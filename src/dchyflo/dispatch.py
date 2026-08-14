"""Transparent Pyomo/HiGHS hydro-FPV cooperative dispatch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyomo.environ as pyo


@dataclass
class DispatchResult:
    hourly: pd.DataFrame
    daily: pd.DataFrame
    termination: str


def _dekad_key(timestamp: pd.Timestamp) -> tuple[int, int, int]:
    dekad = 1 if timestamp.day <= 10 else 2 if timestamp.day <= 20 else 3
    return timestamp.year, timestamp.month, dekad


def optimize_dekadal_dispatch(
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    grid_limit_mw: float,
) -> DispatchResult:
    """Minimize PV curtailment while preserving turbine water by dekad.

    Daily head is fixed to the validated physical baseline. Non-power release
    remains on its historical path. Turbine-water timing may move within each
    dekad, but its total volume is conserved and the implied reservoir storage
    is constrained at every day end.
    """
    h = hourly.copy().reset_index(drop=True)
    d = daily.copy().reset_index(drop=True)
    h["date"] = pd.to_datetime(h["timestamp_lst"]).dt.normalize()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    if not set(h["date"]).issubset(set(d["date"])):
        raise ValueError("Hourly FPV dates do not align with daily hydropower")
    h["day_index"] = h["date"].map({date: i for i, date in enumerate(d["date"])})
    if h["day_index"].isna().any():
        raise ValueError("Missing daily row for one or more hourly timestamps")
    h["day_index"] = h["day_index"].astype(int)

    model = pyo.ConcreteModel()
    model.H = pyo.RangeSet(0, len(h) - 1)
    max_column = "dispatch_max_output_mw" if "dispatch_max_output_mw" in d else "dynamic_max_output_mw"
    max_power = d[max_column].to_numpy(float)[h["day_index"]]
    effective_grid = (
        d["effective_export_limit_mw"].to_numpy(float)[h["day_index"]]
        if "effective_export_limit_mw" in d else np.full(len(h), grid_limit_mw)
    )
    pv = h["pv_power_mw"].to_numpy(float)
    model.hydro = pyo.Var(
        model.H, domain=pyo.NonNegativeReals,
        bounds=lambda _m, i: (0.0, float(max_power[i])),
    )
    model.curtailment = pyo.Var(model.H, domain=pyo.NonNegativeReals)
    model.grid_excess = pyo.Constraint(
        model.H,
        rule=lambda m, i: m.curtailment[i] >= m.hydro[i] + float(pv[i]) - float(effective_grid[i]),
    )
    model.pv_curtailment_cap = pyo.Constraint(
        model.H, rule=lambda m, i: m.curtailment[i] <= float(pv[i])
    )

    hours_by_day = {
        i: h.index[h["day_index"].eq(i)].tolist() for i in range(len(d))
    }
    observed_energy = d["observed_hydro_energy_mwh"].to_numpy(float)
    specific_energy = d["specific_energy_mwh_per_m3"].to_numpy(float)
    observed_water = observed_energy / specific_energy
    model.dev_pos = pyo.Var(range(len(d)), domain=pyo.NonNegativeReals)
    model.dev_neg = pyo.Var(range(len(d)), domain=pyo.NonNegativeReals)
    model.daily_deviation = pyo.ConstraintList()
    for day, indices in hours_by_day.items():
        model.daily_deviation.add(
            sum(model.hydro[i] for i in indices) - float(observed_energy[day])
            == model.dev_pos[day] - model.dev_neg[day]
        )

    groups: dict[tuple[int, int, int], list[int]] = {}
    for i, date in enumerate(d["date"]):
        groups.setdefault(_dekad_key(date), []).append(i)
    model.storage_bounds = pyo.ConstraintList()
    model.dekad_terminal = pyo.ConstraintList()
    for days in groups.values():
        optimized_water_terms_1e8 = []
        observed_water_cumulative_1e8 = 0.0
        for day in days:
            day_water_1e8 = sum(model.hydro[i] for i in hours_by_day[day]) / (
                float(specific_energy[day]) * 1.0e8
            )
            optimized_water_terms_1e8.append(day_water_1e8)
            observed_water_cumulative_1e8 += float(observed_water[day]) / 1.0e8
            simulated_storage = (
                float(d.loc[day, "storage_1e8m3"])
                + observed_water_cumulative_1e8
                - sum(optimized_water_terms_1e8)
            )
            model.storage_bounds.add(simulated_storage >= float(d.loc[day, "storage_lower_1e8m3"]))
            model.storage_bounds.add(simulated_storage <= float(d.loc[day, "storage_upper_1e8m3"]))
        model.dekad_terminal.add(
            sum(optimized_water_terms_1e8) == float(observed_water[days].sum()) / 1.0e8
        )

    model.objective = pyo.Objective(
        expr=sum(model.curtailment[i] for i in model.H)
        + 1.0e-6 * sum(model.dev_pos[i] + model.dev_neg[i] for i in range(len(d))),
        sense=pyo.minimize,
    )
    solver = pyo.SolverFactory("appsi_highs")
    result = solver.solve(model)
    termination = str(result.solver.termination_condition)
    if termination.lower() != "optimal":
        raise RuntimeError(f"HiGHS termination condition: {termination}")

    h["hydro_power_mw"] = [pyo.value(model.hydro[i]) for i in model.H]
    h["effective_export_limit_mw"] = effective_grid
    h["pv_curtailment_mw"] = [pyo.value(model.curtailment[i]) for i in model.H]
    h["hybrid_export_mw"] = h["hydro_power_mw"] + h["pv_power_mw"] - h["pv_curtailment_mw"]
    optimized_daily_energy = h.groupby("day_index")["hydro_power_mw"].sum().reindex(range(len(d))).to_numpy()
    optimized_water = optimized_daily_energy / specific_energy
    storage_sim = np.empty(len(d))
    terminal_error = np.empty(len(d))
    for days in groups.values():
        shift = 0.0
        for day in days:
            shift += observed_water[day] - optimized_water[day]
            storage_sim[day] = d.loc[day, "storage_1e8m3"] + shift / 1.0e8
            terminal_error[day] = shift
    d["optimized_hydro_energy_mwh"] = optimized_daily_energy
    d["optimized_turbine_water_m3"] = optimized_water
    d["simulated_storage_1e8m3"] = storage_sim
    d["dekad_cumulative_water_shift_m3"] = terminal_error
    return DispatchResult(h, d, termination)


def fixed_noncooperative_dispatch(
    hourly: pd.DataFrame, daily: pd.DataFrame, grid_limit_mw: float
) -> pd.DataFrame:
    h = hourly.copy()
    d = daily.set_index(pd.to_datetime(daily["date"]).dt.normalize())
    dates = pd.to_datetime(h["timestamp_lst"]).dt.normalize()
    h["hydro_power_mw"] = dates.map(d["observed_hydro_energy_mwh"] / 24.0).to_numpy()
    h["effective_export_limit_mw"] = (
        dates.map(d["effective_export_limit_mw"]).to_numpy()
        if "effective_export_limit_mw" in d else grid_limit_mw
    )
    total = h["hydro_power_mw"] + h["pv_power_mw"]
    h["pv_curtailment_mw"] = (total - h["effective_export_limit_mw"]).clip(lower=0.0)
    h["hybrid_export_mw"] = total - h["pv_curtailment_mw"]
    return h
