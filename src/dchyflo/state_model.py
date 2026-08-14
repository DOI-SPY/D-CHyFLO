"""Explicit cross-day reservoir-state optimization for D-CHyFLO v4.

The v3 dispatch inferred turbine water from observed generation and forced the
water shift to close every dekad.  This module instead makes end-of-day
storage a decision state and derives hydro generation from optimized turbine
flow.  Historical generation is deliberately not an input to the water
balance; it remains validation evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyomo.environ as pyo


SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86400.0
M3_PER_STORAGE_UNIT = 1.0e8


@dataclass
class StateDispatchResult:
    """Solved hourly dispatch and daily reservoir states."""

    hourly: pd.DataFrame
    daily: pd.DataFrame
    termination: str
    primary_export_mwh: float
    achieved_export_mwh: float
    secondary_score: float
    initial_storage_1e8m3: float
    terminal_storage_target_1e8m3: float


def _finite_daily_series(daily: pd.DataFrame, names: tuple[str, ...]) -> np.ndarray:
    """Coalesce candidate series in priority order and require finite output."""

    values = np.full(len(daily), np.nan, dtype=float)
    for name in names:
        if name in daily:
            candidate = daily[name].to_numpy(float)
            fill = ~np.isfinite(values) & np.isfinite(candidate)
            values[fill] = candidate[fill]
    if not np.isfinite(values).all():
        raise ValueError(f"No complete daily series can be assembled from {names}")
    return values


def infer_beginning_storage_1e8m3(daily: pd.DataFrame) -> float:
    """Infer the first beginning-of-day storage from the observed water balance.

    The first day of each source year has no assimilated closure term.  In that
    case the reported-inflow reconstruction is used and the assumption is
    exposed in the output rather than silently replacing the source record.
    """

    row = daily.iloc[0]
    gross_inflow = row.get("effective_gross_inflow_assimilated_m3s", np.nan)
    if not np.isfinite(gross_inflow):
        gross_inflow = row.get("gross_inflow_reconstructed_m3s", np.nan)
    if not np.isfinite(gross_inflow):
        gross_inflow = float(row["inflow_m3s"]) + float(
            row.get("reservoir_loss_equivalent_m3s", 0.0)
        )
    loss_m3 = float(row.get("reservoir_loss_m3_day", 0.0))
    reference_release_m3 = float(row["outflow_m3s"]) * SECONDS_PER_DAY
    reference_change = (
        float(gross_inflow) * SECONDS_PER_DAY - loss_m3 - reference_release_m3
    ) / M3_PER_STORAGE_UNIT
    return float(row["storage_1e8m3"] - reference_change)


def optimize_energy_aware_state(
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    grid_limit_mw: float,
    rated_turbine_flow_m3s: float = 1270.12,
    initial_storage_1e8m3: float | None = None,
    terminal_storage_target_1e8m3: float | None = None,
    primary_tolerance_mwh: float = 1.0e-6,
    primary_relative_tolerance: float = 0.0,
) -> StateDispatchResult:
    """Solve the v4 energy-aware dispatch with an explicit reservoir state.

    Water balance is daily while turbine dispatch and the shared export limit
    are hourly.  Daily specific energy is fixed for one sequential-linearization
    iteration; a later driver updates it from the nonlinear H-V-A replay.

    The solve is lexicographic:

    1. maximize total hybrid export;
    2. hold that optimum within ``primary_tolerance_mwh`` and minimize a
       dimensionless combination of non-power release, storage-path deviation,
       and hourly hydro ramps.
    """

    h = hourly.copy().reset_index(drop=True)
    d = daily.copy().reset_index(drop=True)
    h["timestamp_lst"] = pd.to_datetime(h["timestamp_lst"])
    h["date"] = h["timestamp_lst"].dt.normalize()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    if d["date"].duplicated().any() or h["timestamp_lst"].duplicated().any():
        raise ValueError("Daily dates and hourly timestamps must be unique")
    day_lookup = {date: i for i, date in enumerate(d["date"])}
    h["day_index"] = h["date"].map(day_lookup)
    if h["day_index"].isna().any():
        raise ValueError("Hourly FPV dates do not align with daily reservoir inputs")
    h["day_index"] = h["day_index"].astype(int)
    hours_by_day = {
        day: h.index[h["day_index"].eq(day)].tolist() for day in range(len(d))
    }
    if any(len(indices) == 0 for indices in hours_by_day.values()):
        raise ValueError("Every reservoir day must have at least one hourly record")

    required = {
        "storage_1e8m3",
        "storage_lower_1e8m3",
        "storage_upper_1e8m3",
        "specific_energy_mwh_per_m3",
        "outflow_m3s",
    }
    missing = sorted(required.difference(d.columns))
    if missing:
        raise ValueError(f"Missing daily columns: {missing}")

    inflow_m3s = _finite_daily_series(
        d,
        (
            "effective_gross_inflow_assimilated_m3s",
            "gross_inflow_reconstructed_m3s",
            "inflow_m3s",
        ),
    )
    loss_m3 = (
        d["reservoir_loss_m3_day"].fillna(0.0).to_numpy(float)
        if "reservoir_loss_m3_day" in d
        else np.zeros(len(d))
    )
    daily_specific_energy = d["specific_energy_mwh_per_m3"].to_numpy(float)
    specific_energy = (
        h["specific_energy_mwh_per_m3_hourly"].to_numpy(float)
        if "specific_energy_mwh_per_m3_hourly" in h
        else daily_specific_energy[h["day_index"].to_numpy(int)]
    )
    if (specific_energy <= 0.0).any():
        raise ValueError("specific_energy_mwh_per_m3 must be positive")
    lower = d["storage_lower_1e8m3"].to_numpy(float)
    upper = d["storage_upper_1e8m3"].to_numpy(float)
    if (lower > upper).any():
        raise ValueError("Daily storage lower bounds exceed upper bounds")

    initial_storage = (
        infer_beginning_storage_1e8m3(d)
        if initial_storage_1e8m3 is None
        else float(initial_storage_1e8m3)
    )
    terminal_target = (
        float(d.iloc[-1]["storage_1e8m3"])
        if terminal_storage_target_1e8m3 is None
        else float(terminal_storage_target_1e8m3)
    )
    if not lower[-1] <= terminal_target <= upper[-1]:
        raise ValueError("Terminal storage target lies outside the final-day bounds")

    pv = h["pv_power_mw"].to_numpy(float)
    if (pv < -1.0e-10).any():
        raise ValueError("pv_power_mw must be non-negative")
    pv = np.maximum(pv, 0.0)
    max_column = (
        "dispatch_max_output_mw"
        if "dispatch_max_output_mw" in d
        else "dynamic_max_output_mw"
    )
    max_power = d[max_column].to_numpy(float)[h["day_index"]]
    turbine_flow_lower = (
        h["turbine_flow_lower_m3s"].to_numpy(float)
        if "turbine_flow_lower_m3s" in h
        else np.zeros(len(h))
    )
    turbine_flow_upper = (
        h["turbine_flow_upper_m3s"].to_numpy(float)
        if "turbine_flow_upper_m3s" in h
        else np.full(len(h), rated_turbine_flow_m3s)
    )
    turbine_flow_lower = np.maximum(turbine_flow_lower, 0.0)
    turbine_flow_upper = np.minimum(turbine_flow_upper, rated_turbine_flow_m3s)
    if (turbine_flow_lower > turbine_flow_upper).any():
        raise ValueError("Hourly turbine-flow trust bounds are inconsistent")
    effective_grid = (
        d["effective_export_limit_mw"].to_numpy(float)[h["day_index"]]
        if "effective_export_limit_mw" in d
        else np.full(len(h), float(grid_limit_mw))
    )

    model = pyo.ConcreteModel(name="D_CHyFLO_v4_explicit_state")
    model.H = pyo.RangeSet(0, len(h) - 1)
    model.D = pyo.RangeSet(0, len(d) - 1)
    model.turbine_flow = pyo.Var(
        model.H,
        domain=pyo.NonNegativeReals,
        bounds=lambda _m, i: (
            float(turbine_flow_lower[i]),
            float(turbine_flow_upper[i]),
        ),
    )
    model.hydro = pyo.Var(
        model.H,
        domain=pyo.NonNegativeReals,
        bounds=lambda _m, i: (0.0, float(max_power[i])),
    )
    model.curtailment = pyo.Var(
        model.H,
        domain=pyo.NonNegativeReals,
        bounds=lambda _m, i: (0.0, float(pv[i])),
    )
    model.storage = pyo.Var(
        model.D,
        bounds=lambda _m, day: (float(lower[day]), float(upper[day])),
    )
    model.nonpower_release = pyo.Var(model.D, domain=pyo.NonNegativeReals)

    model.hydro_conversion = pyo.Constraint(
        model.H,
        rule=lambda m, i: m.hydro[i]
        == m.turbine_flow[i]
        * SECONDS_PER_HOUR
        * float(specific_energy[i]),
    )
    model.export_limit = pyo.Constraint(
        model.H,
        rule=lambda m, i: m.hydro[i] + float(pv[i]) - m.curtailment[i]
        <= float(effective_grid[i]),
    )

    model.water_balance = pyo.ConstraintList()
    for day, indices in hours_by_day.items():
        previous = initial_storage if day == 0 else model.storage[day - 1]
        turbine_volume_1e8 = (
            sum(model.turbine_flow[i] for i in indices)
            * SECONDS_PER_HOUR
            / M3_PER_STORAGE_UNIT
        )
        inflow_volume_1e8 = inflow_m3s[day] * SECONDS_PER_DAY / M3_PER_STORAGE_UNIT
        loss_volume_1e8 = loss_m3[day] / M3_PER_STORAGE_UNIT
        model.water_balance.add(
            model.storage[day]
            == previous
            + float(inflow_volume_1e8 - loss_volume_1e8)
            - turbine_volume_1e8
            - model.nonpower_release[day]
        )
    model.terminal_storage = pyo.Constraint(
        expr=model.storage[len(d) - 1] == terminal_target
    )

    model.total_export = pyo.Expression(
        expr=sum(
            model.hydro[i] + float(pv[i]) - model.curtailment[i] for i in model.H
        )
    )
    model.primary_objective = pyo.Objective(
        expr=model.total_export, sense=pyo.maximize
    )
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    solver = pyo.SolverFactory("appsi_highs")
    primary_result = solver.solve(model)
    primary_termination = str(primary_result.solver.termination_condition)
    if primary_termination.lower() != "optimal":
        raise RuntimeError(f"HiGHS primary termination condition: {primary_termination}")
    primary_export = float(pyo.value(model.total_export))

    model.primary_objective.deactivate()
    lexicographic_tolerance = max(
        float(primary_tolerance_mwh),
        float(primary_relative_tolerance) * abs(primary_export),
    )
    model.primary_floor = pyo.Constraint(
        expr=model.total_export >= primary_export - lexicographic_tolerance
    )
    model.storage_dev_pos = pyo.Var(model.D, domain=pyo.NonNegativeReals)
    model.storage_dev_neg = pyo.Var(model.D, domain=pyo.NonNegativeReals)
    storage_deviation_target = (
        d["slp_storage_target_1e8m3"].to_numpy(float)
        if "slp_storage_target_1e8m3" in d
        else d["storage_1e8m3"].to_numpy(float)
    )
    model.storage_deviation = pyo.Constraint(
        model.D,
        rule=lambda m, day: m.storage[day] - float(storage_deviation_target[day])
        == m.storage_dev_pos[day] - m.storage_dev_neg[day],
    )
    if "slp_turbine_target_m3s" in h:
        turbine_target = h["slp_turbine_target_m3s"].to_numpy(float)
        model.turbine_dev_pos = pyo.Var(model.H, domain=pyo.NonNegativeReals)
        model.turbine_dev_neg = pyo.Var(model.H, domain=pyo.NonNegativeReals)
        model.turbine_deviation = pyo.Constraint(
            model.H,
            rule=lambda m, i: m.turbine_flow[i] - float(turbine_target[i])
            == m.turbine_dev_pos[i] - m.turbine_dev_neg[i],
        )
        turbine_scale = max(
            float(h.get("slp_turbine_scale_m3s", pd.Series([rated_turbine_flow_m3s])).iloc[0])
            * len(h),
            1.0,
        )
        temporal_regularizer = sum(
            model.turbine_dev_pos[i] + model.turbine_dev_neg[i] for i in model.H
        ) / turbine_scale
    else:
        model.R = pyo.RangeSet(1, len(h) - 1)
        model.ramp = pyo.Var(model.R, domain=pyo.NonNegativeReals)
        model.ramp_up = pyo.Constraint(
            model.R,
            rule=lambda m, i: m.ramp[i] >= m.hydro[i] - m.hydro[i - 1],
        )
        model.ramp_down = pyo.Constraint(
            model.R,
            rule=lambda m, i: m.ramp[i] >= m.hydro[i - 1] - m.hydro[i],
        )
        ramp_scale = max(float(grid_limit_mw) * max(len(h) - 1, 1), 1.0)
        temporal_regularizer = sum(
            model.ramp[i] for i in range(1, len(h))
        ) / ramp_scale
    inflow_scale = max(
        float(np.sum(inflow_m3s * SECONDS_PER_DAY / M3_PER_STORAGE_UNIT)), 1.0
    )
    storage_scale = max(float(np.max(upper) - np.min(lower)), 1.0)
    model.secondary_objective = pyo.Objective(
        expr=(sum(model.nonpower_release[day] for day in model.D) / inflow_scale)
        + (
            sum(
                model.storage_dev_pos[day] + model.storage_dev_neg[day]
                for day in model.D
            )
            / (len(d) * storage_scale)
        )
        + temporal_regularizer,
        sense=pyo.minimize,
    )
    secondary_result = solver.solve(model)
    secondary_termination = str(secondary_result.solver.termination_condition)
    if secondary_termination.lower() != "optimal":
        raise RuntimeError(
            f"HiGHS secondary termination condition: {secondary_termination}"
        )

    h["turbine_flow_m3s"] = [pyo.value(model.turbine_flow[i]) for i in model.H]
    h["hydro_power_mw"] = [pyo.value(model.hydro[i]) for i in model.H]
    h["pv_curtailment_mw"] = [pyo.value(model.curtailment[i]) for i in model.H]
    h["pv_used_mw"] = h["pv_power_mw"] - h["pv_curtailment_mw"]
    h["effective_export_limit_mw"] = effective_grid
    h["hybrid_export_mw"] = h["hydro_power_mw"] + h["pv_used_mw"]

    d["optimized_storage_1e8m3"] = [pyo.value(model.storage[day]) for day in model.D]
    d["optimized_nonpower_release_m3"] = [
        pyo.value(model.nonpower_release[day]) * M3_PER_STORAGE_UNIT for day in model.D
    ]
    d["optimized_turbine_volume_m3"] = [
        h.loc[h["day_index"].eq(day), "turbine_flow_m3s"].sum() * SECONDS_PER_HOUR
        for day in range(len(d))
    ]
    d["optimized_hydro_energy_mwh"] = [
        h.loc[h["day_index"].eq(day), "hydro_power_mw"].sum()
        for day in range(len(d))
    ]
    d["water_balance_residual_m3"] = np.nan
    previous_storage = initial_storage
    for day in range(len(d)):
        expected = (
            previous_storage
            + inflow_m3s[day] * SECONDS_PER_DAY / M3_PER_STORAGE_UNIT
            - loss_m3[day] / M3_PER_STORAGE_UNIT
            - d.loc[day, "optimized_turbine_volume_m3"] / M3_PER_STORAGE_UNIT
            - d.loc[day, "optimized_nonpower_release_m3"] / M3_PER_STORAGE_UNIT
        )
        d.loc[day, "water_balance_residual_m3"] = (
            d.loc[day, "optimized_storage_1e8m3"] - expected
        ) * M3_PER_STORAGE_UNIT
        previous_storage = d.loc[day, "optimized_storage_1e8m3"]

    return StateDispatchResult(
        hourly=h,
        daily=d,
        termination=secondary_termination,
        primary_export_mwh=primary_export,
        achieved_export_mwh=float(pyo.value(model.total_export)),
        secondary_score=float(pyo.value(model.secondary_objective)),
        initial_storage_1e8m3=initial_storage,
        terminal_storage_target_1e8m3=terminal_target,
    )
