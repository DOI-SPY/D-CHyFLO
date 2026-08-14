"""Carbon-weighted lexicographic reservoir-FPV dispatch for D-CHyFLO.

This module keeps the v4 water balance, turbine limits, storage bounds and
shared-export constraint.  It adds one transparent epsilon constraint: retain
a registered fraction of the energy-maximizing export before maximizing an
hourly operational carbon-quantity signal.  The signal is not interpreted as
a lifecycle or observed marginal-emissions factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from dchyflo.state_model import (
    M3_PER_STORAGE_UNIT,
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
    _finite_daily_series,
    infer_beginning_storage_1e8m3,
)


@dataclass
class CarbonDispatchResult:
    """Solved energy-constrained, carbon-weighted dispatch."""

    hourly: pd.DataFrame
    daily: pd.DataFrame
    termination: str
    energy_optimum_mwh: float
    energy_floor_mwh: float
    carbon_optimum_kgco2: float
    achieved_export_mwh: float
    achieved_carbon_kgco2: float
    secondary_score: float
    initial_storage_1e8m3: float
    terminal_storage_target_1e8m3: float


def optimize_carbon_aware_state(
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    grid_limit_mw: float,
    energy_retention_fraction: float,
    rated_turbine_flow_m3s: float = 1270.12,
    initial_storage_1e8m3: float | None = None,
    terminal_storage_target_1e8m3: float | None = None,
    energy_floor_tolerance_mwh: float = 1.0e-3,
    path_regularization_weight_mwh_equivalent: float = 1.0,
) -> CarbonDispatchResult:
    """Maximize a carbon signal subject to an energy-retention constraint.

    The energy optimum defines the epsilon floor.  A pure normalized-carbon
    solve records an upper bound, after which a proximal carbon objective uses
    a registered, very small path-regularization weight to stabilize the SLP.
    """

    if not 0.0 < energy_retention_fraction <= 1.0:
        raise ValueError("energy_retention_fraction must lie in (0, 1]")
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
    if any(not indices for indices in hours_by_day.values()):
        raise ValueError("Every reservoir day must have at least one hourly record")

    required_daily = {
        "storage_1e8m3",
        "storage_lower_1e8m3",
        "storage_upper_1e8m3",
        "specific_energy_mwh_per_m3",
        "outflow_m3s",
    }
    missing_daily = sorted(required_daily.difference(d.columns))
    if missing_daily:
        raise ValueError(f"Missing daily columns: {missing_daily}")
    if "carbon_intensity_kgco2_per_mwh" not in h:
        raise ValueError("Hourly carbon_intensity_kgco2_per_mwh is required")

    carbon_intensity = h["carbon_intensity_kgco2_per_mwh"].to_numpy(float)
    if not np.isfinite(carbon_intensity).all() or (carbon_intensity < 0.0).any():
        raise ValueError("Carbon intensity must be finite and non-negative")
    if not np.any(carbon_intensity > 0.0):
        raise ValueError("At least one carbon-intensity value must be positive")

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
    use_tangent = {
        "hydro_linear_slope_mw_per_m3s",
        "hydro_linear_intercept_mw",
    }.issubset(h.columns)
    if use_tangent:
        hydro_slope = h["hydro_linear_slope_mw_per_m3s"].to_numpy(float)
        hydro_intercept = h["hydro_linear_intercept_mw"].to_numpy(float)
        if not np.isfinite(hydro_slope).all() or not np.isfinite(
            hydro_intercept
        ).all():
            raise ValueError("Hydropower tangent coefficients must be finite")
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

    pv = np.maximum(h["pv_power_mw"].to_numpy(float), 0.0)
    if not np.isfinite(pv).all():
        raise ValueError("pv_power_mw must be finite")
    max_column = (
        "dispatch_max_output_mw"
        if "dispatch_max_output_mw" in d
        else "dynamic_max_output_mw"
    )
    max_power = d[max_column].to_numpy(float)[h["day_index"]]
    flow_lower = (
        h["turbine_flow_lower_m3s"].to_numpy(float)
        if "turbine_flow_lower_m3s" in h
        else np.zeros(len(h))
    )
    flow_upper = (
        h["turbine_flow_upper_m3s"].to_numpy(float)
        if "turbine_flow_upper_m3s" in h
        else np.full(len(h), rated_turbine_flow_m3s)
    )
    flow_lower = np.maximum(flow_lower, 0.0)
    flow_upper = np.minimum(flow_upper, rated_turbine_flow_m3s)
    if (flow_lower > flow_upper).any():
        raise ValueError("Hourly turbine-flow trust bounds are inconsistent")
    effective_grid = (
        d["effective_export_limit_mw"].to_numpy(float)[h["day_index"]]
        if "effective_export_limit_mw" in d
        else np.full(len(h), float(grid_limit_mw))
    )

    model = pyo.ConcreteModel(name="D_CHyFLO_v5_carbon_state")
    model.H = pyo.RangeSet(0, len(h) - 1)
    model.D = pyo.RangeSet(0, len(d) - 1)
    model.turbine_flow = pyo.Var(
        model.H,
        domain=pyo.NonNegativeReals,
        bounds=lambda _m, i: (float(flow_lower[i]), float(flow_upper[i])),
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
    if use_tangent:
        model.hydro_conversion = pyo.Constraint(
            model.H,
            rule=lambda m, i: m.hydro[i]
            == float(hydro_slope[i]) * m.turbine_flow[i]
            + float(hydro_intercept[i]),
        )
    else:
        model.hydro_conversion = pyo.Constraint(
            model.H,
            rule=lambda m, i: m.hydro[i]
            == m.turbine_flow[i] * SECONDS_PER_HOUR * float(specific_energy[i]),
        )
    model.export_limit = pyo.Constraint(
        model.H,
        rule=lambda m, i: m.hydro[i] + float(pv[i]) - m.curtailment[i]
        <= float(effective_grid[i]),
    )
    model.water_balance = pyo.ConstraintList()
    for day, indices in hours_by_day.items():
        previous = initial_storage if day == 0 else model.storage[day - 1]
        turbine_volume = (
            sum(model.turbine_flow[i] for i in indices)
            * SECONDS_PER_HOUR
            / M3_PER_STORAGE_UNIT
        )
        inflow_volume = inflow_m3s[day] * SECONDS_PER_DAY / M3_PER_STORAGE_UNIT
        loss_volume = loss_m3[day] / M3_PER_STORAGE_UNIT
        model.water_balance.add(
            model.storage[day]
            == previous
            + float(inflow_volume - loss_volume)
            - turbine_volume
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
    model.operational_carbon = pyo.Expression(
        expr=sum(
            (model.hydro[i] + float(pv[i]) - model.curtailment[i])
            * float(carbon_intensity[i])
            for i in model.H
        )
    )
    carbon_reference = float(np.mean(carbon_intensity))
    model.carbon_score = pyo.Expression(
        expr=model.operational_carbon / carbon_reference
    )
    solver = pyo.SolverFactory("appsi_highs")
    model.energy_objective = pyo.Objective(expr=model.total_export, sense=pyo.maximize)
    energy_result = solver.solve(model)
    energy_termination = str(energy_result.solver.termination_condition)
    if energy_termination.lower() != "optimal":
        raise RuntimeError(f"HiGHS energy termination condition: {energy_termination}")
    energy_optimum = float(pyo.value(model.total_export))

    model.energy_objective.deactivate()
    energy_floor = max(
        0.0,
        energy_retention_fraction * energy_optimum - energy_floor_tolerance_mwh,
    )
    model.energy_floor = pyo.Constraint(expr=model.total_export >= energy_floor)
    model.carbon_objective = pyo.Objective(
        expr=model.carbon_score, sense=pyo.maximize
    )
    carbon_result = solver.solve(model)
    carbon_termination = str(carbon_result.solver.termination_condition)
    if carbon_termination.lower() != "optimal":
        raise RuntimeError(f"HiGHS carbon termination condition: {carbon_termination}")
    carbon_optimum = float(pyo.value(model.operational_carbon))

    model.carbon_objective.deactivate()
    model.storage_dev_pos = pyo.Var(model.D, domain=pyo.NonNegativeReals)
    model.storage_dev_neg = pyo.Var(model.D, domain=pyo.NonNegativeReals)
    storage_target = (
        d["slp_storage_target_1e8m3"].to_numpy(float)
        if "slp_storage_target_1e8m3" in d
        else d["storage_1e8m3"].to_numpy(float)
    )
    model.storage_deviation = pyo.Constraint(
        model.D,
        rule=lambda m, day: m.storage[day] - float(storage_target[day])
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
            float(
                h.get(
                    "slp_turbine_scale_m3s",
                    pd.Series([rated_turbine_flow_m3s]),
                ).iloc[0]
            )
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
        temporal_regularizer = sum(model.ramp[i] for i in model.R) / ramp_scale
    inflow_scale = max(
        float(np.sum(inflow_m3s * SECONDS_PER_DAY / M3_PER_STORAGE_UNIT)), 1.0
    )
    storage_scale = max(float(np.max(upper) - np.min(lower)), 1.0)
    model.secondary_score = pyo.Expression(
        expr=(sum(model.nonpower_release[day] for day in model.D) / inflow_scale)
        + (
            sum(
                model.storage_dev_pos[day] + model.storage_dev_neg[day]
                for day in model.D
            )
            / (len(d) * storage_scale)
        )
        + temporal_regularizer,
    )
    model.proximal_carbon_objective = pyo.Objective(
        expr=model.carbon_score
        - float(path_regularization_weight_mwh_equivalent) * model.secondary_score,
        sense=pyo.maximize,
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
    h["operational_carbon_kgco2"] = (
        h["hybrid_export_mw"] * h["carbon_intensity_kgco2_per_mwh"]
    )
    d["optimized_storage_1e8m3"] = [pyo.value(model.storage[day]) for day in model.D]
    d["optimized_nonpower_release_m3"] = [
        pyo.value(model.nonpower_release[day]) * M3_PER_STORAGE_UNIT for day in model.D
    ]
    d["optimized_turbine_volume_m3"] = [
        h.loc[h["day_index"].eq(day), "turbine_flow_m3s"].sum()
        * SECONDS_PER_HOUR
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

    return CarbonDispatchResult(
        hourly=h,
        daily=d,
        termination=secondary_termination,
        energy_optimum_mwh=energy_optimum,
        energy_floor_mwh=energy_floor,
        carbon_optimum_kgco2=carbon_optimum,
        achieved_export_mwh=float(pyo.value(model.total_export)),
        achieved_carbon_kgco2=float(pyo.value(model.operational_carbon)),
        secondary_score=float(pyo.value(model.secondary_score)),
        initial_storage_1e8m3=initial_storage,
        terminal_storage_target_1e8m3=terminal_target,
    )
