"""Dynamic-head replay for the D-CHyFLO v5 carbon-aware dispatch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dchyflo.curves import PhysicalCurves
from dchyflo.state_model import SECONDS_PER_DAY, infer_beginning_storage_1e8m3

from .carbon_dispatch import CarbonDispatchResult, optimize_carbon_aware_state


@dataclass
class CarbonSLPResult:
    """Converged carbon-aware dispatch and nonlinear replay trace."""

    dispatch: CarbonDispatchResult
    convergence: pd.DataFrame
    success: bool


def optimize_carbon_aware_state_slp(
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    curves: PhysicalCurves,
    grid_limit_mw: float,
    energy_retention_fraction: float,
    output_coefficient: float = 8.0,
    head_loss_coefficient: float = 3.795e-6,
    rated_turbine_flow_m3s: float = 1270.12,
    installed_capacity_mw: float = 250.0,
    damping: float = 0.6,
    max_slp_iterations: int = 64,
    head_tolerance_m: float = 1.0e-3,
    storage_tolerance_1e8m3: float = 1.0e-5,
    relative_objective_tolerance: float = 1.0e-7,
    path_regularization_weight_mwh_equivalent: float = 1.0,
    replay_power_tolerance_mw: float = 1.0e-3,
    operating_corridor_multiplier: float = 1.0,
    iteration_trust_step_fraction: float = 0.25,
    stall_window: int | None = 8,
) -> CarbonSLPResult:
    """Iterate the carbon-aware LP against exact nonlinear reservoir head."""

    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must lie in (0, 1]")
    if operating_corridor_multiplier <= 0.0:
        raise ValueError("operating_corridor_multiplier must be positive")
    if not 0.0 < iteration_trust_step_fraction <= 1.0:
        raise ValueError("iteration_trust_step_fraction must lie in (0, 1]")
    if stall_window is not None and stall_window < 4:
        raise ValueError("stall_window must be at least 4 or None")

    h = hourly.copy().reset_index(drop=True)
    d = daily.copy().reset_index(drop=True)
    h["timestamp_lst"] = pd.to_datetime(h["timestamp_lst"])
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    lookup = {date: day for day, date in enumerate(d["date"])}
    day_index = h["timestamp_lst"].dt.normalize().map(lookup)
    if day_index.isna().any():
        raise ValueError("Hourly timestamps do not align with the daily SLP reference")
    day_index_array = day_index.to_numpy(int)
    if "reconciled_net_head_m" in d:
        initial_daily_head = d["reconciled_net_head_m"].to_numpy(float)
    elif "net_head_m" in d:
        initial_daily_head = d["net_head_m"].to_numpy(float)
    else:
        raise ValueError("Daily SLP input requires reconciled_net_head_m or net_head_m")
    current_head = (
        np.maximum(h["carbon_head_reference_m"].to_numpy(float), 0.1)
        if "carbon_head_reference_m" in h
        else np.maximum(initial_daily_head[day_index_array], 0.1)
    )
    reference_storage = (
        d["carbon_storage_reference_1e8m3"].to_numpy(float)
        if "carbon_storage_reference_1e8m3" in d
        else d["storage_1e8m3"].to_numpy(float)
    )
    previous_storage = reference_storage.copy()
    physical_lower = d["storage_lower_1e8m3"].to_numpy(float)
    physical_upper = d["storage_upper_1e8m3"].to_numpy(float)
    storage_radius = operating_corridor_multiplier * max(
        float(np.quantile(np.abs(np.diff(reference_storage)), 0.90)), 1.0e-3
    )
    if "reconciled_turbine_flow_m3s" in d:
        reference_turbine_daily = d["reconciled_turbine_flow_m3s"].to_numpy(float)
    else:
        reference_turbine_daily = np.minimum(
            np.maximum(d["outflow_m3s"].to_numpy(float), 0.0),
            rated_turbine_flow_m3s,
        )
    turbine_radius = operating_corridor_multiplier * max(
        float(np.quantile(np.abs(np.diff(reference_turbine_daily)), 0.90)), 25.0
    )
    storage_step_radius = iteration_trust_step_fraction * storage_radius
    turbine_step_radius = iteration_trust_step_fraction * turbine_radius
    reference_turbine = reference_turbine_daily[day_index_array]
    previous_turbine = (
        h["carbon_turbine_reference_m3s"].to_numpy(float)
        if "carbon_turbine_reference_m3s" in h
        else reference_turbine.copy()
    )
    initial_storage = infer_beginning_storage_1e8m3(d)
    terminal_storage = float(reference_storage[-1])
    previous_export = np.nan
    previous_carbon = np.nan
    rows: list[dict[str, float | int | bool | str]] = []
    final_dispatch: CarbonDispatchResult | None = None

    for iteration in range(max_slp_iterations):
        h_iteration = h.copy()
        h_iteration["specific_energy_mwh_per_m3_hourly"] = (
            output_coefficient * current_head / 3.6e6
        )
        h_iteration["hydro_linear_slope_mw_per_m3s"] = (
            output_coefficient
            / 1000.0
            * (
                current_head
                - 2.0 * head_loss_coefficient * np.square(previous_turbine)
            )
        )
        h_iteration["hydro_linear_intercept_mw"] = (
            2.0
            * output_coefficient
            * head_loss_coefficient
            * np.power(previous_turbine, 3)
            / 1000.0
        )
        h_iteration["turbine_flow_lower_m3s"] = np.maximum.reduce(
            [
                reference_turbine - turbine_radius,
                previous_turbine - turbine_step_radius,
                np.zeros(len(h_iteration)),
            ]
        )
        h_iteration["turbine_flow_upper_m3s"] = np.minimum.reduce(
            [
                reference_turbine + turbine_radius,
                previous_turbine + turbine_step_radius,
                np.full(len(h_iteration), rated_turbine_flow_m3s),
            ]
        )
        h_iteration["slp_turbine_target_m3s"] = previous_turbine
        h_iteration["slp_turbine_scale_m3s"] = turbine_radius
        d_iteration = d.copy()
        d_iteration["storage_lower_1e8m3"] = np.maximum.reduce(
            [
                physical_lower,
                reference_storage - storage_radius,
                previous_storage - storage_step_radius,
            ]
        )
        d_iteration["storage_upper_1e8m3"] = np.minimum.reduce(
            [
                physical_upper,
                reference_storage + storage_radius,
                previous_storage + storage_step_radius,
            ]
        )
        d_iteration["slp_storage_target_1e8m3"] = previous_storage
        if "slp_dynamic_max_output_mw" in d:
            d_iteration["dispatch_max_output_mw"] = d["slp_dynamic_max_output_mw"]

        result = optimize_carbon_aware_state(
            h_iteration,
            d_iteration,
            grid_limit_mw=grid_limit_mw,
            energy_retention_fraction=energy_retention_fraction,
            rated_turbine_flow_m3s=rated_turbine_flow_m3s,
            initial_storage_1e8m3=initial_storage,
            terminal_storage_target_1e8m3=terminal_storage,
            path_regularization_weight_mwh_equivalent=(
                path_regularization_weight_mwh_equivalent
            ),
        )
        final_dispatch = result
        end_storage = result.daily["optimized_storage_1e8m3"].to_numpy(float)
        beginning_storage = np.concatenate([[initial_storage], end_storage[:-1]])
        mean_storage = 0.5 * (beginning_storage + end_storage)
        mean_level = curves.level_from_storage(mean_storage)
        daily_release_m3 = (
            result.daily["optimized_turbine_volume_m3"].to_numpy(float)
            + result.daily["optimized_nonpower_release_m3"].to_numpy(float)
        )
        daily_release_m3s = daily_release_m3 / SECONDS_PER_DAY
        tailwater = curves.tailwater_from_flow(daily_release_m3s)
        turbine_flow = result.hourly["turbine_flow_m3s"].to_numpy(float)
        exact_head = np.maximum(
            mean_level[day_index_array]
            - tailwater.values[day_index_array]
            - head_loss_coefficient * np.square(turbine_flow),
            0.1,
        )
        exact_power = output_coefficient * turbine_flow * exact_head / 1000.0
        linear_power = result.hourly["hydro_power_mw"].to_numpy(float)
        dynamic_limit = curves.total_max_output_mw(
            mean_level, installed_capacity_mw=installed_capacity_mw
        ).values
        exact_export = exact_power + result.hourly["pv_used_mw"].to_numpy(float)
        carbon_factor = result.hourly[
            "carbon_intensity_kgco2_per_mwh"
        ].to_numpy(float)
        replay_carbon = float(np.sum(exact_export * carbon_factor))
        replay_export = float(np.sum(exact_export))
        max_head_change = float(np.max(np.abs(exact_head - current_head)))
        max_storage_change = float(np.max(np.abs(end_storage - previous_storage)))
        replay_error = float(np.max(np.abs(exact_power - linear_power)))
        power_excess = float(np.max(exact_power - dynamic_limit[day_index_array]))
        export_change = (
            float("nan")
            if not np.isfinite(previous_export)
            else abs(replay_export - previous_export) / max(abs(previous_export), 1.0)
        )
        carbon_change = (
            float("nan")
            if not np.isfinite(previous_carbon)
            else abs(replay_carbon - previous_carbon) / max(abs(previous_carbon), 1.0)
        )
        converged = bool(
            iteration > 0
            and result.termination.lower() == "optimal"
            and max_head_change <= head_tolerance_m
            and max_storage_change <= storage_tolerance_1e8m3
            and export_change <= relative_objective_tolerance
            and carbon_change <= relative_objective_tolerance
            and replay_error <= replay_power_tolerance_mw
            and power_excess <= 1.0e-3
        )
        recent = rows[-(stall_window - 1) :] if stall_window else []
        recent_head = [float(row["max_head_change_m"]) for row in recent] + [
            max_head_change
        ]
        recent_storage = [
            float(row["max_storage_change_1e8m3"]) for row in recent
        ] + [max_storage_change]
        stalled = bool(
            stall_window is not None
            and iteration + 1 >= stall_window
            and len(recent_head) == stall_window
            and min(recent_head) > max(50.0 * head_tolerance_m, 0.05)
            and min(recent_storage)
            > max(50.0 * storage_tolerance_1e8m3, 1.0e-3)
            and recent_head[-1] >= 0.90 * recent_head[0]
            and recent_storage[-1] >= 0.90 * recent_storage[0]
        )
        rows.append(
            {
                "iteration": iteration,
                "termination": result.termination,
                "energy_retention_fraction": energy_retention_fraction,
                "energy_optimum_mwh": result.energy_optimum_mwh,
                "energy_floor_mwh": result.energy_floor_mwh,
                "achieved_export_mwh": result.achieved_export_mwh,
                "replay_export_mwh": replay_export,
                "achieved_carbon_kgco2": result.achieved_carbon_kgco2,
                "replay_carbon_kgco2": replay_carbon,
                "path_regularization_score": result.secondary_score,
                "relative_export_change": export_change,
                "relative_carbon_change": carbon_change,
                "max_head_change_m": max_head_change,
                "max_storage_change_1e8m3": max_storage_change,
                "max_nonlinear_power_replay_error_mw": replay_error,
                "max_dynamic_power_excess_mw": power_excess,
                "storage_trust_radius_1e8m3": storage_radius,
                "turbine_trust_radius_m3s": turbine_radius,
                "storage_iteration_step_radius_1e8m3": storage_step_radius,
                "turbine_iteration_step_radius_m3s": turbine_step_radius,
                "converged": converged,
                "stalled": stalled,
                "stop_reason": (
                    "converged"
                    if converged
                    else "persistent_noncontractive_iteration"
                    if stalled
                    else "continue"
                ),
            }
        )
        result.hourly["nonlinear_replay_head_m"] = exact_head
        result.hourly["nonlinear_replay_hydro_power_mw"] = exact_power
        result.hourly["nonlinear_replay_hybrid_export_mw"] = exact_export
        result.hourly["nonlinear_replay_operational_carbon_kgco2"] = (
            exact_export * carbon_factor
        )
        result.hourly["nonlinear_power_replay_error_mw"] = exact_power - linear_power
        result.daily["optimized_mean_storage_1e8m3"] = mean_storage
        result.daily["optimized_mean_upstream_level_m"] = mean_level
        result.daily["optimized_total_release_m3s"] = daily_release_m3s
        result.daily["optimized_tailwater_level_m"] = tailwater.values
        result.daily["dynamic_max_output_replay_mw"] = dynamic_limit
        if converged:
            return CarbonSLPResult(result, pd.DataFrame(rows), True)
        if stalled:
            return CarbonSLPResult(result, pd.DataFrame(rows), False)
        current_head = (1.0 - damping) * current_head + damping * exact_head
        d["slp_dynamic_max_output_mw"] = dynamic_limit
        previous_storage = end_storage
        previous_turbine = turbine_flow
        previous_export = replay_export
        previous_carbon = replay_carbon

    if final_dispatch is None:
        raise RuntimeError("Carbon-aware state SLP did not start")
    return CarbonSLPResult(final_dispatch, pd.DataFrame(rows), False)
