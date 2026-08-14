"""Nonlinear head replay around the hourly/daily D-CHyFLO state model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .curves import PhysicalCurves
from .state_model import (
    SECONDS_PER_DAY,
    StateDispatchResult,
    infer_beginning_storage_1e8m3,
    optimize_energy_aware_state,
)


@dataclass
class StateSLPResult:
    """Converged nonlinear replay and its deterministic iteration trace."""

    dispatch: StateDispatchResult
    convergence: pd.DataFrame
    success: bool


def optimize_energy_aware_state_slp(
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    curves: PhysicalCurves,
    grid_limit_mw: float,
    output_coefficient: float = 8.0,
    head_loss_coefficient: float = 3.795e-6,
    rated_turbine_flow_m3s: float = 1270.12,
    installed_capacity_mw: float = 250.0,
    damping: float = 0.6,
    max_slp_iterations: int = 40,
    head_tolerance_m: float = 1.0e-3,
    storage_tolerance_1e8m3: float = 1.0e-5,
    relative_export_tolerance: float = 1.0e-7,
    replay_power_tolerance_mw: float = 1.0e-3,
    lexicographic_relative_tolerance: float = 1.0e-2,
    operating_corridor_multiplier: float = 1.0,
    stall_window: int | None = None,
) -> StateSLPResult:
    """Iterate hourly conversion coefficients against exact nonlinear head.

    The linear dispatch uses a fixed hourly specific-energy coefficient in each
    iteration. Exact head is then replayed from mean daily storage, total daily
    release and each hour's turbine-flow head loss. No observed generation enters
    the optimization.
    """

    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must lie in (0, 1]")
    if operating_corridor_multiplier <= 0.0:
        raise ValueError("operating_corridor_multiplier must be positive")
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
    current_head = np.maximum(initial_daily_head[day_index_array], 0.1)
    reference_storage = d["storage_1e8m3"].to_numpy(float)
    previous_storage = reference_storage.copy()
    physical_storage_lower = d["storage_lower_1e8m3"].to_numpy(float)
    physical_storage_upper = d["storage_upper_1e8m3"].to_numpy(float)
    storage_steps = np.abs(np.diff(reference_storage))
    base_storage_trust_radius = max(
        float(np.quantile(storage_steps, 0.90)), 1.0e-3
    )
    storage_trust_radius = (
        operating_corridor_multiplier * base_storage_trust_radius
    )
    if "reconciled_turbine_flow_m3s" in d:
        reference_turbine_daily = d["reconciled_turbine_flow_m3s"].to_numpy(float)
    else:
        reference_turbine_daily = np.minimum(
            np.maximum(d["outflow_m3s"].to_numpy(float), 0.0),
            rated_turbine_flow_m3s,
        )
    turbine_steps = np.abs(np.diff(reference_turbine_daily))
    base_turbine_trust_radius = max(
        float(np.quantile(turbine_steps, 0.90)), 25.0
    )
    turbine_trust_radius = (
        operating_corridor_multiplier * base_turbine_trust_radius
    )
    reference_turbine = reference_turbine_daily[day_index_array]
    previous_turbine = reference_turbine.copy()
    initial_storage = infer_beginning_storage_1e8m3(d)
    terminal_storage = float(reference_storage[-1])
    previous_export = np.nan
    rows: list[dict[str, float | int | bool | str]] = []
    final_dispatch: StateDispatchResult | None = None

    for iteration in range(max_slp_iterations):
        h_iteration = h.copy()
        h_iteration["specific_energy_mwh_per_m3_hourly"] = (
            output_coefficient * current_head / 3.6e6
        )
        h_iteration["turbine_flow_lower_m3s"] = np.maximum(
            reference_turbine - turbine_trust_radius, 0.0
        )
        h_iteration["turbine_flow_upper_m3s"] = np.minimum(
            reference_turbine + turbine_trust_radius, rated_turbine_flow_m3s
        )
        h_iteration["slp_turbine_target_m3s"] = previous_turbine
        h_iteration["slp_turbine_scale_m3s"] = turbine_trust_radius
        d_iteration = d.copy()
        d_iteration["storage_lower_1e8m3"] = np.maximum(
            physical_storage_lower, reference_storage - storage_trust_radius
        )
        d_iteration["storage_upper_1e8m3"] = np.minimum(
            physical_storage_upper, reference_storage + storage_trust_radius
        )
        d_iteration["slp_storage_target_1e8m3"] = previous_storage
        if "slp_dynamic_max_output_mw" in d:
            d_iteration["dispatch_max_output_mw"] = d["slp_dynamic_max_output_mw"]

        result = optimize_energy_aware_state(
            h_iteration,
            d_iteration,
            grid_limit_mw=grid_limit_mw,
            rated_turbine_flow_m3s=rated_turbine_flow_m3s,
            initial_storage_1e8m3=initial_storage,
            terminal_storage_target_1e8m3=terminal_storage,
            primary_tolerance_mwh=1.0e-3,
            primary_relative_tolerance=lexicographic_relative_tolerance,
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
            mean_level,
            installed_capacity_mw=installed_capacity_mw,
        ).values
        max_head_change = float(np.max(np.abs(exact_head - current_head)))
        max_storage_change = float(np.max(np.abs(end_storage - previous_storage)))
        replay_error = float(np.max(np.abs(exact_power - linear_power)))
        power_excess = float(np.max(exact_power - dynamic_limit[day_index_array]))
        storage_trust_hits = int(
            np.sum(
                np.isclose(
                    end_storage,
                    d_iteration["storage_lower_1e8m3"].to_numpy(float),
                    atol=1.0e-6,
                )
                | np.isclose(
                    end_storage,
                    d_iteration["storage_upper_1e8m3"].to_numpy(float),
                    atol=1.0e-6,
                )
            )
        )
        turbine_trust_hits = int(
            np.sum(
                np.isclose(
                    turbine_flow,
                    h_iteration["turbine_flow_lower_m3s"].to_numpy(float),
                    atol=1.0e-5,
                )
                | np.isclose(
                    turbine_flow,
                    h_iteration["turbine_flow_upper_m3s"].to_numpy(float),
                    atol=1.0e-5,
                )
            )
        )
        relative_export_change = (
            float("nan")
            if not np.isfinite(previous_export)
            else float(
                abs(result.achieved_export_mwh - previous_export)
                / max(abs(previous_export), 1.0)
            )
        )
        converged = bool(
            iteration > 0
            and result.termination.lower() == "optimal"
            and max_head_change <= head_tolerance_m
            and max_storage_change <= storage_tolerance_1e8m3
            and relative_export_change <= relative_export_tolerance
            and replay_error <= replay_power_tolerance_mw
            and power_excess <= 1.0e-3
        )
        recent = rows[-(stall_window - 1):] if stall_window else []
        recent_head = [float(row["max_head_change_m"]) for row in recent] + [max_head_change]
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
        rows.append({
            "iteration": iteration,
            "termination": result.termination,
            "primary_export_mwh": result.primary_export_mwh,
            "achieved_export_mwh": result.achieved_export_mwh,
            "lexicographic_export_gap_mwh": (
                result.primary_export_mwh - result.achieved_export_mwh
            ),
            "lexicographic_relative_tolerance": lexicographic_relative_tolerance,
            "relative_export_change": relative_export_change,
            "secondary_score": result.secondary_score,
            "max_head_change_m": max_head_change,
            "max_storage_change_1e8m3": max_storage_change,
            "max_nonlinear_power_replay_error_mw": replay_error,
            "max_dynamic_power_excess_mw": power_excess,
            "tailwater_out_of_range_days": int(tailwater.out_of_range.sum()),
            "storage_trust_radius_1e8m3": storage_trust_radius,
            "turbine_trust_radius_m3s": turbine_trust_radius,
            "base_storage_trust_radius_1e8m3": base_storage_trust_radius,
            "base_turbine_trust_radius_m3s": base_turbine_trust_radius,
            "operating_corridor_multiplier": operating_corridor_multiplier,
            "storage_trust_bound_hits": storage_trust_hits,
            "turbine_trust_bound_hits": turbine_trust_hits,
            "converged": converged,
            "stalled": stalled,
            "stop_reason": (
                "converged" if converged else "persistent_noncontractive_iteration" if stalled else "continue"
            ),
        })

        result.hourly["nonlinear_replay_head_m"] = exact_head
        result.hourly["nonlinear_replay_hydro_power_mw"] = exact_power
        result.hourly["nonlinear_power_replay_error_mw"] = exact_power - linear_power
        result.daily["optimized_mean_storage_1e8m3"] = mean_storage
        result.daily["optimized_mean_upstream_level_m"] = mean_level
        result.daily["optimized_total_release_m3s"] = daily_release_m3s
        result.daily["optimized_tailwater_level_m"] = tailwater.values
        result.daily["dynamic_max_output_replay_mw"] = dynamic_limit
        if converged:
            return StateSLPResult(result, pd.DataFrame(rows), True)
        if stalled:
            return StateSLPResult(result, pd.DataFrame(rows), False)

        current_head = (1.0 - damping) * current_head + damping * exact_head
        d["slp_dynamic_max_output_mw"] = dynamic_limit
        previous_storage = end_storage
        previous_turbine = turbine_flow
        previous_export = result.achieved_export_mwh

    if final_dispatch is None:
        raise RuntimeError("State SLP did not start")
    return StateSLPResult(final_dispatch, pd.DataFrame(rows), False)
