"""Physical reconciliation of historical reservoir operation.

Observed inflow, outflow, storage and generation are treated as noisy
measurements.  Daily mass balance, turbine-flow feasibility and hydropower
conversion are the physical model.  A sparse convex quadratic projection is
solved inside a damped head-update loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, minimize

from .curves import PhysicalCurves


SECONDS_PER_DAY = 86400.0
M3_PER_STORAGE_UNIT = 1.0e8


@dataclass
class ReconciliationResult:
    daily: pd.DataFrame
    convergence: pd.DataFrame
    metrics: dict[str, float | int | str | bool]
    success: bool
    termination: str


def _robust_scale(values: np.ndarray, floor: float) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float(floor)
    q25, q75 = np.quantile(finite, [0.25, 0.75])
    return float(max((q75 - q25) / 1.349, floor))


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def reconcile_historical_operation(
    daily: pd.DataFrame,
    curves: PhysicalCurves,
    output_coefficient: float = 8.0,
    head_loss_coefficient: float = 3.795e-6,
    rated_turbine_flow_m3s: float = 1270.12,
    installed_capacity_mw: float = 250.0,
    weights: dict[str, float] | None = None,
    scales: dict[str, float] | None = None,
    initial_storage_1e8m3: float | None = None,
    damping: float = 0.6,
    max_slp_iterations: int = 40,
    head_tolerance_m: float = 1.0e-3,
    storage_tolerance_1e8m3: float = 1.0e-5,
    relative_objective_tolerance: float = 1.0e-6,
    replay_tolerance_gwh_day: float = 1.0e-3,
) -> ReconciliationResult:
    """Construct a physically consistent daily historical reference.

    ``weights`` and ``scales`` define a deterministic projection metric, not a
    statistical measurement-error likelihood.  Their sensitivity must be
    evaluated before the reconciled series is frozen for scientific use.
    """

    d = daily.copy().reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    required = {
        "date",
        "inflow_m3s",
        "outflow_m3s",
        "storage_1e8m3",
        "observed_generation_gwh",
        "reservoir_loss_m3_day",
        "storage_lower_1e8m3",
        "storage_upper_1e8m3",
        "upstream_level_m",
    }
    missing = sorted(required.difference(d.columns))
    if missing:
        raise ValueError(f"Missing reconciliation columns: {missing}")
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must lie in (0, 1]")

    n = len(d)
    if n < 2:
        raise ValueError("Historical reconciliation requires at least two days")
    obs_s = d["storage_1e8m3"].to_numpy(float)
    obs_i = d["inflow_m3s"].to_numpy(float)
    obs_o = d["outflow_m3s"].to_numpy(float)
    obs_e = d["observed_generation_gwh"].to_numpy(float)
    loss_1e8 = d["reservoir_loss_m3_day"].fillna(0.0).to_numpy(float) / M3_PER_STORAGE_UNIT
    lower_s = d["storage_lower_1e8m3"].to_numpy(float)
    upper_s = d["storage_upper_1e8m3"].to_numpy(float)
    if not np.isfinite(np.column_stack([obs_s, obs_i, obs_o, obs_e])).all():
        raise ValueError("M0 currently requires finite daily core observations")
    if (lower_s > upper_s).any():
        raise ValueError("Storage lower bound exceeds upper bound")

    weight = {"inflow": 1.0, "outflow": 1.0, "storage": 1.0, "energy": 1.0}
    if weights:
        weight.update(weights)
    if any(value <= 0.0 for value in weight.values()):
        raise ValueError("All reconciliation weights must be positive")
    scale = {
        "inflow": _robust_scale(obs_i, 10.0),
        "outflow": _robust_scale(obs_o, 10.0),
        "storage": _robust_scale(obs_s, 1.0),
        "energy": _robust_scale(obs_e, 0.05),
    }
    if scales:
        scale.update(scales)
    if any(value <= 0.0 for value in scale.values()):
        raise ValueError("All reconciliation scales must be positive")

    first_reference_change = (
        (obs_i[0] - obs_o[0]) * SECONDS_PER_DAY / M3_PER_STORAGE_UNIT
        - loss_1e8[0]
    )
    beginning_storage = (
        float(obs_s[0] - first_reference_change)
        if initial_storage_1e8m3 is None
        else float(initial_storage_1e8m3)
    )

    # x = [storage, inflow, outflow, turbine_flow]
    s_slice = slice(0, n)
    i_slice = slice(n, 2 * n)
    o_slice = slice(2 * n, 3 * n)
    t_slice = slice(3 * n, 4 * n)
    conversion = SECONDS_PER_DAY / M3_PER_STORAGE_UNIT

    equality = sparse.lil_matrix((n, 4 * n), dtype=float)
    rhs = np.empty(n, dtype=float)
    for day in range(n):
        equality[day, day] = 1.0
        if day > 0:
            equality[day, day - 1] = -1.0
            rhs[day] = -loss_1e8[day]
        else:
            rhs[day] = beginning_storage - loss_1e8[day]
        equality[day, n + day] = -conversion
        equality[day, 2 * n + day] = conversion
    equality = equality.tocsr()

    turbine_release = sparse.lil_matrix((n, 4 * n), dtype=float)
    for day in range(n):
        turbine_release[day, 2 * n + day] = -1.0
        turbine_release[day, 3 * n + day] = 1.0
    turbine_release = turbine_release.tocsr()
    constraints = [
        LinearConstraint(equality, rhs, rhs),
        LinearConstraint(turbine_release, -np.inf, np.zeros(n)),
    ]
    initial_head = (
        d["net_head_m"].to_numpy(float)
        if "net_head_m" in d
        else d["upstream_level_m"].to_numpy(float)
        - curves.tailwater_from_flow(np.maximum(obs_o, 0.0)).values
        - head_loss_coefficient * np.minimum(np.maximum(obs_o, 0.0), rated_turbine_flow_m3s) ** 2
    )
    head = np.maximum(initial_head, 0.1)
    initial_power_limit = curves.total_max_output_mw(
        d["upstream_level_m"].to_numpy(float),
        installed_capacity_mw=installed_capacity_mw,
    ).values
    initial_turbine_upper = np.minimum(
        rated_turbine_flow_m3s,
        initial_power_limit * 1000.0 / (output_coefficient * head),
    )
    q_turbine_start = np.minimum(
        np.maximum(obs_e * 1.0e6 / (24.0 * output_coefficient * head), 0.0),
        np.minimum(np.maximum(obs_o, 0.0), initial_turbine_upper),
    )
    x = np.concatenate([
        np.clip(obs_s, lower_s, upper_s),
        np.maximum(obs_i, 0.0),
        np.maximum(obs_o, 0.0),
        q_turbine_start,
    ])
    previous_storage = x[s_slice].copy()
    previous_objective = np.nan
    convergence_rows: list[dict[str, float | int | str | bool]] = []
    final_result = None

    for iteration in range(max_slp_iterations):
        energy_factor = 24.0 * output_coefficient * head / 1.0e6
        reference_level = curves.level_from_storage(x[s_slice])
        dynamic_power_limit_mw = curves.total_max_output_mw(
            reference_level,
            installed_capacity_mw=installed_capacity_mw,
        ).values
        turbine_upper = np.minimum(
            rated_turbine_flow_m3s,
            dynamic_power_limit_mw * 1000.0 / (output_coefficient * head),
        )
        turbine_upper = np.maximum(turbine_upper, 0.0)
        bounds = Bounds(
            np.concatenate([lower_s, np.zeros(n), np.zeros(n), np.zeros(n)]),
            np.concatenate([
                upper_s,
                np.full(n, np.inf),
                np.full(n, np.inf),
                turbine_upper,
            ]),
        )
        x[t_slice] = np.minimum(x[t_slice], turbine_upper)
        design = sparse.lil_matrix((4 * n, 4 * n), dtype=float)
        target = np.concatenate([obs_s, obs_i, obs_o, obs_e])
        row_scales = np.concatenate([
            np.full(n, np.sqrt(weight["storage"]) / scale["storage"]),
            np.full(n, np.sqrt(weight["inflow"]) / scale["inflow"]),
            np.full(n, np.sqrt(weight["outflow"]) / scale["outflow"]),
            np.full(n, np.sqrt(weight["energy"]) / scale["energy"]),
        ])
        design[0:n, s_slice] = sparse.eye(n)
        design[n:2 * n, i_slice] = sparse.eye(n)
        design[2 * n:3 * n, o_slice] = sparse.eye(n)
        for day in range(n):
            design[3 * n + day, 3 * n + day] = energy_factor[day]
        design = sparse.diags(row_scales) @ design.tocsr()
        target_weighted = row_scales * target
        hessian = (2.0 * (design.T @ design)).tocsc()

        def objective(vector: np.ndarray) -> float:
            residual = design @ vector - target_weighted
            return float(residual @ residual)

        def gradient(vector: np.ndarray) -> np.ndarray:
            return np.asarray(2.0 * design.T @ (design @ vector - target_weighted)).ravel()

        def hess(_vector: np.ndarray) -> sparse.csc_matrix:
            return hessian

        result = minimize(
            objective,
            x,
            method="trust-constr",
            jac=gradient,
            hess=hess,
            constraints=constraints,
            bounds=bounds,
            options={"maxiter": 500, "gtol": 1.0e-8, "xtol": 1.0e-10, "verbose": 0},
        )
        final_result = result
        x = result.x
        reconciled_storage = x[s_slice]
        reconciled_outflow = x[o_slice]
        reconciled_turbine = x[t_slice]
        level = curves.level_from_storage(reconciled_storage)
        tailwater = curves.tailwater_from_flow(reconciled_outflow)
        exact_head = (
            level
            - tailwater.values
            - head_loss_coefficient * np.square(reconciled_turbine)
        )
        exact_head = np.maximum(exact_head, 0.1)
        linearized_energy = energy_factor * reconciled_turbine
        replay_energy = 24.0 * output_coefficient * exact_head * reconciled_turbine / 1.0e6
        replay_power_mw = replay_energy * 1000.0 / 24.0
        replay_power_limit_mw = curves.total_max_output_mw(
            level,
            installed_capacity_mw=installed_capacity_mw,
        ).values
        max_power_excess = float(np.max(replay_power_mw - replay_power_limit_mw))
        max_head_change = float(np.max(np.abs(exact_head - head)))
        max_storage_change = float(np.max(np.abs(reconciled_storage - previous_storage)))
        objective_change = (
            float("nan")
            if not np.isfinite(previous_objective)
            else float(abs(result.fun - previous_objective))
        )
        relative_objective_change = (
            float("nan")
            if not np.isfinite(previous_objective)
            else float(objective_change / max(abs(previous_objective), 1.0))
        )
        max_replay_error = float(np.max(np.abs(linearized_energy - replay_energy)))
        converged = bool(
            result.success
            and max_head_change <= head_tolerance_m
            and max_storage_change <= storage_tolerance_1e8m3
            and np.isfinite(objective_change)
            and relative_objective_change <= relative_objective_tolerance
            and max_replay_error <= replay_tolerance_gwh_day
            and max_power_excess <= 1.0e-3
        )
        convergence_rows.append({
            "iteration": iteration,
            "optimizer_success": bool(result.success),
            "optimizer_status": int(result.status),
            "optimizer_iterations": int(result.niter),
            "objective": float(result.fun),
            "objective_change": objective_change,
            "relative_objective_change": relative_objective_change,
            "max_head_change_m": max_head_change,
            "max_storage_change_1e8m3": max_storage_change,
            "max_lp_pchip_replay_error_gwh_day": max_replay_error,
            "max_dynamic_power_excess_mw": max_power_excess,
            "converged": converged,
        })
        if converged:
            head = exact_head
            break
        head = (1.0 - damping) * head + damping * exact_head
        previous_storage = reconciled_storage.copy()
        previous_objective = float(result.fun)

    if final_result is None:
        raise RuntimeError("M0 reconciliation did not start")

    reconciled_s = x[s_slice]
    reconciled_i = x[i_slice]
    reconciled_o = x[o_slice]
    reconciled_t = x[t_slice]
    level = curves.level_from_storage(reconciled_s)
    tailwater = curves.tailwater_from_flow(reconciled_o)
    exact_head = np.maximum(
        level
        - tailwater.values
        - head_loss_coefficient * np.square(reconciled_t),
        0.1,
    )
    reconciled_e = 24.0 * output_coefficient * exact_head * reconciled_t / 1.0e6
    reconciled_power_mw = reconciled_e * 1000.0 / 24.0
    dynamic_power_limit_mw = curves.total_max_output_mw(
        level,
        installed_capacity_mw=installed_capacity_mw,
    ).values
    reconstructed_balance = np.empty(n)
    for day in range(n):
        previous = beginning_storage if day == 0 else reconciled_s[day - 1]
        expected = previous + conversion * (reconciled_i[day] - reconciled_o[day]) - loss_1e8[day]
        reconstructed_balance[day] = (reconciled_s[day] - expected) * M3_PER_STORAGE_UNIT

    output = d.copy()
    output["reconciled_inflow_m3s"] = reconciled_i
    output["reconciled_outflow_m3s"] = reconciled_o
    output["reconciled_turbine_flow_m3s"] = reconciled_t
    output["reconciled_storage_1e8m3"] = reconciled_s
    output["reconciled_upstream_level_m"] = level
    output["reconciled_tailwater_level_m"] = tailwater.values
    output["reconciled_net_head_m"] = exact_head
    output["reconciled_generation_gwh"] = reconciled_e
    output["reconciled_power_mw"] = reconciled_power_mw
    output["dynamic_max_output_mw"] = dynamic_power_limit_mw
    output["reconciliation_water_balance_residual_m3"] = reconstructed_balance
    output["reconciled_spill_bypass_m3s"] = reconciled_o - reconciled_t
    output["residual_inflow_m3s"] = reconciled_i - obs_i
    output["residual_outflow_m3s"] = reconciled_o - obs_o
    output["residual_storage_1e8m3"] = reconciled_s - obs_s
    output["residual_level_m"] = level - d["upstream_level_m"].to_numpy(float)
    output["residual_generation_gwh"] = reconciled_e - obs_e

    final_convergence = convergence_rows[-1]
    metrics: dict[str, float | int | str | bool] = {
        "success": bool(final_convergence["converged"]),
        "optimizer_success_last_iteration": bool(final_result.success),
        "termination": str(final_result.message),
        "slp_iterations": len(convergence_rows),
        "beginning_storage_1e8m3": beginning_storage,
        "rmse_level_m": _rmse(output["residual_level_m"].to_numpy(float)),
        "rmse_storage_1e8m3": _rmse(output["residual_storage_1e8m3"].to_numpy(float)),
        "rmse_inflow_m3s": _rmse(output["residual_inflow_m3s"].to_numpy(float)),
        "rmse_outflow_m3s": _rmse(output["residual_outflow_m3s"].to_numpy(float)),
        "rmse_generation_gwh_day": _rmse(output["residual_generation_gwh"].to_numpy(float)),
        "bias_inflow_m3s": float(output["residual_inflow_m3s"].mean()),
        "bias_outflow_m3s": float(output["residual_outflow_m3s"].mean()),
        "bias_generation_gwh_day": float(output["residual_generation_gwh"].mean()),
        "annual_generation_bias_percent": float(
            100.0 * (reconciled_e.sum() / obs_e.sum() - 1.0)
        ),
        "max_water_balance_residual_m3": float(np.max(np.abs(reconstructed_balance))),
        "max_turbine_minus_outflow_m3s": float(np.max(reconciled_t - reconciled_o)),
        "max_dynamic_power_excess_mw": float(
            np.max(reconciled_power_mw - dynamic_power_limit_mw)
        ),
        "dynamic_power_limit_hits": int(
            np.sum(np.isclose(reconciled_power_mw, dynamic_power_limit_mw, atol=1.0e-3))
        ),
        "negative_reconciled_inflow_days": int(np.sum(reconciled_i < -1.0e-8)),
        "storage_lower_bound_hits": int(np.sum(np.isclose(reconciled_s, lower_s, atol=1.0e-6))),
        "storage_upper_bound_hits": int(np.sum(np.isclose(reconciled_s, upper_s, atol=1.0e-6))),
        "projection_weight_inflow": float(weight["inflow"]),
        "projection_weight_outflow": float(weight["outflow"]),
        "projection_weight_storage": float(weight["storage"]),
        "projection_weight_energy": float(weight["energy"]),
        "projection_scale_inflow_m3s": float(scale["inflow"]),
        "projection_scale_outflow_m3s": float(scale["outflow"]),
        "projection_scale_storage_1e8m3": float(scale["storage"]),
        "projection_scale_energy_gwh_day": float(scale["energy"]),
    }
    return ReconciliationResult(
        daily=output,
        convergence=pd.DataFrame(convergence_rows),
        metrics=metrics,
        success=bool(final_convergence["converged"]),
        termination=str(final_result.message),
    )
