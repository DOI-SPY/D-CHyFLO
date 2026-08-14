"""Hydropower equations and historical calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .curves import PhysicalCurves


def _safe_r2(observed: np.ndarray, modeled: np.ndarray) -> float:
    mask = np.isfinite(observed) & np.isfinite(modeled)
    if mask.sum() < 2:
        return float("nan")
    residual = np.sum((observed[mask] - modeled[mask]) ** 2)
    total = np.sum((observed[mask] - observed[mask].mean()) ** 2)
    return float(1.0 - residual / total) if total > 0 else float("nan")


def calibrate_output_coefficient(data: pd.DataFrame, head_loss_coefficient: float = 3.795e-6,
                                 rated_total_flow_m3s: float = 1270.12) -> dict:
    """Robustly estimate K from non-spill historical daily observations.

    Generation hours are unit-hours, so daily energy is divided by 24 h rather
    than by the summed unit operating hours.
    """
    work = data.copy()
    work["gross_head_observed_m"] = work["upstream_level_m"] - work["downstream_level_m"]
    work["head_loss_m"] = head_loss_coefficient * work["outflow_m3s"] ** 2
    work["net_head_observed_m"] = work["gross_head_observed_m"] - work["head_loss_m"]
    work["observed_average_power_kw"] = work["generation_1e4kwh"] * 1e4 / 24.0
    work["implied_output_coefficient"] = work["observed_average_power_kw"] / (
        work["outflow_m3s"] * work["net_head_observed_m"]
    )
    eligible = (
        (work["generation_1e4kwh"] > 0)
        & (work["outflow_m3s"] > 20)
        & (work["outflow_m3s"] <= rated_total_flow_m3s)
        & (work["net_head_observed_m"] > 9.0)
        & work["implied_output_coefficient"].between(5.0, 12.0)
    )
    coefficients = work.loc[eligible, "implied_output_coefficient"]
    return {
        "coefficient_median": float(coefficients.median()),
        "coefficient_mean": float(coefficients.mean()),
        "coefficient_q25": float(coefficients.quantile(0.25)),
        "coefficient_q75": float(coefficients.quantile(0.75)),
        "eligible_days": int(eligible.sum()),
        "total_days": int(len(work)),
        "diagnostics": work,
    }


def evaluate_hydropower(data: pd.DataFrame, curves: PhysicalCurves, output_coefficient: float,
                        head_loss_coefficient: float = 3.795e-6,
                        rated_total_flow_m3s: float = 1270.12,
                        installed_capacity_mw: float = 250.0) -> tuple[pd.DataFrame, dict]:
    """Evaluate daily generation using observed releases and physical limits."""
    result = data.copy()
    result["turbine_flow_upper_m3s"] = np.minimum(
        np.maximum(result["outflow_m3s"], 0.0), rated_total_flow_m3s
    )
    result["spill_lower_bound_m3s"] = np.maximum(result["outflow_m3s"] - rated_total_flow_m3s, 0.0)

    tailwater = curves.tailwater_from_flow(result["outflow_m3s"], version="2013")
    result["tailwater_modeled_2013_m"] = tailwater.values
    result["tailwater_curve_extrapolated"] = tailwater.out_of_range
    result["tailwater_residual_m"] = result["downstream_level_m"] - result["tailwater_modeled_2013_m"]

    result["gross_head_observed_m"] = result["upstream_level_m"] - result["downstream_level_m"]
    result["head_loss_m"] = head_loss_coefficient * result["turbine_flow_upper_m3s"] ** 2
    result["net_head_m"] = result["gross_head_observed_m"] - result["head_loss_m"]
    max_output = curves.total_max_output_mw(result["upstream_level_m"])
    result["dynamic_max_output_mw"] = np.minimum(max_output.values, installed_capacity_mw)
    result["max_output_curve_extrapolated"] = max_output.out_of_range

    hydraulic_power_mw = (
        output_coefficient * result["turbine_flow_upper_m3s"] * result["net_head_m"] / 1000.0
    )
    result["modeled_power_mw"] = np.minimum(
        np.maximum(hydraulic_power_mw, 0.0), result["dynamic_max_output_mw"]
    )
    result["modeled_generation_gwh"] = result["modeled_power_mw"] * 24.0 / 1000.0
    result["observed_generation_gwh"] = result["generation_1e4kwh"] / 100.0
    result["generation_residual_gwh"] = (
        result["observed_generation_gwh"] - result["modeled_generation_gwh"]
    )

    observed = result["observed_generation_gwh"].to_numpy(float)
    modeled = result["modeled_generation_gwh"].to_numpy(float)
    mask = np.isfinite(observed) & np.isfinite(modeled)
    metrics = {
        "r2": _safe_r2(observed, modeled),
        "rmse_gwh_day": float(np.sqrt(np.mean((observed[mask] - modeled[mask]) ** 2))),
        "mae_gwh_day": float(np.mean(np.abs(observed[mask] - modeled[mask]))),
        "bias_gwh_day": float(np.mean(modeled[mask] - observed[mask])),
        "observed_total_gwh": float(np.sum(observed[mask])),
        "modeled_total_gwh": float(np.sum(modeled[mask])),
        "tailwater_rmse_m": float(np.sqrt(np.nanmean(result["tailwater_residual_m"] ** 2))),
        "tailwater_extrapolated_days": int(result["tailwater_curve_extrapolated"].sum()),
    }
    return result, metrics
