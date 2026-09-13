"""Apply the shared export constraint and retained-energy redispatch."""

from __future__ import annotations

import numpy as np
import pandas as pd


def shared_export(hydro_mw, fpv_available_mw, export_limit_mw: float) -> pd.DataFrame:
    """Allocate the shared channel to hydropower first and report FPV use and congestion."""
    hydro = np.minimum(np.asarray(hydro_mw, dtype=float), export_limit_mw)
    fpv_available = np.maximum(np.asarray(fpv_available_mw, dtype=float), 0.0)
    fpv_used = np.minimum(fpv_available, np.maximum(export_limit_mw - hydro, 0.0))
    joint = hydro + fpv_used
    return pd.DataFrame(
        {
            "hydro_export_mw": hydro,
            "fpv_available_mw": fpv_available,
            "fpv_used_mw": fpv_used,
            "curtailment_mw": fpv_available - fpv_used,
            "joint_export_mw": joint,
            "export_congestion": np.isclose(joint, export_limit_mw, atol=1e-4),
        }
    )


def redispatch_hydro(
    baseline_hydro_mw,
    fpv_available_mw,
    export_limit_mw: float,
    carbon_signal=None,
    energy_retention: float = 1.0,
) -> np.ndarray:
    """Move hydro away from FPV congestion while retaining the required energy.

    The routine is a transparent example scheduler. Formal case studies may replace it
    with a solver-backed reservoir optimization while preserving this interface.
    """
    baseline = np.asarray(baseline_hydro_mw, dtype=float)
    fpv = np.asarray(fpv_available_mw, dtype=float)
    upper = np.maximum(export_limit_mw - fpv, 0.0)
    target = baseline.sum() * float(energy_retention)
    result = np.minimum(baseline, upper)
    remaining = target - result.sum()
    if carbon_signal is None:
        score = fpv
    else:
        score = -np.asarray(carbon_signal, dtype=float)
    order = np.argsort(score)
    for index in order:
        addition = min(max(upper[index] - result[index], 0.0), remaining)
        result[index] += addition
        remaining -= addition
        if remaining <= 1e-9:
            break
    if remaining > 1e-6:
        raise ValueError("retained hydro energy is infeasible under the shared export limit")
    return result
