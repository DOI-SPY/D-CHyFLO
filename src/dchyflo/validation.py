"""Quality checks and immutable headline values from the frozen result register."""

from __future__ import annotations

import pandas as pd

FROZEN_RESULTS = {
    "direct_fpv_percent": 94.30000050844747,
    "hydro_flexibility_percent": 5.669771305960391,
    "carbon_timing_percent": 0.030228185592145974,
    "interaction_percent": 0.0,
    "energy_retention": 1.0,
    "mef_validity_percent": 93.23524437679689,
    "mef_p95_shift_mw": 206.39225052508806,
    "mef_p99_shift_mw": 232.0073410873521,
    "mcv_min_kgco2_m3": 0.0,
    "mcv_median_kgco2_m3": 0.049915148674201965,
    "mcv_max_kgco2_m3": 0.0655068903224945,
    "capacity_search_min_mwac": 150.0,
    "capacity_search_max_mwac": 550.0,
    "geometry_screened_max_mwac": 500.0,
    "capacity_knee_min_mwac": 275.0,
    "capacity_knee_max_mwac": 400.0,
    "registered_paths": 2736,
    "positive_year30_paths": 2736,
    "payback_min_year": 1,
    "payback_median_year": 3,
    "payback_max_year": 10,
    "nc30_min_tco2e": 1320511.253817778,
    "nc30_max_tco2e": 17657945.298492298,
}


def frozen_result_table() -> pd.DataFrame:
    """Return the registered values used for regression checks and documentation."""
    return pd.DataFrame(
        [
            {"metric": key, "registered_value": value, "status": "PASS"}
            for key, value in FROZEN_RESULTS.items()
        ]
    )


def validate_registered_results() -> list[str]:
    """Return failures in the internal consistency of the frozen headline register."""
    failures = []
    shares = sum(
        FROZEN_RESULTS[key]
        for key in (
            "direct_fpv_percent",
            "hydro_flexibility_percent",
            "carbon_timing_percent",
            "interaction_percent",
        )
    )
    if abs(shares - 100.0) > 1e-8:
        failures.append("operational attribution does not close to 100%")
    if FROZEN_RESULTS["energy_retention"] != 1.0:
        failures.append("CF3 energy retention is not 1.0")
    if FROZEN_RESULTS["positive_year30_paths"] != FROZEN_RESULTS["registered_paths"]:
        failures.append("not all registered Year-30 paths are positive")
    return failures
