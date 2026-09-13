"""Screen FPV capacity and DC/AC designs under geometry and export limits."""

from __future__ import annotations

import pandas as pd

from .dispatch import shared_export
from .fpv import simulate_fpv


def design_area_km2(
    capacity_mwac: float, dc_ac_ratio: float, power_density_mwp_km2: float
) -> float:
    """Convert an AC capacity and DC/AC ratio to occupied water area."""
    return float(capacity_mwac) * float(dc_ac_ratio) / float(power_density_mwp_km2)


def geometry_screen(
    capacity_mwac: float,
    dc_ac_ratio: float,
    minimum_reservoir_area_km2: float,
    config: dict,
) -> bool:
    """Apply the registered 2.8% geometric water-coverage screen."""
    fpv = config["fpv"]
    area = design_area_km2(
        capacity_mwac, dc_ac_ratio, float(fpv["surface_power_density_mwp_per_km2"])
    )
    coverage = 100.0 * area / float(minimum_reservoir_area_km2)
    return bool(coverage <= float(fpv["coverage_screen_percent"]) + 1e-12)


def run_capacity_scan(
    meteorology: pd.DataFrame,
    hydro_mw,
    minimum_reservoir_area_km2: float,
    config: dict,
) -> pd.DataFrame:
    """Calculate clipping, curtailment and congestion for each registered design."""
    rows = []
    limit = float(config["hydro"]["export_limit_mw"])
    density = float(config["fpv"]["surface_power_density_mwp_per_km2"])
    for capacity in config["capacity"]["capacity_mwac"]:
        for ratio in config["capacity"]["dc_ac_ratios"]:
            feasible = geometry_screen(capacity, ratio, minimum_reservoir_area_km2, config)
            fpv = simulate_fpv(meteorology, config, capacity, ratio)
            dispatch = shared_export(hydro_mw, fpv["fpv_available_mw"], limit)
            rows.append(
                {
                    "capacity_mwac": float(capacity),
                    "dc_ac_ratio": float(ratio),
                    "water_coverage_percent": 100
                    * design_area_km2(capacity, ratio, density)
                    / minimum_reservoir_area_km2,
                    "geometry_screen_pass": feasible,
                    "fpv_available_mwh": fpv["fpv_available_mw"].sum(),
                    "clipping_mwh": fpv["inverter_clipping_mw"].sum(),
                    "curtailment_mwh": dispatch["curtailment_mw"].sum(),
                    "congested_hours": int(dispatch["export_congestion"].sum()),
                }
            )
    return pd.DataFrame(rows)
