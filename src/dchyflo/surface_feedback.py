"""Dynamic water-surface feasibility and first-order evaporation accounting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .curves import PhysicalCurves


@dataclass(frozen=True)
class SurfaceCapacityRequirement:
    capacity_mwac: float
    footprint_km2: float
    required_surface_area_km2: float
    minimum_level_m: float
    minimum_storage_1e8m3: float


def surface_capacity_requirement(
    curves: PhysicalCurves,
    capacity_mwac: float,
    dc_ac_ratio: float,
    surface_power_density_mwp_per_km2: float,
    coverage_limit_fraction: float,
) -> SurfaceCapacityRequirement:
    """Translate fixed installed capacity into a minimum reservoir state."""

    if capacity_mwac < 0.0:
        raise ValueError("capacity_mwac must be non-negative")
    if dc_ac_ratio <= 0.0 or surface_power_density_mwp_per_km2 <= 0.0:
        raise ValueError("DC/AC ratio and surface power density must be positive")
    if not 0.0 < coverage_limit_fraction <= 1.0:
        raise ValueError("coverage_limit_fraction must lie in (0, 1]")
    footprint = capacity_mwac * dc_ac_ratio / surface_power_density_mwp_per_km2
    required_area = footprint / coverage_limit_fraction
    minimum_level = float(curves.level_from_area(np.array([required_area]))[0])
    minimum_storage = float(curves.storage_from_level(np.array([minimum_level]))[0])
    return SurfaceCapacityRequirement(
        capacity_mwac=float(capacity_mwac),
        footprint_km2=float(footprint),
        required_surface_area_km2=float(required_area),
        minimum_level_m=minimum_level,
        minimum_storage_1e8m3=minimum_storage,
    )


def evaluate_surface_trajectory(
    dates: pd.Series,
    storage_1e8m3: np.ndarray | pd.Series,
    curves: PhysicalCurves,
    requirement: SurfaceCapacityRequirement,
    coverage_limit_fraction: float,
    trajectory_label: str,
) -> pd.DataFrame:
    """Evaluate day-scale fixed-array feasibility without changing capacity."""

    storage = np.asarray(storage_1e8m3, dtype=float)
    level = curves.level_from_storage(storage)
    area = curves.area_from_level(level)
    coverage = np.divide(
        requirement.footprint_km2,
        area,
        out=np.zeros_like(area),
        where=area > 0.0,
    )
    return pd.DataFrame({
        "date": pd.to_datetime(dates).to_numpy(),
        "trajectory": trajectory_label,
        "capacity_mwac": requirement.capacity_mwac,
        "storage_1e8m3": storage,
        "upstream_level_m": level,
        "surface_area_km2": area,
        "fpv_footprint_km2": requirement.footprint_km2,
        "coverage_fraction": coverage,
        "coverage_percent": 100.0 * coverage,
        "coverage_limit_percent": 100.0 * coverage_limit_fraction,
        "surface_area_margin_km2": area - requirement.required_surface_area_km2,
        "surface_feasible": coverage <= coverage_limit_fraction + 1.0e-12,
    })


def evaporation_saving_account(
    dates: pd.Series,
    nonlinear_head_m: np.ndarray | pd.Series,
    monthly_evaporation: pd.DataFrame,
    footprint_km2: float,
    suppression_efficiency: float,
    output_coefficient: float,
) -> pd.DataFrame:
    """First-order water saving and potential hydro opportunity.

    Saved water is not injected back into dispatch. The energy column is the
    potential energy if the saved water is eventually turbined at the same
    day's nonlinear head; it is therefore an opportunity quantity, not an
    optimized generation claim.
    """

    if not 0.0 <= suppression_efficiency <= 1.0:
        raise ValueError("suppression_efficiency must lie in [0, 1]")
    frame = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "nonlinear_head_m": np.asarray(nonlinear_head_m, dtype=float),
    })
    evaporation_by_month = monthly_evaporation.set_index("month")[
        "evaporation_increment_mm"
    ].astype(float)
    frame["monthly_evaporation_increment_mm"] = frame["date"].dt.month.map(
        evaporation_by_month
    )
    frame["days_in_month"] = frame["date"].dt.days_in_month
    frame["daily_evaporation_increment_mm"] = (
        frame["monthly_evaporation_increment_mm"] / frame["days_in_month"]
    )
    frame["suppression_efficiency"] = suppression_efficiency
    frame["saved_water_m3"] = (
        frame["daily_evaporation_increment_mm"]
        / 1000.0
        * footprint_km2
        * 1.0e6
        * suppression_efficiency
    )
    frame["potential_hydro_mwh"] = (
        frame["saved_water_m3"]
        * output_coefficient
        * frame["nonlinear_head_m"]
        / 3.6e6
    )
    return frame

