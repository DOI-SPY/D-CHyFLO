"""Generic physical-curve interface for D-CHyFLO reservoir models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


@dataclass
class CurveEvaluation:
    values: np.ndarray
    out_of_range: np.ndarray


class PhysicalCurves:
    """Load and evaluate official H-V-A, tailwater, and output curves.

    H-V-A evaluation is strict and never extrapolates. Tailwater and maximum
    output use linear endpoint extension only when needed and return an
    explicit out-of-range flag.
    """

    def __init__(self, config_dir: Path):
        self.config_dir = Path(config_dir)
        self.hva = pd.read_csv(self.config_dir / "reservoir_hva.csv")
        self.tailwater_2013 = pd.read_csv(self.config_dir / "tailwater_curve_2013.csv")
        self.tailwater_initial = pd.read_csv(self.config_dir / "tailwater_curve_initial_design.csv")
        self.unit_output = pd.read_csv(self.config_dir / "unit_max_output_curve.csv")

        self._level_to_storage = PchipInterpolator(
            self.hva["level_m"], self.hva["storage_1e8m3"], extrapolate=False
        )
        self._storage_to_level = PchipInterpolator(
            self.hva["storage_1e8m3"], self.hva["level_m"], extrapolate=False
        )
        self._level_to_area = PchipInterpolator(
            self.hva["level_m"], self.hva["area_km2"], extrapolate=False
        )
        self._area_to_level = PchipInterpolator(
            self.hva["area_km2"], self.hva["level_m"], extrapolate=False
        )
        self._tailwater = PchipInterpolator(
            self.tailwater_2013["flow_m3s"], self.tailwater_2013["tailwater_level_m"], extrapolate=False
        )
        self._tailwater_initial = PchipInterpolator(
            self.tailwater_initial["flow_m3s"], self.tailwater_initial["tailwater_level_m"], extrapolate=False
        )
        self._unit_max_output = PchipInterpolator(
            self.unit_output["upstream_level_m"], self.unit_output["unit_max_output_mw"], extrapolate=False
        )

    @staticmethod
    def _as_array(values: float | np.ndarray | pd.Series) -> np.ndarray:
        return np.asarray(values, dtype=float)

    @staticmethod
    def _strict(interpolator: PchipInterpolator, values: np.ndarray, lower: float, upper: float,
                name: str) -> np.ndarray:
        outside = (values < lower) | (values > upper)
        if np.any(outside):
            bad = values[outside]
            raise ValueError(f"{name} outside [{lower}, {upper}]: {bad[:5].tolist()}")
        return np.asarray(interpolator(values), dtype=float)

    @staticmethod
    def _linear_endpoint_extension(interpolator: PchipInterpolator, x: np.ndarray,
                                   xp: np.ndarray, yp: np.ndarray) -> CurveEvaluation:
        outside = (x < xp[0]) | (x > xp[-1])
        clipped = np.clip(x, xp[0], xp[-1])
        result = np.asarray(interpolator(clipped), dtype=float)
        low = x < xp[0]
        high = x > xp[-1]
        if np.any(low):
            slope = (yp[1] - yp[0]) / (xp[1] - xp[0])
            result[low] = yp[0] + slope * (x[low] - xp[0])
        if np.any(high):
            slope = (yp[-1] - yp[-2]) / (xp[-1] - xp[-2])
            result[high] = yp[-1] + slope * (x[high] - xp[-1])
        return CurveEvaluation(result, outside)

    def storage_from_level(self, level_m: float | np.ndarray | pd.Series) -> np.ndarray:
        x = self._as_array(level_m)
        return self._strict(
            self._level_to_storage,
            x,
            float(self.hva["level_m"].min()),
            float(self.hva["level_m"].max()),
            "upstream level",
        )

    def level_from_storage(self, storage_1e8m3: float | np.ndarray | pd.Series) -> np.ndarray:
        x = self._as_array(storage_1e8m3)
        return self._strict(
            self._storage_to_level,
            x,
            float(self.hva["storage_1e8m3"].min()),
            float(self.hva["storage_1e8m3"].max()),
            "storage",
        )

    def area_from_level(self, level_m: float | np.ndarray | pd.Series) -> np.ndarray:
        x = self._as_array(level_m)
        return self._strict(
            self._level_to_area,
            x,
            float(self.hva["level_m"].min()),
            float(self.hva["level_m"].max()),
            "upstream level",
        )

    def level_from_area(self, area_km2: float | np.ndarray | pd.Series) -> np.ndarray:
        x = self._as_array(area_km2)
        return self._strict(
            self._area_to_level,
            x,
            float(self.hva["area_km2"].min()),
            float(self.hva["area_km2"].max()),
            "surface area",
        )

    def tailwater_from_flow(self, flow_m3s: float | np.ndarray | pd.Series,
                            version: str = "2013") -> CurveEvaluation:
        x = self._as_array(flow_m3s)
        if version == "2013":
            table, curve = self.tailwater_2013, self._tailwater
        elif version == "initial_design":
            table, curve = self.tailwater_initial, self._tailwater_initial
        else:
            raise ValueError(f"Unknown tailwater curve version: {version}")
        return self._linear_endpoint_extension(
            curve,
            x,
            table["flow_m3s"].to_numpy(dtype=float),
            table["tailwater_level_m"].to_numpy(dtype=float),
        )

    def total_max_output_mw(self, upstream_level_m: float | np.ndarray | pd.Series,
                            units: int = 4, installed_capacity_mw: float = 250.0) -> CurveEvaluation:
        x = self._as_array(upstream_level_m)
        table = self.unit_output
        evaluated = self._linear_endpoint_extension(
            self._unit_max_output,
            x,
            table["upstream_level_m"].to_numpy(dtype=float),
            table["unit_max_output_mw"].to_numpy(dtype=float),
        )
        evaluated.values = np.clip(evaluated.values * units, 0.0, installed_capacity_mw)
        return evaluated
