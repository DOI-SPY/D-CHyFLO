"""Read configuration and tabular inputs with clear field checks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


def load_config(path: str | Path) -> dict:
    """Return a YAML configuration and resolve no paths implicitly."""
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration must contain a YAML mapping")
    return config


def read_csv_checked(path: str | Path, required: Iterable[str]) -> pd.DataFrame:
    """Read a CSV and fail with the missing field names."""
    frame = pd.read_csv(path)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required fields: {', '.join(missing)}")
    return frame


def load_example_inputs(config: dict) -> dict[str, pd.DataFrame]:
    """Load the four public input tables used by the example workflow."""
    paths = config["data"]
    reservoir = read_csv_checked(
        paths["reservoir"],
        ["date", "inflow_m3s", "release_m3s", "storage_million_m3", "reservoir_level_m"],
    )
    hva = read_csv_checked(paths["hva"], ["reservoir_level_m", "storage_million_m3", "area_km2"])
    meteorology = read_csv_checked(
        paths["meteorology"],
        [
            "timestamp",
            "ghi_wm2",
            "dni_wm2",
            "dhi_wm2",
            "temp_air_c",
            "wind_speed_ms",
            "grid_demand_mw",
        ],
    )
    grid = read_csv_checked(
        paths["grid"],
        ["generator", "capacity_mw", "marginal_cost_per_mwh", "emission_kgco2_per_mwh"],
    )
    return {"reservoir": reservoir, "hva": hva, "meteorology": meteorology, "grid": grid}
