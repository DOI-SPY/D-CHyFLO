"""Reconstruct conventional reservoir hydropower and physical checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86_400.0


def reconstruct_hydro(reservoir: pd.DataFrame, hva: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Calculate storage balance, level, net head and hydropower.

    Storage is expressed in million m3 and power in MW. Nierji is represented as a
    conventional reservoir; this calculation contains no pumping operation.
    """
    hydro = config["hydro"]
    data = reservoir.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)
    delta = (data["inflow_m3s"] - data["release_m3s"]) * SECONDS_PER_DAY / 1e6
    data["calculated_storage_million_m3"] = data["storage_million_m3"].iloc[0] + delta.cumsum()
    observed_change = data["storage_million_m3"].shift(-1) - data["storage_million_m3"]
    data["water_balance_residual_million_m3"] = observed_change - delta
    data.loc[data.index[-1], "water_balance_residual_million_m3"] = np.nan
    curve = hva.sort_values("storage_million_m3")
    data["calculated_level_m"] = np.interp(
        data["storage_million_m3"],
        curve["storage_million_m3"],
        curve["reservoir_level_m"],
    )
    tailwater = data.get("tailwater_level_m", pd.Series(0.0, index=data.index))
    data["net_head_m"] = (data["calculated_level_m"] - tailwater).clip(lower=0)
    turbine_flow = data["release_m3s"].clip(upper=float(hydro["turbine_flow_limit_m3s"]))
    power = 1000.0 * 9.80665 * float(hydro["efficiency"]) * turbine_flow * data["net_head_m"] / 1e6
    data["hydro_power_mw"] = power.clip(upper=float(hydro["capacity_mw"]))
    data["level_within_bounds"] = data["calculated_level_m"].between(
        float(hydro["minimum_level_m"]), float(hydro["maximum_level_m"])
    )
    data["power_within_bounds"] = data["hydro_power_mw"].between(0, float(hydro["capacity_mw"]))
    return data


def water_balance_pass(frame: pd.DataFrame, tolerance_million_m3: float = 0.01) -> bool:
    """Return True when all comparable daily storage balances meet tolerance."""
    residual = frame["water_balance_residual_million_m3"].dropna().abs()
    return bool((residual <= tolerance_million_m3).all())


def hourly_hydro_profile(frame: pd.DataFrame, timestamps: pd.Series) -> pd.Series:
    """Map daily reconstructed hydropower to hourly timestamps."""
    daily = frame.set_index(frame["date"].dt.date)["hydro_power_mw"]
    values = [daily[pd.Timestamp(value).date()] for value in timestamps]
    return pd.Series(values, index=pd.DatetimeIndex(timestamps), name="hydro_available_mw")
