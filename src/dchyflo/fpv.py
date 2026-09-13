"""Calculate hourly floating-PV generation from meteorological inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib


def simulate_fpv(
    meteorology: pd.DataFrame,
    config: dict,
    capacity_mwac: float | None = None,
    dc_ac_ratio: float | None = None,
    winter_scenario: str = "moderate_cold",
) -> pd.DataFrame:
    """Calculate POA irradiance, temperature, DC output, clipping and AC output."""
    fpv = config["fpv"]
    ac_capacity = float(capacity_mwac if capacity_mwac is not None else fpv["capacity_mwac"])
    ratio = float(dc_ac_ratio if dc_ac_ratio is not None else fpv["dc_ac_ratio"])
    data = meteorology.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    times = pd.DatetimeIndex(data["timestamp"])
    position = pvlib.solarposition.get_solarposition(
        times, float(fpv["latitude"]), float(fpv["longitude"])
    )
    poa = (
        pvlib.irradiance.get_total_irradiance(
            surface_tilt=float(fpv["tilt_degrees"]),
            surface_azimuth=float(fpv["azimuth_degrees"]),
            solar_zenith=position["apparent_zenith"],
            solar_azimuth=position["azimuth"],
            dni=data["dni_wm2"].to_numpy(),
            ghi=data["ghi_wm2"].to_numpy(),
            dhi=data["dhi_wm2"].to_numpy(),
        )["poa_global"]
        .fillna(0)
        .clip(lower=0)
    )
    cell_temp = pvlib.temperature.sapm_cell(
        poa,
        data["temp_air_c"].to_numpy(),
        data["wind_speed_ms"].to_numpy(),
        **pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"],
    )
    dc_capacity = ac_capacity * ratio
    dc_values = dc_capacity * (poa / 1000.0) * (1 - 0.004 * (cell_temp - 25.0))
    dc = pd.Series(np.maximum(np.asarray(dc_values, dtype=float), 0.0), index=data.index)
    available = np.minimum(dc, ac_capacity)
    availability = float(fpv["winter_scenarios"][winter_scenario])
    month = data["timestamp"].dt.month
    winter_factor = np.where(month.isin([11, 12, 1, 2, 3]), availability, 1.0)
    data["poa_wm2"] = np.asarray(poa)
    data["cell_temperature_c"] = np.asarray(cell_temp)
    data["dc_output_mw"] = dc.to_numpy()
    data["inverter_clipping_mw"] = (dc - available).clip(lower=0).to_numpy()
    data["fpv_available_mw"] = available.to_numpy() * winter_factor
    return data
