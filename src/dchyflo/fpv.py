"""Hourly floating-PV physics and transparent cold-region scenarios."""

from __future__ import annotations

from datetime import timedelta, timezone
from typing import Any

import pandas as pd
import pvlib


def _snow_loss(weather, poa_global, tilt_deg, scenario):
    snowing = weather["temperature_c"] <= scenario["snow_phase_temperature_c"]
    snowfall_cm = (
        weather["precipitation_mm"]
        * scenario["snowfall_cm_per_mm_water"]
        * snowing.astype(float)
    )
    coverage = pvlib.snow.coverage_nrel(
        snowfall=snowfall_cm,
        poa_irradiance=poa_global,
        temp_air=weather["temperature_c"],
        surface_tilt=tilt_deg,
        threshold_snowfall=scenario["threshold_snowfall_cm_h"],
        slide_amount_coefficient=scenario["slide_amount_coefficient"],
    )
    loss = pvlib.snow.dc_loss_nrel(
        coverage, num_strings=int(scenario["module_substrings"])
    )
    return coverage.clip(0.0, 1.0), pd.Series(loss, index=weather.index).clip(0.0, 1.0)


def simulate_unit_fpv(
    weather: pd.DataFrame, site: dict[str, Any], fpv: dict[str, Any]
) -> pd.DataFrame:
    """Simulate hourly output per 1 MWac of installed FPV."""
    required = {
        "timestamp_lst", "ghi_wh_m2", "dni_wh_m2", "dhi_wh_m2",
        "temperature_c", "wind_speed_10m_m_s", "precipitation_mm",
    }
    missing = required - set(weather.columns)
    if missing:
        raise KeyError(f"Hourly weather missing columns: {sorted(missing)}")
    frame = weather.copy()
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp_lst"]))
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize(timezone(timedelta(hours=8)))
    frame.index = timestamps

    # POWER's timestamp labels an hourly interval.  Using its midpoint gives
    # materially better GHI = DNI*cos(zenith) + DHI closure at this site.
    solar_timestamps = timestamps + pd.Timedelta(minutes=30)
    solar = pvlib.solarposition.get_solarposition(
        solar_timestamps, float(site["latitude_deg"]), float(site["longitude_deg"])
    )
    solar.index = timestamps
    dni_extra = pd.Series(
        pvlib.irradiance.get_extra_radiation(solar_timestamps).to_numpy(),
        index=timestamps,
    )
    airmass = pvlib.atmosphere.get_relative_airmass(solar["apparent_zenith"])
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=float(fpv["tilt_deg"]),
        surface_azimuth=float(fpv["azimuth_deg"]),
        solar_zenith=solar["apparent_zenith"],
        solar_azimuth=solar["azimuth"],
        dni=frame["dni_wh_m2"], ghi=frame["ghi_wh_m2"], dhi=frame["dhi_wh_m2"],
        dni_extra=dni_extra, airmass=airmass, albedo=0.08, model="perez",
    )
    poa_global = poa["poa_global"].fillna(0.0).clip(lower=0.0)
    temp_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]
    cell_temperature = pvlib.temperature.sapm_cell(
        poa_global, frame["temperature_c"], frame["wind_speed_10m_m_s"], **temp_params
    )
    dc_w = pvlib.pvsystem.pvwatts_dc(
        poa_global, cell_temperature,
        pdc0=1_000_000.0 * float(fpv["dc_ac_ratio"]),
        gamma_pdc=float(fpv["gamma_pdc_per_c"]),
    ).clip(lower=0.0)
    ac_clear_w = pvlib.inverter.pvwatts(
        dc_w, pdc0=1_000_000.0 / float(fpv["inverter_efficiency"]),
        eta_inv_nom=float(fpv["inverter_efficiency"]),
    ).clip(lower=0.0, upper=1_000_000.0)

    out = pd.DataFrame({
        "timestamp_lst": timestamps.tz_localize(None),
        "solar_zenith_deg": solar["apparent_zenith"].to_numpy(),
        "poa_global_w_m2": poa_global.to_numpy(),
        "cell_temperature_c": cell_temperature.to_numpy(),
        "ac_clear_mw_per_mwac": ac_clear_w.to_numpy() / 1_000_000.0,
    })
    for name, scenario in fpv["cold_scenarios"].items():
        availability = pd.Series(float(scenario["base_availability"]), index=frame.index)
        if scenario.get("snow_model"):
            coverage, snow_loss = _snow_loss(frame, poa_global, float(fpv["tilt_deg"]), scenario)
            availability = availability.where(
                frame["temperature_c"] > scenario["extreme_cold_threshold_c"],
                availability * float(scenario["extreme_cold_availability"]),
            )
            availability = availability.where(
                frame["wind_speed_10m_m_s"] <= scenario["high_wind_threshold_m_s"],
                availability * float(scenario["high_wind_availability"]),
            )
        else:
            coverage = pd.Series(0.0, index=frame.index)
            snow_loss = pd.Series(0.0, index=frame.index)
        out[f"snow_coverage_{name}"] = coverage.to_numpy()
        out[f"availability_{name}"] = availability.to_numpy()
        out[f"ac_mw_per_mwac_{name}"] = (
            ac_clear_w * (1.0 - snow_loss) * availability / 1_000_000.0
        ).to_numpy()
    return out
