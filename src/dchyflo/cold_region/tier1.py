from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def cold_scenario_contract(model_config: Mapping[str, object]) -> pd.DataFrame:
    scenarios = model_config["fpv"]["cold_scenarios"]
    rows = []
    for scenario_id, values in scenarios.items():
        rows.append(
            {
                "scenario_id": scenario_id,
                "snow_model": str(values.get("snow_model", False)),
                "base_availability": float(values["base_availability"]),
                "snow_phase_temperature_c": values.get("snow_phase_temperature_c", np.nan),
                "snowfall_cm_per_mm_water": values.get("snowfall_cm_per_mm_water", np.nan),
                "threshold_snowfall_cm_h": values.get("threshold_snowfall_cm_h", np.nan),
                "slide_amount_coefficient": values.get("slide_amount_coefficient", np.nan),
                "module_substrings": values.get("module_substrings", np.nan),
                "extreme_cold_threshold_c": values.get("extreme_cold_threshold_c", np.nan),
                "extreme_cold_availability": values.get("extreme_cold_availability", np.nan),
                "high_wind_threshold_m_s": values.get("high_wind_threshold_m_s", np.nan),
                "high_wind_availability": values.get("high_wind_availability", np.nan),
                "calibration_status": "PARAMETRIC_BRACKET_NOT_CALIBRATED_TO_LOCAL_OPERATIONS",
                "engineering_status": "NOT_A_STRUCTURAL_OR_MOORING_SAFETY_MODEL",
            }
        )
    return pd.DataFrame(rows)


def annual_tier1_metrics(
    hourly: pd.DataFrame,
    scenario_contract: pd.DataFrame,
) -> pd.DataFrame:
    required = {"timestamp_lst", "ac_clear_mw_per_mwac"}
    for scenario_id in scenario_contract.scenario_id:
        required |= {
            f"snow_coverage_{scenario_id}",
            f"availability_{scenario_id}",
            f"ac_mw_per_mwac_{scenario_id}",
        }
    missing = required.difference(hourly.columns)
    if missing:
        raise ValueError(f"missing Tier-1 FPV columns: {sorted(missing)}")
    frame = hourly.copy()
    frame["timestamp_lst"] = pd.to_datetime(frame["timestamp_lst"])
    if frame.timestamp_lst.duplicated().any():
        raise ValueError("Tier-1 FPV timestamps must be unique")
    frame["year"] = frame.timestamp_lst.dt.year
    no_cold = frame.groupby("year")["ac_mw_per_mwac_no_cold_derate"].sum()
    rows = []
    for scenario in scenario_contract.itertuples(index=False):
        scenario_id = scenario.scenario_id
        energy_column = f"ac_mw_per_mwac_{scenario_id}"
        availability_column = f"availability_{scenario_id}"
        snow_column = f"snow_coverage_{scenario_id}"
        for year, group in frame.groupby("year"):
            energy = float(group[energy_column].sum())
            baseline_energy = float(no_cold.loc[year])
            daylight = group.ac_clear_mw_per_mwac > 0.0
            rows.append(
                {
                    "year": int(year),
                    "scenario_id": scenario_id,
                    "hours": len(group),
                    "energy_yield_mwh_per_mwac": energy,
                    "no_cold_energy_yield_mwh_per_mwac": baseline_energy,
                    "cold_energy_loss_mwh_per_mwac_vs_no_cold": baseline_energy - energy,
                    "cold_energy_loss_fraction_vs_no_cold": 1.0 - energy / baseline_energy,
                    "mean_availability": float(group[availability_column].mean()),
                    "minimum_availability": float(group[availability_column].min()),
                    "hours_below_base_availability": int(
                        (group[availability_column] < float(scenario.base_availability) - 1e-12).sum()
                    ),
                    "mean_snow_coverage_all_hours": float(group[snow_column].mean()),
                    "mean_snow_coverage_daylight": float(group.loc[daylight, snow_column].mean()),
                    "hours_snow_coverage_gt_0_1": int((group[snow_column] > 0.1).sum()),
                    "scenario_status": scenario.calibration_status,
                    "engineering_status": scenario.engineering_status,
                }
            )
    return pd.DataFrame(rows).sort_values(["year", "scenario_id"]).reset_index(drop=True)


def annual_cold_wide(annual: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "energy_yield_mwh_per_mwac",
        "cold_energy_loss_fraction_vs_no_cold",
        "mean_availability",
        "mean_snow_coverage_daylight",
        "hours_below_base_availability",
        "hours_snow_coverage_gt_0_1",
    ]
    wide = annual.pivot(index="year", columns="scenario_id", values=fields)
    wide.columns = [f"tier1_{scenario}_{field}" for field, scenario in wide.columns]
    return wide.reset_index()
