from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProxyScenario:
    scenario_id: str
    grid_factor_kgco2e_per_mwh: float
    fpv_full_lifecycle_gco2e_per_kwh: float
    generic_aquatic_share_fraction: float
    local_aquatic_response_fraction: float
    reservoir_flux_quantile: str
    evidence_class: str

    @property
    def technology_lifecycle_gco2e_per_kwh(self) -> float:
        """Remove the generic aquatic share before adding the local response delta."""
        return self.fpv_full_lifecycle_gco2e_per_kwh * (
            1.0 - self.generic_aquatic_share_fraction
        )


def unique_dispatch_states(unified: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dispatch_case_key",
        "s0_reference",
        "year",
        "hydrologic_regime",
        "capacity_mwac",
        "signal_name",
        "factor_type",
        "signal_role",
        "mean_factor_kgco2_per_mwh",
        "optimized_export_mwh",
        "hydro_control_optimized_export_mwh",
        "incremental_hybrid_export_mwh",
    ]
    return unified[columns].drop_duplicates().sort_values(
        ["capacity_mwac", "dispatch_case_key", "signal_name"]
    ).reset_index(drop=True)


def cold_dispatch_states(stress: pd.DataFrame) -> pd.DataFrame:
    controls = stress.loc[stress.capacity_mwac.eq(0.0), [
        "s0_reference", "year", "optimized_export_mwh"
    ]].rename(columns={"optimized_export_mwh": "hydro_control_optimized_export_mwh"})
    positive = stress.loc[stress.capacity_mwac.gt(0.0)].copy()
    positive = positive.merge(controls, on=["s0_reference", "year"], how="left", validate="many_to_one")
    positive["incremental_hybrid_export_mwh"] = (
        positive.optimized_export_mwh - positive.hydro_control_optimized_export_mwh
    )
    return positive


def capacity_footprints(foreground: pd.DataFrame) -> pd.DataFrame:
    grouped = foreground.groupby("capacity_mwac")["project_screened_footprint_km2"]
    result = grouped.agg(["min", "median", "max"]).reset_index()
    result.columns = [
        "capacity_mwac",
        "footprint_min_km2",
        "footprint_median_km2",
        "footprint_max_km2",
    ]
    return result


def partial_inventory_envelope(climate: pd.DataFrame, service_life_years: float) -> pd.DataFrame:
    rows = []
    for capacity, group in climate.groupby("capacity_mwac"):
        rows.append(
            {
                "capacity_mwac": float(capacity),
                "partial_inventory_low_tco2e_yr": float(group.partial_core_low_tco2e.min() / service_life_years),
                "partial_inventory_central_tco2e_yr": float(group.partial_core_central_tco2e.median() / service_life_years),
                "partial_inventory_high_tco2e_yr": float(group.partial_core_high_tco2e.max() / service_life_years),
            }
        )
    return pd.DataFrame(rows)


def reservoir_flux_anchors(reservoir: pd.DataFrame) -> dict[str, float]:
    values = reservoir.net_ghg_gco2e_m2_yr.astype(float)
    return {
        "minimum": float(values.min()),
        "median": float(values.median()),
        "maximum": float(values.max()),
    }


def calculate_balance(
    dispatch: pd.DataFrame,
    footprints: pd.DataFrame,
    partial_inventory: pd.DataFrame,
    scenarios: Iterable[ProxyScenario],
    flux_anchors: dict[str, float],
    track: str,
) -> pd.DataFrame:
    base = dispatch.merge(footprints, on="capacity_mwac", how="left", validate="many_to_one")
    base = base.merge(partial_inventory, on="capacity_mwac", how="left", validate="many_to_one")
    rows = []
    for scenario in scenarios:
        frame = base.copy()
        frame["proxy_scenario"] = scenario.scenario_id
        frame["accounting_track"] = track
        frame["grid_factor_kgco2e_per_mwh"] = scenario.grid_factor_kgco2e_per_mwh
        frame["fpv_full_lifecycle_gco2e_per_kwh"] = scenario.fpv_full_lifecycle_gco2e_per_kwh
        frame["generic_aquatic_share_fraction"] = scenario.generic_aquatic_share_fraction
        frame["technology_lifecycle_gco2e_per_kwh"] = scenario.technology_lifecycle_gco2e_per_kwh
        frame["local_aquatic_response_fraction"] = scenario.local_aquatic_response_fraction
        frame["reservoir_flux_anchor"] = scenario.reservoir_flux_quantile
        frame["reservoir_flux_gco2e_m2_yr"] = flux_anchors[scenario.reservoir_flux_quantile]
        frame["evidence_class"] = scenario.evidence_class

        if scenario.scenario_id == "favourable":
            footprint = frame.footprint_min_km2
            partial_col = "partial_inventory_low_tco2e_yr"
        elif scenario.scenario_id == "conservative":
            footprint = frame.footprint_max_km2
            partial_col = "partial_inventory_high_tco2e_yr"
        else:
            footprint = frame.footprint_median_km2
            partial_col = "partial_inventory_central_tco2e_yr"

        frame["fpv_footprint_km2"] = footprint
        frame["avoided_grid_tco2e_yr"] = (
            frame.incremental_hybrid_export_mwh * scenario.grid_factor_kgco2e_per_mwh / 1000.0
        )
        frame["technology_lifecycle_tco2e_yr"] = (
            frame.incremental_hybrid_export_mwh * scenario.technology_lifecycle_gco2e_per_kwh / 1000.0
        )
        frame["local_aquatic_delta_tco2e_yr"] = (
            frame.reservoir_flux_gco2e_m2_yr
            * frame.fpv_footprint_km2
            * 1_000_000.0
            * scenario.local_aquatic_response_fraction
            / 1_000_000.0
        )
        frame["total_incremental_burden_tco2e_yr"] = (
            frame.technology_lifecycle_tco2e_yr + frame.local_aquatic_delta_tco2e_yr
        )
        frame["net_avoided_tco2e_yr"] = (
            frame.avoided_grid_tco2e_yr - frame.total_incremental_burden_tco2e_yr
        )
        frame["net_avoided_gco2e_per_kwh"] = (
            frame.net_avoided_tco2e_yr * 1000.0 / frame.incremental_hybrid_export_mwh
        )
        frame["static_net_carbon_status"] = np.where(
            frame.net_avoided_tco2e_yr.gt(0.0), "POSITIVE", "NON_POSITIVE"
        )
        frame["partial_inventory_crosscheck_tco2e_yr"] = frame[partial_col]
        frame["partial_inventory_crosscheck_gco2e_per_kwh"] = (
            frame[partial_col] * 1000.0 / frame.incremental_hybrid_export_mwh
        )
        frame["site_validation_status"] = "HOLD_NOT_SITE_VALIDATED"
        frame["claim_status"] = "REGIONAL_PROXY_SIMULATED_STATIC_LIFECYCLE_NET_CARBON"
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def capacity_envelope(balance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for capacity, group in balance.groupby("capacity_mwac"):
        conservative = group.loc[group.proxy_scenario.eq("conservative")]
        central = group.loc[group.proxy_scenario.eq("central")]
        favourable = group.loc[group.proxy_scenario.eq("favourable")]
        rows.append(
            {
                "capacity_mwac": float(capacity),
                "states": len(group),
                "incremental_export_min_mwh": float(group.incremental_hybrid_export_mwh.min()),
                "incremental_export_median_mwh": float(group.incremental_hybrid_export_mwh.median()),
                "incremental_export_max_mwh": float(group.incremental_hybrid_export_mwh.max()),
                "net_avoided_favourable_max_tco2e_yr": float(favourable.net_avoided_tco2e_yr.max()),
                "net_avoided_central_median_tco2e_yr": float(central.net_avoided_tco2e_yr.median()),
                "net_avoided_conservative_min_tco2e_yr": float(conservative.net_avoided_tco2e_yr.min()),
                "all_proxy_states_positive": bool(group.static_net_carbon_status.eq("POSITIVE").all()),
                "site_validation_status": "HOLD_NOT_SITE_VALIDATED",
            }
        )
    return pd.DataFrame(rows)
