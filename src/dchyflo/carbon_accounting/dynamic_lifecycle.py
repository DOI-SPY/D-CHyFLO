from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DynamicPathway:
    pathway_id: str
    annual_grid_decline_fraction: float
    annual_energy_degradation_fraction: float
    replacement_multiplier: float
    aquatic_response_duration_years: int
    aquatic_response_shape: str
    evidence_class: str


def aquatic_multiplier(year: int, duration: int, shape: str) -> float:
    if year <= 0 or year > duration:
        return 0.0
    if shape == "sustained":
        return 1.0
    if shape == "linear_recovery":
        if duration == 1:
            return 1.0
        return float((duration - year) / (duration - 1))
    if shape == "one_year":
        return 1.0 if year == 1 else 0.0
    raise ValueError(f"unsupported aquatic response shape: {shape}")


def dynamic_ledger(
    static_balance: pd.DataFrame,
    pathways: list[DynamicPathway],
    service_life_years: int,
    initial_manufacture_share: float,
    replacement_share: float,
    end_of_life_share: float,
    source_family: str,
) -> pd.DataFrame:
    if not np.isclose(initial_manufacture_share + replacement_share + end_of_life_share, 1.0):
        raise ValueError("technology lifecycle allocation shares must sum to one")
    records = []
    for source_row_id, row in static_balance.reset_index(drop=True).iterrows():
        base_technology_total = (
            float(row.incremental_hybrid_export_mwh)
            * float(row.technology_lifecycle_gco2e_per_kwh)
            / 1000.0
            * service_life_years
        )
        initial_burden = base_technology_total * initial_manufacture_share
        base_replacement_burden = base_technology_total * replacement_share
        eol_burden = base_technology_total * end_of_life_share
        for pathway in pathways:
            cumulative = -initial_burden
            path_key = f"{source_family}::{source_row_id}::{pathway.pathway_id}"
            common = row.to_dict()
            common.update(
                {
                    "source_family": source_family,
                    "source_row_id": source_row_id,
                    "dynamic_pathway": pathway.pathway_id,
                    "dynamic_path_key": path_key,
                    "annual_grid_decline_fraction": pathway.annual_grid_decline_fraction,
                    "annual_energy_degradation_fraction": pathway.annual_energy_degradation_fraction,
                    "replacement_multiplier": pathway.replacement_multiplier,
                    "aquatic_response_duration_years": pathway.aquatic_response_duration_years,
                    "aquatic_response_shape": pathway.aquatic_response_shape,
                    "dynamic_evidence_class": pathway.evidence_class,
                    "base_technology_lifecycle_total_tco2e": base_technology_total,
                }
            )
            records.append(
                {
                    **common,
                    "lifecycle_year": 0,
                    "grid_factor_kgco2e_per_mwh_dynamic": float(row.grid_factor_kgco2e_per_mwh),
                    "incremental_export_mwh_dynamic": 0.0,
                    "avoided_grid_tco2e": 0.0,
                    "initial_manufacture_tco2e": initial_burden,
                    "replacement_tco2e": 0.0,
                    "end_of_life_tco2e": 0.0,
                    "local_aquatic_delta_tco2e": 0.0,
                    "annual_net_avoided_tco2e": -initial_burden,
                    "cumulative_net_avoided_tco2e": cumulative,
                }
            )
            for year in range(1, service_life_years + 1):
                grid_factor = float(row.grid_factor_kgco2e_per_mwh) * (
                    (1.0 - pathway.annual_grid_decline_fraction) ** (year - 1)
                )
                energy = float(row.incremental_hybrid_export_mwh) * (
                    (1.0 - pathway.annual_energy_degradation_fraction) ** (year - 1)
                )
                avoided = energy * grid_factor / 1000.0
                replacement = (
                    base_replacement_burden * pathway.replacement_multiplier
                    if year == 15
                    else 0.0
                )
                end_of_life = eol_burden if year == service_life_years else 0.0
                aquatic = float(row.local_aquatic_delta_tco2e_yr) * aquatic_multiplier(
                    year,
                    pathway.aquatic_response_duration_years,
                    pathway.aquatic_response_shape,
                )
                annual_net = avoided - replacement - end_of_life - aquatic
                cumulative += annual_net
                records.append(
                    {
                        **common,
                        "lifecycle_year": year,
                        "grid_factor_kgco2e_per_mwh_dynamic": grid_factor,
                        "incremental_export_mwh_dynamic": energy,
                        "avoided_grid_tco2e": avoided,
                        "initial_manufacture_tco2e": 0.0,
                        "replacement_tco2e": replacement,
                        "end_of_life_tco2e": end_of_life,
                        "local_aquatic_delta_tco2e": aquatic,
                        "annual_net_avoided_tco2e": annual_net,
                        "cumulative_net_avoided_tco2e": cumulative,
                    }
                )
    return pd.DataFrame(records)


def summarize_paths(ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for path_key, group in ledger.groupby("dynamic_path_key", sort=False):
        group = group.sort_values("lifecycle_year")
        operating = group.loc[group.lifecycle_year.gt(0)]
        nonnegative = group.loc[group.cumulative_net_avoided_tco2e.ge(0.0)]
        first_nonnegative = int(nonnegative.lifecycle_year.min()) if not nonnegative.empty else None
        row = group.iloc[0]
        rows.append(
            {
                "dynamic_path_key": path_key,
                "source_family": row.source_family,
                "source_row_id": int(row.source_row_id),
                "capacity_mwac": float(row.capacity_mwac),
                "proxy_scenario": row.proxy_scenario,
                "dynamic_pathway": row.dynamic_pathway,
                "s0_reference": row.s0_reference,
                "year": int(row.year),
                "signal_name": row.signal_name,
                "cold_scenario": row.get("cold_scenario", "NOT_APPLICABLE"),
                "capacity_source_status": row.get("capacity_source_status", "CP6A_EXACT_CAPACITY"),
                "lifetime_export_mwh": float(operating.incremental_export_mwh_dynamic.sum()),
                "lifetime_avoided_grid_tco2e": float(operating.avoided_grid_tco2e.sum()),
                "lifetime_initial_manufacture_tco2e": float(group.initial_manufacture_tco2e.sum()),
                "lifetime_replacement_tco2e": float(group.replacement_tco2e.sum()),
                "lifetime_end_of_life_tco2e": float(group.end_of_life_tco2e.sum()),
                "lifetime_local_aquatic_delta_tco2e": float(group.local_aquatic_delta_tco2e.sum()),
                "lifetime_net_avoided_tco2e": float(group.iloc[-1].cumulative_net_avoided_tco2e),
                "carbon_payback_year": first_nonnegative,
                "positive_at_end_of_life": bool(group.iloc[-1].cumulative_net_avoided_tco2e > 0.0),
                "dynamic_claim_status": "CONDITIONAL_DYNAMIC_SCENARIO_NOT_FORECAST",
            }
        )
    return pd.DataFrame(rows)


def capacity_envelope(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.groupby(["source_family", "capacity_mwac", "dynamic_pathway"], as_index=False).agg(
        paths=("dynamic_path_key", "count"),
        lifetime_net_min_tco2e=("lifetime_net_avoided_tco2e", "min"),
        lifetime_net_median_tco2e=("lifetime_net_avoided_tco2e", "median"),
        lifetime_net_max_tco2e=("lifetime_net_avoided_tco2e", "max"),
        payback_year_min=("carbon_payback_year", "min"),
        payback_year_median=("carbon_payback_year", "median"),
        payback_year_max=("carbon_payback_year", "max"),
        all_paths_positive=("positive_at_end_of_life", "all"),
    )
