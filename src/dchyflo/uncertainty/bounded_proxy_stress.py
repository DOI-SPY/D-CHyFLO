from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd


def axis_contract(config: dict) -> pd.DataFrame:
    rows = []
    for gate, levels in config["axes"].items():
        for item in levels:
            row = {"gate": gate, **item}
            row["evidence_class"] = "BOUNDED_REGIONAL_PROXY_OR_SIMULATION"
            row["site_gate_closed"] = False
            rows.append(row)
    return pd.DataFrame(rows)


def static_stress_surface(source: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    axes = config["axes"]
    combos = list(product(axes["H"], axes["K"], axes["M"], axes["L"], axes["G"]))
    base_cols = [
        "dispatch_case_key", "capacity_mwac", "year", "hydrologic_regime", "s0_reference",
        "signal_name", "proxy_scenario", "incremental_hybrid_export_mwh", "fpv_footprint_km2",
    ]
    for source_row_id, base in source.reset_index(drop=True).iterrows():
        initial_equivalent = float(base["technology_lifecycle_tco2e_yr"]) * 30.0
        for h, k, m, lifecycle_axis, g in combos:
            export = float(base["incremental_hybrid_export_mwh"]) * h["energy_multiplier"] * k["availability_multiplier"]
            avoided = export * m["grid_factor_kgco2e_per_mwh"] / 1000.0
            technology = export * lifecycle_axis["lifecycle_gco2e_per_kwh"] / 1000.0 * k["bom_multiplier"]
            aquatic = float(base["fpv_footprint_km2"]) * g["flux_gco2e_m2_yr"] * g["response_fraction"]
            burden = technology + aquatic
            net = avoided - burden
            break_even = (technology + aquatic) * 1000.0 / export if export > 0 else np.nan
            row = {col: base[col] for col in base_cols}
            row.update({
                "source_row_id": source_row_id,
                "H_level": h["level"], "K_level": k["level"], "M_level": m["level"], "L_level": lifecycle_axis["level"], "G_level": g["level"],
                "energy_multiplier": h["energy_multiplier"], "availability_multiplier": k["availability_multiplier"],
                "bom_multiplier": k["bom_multiplier"], "grid_factor_kgco2e_per_mwh_stress": m["grid_factor_kgco2e_per_mwh"],
                "lifecycle_gco2e_per_kwh_stress": lifecycle_axis["lifecycle_gco2e_per_kwh"], "aquatic_flux_gco2e_m2_yr_stress": g["flux_gco2e_m2_yr"],
                "aquatic_response_fraction_stress": g["response_fraction"], "incremental_export_mwh_stress": export,
                "avoided_grid_tco2e_yr_stress": avoided, "technology_lifecycle_tco2e_yr_stress": technology,
                "aquatic_delta_tco2e_yr_stress": aquatic, "net_avoided_tco2e_yr_stress": net,
                "break_even_grid_factor_kgco2e_per_mwh_stress": break_even,
                "net_sign": "POSITIVE" if net > 0 else ("NEGATIVE" if net < 0 else "ZERO"),
                "initial_manufacture_equivalent_tco2e": initial_equivalent * k["bom_multiplier"] * (lifecycle_axis["lifecycle_gco2e_per_kwh"] / float(base["fpv_full_lifecycle_gco2e_per_kwh"])),
                "claim_status": "CONDITIONAL_BOUNDED_SUBSTITUTION_STRESS_NOT_SITE_VALIDATED",
                "site_gate_status": "HOLD",
            })
            rows.append(row)
    return pd.DataFrame(rows)


def static_envelopes(surface: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    capacity = surface.groupby("capacity_mwac").agg(
        states=("net_avoided_tco2e_yr_stress", "size"),
        minimum_net_tco2e_yr=("net_avoided_tco2e_yr_stress", "min"),
        median_net_tco2e_yr=("net_avoided_tco2e_yr_stress", "median"),
        maximum_net_tco2e_yr=("net_avoided_tco2e_yr_stress", "max"),
        positive_states=("net_sign", lambda x: int((x == "POSITIVE").sum())),
        negative_states=("net_sign", lambda x: int((x == "NEGATIVE").sum())),
        maximum_break_even_factor=("break_even_grid_factor_kgco2e_per_mwh_stress", "max"),
    ).reset_index()
    capacity["all_states_positive"] = capacity["negative_states"].eq(0)
    context_cols = ["H_level", "K_level", "M_level", "L_level", "G_level"]
    context = surface.groupby(context_cols).agg(
        states=("net_avoided_tco2e_yr_stress", "size"),
        minimum_net_tco2e_yr=("net_avoided_tco2e_yr_stress", "min"),
        maximum_net_tco2e_yr=("net_avoided_tco2e_yr_stress", "max"),
        positive_states=("net_sign", lambda x: int((x == "POSITIVE").sum())),
        negative_states=("net_sign", lambda x: int((x == "NEGATIVE").sum())),
    ).reset_index()
    robust_by_capacity = surface.groupby(context_cols + ["capacity_mwac"])["net_avoided_tco2e_yr_stress"].min().ge(0).reset_index(name="robust_nonnegative")
    robust_sets = robust_by_capacity.loc[robust_by_capacity["robust_nonnegative"]].groupby(context_cols)["capacity_mwac"].apply(
        lambda values: "|".join(str(int(v)) for v in sorted(values.unique()))
    ).rename("nonnegative_capacity_set_mwac")
    context = context.merge(robust_sets, on=context_cols, how="left")
    context["nonnegative_capacity_set_mwac"] = context["nonnegative_capacity_set_mwac"].fillna("EMPTY")
    return capacity, context


def dynamic_bundle_paths(annual: pd.DataFrame, bundles: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for bundle in bundles:
        frame = annual.copy()
        energy = bundle["H"] * bundle["K_energy"]
        avoided = frame["avoided_grid_tco2e"] * energy * bundle["M_ratio"]
        initial = frame["initial_manufacture_tco2e"] * bundle["K_bom"]
        replacement = frame["replacement_tco2e"] * bundle["K_bom"]
        end = frame["end_of_life_tco2e"] * bundle["K_bom"]
        if bundle["L_target"] is not None:
            ratios = bundle["L_target"] / frame["fpv_full_lifecycle_gco2e_per_kwh"]
            initial *= ratios
            replacement *= ratios
            end *= ratios
        aquatic = frame["local_aquatic_delta_tco2e"].copy()
        if bundle["G_mode"] == "adverse":
            aquatic = frame["fpv_footprint_km2"] * 388.8531 * 0.25
            aquatic = aquatic.where(frame["lifecycle_year"].gt(0), 0.0)
        annual_net = avoided - initial - replacement - end - aquatic
        frame_out = pd.DataFrame({
            "dynamic_path_key": frame["dynamic_path_key"], "bundle": bundle["bundle"], "lifecycle_year": frame["lifecycle_year"],
            "annual_net_avoided_tco2e": annual_net,
        })
        frame_out["cumulative_net_avoided_tco2e"] = frame_out.groupby(["dynamic_path_key", "bundle"])["annual_net_avoided_tco2e"].cumsum()
        rows.append(frame_out)
    ledger = pd.concat(rows, ignore_index=True)
    summary_rows = []
    for (path, bundle), part in ledger.groupby(["dynamic_path_key", "bundle"], sort=False):
        ordered = part.sort_values("lifecycle_year")
        eligible_years = []
        for year in ordered.loc[ordered["lifecycle_year"].gt(0), "lifecycle_year"]:
            tail = ordered.loc[ordered["lifecycle_year"].ge(year), "cumulative_net_avoided_tco2e"]
            if tail.ge(0).all():
                eligible_years.append(int(year))
                break
        summary_rows.append({
            "dynamic_path_key": path, "bundle": bundle,
            "lifetime_net_avoided_tco2e": float(part.iloc[-1]["cumulative_net_avoided_tco2e"]),
            "carbon_payback_year": eligible_years[0] if eligible_years else np.nan,
            "positive_at_year30": bool(part.iloc[-1]["cumulative_net_avoided_tco2e"] > 0),
            "claim_status": "CONDITIONAL_DYNAMIC_PROXY_STRESS_NOT_FORECAST",
        })
    return ledger, pd.DataFrame(summary_rows)


def driver_counterfactual(dynamic: pd.DataFrame) -> pd.DataFrame:
    pivot = dynamic.pivot(index="dynamic_path_key", columns="bundle", values="lifetime_net_avoided_tco2e")
    reference = pivot["reference"]
    rows = []
    for bundle in [c for c in pivot.columns if c not in {"reference", "combined_adversarial"}]:
        loss = reference - pivot[bundle]
        rows.append(pd.DataFrame({"dynamic_path_key": pivot.index, "bundle": bundle, "loss_tco2e": loss.values}))
    losses = pd.concat(rows, ignore_index=True)
    dominant = losses.loc[losses.groupby("dynamic_path_key")["loss_tco2e"].idxmax()].copy()
    dominant = dominant.rename(columns={"bundle": "dominant_single_axis_driver", "loss_tco2e": "dominant_single_axis_loss_tco2e"})
    return dominant.reset_index(drop=True)
