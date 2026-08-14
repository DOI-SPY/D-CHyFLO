from __future__ import annotations

from typing import Sequence

import pandas as pd


CONTROL_GROUP = (
    "reservoir_case_id",
    "s0_reference",
    "year",
    "hydrologic_regime",
    "signal_name",
)


def component_ledger(service_life_years: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "component_id": "CP3_CONTROL_ADJUSTED_REDISPATCH",
                "source_checkpoint": "CP3-C",
                "quantity_type": "ANNUAL_RELATIVE_OPERATIONAL_VALUE",
                "native_unit": "kg CO2 per representative operating year",
                "system_scope": "incremental carbon-aware redispatch value relative to matched 0 MWac hydro control",
                "temporal_alignment": "already annual at a historical representative endpoint",
                "aligned_unit": "t CO2 per representative operating year",
                "alignment_transform": "divide kg by 1000",
                "can_enter_partial_gross_burden": False,
                "can_enter_net_carbon": False,
                "claim_status": "MODELLED_SRMEF_RELATIVE_SENSITIVITY_NOT_TOTAL_AVOIDED_EMISSIONS",
            },
            {
                "component_id": "CP4_PARTIAL_FOREGROUND_GWP",
                "source_checkpoint": "CP4-B",
                "quantity_type": "PARTIAL_LIFETIME_FOREGROUND_BURDEN",
                "native_unit": "t CO2e per FPV project over declared service life",
                "system_scope": "traceably characterized subset of FPV foreground production, replacement and C1-C4; module D excluded",
                "temporal_alignment": f"straight-line annualization over {service_life_years:g} years; not dynamic LCA",
                "aligned_unit": "t CO2e per arithmetic annual-equivalent year",
                "alignment_transform": f"divide lifetime total by {service_life_years:g}",
                "can_enter_partial_gross_burden": True,
                "can_enter_net_carbon": False,
                "claim_status": "PARTIAL_PUBLIC_PROXY_CHARACTERIZATION_WITH_MISSING_PROCESSES",
            },
            {
                "component_id": "CP5_CONDITIONAL_RESERVOIR_GHG",
                "source_checkpoint": "CP5-I/J",
                "quantity_type": "ANNUAL_WHOLE_RESERVOIR_SITE_FLUX",
                "native_unit": "t CO2e per year",
                "system_scope": "whole-reservoir conditional CO2+CH4 emissions; not allocated to FPV capacity",
                "temporal_alignment": "annual model-conditional flux",
                "aligned_unit": "t CO2e per representative operating year",
                "alignment_transform": "none",
                "can_enter_partial_gross_burden": True,
                "can_enter_net_carbon": False,
                "claim_status": "CONDITIONAL_SCENARIO_NOT_VERIFIED_SITE_POINT_BASELINE",
            },
        ]
    )


def functional_unit_registry(service_life_years: float, dc_ac_ratio: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "functional_unit_id": "FU_CAPACITY_DC",
                "functional_unit": "1 kWp_dc installed FPV capacity at commissioning",
                "valid_components": "CP4_PARTIAL_FOREGROUND_GWP",
                "valid_use": "foreground inventory and partial characterized intensity",
                "invalid_use": "reservoir or dispatch allocation",
            },
            {
                "functional_unit_id": "FU_DESIGN_AC",
                "functional_unit": "1 kWac registered FPV design capacity",
                "valid_components": "CP3_CONTROL_ADJUSTED_REDISPATCH | CP4_PARTIAL_FOREGROUND_GWP",
                "valid_use": f"capacity matching with fixed DC/AC ratio {dc_ac_ratio:g}",
                "invalid_use": "energy-service intensity without generation denominator",
            },
            {
                "functional_unit_id": "FU_PROJECT_LIFETIME",
                "functional_unit": f"one FPV project over {service_life_years:g}-year declared service life",
                "valid_components": "CP4_PARTIAL_FOREGROUND_GWP",
                "valid_use": "partial cumulative foreground burden",
                "invalid_use": "direct addition to one-year reservoir or dispatch quantities",
            },
            {
                "functional_unit_id": "FU_REPRESENTATIVE_YEAR",
                "functional_unit": "one representative operating year at a registered historical hydrologic endpoint",
                "valid_components": "CP3_CONTROL_ADJUSTED_REDISPATCH | CP4 annual-equivalent | CP5_CONDITIONAL_RESERVOIR_GHG",
                "valid_use": "aligned ledger and partial gross burden context",
                "invalid_use": "claim of a continuous 30-year hydrologic or grid trajectory",
            },
            {
                "functional_unit_id": "FU_HYBRID_EXPORT",
                "functional_unit": "1 kWh annual optimized hybrid export",
                "valid_components": "CP5 conditional whole-system allocation | CP4 partial annual-equivalent allocation",
                "valid_use": "explicit attributional system-intensity diagnostic",
                "invalid_use": "causal FPV reservoir mitigation or capacity optimum",
            },
            {
                "functional_unit_id": "FU_INCREMENTAL_EXPORT",
                "functional_unit": "1 kWh incremental hybrid export relative to matched 0 MWac hydro control",
                "valid_components": "CP4 partial annual-equivalent | CP3 relative redispatch value",
                "valid_use": "partial FPV service intensity and normalized operational sensitivity",
                "invalid_use": "allocation of the pre-existing whole-reservoir footprint",
            },
        ]
    )


def compatibility_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "combination_id": "CP4_PLUS_CP5",
                "components": "CP4_PARTIAL_FOREGROUND_GWP + CP5_CONDITIONAL_RESERVOIR_GHG",
                "operation": "addition after arithmetic annualization",
                "status": "CONDITIONALLY_COMPATIBLE_PARTIAL_GROSS_CONTEXT",
                "allowed_claim": "partial annual-equivalent gross burden context",
                "forbidden_claim": "complete lifecycle footprint, net carbon, or verified site result",
            },
            {
                "combination_id": "CP3_PLUS_CP4",
                "components": "CP3_CONTROL_ADJUSTED_REDISPATCH + CP4_PARTIAL_FOREGROUND_GWP",
                "operation": "parallel reporting or normalized scale comparison only",
                "status": "NOT_ADDITIVE_FOR_NET_CARBON",
                "allowed_claim": "relative operational sensitivity alongside partial foreground burden",
                "forbidden_claim": "carbon payback, net avoided emissions, or net lifecycle carbon",
            },
            {
                "combination_id": "CP3_PLUS_CP5",
                "components": "CP3_CONTROL_ADJUSTED_REDISPATCH + CP5_CONDITIONAL_RESERVOIR_GHG",
                "operation": "parallel reporting or normalized scale comparison only",
                "status": "NOT_ADDITIVE_FOR_NET_CARBON",
                "allowed_claim": "relative redispatch scale compared with conditional reservoir flux",
                "forbidden_claim": "reservoir offset, net emissions, or physical FPV reservoir mitigation",
            },
            {
                "combination_id": "CP3_PLUS_CP4_PLUS_CP5",
                "components": "all three components",
                "operation": "common ledger without net summation",
                "status": "LEDGER_COMPATIBLE_NET_CARBON_BLOCKED",
                "allowed_claim": "boundary-complete partial evidence ledger",
                "forbidden_claim": "complete lifecycle net carbon or carbon-optimal capacity",
            },
        ]
    )


def invalid_operation_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("IO-01", "subtract CP3 redispatch value from CP4+CP5 and call the result net carbon", "BLOCKED", "CP3 is not total avoided grid emissions"),
            ("IO-02", "allocate the whole-reservoir footprint to incremental FPV export", "BLOCKED", "the reservoir predates FPV and no service-allocation rule is established"),
            ("IO-03", "call CP4-B public-proxy characterization a complete lifecycle result", "BLOCKED", "coverage and background processes remain incomplete"),
            ("IO-04", "treat modelled SRMEF archetypes as observed Northeast marginal emissions", "BLOCKED", "Gate M remains a data gap"),
            ("IO-05", "select an optimal FPV capacity by minimum attributional intensity", "BLOCKED", "denominator allocation, missing costs, cold-region design and incomplete LCIA confound the ranking"),
            ("IO-06", "interpret Cartesian scenario frequencies or quantiles as probabilities", "BLOCKED", "the design is unweighted deterministic screening"),
            ("IO-07", "interpret straight-line annualization as dynamic lifecycle timing", "BLOCKED", "no construction, replacement or end-of-life timing model is applied"),
        ],
        columns=["operation_id", "invalid_operation", "status", "reason"],
    )


def attach_incremental_export(
    coupled: pd.DataFrame,
    group_columns: Sequence[str] = CONTROL_GROUP,
) -> pd.DataFrame:
    required = set(group_columns) | {
        "capacity_mwac",
        "optimized_export_mwh",
        "fpv_control_adjusted_gain_kgco2",
    }
    missing = required.difference(coupled.columns)
    if missing:
        raise ValueError(f"missing dispatch columns: {sorted(missing)}")
    controls = coupled.loc[coupled.capacity_mwac == 0.0, list(group_columns) + ["optimized_export_mwh"]].copy()
    if controls.duplicated(list(group_columns)).any():
        raise ValueError("0 MWac control is not unique within the registered comparison group")
    controls = controls.rename(columns={"optimized_export_mwh": "hydro_control_optimized_export_mwh"})
    positive = coupled.loc[coupled.capacity_mwac > 0.0].copy()
    positive = positive.merge(controls, on=list(group_columns), how="left", validate="many_to_one")
    positive["incremental_hybrid_export_mwh"] = (
        positive["optimized_export_mwh"] - positive["hydro_control_optimized_export_mwh"]
    )
    positive["redispatch_value_tco2"] = positive["fpv_control_adjusted_gain_kgco2"] / 1000.0
    positive["redispatch_value_gco2_per_kwh_incremental_export"] = (
        positive["fpv_control_adjusted_gain_kgco2"] / positive["incremental_hybrid_export_mwh"]
    )
    return positive


def build_unified_states(
    coupled: pd.DataFrame,
    foreground: pd.DataFrame,
    service_life_years: float,
) -> pd.DataFrame:
    positive = attach_incremental_export(coupled)
    positive_fields = [
        "depth_anchor_id",
        "river_length_anchor_id",
        "treatment_factor",
        "landuse_intensity",
        "reservoir_case_id",
        "max_depth_m",
        "inundated_river_length_km",
        "net_ghg_gco2e_m2_yr",
        "net_ghg_tco2e_yr",
        "reservoir_claim_status",
        "dispatch_case_key",
        "s0_reference",
        "year",
        "hydrologic_regime",
        "capacity_mwac",
        "baseline_source",
        "signal_name",
        "factor_type",
        "signal_role",
        "mean_factor_kgco2_per_mwh",
        "optimized_export_mwh",
        "hydro_control_optimized_export_mwh",
        "incremental_hybrid_export_mwh",
        "fpv_control_adjusted_gain_kgco2",
        "control_adjusted_numerical_envelope_kgco2",
        "control_adjusted_sign",
        "redispatch_value_tco2",
        "redispatch_value_gco2_per_kwh_incremental_export",
    ]
    positive = positive[positive_fields]
    foreground_required = {
        "scenario_id",
        "capacity_mwac",
        "dc_capacity_mwp",
        "module_archetype",
        "support_archetype",
        "electrical_archetype",
        "production_mass_coverage",
        "eol_mass_coverage",
        "module_production_mass_coverage",
        "support_production_mass_coverage",
        "electrical_production_mass_coverage",
        "partial_core_low_tco2e",
        "partial_core_central_tco2e",
        "partial_core_high_tco2e",
        "module_d_central_tco2e",
    }
    missing = foreground_required.difference(foreground.columns)
    if missing:
        raise ValueError(f"missing foreground columns: {sorted(missing)}")
    selected = foreground[list(foreground_required)].rename(columns={"scenario_id": "foreground_scenario_id"}).copy()
    selected["foreground_archetype_id"] = (
        selected["module_archetype"] + "__" + selected["support_archetype"] + "__" + selected["electrical_archetype"]
    )
    states = positive.merge(selected, on="capacity_mwac", how="inner", validate="many_to_many")

    for bound in ("low", "central", "high"):
        lifetime = f"partial_core_{bound}_tco2e"
        annualized = f"partial_foreground_{bound}_annualized_tco2e_yr"
        gross = f"partial_gross_context_{bound}_tco2e_yr"
        intensity = f"partial_gross_context_{bound}_gco2e_kwh_hybrid_export"
        fpv_intensity = f"partial_foreground_{bound}_gco2e_kwh_incremental_export"
        states[annualized] = states[lifetime] / float(service_life_years)
        states[gross] = states["net_ghg_tco2e_yr"] + states[annualized]
        states[intensity] = states[gross] * 1000.0 / states["optimized_export_mwh"]
        states[fpv_intensity] = states[annualized] * 1000.0 / states["incremental_hybrid_export_mwh"]

    states["partial_foreground_share_of_central_gross_percent"] = (
        100.0
        * states["partial_foreground_central_annualized_tco2e_yr"]
        / states["partial_gross_context_central_tco2e_yr"]
    )
    states["module_d_status"] = "SEPARATE_NOT_INCLUDED_IN_PARTIAL_CORE"
    states["temporal_alignment_status"] = "REPRESENTATIVE_YEAR_PLUS_STRAIGHT_LINE_ANNUALIZATION_NOT_DYNAMIC_LCA"
    states["joint_scenario_status"] = "UNWEIGHTED_CARTESIAN_SCREENING_NO_PROBABILITIES"
    states["partial_gross_context_status"] = "CONDITIONAL_PARTIAL_GROSS_BURDEN_NOT_COMPLETE_LIFECYCLE"
    states["net_carbon_status"] = "BLOCKED"
    return states


def capacity_envelope(states: pd.DataFrame) -> pd.DataFrame:
    def q05(values: pd.Series) -> float:
        return float(values.quantile(0.05))

    def q95(values: pd.Series) -> float:
        return float(values.quantile(0.95))

    return (
        states.groupby("capacity_mwac", as_index=False)
        .agg(
            states=("foreground_scenario_id", "size"),
            foreground_scenarios=("foreground_scenario_id", "nunique"),
            reservoir_scenarios=("reservoir_case_id", "nunique"),
            dispatch_cases=("dispatch_case_key", "nunique"),
            incremental_export_min_mwh=("incremental_hybrid_export_mwh", "min"),
            incremental_export_median_mwh=("incremental_hybrid_export_mwh", "median"),
            incremental_export_max_mwh=("incremental_hybrid_export_mwh", "max"),
            annualized_foreground_min_tco2e_yr=("partial_foreground_low_annualized_tco2e_yr", "min"),
            annualized_foreground_central_median_tco2e_yr=("partial_foreground_central_annualized_tco2e_yr", "median"),
            annualized_foreground_max_tco2e_yr=("partial_foreground_high_annualized_tco2e_yr", "max"),
            partial_gross_min_tco2e_yr=("partial_gross_context_low_tco2e_yr", "min"),
            partial_gross_median_tco2e_yr=("partial_gross_context_central_tco2e_yr", "median"),
            partial_gross_max_tco2e_yr=("partial_gross_context_high_tco2e_yr", "max"),
            partial_gross_intensity_min_gco2e_kwh=("partial_gross_context_low_gco2e_kwh_hybrid_export", "min"),
            partial_gross_intensity_p05_gco2e_kwh=("partial_gross_context_central_gco2e_kwh_hybrid_export", q05),
            partial_gross_intensity_median_gco2e_kwh=("partial_gross_context_central_gco2e_kwh_hybrid_export", "median"),
            partial_gross_intensity_p95_gco2e_kwh=("partial_gross_context_central_gco2e_kwh_hybrid_export", q95),
            partial_gross_intensity_max_gco2e_kwh=("partial_gross_context_high_gco2e_kwh_hybrid_export", "max"),
            foreground_share_min_percent=("partial_foreground_share_of_central_gross_percent", "min"),
            foreground_share_median_percent=("partial_foreground_share_of_central_gross_percent", "median"),
            foreground_share_max_percent=("partial_foreground_share_of_central_gross_percent", "max"),
            redispatch_value_min_tco2=("redispatch_value_tco2", "min"),
            redispatch_value_median_tco2=("redispatch_value_tco2", "median"),
            redispatch_value_max_tco2=("redispatch_value_tco2", "max"),
        )
    )


def gate_c_readiness() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("numerically_verified_dchyflo_dispatch", True, "CP1-CP3 numerical gates released"),
            ("modelled_grid_signal_sensitivity", True, "CP3-C complete; sensitivity status retained"),
            ("observed_validated_regional_marginal_emissions", False, "Gate M data gap"),
            ("foreground_fpv_inventory", True, "CP4-A complete"),
            ("complete_background_lcia", False, "CP4-C software/license/activity bridge blocked"),
            ("cold_region_engineering_bom", False, "Gate K data gap"),
            ("conditional_reservoir_ghg_ensemble", True, "CP5-I/J complete"),
            ("verified_reservoir_point_baseline", False, "maximum depth and two categorical site inputs unresolved"),
            ("unified_boundary_and_functional_units", True, "CP6-A contract complete when checkpoint passes"),
            ("physical_fpv_reservoir_ghg_response", False, "no transferable response model established"),
            ("complete_transport_replacement_eol_routes", False, "CP4-C blocker"),
        ],
        columns=["prerequisite", "passed", "evidence"],
    )
