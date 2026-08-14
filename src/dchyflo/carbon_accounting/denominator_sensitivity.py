from __future__ import annotations

from itertools import combinations
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


MATCH_KEYS = (
    "reservoir_case_id",
    "s0_reference",
    "year",
    "hydrologic_regime",
    "signal_name",
    "foreground_archetype_id",
)

OPERATIONAL_KEYS = (
    "s0_reference",
    "year",
    "hydrologic_regime",
    "signal_name",
)


def metric_contract() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric_id": "ABSOLUTE_PARTIAL_GROSS",
                "value_column": "partial_gross_context_central_tco2e_yr",
                "direction": "MINIMIZE_FOR_COMPARISON",
                "functional_unit": "one representative operating year",
                "evidence_scope": "conditional whole-reservoir GHG plus annual-equivalent partial FPV foreground GWP",
                "selection_basis": "low, central and high partial-foreground bounds",
                "allowed_interpretation": "absolute partial gross-burden ordering within a matched context",
                "forbidden_interpretation": "complete lifecycle burden, net carbon, or engineering optimum",
            },
            {
                "metric_id": "WHOLE_SYSTEM_ATTRIBUTIONAL_INTENSITY",
                "value_column": "partial_gross_context_central_gco2e_kwh_hybrid_export",
                "direction": "MINIMIZE_FOR_COMPARISON",
                "functional_unit": "one kWh optimized hybrid export",
                "evidence_scope": "the same partial gross numerator allocated to total hybrid export",
                "selection_basis": "low, central and high partial-foreground bounds",
                "allowed_interpretation": "whole-system attributional intensity ordering",
                "forbidden_interpretation": "physical reservoir mitigation or carbon-optimal FPV capacity",
            },
            {
                "metric_id": "INCREMENTAL_FOREGROUND_SERVICE_INTENSITY",
                "value_column": "partial_foreground_central_gco2e_kwh_incremental_export",
                "direction": "MINIMIZE_FOR_COMPARISON",
                "functional_unit": "one kWh incremental hybrid export relative to the matched hydro control",
                "evidence_scope": "annual-equivalent partial FPV foreground GWP only",
                "selection_basis": "low, central and high partial-foreground bounds",
                "allowed_interpretation": "partial foreground service-efficiency ordering",
                "forbidden_interpretation": "whole-reservoir allocation, complete FPV LCA, or net benefit",
            },
            {
                "metric_id": "CAPACITY_NORMALIZED_FOREGROUND",
                "value_column": "capacity_normalized_partial_foreground_central_tco2e_yr_per_mwac",
                "direction": "MINIMIZE_FOR_COMPARISON",
                "functional_unit": "one registered MWac in one representative-year equivalent",
                "evidence_scope": "annual-equivalent partial FPV foreground GWP divided by registered capacity",
                "selection_basis": "low, central and high partial-foreground bounds",
                "allowed_interpretation": "audit of whether the linear inventory contains independent capacity-order information",
                "forbidden_interpretation": "energy-service performance, scale economy, or engineering optimum",
            },
            {
                "metric_id": "OPERATIONAL_REDISPATCH_RELATIVE_VALUE",
                "value_column": "redispatch_value_tco2",
                "direction": "MAXIMIZE_FOR_COMPARISON",
                "functional_unit": "one representative year relative to the matched 0 MWac hydro control",
                "evidence_scope": "CP3-C modelled-SRMEF control-adjusted redispatch value",
                "selection_basis": "point ordering plus propagated numerical replay envelope",
                "allowed_interpretation": "relative operational-sensitivity ordering",
                "forbidden_interpretation": "total avoided emissions, lifecycle credit, probability, or carbon optimum",
            },
        ]
    )


def invalid_inference_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("II-01", "select a capacity from one denominator and call it boundary-independent", "BLOCKED"),
            ("II-02", "average ranks or metrics into a composite score", "BLOCKED"),
            ("II-03", "interpret matched-group selection shares as probabilities", "BLOCKED"),
            ("II-04", "treat all-tied capacity-normalized foreground burden as evidence of equal energy performance", "BLOCKED"),
            ("II-05", "treat falling whole-system intensity as physical reservoir-GHG mitigation", "BLOCKED"),
            ("II-06", "treat CP3 relative redispatch value as total avoided grid emissions", "BLOCKED"),
            ("II-07", "combine CP3 value with CP4 plus CP5 to report net carbon", "BLOCKED"),
            ("II-08", "use a unique point winner when its numerical envelope overlaps another capacity", "BLOCKED"),
            ("II-09", "infer statistical confidence from deterministic low/central/high proxy bounds", "BLOCKED"),
            ("II-10", "release Gate C or recommend construction capacity from CP6-B", "BLOCKED"),
        ],
        columns=["inference_id", "invalid_inference", "status"],
    )


def validate_matched_design(
    states: pd.DataFrame,
    capacities: Sequence[float],
    group_keys: Sequence[str] = MATCH_KEYS,
) -> pd.Series:
    required = set(group_keys) | {"capacity_mwac"}
    missing = required.difference(states.columns)
    if missing:
        raise ValueError(f"missing matched-design columns: {sorted(missing)}")
    if states.duplicated(list(group_keys) + ["capacity_mwac"]).any():
        raise ValueError("matched design contains duplicate group-capacity states")
    expected = tuple(float(value) for value in capacities)
    observed = tuple(sorted(float(value) for value in states.capacity_mwac.unique()))
    if observed != expected:
        raise ValueError(f"capacity grid mismatch: observed={observed}, expected={expected}")
    sizes = states.groupby(list(group_keys), dropna=False).size()
    if not (sizes == len(expected)).all():
        raise ValueError("each matched comparison group must contain every registered capacity exactly once")
    return sizes


def derive_metric_states(states: pd.DataFrame) -> pd.DataFrame:
    result = states.copy()
    for bound in ("low", "central", "high"):
        result[f"capacity_normalized_partial_foreground_{bound}_tco2e_yr_per_mwac"] = (
            result[f"partial_foreground_{bound}_annualized_tco2e_yr"] / result["capacity_mwac"]
        )
    result["redispatch_numerical_envelope_tco2"] = (
        result["control_adjusted_numerical_envelope_kgco2"] / 1000.0
    )
    result["redispatch_lower_tco2"] = (
        result["redispatch_value_tco2"] - result["redispatch_numerical_envelope_tco2"]
    )
    result["redispatch_upper_tco2"] = (
        result["redispatch_value_tco2"] + result["redispatch_numerical_envelope_tco2"]
    )
    result["conditional_reservoir_attributional_intensity_gco2e_kwh_hybrid_export"] = (
        result["net_ghg_tco2e_yr"] * 1000.0 / result["optimized_export_mwh"]
    )
    result["partial_foreground_attributional_intensity_gco2e_kwh_hybrid_export"] = (
        result["partial_foreground_central_annualized_tco2e_yr"]
        * 1000.0
        / result["optimized_export_mwh"]
    )
    for row in metric_contract().itertuples(index=False):
        ascending = row.direction.startswith("MINIMIZE")
        rank_column = f"rank_{row.metric_id.lower()}"
        result[rank_column] = result.groupby(
            list(MATCH_KEYS), dropna=False
        )[row.value_column].rank(method="average", ascending=ascending)
        if row.metric_id == "CAPACITY_NORMALIZED_FOREGROUND":
            grouped = result.groupby(list(MATCH_KEYS), dropna=False)[row.value_column]
            within_group_range = grouped.transform("max") - grouped.transform("min")
            group_size = result.groupby(list(MATCH_KEYS), dropna=False)[row.value_column].transform("size")
            result.loc[within_group_range <= 1e-9, rank_column] = (
                group_size.loc[within_group_range <= 1e-9] + 1.0
            ) / 2.0
    result["screening_status"] = "UNWEIGHTED_MATCHED_DETERMINISTIC_COMPARISON_NOT_PROBABILITY"
    result["recommendation_status"] = "BLOCKED"
    return result


def _selection_set(
    capacities: np.ndarray,
    values: np.ndarray,
    direction: str,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[tuple[float, ...], float]:
    best = float(np.nanmin(values) if direction.startswith("MINIMIZE") else np.nanmax(values))
    tolerance = max(absolute_tolerance, relative_tolerance * max(1.0, abs(best)))
    if direction.startswith("MINIMIZE"):
        mask = values <= best + tolerance
    else:
        mask = values >= best - tolerance
    return tuple(sorted(float(value) for value in capacities[mask])), best


def _capacity_string(values: Iterable[float]) -> str:
    return "|".join(f"{float(value):g}" for value in values)


def build_ordering_audit(
    metric_states: pd.DataFrame,
    absolute_tolerance: float = 1e-9,
    relative_tolerance: float = 1e-10,
) -> pd.DataFrame:
    contract = metric_contract().set_index("metric_id")
    bound_columns: Mapping[str, Mapping[str, str]] = {
        "ABSOLUTE_PARTIAL_GROSS": {
            bound: f"partial_gross_context_{bound}_tco2e_yr" for bound in ("low", "central", "high")
        },
        "WHOLE_SYSTEM_ATTRIBUTIONAL_INTENSITY": {
            bound: f"partial_gross_context_{bound}_gco2e_kwh_hybrid_export"
            for bound in ("low", "central", "high")
        },
        "INCREMENTAL_FOREGROUND_SERVICE_INTENSITY": {
            bound: f"partial_foreground_{bound}_gco2e_kwh_incremental_export"
            for bound in ("low", "central", "high")
        },
        "CAPACITY_NORMALIZED_FOREGROUND": {
            bound: f"capacity_normalized_partial_foreground_{bound}_tco2e_yr_per_mwac"
            for bound in ("low", "central", "high")
        },
    }
    rows: list[dict] = []
    for key_values, group in metric_states.groupby(list(MATCH_KEYS), sort=False, dropna=False):
        group = group.sort_values("capacity_mwac")
        capacities = group.capacity_mwac.to_numpy(float)
        identifiers = dict(zip(MATCH_KEYS, key_values))
        for metric_id, columns in bound_columns.items():
            direction = str(contract.loc[metric_id, "direction"])
            selected: dict[str, tuple[float, ...]] = {}
            best_values: dict[str, float] = {}
            for bound, column in columns.items():
                selected[bound], best_values[bound] = _selection_set(
                    capacities,
                    group[column].to_numpy(float),
                    direction,
                    absolute_tolerance,
                    relative_tolerance,
                )
            central_values = group[columns["central"]].to_numpy(float)
            worst_value = float(
                np.nanmax(central_values) if direction.startswith("MINIMIZE") else np.nanmin(central_values)
            )
            rows.append(
                {
                    **identifiers,
                    "metric_id": metric_id,
                    "direction": direction,
                    "low_selected_capacities_mwac": _capacity_string(selected["low"]),
                    "central_selected_capacities_mwac": _capacity_string(selected["central"]),
                    "high_selected_capacities_mwac": _capacity_string(selected["high"]),
                    "robust_selected_capacities_mwac": _capacity_string(selected["central"]),
                    "central_selected_count": len(selected["central"]),
                    "robust_selected_count": len(selected["central"]),
                    "central_best_value": best_values["central"],
                    "central_worst_value": worst_value,
                    "central_range": float(np.nanmax(central_values) - np.nanmin(central_values)),
                    "boundary_selection_stable": selected["low"] == selected["central"] == selected["high"],
                    "numerical_envelope_applied": False,
                    "ordering_status": (
                        "ALL_CAPACITIES_TIED"
                        if len(selected["central"]) == len(capacities)
                        else "UNIQUE_POINT_SELECTION"
                        if len(selected["central"]) == 1
                        else "TIED_POINT_SELECTION"
                    ),
                    "recommendation_status": "BLOCKED",
                }
            )

        metric_id = "OPERATIONAL_REDISPATCH_RELATIVE_VALUE"
        values = group.redispatch_value_tco2.to_numpy(float)
        point_selected, best_value = _selection_set(
            capacities,
            values,
            "MAXIMIZE_FOR_COMPARISON",
            absolute_tolerance,
            relative_tolerance,
        )
        lower = group.redispatch_lower_tco2.to_numpy(float)
        upper = group.redispatch_upper_tco2.to_numpy(float)
        best_lower = float(np.nanmax(lower))
        tolerance = max(absolute_tolerance, relative_tolerance * max(1.0, abs(best_lower)))
        robust_selected = tuple(sorted(float(value) for value in capacities[upper >= best_lower - tolerance]))
        rows.append(
            {
                **identifiers,
                "metric_id": metric_id,
                "direction": "MAXIMIZE_FOR_COMPARISON",
                "low_selected_capacities_mwac": "NOT_APPLICABLE",
                "central_selected_capacities_mwac": _capacity_string(point_selected),
                "high_selected_capacities_mwac": "NOT_APPLICABLE",
                "robust_selected_capacities_mwac": _capacity_string(robust_selected),
                "central_selected_count": len(point_selected),
                "robust_selected_count": len(robust_selected),
                "central_best_value": best_value,
                "central_worst_value": float(np.nanmin(values)),
                "central_range": float(np.nanmax(values) - np.nanmin(values)),
                "boundary_selection_stable": "NOT_APPLICABLE",
                "numerical_envelope_applied": True,
                "ordering_status": (
                    "NUMERICALLY_UNRESOLVED_POINT_WINNER"
                    if len(robust_selected) > len(point_selected)
                    else "NUMERICALLY_RESOLVED_POINT_WINNER"
                ),
                "recommendation_status": "BLOCKED",
            }
        )
    return pd.DataFrame(rows)


def capacity_selection_frequency(
    audit: pd.DataFrame,
    capacities: Sequence[float],
) -> pd.DataFrame:
    rows = []
    for metric_id, group in audit.groupby("metric_id", sort=False):
        total = len(group)
        for capacity in capacities:
            token = f"{float(capacity):g}"
            central = group.central_selected_capacities_mwac.astype(str).str.split("|").apply(
                lambda values: token in values
            )
            robust = group.robust_selected_capacities_mwac.astype(str).str.split("|").apply(
                lambda values: token in values
            )
            rows.append(
                {
                    "metric_id": metric_id,
                    "capacity_mwac": float(capacity),
                    "matched_groups": total,
                    "central_selected_groups": int(central.sum()),
                    "central_selected_share_of_enumerated_groups": float(central.mean()),
                    "robust_selected_groups": int(robust.sum()),
                    "robust_selected_share_of_enumerated_groups": float(robust.mean()),
                    "frequency_status": "DESCRIPTIVE_UNWEIGHTED_SCREENING_NOT_PROBABILITY",
                }
            )
    return pd.DataFrame(rows)


def build_group_conflicts(audit: pd.DataFrame) -> pd.DataFrame:
    selected = audit.pivot(
        index=list(MATCH_KEYS),
        columns="metric_id",
        values="central_selected_capacities_mwac",
    ).reset_index()
    renamed = {
        metric: f"selected_{metric.lower()}" for metric in metric_contract().metric_id
    }
    selected = selected.rename(columns=renamed)
    absolute = selected[renamed["ABSOLUTE_PARTIAL_GROSS"]]
    system = selected[renamed["WHOLE_SYSTEM_ATTRIBUTIONAL_INTENSITY"]]
    incremental = selected[renamed["INCREMENTAL_FOREGROUND_SERVICE_INTENSITY"]]
    operational = selected[renamed["OPERATIONAL_REDISPATCH_RELATIVE_VALUE"]]
    selected["absolute_vs_system_conflict"] = absolute != system
    selected["absolute_vs_incremental_agreement"] = absolute == incremental
    selected["absolute_vs_operational_agreement"] = absolute == operational

    def intersection(row: pd.Series) -> str:
        metric_sets = []
        for metric in (
            "ABSOLUTE_PARTIAL_GROSS",
            "WHOLE_SYSTEM_ATTRIBUTIONAL_INTENSITY",
            "INCREMENTAL_FOREGROUND_SERVICE_INTENSITY",
            "OPERATIONAL_REDISPATCH_RELATIVE_VALUE",
        ):
            values = str(row[renamed[metric]]).split("|")
            metric_sets.append(set(values))
        common = set.intersection(*metric_sets)
        return "|".join(sorted(common, key=float)) if common else "EMPTY"

    selected["cross_metric_unique_consensus_capacities_mwac"] = selected.apply(intersection, axis=1)
    selected["cross_metric_consensus_status"] = np.where(
        selected.cross_metric_unique_consensus_capacities_mwac == "EMPTY",
        "NO_BOUNDARY_INDEPENDENT_CONSENSUS",
        "POINT_SET_INTERSECTION_EXISTS_NOT_RECOMMENDATION",
    )
    return selected


def pairwise_rank_agreement(metric_states: pd.DataFrame) -> pd.DataFrame:
    rank_columns = {
        metric: f"rank_{metric.lower()}" for metric in metric_contract().metric_id
    }
    rows = []
    ordered = metric_states.sort_values(list(MATCH_KEYS) + ["capacity_mwac"])
    group_count = ordered.groupby(list(MATCH_KEYS), sort=False, dropna=False).ngroups
    capacity_count = ordered.capacity_mwac.nunique()
    rank_arrays = {
        metric: ordered[column].to_numpy(float).reshape(group_count, capacity_count)
        for metric, column in rank_columns.items()
    }
    for metric_a, metric_b in combinations(rank_columns, 2):
        a = rank_arrays[metric_a]
        b = rank_arrays[metric_b]
        a_centered = a - a.mean(axis=1, keepdims=True)
        b_centered = b - b.mean(axis=1, keepdims=True)
        denominator = np.sqrt(
            np.sum(a_centered * a_centered, axis=1)
            * np.sum(b_centered * b_centered, axis=1)
        )
        defined = denominator > 1e-12
        correlations = np.sum(a_centered * b_centered, axis=1)[defined] / denominator[defined]
        same = int(np.isclose(correlations, 1.0, atol=1e-12).sum())
        reverse = int(np.isclose(correlations, -1.0, atol=1e-12).sum())
        undefined = int((~defined).sum())
        rows.append(
            {
                "metric_a": metric_a,
                "metric_b": metric_b,
                "matched_groups_total": group_count,
                "matched_groups_defined": int(defined.sum()),
                "undefined_all_tied_groups": undefined,
                "spearman_mean": float(np.mean(correlations)) if correlations.size else np.nan,
                "spearman_min": float(np.min(correlations)) if correlations.size else np.nan,
                "spearman_max": float(np.max(correlations)) if correlations.size else np.nan,
                "identical_order_groups": same,
                "reverse_order_groups": reverse,
                "interpretation_status": "RANK_DIAGNOSTIC_NOT_COMPOSITE_SCORE",
            }
        )
    return pd.DataFrame(rows)


def operational_unique_ordering(audit: pd.DataFrame) -> pd.DataFrame:
    operational = audit.loc[
        audit.metric_id == "OPERATIONAL_REDISPATCH_RELATIVE_VALUE",
        list(OPERATIONAL_KEYS)
        + [
            "central_selected_capacities_mwac",
            "robust_selected_capacities_mwac",
            "central_selected_count",
            "robust_selected_count",
            "central_best_value",
            "central_worst_value",
            "central_range",
            "ordering_status",
        ],
    ].drop_duplicates()
    if operational.duplicated(list(OPERATIONAL_KEYS)).any():
        raise ValueError("operational ordering depends on reservoir or foreground replication")
    operational["replication_status"] = "UNIQUE_CP3_DISPATCH_CONTEXT_NOT_SCENARIO_WEIGHT"
    return operational.sort_values(list(OPERATIONAL_KEYS)).reset_index(drop=True)


def denominator_identity_groups(metric_states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key_values, group in metric_states.groupby(list(MATCH_KEYS), sort=False, dropna=False):
        group = group.sort_values("capacity_mwac")
        first = group.iloc[0]
        last = group.iloc[-1]
        gross_delta = last.partial_gross_context_central_tco2e_yr - first.partial_gross_context_central_tco2e_yr
        foreground_delta = (
            last.partial_foreground_central_annualized_tco2e_yr
            - first.partial_foreground_central_annualized_tco2e_yr
        )
        attribution_identity = (
            group.partial_gross_context_central_gco2e_kwh_hybrid_export
            - group.conditional_reservoir_attributional_intensity_gco2e_kwh_hybrid_export
            - group.partial_foreground_attributional_intensity_gco2e_kwh_hybrid_export
        ).abs().max()
        rows.append(
            {
                **dict(zip(MATCH_KEYS, key_values)),
                "reservoir_term_capacity_range_tco2e_yr": float(group.net_ghg_tco2e_yr.max() - group.net_ghg_tco2e_yr.min()),
                "gross_delta_150_to_300_tco2e_yr": float(gross_delta),
                "foreground_delta_150_to_300_tco2e_yr": float(foreground_delta),
                "absolute_additivity_delta_error_tco2e_yr": float(abs(gross_delta - foreground_delta)),
                "capacity_normalized_foreground_range_tco2e_yr_per_mwac": float(
                    group.capacity_normalized_partial_foreground_central_tco2e_yr_per_mwac.max()
                    - group.capacity_normalized_partial_foreground_central_tco2e_yr_per_mwac.min()
                ),
                "attributional_intensity_identity_error_gco2e_kwh": float(attribution_identity),
                "gross_intensity_change_150_to_300_percent": float(
                    100.0
                    * (last.partial_gross_context_central_gco2e_kwh_hybrid_export / first.partial_gross_context_central_gco2e_kwh_hybrid_export - 1.0)
                ),
                "incremental_foreground_intensity_change_150_to_300_percent": float(
                    100.0
                    * (last.partial_foreground_central_gco2e_kwh_incremental_export / first.partial_foreground_central_gco2e_kwh_incremental_export - 1.0)
                ),
            }
        )
    return pd.DataFrame(rows)


def denominator_identity_summary(groups: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column, expected, threshold, note in (
        ("reservoir_term_capacity_range_tco2e_yr", 0.0, 1e-9, "whole-reservoir term remains fixed within each matched capacity group"),
        ("absolute_additivity_delta_error_tco2e_yr", 0.0, 1e-9, "gross capacity change equals partial-foreground capacity change"),
        ("capacity_normalized_foreground_range_tco2e_yr_per_mwac", 0.0, 1e-9, "linear foreground inventory contains no independent capacity ordering"),
        ("attributional_intensity_identity_error_gco2e_kwh", 0.0, 1e-9, "gross intensity equals reservoir plus foreground attributional intensities"),
    ):
        maximum = float(groups[column].max())
        rows.append(
            {
                "identity": column,
                "maximum_absolute_error": maximum,
                "expected": expected,
                "threshold": threshold,
                "passed": maximum <= threshold,
                "note": note,
            }
        )
    return pd.DataFrame(rows)
