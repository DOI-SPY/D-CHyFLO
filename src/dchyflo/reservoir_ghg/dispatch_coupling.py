from __future__ import annotations

from itertools import combinations
from math import prod
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


FACTOR_COLUMNS = (
    "depth_anchor_id",
    "river_length_anchor_id",
    "treatment_factor",
    "landuse_intensity",
)

DISPATCH_GROUP_COLUMNS = (
    "reservoir_case_id",
    "s0_reference",
    "year",
    "hydrologic_regime",
    "signal_name",
)


def classify_with_envelope(value: pd.Series, envelope: pd.Series) -> pd.Series:
    """Classify a signed value using its row-specific numerical envelope."""
    result = pd.Series("ZERO_WITHIN_TOLERANCE", index=value.index, dtype=object)
    result.loc[value > envelope] = "POSITIVE"
    result.loc[value < -envelope] = "NEGATIVE"
    return result


def factorial_decomposition(
    frame: pd.DataFrame,
    metric: str = "net_ghg_tco2e_yr",
    factors: Sequence[str] = FACTOR_COLUMNS,
) -> pd.DataFrame:
    """Return an exact orthogonal decomposition for a balanced full factorial.

    One deterministic model result exists per cell, so this is descriptive
    attribution rather than inferential ANOVA: no sampling error, p-values, or
    confidence intervals are implied.
    """
    required = set(factors) | {metric}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing factorial columns: {sorted(missing)}")
    if frame.duplicated(list(factors)).any():
        raise ValueError("factorial design contains duplicate cells")
    expected = prod(frame[factor].nunique() for factor in factors)
    if len(frame) != expected:
        raise ValueError(f"factorial design is incomplete: {len(frame)} rows, expected {expected}")

    values = frame[metric].astype(float)
    grand_mean = float(values.mean())
    total_ss = float(np.square(values - grand_mean).sum())
    components: dict[tuple[str, ...], pd.Series] = {}
    rows: list[dict[str, float | int | str]] = []

    for order in range(1, len(factors) + 1):
        for subset in combinations(factors, order):
            component = frame.groupby(list(subset), dropna=False)[metric].transform("mean").astype(float) - grand_mean
            for lower_order in range(1, order):
                for lower_subset in combinations(subset, lower_order):
                    component = component - components[lower_subset]
            components[subset] = component
            ss = float(np.square(component).sum())
            rows.append(
                {
                    "effect": " × ".join(subset),
                    "order": order,
                    "degrees_of_freedom": prod(frame[factor].nunique() - 1 for factor in subset),
                    "sum_squares_tco2e2_yr2": ss,
                    "variance_share_percent": 100.0 * ss / total_ss if total_ss else 0.0,
                    "rms_effect_tco2e_yr": float(np.sqrt(np.mean(np.square(component)))),
                    "max_abs_effect_tco2e_yr": float(component.abs().max()),
                }
            )

    result = pd.DataFrame(rows).sort_values(
        ["variance_share_percent", "order", "effect"], ascending=[False, True, True]
    )
    result["decomposition_type"] = "EXACT_BALANCED_FACTORIAL_DESCRIPTIVE_NO_P_VALUES"
    return result.reset_index(drop=True)


def couple_reservoir_ghg_and_dispatch(
    reservoir: pd.DataFrame,
    dispatch: pd.DataFrame,
) -> pd.DataFrame:
    """Cross conditional whole-reservoir GHG cases with fixed dispatch cases."""
    reservoir_required = set(FACTOR_COLUMNS) | {
        "case_id",
        "net_ghg_gco2e_m2_yr",
        "net_ghg_tco2e_yr",
        "claim_status",
    }
    dispatch_required = {
        "case_key",
        "s0_reference",
        "year",
        "hydrologic_regime",
        "capacity_mwac",
        "signal_name",
        "optimized_export_mwh",
        "fpv_control_adjusted_gain_kgco2",
        "control_adjusted_numerical_envelope_kgco2",
        "control_adjusted_sign",
    }
    missing_reservoir = reservoir_required.difference(reservoir.columns)
    missing_dispatch = dispatch_required.difference(dispatch.columns)
    if missing_reservoir or missing_dispatch:
        raise ValueError(
            f"missing reservoir columns={sorted(missing_reservoir)}; "
            f"missing dispatch columns={sorted(missing_dispatch)}"
        )

    reservoir_fields = list(FACTOR_COLUMNS) + [
        "case_id",
        "max_depth_m",
        "inundated_river_length_km",
        "net_ghg_gco2e_m2_yr",
        "net_ghg_tco2e_yr",
        "claim_status",
    ]
    left = reservoir[reservoir_fields].rename(
        columns={
            "case_id": "reservoir_case_id",
            "claim_status": "reservoir_claim_status",
        }
    ).copy()
    right = dispatch.copy().rename(columns={"case_key": "dispatch_case_key"})
    left["_join"] = 1
    right["_join"] = 1
    coupled = left.merge(right, on="_join", how="inner", validate="many_to_many").drop(columns="_join")

    coupled["reservoir_ghg_kgco2e_yr"] = coupled["net_ghg_tco2e_yr"].astype(float) * 1000.0
    coupled["contextual_scale_comparator_kgco2e"] = (
        coupled["reservoir_ghg_kgco2e_yr"] + coupled["fpv_control_adjusted_gain_kgco2"]
    )
    coupled["fixed_shift_increment_kgco2"] = (
        coupled["contextual_scale_comparator_kgco2e"] - coupled["reservoir_ghg_kgco2e_yr"]
    )
    coupled["relative_dispatch_scale_percent"] = (
        100.0 * coupled["fpv_control_adjusted_gain_kgco2"] / coupled["reservoir_ghg_kgco2e_yr"]
    )
    coupled["reservoir_attributional_intensity_gco2e_kwh"] = (
        coupled["reservoir_ghg_kgco2e_yr"] / coupled["optimized_export_mwh"]
    )
    coupled["fixed_shift_sign"] = classify_with_envelope(
        coupled["fixed_shift_increment_kgco2"],
        coupled["control_adjusted_numerical_envelope_kgco2"],
    )
    coupled["sign_invariant"] = coupled["fixed_shift_sign"] == coupled["control_adjusted_sign"]
    coupled["capacity_positive"] = coupled["capacity_mwac"].astype(float) > 0.0
    coupled["reservoir_ghg_status"] = "CONDITIONAL_WHOLE_RESERVOIR_NOT_CAPACITY_ALLOCATED"
    coupled["contextual_comparator_status"] = "SCALE_COMPARATOR_NOT_NET_CARBON_BALANCE"
    coupled["intensity_status"] = "ATTRIBUTIONAL_DENOMINATOR_DIAGNOSTIC_NOT_PHYSICAL_REDUCTION"
    return coupled


def ranking_invariance_audit(
    coupled: pd.DataFrame,
    group_columns: Iterable[str] = DISPATCH_GROUP_COLUMNS,
) -> pd.DataFrame:
    """Audit that adding a group-constant reservoir term preserves ranking."""
    rows = []
    group_columns = list(group_columns)
    for keys, group in coupled.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        ordered = group.sort_values("capacity_mwac").copy()
        gain_rank = ordered["fpv_control_adjusted_gain_kgco2"].rank(method="min", ascending=False)
        comparator_rank = ordered["contextual_scale_comparator_kgco2e"].rank(method="min", ascending=False)
        gain_delta = ordered["fpv_control_adjusted_gain_kgco2"].diff()
        comparator_delta = ordered["contextual_scale_comparator_kgco2e"].diff()
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "capacities": len(ordered),
                "ranking_identical": bool(gain_rank.equals(comparator_rank)),
                "max_adjacent_delta_identity_error_kgco2": float((gain_delta - comparator_delta).abs().max()),
                "reservoir_term_range_within_group_kgco2e_yr": float(
                    ordered["reservoir_ghg_kgco2e_yr"].max() - ordered["reservoir_ghg_kgco2e_yr"].min()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def intensity_dilution_audit(
    coupled: pd.DataFrame,
    group_columns: Iterable[str] = DISPATCH_GROUP_COLUMNS,
) -> pd.DataFrame:
    """Compare 0 and 300 MWac attributional intensities within each group."""
    rows = []
    group_columns = list(group_columns)
    for keys, group in coupled.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        indexed = group.set_index("capacity_mwac")
        if 0.0 not in indexed.index or 300.0 not in indexed.index:
            raise ValueError("each dilution group must contain 0 and 300 MWac")
        i0 = float(indexed.loc[0.0, "reservoir_attributional_intensity_gco2e_kwh"])
        i300 = float(indexed.loc[300.0, "reservoir_attributional_intensity_gco2e_kwh"])
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "intensity_0_mwac_gco2e_kwh": i0,
                "intensity_300_mwac_gco2e_kwh": i300,
                "intensity_dilution_percent": 100.0 * (i0 - i300) / i0,
                "interpretation": "ATTRIBUTIONAL_DENOMINATOR_DILUTION_NOT_PHYSICAL_RESERVOIR_GHG_REDUCTION",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
