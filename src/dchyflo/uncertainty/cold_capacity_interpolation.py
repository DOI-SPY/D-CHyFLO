from __future__ import annotations

import numpy as np
import pandas as pd


KEYS = ["s0_reference", "year"]


def interpolate_dispatch(
    stress: pd.DataFrame,
    target_capacities: list[float],
) -> pd.DataFrame:
    controls = stress.loc[stress.capacity_mwac.eq(0.0), KEYS + ["optimized_export_mwh"]].rename(
        columns={"optimized_export_mwh": "hydro_control_optimized_export_mwh"}
    )
    positive = stress.loc[stress.capacity_mwac.gt(0.0)].copy()
    value_columns = [
        "optimized_export_mwh",
        "energy_baseline_export_mwh",
        "baseline_pv_gross_mwh",
        "state_slp_registry_pv_curtailment_mwh",
        "fpv_control_adjusted_gain_kgco2",
        "control_adjusted_numerical_envelope_kgco2",
    ]
    meta_columns = ["s0_reference", "year", "anchor_roles", "cold_scenario", "signal_name", "factor_type", "signal_role"]
    rows = []
    context_keys = KEYS + ["cold_scenario"]
    for key, group in positive.groupby(context_keys, sort=True):
        group = group.sort_values("capacity_mwac")
        source_capacities = group.capacity_mwac.to_numpy(dtype=float)
        if min(target_capacities) < source_capacities.min() or max(target_capacities) > source_capacities.max():
            raise ValueError("target capacity outside sentinel interpolation range")
        for capacity in target_capacities:
            row = {column: group.iloc[0][column] for column in meta_columns}
            row["capacity_mwac"] = float(capacity)
            for column in value_columns:
                row[column] = float(np.interp(capacity, source_capacities, group[column].to_numpy(dtype=float)))
            row["capacity_source_status"] = (
                "CP7B_EXACT_SENTINEL" if capacity in set(source_capacities) else "PIECEWISE_LINEAR_BETWEEN_CP7B_SENTINELS"
            )
            row["interpolation_bounds_mwac"] = "150|225|300"
            rows.append(row)
    result = pd.DataFrame(rows).merge(controls, on=KEYS, how="left", validate="many_to_one")
    result["incremental_hybrid_export_mwh"] = (
        result.optimized_export_mwh - result.hydro_control_optimized_export_mwh
    )
    return result.sort_values(KEYS + ["cold_scenario", "capacity_mwac"]).reset_index(drop=True)


def sentinel_reproduction(interpolated: pd.DataFrame, stress: pd.DataFrame) -> pd.DataFrame:
    context_keys = KEYS + ["cold_scenario"]
    sentinels = stress.loc[stress.capacity_mwac.isin([150.0, 225.0, 300.0]), context_keys + ["capacity_mwac", "optimized_export_mwh"]]
    matched = interpolated.merge(
        sentinels,
        on=context_keys + ["capacity_mwac"],
        how="inner",
        suffixes=("_interpolated", "_source"),
        validate="one_to_one",
    )
    matched["absolute_error_mwh"] = (
        matched.optimized_export_mwh_interpolated - matched.optimized_export_mwh_source
    ).abs()
    return matched[context_keys + ["capacity_mwac", "optimized_export_mwh_source", "optimized_export_mwh_interpolated", "absolute_error_mwh"]]
