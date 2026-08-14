"""Mechanism metrics for D-CHyFLO water-battery dispatch."""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd


def _fifo_shift_pairing(
    timestamps: pd.Series, delta_energy_mwh: np.ndarray, threshold_hours: float = 24.0
) -> dict[str, float]:
    """Pair earlier hydro reductions with later increases using FIFO accounting."""

    stored: deque[list[object]] = deque()
    matched = 0.0
    matched_over_threshold = 0.0
    lag_energy_hours = 0.0
    unpaired_discharge = 0.0
    for timestamp, delta in zip(pd.to_datetime(timestamps), delta_energy_mwh):
        if delta < 0.0:
            stored.append([timestamp, float(-delta)])
            continue
        remaining = float(delta)
        while remaining > 1.0e-12 and stored:
            charge_time, available = stored[0]
            amount = min(remaining, float(available))
            lag_hours = (timestamp - charge_time).total_seconds() / 3600.0
            matched += amount
            lag_energy_hours += amount * lag_hours
            if lag_hours > threshold_hours:
                matched_over_threshold += amount
            remaining -= amount
            available = float(available) - amount
            if available <= 1.0e-12:
                stored.popleft()
            else:
                stored[0][1] = available
        unpaired_discharge += remaining
    return {
        "matched_shift_energy_mwh": matched,
        "cross_day_shift_energy_mwh": matched_over_threshold,
        "cross_day_share_of_matched": (
            matched_over_threshold / matched if matched > 0.0 else float("nan")
        ),
        "mean_matched_lag_hours": lag_energy_hours / matched if matched > 0.0 else float("nan"),
        "unpaired_discharge_energy_mwh": unpaired_discharge,
        "unreleased_charge_energy_mwh": float(sum(float(item[1]) for item in stored)),
    }


def water_battery_metrics(
    optimized_hourly: pd.DataFrame,
    reference_hourly: pd.DataFrame,
    interval_hours: float = 1.0,
) -> dict[str, float]:
    """Calculate preregistered flexibility metrics relative to a reference path."""

    required = {"timestamp_lst", "hydro_power_mw", "turbine_flow_m3s"}
    for label, frame in (("optimized", optimized_hourly), ("reference", reference_hourly)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{label} hourly data missing columns: {missing}")
    optimized = optimized_hourly[list(required)].copy()
    reference = reference_hourly[list(required)].copy()
    optimized["timestamp_lst"] = pd.to_datetime(optimized["timestamp_lst"])
    reference["timestamp_lst"] = pd.to_datetime(reference["timestamp_lst"])
    merged = optimized.merge(
        reference,
        on="timestamp_lst",
        how="inner",
        validate="one_to_one",
        suffixes=("_optimized", "_reference"),
    ).sort_values("timestamp_lst")
    if len(merged) != len(optimized) or len(merged) != len(reference):
        raise ValueError("Optimized and reference timestamps do not match exactly")

    delta_power = (
        merged["hydro_power_mw_optimized"] - merged["hydro_power_mw_reference"]
    ).to_numpy(float)
    delta_energy = delta_power * interval_hours
    positive = np.maximum(delta_power, 0.0)
    e_shift_out = float(np.maximum(delta_energy, 0.0).sum())
    e_shift_in = float(np.maximum(-delta_energy, 0.0).sum())
    positive_hours = positive[positive > 1.0e-9]
    p_flex_max = float(positive.max(initial=0.0))
    p_flex_p95 = float(np.quantile(positive_hours, 0.95)) if len(positive_hours) else 0.0
    delta_flow = (
        merged["turbine_flow_m3s_optimized"]
        - merged["turbine_flow_m3s_reference"]
    ).to_numpy(float)
    shifted_volume = float(
        0.5 * np.abs(delta_flow).sum() * interval_hours * 3600.0
    )
    metrics = {
        "shift_out_energy_mwh": e_shift_out,
        "shift_in_energy_mwh": e_shift_in,
        "shift_energy_imbalance_mwh": e_shift_out - e_shift_in,
        "shifted_water_volume_m3": shifted_volume,
        "flex_up_max_mw": p_flex_max,
        "flex_up_p95_mw": p_flex_p95,
        "equivalent_duration_at_max_h": e_shift_out / p_flex_max if p_flex_max else 0.0,
        "equivalent_duration_at_p95_h": e_shift_out / p_flex_p95 if p_flex_p95 else 0.0,
    }
    metrics.update(_fifo_shift_pairing(merged["timestamp_lst"], delta_energy))
    metrics["matched_share_of_shift_out"] = (
        metrics["matched_shift_energy_mwh"] / e_shift_out if e_shift_out else float("nan")
    )
    metrics["matched_share_of_shift_in"] = (
        metrics["matched_shift_energy_mwh"] / e_shift_in if e_shift_in else float("nan")
    )
    return metrics
