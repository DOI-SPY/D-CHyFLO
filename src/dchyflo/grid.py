"""Calculate operational emissions with full hourly grid redispatch."""

from __future__ import annotations

import numpy as np
import pandas as pd


def full_grid_redispatch(
    demand_mw,
    renewable_injection_mw,
    fleet: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Dispatch the complete generator fleet for each hour and return emissions.

    Final operational carbon is the difference between two complete redispatch runs.
    It is not calculated as renewable generation multiplied by an MEF.
    """
    demand = np.asarray(demand_mw, dtype=float)
    injection = np.asarray(renewable_injection_mw, dtype=float)
    ordered = fleet.sort_values("marginal_cost_per_mwh").reset_index(drop=True)
    records: list[dict] = []
    hourly_emissions: list[float] = []
    for hour, (load, clean) in enumerate(zip(demand, injection, strict=True)):
        remaining = max(load - clean, 0.0)
        emissions = 0.0
        for row in ordered.itertuples(index=False):
            dispatched = min(float(row.capacity_mw), remaining)
            records.append(
                {
                    "hour": hour,
                    "generator": row.generator,
                    "dispatch_mw": dispatched,
                    "emission_kgco2": dispatched * float(row.emission_kgco2_per_mwh),
                }
            )
            emissions += dispatched * float(row.emission_kgco2_per_mwh)
            remaining -= dispatched
        if remaining > 1e-8:
            raise ValueError(f"grid demand exceeds fleet capacity in hour {hour}")
        hourly_emissions.append(emissions)
    return pd.DataFrame(records), pd.Series(hourly_emissions, name="emission_kgco2")


def avoided_emissions_by_redispatch(demand_mw, baseline_injection_mw, case_injection_mw, fleet):
    """Return hourly avoided emissions from two complete grid dispatch runs."""
    _, baseline = full_grid_redispatch(demand_mw, baseline_injection_mw, fleet)
    _, case = full_grid_redispatch(demand_mw, case_injection_mw, fleet)
    return baseline - case


def local_mef(
    demand_mw: float, injection_mw: float, fleet: pd.DataFrame, delta_mw: float = 1.0
) -> float:
    """Estimate a local marginal response for interpretation and validity diagnosis."""
    _, low = full_grid_redispatch([demand_mw], [injection_mw], fleet)
    _, high = full_grid_redispatch([demand_mw], [injection_mw + delta_mw], fleet)
    return float((low.iloc[0] - high.iloc[0]) / delta_mw)
