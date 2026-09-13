"""Build a parameterized annual lifecycle carbon ledger."""

from __future__ import annotations

import numpy as np
import pandas as pd


def lifecycle_ledger(
    annual_avoided_tco2e: float,
    lifecycle_burden_tco2e: float,
    grid_decline_fraction: float,
    degradation_fraction: float,
    reservoir_increment_tco2e_year: float,
    horizon_years: int = 30,
    replacement_year: int = 15,
    replacement_burden_tco2e: float = 0.0,
    end_of_life_burden_tco2e: float = 0.0,
    module_d_credit_tco2e: float = 0.0,
) -> pd.DataFrame:
    """Return Year 0-30 avoided emissions, burdens and cumulative net carbon."""
    rows = []
    cumulative = -float(lifecycle_burden_tco2e)
    rows.append(
        {
            "year": 0,
            "avoided_emission_tco2e": 0.0,
            "lifecycle_event_tco2e": float(lifecycle_burden_tco2e),
            "reservoir_increment_tco2e": 0.0,
            "annual_net_carbon_tco2e": -float(lifecycle_burden_tco2e),
            "cumulative_net_carbon_tco2e": cumulative,
        }
    )
    for year in range(1, int(horizon_years) + 1):
        avoided = float(annual_avoided_tco2e) * (1 - grid_decline_fraction) ** (year - 1)
        avoided *= (1 - degradation_fraction) ** (year - 1)
        event = 0.0
        if year == int(replacement_year):
            event += float(replacement_burden_tco2e)
        if year == int(horizon_years):
            event += float(end_of_life_burden_tco2e) - float(module_d_credit_tco2e)
        annual_net = avoided - float(reservoir_increment_tco2e_year) - event
        cumulative += annual_net
        rows.append(
            {
                "year": year,
                "avoided_emission_tco2e": avoided,
                "lifecycle_event_tco2e": event,
                "reservoir_increment_tco2e": float(reservoir_increment_tco2e_year),
                "annual_net_carbon_tco2e": annual_net,
                "cumulative_net_carbon_tco2e": cumulative,
            }
        )
    return pd.DataFrame(rows)


def carbon_payback_year(ledger: pd.DataFrame) -> float:
    """Return the first post-construction year with positive cumulative net carbon."""
    positive = ledger.loc[
        (ledger["year"] > 0) & (ledger["cumulative_net_carbon_tco2e"] > 0), "year"
    ]
    return float(positive.iloc[0]) if not positive.empty else np.nan
