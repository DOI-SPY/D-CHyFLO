"""Daily reservoir-state accounting for a D-CHyFLO reservoir case."""

from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import pandas as pd

from .curves import PhysicalCurves


def add_physical_losses(data: pd.DataFrame, curves: PhysicalCurves, losses_csv: Path) -> pd.DataFrame:
    """Add explicit evaporation/seepage while preserving reported net inflow.

    Historical inflow is balance-derived and therefore treated as net of
    unmodelled loss. Gross inflow is reconstructed by adding the same loss that
    the replay subsequently subtracts.
    """
    losses = pd.read_csv(losses_csv).set_index("month")
    result = data.copy()
    average_area = (
        result["surface_area_km2"].shift(1).fillna(result["surface_area_km2"])
        + result["surface_area_km2"]
    ) / 2.0
    days = result["date"].map(lambda d: calendar.monthrange(d.year, d.month)[1]).astype(float)
    total_mm = result["date"].dt.month.map(
        (losses["evaporation_increment_mm"] + losses["seepage_mm"]).to_dict()
    ).astype(float)
    result["reservoir_loss_mm_day"] = total_mm / days
    result["reservoir_loss_m3_day"] = result["reservoir_loss_mm_day"] * average_area * 1000.0
    result["reservoir_loss_equivalent_m3s"] = result["reservoir_loss_m3_day"] / 86400.0
    result["gross_inflow_reconstructed_m3s"] = (
        result["inflow_m3s"] + result["reservoir_loss_equivalent_m3s"]
    )
    return result


def replay_water_balance(data: pd.DataFrame, curves: PhysicalCurves, losses_csv: Path) -> pd.DataFrame:
    """Diagnose water balance under alternative reported-inflow semantics.

    The source series mixes balance-derived and hydrologically estimated inflow.
    Two free replays are therefore retained: reported inflow as net inflow, and
    reported inflow as gross inflow before official evaporation/seepage loss.
    A closure-flow term is calculated for later state-assimilated scenarios but
    is never hidden inside the reported inflow column.
    """
    result = add_physical_losses(data, curves, losses_csv)
    simulated_net = np.full(len(result), np.nan, dtype=float)
    simulated_gross = np.full(len(result), np.nan, dtype=float)
    for _, indices in result.groupby("year", sort=True).groups.items():
        positions = result.index.get_indexer(indices)
        simulated_net[positions[0]] = result.loc[indices[0], "storage_1e8m3"]
        simulated_gross[positions[0]] = result.loc[indices[0], "storage_1e8m3"]
        for previous_pos, current_pos in zip(positions[:-1], positions[1:]):
            row = result.iloc[current_pos]
            net_delta = (row["inflow_m3s"] - row["outflow_m3s"]) * 86400.0 / 1e8
            gross_delta = net_delta - row["reservoir_loss_m3_day"] / 1e8
            simulated_net[current_pos] = simulated_net[previous_pos] + net_delta
            simulated_gross[current_pos] = simulated_gross[previous_pos] + gross_delta
    result["simulated_storage_net_inflow_assumption_1e8m3"] = simulated_net
    result["simulated_storage_gross_inflow_assumption_1e8m3"] = simulated_gross
    result["storage_replay_residual_net_1e8m3"] = result["storage_1e8m3"] - simulated_net
    result["storage_replay_residual_gross_1e8m3"] = result["storage_1e8m3"] - simulated_gross

    delta_storage_m3s = result["storage_change_1e8m3_corrected"] * 1e8 / 86400.0
    result["water_balance_closure_m3s"] = (
        delta_storage_m3s - result["inflow_m3s"] + result["outflow_m3s"]
        + result["reservoir_loss_equivalent_m3s"]
    )
    result["effective_gross_inflow_assimilated_m3s"] = (
        result["inflow_m3s"] + result["water_balance_closure_m3s"]
    )
    return result
