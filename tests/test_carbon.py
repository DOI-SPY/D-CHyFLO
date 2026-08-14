import numpy as np
import pandas as pd

from dchyflo.carbon_accounting import DynamicPathway, dynamic_ledger, summarize_paths
from dchyflo.reservoir_ghg import factorial_decomposition


def test_dynamic_ledger_closes_annual_and_cumulative_identity() -> None:
    static = pd.DataFrame(
        [
            {
                "incremental_hybrid_export_mwh": 10_000.0,
                "technology_lifecycle_gco2e_per_kwh": 50.0,
                "grid_factor_kgco2e_per_mwh": 500.0,
                "local_aquatic_delta_tco2e_yr": 20.0,
                "capacity_mwac": 100.0,
                "proxy_scenario": "synthetic_reference",
                "s0_reference": "synthetic_s0",
                "year": 2026,
                "signal_name": "synthetic_signal",
            }
        ]
    )
    paths = [DynamicPathway("p1", 0.02, 0.005, 1.0, 3, "linear_recovery", "synthetic")]
    ledger = dynamic_ledger(static, paths, 30, 0.75, 0.15, 0.10, "synthetic")
    annual = ledger.annual_net_avoided_tco2e.to_numpy(float)
    np.testing.assert_allclose(ledger.cumulative_net_avoided_tco2e, np.cumsum(annual))
    summary = summarize_paths(ledger)
    assert len(summary) == 1
    assert np.isfinite(summary.lifetime_net_avoided_tco2e.iloc[0])


def test_factorial_decomposition_sums_to_total_variance() -> None:
    rows = []
    for depth in ("low", "high"):
        for river in ("short", "long"):
            for treatment in (0, 1):
                for landuse in ("low", "high"):
                    rows.append(
                        {
                            "depth_anchor_id": depth,
                            "river_length_anchor_id": river,
                            "treatment_factor": treatment,
                            "landuse_intensity": landuse,
                            "net_ghg_tco2e_yr": (depth == "high") * 4 + (river == "long") * 2 + treatment,
                        }
                    )
    result = factorial_decomposition(pd.DataFrame(rows))
    assert abs(result.variance_share_percent.sum() - 100.0) < 1e-9
