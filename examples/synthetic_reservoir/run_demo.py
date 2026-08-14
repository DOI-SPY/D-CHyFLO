"""Run a lightweight, entirely synthetic D-CHyFLO demonstration."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dchyflo import PhysicalCurves
from dchyflo.carbon_accounting import DynamicPathway, dynamic_ledger, summarize_paths


ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"
OUTPUT = ROOT / "demo_output"


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    curves = PhysicalCurves(DATA)
    levels = pd.Series([101.0, 104.0, 107.0, 109.0])
    curve_demo = pd.DataFrame(
        {
            "level_m": levels,
            "storage_1e8m3": curves.storage_from_level(levels),
            "area_km2": curves.area_from_level(levels),
        }
    )
    curve_demo.to_csv(OUTPUT / "synthetic_curve_evaluation.csv", index=False)

    static = pd.DataFrame(
        [
            {
                "incremental_hybrid_export_mwh": 100_000.0,
                "technology_lifecycle_gco2e_per_kwh": 45.0,
                "grid_factor_kgco2e_per_mwh": 500.0,
                "local_aquatic_delta_tco2e_yr": 250.0,
                "capacity_mwac": 100.0,
                "proxy_scenario": "synthetic_reference",
                "s0_reference": "synthetic_s0",
                "year": 2026,
                "signal_name": "synthetic_signal",
                "cold_scenario": "synthetic_cold",
                "capacity_source_status": "SYNTHETIC_EXAMPLE",
            }
        ]
    )
    pathways = [
        DynamicPathway("reference", 0.03, 0.005, 1.0, 5, "linear_recovery", "synthetic"),
        DynamicPathway("deep_decarbonization", 0.10, 0.008, 1.2, 10, "sustained", "synthetic"),
    ]
    ledger = dynamic_ledger(static, pathways, 30, 0.75, 0.15, 0.10, "synthetic")
    summary = summarize_paths(ledger)
    ledger.to_csv(OUTPUT / "synthetic_dynamic_ledger.csv", index=False)
    summary.to_csv(OUTPUT / "synthetic_path_summary.csv", index=False)

    status = {
        "curve_rows": len(curve_demo),
        "ledger_rows": len(ledger),
        "pathways": len(summary),
        "data_class": "SYNTHETIC_NO_REAL_SITE_CORRESPONDENCE",
    }
    (OUTPUT / "demo_summary.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
