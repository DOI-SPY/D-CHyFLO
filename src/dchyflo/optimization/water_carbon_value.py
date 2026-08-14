"""Finite-difference marginal carbon value of controllable reservoir water.

This module intentionally uses the full annual storage-balance formulation,
not the intraday PWL model whose daily turbine volume is fixed.  A symmetric
volume perturbation is distributed uniformly across the days of one calendar
month.  The annual energy reference and carbon-aware dispatch are re-solved for
each perturbation, preserving the terminal-storage target.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dchyflo.optimization.carbon_dispatch import optimize_carbon_aware_state


SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class WaterValueTask:
    case_id: str
    s0_reference: str
    year: int
    hydrologic_regime: str
    capacity_mwac: int
    signal_name: str
    month: int
    delta_volume_m3: float
    hourly_path: str
    daily_path: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def perturb_monthly_inflow(
    daily: pd.DataFrame, month: int, delta_volume_m3: float
) -> pd.DataFrame:
    """Add a signed total volume uniformly to one calendar month."""

    result = daily.copy()
    dates = pd.to_datetime(result["date"])
    mask = dates.dt.month.eq(int(month))
    count = int(mask.sum())
    if count == 0:
        raise ValueError(f"No daily records found for month {month}")
    candidates = (
        "effective_gross_inflow_assimilated_m3s",
        "gross_inflow_reconstructed_m3s",
        "inflow_m3s",
    )
    column = next((name for name in candidates if name in result.columns), None)
    if column is None:
        raise ValueError("No supported inflow column is available")
    result.loc[mask, column] = (
        pd.to_numeric(result.loc[mask, column], errors="raise")
        + float(delta_volume_m3) / (count * SECONDS_PER_DAY)
    )
    return result


def _solve_task(task: WaterValueTask, settings: dict[str, Any]) -> dict[str, Any]:
    hourly = pd.read_csv(task.hourly_path)
    daily = pd.read_csv(task.daily_path)
    if task.delta_volume_m3 != 0.0:
        daily = perturb_monthly_inflow(
            daily, task.month, task.delta_volume_m3
        )
    result = optimize_carbon_aware_state(
        hourly=hourly,
        daily=daily,
        grid_limit_mw=float(settings["grid_limit_mw"]),
        energy_retention_fraction=float(settings["energy_retention_fraction"]),
        rated_turbine_flow_m3s=float(settings["rated_turbine_flow_m3s"]),
        energy_floor_tolerance_mwh=float(settings["energy_floor_tolerance_mwh"]),
        path_regularization_weight_mwh_equivalent=float(
            settings["path_regularization_weight_mwh_equivalent"]
        ),
    )
    return {
        "case_id": task.case_id,
        "s0_reference": task.s0_reference,
        "year": task.year,
        "hydrologic_regime": task.hydrologic_regime,
        "capacity_mwac": task.capacity_mwac,
        "signal_name": task.signal_name,
        "month": task.month,
        "delta_volume_m3": task.delta_volume_m3,
        "termination": result.termination,
        "energy_optimum_mwh": result.energy_optimum_mwh,
        "energy_floor_mwh": result.energy_floor_mwh,
        "achieved_export_mwh": result.achieved_export_mwh,
        "carbon_optimum_kgco2": result.carbon_optimum_kgco2,
        "achieved_carbon_kgco2": result.achieved_carbon_kgco2,
        "carbon_optimum_gap_kgco2": (
            result.carbon_optimum_kgco2 - result.achieved_carbon_kgco2
        ),
        "nonpower_release_m3": float(
            result.daily["optimized_nonpower_release_m3"].sum()
        ),
        "max_water_balance_residual_m3": float(
            result.daily["water_balance_residual_m3"].abs().max()
        ),
        "initial_storage_1e8m3": result.initial_storage_1e8m3,
        "terminal_storage_target_1e8m3": result.terminal_storage_target_1e8m3,
    }


def _case_paths(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    source = root / config["source_root"]
    cases: list[dict[str, Any]] = []
    for s0 in config["s0_references"]:
        for year in config["years"]:
            for capacity in config["capacities_mwac"]:
                case_id = f"{str(s0).lower()}_{year}_{capacity}mwac"
                folder = source / case_id
                for signal in config["signal_names"]:
                    hourly = folder / f"{signal}_hourly.csv"
                    daily = folder / f"{signal}_daily.csv"
                    if not hourly.exists() or not daily.exists():
                        raise FileNotFoundError(f"Missing frozen case evidence: {folder}")
                    cases.append(
                        {
                            "case_id": case_id,
                            "s0_reference": s0,
                            "year": int(year),
                            "hydrologic_regime": config["hydrologic_regimes"][str(year)],
                            "capacity_mwac": int(capacity),
                            "signal_name": signal,
                            "hourly_path": str(hourly),
                            "daily_path": str(daily),
                        }
                    )
    return cases


def build_tasks(root: Path, config: dict[str, Any]) -> list[WaterValueTask]:
    tasks: list[WaterValueTask] = []
    for case in _case_paths(root, config):
        tasks.append(WaterValueTask(**case, month=0, delta_volume_m3=0.0))
        for month in range(1, 13):
            for volume in config["perturbation_volumes_m3"]:
                for sign in (-1.0, 1.0):
                    tasks.append(
                        WaterValueTask(
                            **case,
                            month=month,
                            delta_volume_m3=sign * float(volume),
                        )
                    )
    return tasks


def summarize_water_values(raw: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "case_id", "s0_reference", "year", "hydrologic_regime",
        "capacity_mwac", "signal_name", "month",
    ]
    rows: list[dict[str, Any]] = []
    perturb = raw.loc[raw["month"].gt(0)].copy()
    perturb["abs_delta_volume_m3"] = perturb["delta_volume_m3"].abs()
    for group_key, group in perturb.groupby(keys + ["abs_delta_volume_m3"], sort=True):
        plus = group.loc[group["delta_volume_m3"].gt(0)].iloc[0]
        minus = group.loc[group["delta_volume_m3"].lt(0)].iloc[0]
        volume = float(plus["delta_volume_m3"])
        central_carbon = (
            float(plus["achieved_carbon_kgco2"])
            - float(minus["achieved_carbon_kgco2"])
        ) / (2.0 * volume)
        central_energy = (
            float(plus["achieved_export_mwh"])
            - float(minus["achieved_export_mwh"])
        ) / (2.0 * volume)
        row = dict(zip(keys + ["perturbation_volume_m3"], group_key))
        row.update(
            {
                "marginal_carbon_kgco2_per_m3": central_carbon,
                "marginal_carbon_gco2_per_m3": 1000.0 * central_carbon,
                "marginal_export_mwh_per_m3": central_energy,
                "plus_carbon_kgco2": float(plus["achieved_carbon_kgco2"]),
                "minus_carbon_kgco2": float(minus["achieved_carbon_kgco2"]),
                "plus_nonpower_release_m3": float(plus["nonpower_release_m3"]),
                "minus_nonpower_release_m3": float(minus["nonpower_release_m3"]),
                "max_water_balance_residual_m3": float(
                    group["max_water_balance_residual_m3"].max()
                ),
                "max_abs_carbon_optimum_gap_kgco2": float(
                    group["carbon_optimum_gap_kgco2"].abs().max()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys + ["perturbation_volume_m3"])


def add_scale_stability(values: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "case_id", "s0_reference", "year", "hydrologic_regime",
        "capacity_mwac", "signal_name", "month",
    ]
    result = values.copy()
    result["scale_relative_difference"] = np.nan
    for _, index in result.groupby(keys).groups.items():
        subset = result.loc[index].sort_values("perturbation_volume_m3")
        if len(subset) != 2:
            continue
        slopes = subset["marginal_carbon_kgco2_per_m3"].to_numpy(float)
        denominator = max(float(np.max(np.abs(slopes))), 1.0e-12)
        relative = float(abs(slopes[1] - slopes[0]) / denominator)
        result.loc[index, "scale_relative_difference"] = relative
    return result


def run_checkpoint16r1(config_path: Path, output_dir: Path, root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(root, config)
    rows: list[dict[str, Any]] = []
    workers = int(config.get("workers", 1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_solve_task, task, config) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
    raw = pd.DataFrame(rows).sort_values(
        ["case_id", "month", "delta_volume_m3"]
    )
    values = add_scale_stability(summarize_water_values(raw))
    tol = config["tolerances"]
    checks = pd.DataFrame(
        [
            {
                "check": "all_solves_optimal",
                "value": int(raw["termination"].str.lower().eq("optimal").sum()),
                "threshold": len(raw),
                "passed": bool(raw["termination"].str.lower().eq("optimal").all()),
            },
            {
                "check": "all_values_finite",
                "value": int(np.isfinite(values["marginal_carbon_kgco2_per_m3"]).sum()),
                "threshold": len(values),
                "passed": bool(np.isfinite(values["marginal_carbon_kgco2_per_m3"]).all()),
            },
            {
                "check": "water_balance",
                "value": float(raw["max_water_balance_residual_m3"].max()),
                "threshold": float(tol["water_balance_residual_m3"]),
                "passed": bool(raw["max_water_balance_residual_m3"].max() <= float(tol["water_balance_residual_m3"])),
            },
            {
                "check": "carbon_objective_retained",
                "value": float(raw["carbon_optimum_gap_kgco2"].abs().max()),
                "threshold": float(tol["carbon_optimum_gap_kgco2"]),
                "passed": bool(raw["carbon_optimum_gap_kgco2"].abs().max() <= float(tol["carbon_optimum_gap_kgco2"])),
            },
            {
                "check": "dual_scale_stability_fraction",
                "value": float((values["scale_relative_difference"] <= float(tol["central_slope_scale_relative_difference"])).mean()),
                "threshold": 0.8,
                "passed": bool((values["scale_relative_difference"] <= float(tol["central_slope_scale_relative_difference"])).mean() >= 0.8),
            },
        ]
    )
    summary = {
        "artifact_id": config["artifact_id"],
        "checkpoint": "16-R1",
        "checkpoint_status": "COMPLETE_VERIFIED" if bool(checks["passed"].all()) else "DIAGNOSTIC_NOT_RELEASED",
        "contexts": int(raw["case_id"].nunique()),
        "solves": int(len(raw)),
        "monthly_value_rows": int(len(values)),
        "marginal_carbon_range_kgco2_per_m3": [
            float(values["marginal_carbon_kgco2_per_m3"].min()),
            float(values["marginal_carbon_kgco2_per_m3"].max()),
        ],
        "median_marginal_carbon_kgco2_per_m3": float(values["marginal_carbon_kgco2_per_m3"].median()),
        "scale_stable_fraction": float((values["scale_relative_difference"] <= float(tol["central_slope_scale_relative_difference"])).mean()),
        "checks_passed": int(checks["passed"].sum()),
        "checks_total": int(len(checks)),
        "claim_boundary": config["claim_boundary"],
        "next_stage": "CP16-R2: joint design and two-stage risk-aware optimization architecture",
    }
    raw.to_csv(output_dir / "monthly_perturbation_solves.csv", index=False)
    values.to_csv(output_dir / "monthly_marginal_water_carbon_value.csv", index=False)
    checks.to_csv(output_dir / "checkpoint_16r1_checks.csv", index=False)
    (output_dir / "checkpoint_16r1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    hashes = []
    for path in sorted(output_dir.glob("*")):
        if path.is_file() and path.name != "artifact_hashes.csv":
            hashes.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path)})
    pd.DataFrame(hashes).to_csv(output_dir / "artifact_hashes.csv", index=False)
    return summary
