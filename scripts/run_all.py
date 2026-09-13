"""Run the public synthetic workflow and write the core outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dchyflo.break_even import bisect_zero
from dchyflo.capacity import run_capacity_scan
from dchyflo.counterfactuals import run_counterfactuals
from dchyflo.data_io import load_config, load_example_inputs
from dchyflo.fpv import simulate_fpv
from dchyflo.grid import avoided_emissions_by_redispatch, local_mef
from dchyflo.hydro import hourly_hydro_profile, reconstruct_hydro, water_balance_pass
from dchyflo.lifecycle import carbon_payback_year, lifecycle_ledger
from dchyflo.mcv import central_difference_mcv, summarize_mcv
from dchyflo.reservoir_ghg import conditional_scenarios
from dchyflo.validation import frozen_result_table, validate_registered_results


def _absolute_data_paths(config: dict) -> dict:
    result = dict(config)
    result["data"] = {
        name: str((ROOT / path).resolve()) if not Path(path).is_absolute() else path
        for name, path in config["data"].items()
    }
    return result


def run(config_path: str | Path) -> Path:
    """Execute all public stages with synthetic or user-supplied inputs."""
    config = _absolute_data_paths(load_config(config_path))
    inputs = load_example_inputs(config)
    output = ROOT / config["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)

    hydro = reconstruct_hydro(inputs["reservoir"], inputs["hva"], config)
    if not water_balance_pass(hydro):
        raise RuntimeError("water-balance quality check failed")
    hydro.to_csv(output / "hydro_reconstruction.csv", index=False)

    meteorology = inputs["meteorology"]
    meteorology["timestamp"] = pd.to_datetime(meteorology["timestamp"], utc=True)
    hydro_hourly = hourly_hydro_profile(hydro, meteorology["timestamp"]).to_numpy()
    fpv = simulate_fpv(meteorology, config)
    fpv.to_csv(output / "fpv_generation.csv", index=False)

    hourly, attribution = run_counterfactuals(
        hydro_hourly,
        fpv["fpv_available_mw"],
        meteorology["grid_demand_mw"],
        inputs["grid"],
        float(config["hydro"]["export_limit_mw"]),
        float(config["counterfactuals"]["energy_retention"]),
    )
    hourly.insert(0, "timestamp", meteorology["timestamp"].to_numpy())
    hourly.to_csv(output / "counterfactual_hourly.csv", index=False)
    attribution.to_csv(output / "operational_attribution.csv", index=False)

    mef_rows = []
    center = len(meteorology) // 2
    base_demand = float(meteorology["grid_demand_mw"].iloc[center])
    base_mef = local_mef(base_demand, 0.0, inputs["grid"])
    for shift in config["mef"]["perturbation_mw"]:
        shifted = max(base_demand - float(shift), 0.0)
        scan_mef = local_mef(shifted, 0.0, inputs["grid"])
        deviation = abs(scan_mef - base_mef) / max(abs(base_mef), 1e-12)
        mef_rows.append(
            {
                "shift_mw": shift,
                "local_mef_kgco2_per_mwh": scan_mef,
                "relative_deviation": deviation,
                "within_10_percent": deviation <= config["mef"]["relative_deviation_threshold"],
            }
        )
    pd.DataFrame(mef_rows).to_csv(output / "mef_validity.csv", index=False)

    # The example response is deterministic and only demonstrates the central-difference interface.
    slopes = [0.0, 0.035, 0.050, 0.060]
    mcv_values = []
    for perturbation, slope in zip(config["mcv"]["perturbations_million_m3"], slopes, strict=True):
        delta = float(perturbation) * 1e6
        mcv_values.append(central_difference_mcv(slope * delta, -slope * delta, perturbation))
    summarize_mcv(mcv_values).to_csv(output / "mcv_summary.csv", index=False)

    minimum_area = float(inputs["hva"]["area_km2"].min())
    capacity = run_capacity_scan(meteorology, hydro_hourly, minimum_area, config)
    avoided = []
    for row in capacity.itertuples(index=False):
        design = simulate_fpv(meteorology, config, row.capacity_mwac, row.dc_ac_ratio)
        carbon = avoided_emissions_by_redispatch(
            meteorology["grid_demand_mw"],
            hydro_hourly,
            hydro_hourly + design["fpv_available_mw"].to_numpy(),
            inputs["grid"],
        )
        avoided.append(float(carbon.sum()))
    capacity["operational_avoided_kgco2"] = avoided
    capacity["marginal_avoided_kgco2_per_added_mw"] = (
        capacity.groupby("dc_ac_ratio")["operational_avoided_kgco2"].diff()
        / capacity.groupby("dc_ac_ratio")["capacity_mwac"].diff()
    )
    capacity.to_csv(output / "capacity_summary.csv", index=False)

    annual_avoided = float(
        hourly["CF0_grid_emission_kgco2"].sub(hourly["CF3_grid_emission_kgco2"]).sum()
        / 1000.0
        * 8760.0
        / len(hourly)
    )
    annual_generation_mwh = float(fpv["fpv_available_mw"].sum() * 8760.0 / len(fpv))
    lifetime_burden = 36.0 * annual_generation_mwh * 30.0 / 1000.0
    ghg = conditional_scenarios(
        np.linspace(0.2, 1.0, int(config["reservoir_ghg"]["conditional_scenario_count"])),
        covered_area_km2=1.0,
        responses=config["reservoir_ghg"]["covered_area_response_fraction"],
    )
    ghg.to_csv(output / "reservoir_ghg_scenarios.csv", index=False)
    ledger = lifecycle_ledger(
        annual_avoided,
        lifetime_burden,
        grid_decline_fraction=0.03,
        degradation_fraction=0.006,
        reservoir_increment_tco2e_year=0.0,
        horizon_years=int(config["lifecycle"]["horizon_years"]),
        replacement_year=int(config["lifecycle"]["replacement_year"]),
        end_of_life_burden_tco2e=0.01 * lifetime_burden,
        module_d_credit_tco2e=float(config["lifecycle"]["module_d_credit"]),
    )
    ledger.to_csv(output / "lifecycle_ledger.csv", index=False)
    pd.DataFrame(
        [
            {
                "year30_net_carbon_tco2e": ledger["cumulative_net_carbon_tco2e"].iloc[-1],
                "carbon_payback_year": carbon_payback_year(ledger),
                "accounting": "parameterized dynamic lifecycle carbon accounting",
            }
        ]
    ).to_csv(output / "lifecycle_summary.csv", index=False)

    base_net = float(ledger["cumulative_net_carbon_tco2e"].iloc[-1])
    diagnostic_scale = max(abs(base_net), 1.0)
    break_even_multiplier = bisect_zero(
        lambda multiplier: diagnostic_scale * (1.0 - multiplier), 0.0, 2.0
    )
    pd.DataFrame(
        [
            {
                "diagnostic": "combined burden multiplier",
                "break_even_value": break_even_multiplier,
                "domain": "expanded diagnostic search",
                "probability": "not applicable",
            }
        ]
    ).to_csv(output / "break_even_summary.csv", index=False)

    failures = validate_registered_results()
    frozen = frozen_result_table()
    if failures:
        frozen["status"] = "FAIL"
        frozen["details"] = "; ".join(failures)
    frozen.to_csv(output / "frozen_result_check.csv", index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/example.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    output = run(config_path)
    print(f"D-CHyFLO example completed: {output}")


if __name__ == "__main__":
    main()
