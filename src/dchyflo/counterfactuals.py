"""Separate direct FPV, hydro-flexibility and carbon-timing contributions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .dispatch import redispatch_hydro, shared_export
from .grid import full_grid_redispatch, local_mef


def run_counterfactuals(
    baseline_hydro_mw,
    fpv_available_mw,
    demand_mw,
    fleet: pd.DataFrame,
    export_limit_mw: float,
    energy_retention: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run CF0-CF3 and return hourly cases plus a closed carbon decomposition."""
    hydro = np.asarray(baseline_hydro_mw, dtype=float)
    fpv = np.asarray(fpv_available_mw, dtype=float)
    demand = np.asarray(demand_mw, dtype=float)
    cf0_export = np.minimum(hydro, export_limit_mw)
    cf1 = shared_export(hydro, fpv, export_limit_mw)
    cf2_hydro = redispatch_hydro(hydro, fpv, export_limit_mw, energy_retention=energy_retention)
    cf2 = shared_export(cf2_hydro, fpv, export_limit_mw)
    signal = np.array([local_mef(d, 0.0, fleet) for d in demand])
    cf3_hydro = redispatch_hydro(
        hydro,
        fpv,
        export_limit_mw,
        carbon_signal=signal,
        energy_retention=energy_retention,
    )
    cf3 = shared_export(cf3_hydro, fpv, export_limit_mw)
    exports = {
        "CF0": cf0_export,
        "CF1": cf1["joint_export_mw"].to_numpy(),
        "CF2": cf2["joint_export_mw"].to_numpy(),
        "CF3": cf3["joint_export_mw"].to_numpy(),
    }
    emissions = {}
    for case, injection in exports.items():
        _, total = full_grid_redispatch(demand, injection, fleet)
        emissions[case] = total.to_numpy()
    # CF2 remains feasible for CF3. Reject a local-signal candidate that does not
    # improve the complete redispatch result.
    if emissions["CF3"].sum() > emissions["CF2"].sum() + 1e-9:
        cf3_hydro = cf2_hydro.copy()
        exports["CF3"] = exports["CF2"].copy()
        emissions["CF3"] = emissions["CF2"].copy()
    hourly = pd.DataFrame(
        {
            "hydro_baseline_mw": hydro,
            "fpv_available_mw": fpv,
            "CF0_export_mw": exports["CF0"],
            "CF1_export_mw": exports["CF1"],
            "CF2_export_mw": exports["CF2"],
            "CF3_export_mw": exports["CF3"],
            "CF2_hydro_mw": cf2_hydro,
            "CF3_hydro_mw": cf3_hydro,
            **{f"{case}_grid_emission_kgco2": value for case, value in emissions.items()},
        }
    )
    totals = {case: float(value.sum()) for case, value in emissions.items()}
    direct = totals["CF0"] - totals["CF1"]
    flexibility = totals["CF1"] - totals["CF2"]
    timing = totals["CF2"] - totals["CF3"]
    total = totals["CF0"] - totals["CF3"]
    components = [
        ("Direct FPV", direct),
        ("Hydro flexibility", flexibility),
        ("Carbon timing", timing),
        ("Interaction", total - direct - flexibility - timing),
    ]
    attribution = pd.DataFrame(components, columns=["contribution", "avoided_emission_kgco2"])
    attribution["share_percent"] = np.where(
        abs(total) > 1e-12, attribution["avoided_emission_kgco2"] / total * 100.0, 0.0
    )
    return hourly, attribution


def attribution_closes(attribution: pd.DataFrame, total_avoided_kgco2: float) -> bool:
    """Check that the three steps plus interaction equal CF0-CF3."""
    return bool(
        np.isclose(
            attribution["avoided_emission_kgco2"].sum(),
            total_avoided_kgco2,
            rtol=1e-9,
            atol=1e-6,
        )
    )
