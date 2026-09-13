import numpy as np
import pandas as pd

from dchyflo.counterfactuals import attribution_closes, run_counterfactuals


def _fleet():
    return pd.DataFrame(
        {
            "generator": ["gas", "coal"],
            "capacity_mw": [250.0, 500.0],
            "marginal_cost_per_mwh": [40.0, 60.0],
            "emission_kgco2_per_mwh": [390.0, 820.0],
        }
    )


def test_cf_steps_close_and_cf3_retains_all_hydro_energy():
    hydro = np.array([120.0, 120.0, 80.0, 80.0])
    fpv = np.array([180.0, 150.0, 0.0, 0.0])
    hourly, attribution = run_counterfactuals(
        hydro, fpv, np.full(4, 500.0), _fleet(), 250.0, energy_retention=1.0
    )
    total = hourly["CF0_grid_emission_kgco2"].sum() - hourly["CF3_grid_emission_kgco2"].sum()
    assert attribution_closes(attribution, total)
    assert np.isclose(hourly["CF3_hydro_mw"].sum(), hydro.sum())
    assert hourly["CF3_grid_emission_kgco2"].sum() <= hourly["CF2_grid_emission_kgco2"].sum()
