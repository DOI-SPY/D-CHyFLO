import numpy as np
import pandas as pd

from dchyflo.dispatch import fixed_noncooperative_dispatch, optimize_dekadal_dispatch


def test_cooperation_preserves_water_and_reduces_or_matches_curtailment() -> None:
    timestamps = pd.date_range("2026-06-01", periods=48, freq="h")
    pv = np.tile([0.0] * 8 + [100.0] * 8 + [0.0] * 8, 2)
    hourly = pd.DataFrame({"timestamp_lst": timestamps, "pv_power_mw": pv})
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-01", "2026-06-02"]),
            "observed_hydro_energy_mwh": [2400.0, 2400.0],
            "specific_energy_mwh_per_m3": [0.00005, 0.00005],
            "dynamic_max_output_mw": [250.0, 250.0],
            "storage_1e8m3": [3.0, 3.0],
            "storage_lower_1e8m3": [2.0, 2.0],
            "storage_upper_1e8m3": [4.0, 4.0],
        }
    )
    fixed = fixed_noncooperative_dispatch(hourly, daily, 150.0)
    optimized = optimize_dekadal_dispatch(hourly, daily, 150.0)
    assert optimized.hourly.pv_curtailment_mw.sum() <= fixed.pv_curtailment_mw.sum() + 1e-7
    expected_water = (daily.observed_hydro_energy_mwh / daily.specific_energy_mwh_per_m3).sum()
    assert abs(optimized.daily.optimized_turbine_water_m3.sum() - expected_water) < 1e-3
