from pathlib import Path

from dchyflo.data_io import load_config, load_example_inputs
from dchyflo.hydro import reconstruct_hydro, water_balance_pass

ROOT = Path(__file__).resolve().parents[1]


def test_example_water_balance_is_within_tolerance():
    config = load_config(ROOT / "config/example.yaml")
    config["data"] = {key: str(ROOT / value) for key, value in config["data"].items()}
    inputs = load_example_inputs(config)
    result = reconstruct_hydro(inputs["reservoir"], inputs["hva"], config)
    assert water_balance_pass(result)
    assert result["level_within_bounds"].all()
    assert result["power_within_bounds"].all()
