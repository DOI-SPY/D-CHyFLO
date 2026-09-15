from pathlib import Path
import sys

import pandas as pd
import yaml    

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_all import run


def test_example_pipeline_writes_core_outputs(tmp_path):
    config = yaml.safe_load(
        (ROOT / "config/example.yaml").read_text(encoding="utf-8")
    )
    config["outputs"]["directory"] = str(tmp_path)

    config_path = tmp_path / "example.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    output = run(config_path)
    assert output == tmp_path
    expected = {
        "operational_attribution.csv",
        "mcv_summary.csv",
        "capacity_summary.csv",
        "lifecycle_summary.csv",
        "break_even_summary.csv",
        "frozen_result_check.csv",
    }
    assert expected <= {path.name for path in output.iterdir()}
    fpv = pd.read_csv(output / "fpv_generation.csv")
    assert fpv["fpv_available_mw"].notna().all()
    assert (fpv["fpv_available_mw"] >= 0).all()
    attribution = pd.read_csv(output / "operational_attribution.csv")
    assert attribution["avoided_emission_kgco2"].sum() >= 0
