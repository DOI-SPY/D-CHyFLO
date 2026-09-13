from pathlib import Path

from dchyflo.capacity import geometry_screen
from dchyflo.data_io import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_capacity_scan_respects_geometry_screen():
    config = load_config(ROOT / "config/example.yaml")
    assert geometry_screen(500, 1.0, 195.0, config)
    assert not geometry_screen(550, 1.0, 195.0, config)
    assert not geometry_screen(500, 1.5, 195.0, config)
