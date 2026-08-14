from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dchyflo import PhysicalCurves


DATA = Path(__file__).resolve().parents[1] / "examples" / "synthetic_reservoir" / "data"


def test_hva_nodes_round_trip_exactly() -> None:
    curves = PhysicalCurves(DATA)
    table = pd.read_csv(DATA / "reservoir_hva.csv")
    np.testing.assert_allclose(
        curves.storage_from_level(table.level_m), table.storage_1e8m3, atol=1e-12
    )
    np.testing.assert_allclose(
        curves.level_from_storage(table.storage_1e8m3), table.level_m, atol=1e-10
    )


def test_hva_rejects_extrapolation() -> None:
    curves = PhysicalCurves(DATA)
    with pytest.raises(ValueError, match="outside"):
        curves.storage_from_level(np.array([99.9]))


def test_tailwater_marks_endpoint_extension() -> None:
    curves = PhysicalCurves(DATA)
    result = curves.tailwater_from_flow(np.array([100.0, 1200.0]))
    assert result.out_of_range.tolist() == [False, True]
    assert np.isfinite(result.values).all()
