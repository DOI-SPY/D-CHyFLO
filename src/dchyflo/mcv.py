"""Calculate Marginal Water-Carbon Value with symmetric perturbations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def central_difference_mcv(
    minus_emission_kgco2: float,
    plus_emission_kgco2: float,
    perturbation_million_m3: float,
) -> float:
    """Return additional avoided carbon per added m3 of available water."""
    delta_m3 = float(perturbation_million_m3) * 1e6
    if delta_m3 <= 0:
        raise ValueError("water perturbation must be positive")
    value = (float(minus_emission_kgco2) - float(plus_emission_kgco2)) / (2 * delta_m3)
    return float(value)


def summarize_mcv(values) -> pd.DataFrame:
    """Return minimum, median and maximum MCV in kg CO2/m3."""
    array = np.asarray(values, dtype=float)
    return pd.DataFrame(
        {
            "metric": ["minimum", "median", "maximum"],
            "mcv_kgco2_per_m3": [array.min(), np.median(array), array.max()],
        }
    )
