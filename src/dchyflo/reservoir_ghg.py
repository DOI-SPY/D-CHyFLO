"""Keep reservoir background flux separate from the FPV-covered-area response."""

from __future__ import annotations

import pandas as pd


def covered_area_increment(
    background_flux_kgco2e_m2_year: float,
    covered_area_km2: float,
    response_fraction: float,
) -> float:
    """Return only the conditional increment over the FPV-covered area in t CO2e/year."""
    return (
        float(background_flux_kgco2e_m2_year)
        * float(covered_area_km2)
        * 1e6
        * float(response_fraction)
        / 1000.0
    )


def conditional_scenarios(
    background_fluxes, covered_area_km2: float, responses=(-0.10, 0.0, 0.10)
) -> pd.DataFrame:
    """Build conditional response cases without changing reservoir background emissions."""
    rows = []
    for case_id, flux in enumerate(background_fluxes, start=1):
        for response in responses:
            rows.append(
                {
                    "case_id": case_id,
                    "reservoir_background_flux_kgco2e_m2_year": float(flux),
                    "covered_area_response_fraction": float(response),
                    "fpv_covered_area_increment_tco2e_year": covered_area_increment(
                        flux, covered_area_km2, response
                    ),
                }
            )
    return pd.DataFrame(rows)
