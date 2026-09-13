from dchyflo.reservoir_ghg import conditional_scenarios


def test_covered_area_stress_does_not_modify_background_flux():
    result = conditional_scenarios([0.5], 2.0)
    assert result["reservoir_background_flux_kgco2e_m2_year"].nunique() == 1
    zero = result[result["covered_area_response_fraction"] == 0]
    assert zero["fpv_covered_area_increment_tco2e_year"].iloc[0] == 0
