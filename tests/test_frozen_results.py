from dchyflo.validation import FROZEN_RESULTS, validate_registered_results


def test_registered_headline_results_remain_consistent():
    assert validate_registered_results() == []
    assert FROZEN_RESULTS["registered_paths"] == 2736
    assert FROZEN_RESULTS["positive_year30_paths"] == 2736
    assert FROZEN_RESULTS["geometry_screened_max_mwac"] == 500
    assert FROZEN_RESULTS["capacity_search_max_mwac"] == 550
