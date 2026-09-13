from dchyflo.lifecycle import carbon_payback_year, lifecycle_ledger


def test_lifecycle_builds_complete_annual_ledger():
    ledger = lifecycle_ledger(
        annual_avoided_tco2e=100,
        lifecycle_burden_tco2e=150,
        grid_decline_fraction=0.01,
        degradation_fraction=0.005,
        reservoir_increment_tco2e_year=1,
    )
    assert ledger["year"].tolist() == list(range(31))
    assert carbon_payback_year(ledger) == 2
    assert ledger["cumulative_net_carbon_tco2e"].iloc[-1] > 0
