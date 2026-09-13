import numpy as np

from dchyflo.mcv import central_difference_mcv


def test_mcv_central_difference_has_expected_sign_and_range():
    perturbation = 5.0
    delta_m3 = perturbation * 1e6
    value = central_difference_mcv(0.05 * delta_m3, -0.05 * delta_m3, perturbation)
    assert np.isclose(value, 0.05)
    assert 0 <= value <= 0.0655
