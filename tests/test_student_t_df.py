import numpy as np
import pytest

from bdf.distributions.student_t import FrequentistStudentT, NormalMeanStudentT


def _freq_student_t() -> FrequentistStudentT:
    return FrequentistStudentT(
        {
            "df": None,
            "estimation_method": "mom",
            "score_method": "nll",
            "score_correction": None,
            "use_posterior_predictive": False,
        }
    )


def test_mom_nonpositive_excess_kurtosis_maps_to_near_gaussian_df():
    data = np.array([-1.0, -1.0, 1.0, 1.0])

    freq_params = _freq_student_t().calc_posterior_params(data)
    assert freq_params["df"] == pytest.approx(100.0)

    shrinkage_params = NormalMeanStudentT(
        {
            "mu_mean": 0.0,
            "sigma_mean": 10.0,
            "df": None,
            "score_method": "nll",
            "score_correction": None,
            "use_posterior_predictive": False,
        }
    ).calc_posterior_params(data)
    assert shrinkage_params["df"] == pytest.approx(100.0)


def test_mom_heavy_tail_maps_to_low_df_near_four():
    data = np.r_[np.zeros(100), 100.0]

    params = _freq_student_t().calc_posterior_params(data)

    assert 4.0 < params["df"] < 5.0


def test_rust_and_python_student_t_mom_nll_match_df_branches():
    bdf_rs = pytest.importorskip("bdf._bdf_rs")
    dist = _freq_student_t()

    for data in (np.array([-1.0, -1.0, 1.0, 1.0]), np.r_[np.zeros(100), 100.0]):
        data = data.astype(float)
        rust_nll = bdf_rs.calculate_nll(data, dist.to_rust_spec())
        python_nll = dist.nll(data)

        assert rust_nll == pytest.approx(python_nll, rel=1e-10, abs=1e-10)
