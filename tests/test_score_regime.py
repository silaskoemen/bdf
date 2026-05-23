import pytest

from benchmarks.utils.score_regime import expand_score_regime


@pytest.mark.parametrize(
    ("regime", "expected"),
    [
        ("nle", {"score_method": "nle", "score_correction": None}),
        ("nll_bic", {"score_method": "nll", "score_correction": "bic"}),
        ("nll", {"score_method": "nll", "score_correction": None}),
    ],
)
def test_expand_score_regime(regime, expected):
    params = {"score_regime": regime, "mu_mu": "auto"}

    expanded = expand_score_regime(params)

    assert "score_regime" not in expanded
    assert expanded == {"mu_mu": "auto", **expected}


def test_expand_score_regime_rejects_unknown_regime():
    with pytest.raises(ValueError, match="Unknown score_regime"):
        expand_score_regime({"score_regime": "nll_aic"})
