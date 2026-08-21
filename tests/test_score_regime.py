import pytest
from omegaconf import OmegaConf

from benchmarks.conditional_diagnostics_eval import _split_best_params
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


def test_split_best_params_keeps_expanded_score_regime_in_distribution_params():
    model_cfg = OmegaConf.create(
        {
            "tunable_init_kwargs": {"n_trees": {}},
            "tunable_params": {"score_regime": {}, "sigma_mu_auto_scale": {}},
            "fixed_params": {"mu_mu": "auto"},
        }
    )
    best_params = {
        "n_trees": 100,
        "sigma_mu_auto_scale": 1.5,
        "score_method": "nll",
        "score_correction": "bic",
    }

    init_kwargs = _split_best_params(model_cfg, best_params)

    assert init_kwargs == {
        "n_trees": 100,
        "params": {
            "mu_mu": "auto",
            "sigma_mu_auto_scale": 1.5,
            "score_method": "nll",
            "score_correction": "bic",
        },
    }
