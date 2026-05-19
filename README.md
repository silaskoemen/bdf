# Bayesian Distributional Forest

A Python package for Bayesian Distributional Forest (BDF), a probabilistic extension of random forests that returns full predictive distributions rather than point estimates.

## Overview

BDF combines forest-style tree induction with Bayesian inference at the leaves. Each leaf carries a posterior over a user-chosen distribution family, splits are scored under that family (via the marginal likelihood or a plug-in surrogate), and predictions are obtained by aggregating per-tree posterior predictives. The result is a single tree-ensemble template that exposes uncertainty natively across regression, classification, and richer leaf families.

Key features:
- Full posterior predictive distributions for each prediction.
- Pluggable leaf families with Bayesian priors (Normal, Student-t, Beta-Bernoulli, Gamma-Poisson, KDE, and more).
- Sklearn-style `fit` / `predict` API.
- Parallel tree fitting with a Rust split-finding backend (PyO3).

If the Rust extension is unavailable at runtime, split search emits a warning and falls back to the slower Python implementation.

Unsupported in the current release:
- Native categorical features (one-hot encode upstream).
- Native missing-value handling.
- Sample weights.
- Multiclass classification (binary only; Dirichlet–multinomial leaves are implemented in the distribution library but out of scope for the published benchmarks).

## Installation

```bash
pip install bayesian-distributional-forest
```

The benchmark suite optionally compares BDF against [bartpy](https://github.com/JakeColtman/bartpy), which depends on a deprecated `sklearn` package. To install it set:

```bash
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
```

## Quick Start

```python
import numpy as np
from bdf.tree_classes.bdf_regressor import BDFRegressor

rng = np.random.default_rng(0)
X = rng.standard_normal((1000, 5))
y = X[:, 0] + 0.5 * rng.standard_normal(1000)

model = BDFRegressor(
    dist="NormalMuNormal",
    params={"mu_mu": "auto", "sigma_mu": "auto"},
    n_trees=50,
    max_depth=50,
    min_samples_leaf=10,
    gamma=0.1,
)
model.fit(X, y)

mean = model.predict(X)                            # point predictions
variance = model.predict(X, method="var")          # predictive variance
samples = model.predict(X, method="samples",       # posterior predictive draws
                        method_params={"n_samples": 200})
lower, upper = model.predict_quantiles(X, q=[0.05, 0.95]).T   # 90% interval
```

## Regression with Heteroscedastic Uncertainty

```python
import numpy as np
import matplotlib.pyplot as plt
from bdf.tree_classes.bdf_regressor import BDFRegressor

rng = np.random.default_rng(0)
X = np.linspace(-10, 10, 1000).reshape(-1, 1)
y = X.ravel() ** 2 / 20 + rng.normal(scale=np.abs(X.ravel()) / 2 + 0.1)

X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

model = BDFRegressor(dist="NormalMuNormal", n_trees=100)
model.fit(X_train, y_train)

mean = model.predict(X_test)
lower, upper = model.predict_quantiles(X_test, q=[0.025, 0.975]).T

plt.scatter(X_train, y_train, alpha=0.3, label="train")
plt.scatter(X_test, y_test, alpha=0.3, label="test")
plt.plot(X_test, mean, "r-", label="predicted mean")
plt.fill_between(X_test.ravel(), lower, upper, alpha=0.2, color="r", label="95% interval")
plt.legend(); plt.show()
```

## Classification

```python
from bdf.tree_classes.bdf_regressor import BDFClassifier

clf = BDFClassifier(dist="BetaMVBernoulli", n_trees=50)
clf.fit(X_train, y_train)
p_positive = clf.predict_proba(X_test)[:, 1]   # P(y = 1 | x)
predictions = clf.predict(X_test)
```

## Available Distributions

Distribution names are passed through the `dist=` argument. The full registry is in `src/bdf/distributions/`; selected families used in the paper:

- `NormalMuNormal`, `NormalMuInvGammaSigmaNormal` (Normal leaves with conjugate priors)
- `FreqStudentT`, `NormalMeanStudentT` (heavy-tailed)
- `GammaMVLambdaPoisson`, `ExponentialGammaAB` (count / positive)
- `KDE`, `BayesianKDE` (nonparametric)
- `BetaMVBernoulli`, `BetaABBernoulli` (binary classification)

## Reproducing the Paper

Empirical results and the paper build are reproducible end-to-end through pixi tasks. See [`benchmarks/REPRODUCIBILITY.md`](benchmarks/REPRODUCIBILITY.md) for the full pipeline (data fetch, benchmark runs, conditional diagnostics, plot/table aggregation, and paper build) and [`benchmarks/README.md`](benchmarks/README.md) for a tour of all studies.

## Documentation

Local docs build with `pixi run docs-build`; output lands in `docs/build/html/`.

## Contributing

Contributions are welcome. Please open an issue describing the change before submitting a PR.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
