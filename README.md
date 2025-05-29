# Bayesian Distributional Forest

A Python package for Bayesian Distributional Forest (BDF), a probabilistic extension of random forests that provides full predictive distributions rather than just point estimates.

## Overview

Bayesian Distributional Forest is a machine learning model that combines the flexibility of decision trees with Bayesian inference to generate predictive distributions for each data point. Instead of predicting a single value, BDF predicts an entire probability distribution, giving you valuable uncertainty information for your predictions.

Key features:
- Full posterior predictive distributions for each prediction
- Uncertainty quantification out of the box
- Customizable priors for different distribution families
- Parallelized training and prediction

## Installation

```bash
pip install bayesian-distributional-forest
```

## Quick Start

```python
import numpy as np
from bdf import BayesianDistributionalForest

# Generate some example data
X = np.random.randn(1000, 5)
y = np.random.randn(1000)

# Initialize the model
bdf_model = BayesianDistributionalForest(
    n_estimators=100,
    prior_type='gaussian',
    max_depth=10
)

# Train the model
bdf_model.fit(X, y)

# Get predictive distributions
predictive_dists = bdf_model.predict_distribution(X[:5])

# Get mean predictions
y_pred = bdf_model.predict(X[:5])

# Get quantiles of the predictive distribution
y_lower, y_upper = bdf_model.predict_interval(X[:5], interval_width=0.9)
```

## Documentation

For detailed documentation, visit [docs.bayesiandistributionalforest.io](https://docs.bayesiandistributionalforest.io).

## Examples

### Regression with Uncertainty

```python
from bdf import BayesianDistributionalForest
import matplotlib.pyplot as plt
import numpy as np

# Generate data with heteroscedastic noise
X = np.linspace(-10, 10, 1000).reshape(-1, 1)
y = X.ravel() ** 2 / 20 + np.random.normal(scale=np.abs(X.ravel()) / 2)

# Split data
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# Train model
model = BayesianDistributionalForest(n_estimators=100, prior_type='gaussian')
model.fit(X_train, y_train)

# Predict with uncertainty
mean_preds = model.predict(X_test)
lower, upper = model.predict_interval(X_test, interval_width=0.95)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.3, label='Training data')
plt.scatter(X_test, y_test, alpha=0.3, label='Test data')
plt.plot(X_test, mean_preds, 'r-', label='Predicted mean')
plt.fill_between(X_test.ravel(), lower, upper, alpha=0.2, color='r', label='95% predictive interval')
plt.legend()
plt.show()
```

## How It Works

Bayesian Distributional Forest builds on the random forest algorithm with key differences:
1. Each leaf node models a distribution rather than a point estimate
2. Bayesian priors are used to regularize the leaf distributions
3. The final prediction combines distributions from multiple trees to form a mixture distribution

## Available Priors

- Gaussian
- Student's t
- Laplace
- Custom (user-defined)

## Contributing

Contributions are welcome! Please check out our [contributing guidelines](CONTRIBUTING.md).

## Citation

If you use this package in your research, please cite:

```
@software{bayesian_distributional_forest,
  title = {Bayesian Distributional Forest},
  author = {Author, A.},
  url = {https://github.com/username/BayesianDistributionalForest},
  year = {2023}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
