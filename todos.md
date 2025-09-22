# List of things to do


## BDFRegressor
- [x] Reorganize `predict` method in `bdf_regressor.py` to offer separate models for all use cases, separate them logically and offer interface to children.
- [x] Reorganize `predict_posterior_params` methods in all distributions to use prefix `posterior_` and the actual distributional names (i.e. `posterior_mu`, `posterior_sigma` for Normal distribution)
- [x] Redo normal distribution with `mu_mu` and `sigma_mu` as priors, indicate uncertainty update of sigma with suffix `PP` (Posterior Predictive) and create `NormalBase` class.
- [ ] Implement `_validate_data` for all distributions to check for valid data (None, invalid bounds etc.) at start of `fit` method (and possibly `predict` method)
- [ ] Parallelize fit (and possibly predict) methods with `joblib`, compare for small to large data
- [ ] Implement exponential with Lomax PP
- [ ] Implement Poisson-Gamma
- [ ] Implement Gamma-Gamma
- [ ] Implement KDE density (only Gaussian?) with x-fold cv and nll/crps for split finding
- [ ] Implement DistKDL with KDE in terminal leaf, both with `PseudoHKDE` and `PenalizedHKDE` to include (e.g. all-data `h` and strength `m` vs `log(h)` penalty as prior regularizer)
- [ ] Implement Multinomial distribution
- [ ] Implement relevant other distributions for certain use cases (e.g. Weibull, Gompertz, Uniform, Negative Binomial, ...) and use CLT mean/pseudo prior if no conjugate prior is available
- [ ] Build eval metrics for probabilistic outputs
- [ ] Write tests for all distributions, expose nll from Rust & check parity
- [ ] Experiment with creation of candidate thresholds, currently `closest_observation`
- [ ] Implement pipeline for fitting and predicting with all comparison models on same data, output final best scores to report in paper.

## BDFClassifier
- [ ] Implement basic Beta prior, Binomial likelihood
- [ ] Implement Gaussian prior on mean, Gaussian likelihood either clip or trunc or in log space
- [ ] Build eval metrics for uncertainty about uncertainty (Brier score (Bias^2 + Var), calibration in certain bands?, Visualize uncertainty in predictions)

## Future Improvements
- [ ] Implement feature importance in-sample with `count` and `gain`, out of sample with `gain` (crps/nll) and permutation based feature importance (pfi, shuffle feature see how loss changes)
