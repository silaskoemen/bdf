# List of things to do
## Overall updates
- [ ] Write tests for all distributions, expose nll from Rust & check parity
- [ ] Suff stats for python splits as well?
- [ ] Experiment with creation of candidate thresholds, currently `closest_observation`
- [ ] Add allowed distributions to BDFRegressor and BDFClassifier docstrings, validate input
- [ ] Try ty vs pyright as type checker
- [ ] Allow numpy and pandas inputs
- [ ] Correct readme
- [ ] Parallelize predictions over trees, allow qties from all sampled vs params from trees

## BDFRegressor
- [ ] Could use `t`, `skew-t`, `Weibull`, `NegBin`, `Skew-Normal`, `Gamma`, `Beta` distributions to appendix as commonly used distributions, some freq, some approx Bayesian
- [ ] Could think about `HurdleDistribution` (like KDE) which wraps classification with positive values for zero-inflation - only use if benchmark datasets have this property
- [ ] Implement KDE FFT for faster training and prediction

## BDFClassifier
- [ ] Change multinomial to `DirichletCategorical`
- [ ] Clean up `predict` vs `predict_proba` methods

## Future Improvements
- [ ] Implement multivariate targets (e.g. MVN or independent univariate distributions, copula?)
- [ ] Implement truly Gaussian prior on bandwidth `h` in KDE; know is function of `sigma`, find normal distribution of stddev
- [ ] Implement Improved Sheather Jones (ISJ) bandwidth selector for KDE (robust to multimodality)
- [ ] Implement feature importance in-sample with `count` and `gain`, out of sample with `gain` (crps/nll) and permutation based feature importance (pfi, shuffle feature see how loss changes)
- [ ] Implement sample weights (either just magnitude of NLL contribution or weighted version of MLE/MoM estimates)
- [ ] Implement Ranking task
- [ ] Implement pre-computed histograms for splits (like in xgboost, lightgbm, catboost, massive speedup)
- [ ] Implement missing value handling/sparsity aware split finding (e.g. learn optimal direction to send missing values at each split, like in xgboost)
- [ ] Implement categorical support (in-train target encoding, holdout target encoding, catboost style target encoding)
