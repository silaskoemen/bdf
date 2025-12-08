# List of things to do


## BDFRegressor
- [ ] Parallelize fit (and possibly predict) methods with `joblib`, compare for small to large data
- [ ] Build eval metrics for probabilistic outputs
- [ ] Write tests for all distributions, expose nll from Rust & check parity
- [ ] Experiment with creation of candidate thresholds, currently `closest_observation`
- [ ] Could think about `HurdleDistribution` (like KDE) which wraps classification with positive values for zero-inflation - only use if benchmark datasets have this property
- [ ] Could use `t`, `Weibull`, `NegBin`, `Skew-Normal`, `Gamma`, `Beta` distributions to appendix as commonly used distributions, some freq, some approx Bayesian
- [ ] Suff stats for python splits as well?

## BDFClassifier
- [ ] Implement Gaussian prior on mean, Gaussian likelihood either clip or trunc or in log space
- [ ] Build eval metrics for uncertainty about uncertainty (Brier score (Bias^2 + Var), calibration in certain bands?, Visualize uncertainty in predictions)

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
