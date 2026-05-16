# List of things to do
## Overall updates
- [ ] Write tests for all distributions, expose nll from Rust & check parity
- [ ] Add allowed distributions to BDFRegressor and BDFClassifier docstrings, validate input
- [ ] Allow numpy and pandas inputs
- [ ] Evaluate whether Bayesian vs Frequentist warrants a separate experiment or simple loose Bayes approximates frequentist, e.g. student t performs well, another layer of regularization is enough

## BDFRegressor
- [ ] Could think about `HurdleDistribution` (like KDE) which wraps classification with positive values for zero-inflation - only use if benchmark datasets have this property

## BDFClassifier
- [ ] Change multinomial to `DirichletCategorical`

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
