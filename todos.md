# List of things to do


## BDFRegressor
- [ ] Add None/numeric/NaN checks to `validate_data` before `fit` and `predict`
- [ ] Parallelize fit (and possibly predict) methods with `joblib`, compare for small to large data
- [x] Implement exponential with Lomax PP
- [ ] Implement Poisson-Gamma, think about whether to use `n` and `p` for PP case as params bc only those needed, evaluate and possibly change
- [ ] Implement Gamma-Gamma
- [ ] Implement KDE density (only Gaussian?) with x-fold cv and nll/crps for split finding
- [ ] Implement DistKDL with KDE in terminal leaf, both with `PseudoHKDE` and `PenalizedHKDE` to include (e.g. all-data `h` and strength `m` vs `log(h)` penalty as prior regularizer)
- [ ] Implement Multinomial (dirichlet prior) distribution
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
- [ ] Implement multivariate targets (e.g. MVN or independent univariate distributions, copula?)
- [ ] Implement truly Gaussian prior on bandwidth `h` in KDE; know is function of `sigma`, find normal distribution of stddev
- [ ] Implement Improved Sheather Jones (ISJ) bandwidth selector for KDE (robust to multimodality)
- [ ] Implement feature importance in-sample with `count` and `gain`, out of sample with `gain` (crps/nll) and permutation based feature importance (pfi, shuffle feature see how loss changes)
- [ ] Implement sample weights (either just magnitude of NLL contribution or weighted version of MLE/MoM estimates)
- [ ] Implement Ranking task
- [ ] Implement pre-computed histograms for splits (like in xgboost, lightgbm, catboost, massive speedup)
- [ ] Implement missing value handling/sparsity aware split finding (e.g. learn optimal direction to send missing values at each split, like in xgboost)
