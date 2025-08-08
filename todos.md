# List of things to do


## BDFRegressor Enhancements
- [ ] Reorganize `predict` method in `bdf_regressor.py` to offer separate models for all use cases, separate them logically and offer interface to children.
- [ ] Implement exponential with Lomax PP
- [ ] Implement Poisson-Gamma
- [ ] Implement Gamma-Gamma
- [ ] Implement KDE density (only Gaussian?) with x-fold cv and nll/crps for split finding
- [ ] Implement DistKDL with KDE in terminal leaf
- [ ] Build eval metrics for probabilistic outputs
- [ ] Implement pipeline for fitting and predicting with all comparison models on same data, output final best scores to report in paper.

## BDFClassifier Enhancements
- [ ] Implement basic Beta prior, Binomial likelihood
- [ ] Implement Gaussian prior on mean, Gaussian likelihood either clip or trunc or in log space
- [ ] Build eval metrics for uncertainty about uncertainty (Brier score (Bias^2 + Var), calibration in certain bands?, Visualize uncertainty in predictions)
