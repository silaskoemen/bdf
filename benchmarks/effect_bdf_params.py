"""
Benchmarking script to compare how the different parameters of the BDF

Parameters include `reg_lambda` (prior splitting probability), `reg_gamma`
(multiplicity correction), `reg_nu` (depth penalty), `score_correction` (nle, nll+bic, nll alone),
as well as choosing parametric (KDE) vs non-parametric (normal) distributions, where the normal
choice additionally has the grand prior.

Similarly to other benchmarks, uses Friedman #1-3 and `make_regression` DGPs.
"""
