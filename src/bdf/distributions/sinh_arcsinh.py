"""
File for the sinh-arcsinh (SAS/SHASH) distribution.

Due to no closed-form MLE/MoM estimates, BOBYQA is used for parameter estimation
(also in Rust calculations). Permits an additional parameter for the maximum number
of iterations in the optimization process.

Uses the reparameterization of sigma/delta as sigma_delta in estimation,
then changes it back for NLL evaluation.

In the different versions, likelihood will be abbreviated as SHASH. Permits:
- NormalMeanPseudoEpsilonSHASH
- NormalMeanPseudoEpsilonPseudoDeltaSHASH
-
"""
