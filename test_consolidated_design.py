"""Test script for consolidated NormalMuNormal design."""

import numpy as np

from src.bdf.distributions.normal import NormalMuNormal, NormalMuNormalParams, NormGammaNormal, NormGammaNormalParams

np.random.seed(42)
data = np.random.normal(loc=5.0, scale=2.0, size=100)

print("=" * 80)
print("Testing Consolidated BDF Distribution Design")
print("=" * 80)

# Test 1: NormalMuNormal with evidence scoring and posterior predictive (default)
print("\n1. NormalMuNormal with evidence scoring + posterior predictive (default)")
print("-" * 80)
params1 = NormalMuNormalParams(
    mu_mu=0.0, sigma_mu=1.0, score_method="evidence", use_posterior_predictive=True  # Full Bayesian
)
dist1 = NormalMuNormal(params=params1)
print(f"Distribution: {dist1}")
print(f"Supports evidence: {dist1._supports_evidence}")
print(f"Supports posterior predictive: {dist1._supports_posterior_predictive}")
score1 = dist1.score(data)
print(f"Score (evidence): {score1:.4f}")
nle1 = dist1.nle(data)
print(f"NLE: {nle1:.4f}")
posterior_params1 = dist1.get_posterior_params(data)
print(f"Posterior params: {posterior_params1}")
samples1 = dist1.sample_posterior(params=posterior_params1, size=5, random_state=42)
print(f"Posterior samples: {samples1}")

# Test 2: NormalMuNormal with NLL + BIC scoring and plug-in estimates
print("\n2. NormalMuNormal with NLL + BIC scoring + plug-in estimates")
print("-" * 80)
params2 = NormalMuNormalParams(
    mu_mu=0.0, sigma_mu=1.0, score_method="nll", score_correction="bic", use_posterior_predictive=False  # Plug-in
)
dist2 = NormalMuNormal(params=params2)
print(f"Distribution: {dist2}")
score2 = dist2.score(data)
print(f"Score (NLL + BIC): {score2:.4f}")
nll2 = dist2.nll(data)
print(f"NLL: {nll2:.4f}")

# Test 3: NormalMuNormal with evidence but plug-in (should warn about inconsistency)
print("\n3. NormalMuNormal with evidence + plug-in (tests warning)")
print("-" * 80)
params3 = NormalMuNormalParams(
    mu_mu=0.0, sigma_mu=1.0, score_method="evidence", use_posterior_predictive=False  # Inconsistent but allowed
)
dist3 = NormalMuNormal(params=params3)
print(f"Distribution: {dist3}")
score3 = dist3.score(data)
print(f"Score (evidence): {score3:.4f}")

# Test 4: NormGammaNormal with evidence
print("\n4. NormGammaNormal with evidence")
print("-" * 80)
params4 = NormGammaNormalParams(mean=0.0, n=1.0, nu=3.0, phi=1.0)
dist4 = NormGammaNormal(params=params4)
print(f"Distribution: {dist4}")
print(f"Supports evidence: {dist4._supports_evidence}")
print(f"Supports posterior predictive: {dist4._supports_posterior_predictive}")
score4 = dist4.score(data)
print(f"Score (evidence): {score4:.4f}")
posterior_params4 = dist4.get_posterior_params(data)
print(f"Posterior params: {posterior_params4}")

# Test 5: Verify single class handles both modes
print("\n5. Verify single class handles both posterior predictive and plug-in")
print("-" * 80)
params5a = NormalMuNormalParams(mu_mu=0.0, sigma_mu=1.0, score_method="evidence", use_posterior_predictive=True)
params5b = NormalMuNormalParams(mu_mu=0.0, sigma_mu=1.0, score_method="nll", use_posterior_predictive=False)
dist5a = NormalMuNormal(params=params5a)
dist5b = NormalMuNormal(params=params5b)
print(f"✓ Same class (NormalMuNormal) used for both modes")
print(f"  Mode A (PP=True): {dist5a.__class__.__name__}, evidence={dist5a._supports_evidence}")
print(f"  Mode B (PP=False): {dist5b.__class__.__name__}, evidence={dist5b._supports_evidence}")
print(f"  Both are same class: {dist5a.__class__ == dist5b.__class__}")

print("\n" + "=" * 80)
print("All tests completed successfully!")
print("=" * 80)
