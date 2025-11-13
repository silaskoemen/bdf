"""Test script for new BDF distribution design."""

import numpy as np

from src.bdf.distributions.normal import NormalMuNormal, NormalMuNormalPP, NormGammaNormal

# Set random seed for reproducibility
np.random.seed(42)

# Generate some test data
data = np.random.normal(loc=5.0, scale=2.0, size=100)

print("=" * 80)
print("Testing New BDF Distribution Design")
print("=" * 80)

# Test 1: NormalMuNormalPP with evidence scoring (default)
print("\n1. NormalMuNormalPP with evidence scoring (default)")
print("-" * 80)
params1 = {
    "mean": 0.0,
    "std": 1.0,
    # score_method defaults to "evidence"
    # use_posterior_predictive defaults to True
}
dist1 = NormalMuNormalPP(params=params1)
print(f"Distribution: {dist1}")
print(f"Supports evidence: {dist1._supports_evidence}")
print(f"Supports posterior predictive: {dist1._supports_posterior_predictive}")

# Calculate score (should use evidence)
score1 = dist1.score(data)
print(f"Score (evidence): {score1:.4f}")

# Calculate NLE directly
nle1 = dist1.nle(data)
print(f"NLE: {nle1:.4f}")

# Get posterior params
post_params1 = dist1.calc_posterior_params(data, return_dict=True)
print(f"Posterior params: {post_params1}")

# Sample from posterior
samples1 = dist1.sample_posterior(data=data, size=5)
print(f"Posterior samples: {samples1}")

# Test 2: NormalMuNormalPP with NLL + BIC scoring
print("\n2. NormalMuNormalPP with NLL + BIC scoring")
print("-" * 80)
params2 = {
    "mean": 0.0,
    "std": 1.0,
    "score_method": "nll",
    "score_correction": "bic",
    "use_posterior_predictive": False,  # Use plug-in
}
dist2 = NormalMuNormalPP(params=params2)
print(f"Distribution: {dist2}")

# Calculate score (should use NLL + BIC)
score2 = dist2.score(data)
print(f"Score (NLL + BIC): {score2:.4f}")

# Calculate NLL directly
nll2 = dist2.nll(data)
print(f"NLL: {nll2:.4f}")

# Test 3: NormalMuNormal (plug-in only)
print("\n3. NormalMuNormal (plug-in only)")
print("-" * 80)
params3 = {
    "mean": 0.0,
    "std": 1.0,
    "score_method": "nll",
    "score_correction": "aic",
}
dist3 = NormalMuNormal(params=params3)
print(f"Distribution: {dist3}")
print(f"Supports evidence: {dist3._supports_evidence}")

# Calculate score (should use NLL + AIC)
score3 = dist3.score(data)
print(f"Score (NLL + AIC): {score3:.4f}")

# Test 4: NormGammaNormal with evidence
print("\n4. NormGammaNormal with evidence")
print("-" * 80)
params4 = {
    "mean": 0.0,
    "n": 1.0,
    "nu": 3.0,
    "phi": 1.0,
    "score_method": "evidence",
}
dist4 = NormGammaNormal(params=params4)
print(f"Distribution: {dist4}")
print(f"Supports evidence: {dist4._supports_evidence}")
print(f"Supports posterior predictive: {dist4._supports_posterior_predictive}")

# Calculate score (should use evidence)
score4 = dist4.score(data)
print(f"Score (evidence): {score4:.4f}")

# Get posterior params
post_params4 = dist4.calc_posterior_params(data, return_dict=True)
print(f"Posterior params: {post_params4}")

# Test 5: Error handling - try evidence on plug-in model
print("\n5. Error handling - try evidence on NormalMuNormal")
print("-" * 80)
try:
    params5 = {
        "mean": 0.0,
        "std": 1.0,
        "score_method": "evidence",  # This should raise an error
    }
    dist5 = NormalMuNormal(params=params5)
    print("ERROR: Should have raised ValueError!")
except ValueError as e:
    print(f"Correctly caught error: {e}")

print("\n" + "=" * 80)
print("All tests completed successfully!")
print("=" * 80)
