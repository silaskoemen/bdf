"""
Numerical example: Two-stage Bayesian decision process for splitting.
"""

import numpy as np

print("=" * 80)
print("TWO-STAGE BAYESIAN SPLIT DECISION")
print("=" * 80)

# Scenario setup
print("\nScenario: Node at depth 0, considering 10 features with 40 thresholds each")
print("-" * 80)

k = 10  # number of features
m_j = 40  # thresholds per feature
gamma = 1.0  # standard Bayesian multiplicity penalty

# Tree prior (defer mode)
alpha = 0.9
delta = 0.7
depth = 0
p_d = alpha * (delta**depth)
depth_penalty = np.log((1 - p_d) / p_d)

print(f"\nSearch space: k={k} features, m_j={m_j} thresholds")
print(f"Tree prior: alpha={alpha}, delta={delta}, depth={depth}")
print(f"  → p_d = {p_d:.3f}")
print(f"  → depth_penalty = {depth_penalty:.3f}")

# Example splits found during search
splits = [
    {"feature": 0, "threshold": 15, "raw_gain": 3.5, "description": "Strong split"},
    {"feature": 3, "threshold": 22, "raw_gain": 2.0, "description": "Moderate split"},
    {"feature": 7, "threshold": 8, "raw_gain": 0.5, "description": "Weak split"},
    {"feature": 9, "threshold": 35, "raw_gain": -0.5, "description": "Slight negative"},
    {"feature": 2, "threshold": 12, "raw_gain": -2.0, "description": "Strong negative"},
]

print("\n" + "=" * 80)
print("STAGE 1: Multiplicity Correction (Which split?)")
print("=" * 80)
print("\nMultiplicity penalty = γ × [log(k) + log(m_j)]")
print(f"                     = {gamma} × [log({k}) + log({m_j})]")
print(f"                     = {gamma} × [{np.log(k):.3f} + {np.log(m_j):.3f}]")
multiplicity_penalty = gamma * (np.log(k) + np.log(m_j))
print(f"                     = {multiplicity_penalty:.3f}")

print(f"\nProcessing {len(splits)} candidate splits:")
print("-" * 80)
print("Feature | Thresh | Raw Gain | Multiplicity | Penalized Gain | Description")
print("-" * 80)

best_penalized_gain = -np.inf
best_split = None

for split in splits:
    raw = split["raw_gain"]
    penalized = raw - multiplicity_penalty

    if penalized > best_penalized_gain:
        best_penalized_gain = penalized
        best_split = split

    marker = " ← BEST" if split == best_split or penalized == best_penalized_gain else ""

    print(
        f"   {split['feature']:2d}   |   {split['threshold']:2d}   | "
        f"{raw:8.3f} | {multiplicity_penalty:12.3f} | "
        f"{penalized:14.3f} | {split['description']}{marker}"
    )

print("\n" + "=" * 80)
print("STAGE 2: Tree Structure Prior (Split vs No Split?)")
print("=" * 80)

print(f"\nBest penalized gain: {best_penalized_gain:.3f}")
print(f"Depth penalty:       {depth_penalty:.3f}")
print("\nDecision rule: penalized_gain > depth_penalty")
print(f"              {best_penalized_gain:.3f} > {depth_penalty:.3f}")

should_split = best_penalized_gain > depth_penalty
print(f"              {should_split}")
print(f"\n→ Decision: {'SPLIT' if should_split else 'DO NOT SPLIT'}")

if should_split:
    print(f"\nChosen split: Feature {best_split['feature']}, " f"Threshold {best_split['threshold']}")
    print(f"  Raw gain: {best_split['raw_gain']:.3f}")
    print(f"  After multiplicity correction: {best_penalized_gain:.3f}")
    print(f"  Exceeds tree prior penalty: {depth_penalty:.3f}")

print("\n" + "=" * 80)
print("BAYESIAN INTERPRETATION")
print("=" * 80)

print(
    f"""
Stage 1: Account for Multiple Testing
--------------------------------------
We searched over k×m = {k}×{m_j} = {k*m_j} candidates.
Even random data could produce a "good" split by chance.

The multiplicity penalty {multiplicity_penalty:.3f} deflates the evidence:
  - Raw gain {best_split['raw_gain']:.3f} is what we observed
  - But we tried {k*m_j} times, so chance explains {multiplicity_penalty:.3f}
  - True evidence: {best_penalized_gain:.3f}

Stage 2: Compare Evidence to Tree Prior
----------------------------------------
The tree structure prior has p_d={p_d:.3f} for splitting at depth {depth}.
This means depth_penalty = log((1-p_d)/p_d) = {depth_penalty:.3f}

Since depth_penalty is NEGATIVE, the prior FAVORS splitting.
Decision: Is the (penalized) evidence enough to split?

  Posterior odds(split) ∝ exp(penalized_gain) / exp(-depth_penalty)
                        ∝ exp({best_penalized_gain:.3f}) / exp({-depth_penalty:.3f})
                        ∝ exp({best_penalized_gain - (-depth_penalty):.3f})

For splitting, we need posterior odds > 1:
  {best_penalized_gain:.3f} > {depth_penalty:.3f}  →  {should_split}
"""
)

print("=" * 80)
print("WHY BOTH PENALTIES REDUCE THE GAIN")
print("=" * 80)

print(
    """
Full Bayesian posterior odds:

  log[P(split|data)] = log[prior] + log[BF_search] + log[BF_evidence]
                     = log(p_d/(1-p_d)) + [-log(k×m)] + [gain_raw]
                     = [gain_raw - log(k×m)] + log(p_d/(1-p_d))
                     = [gain_raw - log(k×m)] - log((1-p_d)/p_d)
                     = penalized_gain - depth_penalty

For splitting, need: penalized_gain > depth_penalty

Both corrections (multiplicity and tree prior) enter on the GAIN side
because they both reduce the effective evidence for splitting:
  - Multiplicity: "gain is inflated by search"
  - Tree prior: "need to overcome prior against branching"
"""
)

# Show what happens if we move penalties to threshold side
print("\n" + "=" * 80)
print("ALTERNATIVE FORMULATION (Algebraically Equivalent)")
print("=" * 80)

effective_threshold = depth_penalty + multiplicity_penalty
print("\nInstead of: penalized_gain > depth_penalty")
print("Could write: raw_gain > (depth_penalty + multiplicity_penalty)")
print(f"            {best_split['raw_gain']:.3f} > {effective_threshold:.3f}")
print(f"            {best_split['raw_gain'] > effective_threshold}")

print(
    f"""
Conceptual difference:
  - Current:     "Evidence ({best_penalized_gain:.3f}) vs Prior ({depth_penalty:.3f})"
  - Alternative: "Evidence ({best_split['raw_gain']:.3f}) vs Threshold ({effective_threshold:.3f})"

Both are correct, but current separates:
  1. Evidence processing: raw → penalized
  2. Decision making: penalized vs prior
"""
)

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print(
    """
✅ Current implementation is Bayesian correct:
   1. Apply multiplicity penalty to gain (account for search)
   2. Compare penalized gain to depth penalty (account for tree prior)
   3. Both penalties reduce effective evidence for splitting

The two-stage process cleanly separates:
   - Feature/threshold selection (Stage 1)
   - Split/no-split decision (Stage 2)
"""
)
