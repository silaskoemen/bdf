# Mathematical Analysis: Defer Mode and Negative Gains

## The Defer Prior Structure

In 'defer' mode, the split decision compares evidence against a tree structure prior.

### Split Decision Rule (bdf_tree.py:156)
```python
if loss_reduction > depth_penalty:
    split()
```

Where `depth_penalty = log((1-p_d)/p_d)` and `p_d = alpha * delta^depth`

## Bayesian Interpretation

### Posterior Odds of Splitting

Given:
- **Model S** (split): Creates two child nodes
- **Model N** (no split): Remains as leaf
- **Prior**: `P(S) = p_d`, `P(N) = 1 - p_d`
- **Evidence**: `gain = log[P(data|S)] - log[P(data|N)]` (log Bayes factor)

The posterior odds of splitting:
```
posterior_odds(S/N) = [p_d / (1-p_d)] × exp(gain)
```

Taking logarithm:
```
log[posterior_odds] = log(p_d / (1-p_d)) + gain
```

For splitting, we want `posterior_odds > 1`:
```
log(p_d / (1-p_d)) + gain > 0
gain > -log(p_d / (1-p_d))
gain > log((1-p_d) / p_d)    ← This is depth_penalty!
```

**This confirms the implementation is Bayesian correct!**

## When Can Negative Gain Still Lead to Splitting?

### Critical Insight: Sign of depth_penalty

The sign of `depth_penalty = log((1-p_d)/p_d)` depends on `p_d`:

| Condition | depth_penalty | Interpretation |
|-----------|---------------|----------------|
| `p_d > 0.5` | **NEGATIVE** | Prior favors splitting |
| `p_d = 0.5` | **ZERO** | Prior is neutral |
| `p_d < 0.5` | **POSITIVE** | Prior discourages splitting |

### Example: Strong Prior for Splitting

**Scenario:**
- `p_d = 0.9` (strong prior belief that we should split)
- `depth_penalty = log(0.1/0.9) = log(1/9) ≈ -2.197`
- `gain = -1.0` (evidence slightly against splitting)

**Decision:**
```
-1.0 > -2.197  →  TRUE  →  SPLIT!
```

Even though the evidence is negative (data suggests no split), the prior is so strong that we split anyway.

### Example: Weak Prior Against Splitting

**Scenario:**
- `p_d = 0.1` (weak prior for splitting)
- `depth_penalty = log(0.9/0.1) = log(9) ≈ +2.197`
- `gain = +1.5` (moderate evidence for splitting)

**Decision:**
```
+1.5 > +2.197  →  FALSE  →  NO SPLIT
```

Evidence is positive but not strong enough to overcome the prior against splitting.

## CART Prior Behavior with Depth

Using `p_d = alpha * delta^depth`:

### Typical Parameters
- `alpha = 0.95` (high initial splitting probability)
- `delta = 0.5` (decay rate)

| Depth | p_d | depth_penalty | Prior Favors |
|-------|-----|---------------|--------------|
| 0 | 0.950 | **-2.944** | **Split (strong)** |
| 1 | 0.475 | **+0.100** | No split (weak) |
| 2 | 0.238 | **+1.163** | No split (moderate) |
| 3 | 0.119 | **+2.013** | No split (strong) |
| 4 | 0.059 | **+2.813** | No split (very strong) |

### Critical Observation

**At depth 0 with alpha=0.95, delta=0.5:**
- `depth_penalty ≈ -2.944` (NEGATIVE!)
- A split with `gain = -2.0` would still be accepted:
  - `-2.0 > -2.944` → TRUE → SPLIT

**This is where Rust's 0.0 clamping breaks the model:**
- Rust returns: `gain = 0.0` (clamped)
- Decision: `0.0 > -2.944` → TRUE → Split
- **But this is wrong!** The actual gain might be -3.5, which should NOT split:
  - `-3.5 > -2.944` → FALSE → NO SPLIT

## The Problem Illustrated

### Scenario: Shallow depth with strong prior

Configuration:
- depth = 0
- alpha = 0.9, delta = 0.7
- p_d = 0.9
- depth_penalty = log(0.1/0.9) ≈ -2.197

Suppose the true evidence (after multiplicity penalty) is:
- Raw improvement: 1.5
- Multiplicity penalty (gamma × log(k×m)): -2.8
- **True gain: -1.3**

### What SHOULD happen (Python with negative gains enabled):
```
gain = -1.3
-1.3 > -2.197  →  TRUE  →  SPLIT
```
**Interpretation:** Evidence is slightly negative, but prior strongly favors splitting → split!

### What ACTUALLY happens (Rust clamping to 0.0):
```
gain = 0.0  (clamped from -1.3)
0.0 > -2.197  →  TRUE  →  SPLIT
```
**Problem:** Same decision, but for the wrong reason!

### When clamping causes wrong decision:

Suppose true gain = -2.5 (stronger negative evidence):
```
# Correct behavior (should NOT split):
-2.5 > -2.197  →  FALSE  →  NO SPLIT

# Rust behavior (WRONG):
0.0 > -2.197  →  TRUE  →  SPLIT
```

## Conclusion

**YES, negative depth_penalty is not only possible but ESSENTIAL for the defer prior to work correctly!**

### When negative depth_penalty occurs:
1. **Shallow depths** with high alpha (e.g., alpha > 0.5)
2. **CART prior** at root: `p_0 = alpha`, so if alpha > 0.5, depth_penalty is negative

### Why Rust must support negative gains:
1. At shallow depths, prior can strongly favor splitting (p_d > 0.5)
2. This creates negative depth_penalty
3. Negative evidence (gain < 0) can still satisfy `gain > depth_penalty` when both are negative
4. Clamping to 0.0 makes ALL negative evidence look the same, breaking the Bayesian comparison

### The Fix is Essential

Without allowing negative gains, the defer prior cannot:
- Distinguish between "slightly negative evidence" vs "strongly negative evidence"
- Make correct Bayesian decisions when prior favors splitting (p_d > 0.5)
- Implement the intended tree structure prior at all

The 0.0 clamping effectively replaces:
```
gain > depth_penalty
```
with:
```
max(gain, 0) > depth_penalty
```

This is fundamentally different and breaks the Bayesian interpretation!
