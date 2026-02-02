# Mathematical Derivation: Multiplicity Correction Placement

## The Question

Should the multiplicity penalty be:
- **Option A** (current): `(gain - γ×log(k×m)) > threshold`
- **Option B** (alternative): `gain > (threshold + γ×log(k×m))`

While algebraically equivalent, they have different Bayesian interpretations.

## The Bayesian Model Comparison Setup

### What We're Actually Comparing

We're NOT comparing:
- One specific split S vs No split N

We're comparing:
- **H_split**: "There exists a beneficial split among k features × m_j thresholds"
- **H_nosplit**: "No split is better than the current node"

### The Search Process

At each node, we:
1. Consider k features
2. For each feature j, try m_j threshold candidates
3. Total search space: k × m̄ candidates (where m̄ is typical threshold count)
4. Pick the **best** split found

**Key issue:** Selection bias - even random data will produce a "best" split by chance!

## Bayesian Model Comparison

### Model Likelihoods

For **H_nosplit**:
```
P(data | H_nosplit) = L_parent
```

For **H_split** with uniform prior over search space:
```
P(data | H_split) = sum over all (j,c) [ P(feature j) × P(threshold c|j) × L_{j,c} ]
                  = sum_j [ (1/k) × sum_c [ (1/m_j) × L_{j,c} ] ]
```

Where `L_{j,c}` is the likelihood of the split at feature j, threshold c.

### Bayes Factor

The Bayes factor for splitting is:
```
BF = P(data | H_split) / P(data | H_nosplit)
```

Taking logarithm:
```
log BF = log[ sum_j (1/k) sum_c (1/m_j) L_{j,c} ] - log(L_parent)
```

### Two Scenarios

#### Scenario 1: One Split Dominates (MAP approximation)

If one split (j*, c*) is much better than all others:
```
P(data | H_split) ≈ (1/k) × (1/m_j*) × L_{j*,c*}
```

Therefore:
```
log BF ≈ log(L_{j*,c*}) - log(k) - log(m_j*) - log(L_parent)
       = [log(L_{j*,c*}) - log(L_parent)] - log(k) - log(m_j*)
       = gain_raw - log(k) - log(m_j*)
```

**This justifies:** `penalized_gain = gain - log(k) - log(m_j*)`

#### Scenario 2: Multiple Competitive Splits (Model Averaging)

If several splits are competitive, we should integrate:
```
log BF = log[ sum_j (1/k) sum_c (1/m_j) exp(delta_{j,c}) ] - 0
       = log[ (1/k) sum_j (1/m_j) sum_c exp(delta_{j,c}) ]
       = log[ (1/k) sum_j exp(g_j) ]
```

where `g_j = log[(1/m_j) sum_c exp(delta_{j,c})]` is the feature-level integrated score.

This can be further simplified:
```
log BF = log[ sum_j exp(g_j) ] - log(k)
       ≈ max_j(g_j) - log(k)    (if one feature dominates)
       ≈ [max_{j,c}(delta_{j,c}) - log(m_j*)] - log(k)
```

**This ALSO justifies:** `penalized_gain = gain - log(k) - log(m_j*)`

## The Role of γ (Gamma)

The `γ` parameter modulates the strength of the multiplicity penalty:

```
penalized_gain = gain - γ × [log(k) + log(m_j)]
```

### Interpretation of γ Values

| γ | Meaning | Bayesian Prior |
|---|---------|----------------|
| 0 | No penalty | Ignore multiplicity (overfitting) |
| 1 | Standard penalty | Uniform prior over k×m candidates |
| >1 | Conservative | Stronger penalty than uniform (skeptical of search) |
| <1 | Optimistic | Weaker penalty (trust that best split is real) |

### Why γ ≠ 1 Might Make Sense

**γ > 1** (Conservative):
- Accounts for correlation between splits (not all k×m are independent)
- Protects against overfitting when k and m are large
- Similar to Bonferroni correction

**γ < 1** (Optimistic):
- If you believe splits are not uniformly distributed (some features are more likely to be good)
- Structured search (not truly uniform over k×m candidates)

## Two-Stage Decision Process

The split decision involves TWO comparisons:

### Stage 1: Which Split? (Multiplicity Correction)
```
penalized_gain = max_{j,c}[delta_{j,c}] - γ × log(k × m_j)
```
**Question:** "What's the best split, accounting for search over k×m candidates?"

### Stage 2: Split vs No Split? (Tree Structure Prior)
```
penalized_gain > depth_penalty
```
**Question:** "Is the best penalized split worth the complexity of branching?"

### Why Penalties Are Additive on the Gain Side

Mathematically, the full Bayes factor is:
```
log[posterior_odds(split)] = log[prior_odds(split)] + log[BF_multiplicity] + log[BF_data]
```

Expanding:
```
= log(p_d/(1-p_d)) + [gain - log(k×m)] + 0
= [gain - log(k×m)] - log((1-p_d)/p_d)
= [gain - log(k×m)] + [-depth_penalty]
```

For split, we need:
```
log[posterior_odds(split)] > 0
[gain - log(k×m)] > depth_penalty
```

So **both penalties reduce the effective evidence**, not increase the threshold!

## Current Implementation is Correct

The Python implementation:
```python
# Stage 1: Multiplicity correction
feat_map_score = feat_best_delta - gamma * (np.log(k) + np.log(m_j))

# Stage 2: Tree prior comparison
if feat_map_score > depth_penalty:
    split()
```

This is Bayesian correct because:

1. **Multiplicity penalty reduces gain**: We searched k×m candidates, so the "best" is inflated
2. **Tree prior is separate**: Independent decision about tree complexity
3. **Both are on the gain side**: Both reduce evidence for splitting

## Alternative Formulation (Equivalent)

You could equivalently write:
```
gain > (depth_penalty + gamma × log(k × m_j))
```

This moves the multiplicity penalty to the threshold side, but the inequality is identical.

**Conceptual difference:**
- Current way: "Evidence is worth less because we searched"
- Alternative: "We need more evidence because we searched"

Both are mathematically correct, but the current way better separates:
- **Evidence processing** (gain - multiplicity)
- **Decision making** (compare to prior)

## Conclusion

✅ **The current implementation is correct:**
```
penalized_gain = gain - γ × [log(k) + log(m_j)]
if penalized_gain > depth_penalty:
    split()
```

This properly accounts for:
1. Multiple testing over k features and m_j thresholds
2. Tree structure prior for split vs no-split decision
3. Both penalties reduce the effective evidence for splitting

The `γ` parameter allows tuning the strength of multiplicity correction:
- `γ = 1`: Full Bayesian (uniform prior over splits)
- `γ > 1`: Conservative (e.g., Bonferroni-like)
- `γ < 1`: Optimistic (structured/informed search)
