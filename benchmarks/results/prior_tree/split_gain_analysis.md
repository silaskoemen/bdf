# Split Gain Analysis: Negative Values and Tree Structure Prior

## Summary

**Finding:** The Rust implementation cannot return negative gain values, while Python can. This breaks the tree structure prior in 'defer' and 'bernoulli' modes.

## Evidence

### 1. Python Implementation (src/bdf/tree_classes/bdf_node.py)

#### MAP Method (lines 300-331)
```python
# Line 197: Initialize to -inf
best_map_score = -np.inf

# Lines 302-304: Apply penalty (can go negative)
feat_map_score = feat_best_delta
if gamma > 0.0:
    feat_map_score -= gamma * (np.log(k) + np.log(m_j))

# Line 307: Track best even if negative
if feat_map_score > best_map_score and feat_best_threshold is not None:
    best_map_score = feat_map_score

# Line 331: Return the score (CAN BE NEGATIVE)
return (..., float(best_map_score), ...)
```

**Conclusion:** Python CAN return negative `loss_reduction` values after penalty.

### 2. Rust Implementation (compiled/rust/src/splitter.rs)

#### Non-KDE Path (lines 216-283)
```rust
// Line 235: Initialize to 0.0
let mut local_best_loss = 0.0;

// Line 268: Only update if improvement > current best
if improvement > local_best_loss {
    local_best_loss = improvement;
    // ...
}

// Lines 276-278: Apply penalty AFTER tracking (can go negative)
if (gamma > 0.0) && (num_thresholds_tried > 0) && (local_best_threshold.is_some()) {
    local_best_loss -= gamma * ((num_features_tried as f64).ln() + (num_thresholds_tried as f64).ln());
}

// Line 281: Update global best
update_best(&best_results, feature_idx, n_samples, local_best_loss, ...);
```

#### update_best function (lines 1616-1645)
```rust
// Line 202: Global best initialized to 0.0
let best_results = Mutex::new((
    None as Option<usize>,
    None as Option<f64>,
    0.0f64,  // <-- INITIALIZED TO 0.0
    ...
));

// Line 1627: Only update if better than current best (0.0)
if local_best_loss > best.2 {
    *best = (..., local_best_loss, ...);
}
```

**Problem:** Even if `local_best_loss` is negative after penalty, it won't beat the initial 0.0, so it gets rejected.

#### KDE Path (lines 549-562)
```rust
// Lines 554-561: Insert into TopKCandidates
let cand = BestKdeSplit {
    feature_idx,
    threshold,
    gain: local_best_gain,  // Can be negative after penalty
    thresholds_tried: num_thresholds_tried,
};
let mut top_k_results = best_results.lock().unwrap();
top_k_results.try_insert(cand);
```

#### TopKCandidates::try_insert (lines 80-105)
```rust
fn try_insert(&mut self, candidate: BestKdeSplit) -> bool {
    if candidate.gain <= 0.0 {  // <-- EXPLICIT REJECTION
        return false;
    }
    // ...
}
```

**Conclusion:** Rust CANNOT return negative gains:
- Non-KDE: Comparison against 0.0 initial value rejects negative gains
- KDE: Explicit `<= 0.0` check rejects non-positive gains

### 3. Tree Growth Decision (src/bdf/tree_classes/bdf_tree.py)

```python
# Line 147-150: Call split finder (returns loss_reduction)
feature_idx, threshold, loss_reduction, ... = node.find_best_split(
    X, y, self.min_samples_leaf, self.min_child_weight, col_idcs=col_idcs, eta=eta, gamma=self.gamma
)

# Line 153: Calculate tree structure prior penalty
depth_penalty = self._calculate_depth_penalty(node.depth)

# Line 156: Decision to split
if feature_idx is not None and loss_reduction > depth_penalty:
    # Split the node
```

#### Depth Penalty Calculation (lines 81-96)
```python
if self.tree_prior_mode == "linear":
    return self.penalty + self.delta * depth  # Can be positive

elif self.tree_prior_mode == "defer":
    # CART prior: p_d = alpha * delta^depth
    p_d = self.alpha * (self.delta**depth)
    p_d = np.clip(p_d, 1e-10, 1 - 1e-10)
    return np.log((1 - p_d) / p_d)  # POSITIVE when p_d < 0.5

elif self.tree_prior_mode == "bernoulli":
    # Full branching process prior
    p_d = self.alpha * (self.delta**depth)
    p_d1 = self.alpha * (self.delta ** (depth + 1))
    p_d = np.clip(p_d, 1e-10, 1 - 1e-10)
    p_d1 = np.clip(p_d1, 1e-10, 1 - 1e-10)
    return np.log((1 - p_d) / p_d) + 2 * np.log(1 - p_d1)  # POSITIVE
```

## The Problem

There are TWO separate penalties:
1. **Multiplicity penalty** (`gamma`): Applied during split finding for multiple testing correction
2. **Tree structure prior** (`depth_penalty`): Applied after split finding to control tree growth

### Scenario in 'defer' or 'bernoulli' modes:

1. Rust returns `loss_reduction = 0.0` (maximum possible due to clamping)
2. `depth_penalty` is positive (e.g., `log((1-p_d)/p_d)` when `p_d < 0.5`)
3. Condition: `0.0 > positive_value` → **False** → No split

### What SHOULD happen:

1. Split finder could return negative `loss_reduction` (raw improvement minus gamma penalty)
2. If `loss_reduction > depth_penalty`, split anyway (negative but not too negative)
3. This allows tree structure prior to interact with evidence properly

## Impact

The tree structure prior cannot function properly because:
- Rust always returns max(0.0, penalized_gain)
- 'defer' and 'bernoulli' modes have positive depth penalties
- No split will ever satisfy `0.0 > positive_penalty`
- This prevents the "defer" evidence boost from encouraging splitting

## Recommendation

**Fix Rust implementation** to allow negative gains to be returned:

### Option 1: Remove the 0.0 floor in update_best
```rust
// Line 202: Initialize to -inf instead of 0.0
let best_results = Mutex::new((
    None as Option<usize>,
    None as Option<f64>,
    f64::NEG_INFINITY,  // <-- Changed from 0.0
    None as Option<Array1<bool>>,
    None as Option<Array1<bool>>
));
```

### Option 2: Remove the rejection in TopKCandidates (for KDE)
```rust
fn try_insert(&mut self, candidate: BestKdeSplit) -> bool {
    // Remove this check:
    // if candidate.gain <= 0.0 {
    //     return false;
    // }

    if self.candidates.len() < self.k {
        self.candidates.push(candidate);
        return true;
    }
    // ...
}
```

### Option 3: Return 0.0 only when no valid split found
Currently, the code allows negative values to be computed but then clamps them to 0.0. Instead:
- Track whether ANY valid split was found
- If yes, return the best (possibly negative) gain
- If no, return 0.0

## Testing

After fixing, verify that:
1. Rust and Python return similar gains for the same data
2. Negative gains can be returned when penalties are high
3. Tree structure prior works correctly in 'defer' and 'bernoulli' modes
