# Rust Negative Gain Fix - Final Summary

## Problem Identified

The Rust implementation was clamping split gains to a maximum of 0.0, preventing the tree structure prior ('defer' and 'bernoulli' modes) from working correctly. This broke the Bayesian comparison between evidence and prior when both were negative.

## Root Cause

Multiple initialization values were set to `0.0` instead of `f64::NEG_INFINITY`, causing:
1. Negative gains to be filtered out during split finding
2. Incorrect semantics for "no split found" cases
3. Inability to distinguish between "impossible to split" vs "found split with negative gain"

## Changes Made to `compiled/rust/src/splitter.rs`

### 1. Best Score Initializations (7 locations)
Changed from `0.0` to `f64::NEG_INFINITY`:
- Line 202: Global `best_results` (non-KDE path)
- Line 231: Local per-feature (suff stats path)
- Line 297: Local per-feature (slow path)
- Line 487: Local per-feature (KDE fast path)
- Line 580: Local per-feature (KDE fallback path)
- Line 896: Local per-feature (KDE FFT path)

**Effect:** Allows negative gains to be tracked and compared properly.

### 2. TopKCandidates Filter (lines 81-83)
Removed:
```rust
if candidate.gain <= 0.0 {
    return false;
}
```

**Effect:** KDE path can now track negative gains in top-k candidates.

### 3. Error Return Values (10 locations)
Changed all `return (None, None, 0.0, ...)` to `return (None, None, f64::NEG_INFINITY, ...)`:
- Lines 401, 660, 725, 735, 787, 799, 803, 865, 991, 1094

**Effect:** Improved semantics - `NEG_INFINITY` clearly means "impossible/forbidden", not "neutral".

## Semantic Improvements

### Before Fix
- `(None, None, 0.0)` = "no split found"
- `(Some(f), Some(t), 0.0)` = "found split with zero gain"
- **Problem:** Same value (0.0) with different meanings!
- **Risk:** If `depth_penalty < 0` (prior favors splitting), the value 0.0 suggests splitting might be acceptable

### After Fix
- `(None, None, -inf)` = "no split possible/forbidden"
- `(Some(f), Some(t), -20.0)` = "found split with negative gain"
- `(Some(f), Some(t), 0.0)` = "found split with zero gain"
- **Benefit:** Clear, unambiguous semantics
- **Safety:** `NEG_INFINITY > depth_penalty` is always False, regardless of prior

## Verification

### Test Results
```python
# Constant y (negative improvement after penalty)
Rust:   feature=0, gain=-20.01
Python: feature=0, gain=-19.51
✓ Rust now returns negative gains!

# Edge case (n=1, impossible to split)
Rust: feature=None, gain=-inf
✓ Clear "impossible" signal

# Semantic check with depth_penalty = -2.0
-inf > -2.0? False  ← Will not split (correct)
-20.0 > -2.0? False ← Will not split (evidence too weak)
-1.5 > -2.0? True   ← Will split (evidence beats prior)
```

### All existing tests pass
```bash
pytest tests/test_split_finding.py -v
# 4/4 PASSED
```

## Impact on Tree Structure Prior

The 'defer' and 'bernoulli' modes now work correctly:

### Defer Prior Math
```
split if: gain > log((1-p_d)/p_d)
```

Where `p_d = alpha * delta^depth`

**At shallow depths with high alpha (e.g., 0.95):**
- `p_d = 0.95` → `depth_penalty ≈ -2.94` (NEGATIVE!)
- Prior strongly favors splitting
- **Before fix:** Negative gains filtered out → no splits possible → prior ignored!
- **After fix:** Negative gains compared to negative penalty → Bayesian decision works!

### Example
```
depth=0, alpha=0.9, delta=0.7
depth_penalty = -2.197 (prior favors splitting)

gain = -1.0 (weak evidence against splitting)
-1.0 > -2.197? TRUE → SPLIT (prior overrules weak negative evidence)

gain = -3.0 (strong evidence against splitting)
-3.0 > -2.197? FALSE → NO SPLIT (evidence overrules prior)
```

## Conclusion

✅ **Rust can now return negative gains** after multiplicity correction
✅ **Tree structure prior works correctly** in 'defer' and 'bernoulli' modes
✅ **Bayesian comparison** between evidence and prior functions properly
✅ **Clear semantics** distinguish "impossible" from "negative gain"
✅ **Robust against edge cases** with proper NEG_INFINITY handling

The defer prior can now properly implement the intended Bayesian tree structure regularization, where even negative evidence can lead to splits if the prior demands it!
