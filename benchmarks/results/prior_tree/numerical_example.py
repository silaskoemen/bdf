"""
Numerical demonstration of how 0.0 clamping breaks the defer prior.
"""

import numpy as np
import pandas as pd


def calculate_depth_penalty(alpha: float, delta: float, depth: int) -> float:
    """Calculate depth penalty in defer mode."""
    p_d = alpha * (delta**depth)
    p_d = np.clip(p_d, 1e-10, 1 - 1e-10)
    return np.log((1 - p_d) / p_d)


def should_split(gain: float, depth_penalty: float) -> bool:
    """Decision rule: split if gain > depth_penalty."""
    return gain > depth_penalty


# Configuration
alpha = 0.9
delta = 0.7
depths = range(5)

print("=" * 80)
print("DEFER PRIOR: Effect of 0.0 Clamping on Split Decisions")
print("=" * 80)
print(f"\nConfiguration: alpha={alpha}, delta={delta}")
print()

# Calculate depth penalties
results = []
for depth in depths:
    p_d = alpha * (delta**depth)
    penalty = calculate_depth_penalty(alpha, delta, depth)

    print(f"\nDepth {depth}:")
    print(f"  p_d = {p_d:.4f}")
    print(f"  depth_penalty = {penalty:.4f}")
    print(f"  Prior favors: {'SPLIT' if p_d > 0.5 else 'NO SPLIT'}")

    # Test various gain values
    test_gains = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

    print("\n  Gain | Correct | Rust(0.0) | Decision Changes?")
    print("  " + "-" * 55)

    for true_gain in test_gains:
        # Correct decision with negative gains allowed
        correct_decision = should_split(true_gain, penalty)

        # Rust decision (clamped to 0.0)
        rust_gain = max(true_gain, 0.0)
        rust_decision = should_split(rust_gain, penalty)

        changed = "YES! ←" if correct_decision != rust_decision else "No"

        print(f"  {true_gain:5.1f} | {str(correct_decision):7s} | {str(rust_decision):9s} | {changed}")

        if correct_decision != rust_decision:
            results.append(
                {
                    "depth": depth,
                    "p_d": p_d,
                    "depth_penalty": penalty,
                    "true_gain": true_gain,
                    "correct": correct_decision,
                    "rust": rust_decision,
                }
            )

print("\n" + "=" * 80)
print("SUMMARY: Cases where clamping causes WRONG decision")
print("=" * 80)

if results:
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print(f"\nTotal wrong decisions: {len(results)}")
else:
    print("No wrong decisions found (but this means prior isn't testing negative region!)")

print("\n" + "=" * 80)
print("KEY INSIGHT")
print("=" * 80)
print(
    """
When depth_penalty is NEGATIVE (prior favors splitting):
- Negative gains can still lead to correct splits
- Clamping to 0.0 makes all negative evidence indistinguishable
- This breaks the Bayesian comparison between evidence and prior

The defer prior REQUIRES negative gains to function correctly at shallow depths!
"""
)

# Additional analysis: At what depth does penalty become positive?
print("\n" + "=" * 80)
print("TRANSITION DEPTH ANALYSIS")
print("=" * 80)

for alpha_test in [0.5, 0.7, 0.8, 0.9, 0.95]:
    for delta_test in [0.3, 0.5, 0.7, 0.9]:
        transition_depth = None
        for d in range(20):
            p_d = alpha_test * (delta_test**d)
            if p_d < 0.5:
                transition_depth = d
                break

        if transition_depth is not None:
            print(
                f"alpha={alpha_test:.2f}, delta={delta_test:.2f}: "
                f"penalty becomes positive at depth {transition_depth}"
            )
        else:
            print(f"alpha={alpha_test:.2f}, delta={delta_test:.2f}: " f"penalty ALWAYS negative (p_d always > 0.5)")
