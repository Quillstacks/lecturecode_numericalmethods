# Loss of Significance Visualization: a + epsilon - b for small epsilon
# This script demonstrates, by brute force, how floating point arithmetic loses precision for very small values.
# As epsilon gets smaller, a + epsilon - b should always be epsilon (if a == b), but at some point the computed value becomes zero due to loss of significance.
# To make this effect even more visible, we also print 1/(a+epsilon-b), which explodes as the computed value approaches zero or is lost due to rounding.


import time
import numpy as np


# Parameters for exploration
# =====================================================================
# Experiment with these values and observe the impact on computations

# Student: Set a and b here (default to 1)
a = 1.0
b = 1.0
# Experiment with these values and observe the impact on computations
eps_start = 1e-1   # interval start (large epsilon)
eps_end = 1e-18    # interval end (very small epsilon)
num_points = 20    # number of epsilon values to try
# =====================================================================

epsilons = np.logspace(np.log10(eps_start), np.log10(eps_end), num_points)

operation_count = 0

print(f"\n{'='*80}")
print(f"LOSS OF SIGNIFICANCE DEMO: 1 + epsilon - 1 for small epsilon")
print(f"{'='*80}")
print("For very small epsilon, 1 + epsilon - 1 should be epsilon, but floating point arithmetic can lose precision!")
print("Watch how the computed value becomes zero as epsilon gets extremely small.")

print(f"LOSS OF SIGNIFICANCE DEMO: a + epsilon - b for small epsilon (a={a}, b={b})")
print(f"{'='*80}")



print(f"{'#':>3} | {'epsilon':>12} | {'1+eps-1':>14} | {'Δ':>10} | {'1/(1+eps-1)':>16}")
print(f"{'-'*65}")
for i, eps in enumerate(epsilons, 1):
    result = (a + eps) - b
    delta = result - eps
    try:
        inv_result = 1.0 / result
    except ZeroDivisionError:
        inv_result = float('inf')
    print(f"{i:3d} | {eps:12.2e} | {result:14.6e} | {delta:10.2e} | {inv_result:16.6e}")
    time.sleep(0.65)

print(f"{'='*80}")
print("Exploration complete.")
print(f"Tested {num_points} values of epsilon from {eps_start:.1e} to {eps_end:.1e}.")
print(f"{'='*80}\n")