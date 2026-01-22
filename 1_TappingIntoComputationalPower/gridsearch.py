import time
import numpy as np


# Grid search visualization for (w, b) in a linear model xw + b
# Easy to play with interval [a, b] and step size h

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for grid search
# =====================================================================

# Experiment with these values and observe the impact on computations
a = 0   # interval start
b = 10    # interval end
h = 2.5     # step size

# =====================================================================



# Example: y = xw + b, with two data points
X = np.array([2, 5])
Y = np.array([11, 13])


# Track total additions and multiplications
total_additions = 0
total_multiplications = 0
operation_count = 0
best_diff = float('inf')
best_w = None
best_b = None


print(f"\n{'='*70}")
print(f"Grid Search over w and b in [{a}, {b}] with step {h}")
print(f"{'='*70}")
print(f"{'#':>4} | {'w':>7} | {'b':>7} | {'diff':>10} | {'additions':>10} | {'multiplies':>11}")
print(f"{'-'*70}")
for w in np.arange(a, b + h, h):
    for b_ in np.arange(a, b + h, h):
        operation_count += 1
        # Compute predictions: y_pred = X * w + b_
        y_pred = []
        for x in X:
            y = x * w
            total_multiplications += 1  # x * w
            y = y + b_
            total_additions += 1  # + b_
            y_pred.append(y)
        y_pred = np.array(y_pred)
        # Compute simple sum of absolute differences
        diff = np.sum(np.abs(Y - y_pred))
        if diff < best_diff:
            best_diff = diff
            best_w = w
            best_b = b_
        print(f"{operation_count:4d} | {w:7.2f} | {b_:7.2f} | {diff:10.2f} | {total_additions:10d} | {total_multiplications:11d}")
        time.sleep(0.65)



print(f"{'='*70}")
print("Grid search complete.")
print(f"Total additions:        {total_additions}")
print(f"Total multiplications:  {total_multiplications}")
print(f"Total operations:       {total_additions + total_multiplications}")
print(f"Best result: w = {best_w:.2f}, b = {best_b:.2f}, diff = {best_diff:.2f}")
print(f"{'='*70}\n")
