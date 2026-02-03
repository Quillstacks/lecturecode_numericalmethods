import time
import numpy as np

# Numerical Integration: Quadrature Rules
# Trapezoid, Simpson, and Newton-Cotes methods

np.random.seed(42)

# =====================================================================
# Configuration Parameters - Experiment with these!
# =====================================================================

# Integral to compute: integral of f(x) from a to b
def f(x):
    """Test function: sin(x)"""
    return np.sin(x)

# Integration bounds
a = 0
b = np.pi  # Exact integral of sin(x) from 0 to pi = 2

# Exact solution (for error calculation)
exact_integral = 2.0

# Number of subintervals for composite rules
n_intervals = [2, 4, 8, 16, 32, 64]

# =====================================================================


def riemann_left(f, a, b, n):
    """Left Riemann sum."""
    h = (b - a) / n
    x = np.linspace(a, b - h, n)
    return h * np.sum(f(x))


def riemann_right(f, a, b, n):
    """Right Riemann sum."""
    h = (b - a) / n
    x = np.linspace(a + h, b, n)
    return h * np.sum(f(x))


def riemann_midpoint(f, a, b, n):
    """Midpoint Riemann sum."""
    h = (b - a) / n
    x = np.linspace(a + h/2, b - h/2, n)
    return h * np.sum(f(x))


def trapezoid(f, a, b, n):
    """Composite Trapezoid rule."""
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h * (y[0]/2 + np.sum(y[1:-1]) + y[-1]/2)


def simpson(f, a, b, n):
    """Composite Simpson's 1/3 rule (n must be even)."""
    if n % 2 != 0:
        n += 1  # Make n even
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    # Simpson's rule: (h/3) * [y0 + 4*y1 + 2*y2 + 4*y3 + ... + yn]
    result = y[0] + y[-1]
    result += 4 * np.sum(y[1:-1:2])  # Odd indices
    result += 2 * np.sum(y[2:-1:2])  # Even indices (except first and last)

    return h * result / 3


def simpson_38(f, a, b, n):
    """Composite Simpson's 3/8 rule (n must be divisible by 3)."""
    while n % 3 != 0:
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    # Simpson's 3/8: (3h/8) * [y0 + 3*y1 + 3*y2 + 2*y3 + 3*y4 + 3*y5 + 2*y6 + ...]
    result = y[0] + y[-1]
    for i in range(1, n):
        if i % 3 == 0:
            result += 2 * y[i]
        else:
            result += 3 * y[i]

    return 3 * h * result / 8


def count_evaluations(n, method):
    """Count function evaluations for each method."""
    if method in ['riemann_left', 'riemann_right', 'riemann_midpoint']:
        return n
    elif method in ['trapezoid', 'simpson', 'simpson_38']:
        return n + 1
    return n


def main():
    print("\n" + "="*70)
    print("LECTURE 8: Numerical Integration (Quadrature)")
    print("="*70)

    print(f"\nIntegral: integral of sin(x) from {a} to {b}")
    print(f"Exact answer: {exact_integral}")

    # Part 1: Compare methods for fixed n
    print(f"\n{'='*70}")
    print("Part 1: Compare Methods (n=4 intervals)")
    print(f"{'='*70}")

    n = 4
    methods = [
        ("Left Riemann", riemann_left),
        ("Right Riemann", riemann_right),
        ("Midpoint", riemann_midpoint),
        ("Trapezoid", trapezoid),
        ("Simpson 1/3", simpson),
    ]

    print(f"\n{'Method':<20} | {'Approximation':>14} | {'Error':>12} | {'Rel Error':>12}")
    print(f"{'-'*65}")

    for name, method in methods:
        approx = method(f, a, b, n)
        error = abs(approx - exact_integral)
        rel_error = error / exact_integral
        print(f"{name:<20} | {approx:14.10f} | {error:12.2e} | {rel_error:12.2e}")
        time.sleep(0.2)

    # Part 2: Convergence analysis
    print(f"\n{'='*70}")
    print("Part 2: Convergence Analysis (Error vs n)")
    print(f"{'='*70}")

    print("\nTrapezoid Rule (Error should be O(h^2)):")
    print(f"{'n':>6} | {'h':>12} | {'Approx':>14} | {'Error':>12} | {'Error/h^2':>12}")
    print(f"{'-'*65}")

    for n in n_intervals:
        h = (b - a) / n
        approx = trapezoid(f, a, b, n)
        error = abs(approx - exact_integral)
        error_ratio = error / (h**2) if h > 0 else 0
        print(f"{n:6d} | {h:12.6f} | {approx:14.10f} | {error:12.2e} | {error_ratio:12.4f}")
        time.sleep(0.15)

    print("\nSimpson's Rule (Error should be O(h^4)):")
    print(f"{'n':>6} | {'h':>12} | {'Approx':>14} | {'Error':>12} | {'Error/h^4':>12}")
    print(f"{'-'*65}")

    for n in n_intervals:
        if n % 2 != 0:
            continue
        h = (b - a) / n
        approx = simpson(f, a, b, n)
        error = abs(approx - exact_integral)
        error_ratio = error / (h**4) if h > 0 else 0
        print(f"{n:6d} | {h:12.6f} | {approx:14.10f} | {error:12.2e} | {error_ratio:12.4f}")
        time.sleep(0.15)

    # Part 3: Hand calculation example
    print(f"\n{'='*70}")
    print("Part 3: Hand Calculation (Trapezoid with n=2)")
    print(f"{'='*70}")

    n_hand = 2
    h_hand = (b - a) / n_hand
    x_points = np.linspace(a, b, n_hand + 1)
    y_points = f(x_points)

    print(f"\nStep size h = (b-a)/n = ({b:.4f} - {a})/2 = {h_hand:.4f}")
    print(f"\nEvaluation points and function values:")
    for i, (xi, yi) in enumerate(zip(x_points, y_points)):
        print(f"  x_{i} = {xi:.4f}, f(x_{i}) = sin({xi:.4f}) = {yi:.4f}")

    print(f"\nTrapezoid formula: h * [f(a)/2 + f(x_1) + ... + f(x_{n-1}) + f(b)/2]")
    print(f"                 = {h_hand:.4f} * [{y_points[0]:.4f}/2 + {y_points[1]:.4f} + {y_points[2]:.4f}/2]")
    print(f"                 = {h_hand:.4f} * [{y_points[0]/2:.4f} + {y_points[1]:.4f} + {y_points[2]/2:.4f}]")
    print(f"                 = {h_hand:.4f} * {y_points[0]/2 + y_points[1] + y_points[2]/2:.4f}")
    print(f"                 = {trapezoid(f, a, b, n_hand):.4f}")

    # Part 4: Function evaluations vs accuracy
    print(f"\n{'='*70}")
    print("Part 4: Accuracy vs Function Evaluations")
    print(f"{'='*70}")

    print(f"\n{'Method':<15} | {'n':>6} | {'Evals':>6} | {'Error':>12}")
    print(f"{'-'*50}")

    # Same number of evaluations
    for target_evals in [5, 9, 17]:
        trap_n = target_evals - 1
        simp_n = target_evals - 1
        if simp_n % 2 != 0:
            simp_n -= 1

        trap_err = abs(trapezoid(f, a, b, trap_n) - exact_integral)
        simp_err = abs(simpson(f, a, b, simp_n) - exact_integral)

        print(f"{'Trapezoid':<15} | {trap_n:6d} | {trap_n+1:6d} | {trap_err:12.2e}")
        print(f"{'Simpson':<15} | {simp_n:6d} | {simp_n+1:6d} | {simp_err:12.2e}")
        print()

    print("Observation: Simpson achieves better accuracy with same function evaluations!")

    print("\n" + "="*70)
    print("EXERCISES:")
    print("="*70)
    print("""
1. Change the test function to f(x) = exp(-x^2) on [0, 2].
   (No closed-form antiderivative!) Compare methods.

2. Plot error vs n on a log-log scale. Verify that Trapezoid has
   slope -2 and Simpson has slope -4.

3. For a discontinuous function like f(x) = sign(x - 0.5) on [0, 1],
   how do the methods perform? Why does smoothness matter?

4. Implement adaptive quadrature: subdivide intervals where the
   local error estimate is large.

5. Compute the integral of your favorite ML function: e.g.,
   sigmoid(x) from -10 to 10, or the Gaussian PDF.
""")


if __name__ == "__main__":
    main()
