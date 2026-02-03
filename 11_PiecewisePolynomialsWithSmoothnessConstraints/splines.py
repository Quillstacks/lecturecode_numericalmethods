import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Spline Interpolation
# Piecewise polynomials with smoothness constraints

np.random.seed(42)

# =====================================================================
# Configuration Parameters - Experiment with these!
# =====================================================================

# Runge's function for comparison
def runge(x):
    return 1 / (1 + 25 * x**2)

# Domain
x_min, x_max = -1, 1

# Number of data points
n_points = 11

# =====================================================================


def piecewise_linear(x_eval, x_nodes, y_nodes):
    """Piecewise linear interpolation (simplest spline)."""
    result = np.zeros_like(x_eval, dtype=float)

    for i, x in enumerate(x_eval):
        # Find the interval containing x
        for j in range(len(x_nodes) - 1):
            if x_nodes[j] <= x <= x_nodes[j + 1]:
                # Linear interpolation in this interval
                t = (x - x_nodes[j]) / (x_nodes[j + 1] - x_nodes[j])
                result[i] = (1 - t) * y_nodes[j] + t * y_nodes[j + 1]
                break
        else:
            # Extrapolation (use nearest endpoint)
            if x < x_nodes[0]:
                result[i] = y_nodes[0]
            else:
                result[i] = y_nodes[-1]

    return result


def cubic_spline_natural(x_nodes, y_nodes):
    """
    Compute natural cubic spline coefficients.
    Returns coefficients a, b, c, d for each interval.
    S_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3
    """
    n = len(x_nodes) - 1  # Number of intervals
    h = np.diff(x_nodes)  # Interval widths

    # Set up tridiagonal system for second derivatives (c coefficients)
    # Natural spline: c_0 = c_n = 0

    # Build the system Ac = rhs
    A = np.zeros((n + 1, n + 1))
    rhs = np.zeros(n + 1)

    # Boundary conditions (natural spline)
    A[0, 0] = 1
    A[n, n] = 1
    rhs[0] = 0
    rhs[n] = 0

    # Interior equations
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 3 * ((y_nodes[i + 1] - y_nodes[i]) / h[i] -
                       (y_nodes[i] - y_nodes[i - 1]) / h[i - 1])

    # Solve for c coefficients
    c = np.linalg.solve(A, rhs)

    # Compute other coefficients
    a = y_nodes[:-1]
    b = np.zeros(n)
    d = np.zeros(n)

    for i in range(n):
        b[i] = (y_nodes[i + 1] - y_nodes[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    return a, b, c[:-1], d, x_nodes


def evaluate_cubic_spline(x_eval, a, b, c, d, x_nodes):
    """Evaluate cubic spline at given points."""
    result = np.zeros_like(x_eval, dtype=float)

    for i, x in enumerate(x_eval):
        # Find the interval
        for j in range(len(x_nodes) - 1):
            if x_nodes[j] <= x <= x_nodes[j + 1]:
                dx = x - x_nodes[j]
                result[i] = a[j] + b[j] * dx + c[j] * dx**2 + d[j] * dx**3
                break
        else:
            # Extrapolation
            if x < x_nodes[0]:
                dx = x - x_nodes[0]
                result[i] = a[0] + b[0] * dx
            else:
                dx = x - x_nodes[-2]
                result[i] = a[-1] + b[-1] * dx + c[-1] * dx**2 + d[-1] * dx**3

    return result


def main():
    print("\n" + "="*70)
    print("LECTURE 11: Splines")
    print("="*70)

    # Generate data points
    x_nodes = np.linspace(x_min, x_max, n_points)
    y_nodes = runge(x_nodes)
    x_fine = np.linspace(x_min, x_max, 500)
    y_true = runge(x_fine)

    # Part 1: Piecewise Linear vs Polynomial
    print(f"\n{'='*70}")
    print("Part 1: Piecewise Linear Interpolation")
    print(f"{'='*70}")

    y_linear = piecewise_linear(x_fine, x_nodes, y_nodes)
    error_linear = np.max(np.abs(y_linear - y_true))

    print(f"\nData: {n_points} points on Runge function")
    print(f"Piecewise linear max error: {error_linear:.4f}")
    print("\nAdvantage: No oscillation! Always stays between data values.")
    print("Disadvantage: Not smooth - has corners at data points.")

    # Part 2: Cubic Spline
    print(f"\n{'='*70}")
    print("Part 2: Natural Cubic Spline")
    print(f"{'='*70}")

    a, b, c, d, x_knots = cubic_spline_natural(x_nodes, y_nodes)
    y_spline = evaluate_cubic_spline(x_fine, a, b, c, d, x_knots)
    error_spline = np.max(np.abs(y_spline - y_true))

    print(f"\nCubic spline max error: {error_spline:.4f}")
    print(f"Improvement over linear: {error_linear / error_spline:.1f}x")

    print("\nCubic spline properties:")
    print("  - Passes through all data points (interpolation)")
    print("  - Continuous first derivative (C1 smooth)")
    print("  - Continuous second derivative (C2 smooth)")
    print("  - Natural BC: S''(x_0) = S''(x_n) = 0")

    # Part 3: Spline Coefficients
    print(f"\n{'='*70}")
    print("Part 3: Spline Coefficients (First 3 Intervals)")
    print(f"{'='*70}")

    print("\nS_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3")
    print(f"\n{'i':>3} | {'[x_i, x_{i+1}]':>16} | {'a_i':>10} | {'b_i':>10} | {'c_i':>10} | {'d_i':>10}")
    print(f"{'-'*75}")

    for i in range(min(3, len(a))):
        interval = f"[{x_nodes[i]:.2f}, {x_nodes[i+1]:.2f}]"
        print(f"{i:3d} | {interval:>16} | {a[i]:10.4f} | {b[i]:10.4f} | {c[i]:10.4f} | {d[i]:10.4f}")

    # Part 4: Smoothness Verification
    print(f"\n{'='*70}")
    print("Part 4: Smoothness at Knots")
    print(f"{'='*70}")

    print("\nChecking continuity at interior knots:")
    print(f"{'Knot x_i':>10} | {'Left S(x)':>12} | {'Right S(x)':>12} | {'Difference':>12}")
    print(f"{'-'*55}")

    for i in range(1, min(4, len(x_nodes) - 1)):
        x_knot = x_nodes[i]
        # Value from left interval
        dx_left = x_knot - x_nodes[i - 1]
        val_left = a[i-1] + b[i-1]*dx_left + c[i-1]*dx_left**2 + d[i-1]*dx_left**3
        # Value from right interval
        val_right = a[i]
        diff = abs(val_left - val_right)
        print(f"{x_knot:10.4f} | {val_left:12.6f} | {val_right:12.6f} | {diff:12.2e}")

    # Part 5: Comparison with high-degree polynomial
    print(f"\n{'='*70}")
    print("Part 5: Spline vs High-Degree Polynomial (Runge)")
    print(f"{'='*70}")

    # Lagrange polynomial (from lecture 10)
    def lagrange_interpolate_array(x_eval, x_nodes, y_nodes):
        n = len(x_nodes)
        result = np.zeros_like(x_eval)
        for k in range(n):
            L = np.ones_like(x_eval)
            for j in range(n):
                if j != k:
                    L *= (x_eval - x_nodes[j]) / (x_nodes[k] - x_nodes[j])
            result += y_nodes[k] * L
        return result

    y_poly = lagrange_interpolate_array(x_fine, x_nodes, y_nodes)
    error_poly = np.max(np.abs(y_poly - y_true))

    print(f"\n{n_points} data points on Runge function:")
    print(f"  Polynomial (degree {n_points-1}) max error: {error_poly:.4f}")
    print(f"  Cubic spline max error:                    {error_spline:.4f}")
    print(f"  Spline is {error_poly / error_spline:.1f}x better!")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: All methods
    axes[0, 0].plot(x_fine, y_true, 'k-', linewidth=2, label='True function')
    axes[0, 0].plot(x_fine, y_poly, 'r--', linewidth=1.5, label=f'Polynomial (deg {n_points-1})')
    axes[0, 0].plot(x_fine, y_spline, 'b-', linewidth=1.5, label='Cubic spline')
    axes[0, 0].scatter(x_nodes, y_nodes, c='green', s=50, zorder=5, label='Data')
    axes[0, 0].set_ylim(-0.5, 1.5)
    axes[0, 0].legend()
    axes[0, 0].set_title('Polynomial vs Spline')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Spline only
    axes[0, 1].plot(x_fine, y_true, 'k-', linewidth=2, label='True')
    axes[0, 1].plot(x_fine, y_spline, 'b-', linewidth=1.5, label='Spline')
    axes[0, 1].scatter(x_nodes, y_nodes, c='red', s=50, zorder=5)
    axes[0, 1].set_ylim(-0.1, 1.1)
    axes[0, 1].legend()
    axes[0, 1].set_title('Cubic Spline (zoomed)')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Error comparison
    axes[1, 0].semilogy(x_fine, np.abs(y_poly - y_true), 'r-', label='Polynomial error')
    axes[1, 0].semilogy(x_fine, np.abs(y_spline - y_true), 'b-', label='Spline error')
    axes[1, 0].legend()
    axes[1, 0].set_title('Interpolation Error (log scale)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('|error|')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Linear vs Cubic
    axes[1, 1].plot(x_fine, y_true, 'k-', linewidth=2, label='True')
    axes[1, 1].plot(x_fine, y_linear, 'g--', linewidth=1.5, label='Piecewise linear')
    axes[1, 1].plot(x_fine, y_spline, 'b-', linewidth=1.5, label='Cubic spline')
    axes[1, 1].scatter(x_nodes, y_nodes, c='red', s=50, zorder=5)
    axes[1, 1].set_ylim(-0.1, 1.1)
    axes[1, 1].legend()
    axes[1, 1].set_title('Linear vs Cubic Spline')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('splines.png', dpi=150)
    print(f"\nVisualization saved to: splines.png")

    print("\n" + "="*70)
    print("EXERCISES:")
    print("="*70)
    print("""
1. Implement clamped boundary conditions: specify S'(x_0) and S'(x_n).
   Compare to natural splines.

2. What happens to the cubic spline as you increase the number of
   data points? Does it always improve?

3. For image interpolation, we use 2D bicubic splines. Research how
   they're constructed from 1D cubic splines.

4. Compare the number of coefficients: polynomial (n) vs spline (4n).
   Why is spline more stable despite having more parameters?

5. Implement B-spline basis functions. Verify they are non-negative
   and form a partition of unity.
""")


if __name__ == "__main__":
    main()
