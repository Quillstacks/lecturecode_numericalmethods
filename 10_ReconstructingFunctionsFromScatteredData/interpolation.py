import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Polynomial Interpolation and the Runge Phenomenon
# Lagrange interpolation, overfitting, and extrapolation danger

np.random.seed(42)

# =====================================================================
# Configuration Parameters - Experiment with these!
# =====================================================================

# Runge's function: f(x) = 1 / (1 + 25*x^2)
def runge(x):
    return 1 / (1 + 25 * x**2)

# Interpolation domain
x_min, x_max = -1, 1

# Number of interpolation points to try
n_points_list = [5, 9, 15, 21]

# =====================================================================


def lagrange_basis(x, i, x_nodes):
    """
    Compute the i-th Lagrange basis polynomial L_i(x).
    L_i(x) = prod_{j != i} (x - x_j) / (x_i - x_j)
    """
    L = 1.0
    xi = x_nodes[i]
    for j, xj in enumerate(x_nodes):
        if j != i:
            L *= (x - xj) / (xi - xj)
    return L


def lagrange_interpolate(x, x_nodes, y_nodes):
    """
    Lagrange interpolation at point x.
    P(x) = sum_i y_i * L_i(x)
    """
    P = 0.0
    for i, yi in enumerate(y_nodes):
        P += yi * lagrange_basis(x, i, x_nodes)
    return P


def lagrange_interpolate_array(x_eval, x_nodes, y_nodes):
    """Lagrange interpolation at multiple points."""
    return np.array([lagrange_interpolate(x, x_nodes, y_nodes) for x in x_eval])


def chebyshev_nodes(n, a=-1, b=1):
    """
    Chebyshev nodes: x_k = cos((2k+1)*pi / (2n)), k = 0, ..., n-1
    Scaled to interval [a, b]
    """
    k = np.arange(n)
    nodes = np.cos((2*k + 1) * np.pi / (2*n))
    # Scale from [-1, 1] to [a, b]
    return (a + b)/2 + (b - a)/2 * nodes


def main():
    print("\n" + "="*70)
    print("LECTURE 10: Interpolation & The Runge Phenomenon")
    print("="*70)

    # Part 1: Lagrange Interpolation Basics
    print(f"\n{'='*70}")
    print("Part 1: Lagrange Interpolation by Hand")
    print(f"{'='*70}")

    # Simple example: interpolate through (0,1), (1,0), (2,1)
    x_simple = np.array([0, 1, 2])
    y_simple = np.array([1, 0, 1])

    print("\nData points: (0,1), (1,0), (2,1)")
    print("\nLagrange basis polynomials:")
    print("L_0(x) = (x-1)(x-2) / ((0-1)(0-2)) = (x-1)(x-2) / 2")
    print("L_1(x) = (x-0)(x-2) / ((1-0)(1-2)) = -x(x-2)")
    print("L_2(x) = (x-0)(x-1) / ((2-0)(2-1)) = x(x-1) / 2")
    print("\nP(x) = 1*L_0(x) + 0*L_1(x) + 1*L_2(x)")
    print("     = (x-1)(x-2)/2 + x(x-1)/2")
    print("     = (x^2 - 3x + 2 + x^2 - x) / 2")
    print("     = (2x^2 - 4x + 2) / 2")
    print("     = x^2 - 2x + 1 = (x-1)^2")

    print("\nVerification:")
    for xi, yi in zip(x_simple, y_simple):
        P_xi = lagrange_interpolate(xi, x_simple, y_simple)
        print(f"  P({xi}) = {P_xi:.4f}, expected {yi}")

    # Part 2: Runge's Phenomenon
    print(f"\n{'='*70}")
    print("Part 2: Runge's Phenomenon")
    print(f"Function: f(x) = 1 / (1 + 25*x^2) on [{x_min}, {x_max}]")
    print(f"{'='*70}")

    x_fine = np.linspace(x_min, x_max, 500)
    y_true = runge(x_fine)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, n in enumerate(n_points_list):
        # Equally spaced nodes
        x_nodes = np.linspace(x_min, x_max, n)
        y_nodes = runge(x_nodes)

        # Interpolate
        y_interp = lagrange_interpolate_array(x_fine, x_nodes, y_nodes)

        # Compute max error
        max_error = np.max(np.abs(y_interp - y_true))

        print(f"\nn = {n} points (degree {n-1} polynomial):")
        print(f"  Max interpolation error: {max_error:.4f}")

        # Plot
        ax = axes[idx]
        ax.plot(x_fine, y_true, 'k-', linewidth=2, label='True function')
        ax.plot(x_fine, y_interp, 'b--', linewidth=1.5, label=f'Interpolant (n={n})')
        ax.scatter(x_nodes, y_nodes, c='red', s=50, zorder=5, label='Data points')
        ax.set_xlim(x_min - 0.1, x_max + 0.1)
        ax.set_ylim(-0.5, 1.5)
        ax.set_title(f'n = {n} points, max error = {max_error:.2f}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        time.sleep(0.3)

    plt.tight_layout()
    plt.savefig('runge_phenomenon.png', dpi=150)
    print(f"\nVisualization saved to: runge_phenomenon.png")

    print("\nObservation: More points leads to WORSE approximation at the edges!")
    print("This is the Runge phenomenon - polynomial overfitting.")

    # Part 3: Chebyshev Nodes
    print(f"\n{'='*70}")
    print("Part 3: Chebyshev Nodes (Solution to Runge)")
    print(f"{'='*70}")

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

    n = 15

    # Equally spaced
    x_equal = np.linspace(x_min, x_max, n)
    y_equal = runge(x_equal)
    y_interp_equal = lagrange_interpolate_array(x_fine, x_equal, y_equal)
    error_equal = np.max(np.abs(y_interp_equal - y_true))

    # Chebyshev
    x_cheb = chebyshev_nodes(n, x_min, x_max)
    y_cheb = runge(x_cheb)
    y_interp_cheb = lagrange_interpolate_array(x_fine, x_cheb, y_cheb)
    error_cheb = np.max(np.abs(y_interp_cheb - y_true))

    print(f"\nn = {n} interpolation points:")
    print(f"  Equally spaced: max error = {error_equal:.4f}")
    print(f"  Chebyshev nodes: max error = {error_cheb:.4f}")
    print(f"  Improvement: {error_equal / error_cheb:.1f}x better!")

    axes2[0].plot(x_fine, y_true, 'k-', linewidth=2, label='True')
    axes2[0].plot(x_fine, y_interp_equal, 'b--', label='Interpolant')
    axes2[0].scatter(x_equal, y_equal, c='red', s=50)
    axes2[0].set_title(f'Equally Spaced (n={n}), error={error_equal:.2f}')
    axes2[0].set_ylim(-1, 2)
    axes2[0].legend()

    axes2[1].plot(x_fine, y_true, 'k-', linewidth=2, label='True')
    axes2[1].plot(x_fine, y_interp_cheb, 'g--', label='Interpolant')
    axes2[1].scatter(x_cheb, y_cheb, c='red', s=50)
    axes2[1].set_title(f'Chebyshev Nodes (n={n}), error={error_cheb:.4f}')
    axes2[1].set_ylim(-0.2, 1.2)
    axes2[1].legend()

    plt.tight_layout()
    plt.savefig('chebyshev_nodes.png', dpi=150)
    print(f"Visualization saved to: chebyshev_nodes.png")

    # Part 4: Extrapolation Danger
    print(f"\n{'='*70}")
    print("Part 4: Extrapolation Danger")
    print(f"{'='*70}")

    # Fit polynomial to data in [-1, 1], evaluate outside
    n = 5
    x_nodes = np.linspace(-1, 1, n)
    y_nodes = runge(x_nodes)

    x_extrap = np.linspace(-2, 2, 100)
    y_extrap = lagrange_interpolate_array(x_extrap, x_nodes, y_nodes)
    y_true_extrap = runge(x_extrap)

    print(f"\nPolynomial fit on [-1, 1] with n={n} points:")
    print("\nExtrapolation comparison:")
    print(f"{'x':>6} | {'True f(x)':>12} | {'Polynomial':>12} | {'Error':>12}")
    print(f"{'-'*50}")

    test_points = [-1.5, -1.2, 0, 1.2, 1.5]
    for x in test_points:
        true_val = runge(x)
        poly_val = lagrange_interpolate(x, x_nodes, y_nodes)
        error = abs(poly_val - true_val)
        marker = " <-- EXTRAPOLATION" if abs(x) > 1 else ""
        print(f"{x:6.1f} | {true_val:12.4f} | {poly_val:12.4f} | {error:12.4f}{marker}")

    print("\nExtrapolation is DANGEROUS: polynomials grow unboundedly outside")
    print("the data range. This is analogous to distribution shift in ML!")

    print("\n" + "="*70)
    print("EXERCISES:")
    print("="*70)
    print("""
1. Try interpolating f(x) = |x| on [-1, 1]. What happens?
   (Hint: This function isn't smooth at x=0)

2. Implement Newton's divided differences form of interpolation.
   Verify it gives the same polynomial as Lagrange.

3. Plot the Lagrange basis functions L_i(x) for n=5 equally spaced
   points. Note how they oscillate.

4. For a neural network as a function approximator, what plays the
   role of "interpolation points"? How does this relate to overfitting?

5. Compare polynomial interpolation to linear regression. What are
   the trade-offs? When would you prefer each?
""")


if __name__ == "__main__":
    main()
