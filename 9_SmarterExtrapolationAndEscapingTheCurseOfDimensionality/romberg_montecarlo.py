import time
import numpy as np

# Romberg Integration and Monte Carlo Methods
# Smarter extrapolation and escaping the curse of dimensionality

np.random.seed(42)

# =====================================================================
# Configuration Parameters - Experiment with these!
# =====================================================================

# Test function for Romberg
def f_romberg(x):
    return np.exp(x)

a_romberg, b_romberg = 0, 1
exact_romberg = np.exp(1) - 1  # e - 1

# Test function for Monte Carlo (multi-dimensional)
def f_mc(x):
    """Sphere indicator: 1 if ||x|| < 1, else 0"""
    return 1.0 if np.linalg.norm(x) < 1 else 0.0

# Monte Carlo samples
mc_samples = [100, 1000, 10000, 100000]

# Dimensions to test for curse of dimensionality
dimensions = [1, 2, 3, 5, 10]

# =====================================================================


def trapezoid(f, a, b, n):
    """Composite trapezoid rule."""
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.array([f(xi) for xi in x])
    return h * (y[0]/2 + np.sum(y[1:-1]) + y[-1]/2)


def romberg(f, a, b, max_level=5):
    """
    Romberg integration: Richardson extrapolation applied to trapezoid.
    Returns the Romberg tableau.
    """
    R = np.zeros((max_level, max_level))

    # First column: trapezoid rule with 1, 2, 4, 8, ... intervals
    for i in range(max_level):
        n = 2**i
        R[i, 0] = trapezoid(f, a, b, n)

    # Fill in the tableau using Richardson extrapolation
    for j in range(1, max_level):
        for i in range(j, max_level):
            # R[i,j] = (4^j * R[i,j-1] - R[i-1,j-1]) / (4^j - 1)
            factor = 4**j
            R[i, j] = (factor * R[i, j-1] - R[i-1, j-1]) / (factor - 1)

    return R


def monte_carlo_1d(f, a, b, n_samples):
    """Monte Carlo integration in 1D."""
    samples = np.random.uniform(a, b, n_samples)
    values = np.array([f(x) for x in samples])
    integral = (b - a) * np.mean(values)
    std_error = (b - a) * np.std(values) / np.sqrt(n_samples)
    return integral, std_error


def monte_carlo_nd(f, bounds, n_samples):
    """
    Monte Carlo integration in n dimensions.
    bounds: list of (low, high) tuples for each dimension
    """
    d = len(bounds)
    volume = np.prod([b - a for a, b in bounds])

    # Generate random samples in the hypercube
    samples = np.array([
        np.random.uniform(low, high, n_samples)
        for low, high in bounds
    ]).T  # Shape: (n_samples, d)

    values = np.array([f(x) for x in samples])
    integral = volume * np.mean(values)
    std_error = volume * np.std(values) / np.sqrt(n_samples)

    return integral, std_error


def hypersphere_volume(d):
    """Exact volume of unit hypersphere in d dimensions."""
    if d == 1:
        return 2
    elif d == 2:
        return np.pi
    else:
        # V_d = pi^(d/2) / Gamma(d/2 + 1)
        from scipy.special import gamma
        return (np.pi ** (d/2)) / gamma(d/2 + 1)


def trapezoid_nd_grid(f, bounds, n_per_dim):
    """
    n-dimensional trapezoid rule using tensor product grid.
    WARNING: Cost is O(n_per_dim^d) - exponential in dimension!
    """
    d = len(bounds)
    total_points = n_per_dim ** d

    if total_points > 1e6:
        return None, total_points  # Too expensive

    # Create grid
    grids = [np.linspace(a, b, n_per_dim) for a, b in bounds]
    mesh = np.meshgrid(*grids, indexing='ij')
    points = np.stack([m.flatten() for m in mesh], axis=1)

    # Evaluate function
    values = np.array([f(x) for x in points])

    # Simple average * volume (crude approximation)
    volume = np.prod([b - a for a, b in bounds])
    integral = volume * np.mean(values)

    return integral, total_points


def main():
    print("\n" + "="*70)
    print("LECTURE 9: Romberg Integration & Monte Carlo")
    print("="*70)

    # Part 1: Romberg Integration
    print(f"\n{'='*70}")
    print("Part 1: Romberg Integration")
    print(f"Function: exp(x) from {a_romberg} to {b_romberg}")
    print(f"Exact integral: e - 1 = {exact_romberg:.10f}")
    print(f"{'='*70}")

    R = romberg(f_romberg, a_romberg, b_romberg, max_level=5)

    print("\nRomberg Tableau:")
    print("(Rows = more intervals, Columns = higher-order extrapolation)")
    print()

    # Print tableau
    print(f"{'n':>4}", end=" | ")
    for j in range(5):
        print(f"{'R[i,'+str(j)+']':>14}", end=" | ")
    print()
    print("-" * 90)

    for i in range(5):
        n = 2**i
        print(f"{n:4d}", end=" | ")
        for j in range(i + 1):
            error = abs(R[i, j] - exact_romberg)
            if error < 1e-12:
                print(f"{R[i,j]:14.10f}*", end=" | ")
            else:
                print(f"{R[i,j]:14.10f}", end="  | ")
        print()
        time.sleep(0.3)

    print("\n* = converged to machine precision")
    print(f"\nBest estimate: R[4,4] = {R[4,4]:.15f}")
    print(f"True value:           {exact_romberg:.15f}")
    print(f"Error:                {abs(R[4,4] - exact_romberg):.2e}")

    # Part 2: Monte Carlo in 1D
    print(f"\n{'='*70}")
    print("Part 2: Monte Carlo Integration (1D)")
    print(f"Same function: exp(x) from {a_romberg} to {b_romberg}")
    print(f"{'='*70}")

    print(f"\n{'Samples':>10} | {'Estimate':>14} | {'Std Error':>12} | {'True Error':>12}")
    print(f"{'-'*55}")

    for n in mc_samples:
        est, std_err = monte_carlo_1d(f_romberg, a_romberg, b_romberg, n)
        true_err = abs(est - exact_romberg)
        print(f"{n:10d} | {est:14.6f} | {std_err:12.2e} | {true_err:12.2e}")

    print("\nObservation: Error decreases as O(1/sqrt(N)) - need 100x more")
    print("samples for 10x less error. Slow but dimension-independent!")

    # Part 3: Curse of Dimensionality
    print(f"\n{'='*70}")
    print("Part 3: The Curse of Dimensionality")
    print("Computing volume of unit hypersphere in d dimensions")
    print(f"{'='*70}")

    print(f"\n{'d':>4} | {'Grid (n=10/dim)':>18} | {'Grid Points':>12} | {'MC (n=10000)':>14} | {'Exact':>12}")
    print(f"{'-'*75}")

    for d in dimensions:
        bounds = [(-1, 1)] * d

        # Grid-based (exponential cost)
        grid_result, grid_points = trapezoid_nd_grid(f_mc, bounds, n_per_dim=10)

        # Monte Carlo (linear cost in samples)
        mc_result, mc_std = monte_carlo_nd(f_mc, bounds, 10000)

        # Scale by 2^d (we integrated over [-1,1]^d, hypersphere volume formula)
        exact = hypersphere_volume(d)

        if grid_result is not None:
            # Scale indicator function result to get volume
            grid_vol = grid_result
            print(f"{d:4d} | {grid_vol:18.6f} | {grid_points:12d} | {mc_result:14.6f} | {exact:12.6f}")
        else:
            print(f"{d:4d} | {'TOO EXPENSIVE':>18} | {grid_points:12.0e} | {mc_result:14.6f} | {exact:12.6f}")

        time.sleep(0.3)

    print("\nKey insight: Grid methods need O(n^d) points - exponential in dimension!")
    print("Monte Carlo needs O(n) points regardless of dimension.")

    # Part 4: Estimate pi using Monte Carlo
    print(f"\n{'='*70}")
    print("Part 4: Estimate Pi using Monte Carlo")
    print("(Classic example: area of quarter circle in unit square)")
    print(f"{'='*70}")

    def quarter_circle(x):
        return 1.0 if x[0]**2 + x[1]**2 < 1 else 0.0

    print(f"\n{'Samples':>10} | {'Pi Estimate':>14} | {'Error':>12}")
    print(f"{'-'*45}")

    for n in mc_samples:
        est, _ = monte_carlo_nd(quarter_circle, [(0, 1), (0, 1)], n)
        pi_est = 4 * est  # Quarter circle area = pi/4
        error = abs(pi_est - np.pi)
        print(f"{n:10d} | {pi_est:14.10f} | {error:12.2e}")

    print(f"\nTrue pi:     {np.pi:.10f}")

    print("\n" + "="*70)
    print("EXERCISES:")
    print("="*70)
    print("""
1. Modify the Romberg code to stop when |R[i,j] - R[i-1,j-1]| < tolerance.
   How many function evaluations does this save?

2. Implement importance sampling for Monte Carlo: sample more densely
   where f(x) is large. Test on f(x) = exp(-x^2) on [-5, 5].

3. Compare convergence of Romberg vs Monte Carlo for the same number
   of function evaluations. At what dimension does MC win?

4. Implement stratified sampling: divide [0,1]^d into strata and
   sample uniformly within each. Compare variance to simple MC.

5. Use Monte Carlo to estimate the expected loss E[L(theta)] for a
   simple model. This is how SGD approximates the gradient!
""")


if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("Note: scipy is needed for exact hypersphere volume in d>2.")
        print("Install with: pip install scipy")
