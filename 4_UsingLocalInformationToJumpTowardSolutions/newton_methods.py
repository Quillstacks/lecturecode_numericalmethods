import time
import numpy as np

# Newton-Raphson and Secant Method Implementation
# Explore convergence rates and failure modes

# Set random seed for reproducibility
np.random.seed(42)

# =====================================================================
# Configuration Parameters - Experiment with these!
# =====================================================================

# Problem: Find the root of f(x) = x^2 - 2 (i.e., compute sqrt(2))
# Try changing these to x^3 - 5 for cube root of 5

def f(x):
    return x**2 - 2

def f_prime(x):
    return 2*x

# Starting points
x0_newton = 1.0      # Newton starting point
x0_secant = 1.0      # Secant first point
x1_secant = 2.0      # Secant second point

# Convergence settings
tolerance = 1e-12
max_iterations = 20

# Known exact solution (for error calculation)
exact_solution = np.sqrt(2)

# =====================================================================


def newton_raphson(f, f_prime, x0, tol, max_iter):
    """
    Newton-Raphson method for finding roots.

    Returns: list of (iteration, x_n, f(x_n), error) tuples
    """
    history = []
    x = x0

    for n in range(max_iter):
        fx = f(x)
        error = abs(x - exact_solution)
        history.append((n, x, fx, error))

        if abs(fx) < tol:
            break

        fpx = f_prime(x)
        if abs(fpx) < 1e-15:
            print(f"Warning: Derivative near zero at iteration {n}")
            break

        x = x - fx / fpx

    return history


def secant_method(f, x0, x1, tol, max_iter):
    """
    Secant method for finding roots (no derivative needed).

    Returns: list of (iteration, x_n, f(x_n), error) tuples
    """
    history = []

    for n in range(max_iter):
        fx1 = f(x1)
        error = abs(x1 - exact_solution)
        history.append((n, x1, fx1, error))

        if abs(fx1) < tol:
            break

        fx0 = f(x0)
        denominator = fx1 - fx0
        if abs(denominator) < 1e-15:
            print(f"Warning: Division by zero at iteration {n}")
            break

        x_new = x1 - fx1 * (x1 - x0) / denominator
        x0 = x1
        x1 = x_new

    return history


def print_convergence_table(newton_hist, secant_hist):
    """Print side-by-side comparison of Newton and Secant convergence."""

    print(f"\n{'='*80}")
    print(f"Convergence Comparison: Newton-Raphson vs Secant Method")
    print(f"Target: sqrt(2) = {exact_solution:.10f}")
    print(f"{'='*80}")
    print(f"{'n':>3} | {'Newton x_n':>14} | {'Newton Error':>12} | {'Secant x_n':>14} | {'Secant Error':>12}")
    print(f"{'-'*80}")

    max_len = max(len(newton_hist), len(secant_hist))

    for i in range(max_len):
        # Newton values
        if i < len(newton_hist):
            n_x = f"{newton_hist[i][1]:.10f}"
            n_err = f"{newton_hist[i][3]:.2e}"
        else:
            n_x = "converged"
            n_err = "-"

        # Secant values
        if i < len(secant_hist):
            s_x = f"{secant_hist[i][1]:.10f}"
            s_err = f"{secant_hist[i][3]:.2e}"
        else:
            s_x = "converged"
            s_err = "-"

        print(f"{i:3d} | {n_x:>14} | {n_err:>12} | {s_x:>14} | {s_err:>12}")
        time.sleep(0.3)

    print(f"{'='*80}")
    print(f"Newton iterations: {len(newton_hist)}, Secant iterations: {len(secant_hist)}")
    print(f"{'='*80}\n")


def analyze_convergence_rate(history, method_name):
    """
    Analyze the convergence rate by looking at error ratios.

    For quadratic convergence: |e_{n+1}| / |e_n|^2 should be roughly constant
    For superlinear (phi): |e_{n+1}| / |e_n|^phi should be roughly constant
    """
    print(f"\n{method_name} Convergence Rate Analysis:")
    print(f"{'-'*50}")

    errors = [h[3] for h in history if h[3] > 1e-14]  # Skip near-zero errors

    if len(errors) < 3:
        print("Not enough data points for analysis")
        return

    print(f"{'n':>3} | {'Error':>12} | {'e_{n+1}/e_n^2':>14} | {'e_{n+1}/e_n^1.618':>16}")
    print(f"{'-'*50}")

    phi = (1 + np.sqrt(5)) / 2  # Golden ratio ~ 1.618

    for i in range(len(errors) - 1):
        e_n = errors[i]
        e_n1 = errors[i + 1]

        ratio_quad = e_n1 / (e_n ** 2) if e_n > 1e-10 else float('inf')
        ratio_phi = e_n1 / (e_n ** phi) if e_n > 1e-10 else float('inf')

        print(f"{i:3d} | {e_n:.6e} | {ratio_quad:14.6f} | {ratio_phi:16.6f}")

    print(f"{'-'*50}")
    print("Quadratic convergence: ratio in column 3 should stabilize")
    print("Superlinear (phi=1.618): ratio in column 4 should stabilize")


def demonstrate_failure_modes():
    """
    Demonstrate Newton's method failure modes.
    """
    print(f"\n{'='*80}")
    print("Newton's Method Failure Modes")
    print(f"{'='*80}")

    # Failure mode 1: Multiple root (f(x) = x^3, root at 0 with multiplicity 3)
    print("\n1. Multiple Root: f(x) = x^3 (root at x=0 has multiplicity 3)")
    print("   Expected: Linear convergence instead of quadratic")

    def f_cubic(x):
        return x**3

    def f_cubic_prime(x):
        return 3*x**2

    x = 1.0
    print(f"\n   {'n':>3} | {'x_n':>14} | {'x_{n+1}/x_n':>14}")
    print(f"   {'-'*40}")

    for n in range(8):
        fx = f_cubic(x)
        fpx = f_cubic_prime(x)
        x_new = x - fx / fpx
        ratio = x_new / x if abs(x) > 1e-10 else 0
        print(f"   {n:3d} | {x:14.10f} | {ratio:14.10f}")
        x = x_new

    print("   Note: x_{n+1}/x_n = 2/3 (constant ratio = linear convergence)")

    # Failure mode 2: Zero derivative
    print("\n2. Zero Derivative: f(x) = x^2 at x=0")
    print("   If we start exactly at a stationary point, Newton fails.")
    print("   f'(0) = 0, so we can't compute -f(x)/f'(x)")

    # Failure mode 3: Cycling (more complex to demonstrate simply)
    print("\n3. Bad Starting Point / Different Root:")
    print("   f(x) = x^3 - x has roots at x = -1, 0, 1")
    print("   Starting point determines which root we find!")

    def f_multi(x):
        return x**3 - x

    def f_multi_prime(x):
        return 3*x**2 - 1

    for x0 in [0.5, 0.6, -0.5, 1.5]:
        x = x0
        for _ in range(20):
            fx = f_multi(x)
            fpx = f_multi_prime(x)
            if abs(fpx) < 1e-10:
                break
            x = x - fx / fpx
            if abs(fx) < 1e-10:
                break
        print(f"   Starting from x0={x0:5.2f} -> converges to x*={x:8.5f}")


def newton_2d_example():
    """
    2D Newton's method example: Find intersection of circle and line.

    F1(x,y) = x^2 + y^2 - 4 = 0  (circle of radius 2)
    F2(x,y) = x - y = 0          (line y = x)

    Solution: (sqrt(2), sqrt(2)) and (-sqrt(2), -sqrt(2))
    """
    print(f"\n{'='*80}")
    print("2D Newton's Method: Circle x^2 + y^2 = 4 intersects Line y = x")
    print(f"{'='*80}")

    def F(xy):
        x, y = xy
        return np.array([x**2 + y**2 - 4, x - y])

    def J(xy):
        x, y = xy
        return np.array([[2*x, 2*y],
                         [1, -1]])

    # Starting point
    xy = np.array([1.0, 0.5])
    exact_2d = np.array([np.sqrt(2), np.sqrt(2)])

    print(f"\n{'n':>3} | {'x':>12} | {'y':>12} | {'||F||':>12} | {'error':>12}")
    print(f"{'-'*60}")

    for n in range(10):
        Fxy = F(xy)
        error = np.linalg.norm(xy - exact_2d)
        print(f"{n:3d} | {xy[0]:12.8f} | {xy[1]:12.8f} | {np.linalg.norm(Fxy):12.2e} | {error:12.2e}")

        if np.linalg.norm(Fxy) < 1e-12:
            break

        Jxy = J(xy)
        # Solve J * d = -F for the Newton step d
        d = np.linalg.solve(Jxy, -Fxy)
        xy = xy + d
        time.sleep(0.2)

    print(f"{'-'*60}")
    print(f"Solution: ({xy[0]:.10f}, {xy[1]:.10f})")
    print(f"Expected: ({np.sqrt(2):.10f}, {np.sqrt(2):.10f})")


# =====================================================================
# Main execution
# =====================================================================

if __name__ == "__main__":

    print("\n" + "="*80)
    print("LECTURE 4: Root Finding & Newton Methods")
    print("="*80)

    # Run Newton-Raphson
    print("\nRunning Newton-Raphson method...")
    newton_history = newton_raphson(f, f_prime, x0_newton, tolerance, max_iterations)

    # Run Secant method
    print("Running Secant method...")
    secant_history = secant_method(f, x0_secant, x1_secant, tolerance, max_iterations)

    # Compare convergence
    print_convergence_table(newton_history, secant_history)

    # Analyze convergence rates
    analyze_convergence_rate(newton_history, "Newton-Raphson")
    analyze_convergence_rate(secant_history, "Secant")

    # Demonstrate failure modes
    demonstrate_failure_modes()

    # 2D Newton example
    newton_2d_example()

    print("\n" + "="*80)
    print("EXERCISES:")
    print("="*80)
    print("""
1. Change f(x) to x^3 - 5 to compute the cube root of 5.
   Update f_prime(x) accordingly and compare Newton vs Secant.

2. Try different starting points for the multiple-root case (f(x) = x^3).
   What do you observe about the convergence rate?

3. In the 2D example, change the starting point to (1, -1).
   Which solution does it converge to now?

4. Implement a "hybrid" method that starts with a few Secant iterations
   (to get close to the root) then switches to Newton for fast convergence.
   When would this be useful?

5. Add operation counting (like in Lecture 1's grid search) to compare
   the total computational cost of Newton vs Secant.
""")
