import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ODE Solvers: Euler and Runge-Kutta Methods
# Simulating dynamic systems and connection to ResNets

np.random.seed(42)

# =====================================================================
# Configuration Parameters - Experiment with these!
# =====================================================================

# Test ODE: dy/dt = -y, y(0) = 1
# Exact solution: y(t) = exp(-t)

def f_decay(t, y):
    """Simple exponential decay: dy/dt = -y"""
    return -y

def exact_decay(t):
    """Exact solution: y = exp(-t)"""
    return np.exp(-t)

# Another test: dy/dt = -2ty, y(0) = 1
# Exact solution: y(t) = exp(-t^2)

def f_gaussian(t, y):
    """Gaussian decay: dy/dt = -2ty"""
    return -2 * t * y

def exact_gaussian(t):
    """Exact solution: y = exp(-t^2)"""
    return np.exp(-t**2)

# Time domain
t_start = 0
t_end = 5
y0 = 1.0

# Step sizes to compare
step_sizes = [1.0, 0.5, 0.25, 0.1]

# =====================================================================


def euler_method(f, y0, t_start, t_end, h):
    """
    Euler's method: y_{n+1} = y_n + h * f(t_n, y_n)
    Returns: t_values, y_values
    """
    t_values = [t_start]
    y_values = [y0]

    t = t_start
    y = y0

    while t < t_end - 1e-10:
        y = y + h * f(t, y)
        t = t + h
        t_values.append(t)
        y_values.append(y)

    return np.array(t_values), np.array(y_values)


def rk4_method(f, y0, t_start, t_end, h):
    """
    Classic 4th-order Runge-Kutta method.
    Returns: t_values, y_values
    """
    t_values = [t_start]
    y_values = [y0]

    t = t_start
    y = y0

    while t < t_end - 1e-10:
        k1 = f(t, y)
        k2 = f(t + h/2, y + h*k1/2)
        k3 = f(t + h/2, y + h*k2/2)
        k4 = f(t + h, y + h*k3)

        y = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        t = t + h
        t_values.append(t)
        y_values.append(y)

    return np.array(t_values), np.array(y_values)


def midpoint_method(f, y0, t_start, t_end, h):
    """
    Midpoint method (RK2): y_{n+1} = y_n + h * f(t_n + h/2, y_n + h*f(t_n,y_n)/2)
    """
    t_values = [t_start]
    y_values = [y0]

    t = t_start
    y = y0

    while t < t_end - 1e-10:
        k1 = f(t, y)
        k2 = f(t + h/2, y + h*k1/2)
        y = y + h * k2
        t = t + h
        t_values.append(t)
        y_values.append(y)

    return np.array(t_values), np.array(y_values)


def analyze_stability(f, exact, method_name, method_func, h_values):
    """Analyze error vs step size for a method."""
    print(f"\n{method_name}:")
    print(f"{'h':>8} | {'Final y':>12} | {'Exact':>12} | {'Error':>12} | {'Error/h^p':>12}")
    print(f"{'-'*65}")

    errors = []
    for h in h_values:
        t_vals, y_vals = method_func(f, y0, t_start, t_end, h)
        exact_final = exact(t_end)
        error = abs(y_vals[-1] - exact_final)
        errors.append(error)

        # Estimate order
        if method_name == "Euler":
            error_ratio = error / h if h > 0 else 0
        else:  # RK4
            error_ratio = error / (h**4) if h > 0 else 0

        print(f"{h:8.4f} | {y_vals[-1]:12.6f} | {exact_final:12.6f} | {error:12.2e} | {error_ratio:12.4f}")

    return errors


def main():
    print("\n" + "="*70)
    print("LECTURE 13: ODE Solvers (Euler & Runge-Kutta)")
    print("="*70)

    # Part 1: Euler method by hand
    print(f"\n{'='*70}")
    print("Part 1: Euler Method by Hand")
    print(f"ODE: dy/dt = -y, y(0) = 1")
    print(f"Exact solution: y(t) = exp(-t)")
    print(f"{'='*70}")

    h = 0.5
    print(f"\nStep size h = {h}")
    print(f"\n{'n':>3} | {'t_n':>8} | {'y_n (Euler)':>14} | {'y(t) exact':>14} | {'Error':>12}")
    print(f"{'-'*60}")

    t, y = 0, 1.0
    for n in range(5):
        exact_val = exact_decay(t)
        error = abs(y - exact_val)
        print(f"{n:3d} | {t:8.2f} | {y:14.6f} | {exact_val:14.6f} | {error:12.2e}")

        # Euler update
        y = y + h * f_decay(t, y)
        t = t + h
        time.sleep(0.2)

    print("\nEuler formula: y_{n+1} = y_n + h * f(t_n, y_n)")
    print(f"              = y_n + {h} * (-y_n)")
    print(f"              = y_n * (1 - {h})")
    print(f"              = y_n * {1 - h}")

    # Part 2: RK4 derivation
    print(f"\n{'='*70}")
    print("Part 2: Runge-Kutta 4th Order (RK4)")
    print(f"{'='*70}")

    print("\nRK4 samples the slope at 4 points:")
    print("  k1 = f(t_n, y_n)                    -- slope at start")
    print("  k2 = f(t_n + h/2, y_n + h*k1/2)     -- slope at midpoint (using k1)")
    print("  k3 = f(t_n + h/2, y_n + h*k2/2)     -- slope at midpoint (using k2)")
    print("  k4 = f(t_n + h, y_n + h*k3)         -- slope at end")
    print("\nWeighted average: y_{n+1} = y_n + h*(k1 + 2*k2 + 2*k3 + k4)/6")

    # One RK4 step example
    t, y = 0, 1.0
    h = 0.5
    k1 = f_decay(t, y)
    k2 = f_decay(t + h/2, y + h*k1/2)
    k3 = f_decay(t + h/2, y + h*k2/2)
    k4 = f_decay(t + h, y + h*k3)

    print(f"\nExample: t=0, y=1, h={h}")
    print(f"  k1 = f(0, 1) = -1 = {k1:.4f}")
    print(f"  k2 = f(0.25, 1 + 0.5*(-1)/2) = f(0.25, 0.75) = -0.75 = {k2:.4f}")
    print(f"  k3 = f(0.25, 1 + 0.5*(-0.75)/2) = f(0.25, 0.8125) = {k3:.4f}")
    print(f"  k4 = f(0.5, 1 + 0.5*{k3:.4f}) = {k4:.4f}")
    print(f"  y_1 = 1 + 0.5*({k1:.4f} + 2*{k2:.4f} + 2*{k3:.4f} + {k4:.4f})/6")
    y_rk4 = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6
    print(f"      = {y_rk4:.6f}")
    print(f"  Exact y(0.5) = exp(-0.5) = {exact_decay(0.5):.6f}")
    print(f"  Error: {abs(y_rk4 - exact_decay(0.5)):.2e}")

    # Part 3: Convergence comparison
    print(f"\n{'='*70}")
    print("Part 3: Convergence Analysis (Error vs Step Size)")
    print(f"{'='*70}")

    errors_euler = analyze_stability(f_decay, exact_decay, "Euler (O(h))", euler_method, step_sizes)
    errors_rk4 = analyze_stability(f_decay, exact_decay, "RK4 (O(h^4))", rk4_method, step_sizes)

    print("\nObservation:")
    print("  Euler: halving h → halves the error (linear convergence)")
    print("  RK4: halving h → reduces error by 16x (4th order convergence)")

    # Part 4: ResNet connection
    print(f"\n{'='*70}")
    print("Part 4: Connection to ResNets")
    print(f"{'='*70}")

    print("\nResNet layer: h_{l+1} = h_l + f(h_l)")
    print("Euler step:   y_{n+1} = y_n + h * f(t_n, y_n)")
    print("\nThey are THE SAME! (with h=1 and no explicit time dependence)")
    print("\nImplications:")
    print("  - ResNets ≈ Euler discretization of a continuous ODE")
    print("  - Neural ODEs: solve dh/dt = f(h,t) with adaptive step size")
    print("  - Diffusion models: solve reverse SDEs numerically")
    print("  - Better ODE solvers → fewer function evaluations → faster sampling")

    # Part 5: Stability demonstration
    print(f"\n{'='*70}")
    print("Part 5: Stability (Stiff Equation)")
    print(f"ODE: dy/dt = -100*y (fast decay)")
    print(f"{'='*70}")

    def f_stiff(t, y):
        return -100 * y

    def exact_stiff(t):
        return np.exp(-100 * t)

    print("\nEuler with h=0.025 (stable: h*lambda = -2.5):")
    t_euler, y_euler = euler_method(f_stiff, 1.0, 0, 0.1, 0.025)
    for t, y in zip(t_euler[:5], y_euler[:5]):
        print(f"  t={t:.3f}: y={y:.6f}, exact={exact_stiff(t):.6f}")

    print("\nEuler with h=0.025 is stable for this problem.")
    print("Stability requires |1 + h*lambda| < 1, i.e., h < 2/|lambda|")
    print(f"For lambda=-100: h < 0.02")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Euler vs RK4 vs Exact
    t_exact = np.linspace(t_start, t_end, 200)
    y_exact = exact_decay(t_exact)

    t_e, y_e = euler_method(f_decay, y0, t_start, t_end, 0.5)
    t_rk, y_rk = rk4_method(f_decay, y0, t_start, t_end, 0.5)

    axes[0, 0].plot(t_exact, y_exact, 'k-', linewidth=2, label='Exact')
    axes[0, 0].plot(t_e, y_e, 'ro-', label='Euler (h=0.5)')
    axes[0, 0].plot(t_rk, y_rk, 'bs-', label='RK4 (h=0.5)')
    axes[0, 0].set_xlabel('t')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_title("dy/dt = -y: Euler vs RK4")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Convergence (error vs h)
    axes[0, 1].loglog(step_sizes, errors_euler, 'ro-', label='Euler')
    axes[0, 1].loglog(step_sizes, errors_rk4, 'bs-', label='RK4')
    axes[0, 1].loglog(step_sizes, [h for h in step_sizes], 'r--', alpha=0.5, label='O(h)')
    axes[0, 1].loglog(step_sizes, [h**4 for h in step_sizes], 'b--', alpha=0.5, label='O(h^4)')
    axes[0, 1].set_xlabel('Step size h')
    axes[0, 1].set_ylabel('Error at t=5')
    axes[0, 1].set_title('Convergence Order')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Multiple step sizes (Euler)
    for h in [1.0, 0.5, 0.25]:
        t_e, y_e = euler_method(f_decay, y0, t_start, t_end, h)
        axes[1, 0].plot(t_e, y_e, 'o-', label=f'h={h}')
    axes[1, 0].plot(t_exact, y_exact, 'k-', linewidth=2, label='Exact')
    axes[1, 0].set_xlabel('t')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_title('Euler: Effect of Step Size')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: ResNet visualization
    # Simulate ResNet as Euler on a toy transformation
    np.random.seed(42)
    x = np.array([1.0, 0.5])  # 2D input
    trajectory = [x.copy()]

    for layer in range(10):
        # ResNet: x = x + f(x) where f is some nonlinear transformation
        # Simulating with a simple rotation + scaling
        W = np.array([[0.9, -0.1], [0.1, 0.9]])
        f_x = (W @ x) * 0.1 - x * 0.05
        x = x + f_x  # Euler step with h=1
        trajectory.append(x.copy())

    trajectory = np.array(trajectory)
    axes[1, 1].plot(trajectory[:, 0], trajectory[:, 1], 'bo-')
    axes[1, 1].plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Input')
    axes[1, 1].plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', markersize=15, label='Output')
    axes[1, 1].set_xlabel('Dimension 1')
    axes[1, 1].set_ylabel('Dimension 2')
    axes[1, 1].set_title('ResNet as Euler Integration')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig('ode_solvers.png', dpi=150)
    print(f"\nVisualization saved to: ode_solvers.png")

    print("\n" + "="*70)
    print("EXERCISES:")
    print("="*70)
    print("""
1. Implement the implicit Euler method: y_{n+1} = y_n + h*f(t_{n+1}, y_{n+1}).
   For linear ODEs, this requires solving a linear equation each step.

2. Solve the pendulum equation: d^2(theta)/dt^2 = -(g/L)*sin(theta).
   Convert to a system of two first-order ODEs and solve with RK4.

3. Compare Euler and RK4 on dy/dt = y^2 - 1 (which has finite-time blowup).
   How do the methods behave near the singularity?

4. Implement adaptive step size control: estimate error by comparing
   two methods, adjust h to keep error below tolerance.

5. Implement a simple Neural ODE: define f as a small neural network,
   solve the ODE, and backpropagate through the solver.
""")


if __name__ == "__main__":
    main()
