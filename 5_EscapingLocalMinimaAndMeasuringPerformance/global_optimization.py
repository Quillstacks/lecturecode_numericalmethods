import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Global Optimization and Complexity Analysis
# Explore multimodal landscapes, random restarts, and Big-O complexity

np.random.seed(42)

# =====================================================================
# Configuration Parameters - Experiment with these!
# =====================================================================

# Multimodal function: f(x) = sin(5x) + 0.1*x^2
def f(x):
    return np.sin(5*x) + 0.1*x**2

def f_prime(x):
    return 5*np.cos(5*x) + 0.2*x

# Search domain
x_min, x_max = -3, 3

# Gradient descent settings
learning_rate = 0.05
max_iterations = 100
tolerance = 1e-6

# Random restarts
num_restarts = 10

# =====================================================================


def gradient_descent(f, f_prime, x0, lr, max_iter, tol):
    """
    Simple gradient descent for minimization.
    Returns: final x, final f(x), trajectory
    """
    x = x0
    trajectory = [x]

    for _ in range(max_iter):
        grad = f_prime(x)
        x_new = x - lr * grad
        trajectory.append(x_new)

        if abs(x_new - x) < tol:
            break
        x = x_new

    return x, f(x), trajectory


def random_restarts(f, f_prime, x_min, x_max, num_starts, lr, max_iter, tol):
    """
    Run gradient descent from multiple random starting points.
    Returns: best x, best f(x), all results
    """
    results = []

    for i in range(num_starts):
        x0 = np.random.uniform(x_min, x_max)
        x_final, f_final, traj = gradient_descent(f, f_prime, x0, lr, max_iter, tol)
        results.append({
            'start': x0,
            'final': x_final,
            'value': f_final,
            'trajectory': traj
        })

    # Find best result
    best = min(results, key=lambda r: r['value'])
    return best, results


def shekel_function(x, a, c):
    """
    Shekel's foxholes in 1D: f(x) = -sum(1 / ((x - a_i)^2 + c_i))
    """
    total = 0
    for ai, ci in zip(a, c):
        total += 1 / ((x - ai)**2 + ci)
    return -total


def complexity_demo():
    """
    Demonstrate Big-O complexity with different algorithms.
    """
    print(f"\n{'='*70}")
    print("Complexity Analysis: Comparing Algorithm Scaling")
    print(f"{'='*70}")

    sizes = [10, 100, 1000, 10000]

    print(f"\n{'n':>8} | {'O(1)':>10} | {'O(log n)':>10} | {'O(n)':>10} | {'O(n^2)':>12}")
    print(f"{'-'*60}")

    for n in sizes:
        o_1 = 1
        o_logn = np.log2(n)
        o_n = n
        o_n2 = n**2
        print(f"{n:8d} | {o_1:10.0f} | {o_logn:10.1f} | {o_n:10.0f} | {o_n2:12.0f}")

    print(f"\n{'='*70}")
    print("Convergence Rate Comparison")
    print(f"{'='*70}")

    # Simulate convergence rates
    print(f"\n{'Iter':>6} | {'Linear (0.5)':>14} | {'Superlinear (1.6)':>18} | {'Quadratic':>14}")
    print(f"{'-'*60}")

    e_linear = 1.0
    e_super = 1.0
    e_quad = 1.0

    for i in range(8):
        print(f"{i:6d} | {e_linear:14.2e} | {e_super:18.2e} | {e_quad:14.2e}")
        e_linear *= 0.5           # Linear: error halves
        e_super = e_super ** 1.618  # Superlinear (golden ratio)
        e_quad = e_quad ** 2       # Quadratic: error squares

        if e_quad < 1e-16:
            e_quad = 0


def main():
    print("\n" + "="*70)
    print("LECTURE 5: Global Optimization & Complexity")
    print("="*70)

    # Part 1: Demonstrate local minima problem
    print(f"\n{'='*70}")
    print("Part 1: Local vs Global Minima")
    print(f"Function: f(x) = sin(5x) + 0.1*x^2 on [{x_min}, {x_max}]")
    print(f"{'='*70}")

    # Single gradient descent run
    x0 = 2.0
    x_final, f_final, traj = gradient_descent(f, f_prime, x0, learning_rate, max_iterations, tolerance)
    print(f"\nSingle run from x0={x0}:")
    print(f"  Final x = {x_final:.6f}, f(x) = {f_final:.6f}")

    # Part 2: Random restarts
    print(f"\n{'='*70}")
    print(f"Part 2: Random Restarts ({num_restarts} trials)")
    print(f"{'='*70}")

    best, all_results = random_restarts(f, f_prime, x_min, x_max, num_restarts,
                                         learning_rate, max_iterations, tolerance)

    print(f"\n{'#':>3} | {'Start':>10} | {'Final x':>10} | {'f(x)':>10}")
    print(f"{'-'*45}")

    for i, r in enumerate(all_results):
        marker = " <-- BEST" if r == best else ""
        print(f"{i+1:3d} | {r['start']:10.4f} | {r['final']:10.4f} | {r['value']:10.4f}{marker}")

    print(f"\nBest solution found: x* = {best['final']:.6f}, f(x*) = {best['value']:.6f}")

    # Part 3: Shekel's Foxholes
    print(f"\n{'='*70}")
    print("Part 3: Shekel's Foxholes (Controlled Multimodality)")
    print(f"{'='*70}")

    # Define 5 foxholes with different depths
    a = [-2.0, -0.5, 0.5, 1.5, 2.5]  # Hole locations
    c = [0.2, 0.1, 0.05, 0.15, 0.25]   # Hole depths (smaller = deeper)

    print(f"\nFoxhole locations: {a}")
    print(f"Foxhole depths (c): {c}")
    print("(Smaller c = deeper hole)")

    # Evaluate at each hole
    print(f"\n{'Location':>10} | {'f(x)':>12} | {'Depth Rank':>12}")
    print(f"{'-'*40}")

    values = [(ai, shekel_function(ai, a, c)) for ai in a]
    values.sort(key=lambda x: x[1])

    for rank, (ai, fi) in enumerate(values, 1):
        print(f"{ai:10.2f} | {fi:12.4f} | {rank:12d}")

    print(f"\nDeepest hole (global min) at x = {values[0][0]}")

    # Part 4: Complexity analysis
    complexity_demo()

    # Save visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Multimodal function with trajectories
    x_plot = np.linspace(x_min, x_max, 500)
    axes[0].plot(x_plot, f(x_plot), 'k-', linewidth=2)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f(x)')
    axes[0].set_title('Random Restarts on Multimodal Function')

    colors = plt.cm.viridis(np.linspace(0, 1, len(all_results)))
    for r, col in zip(all_results, colors):
        traj = np.array(r['trajectory'])
        axes[0].plot(traj, f(traj), 'o-', color=col, markersize=3, alpha=0.7)
        axes[0].plot(r['start'], f(r['start']), 's', color=col, markersize=8)

    axes[0].plot(best['final'], best['value'], 'r*', markersize=15, label='Best found')
    axes[0].legend()

    # Plot 2: Shekel's foxholes
    x_shekel = np.linspace(-3, 4, 500)
    y_shekel = [shekel_function(xi, a, c) for xi in x_shekel]
    axes[1].plot(x_shekel, y_shekel, 'k-', linewidth=2)
    axes[1].scatter(a, [shekel_function(ai, a, c) for ai in a], c='red', s=100, zorder=5)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('f(x)')
    axes[1].set_title("Shekel's Foxholes")

    plt.tight_layout()
    plt.savefig('global_optimization.png', dpi=150)
    print(f"\nVisualization saved to: global_optimization.png")

    print("\n" + "="*70)
    print("EXERCISES:")
    print("="*70)
    print("""
1. Increase num_restarts to 50 and 100. How does the probability of
   finding the global minimum change?

2. Modify the Shekel function to 2D: f(x,y) = -sum(1/((x-a_i)^2 + (y-b_i)^2 + c_i))
   How does the difficulty of finding the global minimum scale?

3. Implement simulated annealing: accept worse solutions with probability
   exp(-delta/T) where T decreases over time.

4. Plot wall-clock time vs problem size for O(n) and O(n^2) algorithms.
   At what n does O(n^2) become impractical?

5. Compare convergence: run GD for 1000 iterations vs Newton for 10.
   Count total function/gradient evaluations for each.
""")


if __name__ == "__main__":
    main()
