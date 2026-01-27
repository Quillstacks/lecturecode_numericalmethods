import time
import numpy as np
import sys

# Set random seed for reproducibility
np.random.seed(42)

# Linear system: [2 1; 5 1][w; b] = [11; 13]
# X is a matrix of shape (2,2), Y is a vector
X = np.array([[2, 1], [5, 1]])
Y = np.array([11, 13])
a = 0   # interval start
b = 10    # interval end
w_real = 2/3
b_real = 29/3


def conditioning_analysis(X, Y, perturb):
    print("\nConditioning Analysis:")
    for p in perturb:
        K_1 = (X[0, 0] + p) * w_real + (X[0, 1] + p) * b_real - Y[0]
        K_2 = (X[1, 0] + p) * w_real + (X[1, 1] + p) * b_real - Y[1]
        mean_abs = (abs(K_1) + abs(K_2)) / 2
    print(f"Condition number with perturbation {p}: {mean_abs:.2f}")

def stability_analysis(X, w_best, b_best, perturb):
    print("\nStability Analysis:")
    mean_abs_list = []
    for p in perturb:
        s_1 = (X[0, 0] + p) * w_best + (X[0, 1] + p) * b_best - ((X[0, 0]) * w_best + (X[0, 1]) * b_best)
        s_2 = (X[1, 0] + p) * w_best + (X[1, 1] + p) * b_best - ((X[1, 0]) * w_best + (X[1, 1]) * b_best)
        mean_abs = (abs(s_1) + abs(s_2)) / 2   
        mean_abs_list.append(mean_abs)
    mean_of_means = np.mean(mean_abs_list)
    std_of_means = np.std(mean_abs_list)
    print(f"Mean of mean_abs: {mean_of_means:.4f}, Std: {std_of_means:.4f}")

def consistency_analysis(X, Y, w_best, b_best):
    # Pay attention to how this is implemented, we know this metric very well from validating ML models.
    print("\nConsistency Analysis:")
    y_pred = X @ np.array([w_best, b_best])
    residuals = Y - y_pred
    mean_residual = np.mean(residuals)
    print(f"Mean of residuals of best fit: {mean_residual:.2f}")
    # Is there a better way to measure consistency? Another loss function? 

def convergence_analysis(X, Y, w_best, b_best, perturb):
    print("\nConvergence Analysis:")
    mean_abs_list = []
    for p in perturb:
        y_perturbed = (X + p) @ np.array([w_best, b_best])
        residuals = Y - y_perturbed
        mean_abs = np.mean(np.abs(residuals))
        mean_abs_list.append(mean_abs)
    mean_of_means = np.mean(mean_abs_list)
    std_of_means = np.std(mean_abs_list)
    print(f"Mean of mean_abs: {mean_of_means:.4f}, Std: {std_of_means:.4f}")


def brute_force(X, Y, a, b, h):
    total_additions = 0
    total_multiplications = 0
    operation_count = 0
    best_diff = float('inf')
    best_w = None
    best_b = None

    w_range = np.arange(a, b + h, h)
    b_range = np.arange(a, b + h, h)
    total_steps = len(w_range) * len(b_range)
    progress_bar_len = 40

    def print_progress(progress, total):
        percent = progress / total
        filled = int(progress_bar_len * percent)
        bar = '=' * filled + '-' * (progress_bar_len - filled)
        print(f"\rProgress: [{bar}] {progress}/{total}", end='', flush=True)

    print()  # Newline before progress bar

    for w in w_range:
        for b_ in b_range:
            operation_count += 1
            y_pred = X @ np.array([w, b_])
            diff = np.mean(np.abs(Y - y_pred))
            if diff < best_diff:
                best_diff = diff
                best_w = w
                best_b = b_
            total_multiplications += X.size
            total_additions += X.size - X.shape[0]
            print_progress(operation_count, total_steps)

    print()  # Newline after progress bar
    print(f"Best w: {best_w}, Best b: {best_b}, Error: {best_diff}")
    return best_w, best_b






if __name__ == "__main__":

    # ====================================================================
    # Configuration Parameters 

    h = 5    # step size
    perturb = np.array([-0.5, 0.5])  # small perturbation to get \tilde{x}
    

    # ====================================================================          
    # It is easiest to comment out analysis runs, but the one you want to 
    # explore.
    # ====================================================================

    conditioning_analysis(X, Y, perturb)

    w_best, b_best = brute_force(X, Y, a, b, h)  
    # Note that we brute force on x and not on \tilde{x}
    # print(f"w_real: {w_real}, b_real: {b_real}")

    stability_analysis(X, w_best, b_best, perturb)  

    consistency_analysis(X, Y, w_best, b_best)

    convergence_analysis(X, Y, w_best, b_best, perturb)


    # This shows that this single numerical solution converges. 
    # Can you also show that the method of brute forcing 
    # itself converges as h -> 0?

    # ===================================================================