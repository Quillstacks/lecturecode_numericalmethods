import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Stochastic Gradient Descent and Modern Optimizers
# Compare SGD, Momentum, and Adam on a simple optimization problem

np.random.seed(42)

# =====================================================================
# Configuration Parameters - Experiment with these!
# =====================================================================

# Dataset: Find theta that minimizes sum of (x_i - theta)^2
data = np.array([1, 3, 5, 7, 9, 11, 13, 15])
optimal_theta = np.mean(data)  # Analytical solution

# Training settings
learning_rate = 0.1
batch_size = 2
num_epochs = 20
theta_init = 0.0

# Adam hyperparameters
beta1 = 0.9   # First moment decay
beta2 = 0.999  # Second moment decay
epsilon = 1e-8

# Momentum hyperparameter
momentum = 0.9

# =====================================================================


def loss(theta, x):
    """Mean squared error loss."""
    return np.mean((x - theta)**2)


def gradient(theta, x):
    """Gradient of MSE loss: d/dtheta mean((x - theta)^2) = -2*mean(x - theta)"""
    return -2 * np.mean(x - theta)


def full_gradient(theta, data):
    """Full batch gradient."""
    return gradient(theta, data)


def stochastic_gradient(theta, data, batch_size):
    """Stochastic gradient from random mini-batch."""
    batch_idx = np.random.choice(len(data), batch_size, replace=False)
    batch = data[batch_idx]
    return gradient(theta, batch), batch_idx


def sgd(theta, data, lr, batch_size, num_epochs):
    """Vanilla SGD."""
    history = {'theta': [theta], 'loss': [loss(theta, data)]}

    for epoch in range(num_epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(len(data))

        for i in range(0, len(data), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch = data[batch_idx]

            grad = gradient(theta, batch)
            theta = theta - lr * grad

            history['theta'].append(theta)
            history['loss'].append(loss(theta, data))

    return theta, history


def sgd_momentum(theta, data, lr, batch_size, num_epochs, momentum):
    """SGD with momentum."""
    history = {'theta': [theta], 'loss': [loss(theta, data)]}
    velocity = 0

    for epoch in range(num_epochs):
        indices = np.random.permutation(len(data))

        for i in range(0, len(data), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch = data[batch_idx]

            grad = gradient(theta, batch)
            velocity = momentum * velocity + grad
            theta = theta - lr * velocity

            history['theta'].append(theta)
            history['loss'].append(loss(theta, data))

    return theta, history


def adam(theta, data, lr, batch_size, num_epochs, beta1, beta2, eps):
    """Adam optimizer."""
    history = {'theta': [theta], 'loss': [loss(theta, data)]}
    m = 0  # First moment
    v = 0  # Second moment
    t = 0  # Timestep

    for epoch in range(num_epochs):
        indices = np.random.permutation(len(data))

        for i in range(0, len(data), batch_size):
            t += 1
            batch_idx = indices[i:i+batch_size]
            batch = data[batch_idx]

            grad = gradient(theta, batch)

            # Update biased first moment estimate
            m = beta1 * m + (1 - beta1) * grad
            # Update biased second moment estimate
            v = beta2 * v + (1 - beta2) * grad**2

            # Bias-corrected estimates
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)

            # Update
            theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)

            history['theta'].append(theta)
            history['loss'].append(loss(theta, data))

    return theta, history


def gd_full_batch(theta, data, lr, num_epochs):
    """Full batch gradient descent for comparison."""
    history = {'theta': [theta], 'loss': [loss(theta, data)]}

    for epoch in range(num_epochs):
        grad = full_gradient(theta, data)
        theta = theta - lr * grad

        history['theta'].append(theta)
        history['loss'].append(loss(theta, data))

    return theta, history


def main():
    print("\n" + "="*70)
    print("LECTURE 6: SGD & Modern Optimizers")
    print("="*70)

    print(f"\nDataset: {data}")
    print(f"Optimal theta (analytical): {optimal_theta}")
    print(f"Initial theta: {theta_init}")
    print(f"Learning rate: {learning_rate}, Batch size: {batch_size}")

    # Run all optimizers
    print(f"\n{'='*70}")
    print("Running Optimizers...")
    print(f"{'='*70}")

    theta_gd, hist_gd = gd_full_batch(theta_init, data, learning_rate, num_epochs)
    theta_sgd, hist_sgd = sgd(theta_init, data, learning_rate, batch_size, num_epochs)
    theta_mom, hist_mom = sgd_momentum(theta_init, data, learning_rate, batch_size, num_epochs, momentum)
    theta_adam, hist_adam = adam(theta_init, data, learning_rate, batch_size, num_epochs, beta1, beta2, epsilon)

    print(f"\n{'Method':<20} | {'Final theta':>12} | {'Final Loss':>12} | {'Updates':>10}")
    print(f"{'-'*60}")
    print(f"{'Full Batch GD':<20} | {theta_gd:12.6f} | {hist_gd['loss'][-1]:12.6f} | {len(hist_gd['theta'])-1:10d}")
    print(f"{'SGD':<20} | {theta_sgd:12.6f} | {hist_sgd['loss'][-1]:12.6f} | {len(hist_sgd['theta'])-1:10d}")
    print(f"{'SGD + Momentum':<20} | {theta_mom:12.6f} | {hist_mom['loss'][-1]:12.6f} | {len(hist_mom['theta'])-1:10d}")
    print(f"{'Adam':<20} | {theta_adam:12.6f} | {hist_adam['loss'][-1]:12.6f} | {len(hist_adam['theta'])-1:10d}")

    # Detailed SGD trace
    print(f"\n{'='*70}")
    print("Detailed SGD Trace (first 10 updates)")
    print(f"{'='*70}")

    np.random.seed(42)  # Reset for reproducibility
    theta = theta_init
    print(f"\n{'Update':>6} | {'Batch':>10} | {'Gradient':>10} | {'Theta':>10} | {'Loss':>10}")
    print(f"{'-'*55}")

    for i in range(10):
        grad, batch_idx = stochastic_gradient(theta, data, batch_size)
        batch_str = str(data[batch_idx].tolist())
        theta_old = theta
        theta = theta - learning_rate * grad
        print(f"{i+1:6d} | {batch_str:>10} | {grad:10.4f} | {theta:10.4f} | {loss(theta, data):10.4f}")
        time.sleep(0.2)

    # Variance analysis
    print(f"\n{'='*70}")
    print("Gradient Variance Analysis")
    print(f"{'='*70}")

    theta_test = 4.0  # Near optimal
    full_grad = full_gradient(theta_test, data)

    print(f"\nFull gradient at theta={theta_test}: {full_grad:.6f}")
    print(f"\nStochastic gradients (batch_size={batch_size}):")

    stoch_grads = []
    for i in range(20):
        sg, _ = stochastic_gradient(theta_test, data, batch_size)
        stoch_grads.append(sg)
        print(f"  Sample {i+1:2d}: {sg:8.4f}")

    print(f"\nMean of stochastic gradients: {np.mean(stoch_grads):.6f}")
    print(f"Variance of stochastic gradients: {np.var(stoch_grads):.6f}")
    print(f"True gradient: {full_grad:.6f}")
    print("(Mean should be close to true gradient - unbiased estimator)")

    # Save visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Loss curves
    axes[0].plot(hist_gd['loss'], 'k-', label='Full Batch GD', linewidth=2)
    axes[0].plot(hist_sgd['loss'], 'b-', label='SGD', alpha=0.7)
    axes[0].plot(hist_mom['loss'], 'g-', label='SGD + Momentum', alpha=0.7)
    axes[0].plot(hist_adam['loss'], 'r-', label='Adam', alpha=0.7)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Update Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Convergence Comparison')
    axes[0].legend()
    axes[0].set_yscale('log')

    # Plot 2: Theta trajectories
    axes[1].plot(hist_gd['theta'], 'k-', label='Full Batch GD', linewidth=2)
    axes[1].plot(hist_sgd['theta'], 'b-', label='SGD', alpha=0.7)
    axes[1].plot(hist_mom['theta'], 'g-', label='SGD + Momentum', alpha=0.7)
    axes[1].plot(hist_adam['theta'], 'r-', label='Adam', alpha=0.7)
    axes[1].axhline(y=optimal_theta, color='gray', linestyle='--', label=f'Optimal ({optimal_theta})')
    axes[1].set_xlabel('Update Step')
    axes[1].set_ylabel('Theta')
    axes[1].set_title('Parameter Trajectory')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('sgd_optimizers.png', dpi=150)
    print(f"\nVisualization saved to: sgd_optimizers.png")

    print("\n" + "="*70)
    print("EXERCISES:")
    print("="*70)
    print("""
1. Change batch_size to 1, 4, and 8. How does the variance of SGD
   updates change? How does convergence speed change?

2. Try different learning rates (0.01, 0.1, 0.5). When does SGD diverge?
   Does Adam handle larger learning rates better?

3. Implement learning rate decay: lr_t = lr_0 / (1 + decay * t).
   Compare convergence with constant vs decaying learning rate.

4. Add noise to the gradients to simulate a more realistic scenario.
   How do the optimizers handle noisy gradients?

5. Implement AdaGrad: accumulate squared gradients and divide learning
   rate by sqrt(sum of squared gradients). Compare to Adam.
""")


if __name__ == "__main__":
    main()
