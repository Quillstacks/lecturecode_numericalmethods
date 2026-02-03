import time
import numpy as np

# Numerical Stability in Deep Learning
# Demonstrate log-sum-exp trick, stable softmax, and gradient clipping

np.random.seed(42)

# =====================================================================
# Configuration Parameters - Experiment with these!
# =====================================================================

# For softmax/log-sum-exp demonstrations
large_values = np.array([1000, 1001, 1002])
small_values = np.array([-1000, -1001, -1002])
normal_values = np.array([1.0, 2.0, 3.0])

# Gradient clipping threshold
clip_threshold = 1.0

# Weight initialization demo
layer_sizes = [784, 256, 128, 64, 10]  # MLP architecture

# =====================================================================


def naive_softmax(x):
    """Naive softmax - will overflow for large x."""
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def stable_softmax(x):
    """Stable softmax - subtract max before exp."""
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)


def naive_logsumexp(x):
    """Naive log-sum-exp - will overflow for large x."""
    return np.log(np.sum(np.exp(x)))


def stable_logsumexp(x):
    """Stable log-sum-exp using the max trick."""
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(x - x_max)))


def clip_by_value(gradient, threshold):
    """Clip gradient element-wise."""
    return np.clip(gradient, -threshold, threshold)


def clip_by_norm(gradient, threshold):
    """Clip gradient by global norm (preserves direction)."""
    norm = np.linalg.norm(gradient)
    if norm > threshold:
        return gradient * (threshold / norm)
    return gradient


def xavier_init(fan_in, fan_out):
    """Xavier/Glorot initialization for tanh/sigmoid."""
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(fan_in, fan_out) * std


def he_init(fan_in, fan_out):
    """He/Kaiming initialization for ReLU."""
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out) * std


def random_init(fan_in, fan_out, scale=1.0):
    """Random initialization (often bad)."""
    return np.random.randn(fan_in, fan_out) * scale


def forward_pass_variance(layer_sizes, init_fn, activation='relu'):
    """Track variance through a forward pass."""
    x = np.random.randn(1, layer_sizes[0])  # Input with unit variance
    variances = [np.var(x)]

    for i in range(len(layer_sizes) - 1):
        W = init_fn(layer_sizes[i], layer_sizes[i + 1])
        x = x @ W

        if activation == 'relu':
            x = np.maximum(0, x)
        elif activation == 'tanh':
            x = np.tanh(x)

        variances.append(np.var(x))

    return variances


def main():
    print("\n" + "="*70)
    print("LECTURE 7: Numerical Stability")
    print("="*70)

    # Part 1: Softmax stability
    print(f"\n{'='*70}")
    print("Part 1: Stable Softmax")
    print(f"{'='*70}")

    print("\n--- Normal values:", normal_values)
    print(f"Naive softmax:  {naive_softmax(normal_values)}")
    print(f"Stable softmax: {stable_softmax(normal_values)}")

    print("\n--- Large values:", large_values)
    try:
        naive_result = naive_softmax(large_values)
        print(f"Naive softmax:  {naive_result}")
    except Exception as e:
        print(f"Naive softmax:  OVERFLOW! (returns nan/inf)")

    with np.errstate(over='ignore', invalid='ignore'):
        naive_result = naive_softmax(large_values)
        print(f"Naive softmax:  {naive_result}")

    print(f"Stable softmax: {stable_softmax(large_values)}")

    print("\n--- Small values:", small_values)
    with np.errstate(divide='ignore', invalid='ignore'):
        naive_result = naive_softmax(small_values)
        print(f"Naive softmax:  {naive_result} (underflow to 0, then 0/0)")

    print(f"Stable softmax: {stable_softmax(small_values)}")

    # Part 2: Log-sum-exp stability
    print(f"\n{'='*70}")
    print("Part 2: Stable Log-Sum-Exp")
    print(f"{'='*70}")

    print("\nlog(sum(exp(x))) is used in:")
    print("  - Cross-entropy loss")
    print("  - Attention scores in Transformers")
    print("  - Log-likelihood computations")

    print("\n--- Normal values:", normal_values)
    print(f"Naive log-sum-exp:  {naive_logsumexp(normal_values):.6f}")
    print(f"Stable log-sum-exp: {stable_logsumexp(normal_values):.6f}")

    print("\n--- Large values:", large_values)
    with np.errstate(over='ignore'):
        naive_lse = naive_logsumexp(large_values)
        print(f"Naive log-sum-exp:  {naive_lse}")

    print(f"Stable log-sum-exp: {stable_logsumexp(large_values):.6f}")
    print("(Note: The answer should be close to max(x) = 1002)")

    # Part 3: Gradient clipping
    print(f"\n{'='*70}")
    print("Part 3: Gradient Clipping")
    print(f"{'='*70}")

    # Simulated exploding gradient
    gradient = np.array([10.0, -50.0, 100.0, -5.0])
    print(f"\nOriginal gradient: {gradient}")
    print(f"Gradient norm: {np.linalg.norm(gradient):.4f}")
    print(f"Clip threshold: {clip_threshold}")

    clipped_value = clip_by_value(gradient, clip_threshold)
    clipped_norm = clip_by_norm(gradient, clip_threshold)

    print(f"\nClip by value: {clipped_value}")
    print(f"  (Each element clipped to [-{clip_threshold}, {clip_threshold}])")
    print(f"  Direction preserved? No - gradient direction changed")

    print(f"\nClip by norm: {clipped_norm}")
    print(f"  Clipped norm: {np.linalg.norm(clipped_norm):.4f}")
    print(f"  Direction preserved? Yes - only magnitude scaled")

    # Part 4: Weight initialization
    print(f"\n{'='*70}")
    print("Part 4: Weight Initialization")
    print(f"{'='*70}")

    print(f"\nNetwork architecture: {layer_sizes}")
    print("Tracking variance through forward pass...")

    print(f"\n{'Layer':<10} | {'Random (1.0)':<15} | {'Xavier':<15} | {'He (ReLU)':<15}")
    print(f"{'-'*60}")

    var_random = forward_pass_variance(layer_sizes, lambda fi, fo: random_init(fi, fo, 1.0), 'relu')
    var_xavier = forward_pass_variance(layer_sizes, xavier_init, 'relu')
    var_he = forward_pass_variance(layer_sizes, he_init, 'relu')

    for i, (vr, vx, vh) in enumerate(zip(var_random, var_xavier, var_he)):
        layer_name = "Input" if i == 0 else f"Layer {i}"
        print(f"{layer_name:<10} | {vr:15.6f} | {vx:15.6f} | {vh:15.6f}")
        time.sleep(0.2)

    print("\nObservation:")
    print("  - Random init: variance explodes or vanishes")
    print("  - Xavier init: designed for tanh/sigmoid, okay for ReLU")
    print("  - He init: designed for ReLU, maintains variance better")

    # Part 5: Batch normalization epsilon
    print(f"\n{'='*70}")
    print("Part 5: BatchNorm Epsilon")
    print(f"{'='*70}")

    x = np.array([1.0, 1.0, 1.0, 1.0])  # Constant activations
    mean = np.mean(x)
    var = np.var(x)

    print(f"\nActivations: {x}")
    print(f"Mean: {mean}, Variance: {var}")

    print("\nNormalization: (x - mean) / sqrt(var + epsilon)")

    for eps in [0, 1e-8, 1e-5, 1e-3]:
        if eps == 0 and var == 0:
            print(f"  epsilon = {eps}: Division by zero!")
        else:
            normalized = (x - mean) / np.sqrt(var + eps)
            print(f"  epsilon = {eps}: {normalized}")

    print("\nWhy epsilon matters: when variance is near zero (constant activations),")
    print("we need epsilon to prevent division by zero.")

    print("\n" + "="*70)
    print("EXERCISES:")
    print("="*70)
    print("""
1. Implement stable cross-entropy: -log(softmax(x)[correct_class])
   using log-sum-exp. Compare to naive implementation.

2. Simulate training with exploding gradients: start with gradient = 1,
   multiply by 2 each step. At which step does it overflow in float32?
   Add gradient clipping and observe the difference.

3. Compare Xavier vs He initialization with tanh activation instead of ReLU.
   Which maintains variance better now?

4. Implement "loss scaling" for mixed precision: multiply loss by 1024
   before backward pass, divide gradients by 1024 after. Why does this help?

5. Create a scenario where BatchNorm's running mean/variance becomes
   problematic during inference (hint: train on one distribution, test
   on another).
""")


if __name__ == "__main__":
    main()
