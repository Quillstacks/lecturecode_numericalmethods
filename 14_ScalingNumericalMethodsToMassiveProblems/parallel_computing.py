import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

# Parallel and Distributed Computing
# Understanding GPU parallelism, data parallelism, and scaling

np.random.seed(42)

# =====================================================================
# Configuration Parameters - Experiment with these!
# =====================================================================

# Matrix sizes for benchmarking
matrix_sizes = [100, 500, 1000, 2000]

# Batch sizes for data parallelism demo
batch_sizes = [32, 64, 128, 256]

# Number of "GPUs" to simulate
num_workers = 4

# =====================================================================


def matrix_multiply_sequential(A, B):
    """Sequential matrix multiplication."""
    return A @ B


def matrix_multiply_blocked(A, B, block_size=64):
    """
    Blocked matrix multiplication (cache-friendly).
    This is how GPUs organize work!
    """
    n = A.shape[0]
    C = np.zeros((n, n))

    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            for k in range(0, n, block_size):
                # Compute block C[i:i+bs, j:j+bs]
                i_end = min(i + block_size, n)
                j_end = min(j + block_size, n)
                k_end = min(k + block_size, n)

                C[i:i_end, j:j_end] += A[i:i_end, k:k_end] @ B[k:k_end, j:j_end]

    return C


def simulate_gpu_kernel(data, kernel_fn, num_threads=256):
    """
    Simulate GPU-style SIMD execution.
    Same operation applied to all elements in parallel.
    """
    # In reality, GPUs have thousands of threads
    # Here we simulate the concept
    results = np.zeros_like(data)
    chunk_size = max(1, len(data) // num_threads)

    for i in range(0, len(data), chunk_size):
        end = min(i + chunk_size, len(data))
        results[i:end] = kernel_fn(data[i:end])

    return results


def compute_gradient(model_params, batch_data):
    """Simulate gradient computation for one batch."""
    # Simulate some computation
    gradient = np.random.randn(*model_params.shape) * 0.01
    time.sleep(0.01)  # Simulate computation time
    return gradient


def data_parallel_training(model_params, data, num_gpus, batch_size):
    """
    Simulate data-parallel training across multiple GPUs.

    1. Split batch across GPUs
    2. Each GPU computes local gradient
    3. AllReduce: average gradients
    4. Update model (same on all GPUs)
    """
    samples_per_gpu = batch_size // num_gpus
    gradients = []

    # Simulate parallel gradient computation
    print(f"\n    Batch size: {batch_size}, GPUs: {num_gpus}, Samples/GPU: {samples_per_gpu}")

    for gpu_id in range(num_gpus):
        start_idx = gpu_id * samples_per_gpu
        end_idx = start_idx + samples_per_gpu
        batch = data[start_idx:end_idx]

        grad = compute_gradient(model_params, batch)
        gradients.append(grad)
        print(f"    GPU {gpu_id}: computed gradient (norm={np.linalg.norm(grad):.4f})")

    # AllReduce: average gradients
    avg_gradient = np.mean(gradients, axis=0)
    print(f"    AllReduce: averaged {num_gpus} gradients")

    # Update model
    learning_rate = 0.01
    model_params = model_params - learning_rate * avg_gradient

    return model_params


def analyze_scaling():
    """Analyze strong and weak scaling."""
    print(f"\n{'='*70}")
    print("Scaling Analysis")
    print(f"{'='*70}")

    print("\nStrong Scaling: Fixed problem size, increase workers")
    print("Ideal: Time = T_1 / P (where P = number of processors)")
    print()

    fixed_work = 1000000  # Fixed amount of work
    print(f"{'Workers':>8} | {'Work/Worker':>14} | {'Ideal Speedup':>14} | {'Comm Overhead':>14}")
    print(f"{'-'*55}")

    for p in [1, 2, 4, 8, 16]:
        work_per_worker = fixed_work // p
        ideal_speedup = p
        # Communication overhead grows with number of workers
        comm_overhead = np.log2(p) * 1000 if p > 1 else 0
        print(f"{p:8d} | {work_per_worker:14d} | {ideal_speedup:14.1f}x | {comm_overhead:14.0f}")

    print("\nWeak Scaling: Work/worker is constant, increase total work")
    print("Ideal: Time stays constant as we add workers")
    print()

    work_per_worker = 100000
    print(f"{'Workers':>8} | {'Total Work':>14} | {'Ideal Time':>14} | {'With Comm':>14}")
    print(f"{'-'*55}")

    base_time = 1.0
    for p in [1, 2, 4, 8, 16]:
        total_work = work_per_worker * p
        ideal_time = base_time
        # Communication adds overhead
        actual_time = base_time + np.log2(p) * 0.1 if p > 1 else base_time
        print(f"{p:8d} | {total_work:14d} | {ideal_time:14.2f}s | {actual_time:14.2f}s")


def simulate_pipeline_parallelism():
    """Demonstrate pipeline parallelism for large models."""
    print(f"\n{'='*70}")
    print("Pipeline Parallelism (for models too large for one GPU)")
    print(f"{'='*70}")

    num_stages = 4
    num_microbatches = 8

    print(f"\nModel split across {num_stages} GPUs (pipeline stages)")
    print(f"Batch split into {num_microbatches} microbatches")
    print("\nPipeline schedule (F=forward, B=backward):")
    print()

    # Create pipeline schedule
    schedule = [[' . ' for _ in range(num_microbatches * 2)] for _ in range(num_stages)]

    # Forward passes
    for mb in range(num_microbatches):
        for stage in range(num_stages):
            time_slot = mb + stage
            if time_slot < len(schedule[0]):
                schedule[stage][time_slot] = f'F{mb}'

    # Backward passes (after all forwards for that microbatch)
    for mb in range(num_microbatches):
        for stage in range(num_stages - 1, -1, -1):
            time_slot = num_microbatches + mb + (num_stages - 1 - stage)
            if time_slot < len(schedule[0]):
                schedule[stage][time_slot] = f'B{mb}'

    # Print schedule
    print("Time ->  ", end='')
    for t in range(len(schedule[0])):
        print(f'{t:>4}', end='')
    print()
    print("-" * (9 + 4 * len(schedule[0])))

    for stage in range(num_stages):
        print(f"GPU {stage}:   ", end='')
        for slot in schedule[stage]:
            print(f'{slot:>4}', end='')
        print()

    # Calculate bubble
    total_slots = len(schedule[0])
    active_slots = sum(1 for stage in schedule for slot in stage if slot != ' . ')
    bubble = 1 - active_slots / (num_stages * total_slots)

    print(f"\nPipeline bubble (idle time): {bubble:.1%}")
    print("Bubble = (P-1) / (P-1+M) where P=stages, M=microbatches")


def main():
    print("\n" + "="*70)
    print("LECTURE 14: Parallel & Distributed Computing")
    print("="*70)

    # Part 1: Why parallelism?
    print(f"\n{'='*70}")
    print("Part 1: The Scale Problem")
    print(f"{'='*70}")

    print("\nGPT-3 Training:")
    print("  Parameters: 175 billion")
    print("  Training FLOPs: ~3.14 × 10^23")
    print("  Single GPU (A100, 312 TFLOPS): would take ~32 years")
    print("  1000 GPUs: ~12 days")
    print("\nParallelism is not optional at this scale!")

    # Part 2: Matrix multiplication
    print(f"\n{'='*70}")
    print("Part 2: Matrix Multiplication (Core Operation)")
    print(f"{'='*70}")

    print("\nMatrix multiply is O(n^3) - highly parallelizable!")
    print("C[i,j] = sum_k A[i,k] * B[k,j]")
    print("\nAll C[i,j] can be computed independently (embarrassingly parallel)")

    print(f"\n{'Size':>8} | {'Operations':>14} | {'Time (simulated)':>18}")
    print(f"{'-'*45}")

    for n in matrix_sizes:
        ops = n ** 3
        # Simulate time (assuming 10 GFLOPS)
        time_s = ops / 10e9
        print(f"{n:8d} | {ops:14.2e} | {time_s:18.4f}s")

    # Part 3: Data parallelism
    print(f"\n{'='*70}")
    print("Part 3: Data Parallel Training")
    print(f"{'='*70}")

    print("\nData parallelism strategy:")
    print("  1. Replicate model on each GPU")
    print("  2. Split batch across GPUs")
    print("  3. Each GPU computes gradient on its portion")
    print("  4. AllReduce: average gradients across GPUs")
    print("  5. Each GPU updates its model copy (stays synchronized)")

    # Simulate data parallel training
    model_params = np.random.randn(100, 100)
    data = np.random.randn(256, 10)

    print("\nSimulating one training step:")
    model_params = data_parallel_training(model_params, data, num_workers, batch_size=256)

    # Part 4: Scaling analysis
    analyze_scaling()

    # Part 5: Pipeline parallelism
    simulate_pipeline_parallelism()

    # Part 6: GPU vs CPU comparison
    print(f"\n{'='*70}")
    print("Part 6: GPU vs CPU Architecture")
    print(f"{'='*70}")

    print("\n" + "-"*50)
    print(f"{'Aspect':<20} | {'CPU':>12} | {'GPU':>12}")
    print("-"*50)
    print(f"{'Cores':<20} | {'8-64':>12} | {'10,000+':>12}")
    print(f"{'Clock Speed':<20} | {'3-5 GHz':>12} | {'1-2 GHz':>12}")
    print(f"{'Memory':<20} | {'64-256 GB':>12} | {'16-80 GB':>12}")
    print(f"{'Memory BW':<20} | {'50 GB/s':>12} | {'1-2 TB/s':>12}")
    print(f"{'Peak FLOPS':<20} | {'~1 TFLOPS':>12} | {'~300 TFLOPS':>12}")
    print(f"{'Best For':<20} | {'Serial':>12} | {'Parallel':>12}")
    print("-"*50)

    print("\nGPU excels at SIMD: Same Instruction, Multiple Data")
    print("Matrix multiply, convolutions, attention - all SIMD-friendly")

    # Part 7: Communication costs
    print(f"\n{'='*70}")
    print("Part 7: Communication Costs (AllReduce)")
    print(f"{'='*70}")

    print("\nAllReduce: combine values across all workers")
    print("For P workers, each with N parameters:")
    print("  Ring AllReduce: 2(P-1) * N / P communication")
    print("  Tree AllReduce: 2 * log(P) * N communication")

    print(f"\n{'Workers':>8} | {'Ring Comms':>14} | {'Tree Comms':>14}")
    print(f"{'-'*42}")

    N = 1e9  # 1B parameters
    for P in [2, 4, 8, 16, 32]:
        ring = 2 * (P - 1) * N / P / 1e9
        tree = 2 * np.log2(P) * N / 1e9
        print(f"{P:8d} | {ring:14.2f} GB | {tree:14.2f} GB")

    print("\nKey insight: communication overhead grows with number of workers")
    print("This limits scaling efficiency!")

    # Part 8: Summary
    print(f"\n{'='*70}")
    print("Part 8: Course Synthesis")
    print(f"{'='*70}")

    print("""
Every large model training uses ALL the numerical methods from this course:

  Ch 1-3: FOUNDATIONS
    - Discretization: batching, tokenization, finite parameters
    - Floating-point: FP16/BF16 mixed precision training
    - Error analysis: monitoring loss curves, gradient norms

  Ch 4-6: OPTIMIZATION
    - Newton concepts: understanding curvature and step sizes
    - SGD + Adam: THE training algorithm
    - Momentum, adaptive learning rates, schedules

  Ch 7: STABILITY
    - log-sum-exp in attention and cross-entropy
    - Gradient clipping for stable training
    - Proper initialization (He, Xavier)

  Ch 8-9: INTEGRATION (indirect)
    - Expected loss is an integral over data distribution
    - SGD = Monte Carlo gradient estimation

  Ch 10-11: APPROXIMATION
    - Neural networks = function approximators
    - Regularization analogous to spline smoothness

  Ch 12: FOURIER
    - Positional encoding in Transformers
    - Spectral normalization in GANs

  Ch 13: DIFFERENTIAL EQUATIONS
    - ResNets ≈ Euler discretization
    - Neural ODEs, Diffusion models

  Ch 14: PARALLELISM (this lecture!)
    - GPU computing enables everything
    - Data/model parallelism for scale

Numerical thinking: "What's the error? Is it stable? How fast?
Will it converge? HOW DOES IT SCALE?"
""")

    print("\n" + "="*70)
    print("EXERCISES:")
    print("="*70)
    print("""
1. Implement a simple matrix multiply using blocking. Compare cache
   performance to naive triple loop (use time.time()).

2. Simulate the effect of batch size on gradient variance. Plot
   variance vs batch size for different "dataset" distributions.

3. Calculate the communication-to-computation ratio for your favorite
   model. When does communication become the bottleneck?

4. Implement a toy version of gradient compression: quantize gradients
   to 8 bits before AllReduce. How much does this hurt convergence?

5. Design an optimal parallelization strategy for a model with:
   - 10B parameters
   - 8 GPUs with 40GB each
   - Training batch size of 2048
   Should you use data parallelism, model parallelism, or both?
""")


if __name__ == "__main__":
    main()
