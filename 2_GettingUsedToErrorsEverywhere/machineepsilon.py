# Machine Epsilon Exploration: Find the smallest number eps such that 1 + eps > 1
# for different floating point types (half, single, double, etc.)

import time
import numpy as np



def explore_machine_epsilon(explore_type='float32'):
	type_map = {
		'float16': (np.float16, 'float16 (half)'),
		'float32': (np.float32, 'float32 (single)'),
		'float64': (np.float64, 'float64 (double)'),
		'int32': (np.int32, 'int32 (integer)'),
		'int64': (np.int64, 'int64 (integer)'),
	}
	if explore_type not in type_map:
		raise ValueError(f"Unknown type '{explore_type}'. Choose from: {list(type_map.keys())}")
	dtype, label = type_map[explore_type]
	sleep_time = 0.65  # seconds between steps (fixed)

	print(f"\n{'='*70}")
	print(f"Exploring Machine Epsilon for {label}")
	print(f"{'='*70}")
	print(f"{'Step':>4} | {'Epsilon':<14} | {'1+eps':<14} | {'1+eps > 1?':<12}")
	print(f"{'-'*70}")
	eps = dtype(1)
	step = 0
	while True:
		one_plus_eps = dtype(1) + eps
		is_greater = one_plus_eps > dtype(1)
		print(f"{step:4d} | {eps:.10e} | {one_plus_eps:.10e} | {is_greater}")
		last_eps = eps
		# For integer types, halving will always round to zero for eps < 1, so we break and explain
		if np.issubdtype(dtype, np.integer):
			if eps <= 1:
				explanation = (
					f"\n{'='*70}\n"
					f"For integer types, any epsilon < 1 is cast to 0, so 1+epsilon == 1 for all epsilon < 1.\n"
					f"There is no sub-integer resolution: the smallest possible difference is 1.\n"
					f"This is fundamentally different from floating-point types, which can represent ever-smaller differences.\n"
					f"{'='*70}\n"
				)
				raise RuntimeError(explanation)
		else:
			if not is_greater:
				break
		eps = dtype(eps) / dtype(2)
		step += 1
		time.sleep(sleep_time)
	print(f"{'='*70}")
	print(f"First sub-resolution epsilon for {label}: {last_eps:.10e}")
	print("Exploration complete.")
	print(f"You can compare this value to numpy.finfo(dtype).eps: {np.finfo(dtype).eps:.10e}")
	print(f"{'='*70}\n")


if __name__ == "__main__":
	# Student: Set which type to explore here!
	explore_type = 'float32'  # Options: 'float16', 'float32', 'float64', 'int32', 'int64'
	explore_machine_epsilon(explore_type)
