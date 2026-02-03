import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Fourier Transform
# DFT, FFT, and applications to signal processing and deep learning

np.random.seed(42)

# =====================================================================
# Configuration Parameters - Experiment with these!
# =====================================================================

# Signal parameters
sample_rate = 100  # Hz
duration = 1.0     # seconds
n_samples = int(sample_rate * duration)

# Frequencies to include in test signal
frequencies = [5, 12, 20]  # Hz
amplitudes = [1.0, 0.5, 0.3]

# =====================================================================


def create_signal(t, freqs, amps):
    """Create a signal as sum of sinusoids."""
    signal = np.zeros_like(t)
    for freq, amp in zip(freqs, amps):
        signal += amp * np.sin(2 * np.pi * freq * t)
    return signal


def dft_naive(x):
    """
    Naive Discrete Fourier Transform: O(N^2).
    X[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*k*n/N)
    """
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X


def idft_naive(X):
    """Naive Inverse DFT."""
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
    return x / N


def fft_cooley_tukey(x):
    """
    Cooley-Tukey FFT algorithm: O(N log N).
    Requires N to be a power of 2.
    """
    N = len(x)
    if N <= 1:
        return x

    # Pad to power of 2 if necessary
    if N & (N - 1) != 0:
        next_pow2 = 1 << (N - 1).bit_length()
        x = np.pad(x, (0, next_pow2 - N), mode='constant')
        N = next_pow2

    if N == 1:
        return x

    # Divide
    even = fft_cooley_tukey(x[0::2])
    odd = fft_cooley_tukey(x[1::2])

    # Conquer
    T = np.exp(-2j * np.pi * np.arange(N // 2) / N) * odd
    return np.concatenate([even + T, even - T])


def positional_encoding(pos, d_model, max_len=1000):
    """
    Transformer positional encoding (sinusoidal).
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe[pos]


def main():
    print("\n" + "="*70)
    print("LECTURE 12: Fourier Transform")
    print("="*70)

    # Generate test signal
    t = np.linspace(0, duration, n_samples, endpoint=False)
    signal = create_signal(t, frequencies, amplitudes)

    # Part 1: DFT by hand (small example)
    print(f"\n{'='*70}")
    print("Part 1: DFT by Hand (4-point example)")
    print(f"{'='*70}")

    x_small = np.array([1, 0, -1, 0])
    print(f"\nInput signal: x = {x_small}")
    print("\nDFT formula: X[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*k*n/N)")

    X_small = dft_naive(x_small)
    print("\nManual computation:")
    print("X[0] = 1*1 + 0*1 + (-1)*1 + 0*1 = 0")
    print("X[1] = 1*1 + 0*(-i) + (-1)*(-1) + 0*(i) = 1 + 1 = 2")
    print("X[2] = 1*1 + 0*(-1) + (-1)*1 + 0*(-1) = 0")
    print("X[3] = 1*1 + 0*(i) + (-1)*(-1) + 0*(-i) = 2")

    print(f"\nComputed X: {np.round(X_small.real).astype(int)}")
    print("(Imaginary parts are zero for this symmetric signal)")

    # Part 2: FFT vs DFT performance
    print(f"\n{'='*70}")
    print("Part 2: FFT Performance (O(N^2) vs O(N log N))")
    print(f"{'='*70}")

    print(f"\n{'N':>8} | {'DFT O(N^2)':>12} | {'FFT O(NlogN)':>14} | {'Speedup':>10}")
    print(f"{'-'*50}")

    for N in [16, 64, 256, 1024]:
        ops_dft = N * N
        ops_fft = N * np.log2(N)
        speedup = ops_dft / ops_fft
        print(f"{N:8d} | {ops_dft:12.0f} | {ops_fft:14.0f} | {speedup:10.1f}x")

    # Part 3: Frequency analysis
    print(f"\n{'='*70}")
    print("Part 3: Frequency Analysis of Test Signal")
    print(f"Signal: sum of sinusoids at {frequencies} Hz")
    print(f"{'='*70}")

    # Compute FFT
    X = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n_samples, 1/sample_rate)

    # Get positive frequencies and magnitudes
    pos_mask = freqs >= 0
    pos_freqs = freqs[pos_mask]
    magnitudes = np.abs(X[pos_mask]) / n_samples * 2

    print("\nDetected frequencies (peaks in magnitude spectrum):")
    print(f"{'Frequency (Hz)':>15} | {'Magnitude':>12} | {'Expected':>12}")
    print(f"{'-'*45}")

    # Find peaks
    threshold = 0.1
    peaks = []
    for i in range(1, len(magnitudes) - 1):
        if magnitudes[i] > threshold and magnitudes[i] > magnitudes[i-1] and magnitudes[i] > magnitudes[i+1]:
            peaks.append((pos_freqs[i], magnitudes[i]))

    for freq, mag in sorted(peaks):
        expected = ""
        for f, a in zip(frequencies, amplitudes):
            if abs(freq - f) < 1:
                expected = f"{a:.1f}"
                break
        print(f"{freq:15.1f} | {mag:12.3f} | {expected:>12}")

    # Part 4: Convolution Theorem
    print(f"\n{'='*70}")
    print("Part 4: Convolution Theorem")
    print("F(f * g) = F(f) · F(g)")
    print(f"{'='*70}")

    # Simple convolution example
    f_sig = np.array([1, 2, 3, 2, 1])
    g_sig = np.array([1, 1, 1])

    # Pad for circular convolution
    N_conv = len(f_sig) + len(g_sig) - 1
    f_pad = np.pad(f_sig, (0, N_conv - len(f_sig)))
    g_pad = np.pad(g_sig, (0, N_conv - len(g_sig)))

    # Direct convolution
    conv_direct = np.convolve(f_sig, g_sig, mode='full')

    # FFT convolution
    F_f = np.fft.fft(f_pad)
    F_g = np.fft.fft(g_pad)
    conv_fft = np.real(np.fft.ifft(F_f * F_g))

    print(f"\nf = {f_sig}")
    print(f"g = {g_sig}")
    print(f"\nDirect convolution:  {conv_direct}")
    print(f"FFT convolution:     {np.round(conv_fft).astype(int)}")
    print("\nFor large N, FFT convolution is O(N log N) vs O(N^2) direct!")

    # Part 5: Positional Encoding
    print(f"\n{'='*70}")
    print("Part 5: Transformer Positional Encoding (Fourier Features)")
    print(f"{'='*70}")

    d_model = 8
    positions = [0, 1, 10, 100]

    print(f"\nPositional encoding with d_model={d_model}:")
    print("PE(pos, 2i) = sin(pos / 10000^(2i/d))")
    print("PE(pos, 2i+1) = cos(pos / 10000^(2i/d))")

    print(f"\n{'pos':>5}", end=" | ")
    for i in range(d_model):
        print(f"dim {i:2d}", end=" | ")
    print()
    print("-" * (8 + 10 * d_model))

    for pos in positions:
        pe = positional_encoding(pos, d_model)
        print(f"{pos:5d}", end=" | ")
        for val in pe:
            print(f"{val:7.3f}", end=" | ")
        print()

    print("\nKey insight: Different positions have unique 'fingerprints'")
    print("Low dimensions = low frequencies (change slowly)")
    print("High dimensions = high frequencies (change rapidly)")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Time domain signal
    axes[0, 0].plot(t, signal, 'b-', linewidth=0.8)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title(f'Time Domain Signal ({frequencies} Hz)')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Frequency spectrum
    axes[0, 1].stem(pos_freqs[:50], magnitudes[:50], linefmt='b-', markerfmt='bo', basefmt='k-')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Magnitude')
    axes[0, 1].set_title('Frequency Spectrum (FFT)')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Positional encoding heatmap
    pe_matrix = np.array([positional_encoding(p, 64) for p in range(50)])
    im = axes[1, 0].imshow(pe_matrix.T, aspect='auto', cmap='RdBu')
    axes[1, 0].set_xlabel('Position')
    axes[1, 0].set_ylabel('Dimension')
    axes[1, 0].set_title('Transformer Positional Encoding')
    plt.colorbar(im, ax=axes[1, 0])

    # Plot 4: PE similarity matrix
    pe_long = np.array([positional_encoding(p, 64) for p in range(100)])
    similarity = pe_long @ pe_long.T
    axes[1, 1].imshow(similarity, cmap='viridis')
    axes[1, 1].set_xlabel('Position')
    axes[1, 1].set_ylabel('Position')
    axes[1, 1].set_title('PE Dot Product Similarity')

    plt.tight_layout()
    plt.savefig('fourier.png', dpi=150)
    print(f"\nVisualization saved to: fourier.png")

    print("\n" + "="*70)
    print("EXERCISES:")
    print("="*70)
    print("""
1. Add noise to the signal: signal + 0.5*np.random.randn(n_samples).
   Can you still identify the original frequencies?

2. Implement windowing (Hann window) before FFT. Compare the spectrum
   with and without windowing.

3. Use FFT to implement fast polynomial multiplication. Verify it
   works on (1 + 2x) * (3 + 4x) = 3 + 10x + 8x^2.

4. Investigate the relationship between positional encoding frequency
   and the "wavelength" of patterns it can represent.

5. Implement 2D FFT for image processing. Apply a low-pass filter
   to blur an image.
""")


if __name__ == "__main__":
    main()
