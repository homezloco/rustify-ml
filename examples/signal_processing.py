"""
scipy.signal-style 1D signal processing — rustify-ml acceleration targets.

Pure-Python implementations of common signal processing operations.
The nested for-loop patterns translate cleanly to Rust with zero fallbacks.

Expected Rust speedup: 10–40x via Vec<f64> + nested loops.
"""

from __future__ import annotations


def convolve1d(signal: list, kernel: list) -> list:
    """1D convolution: output[i] = sum(signal[i+j] * kernel[j] for j in range(k)).

    Equivalent to scipy.signal.convolve(signal, kernel, mode='valid').
    """
    n = len(signal)
    k = len(kernel)
    result = [0.0] * (n - k + 1)
    for i in range(n - k + 1):
        total = 0.0
        for j in range(k):
            total += signal[i + j] * kernel[j]
        result[i] = total
    return result


def moving_average(signal: list, window: int) -> list:
    """Sliding window mean: result[i] = mean(signal[i:i+window]).

    Equivalent to pandas rolling(window).mean() or scipy uniform_filter1d.
    """
    n = len(signal)
    result = [0.0] * (n - window + 1)
    for i in range(n - window + 1):
        total = 0.0
        for j in range(window):
            total += signal[i + j]
        result[i] = total / window
    return result


def cross_correlate(a: list, b: list) -> list:
    """1D cross-correlation: output[i] = sum(a[j] * b[j+i] for j).

    Equivalent to numpy.correlate(a, b, mode='valid').
    """
    n = len(a)
    m = len(b)
    out_len = m - n + 1
    result = [0.0] * out_len
    for i in range(out_len):
        total = 0.0
        for j in range(n):
            total += a[j] * b[i + j]
        result[i] = total
    return result


def diff(signal: list) -> list:
    """First-order difference: result[i] = signal[i+1] - signal[i].

    Equivalent to numpy.diff(signal).
    """
    n = len(signal)
    result = [0.0] * (n - 1)
    for i in range(n - 1):
        result[i] = signal[i + 1] - signal[i]
    return result


def cumsum(signal: list) -> list:
    """Cumulative sum: result[i] = sum(signal[0:i+1]).

    Equivalent to numpy.cumsum(signal).
    """
    n = len(signal)
    result = [0.0] * n
    total = 0.0
    for i in range(n):
        total += signal[i]
        result[i] = total
    return result


if __name__ == "__main__":
    import time

    n = 10_000
    signal = [float(i % 100) for i in range(n)]
    kernel = [0.25, 0.5, 0.25]
    n_iters = 5_000

    print(f"Benchmarking signal processing (n={n}, iters={n_iters})...")

    start = time.perf_counter()
    for _ in range(n_iters):
        convolve1d(signal, kernel)
    elapsed = time.perf_counter() - start
    print(f"  convolve1d (k=3):    {elapsed:.3f}s ({elapsed/n_iters*1000:.2f}ms/call)")

    start = time.perf_counter()
    for _ in range(n_iters):
        moving_average(signal, 10)
    elapsed = time.perf_counter() - start
    print(f"  moving_average (w=10): {elapsed:.3f}s ({elapsed/n_iters*1000:.2f}ms/call)")

    start = time.perf_counter()
    for _ in range(n_iters):
        cumsum(signal)
    elapsed = time.perf_counter() - start
    print(f"  cumsum:              {elapsed:.3f}s ({elapsed/n_iters*1000:.2f}ms/call)")
