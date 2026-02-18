"""
sklearn-style StandardScaler and MinMaxScaler — rustify-ml acceleration targets.

These are pure-Python implementations of the most common sklearn preprocessing
operations. The loops are identical in structure to normalize_pixels and translate
with zero fallbacks.

Expected Rust speedup: 5–20x via Vec<f64> math.
"""

from __future__ import annotations


def standard_scale(data: list, mean: float, std: float) -> list:
    """Standardize: (x - mean) / std for each element.

    Equivalent to sklearn.preprocessing.StandardScaler.transform on a 1D array.
    """
    result = [0.0] * len(data)
    for i in range(len(data)):
        result[i] = (data[i] - mean) / std
    return result


def min_max_scale(data: list, min_val: float, max_val: float) -> list:
    """Min-max normalize: (x - min) / (max - min) for each element.

    Equivalent to sklearn.preprocessing.MinMaxScaler.transform on a 1D array.
    """
    result = [0.0] * len(data)
    range_val = max_val - min_val
    for i in range(len(data)):
        result[i] = (data[i] - min_val) / range_val
    return result


def robust_scale(data: list, median: float, iqr: float) -> list:
    """Robust scaling: (x - median) / IQR for each element.

    Equivalent to sklearn.preprocessing.RobustScaler.transform on a 1D array.
    """
    result = [0.0] * len(data)
    for i in range(len(data)):
        result[i] = (data[i] - median) / iqr
    return result


def l2_normalize(data: list) -> list:
    """L2 normalize a vector: x / ||x||_2.

    Equivalent to sklearn.preprocessing.normalize(x, norm='l2').
    """
    total = 0.0
    for i in range(len(data)):
        total += data[i] * data[i]
    norm = total ** 0.5
    result = [0.0] * len(data)
    for i in range(len(data)):
        result[i] = data[i] / norm
    return result


if __name__ == "__main__":
    import time

    n = 100_000
    data = [float(i % 256) for i in range(n)]
    n_iters = 1_000

    print(f"Benchmarking sklearn-style scalers (n={n}, iters={n_iters})...")

    start = time.perf_counter()
    for _ in range(n_iters):
        standard_scale(data, 128.0, 64.0)
    elapsed = time.perf_counter() - start
    print(f"  standard_scale:  {elapsed:.3f}s ({elapsed/n_iters*1000:.2f}ms/call)")

    start = time.perf_counter()
    for _ in range(n_iters):
        min_max_scale(data, 0.0, 255.0)
    elapsed = time.perf_counter() - start
    print(f"  min_max_scale:   {elapsed:.3f}s ({elapsed/n_iters*1000:.2f}ms/call)")

    start = time.perf_counter()
    for _ in range(n_iters):
        l2_normalize(data[:1000])
    elapsed = time.perf_counter() - start
    print(f"  l2_normalize:    {elapsed:.3f}s ({elapsed/n_iters*1000:.2f}ms/call)")
