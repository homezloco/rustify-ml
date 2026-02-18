"""
Pure Python matrix multiply - target for rustify-ml acceleration.
Expected gain: 50-100x via ndarray matmul in Rust.
"""


def matmul(a, b, n):
    """Multiply two n×n matrices stored as flat Vec<f64> (row-major)."""
    result = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            total = 0.0
            for k in range(n):
                total += a[i * n + k] * b[k * n + j]
            result[i * n + j] = total
    return result


def dot_product(a, b):
    """Compute dot product of two equal-length vectors."""
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


if __name__ == "__main__":
    import time

    n = 32
    a = [float(i % 7) for i in range(n * n)]
    b = [float(i % 5) for i in range(n * n)]
    start = time.time()
    for _ in range(200):
        matmul(a, b, n)
    elapsed = time.time() - start
    print(f"matmul {n}x{n}: {elapsed:.3f}s for 200 iterations")

    va = [float(i) for i in range(1000)]
    vb = [float(i) for i in range(1000)]
    start = time.time()
    for _ in range(10000):
        dot_product(va, vb)
    elapsed = time.time() - start
    print(f"dot_product len=1000: {elapsed:.3f}s for 10000 iterations")
