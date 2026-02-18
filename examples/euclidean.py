def euclidean(p1, p2):
    total = 0.0
    for i in range(len(p1)):
        diff = p1[i] - p2[i]
        total += diff * diff
    return total ** 0.5


if __name__ == "__main__":
    import time

    p1 = [float(i) for i in range(1000)]
    p2 = [float(i * 2) for i in range(1000)]
    start = time.time()
    for _ in range(10000):
        euclidean(p1, p2)
    elapsed = time.time() - start
    print(f"euclidean len=1000: {elapsed:.3f}s for 10000 iterations")
