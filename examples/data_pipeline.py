"""
CSV row processing pipeline - target for rustify-ml acceleration.
Expected gain: 10-30x via Rust string parsing + Vec operations.
"""


def parse_csv_row(row):
    """Parse a comma-separated row of floats into a list."""
    result = []
    for field in row.split(","):
        field = field.strip()
        if field:
            try:
                result.append(float(field))
            except ValueError:
                result.append(0.0)
    return result


def running_mean(values, window):
    """Compute a simple running mean with the given window size."""
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        total = 0.0
        count = 0
        for j in range(start, i + 1):
            total += values[j]
            count += 1
        result.append(total / count if count > 0 else 0.0)
    return result


def zscore_normalize(values):
    """Z-score normalize a list of floats."""
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    std = variance ** 0.5
    if std == 0.0:
        return [0.0] * n
    return [(x - mean) / std for x in values]


if __name__ == "__main__":
    import time

    rows = [",".join(str(float(i + j)) for j in range(20)) for i in range(1000)]
    start = time.time()
    for _ in range(100):
        for row in rows:
            parse_csv_row(row)
    elapsed = time.time() - start
    print(f"parse_csv_row 1000 rows x100: {elapsed:.3f}s")

    values = [float(i % 100) for i in range(10000)]
    start = time.time()
    for _ in range(10):
        running_mean(values, 50)
    elapsed = time.time() - start
    print(f"running_mean len=10000 window=50 x10: {elapsed:.3f}s")
