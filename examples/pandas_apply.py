"""
Pandas-like apply over rows: simulate a small feature engineering pipeline.
Intended for rustify-ml acceleration (loops and per-field math).
"""

from math import log1p


def featurize_row(row: dict[str, float]) -> dict[str, float]:
    # Pure-Python loop; good target for rustify-ml.
    out = {}
    for key, value in row.items():
        scaled = value * 1.5 + 2.0
        if value > 0:
            out[f"log_{key}"] = log1p(value)
        out[f"scaled_{key}"] = scaled
        out[f"centered_{key}"] = value - 0.5
    return out


def apply_rows(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    return [featurize_row(row) for row in rows]


if __name__ == "__main__":
    import time
    rows = [
        {"a": float(i % 7), "b": float(i % 5), "c": float(i % 3)}
        for i in range(50_000)
    ]
    start = time.perf_counter()
    apply_rows(rows)
    elapsed = time.perf_counter() - start
    print(f"apply_rows 50k rows: {elapsed:.3f}s")
