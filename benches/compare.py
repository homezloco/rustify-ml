#!/usr/bin/env python3
"""
rustify-ml speedup comparison script.

Runs each example fixture and prints a before/after timing table.
Use this to validate that generated Rust extensions are faster.

Usage:
    python benches/compare.py                    # Python-only baseline
    python benches/compare.py --with-rust        # Compare Python vs Rust (requires maturin develop)
"""

from __future__ import annotations

import argparse
import importlib
import sys
import timeit
from dataclasses import dataclass
from typing import Callable


@dataclass
class Benchmark:
    name: str
    func_name: str
    setup: Callable
    args_factory: Callable
    iters: int = 10_000


def run_benchmarks(with_rust: bool) -> None:
    results = []

    # ── euclidean distance ────────────────────────────────────────────────────
    from examples.euclidean import euclidean  # type: ignore

    bench_euclidean = Benchmark(
        name="euclidean distance",
        func_name="euclidean",
        setup=lambda: None,
        args_factory=lambda: ([float(i) for i in range(100)], [float(i * 2) for i in range(100)]),
        iters=10_000,
    )

    # ── dot product ───────────────────────────────────────────────────────────
    from examples.matrix_ops import dot_product  # type: ignore

    bench_dot = Benchmark(
        name="dot product (n=1000)",
        func_name="dot_product",
        setup=lambda: None,
        args_factory=lambda: ([float(i) for i in range(1000)], [float(i) for i in range(1000)]),
        iters=5_000,
    )

    # ── normalize pixels ──────────────────────────────────────────────────────
    from examples.image_preprocess import normalize_pixels  # type: ignore

    bench_norm = Benchmark(
        name="normalize pixels (n=1000)",
        func_name="normalize_pixels",
        setup=lambda: None,
        args_factory=lambda: ([float(i % 256) for i in range(1000)], 128.0, 64.0),
        iters=5_000,
    )

    # ── count pairs (BPE) ─────────────────────────────────────────────────────
    from examples.bpe_tokenizer import count_pairs  # type: ignore

    bench_bpe = Benchmark(
        name="count_pairs BPE (n=500)",
        func_name="count_pairs",
        setup=lambda: None,
        args_factory=lambda: ([i % 256 for i in range(500)],),
        iters=5_000,
    )

    py_funcs = {
        "euclidean": euclidean,
        "dot_product": dot_product,
        "normalize_pixels": normalize_pixels,
        "count_pairs": count_pairs,
    }

    benches = [bench_euclidean, bench_dot, bench_norm, bench_bpe]

    # Try to import Rust extension
    rs_mod = None
    if with_rust:
        try:
            rs_mod = importlib.import_module("rustify_ml_ext")
            print(f"✓ Loaded rustify_ml_ext (Rust extension)")
        except ImportError as e:
            print(f"✗ Could not import rustify_ml_ext: {e}")
            print("  Run: cd dist/rustify_ml_ext && maturin develop --release")
            sys.exit(1)

    print()
    print("=" * 70)
    print("  rustify-ml speedup comparison")
    print("=" * 70)
    if with_rust:
        print(f"  {'Benchmark':<30} | {'Python':>9} | {'Rust':>9} | {'Speedup':>8}")
        print(f"  {'-'*30}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}")
    else:
        print(f"  {'Benchmark':<30} | {'Python':>9} | {'Iters':>6}")
        print(f"  {'-'*30}-+-{'-'*9}-+-{'-'*6}")

    for bench in benches:
        py_fn = py_funcs[bench.func_name]
        args = bench.args_factory()

        py_time = timeit.timeit(lambda: py_fn(*args), number=bench.iters)

        if with_rust and rs_mod is not None:
            rs_fn = getattr(rs_mod, bench.func_name, None)
            if rs_fn is None:
                print(f"  {bench.name:<30} | {py_time:>8.3f}s | {'N/A':>9} | {'N/A':>8}")
                continue
            rs_time = timeit.timeit(lambda: rs_fn(*args), number=bench.iters)
            speedup = py_time / rs_time if rs_time > 0 else float("inf")
            print(
                f"  {bench.name:<30} | {py_time:>8.3f}s | {rs_time:>8.3f}s | {speedup:>7.1f}x"
            )
        else:
            print(f"  {bench.name:<30} | {py_time:>8.3f}s | {bench.iters:>6}")

    print("=" * 70)
    print()
    if not with_rust:
        print("Tip: Run with --with-rust after `maturin develop --release` to see speedups.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rustify-ml speedup comparison")
    parser.add_argument(
        "--with-rust",
        action="store_true",
        help="Compare Python vs Rust extension (requires maturin develop --release)",
    )
    args = parser.parse_args()

    # Add project root to path so `examples.*` imports work
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    run_benchmarks(args.with_rust)
