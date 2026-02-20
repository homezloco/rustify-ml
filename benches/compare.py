#!/usr/bin/env python3
"""
rustify-ml comprehensive benchmark suite.

Runs every example fixture and prints a before/after timing table.
Use this to validate that generated Rust extensions are faster.

Usage:
    python benches/compare.py                    # Python-only baseline
    python benches/compare.py --with-rust        # Compare Python vs Rust (requires maturin develop)
    python benches/compare.py --markdown         # Output markdown table for docs
"""

from __future__ import annotations

import argparse
import importlib
import sys
import timeit
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Benchmark:
    name: str
    module: str
    func_name: str
    args_factory: Callable[[], tuple]
    iters: int = 10_000
    category: str = "general"


BENCHMARKS = [
    # ── Distance metrics ──────────────────────────────────────────────────────
    Benchmark(
        name="euclidean (n=1000)",
        module="examples.euclidean",
        func_name="euclidean",
        args_factory=lambda: (
            [float(i) for i in range(1000)],
            [float(i * 2) for i in range(1000)],
        ),
        iters=5_000,
        category="distance",
    ),

    # ── Linear algebra ────────────────────────────────────────────────────────
    Benchmark(
        name="dot_product (n=1000)",
        module="examples.matrix_ops",
        func_name="dot_product",
        args_factory=lambda: (
            [float(i) for i in range(1000)],
            [float(i) for i in range(1000)],
        ),
        iters=5_000,
        category="linalg",
    ),

    # ── Image preprocessing ───────────────────────────────────────────────────
    Benchmark(
        name="normalize_pixels (n=1000)",
        module="examples.image_preprocess",
        func_name="normalize_pixels",
        args_factory=lambda: (
            [float(i % 256) for i in range(1000)],
            128.0,
            64.0,
        ),
        iters=5_000,
        category="image",
    ),

    # ── Data pipeline ─────────────────────────────────────────────────────────
    Benchmark(
        name="running_mean (n=500, w=10)",
        module="examples.data_pipeline",
        func_name="running_mean",
        args_factory=lambda: (
            [float(i) for i in range(500)],
            10,
        ),
        iters=5_000,
        category="pipeline",
    ),

    # ── BPE tokenizer ─────────────────────────────────────────────────────────
    Benchmark(
        name="count_pairs (n=500)",
        module="examples.bpe_tokenizer",
        func_name="count_pairs",
        args_factory=lambda: ([i % 256 for i in range(500)],),
        iters=5_000,
        category="nlp",
    ),
    Benchmark(
        name="bpe_encode (len=100)",
        module="examples.bpe_tokenizer",
        func_name="bpe_encode",
        args_factory=lambda: ("hello world " * 10, []),
        iters=5_000,
        category="nlp",
    ),

    # ── sklearn-style scalers ─────────────────────────────────────────────────
    Benchmark(
        name="standard_scale (n=1000)",
        module="examples.sklearn_scaler",
        func_name="standard_scale",
        args_factory=lambda: (
            [float(i % 256) for i in range(1000)],
            128.0,
            64.0,
        ),
        iters=5_000,
        category="sklearn",
    ),
    Benchmark(
        name="min_max_scale (n=1000)",
        module="examples.sklearn_scaler",
        func_name="min_max_scale",
        args_factory=lambda: (
            [float(i % 256) for i in range(1000)],
            0.0,
            255.0,
        ),
        iters=5_000,
        category="sklearn",
    ),
    Benchmark(
        name="l2_normalize (n=1000)",
        module="examples.sklearn_scaler",
        func_name="l2_normalize",
        args_factory=lambda: ([float(i) for i in range(1000)],),
        iters=5_000,
        category="sklearn",
    ),

    # ── Signal processing ─────────────────────────────────────────────────────
    Benchmark(
        name="convolve1d (n=1000, k=5)",
        module="examples.signal_processing",
        func_name="convolve1d",
        args_factory=lambda: (
            [float(i % 100) for i in range(1000)],
            [0.1, 0.2, 0.4, 0.2, 0.1],
        ),
        iters=3_000,
        category="signal",
    ),
    Benchmark(
        name="moving_average (n=1000, w=10)",
        module="examples.signal_processing",
        func_name="moving_average",
        args_factory=lambda: (
            [float(i % 100) for i in range(1000)],
            10,
        ),
        iters=3_000,
        category="signal",
    ),
    Benchmark(
        name="diff (n=1000)",
        module="examples.signal_processing",
        func_name="diff",
        args_factory=lambda: ([float(i) for i in range(1000)],),
        iters=5_000,
        category="signal",
    ),
    Benchmark(
        name="cumsum (n=1000)",
        module="examples.signal_processing",
        func_name="cumsum",
        args_factory=lambda: ([float(i) for i in range(1000)],),
        iters=5_000,
        category="signal",
    ),
]


def run_benchmarks(with_rust: bool, markdown: bool) -> int:
    """Run all benchmarks. Returns number of failures (0 = all pass)."""
    rs_mod = None
    if with_rust:
        try:
            rs_mod = importlib.import_module("rustify_ml_ext")
        except ImportError as e:
            print(f"FAIL: Could not import rustify_ml_ext: {e}")
            print("  Run: cd dist/rustify_ml_ext && maturin develop --release")
            return 1

    results = []

    for bench in BENCHMARKS:
        try:
            mod = importlib.import_module(bench.module)
            py_fn = getattr(mod, bench.func_name)
        except (ImportError, AttributeError) as e:
            print(f"SKIP {bench.name}: {e}")
            continue

        args = bench.args_factory()
        py_time = timeit.timeit(lambda: py_fn(*args), number=bench.iters)
        py_per_call_us = (py_time / bench.iters) * 1_000_000

        rs_time = None
        rs_per_call_us = None
        speedup = None

        if with_rust and rs_mod is not None:
            rs_fn = getattr(rs_mod, bench.func_name, None)
            if rs_fn is not None:
                rs_time = timeit.timeit(lambda: rs_fn(*args), number=bench.iters)
                rs_per_call_us = (rs_time / bench.iters) * 1_000_000
                speedup = py_time / rs_time if rs_time > 0 else float("inf")

        results.append({
            "name": bench.name,
            "category": bench.category,
            "iters": bench.iters,
            "py_time": py_time,
            "py_us": py_per_call_us,
            "rs_time": rs_time,
            "rs_us": rs_per_call_us,
            "speedup": speedup,
        })

    # ── Output ────────────────────────────────────────────────────────────────
    if markdown:
        print("# rustify-ml Benchmark Results\n")
        print(f"Tested {len(results)} functions. Python vs Rust speedup comparison.\n")
        if with_rust:
            print("| Function | Python (us/call) | Rust (us/call) | Speedup |")
            print("|----------|------------------|----------------|---------|")
            for r in results:
                speedup_str = f"{r['speedup']:.1f}x" if r['speedup'] else "N/A"
                rs_us_str = f"{r['rs_us']:.1f}" if r['rs_us'] else "N/A"
                print(f"| {r['name']} | {r['py_us']:.1f} | {rs_us_str} | {speedup_str} |")
        else:
            print("| Function | Python (us/call) | Iters |")
            print("|----------|------------------|-------|")
            for r in results:
                print(f"| {r['name']} | {r['py_us']:.1f} | {r['iters']} |")
        print()
    else:
        print("\n" + "=" * 80)
        print("  rustify-ml benchmark results")
        print("=" * 80)
        if with_rust:
            print(f"  {'Function':<35} | {'Python us':>10} | {'Rust us':>10} | {'Speedup':>8}")
            print(f"  {'-'*35}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
            for r in results:
                speedup_str = f"{r['speedup']:.1f}x" if r['speedup'] else "N/A"
                rs_us_str = f"{r['rs_us']:.1f}" if r['rs_us'] else "N/A"
                print(f"  {r['name']:<35} | {r['py_us']:>10.1f} | {rs_us_str:>10} | {speedup_str:>8}")
        else:
            print(f"  {'Function':<35} | {'Python us':>10} | {'Iters':>6}")
            print(f"  {'-'*35}-+-{'-'*10}-+-{'-'*6}")
            for r in results:
                print(f"  {r['name']:<35} | {r['py_us']:>10.1f} | {r['iters']:>6}")
        print("=" * 80)

        if not with_rust:
            print("\nTip: Run with --with-rust after `maturin develop --release` to see speedups.")
            print("     Run with --markdown to generate a docs-ready table.\n")
            sys.stdout.flush()

    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rustify-ml benchmark suite")
    parser.add_argument(
        "--with-rust",
        action="store_true",
        help="Compare Python vs Rust extension (requires maturin develop --release)",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output markdown table for documentation",
    )
    args = parser.parse_args()

    # Add project root to path so `examples.*` imports work
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    sys.exit(run_benchmarks(args.with_rust, args.markdown))