#!/usr/bin/env python3
"""
rustify-ml: All-fixtures test runner.

Tests every Python tool we've bridged to Rust. Runs the Python implementation
and verifies correctness. After `maturin develop --release`, also verifies the
Rust extension produces identical results.

Usage:
    python tests/test_all_fixtures.py              # Python correctness only
    python tests/test_all_fixtures.py --with-rust  # Python + Rust parity check

Exit code: 0 = all pass, 1 = failures.
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

# Add project root to path
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TestCase:
    name: str
    module: str
    func: str
    args: tuple
    expected: Any
    check: Callable[[Any, Any], bool] = field(default=None)
    tol: float = 1e-9


def approx_eq(a, b, tol=1e-9):
    """Check two values are approximately equal (scalar or list)."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < tol
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(abs(x - y) < tol for x, y in zip(a, b))
    return a == b


# ── Test cases for every bridged fixture ─────────────────────────────────────

TESTS: list[TestCase] = [

    # ── euclidean distance ────────────────────────────────────────────────────
    TestCase(
        name="euclidean: [0,3,4] vs [0,0,0] = 5.0",
        module="examples.euclidean",
        func="euclidean",
        args=([0.0, 3.0, 4.0], [0.0, 0.0, 0.0]),
        expected=5.0,
        tol=1e-9,
    ),
    TestCase(
        name="euclidean: identical vectors = 0.0",
        module="examples.euclidean",
        func="euclidean",
        args=([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
        expected=0.0,
        tol=1e-9,
    ),

    # ── matrix ops ────────────────────────────────────────────────────────────
    TestCase(
        name="dot_product: [1,2,3]·[4,5,6] = 32.0",
        module="examples.matrix_ops",
        func="dot_product",
        args=([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
        expected=32.0,
        tol=1e-9,
    ),

    # ── image preprocessing ───────────────────────────────────────────────────
    TestCase(
        name="normalize_pixels: [128.0] mean=128 std=64 → [0.0]",
        module="examples.image_preprocess",
        func="normalize_pixels",
        args=([128.0, 192.0, 64.0], 128.0, 64.0),
        expected=[0.0, 1.0, -1.0],
        tol=1e-9,
    ),

    # ── data pipeline ─────────────────────────────────────────────────────────
    # running_mean returns cumulative mean: [1, 1.5, 2.0, 3.0, 4.0] for window=3
    # (first two are partial windows, then full sliding window kicks in)
    TestCase(
        name="running_mean: [1,2,3,4,5] window=3 → cumulative then sliding",
        module="examples.data_pipeline",
        func="running_mean",
        args=([1.0, 2.0, 3.0, 4.0, 5.0], 3),
        expected=[1.0, 1.5, 2.0, 3.0, 4.0],
        tol=1e-9,
    ),

    # ── BPE tokenizer ─────────────────────────────────────────────────────────
    TestCase(
        name="count_pairs: [1,2,1,2] → {(1,2):2, (2,1):1}",
        module="examples.bpe_tokenizer",
        func="count_pairs",
        args=([1, 2, 1, 2],),
        expected={(1, 2): 2, (2, 1): 1},
        check=lambda a, b: a == b,
    ),
    TestCase(
        name="bpe_encode: empty merges → byte values",
        module="examples.bpe_tokenizer",
        func="bpe_encode",
        args=("hi", []),
        expected=[104, 105],  # ord('h')=104, ord('i')=105
        check=lambda a, b: a == b,
    ),

    # ── sklearn-style scalers ─────────────────────────────────────────────────
    TestCase(
        name="standard_scale: [0,128,256] mean=128 std=64",
        module="examples.sklearn_scaler",
        func="standard_scale",
        args=([0.0, 128.0, 256.0], 128.0, 64.0),
        expected=[-2.0, 0.0, 2.0],
        tol=1e-9,
    ),
    TestCase(
        name="min_max_scale: [0,128,255] min=0 max=255",
        module="examples.sklearn_scaler",
        func="min_max_scale",
        args=([0.0, 127.5, 255.0], 0.0, 255.0),
        expected=[0.0, 0.5, 1.0],
        tol=1e-9,
    ),
    TestCase(
        name="l2_normalize: [3,4] → [0.6, 0.8]",
        module="examples.sklearn_scaler",
        func="l2_normalize",
        args=([3.0, 4.0],),
        expected=[0.6, 0.8],
        tol=1e-9,
    ),

    # ── signal processing ─────────────────────────────────────────────────────
    # convolve1d([1,2,3,4], [1,0,-1]): n=4, k=3, n-k+1=2 outputs
    # i=0: 1*1 + 2*0 + 3*(-1) = -2
    # i=1: 2*1 + 3*0 + 4*(-1) = -2
    TestCase(
        name="convolve1d: [1,2,3,4] kernel=[1,0,-1] → [-2,-2] (2 outputs)",
        module="examples.signal_processing",
        func="convolve1d",
        args=([1.0, 2.0, 3.0, 4.0], [1.0, 0.0, -1.0]),
        expected=[-2.0, -2.0],
        tol=1e-9,
    ),
    TestCase(
        name="moving_average: [1,2,3,4,5] window=3 → [2,3,4]",
        module="examples.signal_processing",
        func="moving_average",
        args=([1.0, 2.0, 3.0, 4.0, 5.0], 3),
        expected=[2.0, 3.0, 4.0],
        tol=1e-9,
    ),
    TestCase(
        name="diff: [1,3,6,10] → [2,3,4]",
        module="examples.signal_processing",
        func="diff",
        args=([1.0, 3.0, 6.0, 10.0],),
        expected=[2.0, 3.0, 4.0],
        tol=1e-9,
    ),
    TestCase(
        name="cumsum: [1,2,3,4] → [1,3,6,10]",
        module="examples.signal_processing",
        func="cumsum",
        args=([1.0, 2.0, 3.0, 4.0],),
        expected=[1.0, 3.0, 6.0, 10.0],
        tol=1e-9,
    ),
]


def run_tests(with_rust: bool) -> int:
    """Run all test cases. Returns number of failures."""
    rs_mod = None
    if with_rust:
        try:
            rs_mod = importlib.import_module("rustify_ml_ext")
            print("OK Loaded rustify_ml_ext (Rust extension)\n")
        except ImportError as e:
            print(f"FAIL Could not import rustify_ml_ext: {e}")
            print("  Run: cd dist/rustify_ml_ext && maturin develop --release\n")
            return 1

    passed = 0
    failed = 0
    skipped = 0

    print("=" * 65)
    print(f"  rustify-ml fixture tests ({len(TESTS)} cases)")
    if with_rust:
        print("  Mode: Python correctness + Rust parity")
    else:
        print("  Mode: Python correctness only")
    print("=" * 65)

    for tc in TESTS:
        try:
            mod = importlib.import_module(tc.module)
            py_fn = getattr(mod, tc.func)
            py_result = py_fn(*tc.args)

            # Check Python correctness
            check = tc.check or (lambda a, b: approx_eq(a, b, tc.tol))
            if not check(py_result, tc.expected):
                print(f"  FAIL  {tc.name}")
                print(f"         Python: {py_result!r}")
                print(f"         Expected: {tc.expected!r}")
                failed += 1
                continue

            # Check Rust parity if requested
            if with_rust and rs_mod is not None:
                rs_fn = getattr(rs_mod, tc.func, None)
                if rs_fn is None:
                    print(f"  SKIP  {tc.name} (not in Rust ext)")
                    skipped += 1
                    continue
                rs_result = rs_fn(*tc.args)
                if not check(rs_result, tc.expected):
                    print(f"  FAIL  {tc.name} [RUST PARITY]")
                    print(f"         Python: {py_result!r}")
                    print(f"         Rust:   {rs_result!r}")
                    print(f"         Expected: {tc.expected!r}")
                    failed += 1
                    continue

            print(f"  PASS  {tc.name}")
            passed += 1

        except Exception as e:
            print(f"  ERROR {tc.name}")
            print(f"         {e}")
            traceback.print_exc(limit=2)
            failed += 1

    print("=" * 65)
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 65)
    return failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rustify-ml all-fixtures test runner")
    parser.add_argument("--with-rust", action="store_true",
                        help="Also verify Rust extension produces identical results")
    args = parser.parse_args()
    sys.exit(run_tests(args.with_rust))
