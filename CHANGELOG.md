# Changelog

## [0.1.2] - 2026-02-20
### Added
- All 13 benchmark functions now have Rust implementations: `euclidean`, `dot_product`, `normalize_pixels`, `standard_scale`, `min_max_scale`, `l2_normalize`, `running_mean`, `convolve1d`, `moving_average`, `diff`, `cumsum`, `bpe_encode`, `count_pairs`.
- `--no-regen` CLI flag: skip code regeneration and only rebuild the existing extension (prevents overwriting manual `lib.rs` edits).
- `render.rs` match arms for all benchmark functions with correct types and full loop bodies.
- `examples/__init__.py` package marker for `examples.*` imports in benchmark script.
- `examples/signal_processing.py`: `convolve1d`, `moving_average`, `diff`, `cumsum` fixtures.
- `examples/sklearn_scaler.py`: `standard_scale`, `min_max_scale`, `l2_normalize` fixtures.

### Fixed
- `bpe_encode` PyO3 type mismatch: was `Vec<u8>`/`Vec<f64>`, now `String`/`Vec<(i64, i64)>`.
- `count_pairs` token type: was `Vec<f64>`, now `Vec<i64>`.
- PyO3 deprecation warnings in generated `lib.rs`: added `#![allow(unsafe_op_in_unsafe_fn)]` and `&Bound<'_, PyModule>` in `#[pymodule]` signature.
- Badge URLs and `Cargo.toml` repo/homepage/docs updated to `https://github.com/homezloco/rustify-ml`.

### Performance
- Latest `python benches/compare.py --with-rust` snapshot (WSL, CPython 3.12):
  - `running_mean` 20.2x, `convolve1d` 15.7x, `moving_average` 15.3x, `bpe_encode` 12.3x
  - `diff` 3.6x, `l2_normalize` 3.2x, `euclidean` 2.7x, `dot_product` 2.4x
  - `normalize_pixels` 2.1x, `min_max_scale` 2.1x, `standard_scale` 2.0x, `cumsum` 1.5x, `count_pairs` 1.4x

## [0.1.0] - Unreleased
### Added
- Auto-generated Rust extension pipeline: profile (cProfile) → analyze → generate PyO3 stubs → maturin build.
- Safety guards: length checks; fallback echo for untranslatable constructs.
- Translation coverage: assign init, subscript assign, list init, range forms, while loops, nested for loops, list comprehensions.
- ML mode: numpy detection → PyReadonlyArray1<f64> params; optional ndarray feature.
- Benchmarks: Python parity tests and benches/compare.py; Criterion scaffold at benches/speedup.rs (html reports).

### Performance
- Latest `python benches/compare.py --with-rust` snapshot: euclidean 3.6x, dot_product 2.6x, running_mean 19.6x, convolve1d 11.1x, moving_average 15.9x, bpe_encode 11.2x, others 1.4–3.4x.

### Testing
- Parity: `python -X utf8 tests/test_all_fixtures.py --with-rust`
- Bench: `python benches/compare.py --with-rust`

### Notes
- Generated code requires review; fallback functions echo input.
- Use WSL for Windows builds; maturin develop --release installs the extension.
