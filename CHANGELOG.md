# Changelog

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
