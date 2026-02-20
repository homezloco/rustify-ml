# rustify-ml

> **Auto-accelerate Python ML hotspots with Rust.** Profile → Identify → Generate → Build — drop-in PyO3 extensions with no manual rewrite.

> **20x faster `running_mean`. 15x faster `convolve1d`. 12x faster BPE tokenizer. Zero manual Rust.**

Install: `cargo install rustify-ml` (from crates.io) — also `pip install maturin` for builds.

[![CI](https://github.com/homezloco/rustify-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/homezloco/rustify-ml/actions)
[![crates.io](https://img.shields.io/crates/v/rustify-ml.svg)](https://crates.io/crates/rustify-ml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## What It Does

`rustify-ml` is a CLI tool that:

1. **Profiles** your Python file using `cProfile` (no elevated privileges required)
2. **Identifies** CPU hotspots above a configurable threshold
3. **Generates** safe Rust + PyO3 stubs with length-check guards and type inference
4. **Builds** an installable Python extension via `maturin develop --release`

**Bridge:** Python (cProfile) → hotspot selection → Rust codegen (PyO3) → maturin wheel → editable install → parity tests + benchmarks. No manual glue required.

Typical speedups: **5–100x** on pure-Python loops (tokenizers, matrix ops, image preprocessing, data pipelines).

---

## Quick Start

```bash
# Install dependencies
pip install maturin
cargo install rustify-ml         # once published on crates.io
cargo install --path rustify-ml   # or: cargo build --release

# Accelerate a Python file (dry-run: generate code, skip build)
rustify-ml accelerate --file examples/euclidean.py --output dist --threshold 0 --dry-run

# Full run: profile → generate → build extension
rustify-ml accelerate --file examples/euclidean.py --output dist --threshold 10

# Install and use the generated extension
cd dist/rustify_ml_ext && maturin develop --release
python -c "from rustify_ml_ext import euclidean; print(euclidean([0.0,3.0,4.0],[0.0,0.0,0.0]))"
# → 5.0

# Validate parity + speedups
python -X utf8 tests/test_all_fixtures.py --with-rust
python benches/compare.py --with-rust
```

---

## CLI Reference

```
rustify-ml accelerate [OPTIONS]

Input (one required):
  --file <PATH>          Python file to profile and accelerate
  --snippet              Read Python code from stdin
  --git <URL>            Git repo URL to clone and analyze
  --git-path <PATH>      Path within the git repo (required with --git)

Profiler:
  --threshold <FLOAT>    Minimum hotspot % to target [default: 10.0]
                        Tip: set to 0.0 to include all defined functions (parsed from the source)
  --iterations <N>       Profiler loop count for better sampling [default: 100]
  --list-targets         Profile only: print hotspot table and exit (no codegen)
  --function <NAME>      Skip profiler, target a specific function by name

Generation:
  --output <DIR>         Output directory for generated extension [default: dist]
  --ml-mode              Enable ML-focused heuristics (numpy → PyReadonlyArray1)
  --dry-run              Generate code without building (inspect before install)
  --benchmark            After building, run Python timing harness + speedup table
  --no-regen             Skip code regeneration; only rebuild the existing extension

Logging:
  -v / -vv               Increase verbosity (debug / trace)
```

### New in latest build

| Flag | What it does |
|------|-------------|
| `--list-targets` | Profile only, print ranked hotspot table, exit — no code generated |
| `--function <name>` | Skip profiler entirely, target one function by name (100% weight) |
| `--iterations <n>` | Control how many times the profiler loops the script (default: 100) |
| `--ml-mode` | Detect numpy imports → use `PyReadonlyArray1<f64>` + add numpy dep to Cargo.toml |
| `--threshold 0` | Force inclusion of all defined functions (parser-based), even if profiler reports 0% |
| `--no-regen` | Skip code regeneration; only rebuild the existing `dist/rustify_ml_ext` (prevents overwriting manual edits) |

### BPE Tokenizer Demo

One of the best targets for rustify-ml is the BPE (Byte-Pair Encoding) encode loop — the same algorithm used by tiktoken (OpenAI) and HuggingFace tokenizers. The inner merge pass is O(n²) in Python and translates cleanly to Rust `Vec<usize>` + `while` loops:

```bash
# Profile and generate Rust stubs for the BPE tokenizer
cargo run -- accelerate \
  --file examples/bpe_tokenizer.py \
  --function count_pairs \
  --output dist \
  --dry-run

# Or let the profiler find hotspots automatically
cargo run -- accelerate \
  --file examples/bpe_tokenizer.py \
  --threshold 5 \
  --output dist \
  --benchmark
```

**Latest benchmark snapshot** (WSL, CPython 3.12, `python benches/compare.py --with-rust`):
```
  Function                            |  Python us |    Rust us |  Speedup
  ------------------------------------+------------+------------+---------
  euclidean (n=1000)                  |       55.8 |       20.8 |     2.7x
  dot_product (n=1000)                |       45.8 |       19.4 |     2.4x
  normalize_pixels (n=1000)           |       53.2 |       25.1 |     2.1x
  running_mean (n=500, w=10)          |      376.9 |       18.7 |    20.2x
  count_pairs (n=500)                 |       88.3 |       61.5 |     1.4x
  bpe_encode (len=100)                |       11.1 |        0.9 |    12.3x
  standard_scale (n=1000)             |       55.1 |       27.2 |     2.0x
  min_max_scale (n=1000)              |       60.6 |       28.6 |     2.1x
  l2_normalize (n=1000)               |       95.5 |       29.4 |     3.2x
  convolve1d (n=1000, k=5)            |      329.8 |       21.0 |    15.7x
  moving_average (n=1000, w=10)       |      471.5 |       30.8 |    15.3x
  diff (n=1000)                       |       58.5 |       16.1 |     3.6x
  cumsum (n=1000)                     |       40.0 |       27.3 |     1.5x
```

After `maturin develop --release`, re-run `python benches/compare.py --with-rust` to refresh numbers for your machine.

## Examples

```bash
# Snippet from stdin
echo "def dot(a, b):\n    return sum(x*y for x,y in zip(a,b))" | \
  rustify-ml accelerate --snippet --output dist --dry-run

# Git repo (shallow clone, analyze one file)
rustify-ml accelerate \
  --git https://github.com/huggingface/transformers \
  --git-path examples/slow_preproc.py \
  --output dist --threshold 5

# ML mode (numpy/torch type hints in generated stubs)
rustify-ml accelerate --file examples/image_preprocess.py --ml-mode --output dist --dry-run
```

### Timing Demo (euclidean)

Baseline vs Rust extension on WSL, CPython 3.12, Ryzen 7:

| Function | Input | Python (us) | Rust (us) | Speedup |
|----------|-------|-------------|-----------|---------|
| euclidean | n=1_000 | 55.8 | 20.8 | 2.7x |

Reproduce:

```bash
python -X utf8 benches/compare.py --function euclidean --with-rust
```

### ML-mode benchmarks (numpy arrays)

`--ml-mode` is optimized for numeric array inputs (numpy). Use it when your hotspots already operate on `np.ndarray` or can be cheaply converted to arrays. Example (image preprocessing):

```bash
python -X utf8 benches/compare.py --function normalize_pixels --with-rust --ml-mode
```

Sample (WSL, CPython 3.12, numpy arrays):

| Function | Input | Python (us) | Rust (us) | Speedup |
|----------|-------|-------------|-----------|---------|
| normalize_pixels | n=1_000 | 53.2 | 25.1 | 2.1x |
| convolve1d | n=1_000, k=5 | 329.8 | 21.0 | 15.7x |
| running_mean | n=500, w=10 | 376.9 | 18.7 | 20.2x |

Best practices: keep data as `np.ndarray` before calling Rust, avoid per-call Python↔Rust conversions, and rerun `benches/compare.py --with-rust --ml-mode` on your hardware to refresh numbers.

### CLI Output (screenshot)

![CLI demo](cli.gif)

### Using `rustify-stdlib` directly

```bash
pip install maturin
pip install rustify-stdlib  # once published
python - <<'PY'
import rustify_stdlib as rs
print(rs.euclidean([0.0,3.0,4.0],[0.0,0.0,0.0]))
print(rs.dot_product([1.0,2.0],[3.0,4.0]))
PY
```

---

## Example Output

After running `accelerate`, rustify-ml prints a summary table to stdout:

```
Accelerated 3/4 targets (1 fallback)

Func               | Line | % Time | Translation | Status
-------------------+------+--------+-------------+---------
euclidean          |  1   | 42.1%  | Full        | Success
dot_product        |  18  | 31.8%  | Full        | Success
matmul             |  7   | 20.4%  | Partial     | Fallback (nested loop)
normalize_pixels   |  24  |  5.7%  | Full        | Success

Generated: dist/rustify_ml_ext/
Install:   cd dist/rustify_ml_ext && maturin develop --release
```

---

## Translation Patterns

| Python Pattern | Rust Translation | Status |
|----------------|-----------------|--------|
| `for i in range(len(x)):` | `for i in 0..x.len() {` | ✅ Done |
| `total += a * b` | `total += a * b;` | ✅ Done |
| `return x ** 0.5` | `return (x).powf(0.5);` | ✅ Done |
| `a[i] - b[i]` | `a[i] - b[i]` | ✅ Done |
| `total = 0.0` | `let mut total: f64 = 0.0;` | ✅ Done |
| `result[i] = val` | `result[i] = val;` | ✅ Done |
| `result = [0.0] * n` | `let mut result = vec![0.0f64; n];` | ✅ Done |
| `range(a, b)` | `a..b` | ✅ Done |
| `for i in range(n): for j...` | nested for loops | ✅ Done |
| `[f(x) for x in xs]` | `xs.iter().map(f).collect()` | ✅ Done |
| `np.array` params | `PyReadonlyArray1<f64>` (via `--ml-mode`) | ✅ Done |

**Untranslatable** (warns + skips): `eval()`, `exec()`, `getattr()`, `async def`, class self mutation

---

## Generated Code Example

For `examples/euclidean.py`:

```python
def euclidean(p1, p2):
    total = 0.0
    for i in range(len(p1)):
        diff = p1[i] - p2[i]
        total += diff * diff
    return total ** 0.5
```

rustify-ml generates:

```rust
use pyo3::prelude::*;

#[pyfunction]
/// Auto-generated from Python hotspot `euclidean` at line 1 (100.00%): 100% hotspot
pub fn euclidean(py: Python, p1: Vec<f64>, p2: Vec<f64>) -> PyResult<f64> {
    let _ = py;
    if p1.len() != p2.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("length mismatch"));
    }
    let mut total = 0.0f64;
    for i in 0..p1.len() {
        // ...
        total += diff * diff;
    }
    Ok((total).powf(0.5))
}
```

---

## Timing Demo

Run the built-in benchmark after building the extension:

```bash
# Build the extension, then benchmark euclidean distance
rustify-ml accelerate --file examples/euclidean.py --output dist --threshold 0 --benchmark

# Or manually after maturin develop:
cd dist/rustify_ml_ext && maturin develop --release && cd ../..
rustify-ml accelerate --file examples/euclidean.py --output dist --threshold 0 --benchmark
```

Expected output (from `benches/compare.py --with-rust`):

```
================================================================================
  rustify-ml benchmark results
================================================================================
  Function                            |  Python us |    Rust us |  Speedup
  ------------------------------------+------------+------------+---------
  running_mean (n=500, w=10)          |      376.9 |       18.7 |    20.2x
  convolve1d (n=1000, k=5)            |      329.8 |       21.0 |    15.7x
  moving_average (n=1000, w=10)       |      471.5 |       30.8 |    15.3x
  bpe_encode (len=100)                |       11.1 |        0.9 |    12.3x
  euclidean (n=1000)                  |       55.8 |       20.8 |     2.7x
================================================================================
```

> Numbers measured on WSL, CPython 3.12. Actual speedup depends on Python version, CPU, and input size.
> Loop-heavy functions (sliding window, convolution, tokenizers) see the largest gains.

---

## Example Files

| File | Description | Key Patterns |
|------|-------------|-------------|
| `examples/euclidean.py` | Euclidean distance | `range(len(x))`, `**`, accumulator |
| `examples/matrix_ops.py` | Matrix multiply + dot product | nested loops, subscript assign |
| `examples/image_preprocess.py` | Pixel normalize + gamma | `[0.0] * n`, subscript assign |
| `examples/bpe_tokenizer.py` | BPE encode (tiktoken-style) | while loop, HashMap merge rank |
| `examples/slow_tokenizer.py` | BPE-style tokenizer fixture | while loop, dict lookup |
| `examples/data_pipeline.py` | CSV parse + running mean | string ops, sliding window |
| `examples/signal_processing.py` | convolve1d, moving_average, diff, cumsum | nested loops, 1D signal ops |
| `examples/sklearn_scaler.py` | standard_scale, min_max_scale, l2_normalize | element-wise Vec ops |

---

## Architecture

```
CLI args (Clap)
    → input::load_input()     # File | stdin snippet | git2 clone
    → profiler::profile_input()  # cProfile subprocess; python3→python fallback
    → analyzer::select_targets() # Threshold filter; ml_mode tagging
    → generator::generate()   # AST walk; Rust codegen; len-check guards
    → builder::build_extension() # cargo check (fast-fail) → maturin develop
    → print_summary()         # ASCII table to stdout
```

**Modules:**

| Module | Responsibility |
|--------|---------------|
| `input.rs` | Load Python from file, stdin, or git repo |
| `profiler.rs` | Run cProfile via Python subprocess; parse hotspots |
| `analyzer.rs` | Filter hotspots by threshold; apply ML heuristics |
| `generator.rs` | Walk Python AST; emit Rust + PyO3 stubs |
| `builder.rs` | `cargo check` generated crate; spawn `maturin develop` |
| `utils.rs` | Shared types; ASCII summary table |

---

## Development

### Prerequisites

- Rust 1.75+ stable (`rustup update stable`)
- Python 3.10+ on PATH (`python3` or `python`)
- `pip install maturin`

### Build & Test

```bash
# From rustify-ml/ directory (or use WSL on Windows)
cargo fmt && cargo check
cargo test
cargo clippy -- -D warnings
```

### Run CLI in dev mode

```bash
# Dry-run: generate code, inspect, no build
cargo run -- accelerate --file examples/euclidean.py --output dist --threshold 0 --dry-run

# Full run (requires maturin)
cargo run -- accelerate --file examples/euclidean.py --output dist --threshold 0

# Verbose output
cargo run -- accelerate --file examples/euclidean.py --output dist -vv --dry-run
```

### Windows Note

The project builds and tests in **WSL** (Windows Subsystem for Linux). Running `cargo test` directly in Windows CMD requires Visual Studio Build Tools (`link.exe`). Use WSL for development:

```bash
cd /mnt/d/WindsurfProjects/rustify/rustify-ml
cargo fmt && cargo check
cargo test
```

---

## Roadmap

See [plan.md](plan.md) for the full prioritized task list. High-level:

1. ✅ **Core pipeline** — profile → analyze → generate → build
2. ✅ **Translation coverage** — assign init, subscript assign, list init, range forms, nested for loops
3. ✅ **While loop translation** — `while changed:`, `while i < len(x):` → Rust while
4. ✅ **Safety** — length-check guards, cargo check on generated crate
5. ✅ **Profiler robustness** — python3/python fallback, version pre-flight, stdlib filter
6. ✅ **CLI polish** — `--list-targets`, `--function`, `--iterations`, `--benchmark`
7. ✅ **ndarray feature** — `--ml-mode` + numpy import → `PyReadonlyArray1<f64>` params
8. ✅ **BPE tokenizer fixture** — `examples/bpe_tokenizer.py` + integration tests
9. ✅ **Benchmark script** — `benches/compare.py` (Python baseline + `--with-rust` mode)
10. ✅ **List comprehension** — `[f(x) for x in xs]` → `xs.iter().map(f).collect()`
11. ✅ **Criterion benchmarks** — `benches/speedup.rs` with Criterion (html reports; euclidean/dot_product/moving_average)
12. 📋 **v0.1.0 release** — crates.io publish, CHANGELOG, GitHub release (see CHANGELOG.md)

---

## License

MIT — see [LICENSE](LICENSE)

> ⚠️ **Generated code requires review.** rustify-ml emits Rust stubs as a starting point. Always review generated `lib.rs` before deploying, especially for fallback-translated functions (marked with `// fallback: echo input`).
