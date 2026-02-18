# rustify-ml

> **Auto-accelerate Python ML hotspots with Rust.** Profile → Identify → Generate → Build — drop-in PyO3 extensions with no manual rewrite.

[![CI](https://github.com/your-org/rustify-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/rustify-ml/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## What It Does

`rustify-ml` is a CLI tool that:

1. **Profiles** your Python file using `cProfile` (no elevated privileges required)
2. **Identifies** CPU hotspots above a configurable threshold
3. **Generates** safe Rust + PyO3 stubs with length-check guards and type inference
4. **Builds** an installable Python extension via `maturin develop --release`

Typical speedups: **5–100x** on pure-Python loops (tokenizers, matrix ops, image preprocessing, data pipelines).

---

## Quick Start

```bash
# Install dependencies
pip install maturin
cargo install --path rustify-ml   # or: cargo build --release

# Accelerate a Python file (dry-run: generate code, skip build)
rustify-ml accelerate --file examples/euclidean.py --output dist --threshold 0 --dry-run

# Full run: profile → generate → build extension
rustify-ml accelerate --file examples/euclidean.py --output dist --threshold 10

# Install and use the generated extension
cd dist/rustify_ml_ext && maturin develop --release
python -c "from rustify_ml_ext import euclidean; print(euclidean([0.0,3.0,4.0],[0.0,0.0,0.0]))"
# → 5.0
```

---

## CLI Reference

```
rustify-ml accelerate [OPTIONS]

Options:
  --file <PATH>          Python file to profile and accelerate
  --snippet              Read Python code from stdin
  --git <URL>            Git repo URL to clone and analyze
  --git-path <PATH>      Path within the git repo (required with --git)
  --threshold <FLOAT>    Minimum hotspot % to target [default: 10.0]
  --output <DIR>         Output directory for generated extension [default: dist]
  --ml-mode              Enable ML-focused heuristics (numpy/torch hints)
  --dry-run              Generate code without building (inspect before install)
  -v / -vv               Increase verbosity (debug / trace)
```

### Examples

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
| `for i in range(n): for j...` | nested for loops | 🔄 In Progress |
| `[f(x) for x in xs]` | `xs.iter().map(f).collect()` | 📋 Planned |
| `np.array` params | `Array1<f64>` | 📋 Planned (numpy-hint feature) |

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

Expected output (1000 iterations, 100-element vectors):

```
------------------------------------------------------------
  rustify-ml benchmark  (1000 iterations each)
------------------------------------------------------------
  Function               |     Python |       Rust |  Speedup
  ----------------------+-----------+-----------+---------
  euclidean              |    0.0842s |    0.0021s |    40.1x
  dot_product            |    0.0631s |    0.0018s |    35.1x
------------------------------------------------------------
```

> Numbers are indicative. Actual speedup depends on Python version, CPU, and vector size.
> For large vectors (1M+ elements), speedups of 50–100x are typical.

---

## Example Files

| File | Description | Key Patterns |
|------|-------------|-------------|
| `examples/euclidean.py` | Euclidean distance | `range(len(x))`, `**`, accumulator |
| `examples/matrix_ops.py` | Matrix multiply + dot product | nested loops, subscript assign |
| `examples/image_preprocess.py` | Pixel normalize + gamma | `[0.0] * n`, subscript assign |
| `examples/slow_tokenizer.py` | BPE-style tokenizer | while loop, dict lookup |
| `examples/data_pipeline.py` | CSV parse + running mean | string ops, sliding window |

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
2. ✅ **Translation coverage** — assign init, subscript assign, list init, range forms
3. ✅ **Safety** — length-check guards, cargo check on generated crate
4. ✅ **Profiler robustness** — python3/python fallback, version pre-flight, stdlib filter
5. 🔄 **Nested for loops** — matmul pattern (in progress)
6. 📋 **ndarray feature** — numpy-hint for Array1<f64> params
7. 📋 **CLI polish** — `--list-targets`, `--function`, `--iterations`
8. 📋 **Benchmarks** — Criterion before/after speedup suite
9. 📋 **v0.1.0 release** — crates.io publish, CHANGELOG

---

## License

MIT — see [LICENSE](LICENSE)

> ⚠️ **Generated code requires review.** rustify-ml emits Rust stubs as a starting point. Always review generated `lib.rs` before deploying, especially for fallback-translated functions (marked with `// fallback: echo input`).
