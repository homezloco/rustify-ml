# rustify-ml: Complete Build Plan

## Polyglot Profiler and Accelerator - Python to Rust Bridge for AI/ML

---

## 1. Current State Snapshot (Updated 2026-02-18)

| File | Status | Notes |
|------|--------|-------|
| src/main.rs | ✅ Done | Clap CLI; lib crate imports; AccelerateRow summary table |
| src/lib.rs | ✅ Done | Public lib crate; exposes all modules for integration tests |
| src/input.rs | ✅ Done | File + stdin + git2 clone |
| src/utils.rs | ✅ Done | InputSource, Hotspot, ProfileSummary, TargetSpec, GenerationResult, AccelerateRow, print_summary |
| src/profiler.rs | ✅ Done | cProfile harness; detect_python (python3→python); version pre-flight; stdlib filter |
| src/analyzer.rs | ✅ Done | Hotspot threshold filter; ml_mode reason tagging |
| src/generator.rs | ✅ Done | AST walk; assign init; subscript assign; list init; range(a,b); nested for loops (translate_body_inner depth-aware); infer_assign_type; translate_for_iter; fallback tracking |
| src/builder.rs | ✅ Done | cargo_check_generated (fast-fail); maturin develop; run_benchmark (Python timing harness; speedup table) |
| Cargo.toml | ✅ Done | lib + bin targets; all deps; optional ndarray/tch features |
| tests/integration.rs | ✅ Done | 8 integration tests for generate() pipeline |
| tests/integration_cli.rs | ✅ Done | 3 CLI end-to-end tests (dry-run; python_available guard) |
| examples/euclidean.py | ✅ Done | Euclidean distance with __main__ block for profiler |
| examples/slow_tokenizer.py | ✅ Done | BPE tokenizer loop fixture |
| examples/matrix_ops.py | ✅ Done | matmul + dot_product fixtures |
| examples/image_preprocess.py | ✅ Done | normalize_pixels + apply_gamma fixtures |
| examples/data_pipeline.py | ✅ Done | CSV parsing + running_mean + zscore fixtures |
| .github/workflows/ci.yml | ✅ Done | dtolnay/rust-toolchain; Python 3.11; maturin; 3-OS matrix; cargo cache |
| README.md | ✅ Done | Full docs: usage, CLI ref, translation table, architecture, roadmap |
| plan.md | ✅ Done | This file |

**Build status:** `cargo fmt && cargo check` passes (verified in WSL 2026-02-18)

---

## 2. Architecture Flow

```
CLI args (Clap)
    → input::load_input()        # File | stdin snippet | git2 clone
    → profiler::profile_input()  # cProfile subprocess; python3→python fallback; stdlib filter
    → analyzer::select_targets() # Threshold filter; ml_mode tagging
    → generator::generate()      # AST walk; Rust codegen; len-check guards; fallback tracking
    → builder::cargo_check_generated()  # Fast-fail: cargo check on generated crate
    → builder::build_extension() # maturin develop --release
    → utils::print_summary()     # ASCII table to stdout
```

---

## 3. Prioritized Next Tasks

### Task 1 (CRITICAL): Translation Robustness — Zero Fallback Demo
- [x] Translate Assign with float literal: `total = 0.0` → `let mut total: f64 = 0.0;`
- [x] Translate subscript assign: `result[i] = val` → `result[i] = val;`
- [x] Translate list init: `result = [0.0] * n` → `let mut result = vec![0.0f64; n];`
- [x] Translate range two-arg form: `range(a, b)` → `a..b`
- [x] Add `infer_assign_type()` helper (Float → `: f64`, Int → `: i64`)
- [x] Add `translate_for_iter()` helper (range(n), range(a,b), fallback)
- [x] Add unit tests: float_assign_init, subscript_assign, list_init, range_two_args, normalize_pixels
- [x] Achieve 0 fallbacks on `examples/euclidean.py`
- [x] Verify 0 fallbacks on `dot_product` in `matrix_ops.py` (run `cargo test`)
- [x] Translate nested for loops: `for i in range(n): for j in range(n):` → nested for (translate_body_inner depth-aware recursion)
- [x] Add matmul nested-loop unit test (test_translate_matmul_nested_loops)
- [x] Verify matmul generates 0 fallbacks end-to-end (run `cargo test`)

### Task 2 (HIGH): More Unit Tests
- [x] Add snapshot test: assert generated lib.rs matches expected golden file
- [x] Add unit test: translate_body for nested for loop (expect partial/fallback with clear comment)
- [x] Add profiler unit test with mock cProfile stdout output (parse_hotspots helper)
- [x] Add integration test: normalize_pixels achieves 0 fallbacks with new list init + subscript assign

### Task 3 (HIGH): CLI Polish ✅
- [x] Add `--list-targets` flag: profile only, print hotspots, no generation
- [x] Add `--function` flag: manually specify function name (skip profiler)
- [x] Add `--iterations` flag (default 100): control profiler loop count
- [x] `print_hotspot_table()` added to utils.rs
- [x] `profile_input_with_iterations()` added to profiler.rs

### Task 4 (MEDIUM): Profiler Polish
- [x] `python3` → `python` fallback (cross-platform)
- [x] Python version pre-flight check (warn if < 3.10)
- [x] Filter `<built-in>` and `<frozen>` frames from hotspot list
- [x] Add `--profile-only` flag: run profiler, print hotspots, exit

### Task 5 (MEDIUM): ndarray Optional Feature ✅
- [x] Trigger on `uses_numpy=true` (detects_numpy() checks import numpy/from numpy/import np)
- [x] Generated signature: `PyReadonlyArray1<f64>` when ml_mode + numpy detected
- [x] Add numpy dep to generated Cargo.toml when feature active
- [x] `generate_ml()` public API; `--ml-mode` flag wires into main.rs
- [x] Tests: test_ndarray_mode_replaces_vec_params, test_ndarray_mode_no_numpy_import_stays_vec

### Task 6 (LOW): README + Demo
- [x] Full README with usage, CLI ref, translation table, architecture, roadmap
- [ ] Add timing demo section: before/after Python vs Rust for euclidean (actual numbers)
- [ ] Add GIF/screenshot of CLI output
- [ ] Add crates.io badge once published

### Task 7 (LOW): Release Prep
- [x] Add `benches/speedup.rs` with Criterion before/after benchmarks
- [ ] `cargo publish --dry-run` check
- [x] Write CHANGELOG.md
- [ ] Tag v0.1.0 release

### Task 8 (MEDIUM): rustify-stdlib packaging
- [x] Add `rustify-stdlib/` crate to workspace and maturin config
- [x] Add Python usage example (maturin develop) and parity tests at crate level
- [ ] Publish/push once wired; share import snippet

---

## 4. Translation Patterns Reference

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

**Untranslatable** (warn + skip): `eval()`, `exec()`, `getattr()`, `async def`, class self mutation

---

## 5. CLI Summary Table Format (Target Output)

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

## 6. Key Commands

```bash
# Build and test (run from rustify-ml/ in WSL)
cargo fmt && cargo check
cargo test
cargo clippy -- -D warnings

# Run CLI (dry-run: generate code, no build)
cargo run -- accelerate --file examples/euclidean.py --output dist --threshold 0 --dry-run
cargo run -- accelerate --file examples/matrix_ops.py --output dist --threshold 0 --dry-run
cargo run -- accelerate --file examples/image_preprocess.py --output dist --ml-mode --dry-run

# Install and use generated extension
cd dist/rustify_ml_ext && maturin develop --release
python -c "import rustify_ml_ext; print(rustify_ml_ext.euclidean([0.0,3.0,4.0],[0.0,0.0,0.0]))"
# → 5.0
```

---

## 7. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Generated Rust does not compile | High | High | `cargo_check_generated()` fast-fail (done); dry-run mode; fallback tracking |
| Nested loop translation complexity | High | Medium | Warn partial; suggest rayon in comment; matmul in progress |
| maturin not installed | High | High | Clear error message with install hint |
| python vs python3 command | Medium | Medium | `detect_python()` tries python3 first (done) |
| Python version < 3.10 | Medium | Medium | `check_python_version()` warns if < 3.10 (done) |
| rustpython-parser 0.4.0 API changes | Low | Medium | Pinned; integration tests catch regressions |
| Complex Python fails analysis | High | Low | Warn+skip; translate simple functions only |
| py-spy needs elevated privileges on Windows | High | Medium | cProfile harness used instead (done) |

---

## 8. Windows Development Note

The project builds and tests in **WSL** (Windows Subsystem for Linux). Running `cargo test` directly in Windows CMD requires Visual Studio Build Tools (`link.exe`). Use WSL:

```bash
cd /mnt/d/WindsurfProjects/rustify/rustify-ml
cargo fmt && cargo check   # ✅ verified passing
cargo test
```

---

*Updated 2026-02-18 — cargo fmt && cargo check passes; translation coverage expanded; cargo check on generated crate; README fully updated*
