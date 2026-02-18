# rustify-ml: Build Plan (Current)

## 1. Current State Snapshot

| File | Status | Notes |
|------|--------|-------|
| src/main.rs | Done | Clap CLI, tracing, accelerate flow wired (input→profile→targets→generate→build) |
| src/input.rs | Done | File/stdin; git clone+path supported |
| src/utils.rs | Done | InputSource, ProfileSummary, TargetSpec, GenerationResult (fallback count) |
| src/profiler.rs | Working | cProfile subprocess → hotspots filtered by threshold |
| src/analyzer.rs | Working | Select targets by hotspot pct with ml_mode tag |
| src/generator.rs | Working | rustpython-parser Suite::parse; type inference (int→usize, float→f64, numpy/torch hints→Vec<f64>); range/len/binop/pow translation; len checks; fallback tracking; unit tests added |
| src/builder.rs | Working | maturin develop --release; logs fallback count |
| Cargo.toml | Updated | anyhow, clap, tracing, git2, rustpython-parser, heck, tempfile; (ndarray/tch not yet added) |
| tests | Partial | Generator unit tests (range/len, pow, len-check); no integration yet |

## 2. Architecture Flow (Actual)

CLI → input::load_input (file/snippet/git) → profiler (cProfile) → analyzer::select_targets → generator::generate (PyO3 crate, len guards, fallback logging) → builder::build_extension (maturin).

## 3. Prioritized Tasks (Next)
1) Translation robustness + explicit euclidean win: ensure 0 fallbacks for pure-Python euclidean_distance (len guard → PyValueError, for i in 0..len, total += diff*diff, return powf(0.5)); broaden binops/returns and reduce fallbacks overall.
2) Tests & fixtures: add examples/euclidean.py; integration test (dry-run snapshot OK); expand unit coverage for translate_body/expr (range loop/binop/return).
3) CLI warnings/summary: surface fallback_functions prominently in accelerate output; remind users to review generated code; include install hint when build succeeds.
4) Type/signature polish: optional ndarray hint when numpy detected; better slice types for Vec-like params; consider GIL release for loop-heavy code.
5) Docs/CI: README demo with before/after timing; GitHub Actions (fmt/clippy/test); prep release artifacts.
6) Performance polish: profiling duration/rate flags; cache temp dirs; parallel generation for multiple targets.

## 4. Dependencies (current + upcoming)
- Current: anyhow, clap, tracing, tracing-subscriber, tempfile, git2, rustpython-parser (pin 0.4.0; legacy but stable), heck.
- Upcoming (as needed): quote/syn/prettyplease (cleaner codegen/formatting), ndarray (numpy paths), tch (optional torch), serde/serde_json (config/reporting), criterion (bench).
- Parser note: rustpython-parser 0.4.0 is legacy; if modern syntax issues arise (3.11+), evaluate ruff_python_parser post-MVP.

## 5. Notes
- Profiling uses cProfile (py-spy skipped for Windows privilege issues).
- Fallbacks are logged; generator tracks count and builder reports it.
- Profiling harness: keep stdout JSON/log approach for Stats; avoid binary .prof parsing; ensure parsing of printed stats/JSON is robust.

---

## 4. Module Implementation Plan

### 4.1 src/utils.rs - Expand Shared Types

    pub struct Hotspot { func_name, module, line_start, line_end, pct_time: f32, call_count: u64 }
    pub struct Translatable { hotspot, params: Vec<ParamHint>, return_hint: TypeHint, ml_hints: MlHints, body_complexity: Complexity }
    pub enum TypeHint { Float64, Int64, Bool, Str, ListOf(Box<TypeHint>), NdArray { dtype, ndim }, Unknown }
    pub struct MlHints { uses_numpy, uses_torch, uses_sklearn, is_pure, has_inner_loop: bool }
    pub enum Complexity { Simple, Moderate, Complex }

### 4.2 src/profiler.rs - cProfile Harness

Strategy: Write temp Python harness, run N times, dump cProfile stats to JSON, parse in Rust.

    pub fn profile_code(source: &InputSource, threshold: f32) -> Result<Vec<Hotspot>>
    fn write_harness(code: &str, func_name: &str, temp_dir: &Path) -> Result<PathBuf>
    fn run_python_harness(harness_path: &Path) -> Result<String>
    fn parse_profile_output(json: &str, threshold: f32) -> Result<Vec<Hotspot>>

Harness template: import cProfile json sys; pr.enable(); for _ in range(100): func_call; pr.disable(); print(json.dumps(stats))
Windows: use python not python3; detect via where python at startup.

### 4.3 src/analyzer.rs - AST Analysis

Strategy: Use rustpython-parser to parse Python source into AST, walk to find hotspot functions.

    pub fn analyze_hotspots(source: &str, hotspots: &[Hotspot], ml_mode: bool) -> Result<Vec<Translatable>>
    fn find_function_at_line / infer_param_types / check_purity / detect_ml_hints / score_complexity

Heuristics: import numpy/torch/sklearn -> flags; StmtFor in body -> has_inner_loop; no eval/exec -> is_pure

### 4.4 src/generator.rs - Rust + PyO3 Codegen

Strategy: Use quote! macro for token streams, prettyplease for formatting.

    pub fn generate_stubs(translatables: &[Translatable], output_dir: &Path, dry_run: bool) -> Result<GeneratedOutput>
    pub struct GeneratedOutput { lib_rs, cargo_toml, python_wrapper, functions: Vec<String> }

Generated lib.rs: #[pyfunction] fn func_rs(py: Python, input: Vec<f64>) -> PyResult<Vec<f64>>
ML-smart: uses_numpy -> ndarray + Array1<f64>; has_inner_loop -> py.allow_threads()

### 4.5 src/builder.rs - Maturin Integration

    pub fn build_extension(output_dir: &Path, dry_run: bool) -> Result<BuildResult>
    pub struct BuildResult { success, wheel_path, install_cmd, elapsed_secs }

Maturin: Command::new(maturin).args([develop,--release]).current_dir(output_dir).output()
Pre-flight: check maturin version; check python >= 3.10; check Cargo.toml exists

### 4.6 src/input.rs - Git Mode

    fn clone_git_repo(url, path) -> Result<InputSource>
    // tempfile::TempDir auto-cleans on drop; Repository::clone(url, temp_dir.path())
    // InputSource::Git { url, code, _temp: TempDir }

---

## 5. Python to Rust Translation Patterns

| Python Pattern | Rust Translation | Crate |
|----------------|-----------------|-------|
| for x in list: acc+=f(x) | iter().map().sum() | std |
| [f(x) for x in xs] | xs.iter().map(f).collect() | std |
| np.array([...]) | Array1::from_vec(...) | ndarray |
| np.dot(a, b) | a.dot(&b) | ndarray |
| np.sum(arr) | arr.sum() | ndarray |
| str.split().lower() | split_whitespace().map(to_lowercase) | std |
| dict[key] = val | HashMap::insert(key, val) | std |
| torch.tensor(data) | comment: use tch::Tensor | tch |
| time.sleep(n) | thread::sleep(Duration::from_secs(n)) | std |

Untranslatable (warn + skip): eval(), exec(), getattr(), async def, class self mutation

---

## 6. Broader Ecosystem: High-Value Python to Rust Bridges

Data Processing: pandas->polars (done); numpy loops->ndarray+rayon (our target); PIL->image crate (5-20x)
Inference: ONNX->tract; Torch->tch-rs; custom attention->candle; embeddings->usearch
Pipelines: asyncio->tokio+PyO3 (no GIL); multiprocessing->rayon (no pickle); requests->reqwest
Embedded: maturin cross for ARM; scipy.signal->Rust DSP; ROS2->rclrs+PyO3

---

## 7. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| py-spy needs elevated privileges on Windows | High | Medium | Use cProfile harness for MVP |
| rustpython-parser API changes | Medium | Medium | Pin to 0.3.*; integration test |
| Generated Rust does not compile | High | High | Default Vec<f64>+PyAny fallback; dry-run |
| maturin not installed | High | High | Pre-flight check with install instructions |
| Complex Python fails analysis | High | Low | Warn+skip; translate simple functions only |
| Windows path separators | Medium | Medium | Use Path::display() and forward slashes |
| Python version mismatch | Medium | Medium | Detect version; warn if < 3.10 |

---

## 8. Implementation Milestones

### Milestone 1: Profiling (Week 1)
- [ ] Add serde/serde_json to Cargo.toml
- [ ] Implement write_harness() in profiler.rs
- [ ] Implement run_python_harness() with process spawn
- [ ] Implement parse_profile_output() -> Vec<Hotspot>
- [ ] Wire profiler into main.rs accelerate flow
- [ ] Test with examples/slow_tokenizer.py

### Milestone 2: AST Analysis (Week 2)
- [ ] Add rustpython-parser to Cargo.toml
- [ ] Implement analyze_hotspots() with AST walk
- [ ] Add Hotspot, Translatable, TypeHint, MlHints to utils.rs
- [ ] Implement import scanning for numpy/torch detection
- [ ] Implement purity checker
- [ ] Unit test with sample Python ASTs

### Milestone 3: Code Generation (Week 3)
- [ ] Add quote, proc-macro2, syn, prettyplease to Cargo.toml
- [ ] Implement generate_stubs() with quote! templates
- [ ] Generate lib.rs, Cargo.toml, Python wrapper
- [ ] ML-smart: ndarray for numpy, allow_threads for loops
- [ ] Implement --dry-run output (print to stdout)
- [ ] Test generated code compiles with cargo check

### Milestone 4: Build Integration (Week 4)
- [ ] Implement check_maturin_installed() pre-flight
- [ ] Implement run_maturin_develop() subprocess
- [ ] Add before/after timing benchmark
- [ ] Wire full pipeline: input -> profile -> analyze -> gen -> build
- [ ] End-to-end test: slow_tokenizer.py -> installed accel_rs module

### Milestone 5: Polish and Release (Week 5-6)
- [ ] Add git2 support for --git flag
- [ ] GitHub Actions CI (Linux/macOS/Windows)
- [ ] Criterion benchmarks in benches/
- [ ] Publish to crates.io
- [ ] Demo: accelerate HuggingFace tokenizer example
- [ ] Post to r/rust, r/MachineLearning, Hacker News

---

## 9. Quick Reference: Key Commands

    cargo build
    cargo run -- accelerate --file examples/slow_tokenizer.py --output dist
    cargo run -- accelerate --snippet --output dist < snippet.py
    cargo test && cargo clippy -- -D warnings && cargo fmt
    cargo add rustpython-parser@0.3 quote proc-macro2 prettyplease ndarray serde_json
    cargo add syn --features full,parsing
    cargo add git2 --no-default-features --features https
    cargo add serde --features derive
    pip install maturin

---

## 10. Target File Structure

    rustify-ml/
    +-- Cargo.toml / plan.md / README.md / .gitignore
    +-- .github/workflows/ci.yml
    +-- src/ main.rs input.rs profiler.rs analyzer.rs generator.rs builder.rs utils.rs
    +-- tests/ test_profiler.rs test_analyzer.rs test_generator.rs test_builder.rs
    +-- examples/ slow_tokenizer.py image_preprocess.py matrix_ops.py data_pipeline.py
    +-- benches/ speedup.rs

---

*Generated by rustify-ml plan - last updated 2026-02-18*
