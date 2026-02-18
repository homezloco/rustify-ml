# rustify-ml

CLI to profile Python ML hotspots and generate Rust/PyO3 stubs for acceleration.

## Status
- MVP scaffold: CLI args, module layout, logging/tracing configured.
- Next phases: profiling via py-spy, AST analysis (rustpython-parser), codegen (quote/syn), maturin build path.

Flow (current placeholders): input -> profile -> analyze -> generate -> build. Profiling/analysis/gen/build are wired with placeholder structs and logging; implementation to come.

## Usage (planned)
```bash
rustify-ml accelerate --file path/to/script.py --threshold 15 --output dist
# or
cat script.py | rustify-ml accelerate --snippet --output dist
```

### Example: Euclidean distance
```bash
# generate and build an extension from the example
cargo run -- accelerate --file examples/euclidean.py --output dist

# if built (non-dry-run), the crate lives at dist/rustify_ml_ext
cd dist/rustify_ml_ext
maturin develop --release

# then in Python
python - <<'PY'
from rustify_ml_ext import euclidean
print(euclidean([0.0, 0.0], [3.0, 4.0]))  # expect 5.0
PY
```

## Development
- Rust 1.75+ recommended
- Install maturin and Python 3.10+
- Format & check:
```bash
cd /mnt/d/WindsurfProjects/rustify/rustify-ml && cargo fmt && cargo check
# or keep cwd elsewhere and pass manifest explicitly:
cargo fmt --manifest-path /mnt/d/WindsurfProjects/rustify/rustify-ml/Cargo.toml
cargo check --manifest-path /mnt/d/WindsurfProjects/rustify/rustify-ml/Cargo.toml
```

### Install maturin (if not already)
```bash
pip install maturin
```

### Notes
- CLI warns when any target falls back to echo translation; review generated lib.rs when warned.
- Length mismatches on Vec-like params produce PyValueError guards in generated stubs.

## Roadmap
1) Profiling integration (py-spy) with harness generation
2) AST analysis for translatable hotspots (rustpython-parser)
3) Rust stub generation with PyO3 + ndarray helpers
4) Maturin build/install automation
5) Examples/benchmarks against common ML preprocessing loops
