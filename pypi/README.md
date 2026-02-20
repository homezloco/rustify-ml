# rustify-ml

> **Profile Python ML hotspots and auto-generate Rust PyO3 bindings. 20x speedup with one command.**

This PyPI package is a companion stub. The CLI is written in Rust and distributed via **crates.io**.

## Install the CLI

```bash
cargo install rustify-ml
pip install maturin
```

## Usage

```bash
# Profile a Python file and generate a Rust extension
rustify-ml accelerate --file your_script.py --output dist --threshold 10

# Install and use the generated extension
cd dist/rustify_ml_ext && maturin develop --release
python -c "import rustify_ml_ext; print(rustify_ml_ext.euclidean([0,3,4],[0,0,0]))"
```

## Benchmark Results (CPython 3.12)

| Function | Python µs | Rust µs | Speedup |
|---|---|---|---|
| running_mean | 376.9 | 18.7 | **20.2x** |
| convolve1d | 329.8 | 21.0 | **15.7x** |
| moving_average | 471.5 | 30.8 | **15.3x** |
| bpe_encode | 11.1 | 0.9 | **12.3x** |
| euclidean | 55.8 | 20.8 | 2.7x |

## Links

- **GitHub**: https://github.com/homezloco/rustify-ml
- **crates.io**: https://crates.io/crates/rustify-ml
- **Docs**: https://github.com/homezloco/rustify-ml#readme
