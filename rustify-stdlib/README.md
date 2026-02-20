# rustify-stdlib

Prebuilt Rust routines (PyO3) for Python acceleration. Build with maturin.

## Install (dev)
```bash
# from rustify-ml/rustify-stdlib
maturin develop --release
python - <<'PY'
import rustify_stdlib as rs
print(rs.euclidean([0.0,3.0,4.0],[0.0,0.0,0.0]))
PY
```

## Install (crates.io + pip)

```bash
pip install maturin
pip install rustify-stdlib  # once published
python - <<'PY'
import rustify_stdlib as rs
print(rs.dot_product([1.0,2.0],[3.0,4.0]))
PY
```

## Functions
- euclidean(p1, p2) -> float
- dot_product(a, b) -> float
- moving_average(signal, window) -> list[float]
- convolve1d(signal, kernel) -> list[float]
- bpe_encode(text, merges=[]) -> list[int]

## Testing
```bash
cargo test
cargo bench --bench speed -- --quick
```

## PyO3 linking (Linux)

If you hit missing Python symbols at link time, ensure the build knows where `libpython` lives. The crate ships a `build.rs` that reads the following environment variables and sets `-L`, `-l`, and an rpath:

```bash
# common defaults used during debugging
export PYO3_PYTHON=/usr/bin/python3.12
export PYO3_CONFIG_FILE=/mnt/d/WindsurfProjects/rustify/rustify-ml/pyo3.cfg
export PYO3_NO_PKG_CONFIG=1
export PYO3_LIB_DIR=/usr/lib/x86_64-linux-gnu
export PYO3_LIB_NAME=python3.12
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu

cargo test -vv
```

Customize `PYO3_LIB_DIR` and `PYO3_LIB_NAME` if your Python install differs (e.g., `/opt/python/3.10/lib` and `python3.10`). The rpath is emitted automatically so the test binary can locate `libpython` at runtime.
