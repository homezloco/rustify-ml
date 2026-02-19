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
