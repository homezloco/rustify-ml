# Contributing

Thanks for helping improve rustify-ml!

## How to report issues
- Open a GitHub issue with a **minimal Python snippet** that fails translation (or produces incorrect output).
- Include the `rustify-ml` command you ran (flags, file path, threshold, ml-mode), plus the generated Rust snippet if possible.
- Mention your Python version and OS.

## Development quickstart
- Install: `cargo install rustify-ml` and `pip install maturin` (needed for builds)
- Run tests: `cargo test --all --all-targets -- --nocapture`
- Bench (HTML reports): `cargo bench --bench python_vs_rust`

## Style
- Use Rust 2024 edition, `cargo fmt` and `cargo clippy`.
- Prefer explicit types and prepared statements when adding database code (future-proofing).
- Keep examples realistic and performance-focused.
