"""
rustify-ml: Profile Python ML hotspots and auto-generate Rust PyO3 bindings.

The CLI is written in Rust and distributed via crates.io.

Install the CLI:
    cargo install rustify-ml

Usage:
    rustify-ml accelerate --file your_script.py --output dist --threshold 10

GitHub:   https://github.com/homezloco/rustify-ml
crates.io: https://crates.io/crates/rustify-ml
"""

__version__ = "0.1.2"


def _cli_hint() -> None:
    print(
        "rustify-ml is a Rust CLI tool. Install it with:\n"
        "\n"
        "    cargo install rustify-ml\n"
        "\n"
        "Then run:\n"
        "\n"
        "    rustify-ml accelerate --file your_script.py --output dist --threshold 10\n"
        "\n"
        "Docs: https://github.com/homezloco/rustify-ml"
    )
