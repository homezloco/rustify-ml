use std::process::Command;

use anyhow::{Context, Result, anyhow};
use tracing::{info, warn};

use crate::utils::GenerationResult;

/// Run `cargo check` on the generated crate to catch translation errors early.
/// Returns Ok(()) if the check passes, or an error with the compiler output.
/// This is a fast-fail step: it does NOT require maturin or a Python environment.
pub fn cargo_check_generated(r#gen: &GenerationResult) -> Result<()> {
    info!(path = %r#gen.crate_dir.display(), "running cargo check on generated crate");

    let output = Command::new("cargo")
        .args(["check", "--message-format=short"])
        .current_dir(&r#gen.crate_dir)
        .output()
        .context("failed to spawn cargo; ensure Rust toolchain is installed")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let combined = format!("{}\n{}", stdout.trim(), stderr.trim());
        warn!(
            path = %r#gen.crate_dir.display(),
            "cargo check failed on generated crate — review generated code:\n{}",
            combined
        );
        return Err(anyhow!(
            "generated Rust code failed cargo check. Review {} and fix translation issues.\n\nCompiler output:\n{}",
            r#gen.crate_dir.join("src/lib.rs").display(),
            combined
        ));
    }

    info!(path = %r#gen.crate_dir.display(), "cargo check passed on generated crate");
    Ok(())
}

pub fn build_extension(r#gen: &GenerationResult, dry_run: bool) -> Result<()> {
    if dry_run {
        info!(path = %r#gen.crate_dir.display(), "dry-run: skipping maturin build");
        return Ok(());
    }

    // Run cargo check first as a fast-fail before the full maturin build.
    // Warn but don't abort if cargo is not available (e.g., unusual CI setups).
    if let Err(e) = cargo_check_generated(r#gen) {
        warn!(
            path = %r#gen.crate_dir.display(),
            err = %e,
            "cargo check failed; proceeding with maturin anyway (review generated code)"
        );
    }

    let status = Command::new("maturin")
        .args(["develop", "--release"])
        .current_dir(&r#gen.crate_dir)
        .status()
        .context("failed to spawn maturin; install via `pip install maturin` and ensure on PATH")?;

    if !status.success() {
        return Err(anyhow!("maturin build failed with status {status}"));
    }

    info!(
        path = %r#gen.crate_dir.display(),
        fallback_functions = r#gen.fallback_functions,
        "maturin build completed"
    );
    Ok(())
}
