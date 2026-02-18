use std::process::Command;

use anyhow::{Context, Result, anyhow};
use tracing::info;

use crate::utils::GenerationResult;

pub fn build_extension(r#gen: &GenerationResult, dry_run: bool) -> Result<()> {
    if dry_run {
        info!(path = %r#gen.crate_dir.display(), "dry-run: skipping maturin build");
        return Ok(());
    }

    let status = Command::new("maturin")
        .args(["develop", "--release"])
        .current_dir(&r#gen.crate_dir)
        .status()
        .context("failed to spawn maturin; install via `pip install maturin` and ensure on PATH")?;

    if !status.success() {
        return Err(anyhow!("maturin build failed with status {status}"));
    }

    info!(path = %r#gen.crate_dir.display(), "maturin build completed");
    Ok(())
}
