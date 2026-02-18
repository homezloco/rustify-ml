use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use tempfile::TempDir;

#[derive(Debug, Clone)]
pub enum InputSource {
    File {
        path: PathBuf,
        code: String,
    },
    Snippet(String),
    Git {
        repo: String,
        path: PathBuf,
        code: String,
    },
}

#[derive(Debug, Clone)]
pub struct Hotspot {
    pub func: String,
    pub line: u32,
    pub percent: f32,
}

#[derive(Debug, Clone, Default)]
pub struct ProfileSummary {
    pub hotspots: Vec<Hotspot>,
}

#[derive(Debug, Clone)]
pub struct TargetSpec {
    pub func: String,
    pub line: u32,
    pub percent: f32,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub crate_dir: PathBuf,
    pub generated_functions: Vec<String>,
    pub fallback_functions: usize,
}

/// Materialize the input source into a concrete file path for profiling/build steps.
/// Returns the path and a TempDir to keep the file alive for the caller's scope.
pub fn materialize_input(source: &InputSource) -> Result<(PathBuf, TempDir)> {
    let tmpdir = tempfile::tempdir().context("failed to create temp dir for input")?;
    let filename = match source {
        InputSource::File { path, .. } => path
            .file_name()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("input.py")),
        InputSource::Snippet(_) => PathBuf::from("input.py"),
        InputSource::Git { path, .. } => path
            .file_name()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("input.py")),
    };
    let path = tmpdir.path().join(filename);

    match source {
        InputSource::File { path: src, .. } => {
            fs::copy(src, &path)
                .with_context(|| format!("failed to copy input from {}", src.display()))?;
        }
        InputSource::Snippet(code) => {
            fs::write(&path, code).context("failed to write snippet to temp file")?;
        }
        InputSource::Git { code, .. } => {
            fs::write(&path, code).context("failed to write git file to temp file")?;
        }
    }

    Ok((path, tmpdir))
}

/// One row in the post-generation summary table printed to stdout.
#[derive(Debug, Clone)]
pub struct AccelerateRow {
    pub func: String,
    pub line: u32,
    pub pct_time: f32,
    pub translation: &'static str, // "Full" | "Partial"
    pub status: String,            // "Success" | "Fallback: <reason>"
}

/// Print a simple ASCII summary table to stdout.
pub fn print_summary(rows: &[AccelerateRow], crate_dir: &std::path::Path) {
    let total = rows.len();
    let fallbacks = rows.iter().filter(|r| r.translation == "Partial").count();
    println!();
    println!(
        "Accelerated {}/{} targets ({} fallback{})",
        total - fallbacks,
        total,
        fallbacks,
        if fallbacks == 1 { "" } else { "s" }
    );
    println!();
    println!(
        "{:<22} | {:>4} | {:>6} | {:<11} | Status",
        "Func", "Line", "% Time", "Translation"
    );
    println!("{}", "-".repeat(22 + 3 + 4 + 3 + 6 + 3 + 11 + 3 + 20));
    for row in rows {
        println!(
            "{:<22} | {:>4} | {:>5.1}% | {:<11} | {}",
            row.func, row.line, row.pct_time, row.translation, row.status
        );
    }
    println!();
    println!("Generated: {}", crate_dir.display());
    println!(
        "Install:   cd {} && maturin develop --release",
        crate_dir.display()
    );
    println!();
}

pub fn extract_code(source: &InputSource) -> Result<String> {
    match source {
        InputSource::File { code, .. } => Ok(code.clone()),
        InputSource::Snippet(code) => Ok(code.clone()),
        InputSource::Git { code, .. } => Ok(code.clone()),
    }
}
