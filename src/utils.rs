use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use tempfile::TempDir;

#[derive(Debug, Clone)]
pub enum InputSource {
    File { path: PathBuf, code: String },
    Snippet(String),
    GitPlaceholder(String),
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
}

/// Materialize the input source into a concrete file path for profiling/build steps.
/// Returns the path and a TempDir to keep the file alive for the caller's scope.
pub fn materialize_input(source: &InputSource) -> Result<(PathBuf, TempDir)> {
    let tmpdir = tempfile::tempdir().context("failed to create temp dir for input")?;
    let path = tmpdir.path().join("input.py");

    match source {
        InputSource::File { path: src, .. } => {
            fs::copy(src, &path)
                .with_context(|| format!("failed to copy input from {}", src.display()))?;
        }
        InputSource::Snippet(code) => {
            fs::write(&path, code).context("failed to write snippet to temp file")?;
        }
        InputSource::GitPlaceholder(repo) => {
            return Err(anyhow!("git input not yet implemented: {}", repo));
        }
    }

    Ok((path, tmpdir))
}

pub fn extract_code(source: &InputSource) -> Result<String> {
    match source {
        InputSource::File { code, .. } => Ok(code.clone()),
        InputSource::Snippet(code) => Ok(code.clone()),
        InputSource::GitPlaceholder(repo) => {
            Err(anyhow!("git input not yet implemented: {}", repo))
        }
    }
}
