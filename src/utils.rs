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

pub fn extract_code(source: &InputSource) -> Result<String> {
    match source {
        InputSource::File { code, .. } => Ok(code.clone()),
        InputSource::Snippet(code) => Ok(code.clone()),
        InputSource::Git { code, .. } => Ok(code.clone()),
    }
}
