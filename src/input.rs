use std::io::{self, Read};
use std::path::Path;

use anyhow::{Context, Result};
use tracing::info;

use crate::utils::InputSource;

pub fn load_input(file: Option<&Path>, snippet: bool, git: Option<&str>) -> Result<InputSource> {
    if snippet {
        let mut buffer = String::new();
        io::stdin()
            .read_to_string(&mut buffer)
            .context("failed to read Python snippet from stdin")?;
        info!(chars = buffer.len(), "loaded snippet from stdin");
        return Ok(InputSource::Snippet(buffer));
    }

    if let Some(path) = file {
        let code = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read Python file at {}", path.display()))?;
        info!(path = %path.display(), bytes = code.len(), "loaded file input");
        return Ok(InputSource::File {
            path: path.to_path_buf(),
            code,
        });
    }

    if let Some(repo) = git {
        info!(repo, "git input requested (not yet implemented)");
        // TODO: implement git clone + path selection
        return Ok(InputSource::GitPlaceholder(repo.to_string()));
    }

    Err(anyhow::anyhow!(
        "no input provided; pass --file, --snippet, or --git"
    ))
}
