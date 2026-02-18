use std::io::{self, Read};
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use tracing::{info, warn};

use crate::utils::InputSource;

pub fn load_input(
    file: Option<&Path>,
    snippet: bool,
    git: Option<&str>,
    git_path: Option<&Path>,
) -> Result<InputSource> {
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
        let git_path =
            git_path.ok_or_else(|| anyhow!("--git-path is required when using --git"))?;

        let tmpdir = tempfile::tempdir().context("failed to create temp dir for git clone")?;
        let repo_dir = tmpdir.path().join("repo");
        info!(repo, path = %git_path.display(), "cloning git repo (shallow if supported)");
        let mut fo = git2::FetchOptions::new();
        fo.download_tags(git2::AutotagOption::None);
        fo.update_fetchhead(true);
        let mut co = git2::build::RepoBuilder::new();
        co.fetch_options(fo);
        co.clone(repo, &repo_dir)
            .with_context(|| format!("failed to clone repo {repo}"))?;

        let target_path = repo_dir.join(git_path);
        if !target_path.exists() {
            warn!(path = %target_path.display(), "git path not found in repo");
            return Err(anyhow!("git path not found: {}", target_path.display()));
        }
        let code = std::fs::read_to_string(&target_path).with_context(|| {
            format!(
                "failed to read file {} from git repo",
                target_path.display()
            )
        })?;
        info!(path = %target_path.display(), bytes = code.len(), "loaded git input");
        return Ok(InputSource::Git {
            repo: repo.to_string(),
            path: target_path,
            code,
        });
    }

    Err(anyhow::anyhow!(
        "no input provided; pass --file, --snippet, or --git"
    ))
}
