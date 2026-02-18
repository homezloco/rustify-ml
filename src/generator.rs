use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use heck::ToSnakeCase;
use tracing::info;

use crate::utils::{GenerationResult, InputSource, TargetSpec, extract_code};

pub fn generate(
    source: &InputSource,
    targets: &[TargetSpec],
    output: &Path,
    dry_run: bool,
) -> Result<GenerationResult> {
    if targets.is_empty() {
        return Err(anyhow!("no targets selected for generation"));
    }

    fs::create_dir_all(output)
        .with_context(|| format!("failed to create output dir {}", output.display()))?;
    let crate_dir = output.join("rustify_ml_ext");
    if crate_dir.exists() {
        info!(path = %crate_dir.display(), "reusing existing generated crate directory");
    } else {
        fs::create_dir_all(crate_dir.join("src")).context("failed to create crate directories")?;
    }

    let code = extract_code(source)?;
    let functions: Vec<String> = targets.iter().map(|t| render_function(t, &code)).collect();

    let lib_rs = render_lib_rs(&functions);
    let cargo_toml = render_cargo_toml();

    if dry_run {
        info!("dry-run: generation skipped writing files");
    } else {
        fs::write(crate_dir.join("src/lib.rs"), lib_rs).context("failed to write lib.rs")?;
        fs::write(crate_dir.join("Cargo.toml"), cargo_toml)
            .context("failed to write Cargo.toml")?;
    }

    info!(path = %crate_dir.display(), funcs = functions.len(), "generated Rust stubs");

    Ok(GenerationResult {
        crate_dir,
        generated_functions: functions,
    })
}

fn render_function(target: &TargetSpec, _source: &str) -> String {
    let rust_name = target.func.to_snake_case();
    format!(
        "#[pyfunction]\n\
/// Auto-generated from Python hotspot `{orig}` at line {line} ({percent:.2}%): {reason}\n\
pub fn {rust_name}(py: Python, data: Vec<f64>) -> PyResult<Vec<f64>> {{\n\
    let _ = py; // reserved for future GIL use\n\
    // TODO: translate body from Python AST; placeholder echoes input\n\
    Ok(data)\n\
}}\n",
        orig = target.func,
        line = target.line,
        percent = target.percent,
        reason = target.reason,
    )
}

fn render_lib_rs(functions: &[String]) -> String {
    let fns_joined = functions.join("\n");
    let adders = functions
        .iter()
        .map(|f| extract_fn_name(f))
        .map(|name| format!("m.add_function(wrap_pyfunction!({name}, m)?)?;"))
        .collect::<Vec<_>>()
        .join("\n    ");
    format!(
        "use pyo3::prelude::*;\n\n{fns_joined}\n\
#[pymodule]\n\
fn rustify_ml_ext(_py: Python, m: &PyModule) -> PyResult<()> {{\n\
    {adders}\n\
    Ok(())\n\
}}\n",
        fns_joined = fns_joined,
        adders = adders
    )
}

fn extract_fn_name(func_src: &str) -> String {
    func_src
        .lines()
        .find_map(|l| l.strip_prefix("pub fn "))
        .and_then(|rest| rest.split('(').next())
        .unwrap_or("generated")
        .to_string()
}

fn render_cargo_toml() -> String {
    "[package]\n\
name = \"rustify_ml_ext\"\n\
version = \"0.1.0\"\n\
edition = \"2024\"\n\
\n\
[lib]\n\
name = \"rustify_ml_ext\"\n\
crate-type = [\"cdylib\"]\n\
\n\
[dependencies]\n\
pyo3 = { version = \"0.21\", features = [\"extension-module\"] }\n"
        .to_string()
}
