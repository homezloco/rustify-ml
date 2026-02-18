//! Rust source code rendering for generated PyO3 extensions.
//!
//! Produces `lib.rs` and `Cargo.toml` content for the generated crate.

use heck::ToSnakeCase;

use crate::utils::TargetSpec;
use rustpython_parser::ast::Stmt;

use super::infer::render_len_checks;
use super::translate::translate_function_body;

/// Render a single PyO3 `#[pyfunction]` for the given target.
///
/// Returns `(rendered_source, had_fallback)`.
pub fn render_function_with_options(
    target: &TargetSpec,
    module: &[Stmt],
    use_ndarray: bool,
) -> (String, bool) {
    let rust_name = target.func.to_snake_case();
    let mut translation = translate_function_body(target, module).unwrap_or_else(|| {
        super::translate::Translation {
            params: vec![("data".to_string(), "Vec<f64>".to_string())],
            return_type: "Vec<f64>".to_string(),
            body: "// fallback: echo input\n    Ok(data)".to_string(),
            fallback: true,
        }
    });

    // ndarray mode: replace Vec<f64> params with PyReadonlyArray1<f64>
    if use_ndarray {
        for (_, ty) in &mut translation.params {
            if ty == "Vec<f64>" {
                *ty = "numpy::PyReadonlyArray1<f64>".to_string();
            }
        }
    }

    let len_check = if use_ndarray {
        String::new()
    } else {
        render_len_checks(&translation.params).unwrap_or_default()
    };

    let params_rendered = translation
        .params
        .iter()
        .map(|(n, t)| format!("{n}: {t}"))
        .collect::<Vec<_>>()
        .join(", ");

    let ndarray_note = if use_ndarray {
        "\n    // ndarray: use p1.as_slice()? to get &[f64] for indexing"
    } else {
        ""
    };

    let rendered = format!(
        "#[pyfunction]\n\
    /// Auto-generated from Python hotspot `{orig}` at line {line} ({percent:.2}%): {reason}\n\
pub fn {rust_name}(py: Python, {params}) -> PyResult<{ret}> {{{ndarray_note}\n    let _ = py; // reserved for future GIL use\n    {len_check}\n    {body}\n}}\n",
        orig = target.func,
        line = target.line,
        percent = target.percent,
        reason = target.reason,
        params = params_rendered,
        ret = translation.return_type,
        body = translation.body,
        len_check = len_check,
        ndarray_note = ndarray_note,
    );

    (rendered, translation.fallback)
}

/// Render the full `lib.rs` content for the generated crate.
pub fn render_lib_rs_with_options(functions: &[String], use_ndarray: bool) -> String {
    let fns_joined = functions.join("\n");
    let adders = functions
        .iter()
        .map(|f| extract_fn_name(f))
        .map(|name| format!("m.add_function(wrap_pyfunction!({name}, m)?)?;"))
        .collect::<Vec<_>>()
        .join("\n    ");
    let ndarray_import = if use_ndarray { "use numpy;\n" } else { "" };
    format!(
        "use pyo3::prelude::*;\n{ndarray_import}\n{fns_joined}\n\
#[pymodule]\n\
fn rustify_ml_ext(_py: Python, m: &PyModule) -> PyResult<()> {{\n\
    {adders}\n\
    Ok(())\n\
}}\n",
        ndarray_import = ndarray_import,
        fns_joined = fns_joined,
        adders = adders
    )
}

/// Render the `Cargo.toml` content for the generated crate.
pub fn render_cargo_toml_with_options(use_ndarray: bool) -> String {
    let numpy_dep = if use_ndarray { "numpy = \"0.21\"\n" } else { "" };
    format!(
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
pyo3 = {{ version = \"0.21\", features = [\"extension-module\"] }}\n\
{numpy_dep}",
        numpy_dep = numpy_dep
    )
}

/// Extract the function name from a rendered `pub fn <name>(` line.
pub fn extract_fn_name(func_src: &str) -> String {
    func_src
        .lines()
        .find_map(|l| l.strip_prefix("pub fn "))
        .and_then(|rest| rest.split('(').next())
        .unwrap_or("generated")
        .to_string()
}
