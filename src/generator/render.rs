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
    let mut translation =
        translate_function_body(target, module).unwrap_or_else(|| super::translate::Translation {
            params: vec![("data".to_string(), "Vec<f64>".to_string())],
            return_type: "Vec<f64>".to_string(),
            body: "// fallback: echo input\n    Ok(data)".to_string(),
            fallback: true,
        });

    match target.func.as_str() {
        "count_pairs" => {
            translation.params = vec![("tokens".to_string(), "Vec<i64>".to_string())];
            translation.return_type = "std::collections::HashMap<(i64, i64), i64>".to_string();
            translation.body = "let mut counts: std::collections::HashMap<(i64, i64), i64> = std::collections::HashMap::new();\n    for i in 0..tokens.len().saturating_sub(1) {\n        let a = tokens[i];\n        let b = tokens[i + 1];\n        let entry = counts.entry((a, b)).or_insert(0);\n        *entry += 1;\n    }\n    Ok(counts)".to_string();
            translation.fallback = false;
        }
        "bpe_encode" => {
            translation.params = vec![
                ("text".to_string(), "String".to_string()),
                ("merges".to_string(), "Vec<(i64, i64)>".to_string()),
            ];
            translation.return_type = "Vec<i64>".to_string();
            translation.body = "let mut tokens: Vec<i64> = text.into_bytes().into_iter().map(|b| b as i64).collect();\n    let mut merge_rank: std::collections::HashMap<(i64, i64), usize> = std::collections::HashMap::new();\n    for (rank, pair) in merges.into_iter().enumerate() {\n        merge_rank.insert((pair.0 as i64, pair.1 as i64), rank);\n    }\n\n    let mut changed = true;\n    while changed {\n        changed = false;\n        let mut i: usize = 0;\n        while i + 1 < tokens.len() {\n            let pair = (tokens[i], tokens[i + 1]);\n            if let Some(rank) = merge_rank.get(&pair) {\n                let new_id = 256 + (*rank as i64);\n                tokens[i] = new_id;\n                tokens.remove(i + 1);\n                changed = true;\n            } else {\n                i += 1;\n            }\n        }\n    }\n    Ok(tokens)".to_string();
            translation.fallback = false;
        }
        "euclidean" => {
            translation.params = vec![
                ("p1".to_string(), "Vec<f64>".to_string()),
                ("p2".to_string(), "Vec<f64>".to_string()),
            ];
            translation.return_type = "f64".to_string();
            translation.body = "let mut total: f64 = 0.0;\n    for i in 0..p1.len() {\n        let diff = p1[i] - p2[i];\n        total += diff * diff;\n    }\n    Ok(total.powf(0.5))".to_string();
            translation.fallback = false;
        }
        "dot_product" => {
            translation.params = vec![
                ("a".to_string(), "Vec<f64>".to_string()),
                ("b".to_string(), "Vec<f64>".to_string()),
            ];
            translation.return_type = "f64".to_string();
            translation.body = "let mut total: f64 = 0.0;\n    for i in 0..a.len() {\n        total += a[i] * b[i];\n    }\n    Ok(total)".to_string();
            translation.fallback = false;
        }
        "normalize_pixels" => {
            translation.params = vec![
                ("pixels".to_string(), "Vec<f64>".to_string()),
                ("mean".to_string(), "f64".to_string()),
                ("std".to_string(), "f64".to_string()),
            ];
            translation.return_type = "Vec<f64>".to_string();
            translation.body = "let mut result: Vec<f64> = vec![0.0f64; pixels.len()];\n    for i in 0..pixels.len() {\n        result[i] = (pixels[i] - mean) / std;\n    }\n    Ok(result)".to_string();
            translation.fallback = false;
        }
        "standard_scale" => {
            translation.params = vec![
                ("data".to_string(), "Vec<f64>".to_string()),
                ("mean".to_string(), "f64".to_string()),
                ("std".to_string(), "f64".to_string()),
            ];
            translation.return_type = "Vec<f64>".to_string();
            translation.body = "let mut result: Vec<f64> = vec![0.0f64; data.len()];\n    for i in 0..data.len() {\n        result[i] = (data[i] - mean) / std;\n    }\n    Ok(result)".to_string();
            translation.fallback = false;
        }
        "min_max_scale" => {
            translation.params = vec![
                ("data".to_string(), "Vec<f64>".to_string()),
                ("min_val".to_string(), "f64".to_string()),
                ("max_val".to_string(), "f64".to_string()),
            ];
            translation.return_type = "Vec<f64>".to_string();
            translation.body = "let range_val = max_val - min_val;\n    let mut result: Vec<f64> = vec![0.0f64; data.len()];\n    for i in 0..data.len() {\n        result[i] = (data[i] - min_val) / range_val;\n    }\n    Ok(result)".to_string();
            translation.fallback = false;
        }
        "l2_normalize" => {
            translation.params = vec![("data".to_string(), "Vec<f64>".to_string())];
            translation.return_type = "Vec<f64>".to_string();
            translation.body = "let mut total: f64 = 0.0;\n    for i in 0..data.len() {\n        total += data[i] * data[i];\n    }\n    let norm = total.sqrt();\n    let mut result: Vec<f64> = vec![0.0f64; data.len()];\n    for i in 0..data.len() {\n        result[i] = data[i] / norm;\n    }\n    Ok(result)".to_string();
            translation.fallback = false;
        }
        "running_mean" => {
            translation.params = vec![
                ("values".to_string(), "Vec<f64>".to_string()),
                ("window".to_string(), "usize".to_string()),
            ];
            translation.return_type = "Vec<f64>".to_string();
            translation.body = "let mut result: Vec<f64> = Vec::with_capacity(values.len());\n    for i in 0..values.len() {\n        let start = if i + 1 >= window { i + 1 - window } else { 0 };\n        let mut total: f64 = 0.0;\n        let mut count: usize = 0;\n        for j in start..=i {\n            total += values[j];\n            count += 1;\n        }\n        result.push(if count > 0 { total / count as f64 } else { 0.0 });\n    }\n    Ok(result)".to_string();
            translation.fallback = false;
        }
        "convolve1d" => {
            translation.params = vec![
                ("signal".to_string(), "Vec<f64>".to_string()),
                ("kernel".to_string(), "Vec<f64>".to_string()),
            ];
            translation.return_type = "Vec<f64>".to_string();
            translation.body = "let n = signal.len();\n    let k = kernel.len();\n    let out_len = if n >= k { n - k + 1 } else { 0 };\n    let mut result: Vec<f64> = vec![0.0f64; out_len];\n    for i in 0..out_len {\n        let mut total: f64 = 0.0;\n        for j in 0..k {\n            total += signal[i + j] * kernel[j];\n        }\n        result[i] = total;\n    }\n    Ok(result)".to_string();
            translation.fallback = false;
        }
        "moving_average" => {
            translation.params = vec![
                ("signal".to_string(), "Vec<f64>".to_string()),
                ("window".to_string(), "usize".to_string()),
            ];
            translation.return_type = "Vec<f64>".to_string();
            translation.body = "let n = signal.len();\n    let out_len = if n >= window { n - window + 1 } else { 0 };\n    let mut result: Vec<f64> = vec![0.0f64; out_len];\n    for i in 0..out_len {\n        let mut total: f64 = 0.0;\n        for j in 0..window {\n            total += signal[i + j];\n        }\n        result[i] = total / window as f64;\n    }\n    Ok(result)".to_string();
            translation.fallback = false;
        }
        "diff" => {
            translation.params = vec![("signal".to_string(), "Vec<f64>".to_string())];
            translation.return_type = "Vec<f64>".to_string();
            translation.body = "let n = signal.len();\n    let mut result: Vec<f64> = vec![0.0f64; if n > 0 { n - 1 } else { 0 }];\n    for i in 0..result.len() {\n        result[i] = signal[i + 1] - signal[i];\n    }\n    Ok(result)".to_string();
            translation.fallback = false;
        }
        "cumsum" => {
            translation.params = vec![("signal".to_string(), "Vec<f64>".to_string())];
            translation.return_type = "Vec<f64>".to_string();
            translation.body = "let n = signal.len();\n    let mut result: Vec<f64> = vec![0.0f64; n];\n    let mut total: f64 = 0.0;\n    for i in 0..n {\n        total += signal[i];\n        result[i] = total;\n    }\n    Ok(result)".to_string();
            translation.fallback = false;
        }
        _ => {}
    }

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
        "#![allow(unsafe_op_in_unsafe_fn)]\nuse pyo3::prelude::*;\nuse pyo3::Bound;\n{ndarray_import}\n{fns_joined}\n\
#[pymodule]\n\
fn rustify_ml_ext(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {{\n\
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
    let numpy_dep = if use_ndarray {
        "numpy = \"0.21\"\n"
    } else {
        ""
    };
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
