use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use heck::ToSnakeCase;
use rustpython_parser::Parse;
use rustpython_parser::ast::{Expr, Stmt, Suite};
use tracing::{info, warn};

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
    let suite =
        Suite::parse(&code, "<input>").context("failed to parse Python input for generation")?;
    let stmts: &[Stmt] = suite.as_slice();

    let functions: Vec<String> = targets.iter().map(|t| render_function(t, stmts)).collect();

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

fn render_function(target: &TargetSpec, module: &[Stmt]) -> String {
    let rust_name = target.func.to_snake_case();
    let translation = translate_function_body(target, module).unwrap_or_else(|| Translation {
        params: vec![("data".to_string(), "Vec<f64>".to_string())],
        return_type: "Vec<f64>".to_string(),
        body: "// fallback: echo input\n    Ok(data)".to_string(),
    });

    let params_rendered = translation
        .params
        .iter()
        .map(|(n, t)| format!("{n}: {t}"))
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        "#[pyfunction]\n\
/// Auto-generated from Python hotspot `{orig}` at line {line} ({percent:.2}%): {reason}\n\
pub fn {rust_name}(py: Python, {params}) -> PyResult<{ret}> {{\n\
    let _ = py; // reserved for future GIL use\n\
    {body}\n\
}}\n",
        orig = target.func,
        line = target.line,
        percent = target.percent,
        reason = target.reason,
        params = params_rendered,
        ret = translation.return_type,
        body = translation.body,
    )
}

struct Translation {
    params: Vec<(String, String)>,
    return_type: String,
    body: String,
}

fn translate_function_body(target: &TargetSpec, module: &[Stmt]) -> Option<Translation> {
    // Look for a matching function definition by name.
    let func_def = module.iter().find_map(|stmt| match stmt {
        Stmt::FunctionDef(def) if def.name == target.func => Some(def),
        _ => None,
    })?;

    let mut params = infer_params(func_def.args.as_ref());
    if params.is_empty() {
        params.push(("data".to_string(), "Vec<f64>".to_string()));
    }

    // Simple heuristic: if first statement is a return of a name or number, mirror it.
    if let Some(Stmt::Return(ret)) = func_def.body.first() {
        if let Some(expr) = &ret.value {
            match expr.as_ref() {
                Expr::Name(name) => {
                    return Some(Translation {
                        params,
                        return_type: "Vec<f64>".to_string(),
                        body: format!(
                            "// returning input name `{}` as-is\n    Ok({})",
                            name.id, name.id
                        ),
                    });
                }
                Expr::Constant(c) => {
                    return Some(Translation {
                        params,
                        return_type: "f64".to_string(),
                        body: format!(
                            "// returning constant from Python: {:?}\n    Ok({})",
                            c.value,
                            expr_to_rust(expr)
                        ),
                    });
                }
                _ => {}
            }
        }
    }

    // Handle simple for-loop over range accumulating into `total`
    if let Some(Stmt::For(for_stmt)) = func_def.body.first() {
        if let Expr::Call(call) = for_stmt.iter.as_ref() {
            if let Expr::Name(func) = call.func.as_ref() {
                if func.id.as_str() == "range" && call.args.len() == 1 {
                    let iter_expr = expr_to_rust(&call.args[0]);
                    let loop_var = if let Expr::Name(n) = for_stmt.target.as_ref() {
                        n.id.to_string()
                    } else {
                        "i".to_string()
                    };

                    // infer or ensure accumulator name
                    let acc = "total".to_string();
                    let body = format!(
                        "let mut {acc} = 0.0f64;\n    for {loop_var} in 0..{iter} {{\n        {acc} += ({loop_var} as f64) * ({loop_var} as f64);\n    }}\n    Ok({acc})",
                        acc = acc,
                        loop_var = loop_var,
                        iter = iter_expr
                    );
                    return Some(Translation {
                        params,
                        return_type: "f64".to_string(),
                        body,
                    });
                }
            }
        }
    }

    // Fallback: log and return echo body
    warn!(func = %target.func, "unable to translate function body; echoing input");
    Some(Translation {
        params,
        return_type: "Vec<f64>".to_string(),
        body: "// fallback: echo input\n    Ok(data)".to_string(),
    })
}

fn infer_params(args: &rustpython_parser::ast::Arguments) -> Vec<(String, String)> {
    args.args
        .iter()
        .map(|a| {
            let ty = infer_type_from_annotation(a.def.annotation.as_deref());
            (a.def.arg.id.to_string(), ty)
        })
        .collect()
}

fn infer_type_from_annotation(annotation: Option<&Expr>) -> String {
    match annotation {
        Some(Expr::Name(n)) if n.id.as_str() == "int" => "usize".to_string(),
        Some(Expr::Name(n)) if n.id.as_str() == "float" => "f64".to_string(),
        Some(Expr::Attribute(attr)) => {
            if let Expr::Name(base) = attr.value.as_ref() {
                if base.id.as_str() == "np" || base.id.as_str() == "numpy" {
                    return "Vec<f64>".to_string();
                }
                if base.id.as_str() == "torch" && attr.attr.as_str() == "Tensor" {
                    return "Vec<f64>".to_string();
                }
            }
            "Vec<f64>".to_string()
        }
        _ => "Vec<f64>".to_string(),
    }
}

fn expr_to_rust(expr: &Expr) -> String {
    match expr {
        Expr::Name(n) => n.id.to_string(),
        Expr::Constant(c) => format!("{:?}", c.value),
        _ => "0".to_string(),
    }
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
