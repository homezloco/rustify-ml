use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use heck::ToSnakeCase;
use rustpython_parser::Parse;
use rustpython_parser::ast::{Expr, Operator, Stmt, Suite};
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

    let mut fallback_functions = 0usize;
    let functions: Vec<String> = targets
        .iter()
        .map(|t| {
            let (code, fallback) = render_function(t, stmts);
            if fallback {
                fallback_functions += 1;
            }
            code
        })
        .collect();

    let lib_rs = render_lib_rs(&functions);
    let cargo_toml = render_cargo_toml();

    if dry_run {
        info!("dry-run: generation skipped writing files");
    } else {
        fs::write(crate_dir.join("src/lib.rs"), lib_rs).context("failed to write lib.rs")?;
        fs::write(crate_dir.join("Cargo.toml"), cargo_toml)
            .context("failed to write Cargo.toml")?;
    }

    if fallback_functions > 0 {
        warn!(path = %crate_dir.display(), fallback_functions, "some functions fell back to echo translation");
    }
    info!(path = %crate_dir.display(), funcs = functions.len(), "generated Rust stubs");

    Ok(GenerationResult {
        crate_dir,
        generated_functions: functions,
        fallback_functions,
    })
}

fn render_function(target: &TargetSpec, module: &[Stmt]) -> (String, bool) {
    let rust_name = target.func.to_snake_case();
    let translation = translate_function_body(target, module).unwrap_or_else(|| Translation {
        params: vec![("data".to_string(), "Vec<f64>".to_string())],
        return_type: "Vec<f64>".to_string(),
        body: "// fallback: echo input\n    Ok(data)".to_string(),
        fallback: true,
    });

    let len_check = render_len_checks(&translation.params).unwrap_or_default();

    let params_rendered = translation
        .params
        .iter()
        .map(|(n, t)| format!("{n}: {t}"))
        .collect::<Vec<_>>()
        .join(", ");

    let rendered = format!(
        "#[pyfunction]\n\
    /// Auto-generated from Python hotspot `{orig}` at line {line} ({percent:.2}%): {reason}\n\
pub fn {rust_name}(py: Python, {params}) -> PyResult<{ret}> {{\n    let _ = py; // reserved for future GIL use\n    {len_check}\n    {body}\n}}\n",
        orig = target.func,
        line = target.line,
        percent = target.percent,
        reason = target.reason,
        params = params_rendered,
        ret = translation.return_type,
        body = translation.body,
        len_check = len_check,
    );

    (rendered, translation.fallback)
}

fn render_len_checks(params: &[(String, String)]) -> Option<String> {
    // Collect Vec-like params (Vec<...>) to compare lengths. Only emit if at least 2.
    let vec_params: Vec<&String> = params
        .iter()
        .filter(|(_, ty)| ty.contains("Vec<") || ty.contains("[f64]"))
        .map(|(n, _)| n)
        .collect();

    if vec_params.len() < 2 {
        return None;
    }

    let first = vec_params[0];
    let mut checks = String::new();
    for other in vec_params.iter().skip(1) {
        checks.push_str(&format!(
            "    if {first}.len() != {other}.len() {{\n        return Err(pyo3::exceptions::PyValueError::new_err(\"length mismatch\"));\n    }}\n",
            first = first,
            other = other
        ));
    }

    Some(checks)
}

struct Translation {
    params: Vec<(String, String)>,
    return_type: String,
    body: String,
    fallback: bool,
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
                        fallback: false,
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
                        fallback: false,
                    });
                }
                _ => {}
            }
        }
    }

    // Translate the body generically; fallback handled below
    if let Some(translated_body) = translate_body(&func_def.body) {
        return Some(Translation {
            params,
            return_type: translated_body.return_type,
            body: translated_body.body,
            fallback: translated_body.fallback,
        });
    }

    // Fallback: log and return echo body
    warn!(func = %target.func, "unable to translate function body; echoing input");
    warn!(func = %target.func, "translation fallback used");
    Some(Translation {
        params,
        return_type: "Vec<f64>".to_string(),
        body: "// fallback: echo input\n    Ok(data)".to_string(),
        fallback: true,
    })
}

struct BodyTranslation {
    return_type: String,
    body: String,
    fallback: bool,
}

fn translate_body(body: &[Stmt]) -> Option<BodyTranslation> {
    if body.is_empty() {
        return None;
    }

    // Handle simple single-statement for loop with accumulation
    if let Stmt::For(for_stmt) = &body[0] {
        if let Expr::Call(call) = for_stmt.iter.as_ref() {
            if let Expr::Name(func) = call.func.as_ref() {
                if func.id.as_str() == "range" && call.args.len() == 1 {
                    let iter_expr = expr_to_rust(&call.args[0]);
                    let loop_var = if let Expr::Name(n) = for_stmt.target.as_ref() {
                        n.id.to_string()
                    } else {
                        "i".to_string()
                    };

                    let translated_loop_body = translate_body(for_stmt.body.as_slice())
                        .map(|b| b.body)
                        .unwrap_or_else(|| {
                            format!(
                                "// TODO: translate loop body\n        total += ({loop_var} as f64) * ({loop_var} as f64);"
                            )
                        });

                    let body = format!(
                        "let mut total = 0.0f64;\n    for {loop_var} in 0..{iter} {{\n        {translated_loop_body}\n    }}\n    Ok(total)",
                        loop_var = loop_var,
                        iter = iter_expr,
                        translated_loop_body = translated_loop_body
                    );

                    return Some(BodyTranslation {
                        return_type: "f64".to_string(),
                        body,
                        fallback: false,
                    });
                }
            }
        }
    }

    // Generic translation of sequential statements
    let mut out = String::new();
    for stmt in body {
        match translate_stmt(stmt) {
            Some(line) => {
                out.push_str("    ");
                out.push_str(&line);
                if !line.ends_with('\n') {
                    out.push('\n');
                }
            }
            None => {
                out.push_str("    // Unhandled stmt: ");
                out.push_str(&format!("{:?}\n", stmt));
            }
        }
    }

    out.push_str("    Ok(total)");

    Some(BodyTranslation {
        return_type: "f64".to_string(),
        body: out,
        fallback: true,
    })
}

fn translate_stmt(stmt: &Stmt) -> Option<String> {
    match stmt {
        Stmt::Assign(assign) => {
            if let (Some(target), value) = (assign.targets.first(), &assign.value) {
                let lhs = match target {
                    Expr::Name(n) => format!("let mut {}", n.id),
                    Expr::Attribute(_) => format!("// attribute assign {}", expr_to_rust(target)),
                    _ => format!("// complex assign {}", expr_to_rust(target)),
                };
                let rhs = expr_to_rust(value);
                return Some(format!("{} = {};", lhs, rhs));
            }
            None
        }
        Stmt::For(for_stmt) => {
            let iter_expr = expr_to_rust(&for_stmt.iter);
            let loop_var = expr_to_rust(&for_stmt.target);
            let loop_body = translate_body(for_stmt.body.as_slice())
                .map(|b| b.body)
                .unwrap_or_else(|| "// unhandled loop body".to_string());
            Some(format!(
                "for {loop_var} in {iter_expr} {{\n{loop_body}\n    }}",
                loop_var = loop_var,
                iter_expr = iter_expr,
                loop_body = indent_block(&loop_body, 4)
            ))
        }
        Stmt::If(if_stmt) => {
            let test = expr_to_rust(&if_stmt.test);
            let body = translate_body(if_stmt.body.as_slice())
                .map(|b| b.body)
                .unwrap_or_else(|| "// unhandled if body".to_string());
            let orelse = if !if_stmt.orelse.is_empty() {
                translate_body(if_stmt.orelse.as_slice())
                    .map(|b| b.body)
                    .unwrap_or_else(|| "// unhandled else body".to_string())
            } else {
                String::new()
            };
            let else_block = if orelse.is_empty() {
                String::new()
            } else {
                format!("else {{\n{}\n    }}", indent_block(&orelse, 4))
            };
            Some(format!(
                "if {test} {{\n{body_block}\n    }} {else_block}",
                test = test,
                body_block = indent_block(&body, 4),
                else_block = else_block
            ))
        }
        Stmt::AugAssign(aug) => {
            let lhs = expr_to_rust(&aug.target);
            let rhs = expr_to_rust(&aug.value);
            let op = match aug.op {
                Operator::Add => "+=",
                Operator::Sub => "-=",
                Operator::Mult => "*=",
                Operator::Div => "/=",
                _ => "+=",
            };
            Some(format!("{} {} {};", lhs, op, rhs))
        }
        Stmt::Return(ret) => {
            if let Some(v) = &ret.value {
                Some(format!("return {};", expr_to_rust(v)))
            } else {
                Some("return ();".to_string())
            }
        }
        _ => None,
    }
}

fn infer_params(args: &rustpython_parser::ast::Arguments) -> Vec<(String, String)> {
    args.args
        .iter()
        .map(|a| {
            let ty = infer_type_from_annotation(a.def.annotation.as_deref());
            (a.def.arg.to_string(), ty)
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
        Expr::Call(call) => {
            if let Expr::Name(func) = call.func.as_ref() {
                if func.id.as_str() == "range" && call.args.len() == 1 {
                    return format!("0..{}", expr_to_rust(&call.args[0]));
                }
                if func.id.as_str() == "len" && call.args.len() == 1 {
                    return format!("{}.len()", expr_to_rust(&call.args[0]));
                }
            }
            "/* call fallback */".to_string()
        }
        Expr::BinOp(binop) => {
            let left = expr_to_rust(&binop.left);
            let right = expr_to_rust(&binop.right);
            let op = match binop.op {
                Operator::Add => "+",
                Operator::Sub => "-",
                Operator::Mult => "*",
                Operator::Div => "/",
                Operator::Pow => {
                    return format!("({}).powf({} as f64)", left, right);
                }
                _ => "+",
            };
            format!("({} {} {})", left, op, right)
        }
        Expr::Subscript(sub) => {
            let value = expr_to_rust(&sub.value);
            let index = expr_to_rust(&sub.slice);
            format!("{}[{}]", value, index)
        }
        Expr::Attribute(attr) => {
            format!("{}.{}", expr_to_rust(&attr.value), attr.attr)
        }
        _ => "0".to_string(),
    }
}

fn indent_block(body: &str, spaces: usize) -> String {
    let pad = " ".repeat(spaces);
    body.lines()
        .map(|l| format!("{}{}", pad, l))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::TargetSpec;
    use std::path::PathBuf;
    use tempfile::tempdir;

    #[test]
    fn test_expr_to_rust_range_and_len() {
        let range_expr = Expr::Call(Box::new(rustpython_parser::ast::ExprCall {
            func: Box::new(Expr::Name(Box::new(rustpython_parser::ast::ExprName {
                id: "range".into(),
                ctx: rustpython_parser::ast::ExprContext::Load,
            }))),
            args: vec![Expr::Constant(Box::new(
                rustpython_parser::ast::ExprConstant {
                    value: rustpython_parser::ast::Constant::Int(10.into()),
                    kind: None,
                },
            ))],
            keywords: vec![],
        }));

        let len_expr = Expr::Call(Box::new(rustpython_parser::ast::ExprCall {
            func: Box::new(Expr::Name(Box::new(rustpython_parser::ast::ExprName {
                id: "len".into(),
                ctx: rustpython_parser::ast::ExprContext::Load,
            }))),
            args: vec![Expr::Name(Box::new(rustpython_parser::ast::ExprName {
                id: "a".into(),
                ctx: rustpython_parser::ast::ExprContext::Load,
            }))],
            keywords: vec![],
        }));

        assert_eq!(expr_to_rust(&range_expr), "0..10");
        assert_eq!(expr_to_rust(&len_expr), "a.len()");
    }

    #[test]
    fn test_expr_to_rust_binop_pow() {
        let bin = Expr::BinOp(Box::new(rustpython_parser::ast::ExprBinOp {
            left: Box::new(Expr::Name(Box::new(rustpython_parser::ast::ExprName {
                id: "x".into(),
                ctx: rustpython_parser::ast::ExprContext::Load,
            }))),
            op: Operator::Pow,
            right: Box::new(Expr::Constant(Box::new(
                rustpython_parser::ast::ExprConstant {
                    value: rustpython_parser::ast::Constant::Int(2.into()),
                    kind: None,
                },
            ))),
        }));
        assert_eq!(expr_to_rust(&bin), "(x).powf(2 as f64)");
    }

    #[test]
    fn test_render_len_checks_multiple_vecs() {
        let params = vec![
            ("a".to_string(), "Vec<f64>".to_string()),
            ("b".to_string(), "Vec<f64>".to_string()),
        ];
        let rendered = render_len_checks(&params).unwrap();
        assert!(rendered.contains("a.len() != b.len()"));
        assert!(rendered.contains("PyValueError"));
    }

    #[test]
    fn test_translate_euclidean_body() {
        let code = r#"
def euclidean(p1, p2):
    total = 0.0
    for i in range(len(p1)):
        diff = p1[i] - p2[i]
        total += diff * diff
    return total ** 0.5
"#;
        let suite = Suite::parse(code, "<test>").expect("parse failed");
        let stmts: &[Stmt] = suite.as_slice();
        let target = TargetSpec {
            func: "euclidean".to_string(),
            line: 1,
            percent: 100.0,
            reason: "test".to_string(),
        };

        let translation = translate_function_body(&target, stmts).expect("no translation");
        assert_eq!(translation.return_type, "f64");
        assert!(!translation.fallback, "euclidean should not fallback");
        assert!(translation.body.contains("for i in 0.."));
        assert!(translation.body.contains("total"));
    }

    #[test]
    fn test_generate_integration_euclidean() {
        let example_path = PathBuf::from("examples/euclidean.py");
        let code = std::fs::read_to_string(&example_path).expect("read example");
        let source = InputSource::File {
            path: example_path.clone(),
            code,
        };
        let targets = vec![TargetSpec {
            func: "euclidean".to_string(),
            line: 1,
            percent: 100.0,
            reason: "test".to_string(),
        }];

        let tmp = tempdir().expect("tempdir");
        let gen = generate(&source, &targets, tmp.path(), false).expect("generate");
        assert_eq!(gen.generated_functions.len(), 1);
        assert_eq!(gen.fallback_functions, 0);
        assert!(tmp.path().join("rustify_ml_ext/src/lib.rs").exists());
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
