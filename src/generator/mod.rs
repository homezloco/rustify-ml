//! Generator module — public API for Python → Rust PyO3 stub generation.
//!
//! # Submodules
//! - [`expr`]      — expression-to-Rust translation (pure, no I/O)
//! - [`infer`]     — type inference for params and assignments
//! - [`translate`] — statement/body translation (AST walk)
//! - [`render`]    — PyO3 function + lib.rs + Cargo.toml rendering
//!
//! # Entry points
//! - [`generate`]    — standard generation
//! - [`generate_ml`] — ML mode (numpy → PyReadonlyArray1)

pub mod expr;
pub mod infer;
pub mod render;
pub mod translate;

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use rustpython_parser::Parse;
use rustpython_parser::ast::{Stmt, Suite};
use tracing::{info, warn};

use crate::utils::{GenerationResult, InputSource, TargetSpec, extract_code};
use render::{
    render_cargo_toml_with_options, render_function_with_options, render_lib_rs_with_options,
};

/// Detect numpy usage in Python source (triggers ndarray mode).
fn detects_numpy(code: &str) -> bool {
    code.contains("import numpy") || code.contains("from numpy") || code.contains("import np")
}

/// Parse existing lib.rs content and return complete #[pyfunction] blocks using brace balance.
fn parse_existing_functions(lib_rs: &str) -> Vec<String> {
    let mut funcs = Vec::new();
    let mut current = Vec::new();
    let mut in_fn = false;
    let mut brace_balance: i32 = 0;
    let mut seen_open = false;
    for line in lib_rs.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("#[pyfunction]") {
            current.clear();
            in_fn = true;
            brace_balance = 0;
            seen_open = false;
            current.push(line.to_string());
            continue;
        }

        if in_fn {
            current.push(line.to_string());
            let opens = line.matches('{').count() as i32;
            let closes = line.matches('}').count() as i32;
            if opens > 0 {
                seen_open = true;
            }
            brace_balance += opens;
            brace_balance -= closes;

            if seen_open && brace_balance <= 0 {
                funcs.push(current.join("\n"));
                current.clear();
                in_fn = false;
            }
        }
    }
    funcs
}

/// Generate Rust + PyO3 stubs for the given targets.
pub fn generate(
    source: &InputSource,
    targets: &[TargetSpec],
    output: &Path,
    dry_run: bool,
) -> Result<GenerationResult> {
    generate_with_options(source, targets, output, dry_run, false)
}

/// Generate with ML mode: detects numpy imports → uses `PyReadonlyArray1<f64>` params.
pub fn generate_ml(
    source: &InputSource,
    targets: &[TargetSpec],
    output: &Path,
    dry_run: bool,
) -> Result<GenerationResult> {
    generate_with_options(source, targets, output, dry_run, true)
}

fn generate_with_options(
    source: &InputSource,
    targets: &[TargetSpec],
    output: &Path,
    dry_run: bool,
    ml_mode: bool,
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
    let use_ndarray = ml_mode && detects_numpy(&code);
    if use_ndarray {
        info!("numpy detected + ml_mode: using PyReadonlyArray1<f64> params");
    }

    let suite =
        Suite::parse(&code, "<input>").context("failed to parse Python input for generation")?;
    let stmts: &[Stmt] = suite.as_slice();

    // Merge previously generated functions (if crate already exists) with newly generated ones.
    let mut functions_by_name: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    let existing_lib = crate_dir.join("src/lib.rs");
    if existing_lib.exists()
        && let Ok(existing_src) = std::fs::read_to_string(&existing_lib)
    {
        for func_src in parse_existing_functions(&existing_src) {
            let name = render::extract_fn_name(&func_src);
            functions_by_name.insert(name, func_src);
        }
    }

    let mut fallback_functions = 0usize;
    for t in targets.iter() {
        let (code, fallback) = render_function_with_options(t, stmts, use_ndarray);
        let name = render::extract_fn_name(&code);
        if fallback {
            fallback_functions += 1;
        }
        functions_by_name.insert(name, code);
    }

    let functions: Vec<String> = functions_by_name.into_values().collect();

    let lib_rs = render_lib_rs_with_options(&functions, use_ndarray);
    let cargo_toml = render_cargo_toml_with_options(use_ndarray);

    fs::write(crate_dir.join("src/lib.rs"), lib_rs).context("failed to write lib.rs")?;
    fs::write(crate_dir.join("Cargo.toml"), cargo_toml).context("failed to write Cargo.toml")?;

    if dry_run {
        info!(path = %crate_dir.display(), "dry-run: wrote generated files (no build)");
    }
    if fallback_functions > 0 {
        warn!(
            fallback_functions,
            "some functions fell back to echo translation"
        );
    }
    info!(path = %crate_dir.display(), funcs = functions.len(), "generated Rust stubs");

    Ok(GenerationResult {
        crate_dir,
        generated_functions: functions,
        fallback_functions,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use rustpython_parser::Parse;
    use rustpython_parser::ast::{Expr, Operator, Suite};
    use rustpython_parser::text_size::TextRange;
    use tempfile::tempdir;

    use crate::utils::{InputSource, TargetSpec};

    use super::expr::expr_to_rust;
    use super::infer::render_len_checks;
    use super::translate::translate_function_body;
    use super::*;

    // ── expr tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_expr_to_rust_range_and_len() {
        let range_expr = Expr::Call(rustpython_parser::ast::ExprCall {
            func: Box::new(Expr::Name(rustpython_parser::ast::ExprName {
                range: TextRange::default(),
                id: "range".into(),
                ctx: rustpython_parser::ast::ExprContext::Load,
            })),
            args: vec![Expr::Constant(rustpython_parser::ast::ExprConstant {
                range: TextRange::default(),
                value: rustpython_parser::ast::Constant::Int(10.into()),
                kind: None,
            })],
            keywords: vec![],
            range: TextRange::default(),
        });
        let len_expr = Expr::Call(rustpython_parser::ast::ExprCall {
            func: Box::new(Expr::Name(rustpython_parser::ast::ExprName {
                range: TextRange::default(),
                id: "len".into(),
                ctx: rustpython_parser::ast::ExprContext::Load,
            })),
            args: vec![Expr::Name(rustpython_parser::ast::ExprName {
                range: TextRange::default(),
                id: "a".into(),
                ctx: rustpython_parser::ast::ExprContext::Load,
            })],
            keywords: vec![],
            range: TextRange::default(),
        });
        assert_eq!(expr_to_rust(&range_expr), "0..10");
        assert_eq!(expr_to_rust(&len_expr), "a.len()");
    }

    #[test]
    fn test_expr_to_rust_binop_pow() {
        let bin = Expr::BinOp(rustpython_parser::ast::ExprBinOp {
            range: TextRange::default(),
            left: Box::new(Expr::Name(rustpython_parser::ast::ExprName {
                range: TextRange::default(),
                id: "x".into(),
                ctx: rustpython_parser::ast::ExprContext::Load,
            })),
            op: Operator::Pow,
            right: Box::new(Expr::Constant(rustpython_parser::ast::ExprConstant {
                range: TextRange::default(),
                value: rustpython_parser::ast::Constant::Int(2.into()),
                kind: None,
            })),
        });
        assert_eq!(expr_to_rust(&bin), "(x).powf(2)");
    }

    // ── infer tests ───────────────────────────────────────────────────────────

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

    // ── translate tests ───────────────────────────────────────────────────────

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
        let target = TargetSpec {
            func: "euclidean".to_string(),
            line: 1,
            percent: 100.0,
            reason: "test".to_string(),
        };
        let t = translate_function_body(&target, suite.as_slice()).expect("no translation");
        assert_eq!(t.return_type, "f64");
        assert!(!t.fallback);
        assert!(t.body.contains("for i in 0.."));
    }

    #[test]
    fn test_translate_stmt_float_assign_init() {
        let code = "def f(x):\n    total = 0.0\n    return total\n";
        let suite = Suite::parse(code, "<test>").expect("parse");
        let target = TargetSpec {
            func: "f".to_string(),
            line: 1,
            percent: 100.0,
            reason: "test".to_string(),
        };
        let t = translate_function_body(&target, suite.as_slice()).expect("no translation");
        assert!(t.body.contains("let mut total: f64"), "got: {}", t.body);
    }

    #[test]
    fn test_translate_stmt_subscript_assign() {
        let code = "def f(result, i, val):\n    result[i] = val\n    return result\n";
        let suite = Suite::parse(code, "<test>").expect("parse");
        let target = TargetSpec {
            func: "f".to_string(),
            line: 1,
            percent: 100.0,
            reason: "test".to_string(),
        };
        let t = translate_function_body(&target, suite.as_slice()).expect("no translation");
        assert!(t.body.contains("result[i] = val"), "got: {}", t.body);
    }

    #[test]
    fn test_translate_stmt_list_init() {
        let code = "def f(n):\n    result = [0.0] * n\n    return result\n";
        let suite = Suite::parse(code, "<test>").expect("parse");
        let target = TargetSpec {
            func: "f".to_string(),
            line: 1,
            percent: 100.0,
            reason: "test".to_string(),
        };
        let t = translate_function_body(&target, suite.as_slice()).expect("no translation");
        assert!(
            t.body.contains("vec![") && t.body.contains("result"),
            "got: {}",
            t.body
        );
    }

    #[test]
    fn test_translate_list_comprehension() {
        // result = [x * 2.0 for x in data] → let result: Vec<f64> = data.iter().map(|x| ...).collect();
        let code = "def f(data):\n    result = [x * 2.0 for x in data]\n    return result\n";
        let suite = Suite::parse(code, "<test>").expect("parse");
        let target = TargetSpec {
            func: "f".to_string(),
            line: 1,
            percent: 100.0,
            reason: "test".to_string(),
        };
        let t = translate_function_body(&target, suite.as_slice()).expect("no translation");
        assert!(
            t.body.contains(".iter().map(") && t.body.contains(".collect()"),
            "expected iter().map().collect() for list comp, got: {}",
            t.body
        );
        assert_eq!(t.return_type, "Vec<f64>");
        assert!(!t.fallback);
    }

    #[test]
    fn test_translate_argmax_tuple_return() {
        // returns (index, value) tuple
        let code = "def argmax(xs):\n    best_idx = 0\n    best_val = xs[0]\n    for i in range(len(xs)):\n        if xs[i] > best_val:\n            best_val = xs[i]\n            best_idx = i\n    return (best_idx, best_val)\n";
        let suite = Suite::parse(code, "<test>").expect("parse");
        let target = TargetSpec {
            func: "argmax".to_string(),
            line: 1,
            percent: 100.0,
            reason: "argmax tuple".to_string(),
        };
        let t = translate_function_body(&target, suite.as_slice()).expect("no translation");
        assert_eq!(t.return_type, "(usize, f64)");
        assert!(
            t.body.contains("return Ok((best_idx, best_val));"),
            "body: {}",
            t.body
        );
        assert!(!t.fallback);
    }

    #[test]
    fn test_translate_nested_for_else_triggers_fallback() {
        // for..else is unsupported; expect fallback
        let code = "def f(n):\n    total = 0\n    for i in range(n):\n        for j in range(n):\n            total += i + j\n    else:\n        total += 1\n    return total\n";
        let suite = Suite::parse(code, "<test>").expect("parse");
        let target = TargetSpec {
            func: "f".to_string(),
            line: 1,
            percent: 100.0,
            reason: "test nested for".to_string(),
        };
        let t = translate_function_body(&target, suite.as_slice()).expect("no translation");
        assert!(
            t.fallback,
            "expected fallback for for..else, got body:\n{}",
            t.body
        );
    }

    #[test]
    fn test_translate_dot_product_zero_fallback() {
        let code = "def dot_product(a, b):\n    total = 0.0\n    for i in range(len(a)):\n        total += a[i] * b[i]\n    return total\n";
        let suite = Suite::parse(code, "<test>").expect("parse");
        let target = TargetSpec {
            func: "dot_product".to_string(),
            line: 1,
            percent: 80.0,
            reason: "test".to_string(),
        };
        let t = translate_function_body(&target, suite.as_slice()).expect("no translation");
        assert!(
            !t.fallback,
            "dot_product should not fallback; body:\n{}",
            t.body
        );
        assert!(t.body.contains("total +="), "got: {}", t.body);
    }

    #[test]
    fn test_translate_matmul_nested_loops() {
        let code = r#"
def matmul(a, b, n):
    result = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            total = 0.0
            for k in range(n):
                total += a[i * n + k] * b[k * n + j]
            result[i * n + j] = total
    return result
"#;
        let suite = Suite::parse(code, "<test>").expect("parse");
        let target = TargetSpec {
            func: "matmul".to_string(),
            line: 1,
            percent: 100.0,
            reason: "test".to_string(),
        };
        let t = translate_function_body(&target, suite.as_slice()).expect("no translation");
        assert!(t.body.contains("for i in 0..n"), "got: {}", t.body);
        assert!(t.body.contains("for j in 0..n"), "got: {}", t.body);
        assert!(t.body.contains("for k in 0..n"), "got: {}", t.body);
        assert!(t.body.contains("vec!["), "got: {}", t.body);
    }

    #[test]
    fn test_translate_while_loop_bool_flag() {
        let code = "def count_pairs(tokens):\n    counts = 0\n    changed = True\n    while changed:\n        changed = False\n        counts += 1\n    return counts\n";
        let suite = Suite::parse(code, "<test>").expect("parse");
        let target = TargetSpec {
            func: "count_pairs".to_string(),
            line: 1,
            percent: 100.0,
            reason: "test".to_string(),
        };
        let t = translate_function_body(&target, suite.as_slice()).expect("no translation");
        assert!(t.body.contains("while changed"), "got:\n{}", t.body);
    }

    #[test]
    fn test_translate_while_comparison() {
        let code = "def scan(tokens):\n    i = 0\n    while i < len(tokens):\n        i += 1\n    return i\n";
        let suite = Suite::parse(code, "<test>").expect("parse");
        let target = TargetSpec {
            func: "scan".to_string(),
            line: 1,
            percent: 100.0,
            reason: "test".to_string(),
        };
        let t = translate_function_body(&target, suite.as_slice()).expect("no translation");
        assert!(t.body.contains("while i <"), "got:\n{}", t.body);
        assert!(t.body.contains("tokens.len()"), "got:\n{}", t.body);
    }

    // ── ndarray / ml_mode tests ───────────────────────────────────────────────

    #[test]
    fn test_ndarray_mode_replaces_vec_params() {
        let code = "import numpy as np\ndef dot_product(a, b):\n    total = 0.0\n    for i in range(len(a)):\n        total += a[i] * b[i]\n    return total\n";
        let source = InputSource::Snippet(code.to_string());
        let targets = vec![TargetSpec {
            func: "dot_product".to_string(),
            line: 1,
            percent: 100.0,
            reason: "test".to_string(),
        }];
        let tmp = tempdir().expect("tempdir");
        let result = generate_ml(&source, &targets, tmp.path(), false).expect("generate_ml");
        let lib_rs = std::fs::read_to_string(tmp.path().join("rustify_ml_ext/src/lib.rs")).unwrap();
        assert_eq!(result.fallback_functions, 0);
        assert!(lib_rs.contains("PyReadonlyArray1<f64>"), "got:\n{}", lib_rs);
        assert!(lib_rs.contains("use numpy;"), "got:\n{}", lib_rs);
        let cargo = std::fs::read_to_string(tmp.path().join("rustify_ml_ext/Cargo.toml")).unwrap();
        assert!(cargo.contains("numpy"), "got:\n{}", cargo);
    }

    #[test]
    fn test_ndarray_mode_no_numpy_import_stays_vec() {
        let code = "def dot_product(a, b):\n    total = 0.0\n    for i in range(len(a)):\n        total += a[i] * b[i]\n    return total\n";
        let source = InputSource::Snippet(code.to_string());
        let targets = vec![TargetSpec {
            func: "dot_product".to_string(),
            line: 1,
            percent: 100.0,
            reason: "test".to_string(),
        }];
        let tmp = tempdir().expect("tempdir");
        let result = generate_ml(&source, &targets, tmp.path(), false).expect("generate_ml");
        let lib_rs = std::fs::read_to_string(tmp.path().join("rustify_ml_ext/src/lib.rs")).unwrap();
        assert_eq!(result.fallback_functions, 0);
        assert!(!lib_rs.contains("PyReadonlyArray1"), "got:\n{}", lib_rs);
        assert!(lib_rs.contains("Vec<f64>"), "got:\n{}", lib_rs);
    }

    // ── integration tests ─────────────────────────────────────────────────────

    #[test]
    fn test_generate_integration_euclidean() {
        let path = PathBuf::from("examples/euclidean.py");
        let code = std::fs::read_to_string(&path).expect("read example");
        let source = InputSource::File {
            path: path.clone(),
            code,
        };
        let targets = vec![TargetSpec {
            func: "euclidean".to_string(),
            line: 1,
            percent: 100.0,
            reason: "test".to_string(),
        }];
        let tmp = tempdir().expect("tempdir");
        let result = generate(&source, &targets, tmp.path(), false).expect("generate");
        assert_eq!(result.generated_functions.len(), 1);
        assert_eq!(result.fallback_functions, 0);
        assert!(tmp.path().join("rustify_ml_ext/src/lib.rs").exists());
    }
}
