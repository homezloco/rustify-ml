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
use render::{render_cargo_toml_with_options, render_function_with_options, render_lib_rs_with_options};

/// Detect numpy usage in Python source (triggers ndarray mode).
fn detects_numpy(code: &str) -> bool {
    code.contains("import numpy") || code.contains("from numpy") || code.contains("import np")
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

    let mut fallback_functions = 0usize;
    let functions: Vec<String> = targets
        .iter()
        .map(|t| {
            let (code, fallback) = render_function_with_options(t, stmts, use_ndarray);
            if fallback {
                fallback_functions += 1;
            }
            code
        })
        .collect();

    let lib_rs = render_lib_rs_with_options(&functions, use_ndarray);
    let cargo_toml = render_cargo_toml_with_options(use_ndarray);

    fs::write(crate_dir.join("src/lib.rs"), lib_rs).context("failed to write lib.rs")?;
    fs::write(crate_dir.join("Cargo.toml"), cargo_toml).context("failed to write Cargo.toml")?;

    if dry_run {
        info!(path = %crate_dir.display(), "dry-run: wrote generated files (no build)");
    }
    if fallback_functions > 0 {
        warn!(fallback_functions, "some functions fell back to echo translation");
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
    use rustpython_parser::ast::{Expr, Operator, Stmt, Suite};
    use rustpython_parser::text_size::TextRange;
    use tempfile::tempdir;

    use crate::utils::{InputSource, TargetSpec};

    use super::*;
    use super::expr::expr_to_rust;
    use super::infer::render_len_checks;
    use super::translate::translate_function_body;

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
        let target = TargetSpec { func: "euclidean".to_string(), line: 1, percent: 100.0, reason: "test".to_string() };
        let t = translate_function_body(&target, suite.as_slice()).expect("no translation");
        assert_eq!(t.return_type, "f64");
        assert!(!t.fallback);
        assert!(t.body.contains("for i in 0.."));
    }

    #[test]
    fn test_translate_stmt_float_assign_init() {
        let code = "def f(x):\n    total = 0.0\n    return total\n";
        let suite = Suite::parse(code, "<test>").expect("parse");
        let target = TargetSpec { func: "f".to_string(), line: 1, percent: 100.0, reason: "test".to_string() };
        let t = translate_function_body(&target, suite.as_slice()).expect("no translation");
        assert!(t.body.contains("let mut total: f64"), "got: {}", t.body);
    }

    #[test]
    fn test_translate_stmt_subscript_assign() {
        let code = "def f(result, i, val):\n    result[i] = val\n    return result\n";
        let suite = Suite::parse(code, "<test>").expect("parse");
        let target = TargetSpec { func: "f".to_string(), line: 1, percent: 100.0, reason: "test".to_string() };
        let t = translate_function_body(&target, suite.as_slice()).expect("no translation");
        assert!(t.body.contains("result[i] = val"), "got: {}", t.body);
    }

    #[test]
    fn test_translate_stmt_list_init() {
        let code = "def f(n):\n    result = [0.0] * n\n    return result\n";
        let suite = Suite::parse(code, "<test>").expect("parse");
        let target = TargetSpec { func: "f".to_string(), line: 1, percent: 100.0, reason: "test".to_string() };
        let t = translate_function_body(&target, suite.as_slice()).expect("no translation");
        assert!(t.body.contains("vec![") && t.body.contains("result"), "got: {}", t.body);
    }

    #[test]
    fn test_translate_list_comprehension() {
        // result = [x * 2.0 for x in data] → let result: Vec<f64> = data.iter().map(|x| ...).collect();
        let code = "def f(data):\n    result = [x * 2.0 for x in data]\n    return result\n";
        let suite = Suite::parse(code, "<test>").expect("parse");
        let target = TargetSpec { func: "f".to_string(), line: 1, percent: 100.0, reason: "test".to_string() };
        let t = translate_function_body(&target, suite.as_slice()).expect("no translation");
        assert!(
            t.body.contains(".iter().map(") && t.body.contains(".collect()"),
            "expected iter().map().collect() for list comp, got: {}",
            t.body
        );
    }

    #[test]
    fn test_translate_dot_product_zero_fallback() {
        let code = "def dot_product(a, b):\n    total = 0.0\n    for i in range(len(a)):\n        total += a[i] * b[i]\n    return total\n";
        let suite = Suite::parse(code, "<test>").expect("parse");
        let target = TargetSpec { func: "dot_product".to_string(), line: 1, percent: 80.0, reason: "test".to_string() };
        let t = translate_function_body(&target, suite.as_slice()).expect("no translation");
        assert!(!t.fallback, "dot_product should not fallback; body:\n{}", t.body);
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
        let target = TargetSpec { func: "matmul".to_string(), line: 1, percent: 100.0, reason: "test".to_string() };
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
        let target = TargetSpec { func: "count_pairs".to_string(), line: 1, percent: 100.0, reason: "test".to_string() };
        let t = translate_function_body(&target, suite.as_slice()).expect("no translation");
        assert!(t.body.contains("while changed"), "got:\n{}", t.body);
    }

    #[test]
    fn test_translate_while_comparison() {
        let code = "def scan(tokens):\n    i = 0\n    while i < len(tokens):\n        i += 1\n    return i\n";
        let suite = Suite::parse(code, "<test>").expect("parse");
        let target = TargetSpec { func: "scan".to_string(), line: 1, percent: 100.0, reason: "test".to_string() };
        let t = translate_function_body(&target, suite.as_slice()).expect("no translation");
        assert!(t.body.contains("while i <"), "got:\n{}", t.body);
        assert!(t.body.contains("tokens.len()"), "got:\n{}", t.body);
    }

    // ── ndarray / ml_mode tests ───────────────────────────────────────────────

    #[test]
    fn test_ndarray_mode_replaces_vec_params() {
        let code = "import numpy as np\ndef dot_product(a, b):\n    total = 0.0\n    for i in range(len(a)):\n        total += a[i] * b[i]\n    return total\n";
        let source = InputSource::Snippet(code.to_string());
        let targets = vec![TargetSpec { func: "dot_product".to_string(), line: 1, percent: 100.0, reason: "test".to_string() }];
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
        let targets = vec![TargetSpec { func: "dot_product".to_string(), line: 1, percent: 100.0, reason: "test".to_string() }];
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
        let source = InputSource::File { path: path.clone(), code };
        let targets = vec![TargetSpec { func: "euclidean".to_string(), line: 1, percent: 100.0, reason: "test".to_string() }];
        let tmp = tempdir().expect("tempdir");
        let result = generate(&source, &targets, tmp.path(), false).expect("generate");
        assert_eq!(result.generated_functions.len(), 1);
        assert_eq!(result.fallback_functions, 0);
        assert!(tmp.path().join("rustify_ml_ext/src/lib.rs").exists());
    }
}
