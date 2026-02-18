//! Type inference helpers for Python → Rust parameter and assignment types.

use rustpython_parser::ast::Expr;

use super::expr::expr_to_rust;

/// Infer Rust parameter types from a Python function's argument list.
pub fn infer_params(args: &rustpython_parser::ast::Arguments) -> Vec<(String, String)> {
    args.args
        .iter()
        .map(|a| {
            let ty = infer_type_from_annotation(a.def.annotation.as_deref());
            (a.def.arg.to_string(), ty)
        })
        .collect()
}

/// Infer a Rust type string from a Python type annotation expression.
///
/// Supported annotations:
/// - `int` → `usize`
/// - `float` → `f64`
/// - `np.ndarray`, `numpy.ndarray` → `Vec<f64>`
/// - `torch.Tensor` → `Vec<f64>`
/// - anything else → `Vec<f64>` (safe default)
pub fn infer_type_from_annotation(annotation: Option<&Expr>) -> String {
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

/// Infer a Rust type annotation suffix for a simple assignment RHS.
///
/// Returns `": f64"` for float literals, `": i64"` for int literals, `""` otherwise.
pub fn infer_assign_type(value: &Expr) -> &'static str {
    match value {
        Expr::Constant(c) => match &c.value {
            rustpython_parser::ast::Constant::Float(_) => ": f64",
            rustpython_parser::ast::Constant::Int(_) => ": i64",
            _ => "",
        },
        _ => "",
    }
}

/// Emit Rust length-check guards for Vec parameters.
///
/// If two or more `Vec<...>` params are present, emits:
/// ```rust
/// if a.len() != b.len() {
///     return Err(PyValueError::new_err("length mismatch"));
/// }
/// ```
pub fn render_len_checks(params: &[(String, String)]) -> Option<String> {
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

// Suppress unused import warning — expr_to_rust is used transitively via infer_type_from_annotation
#[allow(unused_imports)]
use super::expr::expr_to_rust as _expr_to_rust_unused;
