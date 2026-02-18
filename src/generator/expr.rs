//! Expression-to-Rust translation helpers.
//!
//! Converts Python AST expression nodes into Rust source strings.
//! All functions are pure (no I/O, no state).

use rustpython_parser::ast::{CmpOp, Expr, Operator};

/// Translate a Python expression to a Rust expression string.
pub fn expr_to_rust(expr: &Expr) -> String {
    match expr {
        Expr::Name(n) => n.id.to_string(),
        Expr::Constant(c) => constant_to_rust(&c.value),
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
                    return format!("({}).powf({})", left, right);
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

/// Convert a Python constant value to its Rust literal equivalent.
pub fn constant_to_rust(value: &rustpython_parser::ast::Constant) -> String {
    use rustpython_parser::ast::Constant;
    match value {
        Constant::Int(i) => i.to_string(),
        Constant::Float(f) => {
            let s = format!("{}", f);
            if s.contains('.') || s.contains('e') {
                s
            } else {
                format!("{}.0", s)
            }
        }
        Constant::Bool(b) => b.to_string(),
        Constant::Str(s) => format!("\"{}\"", s.escape_default()),
        Constant::None => "()".to_string(),
        _ => "0".to_string(),
    }
}

/// Translate a Python for-loop iterator expression to a Rust range string.
/// Handles: `range(n)` → `0..n`, `range(a, b)` → `a..b`, fallback to expr_to_rust.
pub fn translate_for_iter(iter: &Expr) -> String {
    if let Expr::Call(call) = iter
        && let Expr::Name(func) = call.func.as_ref()
        && func.id.as_str() == "range"
    {
        match call.args.len() {
            1 => return format!("0..{}", expr_to_rust(&call.args[0])),
            2 => {
                return format!(
                    "{}..{}",
                    expr_to_rust(&call.args[0]),
                    expr_to_rust(&call.args[1])
                );
            }
            _ => {}
        }
    }
    expr_to_rust(iter)
}

/// Translate a Python while-loop test expression to Rust.
///
/// Handles:
/// - `while changed:` → `while changed`
/// - `while not changed:` → `while !changed`
/// - `while i < len(x):` → `while i < x.len()`
pub fn translate_while_test(test: &Expr) -> String {
    match test {
        Expr::Name(n) => n.id.to_string(),
        Expr::UnaryOp(unary) => {
            use rustpython_parser::ast::UnaryOp;
            if matches!(unary.op, UnaryOp::Not) {
                format!("!{}", translate_while_test(&unary.operand))
            } else {
                expr_to_rust(test)
            }
        }
        Expr::Compare(comp) if comp.ops.len() == 1 && comp.comparators.len() == 1 => {
            let left = expr_to_rust(&comp.left);
            let right = expr_to_rust(&comp.comparators[0]);
            let op = match comp.ops[0] {
                CmpOp::Lt => "<",
                CmpOp::LtE => "<=",
                CmpOp::Gt => ">",
                CmpOp::GtE => ">=",
                CmpOp::Eq => "==",
                CmpOp::NotEq => "!=",
                _ => "<",
            };
            format!("{left} {op} {right}")
        }
        _ => expr_to_rust(test),
    }
}

/// Translate a Python if-test that guards a length check into a Rust guard string.
/// Returns `None` if the test is not a simple equality/inequality comparison.
pub fn translate_len_guard(test: &Expr) -> Option<String> {
    if let Expr::Compare(comp) = test
        && comp.ops.len() == 1
        && comp.comparators.len() == 1
    {
        let op = &comp.ops[0];
        let left = expr_to_rust(&comp.left);
        let right = expr_to_rust(&comp.comparators[0]);
        let cond = match op {
            CmpOp::Eq => format!("{left} == {right}"),
            CmpOp::NotEq => format!("{left} != {right}"),
            _ => return None,
        };
        return Some(format!(
            "if {cond} {{\n        return Err(pyo3::exceptions::PyValueError::new_err(\"Vectors must be same length\"));\n    }}",
            cond = cond
        ));
    }
    None
}
