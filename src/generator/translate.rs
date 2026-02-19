//! Python AST → Rust body translation.
//!
//! Walks Python statement/expression trees and emits Rust source strings.
//! Entry point: `translate_function_body`.

use std::collections::HashMap;

use rustpython_parser::ast::{Expr, Operator, Stmt};
use tracing::warn;

use crate::utils::TargetSpec;

use super::expr::{expr_to_rust, translate_for_iter, translate_len_guard, translate_while_test};
use super::infer::{infer_assign_type, infer_params};

/// Result of translating a single Python function body.
pub struct Translation {
    pub params: Vec<(String, String)>,
    pub return_type: String,
    pub body: String,
    pub fallback: bool,
}

/// Result of translating a block of Python statements.
pub(super) struct BodyTranslation {
    pub return_type: String,
    pub body: String,
    pub fallback: bool,
}

/// Find and translate the body of the named function in `module`.
///
/// Returns `None` only if the function is not found.
/// Returns a `Translation` with `fallback: true` if the body cannot be translated.
pub fn translate_function_body(target: &TargetSpec, module: &[Stmt]) -> Option<Translation> {
    let func_def = module.iter().find_map(|stmt| match stmt {
        Stmt::FunctionDef(def) if def.name == target.func => Some(def),
        _ => None,
    })?;

    let mut params = infer_params(func_def.args.as_ref());
    if params.is_empty() {
        params.push(("data".to_string(), "Vec<f64>".to_string()));
    }

    // Fast path: single-statement return of a name or constant
    if let Some(Stmt::Return(ret)) = func_def.body.first()
        && let Some(expr) = &ret.value
    {
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

    // Generic body translation
    if let Some(translated) = translate_body(&func_def.body) {
        let first_param_name = params
            .first()
            .map(|(n, _)| n.as_str())
            .unwrap_or("data")
            .to_string();
        if translated.fallback {
            return Some(Translation {
                params,
                return_type: "Vec<f64>".to_string(),
                body: format!("// fallback: echo input\n    Ok({first_param_name})"),
                fallback: true,
            });
        }
        return Some(Translation {
            params,
            return_type: translated.return_type,
            body: translated.body,
            fallback: translated.fallback,
        });
    }

    warn!(func = %target.func, "unable to translate function body; echoing input");
    Some(Translation {
        params,
        return_type: "Vec<f64>".to_string(),
        body: "// fallback: echo input\n    Ok(data)".to_string(),
        fallback: true,
    })
}

pub(super) fn translate_body(body: &[Stmt]) -> Option<BodyTranslation> {
    translate_body_inner(body, 1)
}

/// Recursive body translator. `depth` tracks nesting level for indentation.
pub(super) fn translate_body_inner(body: &[Stmt], depth: usize) -> Option<BodyTranslation> {
    if body.is_empty() {
        return None;
    }

    let indent = "    ".repeat(depth);
    let mut var_types: HashMap<String, &str> = HashMap::new();

    // Generic sequential statement translation
    let mut out = String::new();
    let mut had_unhandled = false;
    let mut inferred_return: Option<String> = None;
    let mut had_return = false;

    for stmt in body {
        // Track simple vector-producing assignments for later return inference
        if let Stmt::Assign(assign) = stmt
            && let Some(target) = assign.targets.first()
        {
            // result = [0.0] * n or result = [expr for ...]
            if let Expr::BinOp(binop) = assign.value.as_ref()
                && matches!(binop.op, Operator::Mult)
                && let Expr::List(lst) = binop.left.as_ref()
                && lst.elts.len() == 1
                && let Expr::Name(name_target) = target
            {
                var_types.insert(name_target.id.to_string(), "vec");
            }
            if let Expr::ListComp(_) = assign.value.as_ref()
                && let Expr::Name(name_target) = target
            {
                var_types.insert(name_target.id.to_string(), "vec");
            }
        }

        match translate_stmt_inner(stmt, depth) {
            Some(line) => {
                if line.trim_start().starts_with("return ") {
                    had_return = true;
                    if let Stmt::Return(ret) = stmt {
                        if let Some(expr) = &ret.value {
                            let ret_ty = infer_return_type(expr.as_ref(), &var_types);
                            inferred_return = Some(ret_ty);
                        }
                    }
                }
                out.push_str(&indent);
                out.push_str(&line);
                if !line.ends_with('\n') {
                    out.push('\n');
                }
            }
            None => {
                had_unhandled = true;
                out.push_str(&indent);
                out.push_str("// Unhandled stmt\n");
            }
        }
    }

    if !had_return {
        if let Some(ret_var) = var_types.keys().find(|k| *k == "result" || *k == "output") {
            inferred_return = Some("Vec<f64>".to_string());
            out.push_str(&format!("{indent}return Ok({});\n", ret_var));
        }
    }

    Some(BodyTranslation {
        return_type: inferred_return.unwrap_or_else(|| "f64".to_string()),
        body: out,
        fallback: had_unhandled,
    })
}

/// Translate a single Python statement to a Rust statement string.
/// Returns `None` for unhandled statement types (triggers fallback).
pub(super) fn translate_stmt_inner(stmt: &Stmt, depth: usize) -> Option<String> {
    match stmt {
        Stmt::Assign(assign) => {
            if let (Some(target), value) = (assign.targets.first(), &assign.value) {
                // Subscript assign: result[i] = val → result[i] = val;
                if let Expr::Subscript(sub) = target {
                    let lhs = format!("{}[{}]", expr_to_rust(&sub.value), expr_to_rust(&sub.slice));
                    let rhs = expr_to_rust(value);
                    return Some(format!("{} = {};", lhs, rhs));
                }
                // List init: result = [0.0] * n → let mut result = vec![0.0f64; n];
                if let Expr::BinOp(binop) = value.as_ref()
                    && matches!(binop.op, Operator::Mult)
                    && let Expr::List(lst) = binop.left.as_ref()
                    && lst.elts.len() == 1
                {
                    let fill = expr_to_rust(&lst.elts[0]);
                    let size = expr_to_rust(&binop.right);
                    let var_name = match target {
                        Expr::Name(n) => n.id.to_string(),
                        _ => "result".to_string(),
                    };
                    let fill_rust = if fill.contains('.') {
                        format!("{}f64", fill)
                    } else {
                        fill.clone()
                    };
                    return Some(format!(
                        "let mut {var} = vec![{fill}; {size}];",
                        var = var_name,
                        fill = fill_rust,
                        size = size
                    ));
                }
                // List comprehension: result = [expr for var in iterable]
                // → let result: Vec<f64> = iterable.iter().map(|var| expr).collect();
                if let Expr::ListComp(lc) = value.as_ref()
                    && lc.generators.len() == 1
                {
                    let comprehension = &lc.generators[0];
                    let iter_str = expr_to_rust(&comprehension.iter);
                    let loop_var = expr_to_rust(&comprehension.target);
                    let elt = expr_to_rust(&lc.elt);
                    let var_name = match target {
                        Expr::Name(n) => n.id.to_string(),
                        _ => "result".to_string(),
                    };
                    return Some(format!(
                        "let {var}: Vec<f64> = {iter}.iter().map(|{lv}| {elt}).collect();",
                        var = var_name,
                        iter = iter_str,
                        lv = loop_var,
                        elt = elt,
                    ));
                }
                // Simple name assign
                let lhs = match target {
                    Expr::Name(n) => {
                        let type_suffix = infer_assign_type(value);
                        format!("let mut {}{}", n.id, type_suffix)
                    }
                    Expr::Attribute(_) => format!("// attribute assign {}", expr_to_rust(target)),
                    _ => format!("// complex assign {}", expr_to_rust(target)),
                };
                let rhs = expr_to_rust(value);
                return Some(format!("{} = {};", lhs, rhs));
            }
            None
        }
        Stmt::For(for_stmt) => {
            let iter_str = translate_for_iter(&for_stmt.iter);
            let loop_var = expr_to_rust(&for_stmt.target);
            let inner = translate_body_inner(for_stmt.body.as_slice(), depth + 1);
            let loop_body = inner
                .map(|b| b.body)
                .unwrap_or_else(|| "    // unhandled loop body".to_string());
            Some(format!(
                "for {loop_var} in {iter_str} {{\n{loop_body}\n{indent}}}",
                loop_var = loop_var,
                iter_str = iter_str,
                loop_body = loop_body,
                indent = "    ".repeat(depth)
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
        Stmt::While(while_stmt) => {
            let test = translate_while_test(&while_stmt.test);
            let inner = translate_body_inner(while_stmt.body.as_slice(), depth + 1);
            let loop_body = inner
                .map(|b| b.body)
                .unwrap_or_else(|| format!("{}    // unhandled while body", "    ".repeat(depth)));
            Some(format!(
                "while {test} {{\n{loop_body}\n{indent}}}",
                test = test,
                loop_body = loop_body,
                indent = "    ".repeat(depth)
            ))
        }
        Stmt::Return(ret) => {
            if let Some(v) = &ret.value {
                Some(format!("return Ok({});", expr_to_rust(v)))
            } else {
                Some("return Ok(());".to_string())
            }
        }
        Stmt::Expr(expr_stmt) => {
            // Docstring (string constant) → comment, not fallback
            if let Expr::Constant(c) = expr_stmt.value.as_ref()
                && matches!(c.value, rustpython_parser::ast::Constant::Str(_))
            {
                return Some("// docstring omitted".to_string());
            }
            if let Expr::Call(call) = expr_stmt.value.as_ref()
                && let Expr::Attribute(attr) = call.func.as_ref()
                && attr.attr.as_str() == "append"
                && call.args.len() == 1
            {
                let target = expr_to_rust(&attr.value);
                let arg = expr_to_rust(&call.args[0]);
                return Some(format!("{target}.push({arg});"));
            }
            Some(format!("// expr: {}", expr_to_rust(&expr_stmt.value)))
        }
        Stmt::If(if_stmt) => {
            if let Some(guard) = translate_len_guard(&if_stmt.test) {
                return Some(guard);
            }
            let test = expr_to_rust(&if_stmt.test);
            let body = translate_body_inner(if_stmt.body.as_slice(), depth + 1)
                .map(|b| b.body)
                .unwrap_or_else(|| "// unhandled if body".to_string());
            let orelse = if !if_stmt.orelse.is_empty() {
                translate_body_inner(if_stmt.orelse.as_slice(), depth + 1)
                    .map(|b| b.body)
                    .unwrap_or_else(|| "// unhandled else body".to_string())
            } else {
                String::new()
            };
            let else_block = if orelse.is_empty() {
                String::new()
            } else {
                format!(" else {{\n{}\n{}}}", orelse, "    ".repeat(depth))
            };
            Some(format!(
                "if {test} {{\n{body}\n{indent}}}{else_block}",
                test = test,
                body = body,
                indent = "    ".repeat(depth),
                else_block = else_block
            ))
        }
        _ => None,
    }
}

fn infer_return_type(expr: &Expr, var_types: &HashMap<String, &str>) -> String {
    match expr {
        Expr::Name(n) => {
            if let Some(&"vec") = var_types.get(n.id.as_str()) {
                return "Vec<f64>".to_string();
            }
            "f64".to_string()
        }
        Expr::List(_) | Expr::ListComp(_) => "Vec<f64>".to_string(),
        Expr::Tuple(_) => "Vec<f64>".to_string(),
        _ => "f64".to_string(),
    }
}
