use rustpython_parser::Parse;
use rustpython_parser::ast::{Stmt, Suite};
use tracing::info;

use crate::utils::{InputSource, ProfileSummary, TargetSpec, extract_code};

/// Select target functions to generate based on hotspot percentages and ML mode heuristics (placeholder heuristics).
pub fn select_targets(
    profile: &ProfileSummary,
    source: &InputSource,
    threshold: f32,
    ml_mode: bool,
) -> Vec<TargetSpec> {
    let mut targets = Vec::new();
    for hs in &profile.hotspots {
        if hs.percent < threshold {
            continue;
        }
        let reason = if ml_mode {
            format!("{}% hotspot (ml-mode)", hs.percent)
        } else {
            format!("{}% hotspot", hs.percent)
        };
        targets.push(TargetSpec {
            func: hs.func.clone(),
            line: hs.line,
            percent: hs.percent,
            reason,
        });
    }

    if threshold <= 0.0
        && let Ok(code) = extract_code(source)
        && let Ok(suite) = Suite::parse(&code, "<input>")
    {
        for stmt in suite.iter() {
            if let Stmt::FunctionDef(func_def) = stmt {
                let name = func_def.name.to_string();
                if targets.iter().any(|t| t.func == name) {
                    continue;
                }
                // Source line is optional here; default to 1 if unavailable.
                targets.push(TargetSpec {
                    func: name,
                    line: 1,
                    percent: 0.0,
                    reason: "threshold<=0: include all defs".to_string(),
                });
            }
        }
    }

    info!(count = targets.len(), "selected targets for generation");
    targets
}
