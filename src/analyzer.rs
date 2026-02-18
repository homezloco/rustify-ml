use tracing::info;

use crate::utils::{ProfileSummary, TargetSpec};

/// Select target functions to generate based on hotspot percentages and ML mode heuristics (placeholder heuristics).
pub fn select_targets(profile: &ProfileSummary, threshold: f32, ml_mode: bool) -> Vec<TargetSpec> {
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
    info!(count = targets.len(), "selected targets for generation");
    targets
}
