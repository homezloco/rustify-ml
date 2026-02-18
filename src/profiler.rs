use std::process::Command;

use anyhow::{Context, Result, anyhow};
use tracing::{info, warn};

use crate::utils::{Hotspot, InputSource, ProfileSummary};

/// Profile Python code using the built-in cProfile via a Python subprocess.
/// Parses cumulative time percentages into Hotspot records.
pub fn profile_input(source: &InputSource, threshold: f32) -> Result<ProfileSummary> {
    // Write the Python script to a temp file for execution; keep dir alive for the duration of profiling
    let (path, _tmpdir) = crate::utils::materialize_input(source)?;

    let profiler = format!(
        r#"
import cProfile, pstats, runpy
prof = cProfile.Profile()
prof.enable()
runpy.run_path(r"{path}", run_name="__main__")
prof.disable()
stats = pstats.Stats(prof)
total = sum(v[3] for v in stats.stats.values()) or 1e-9
for (fname, line, func), stat in stats.stats.items():
    ct = stat[3]
    pct = (ct / total) * 100.0
    print(f"{{pct:.2f}}% {{func}} {{fname}}:{{line}}")
"#,
        path = path.display()
    );

    let output = Command::new("python")
        .args(["-c", &profiler])
        .output()
        .context("failed to run python for profiling; ensure Python is installed")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        warn!("python profiling failed: {}", stderr.trim());
        return Err(anyhow!("python profiling failed: {}", stderr.trim()));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut hotspots = Vec::new();
    for line in stdout.lines() {
        // expected line: "pct% func file:line"
        if let Some((percent_part, rest)) = line.split_once(' ') {
            if let Ok(percent) = percent_part.trim().trim_end_matches('%').parse::<f32>() {
                let mut parts = rest.rsplitn(2, ':');
                if let (Some(line_part), Some(func_part)) = (parts.next(), parts.next()) {
                    if let Ok(line_no) = line_part.parse::<u32>() {
                        hotspots.push(Hotspot {
                            func: func_part.trim().to_string(),
                            line: line_no,
                            percent,
                        });
                    }
                }
            }
        }
    }

    hotspots.retain(|h| h.percent >= threshold);
    hotspots.sort_by(|a, b| b.percent.total_cmp(&a.percent));
    info!(
        count = hotspots.len(),
        threshold, "profiled hotspots collected"
    );

    Ok(ProfileSummary { hotspots })
}
