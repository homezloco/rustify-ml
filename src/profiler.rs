use std::process::Command;

use anyhow::{Context, Result, anyhow};
use tracing::{info, warn};

use crate::utils::{Hotspot, InputSource, ProfileSummary};

/// Detect the Python executable name available on this system.
/// Tries `python3` first (Linux/macOS convention), then falls back to `python`.
/// Returns an error if neither is found.
pub fn detect_python() -> Result<String> {
    for candidate in &["python3", "python"] {
        if let Ok(output) = Command::new(candidate).arg("--version").output()
            && output.status.success()
        {
            return Ok(candidate.to_string());
        }
    }
    Err(anyhow!(
        "Python not found on PATH. Install Python 3.10+ and ensure it is on PATH."
    ))
}

/// Check that the detected Python is >= 3.10. Warns (does not error) if older.
fn check_python_version(python: &str) {
    let check = Command::new(python)
        .args([
            "-c",
            "import sys; ok = sys.version_info >= (3, 10); print('ok' if ok else f'old:{sys.version}')",
        ])
        .output();

    match check {
        Ok(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let result = stdout.trim();
            if result == "ok" {
                info!(python, "Python version check passed (>= 3.10)");
            } else {
                warn!(
                    python,
                    version = result.trim_start_matches("old:"),
                    "Python < 3.10 detected; some profiling features may behave differently"
                );
            }
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            warn!(python, err = %stderr.trim(), "Python version check failed");
        }
        Err(e) => {
            warn!(python, err = %e, "could not run Python version check");
        }
    }
}

/// Profile Python code with a configurable iteration count.
/// Wraps `profile_input_core` with the given loop count.
pub fn profile_input_with_iterations(
    source: &InputSource,
    threshold: f32,
    iterations: u32,
) -> Result<ProfileSummary> {
    profile_input_core(source, threshold, iterations)
}

/// Profile Python code using the built-in cProfile via a Python subprocess.
/// Uses a default of 100 iterations.
pub fn profile_input(source: &InputSource, threshold: f32) -> Result<ProfileSummary> {
    profile_input_core(source, threshold, 100)
}

/// Core profiling implementation.
fn profile_input_core(source: &InputSource, threshold: f32, iterations: u32) -> Result<ProfileSummary> {
    let python = detect_python()?;
    check_python_version(&python);

    // Write the Python script to a temp file for execution; keep dir alive for the duration of profiling
    let (path, _tmpdir) = crate::utils::materialize_input(source)?;

    let profiler = format!(
        r#"
import cProfile, pstats, runpy
# Use a non-__main__ run_name to avoid executing script-side benchmarks guarded by
# if __name__ == "__main__": blocks (prevents hangs during profiling).
_iters = {iterations}
prof = cProfile.Profile()
prof.enable()
for _ in range(_iters):
    runpy.run_path(r"{path}", run_name="__rustify_profile__")
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

    let output = Command::new(&python)
        .args(["-c", &profiler])
        .output()
        .with_context(|| {
            format!(
                "failed to run {} for profiling; ensure Python is installed",
                python
            )
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        warn!("python profiling failed: {}", stderr.trim());
        return Err(anyhow!("python profiling failed: {}", stderr.trim()));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut hotspots = Vec::new();
    for line in stdout.lines() {
        // expected line: "pct% func file:line"
        if let Some((percent_part, rest)) = line.split_once(' ')
            && let Ok(percent) = percent_part.trim().trim_end_matches('%').parse::<f32>()
        {
            // Skip built-in and internal Python frames
            if rest.contains("<built-in") || rest.contains("<frozen") {
                continue;
            }
            let mut parts = rest.rsplitn(2, ':');
            if let (Some(line_part), Some(func_part)) = (parts.next(), parts.next())
                && let Ok(line_no) = line_part.parse::<u32>()
            {
                hotspots.push(Hotspot {
                    func: func_part.trim().to_string(),
                    line: line_no,
                    percent,
                });
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
