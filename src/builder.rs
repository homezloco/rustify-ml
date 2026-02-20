use std::process::Command;

use anyhow::{Context, Result, anyhow};
use tracing::{info, warn};

use crate::utils::{GenerationResult, InputSource, TargetSpec, extract_code};

/// Run `cargo check` on the generated crate to catch translation errors early.
/// Returns Ok(()) if the check passes, or an error with the compiler output.
/// This is a fast-fail step: it does NOT require maturin or a Python environment.
pub fn cargo_check_generated(r#gen: &GenerationResult) -> Result<()> {
    info!(path = %r#gen.crate_dir.display(), "running cargo check on generated crate");

    let output = Command::new("cargo")
        .args(["check", "--message-format=short"])
        .current_dir(&r#gen.crate_dir)
        .output()
        .context("failed to spawn cargo; ensure Rust toolchain is installed")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let combined = format!("{}\n{}", stdout.trim(), stderr.trim());
        warn!(
            path = %r#gen.crate_dir.display(),
            "cargo check failed on generated crate — review generated code:\n{}",
            combined
        );
        return Err(anyhow!(
            "generated Rust code failed cargo check. Review {} and fix translation issues.\n\nCompiler output:\n{}",
            r#gen.crate_dir.join("src/lib.rs").display(),
            combined
        ));
    }

    info!(path = %r#gen.crate_dir.display(), "cargo check passed on generated crate");
    Ok(())
}

pub fn build_extension(r#gen: &GenerationResult, dry_run: bool) -> Result<()> {
    if dry_run {
        info!(path = %r#gen.crate_dir.display(), "dry-run: skipping maturin build");
        return Ok(());
    }

    // Run cargo check first as a fast-fail before the full maturin build.
    // Warn but don't abort if cargo is not available (e.g., unusual CI setups).
    if let Err(e) = cargo_check_generated(r#gen) {
        warn!(
            path = %r#gen.crate_dir.display(),
            err = %e,
            "cargo check failed; proceeding with maturin anyway (review generated code)"
        );
    }

    // Ensure maturin is available with a user-friendly hint.
    if let Err(e) = Command::new("maturin").arg("--version").output() {
        return Err(anyhow!(
            "maturin not found: install with `pip install maturin` and ensure it is on PATH (error: {e})"
        ));
    }

    let status = Command::new("maturin")
        .args(["develop", "--release"])
        .current_dir(&r#gen.crate_dir)
        .status()
        .context("failed to spawn maturin; install via `pip install maturin` and ensure on PATH")?;

    if !status.success() {
        return Err(anyhow!("maturin build failed with status {status}"));
    }

    info!(
        path = %r#gen.crate_dir.display(),
        fallback_functions = r#gen.fallback_functions,
        "maturin build completed"
    );
    Ok(())
}

/// Run a Python timing harness comparing the original Python function against the
/// generated Rust extension. Prints a speedup table to stdout.
///
/// Requires: maturin develop already run (extension importable), Python on PATH.
pub fn run_benchmark(
    source: &InputSource,
    result: &GenerationResult,
    targets: &[TargetSpec],
) -> Result<()> {
    use crate::profiler::detect_python;

    let python = detect_python()?;
    let code = extract_code(source)?;
    let module_name = result
        .crate_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("rustify_ml_ext");

    // Build a self-contained Python benchmark script
    let func_names: Vec<String> = targets
        .iter()
        .filter(|t| {
            // Only benchmark functions that were fully translated (no fallback)
            result
                .generated_functions
                .iter()
                .any(|f| f.contains(&format!("pub fn {}", t.func)) && !f.contains("// fallback"))
        })
        .map(|t| t.func.clone())
        .collect();

    if func_names.is_empty() {
        warn!("no fully-translated functions to benchmark; skipping");
        return Ok(());
    }

    let harness = build_benchmark_harness(&code, module_name, &func_names);

    let output = Command::new(&python)
        .args(["-c", &harness])
        .output()
        .with_context(|| format!("failed to run {} for benchmark", python))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("benchmark harness failed: {}", stderr.trim()));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("\n{}", stdout.trim());
    Ok(())
}

/// Generate a Python benchmark script that times original vs Rust for each function.
fn build_benchmark_harness(code: &str, module_name: &str, func_names: &[String]) -> String {
    let escaped_code = code.replace('\\', "\\\\").replace('"', "\\\"");
    let funcs_list = func_names
        .iter()
        .map(|f| format!("\"{}\"", f))
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        r#"
import timeit, sys, importlib, types

# --- original Python code ---
_src = """{code}"""
_mod = types.ModuleType("_orig")
exec(compile(_src, "<rustify_bench>", "exec"), _mod.__dict__)

# --- accelerated Rust extension ---
try:
    _ext = importlib.import_module("{module}")
except ImportError as e:
    print(f"Could not import {module}: {{e}}")
    sys.exit(1)

_funcs = [{funcs}]
_iters = 1000

print()
print(f"{{'':-<60}}")
print(f"  rustify-ml benchmark  ({{_iters}} iterations each)")
print(f"{{'':-<60}}")
print(f"  {{\"Function\":<22}} | {{\"Python\":>10}} | {{\"Rust\":>10}} | {{\"Speedup\":>8}}")
print(f"  {{'':-<22}}-+-{{'':-<10}}-+-{{'':-<10}}-+-{{'':-<8}}")

for fn_name in _funcs:
    py_fn = getattr(_mod, fn_name, None)
    rs_fn = getattr(_ext, fn_name, None)
    if py_fn is None or rs_fn is None:
        print(f"  {{fn_name:<22}} | skipped (not found)")
        continue
    # Build a simple call with dummy float vectors
    try:
        import inspect
        sig = inspect.signature(py_fn)
        n_params = len(sig.parameters)
        dummy = [float(i) for i in range(100)]
        args = tuple(dummy for _ in range(n_params))
        py_time = timeit.timeit(lambda: py_fn(*args), number=_iters)
        rs_time = timeit.timeit(lambda: rs_fn(*args), number=_iters)
        speedup = py_time / rs_time if rs_time > 0 else float("inf")
        print(f"  {{fn_name:<22}} | {{py_time:>9.4f}}s | {{rs_time:>9.4f}}s | {{speedup:>7.1f}}x")
    except Exception as e:
        print(f"  {{fn_name:<22}} | error: {{e}}")

print(f"{{'':-<60}}")
print()
"#,
        code = escaped_code,
        module = module_name,
        funcs = funcs_list,
    )
}
