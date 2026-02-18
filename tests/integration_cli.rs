/// CLI integration tests for rustify-ml.
///
/// These tests invoke the compiled binary end-to-end using `--dry-run`
/// so they do NOT require maturin or a Python build environment.
/// They verify the full pipeline: input → profile → analyze → generate.
///
/// NOTE: These tests require Python to be on PATH (for the profiler step).
/// They are skipped gracefully if Python is not available.
use std::process::Command;

use tempfile::tempdir;

/// Returns true if `python` is available on PATH.
fn python_available() -> bool {
    Command::new("python")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[test]
fn accelerate_euclidean_dry_run_writes_generated_lib() {
    if !python_available() {
        eprintln!("SKIP: python not available on PATH");
        return;
    }

    let bin = env!("CARGO_BIN_EXE_rustify-ml");
    let tmp = tempdir().expect("tempdir");
    let output_dir = tmp.path().join("dist");

    // Use --threshold 0 to ensure all profiled functions are selected,
    // regardless of how cProfile distributes time across the short run.
    let status = Command::new(bin)
        .args([
            "accelerate",
            "--file",
            "examples/euclidean.py",
            "--output",
            output_dir.to_str().unwrap(),
            "--threshold",
            "0",
            "--dry-run",
        ])
        .status()
        .expect("failed to spawn rustify-ml binary");

    assert!(status.success(), "accelerate command failed");

    let lib_rs = output_dir.join("rustify_ml_ext/src/lib.rs");
    assert!(lib_rs.exists(), "lib.rs not generated in dry-run");

    let lib_contents = std::fs::read_to_string(&lib_rs).expect("read lib.rs");
    assert!(
        lib_contents.contains("euclidean"),
        "expected 'euclidean' in generated lib.rs"
    );
    assert!(
        lib_contents.contains("powf"),
        "expected 'powf' (from total ** 0.5) in generated lib.rs"
    );
    assert!(
        lib_contents.contains("length mismatch") || lib_contents.contains("len() !="),
        "expected length-check guard for two Vec params"
    );
}

#[test]
fn accelerate_missing_input_exits_nonzero() {
    let bin = env!("CARGO_BIN_EXE_rustify-ml");
    let tmp = tempdir().expect("tempdir");
    let output_dir = tmp.path().join("dist");

    // No --file, --snippet, or --git → should fail with non-zero exit
    let status = Command::new(bin)
        .args([
            "accelerate",
            "--output",
            output_dir.to_str().unwrap(),
            "--dry-run",
        ])
        .status()
        .expect("failed to spawn rustify-ml binary");

    assert!(
        !status.success(),
        "expected non-zero exit when no input provided"
    );
}

#[test]
fn accelerate_nonexistent_file_exits_nonzero() {
    let bin = env!("CARGO_BIN_EXE_rustify-ml");
    let tmp = tempdir().expect("tempdir");
    let output_dir = tmp.path().join("dist");

    let status = Command::new(bin)
        .args([
            "accelerate",
            "--file",
            "examples/does_not_exist.py",
            "--output",
            output_dir.to_str().unwrap(),
            "--dry-run",
        ])
        .status()
        .expect("failed to spawn rustify-ml binary");

    assert!(
        !status.success(),
        "expected non-zero exit for missing file"
    );
}
