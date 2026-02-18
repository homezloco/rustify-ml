use std::process::Command;

use tempfile::tempdir;

#[test]
fn accelerate_euclidean_dry_run_writes_generated_lib() {
    let bin = env!("CARGO_BIN_EXE_rustify-ml");
    let tmp = tempdir().expect("tempdir");
    let output_dir = tmp.path().join("dist");

    let status = Command::new(bin)
        .args([
            "accelerate",
            "--file",
            "examples/euclidean.py",
            "--output",
            output_dir.to_str().unwrap(),
            "--dry-run",
        ])
        .status()
        .expect("failed to spawn rustify-ml binary");

    assert!(status.success(), "accelerate command failed");

    let lib_rs = output_dir.join("rustify_ml_ext/src/lib.rs");
    assert!(lib_rs.exists(), "lib.rs not generated in dry-run");

    let lib_contents = std::fs::read_to_string(&lib_rs).expect("read lib.rs");
    assert!(lib_contents.contains("euclidean"));
    assert!(lib_contents.contains("powf"));
    assert!(lib_contents.contains("Vectors must be same length"));
}
