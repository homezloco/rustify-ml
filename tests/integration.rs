/// Integration tests for rustify-ml pipeline stages.
///
/// These tests exercise the full generate() pipeline using the example
/// Python fixtures in examples/. They do NOT invoke maturin (no build
/// toolchain required) — they verify that:
///   1. The generator produces valid Rust source files on disk.
///   2. Zero fallback functions for well-formed translatable hotspots.
///   3. Generated lib.rs contains expected function names and PyO3 boilerplate.
use std::path::PathBuf;

use rustify_ml::generator::generate;
use rustify_ml::utils::{InputSource, TargetSpec};
use tempfile::tempdir;

fn load_example(name: &str) -> (PathBuf, InputSource) {
    let path = PathBuf::from(format!("examples/{}", name));
    let code = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read examples/{}: {}", name, e));
    let source = InputSource::File {
        path: path.clone(),
        code,
    };
    (path, source)
}

// ── euclidean ────────────────────────────────────────────────────────────────

#[test]
fn integration_euclidean_zero_fallback() {
    let (_, source) = load_example("euclidean.py");
    let targets = vec![TargetSpec {
        func: "euclidean".to_string(),
        line: 1,
        percent: 100.0,
        reason: "integration test".to_string(),
    }];
    let tmp = tempdir().expect("tempdir");
    let result = generate(&source, &targets, tmp.path(), false).expect("generate failed");

    assert_eq!(result.generated_functions.len(), 1, "expected 1 function");
    assert_eq!(
        result.fallback_functions, 0,
        "euclidean should not fallback"
    );
    assert!(
        tmp.path().join("rustify_ml_ext/src/lib.rs").exists(),
        "lib.rs not written"
    );
    assert!(
        tmp.path().join("rustify_ml_ext/Cargo.toml").exists(),
        "Cargo.toml not written"
    );
}

#[test]
fn integration_euclidean_lib_rs_content() {
    let (_, source) = load_example("euclidean.py");
    let targets = vec![TargetSpec {
        func: "euclidean".to_string(),
        line: 1,
        percent: 100.0,
        reason: "integration test".to_string(),
    }];
    let tmp = tempdir().expect("tempdir");
    generate(&source, &targets, tmp.path(), false).expect("generate failed");

    let lib_rs =
        std::fs::read_to_string(tmp.path().join("rustify_ml_ext/src/lib.rs")).expect("read lib.rs");

    assert!(
        lib_rs.contains("use pyo3::prelude::*"),
        "missing pyo3 import"
    );
    assert!(lib_rs.contains("#[pyfunction]"), "missing #[pyfunction]");
    assert!(lib_rs.contains("#[pymodule]"), "missing #[pymodule]");
    assert!(lib_rs.contains("fn euclidean"), "missing fn euclidean");
    assert!(
        lib_rs.contains("for i in 0.."),
        "missing translated for loop"
    );
    assert!(lib_rs.contains("total"), "missing accumulator variable");
}

#[test]
fn integration_euclidean_len_check_emitted() {
    let (_, source) = load_example("euclidean.py");
    let targets = vec![TargetSpec {
        func: "euclidean".to_string(),
        line: 1,
        percent: 100.0,
        reason: "integration test".to_string(),
    }];
    let tmp = tempdir().expect("tempdir");
    generate(&source, &targets, tmp.path(), false).expect("generate failed");

    let lib_rs =
        std::fs::read_to_string(tmp.path().join("rustify_ml_ext/src/lib.rs")).expect("read lib.rs");

    // euclidean(p1, p2) has two Vec params → length check must be emitted
    assert!(
        lib_rs.contains("len() != ") || lib_rs.contains("length mismatch"),
        "expected length-check guard for two Vec params"
    );
}

// ── dot_product (matrix_ops) ─────────────────────────────────────────────────

#[test]
fn integration_dot_product_zero_fallback() {
    let (_, source) = load_example("matrix_ops.py");
    let targets = vec![TargetSpec {
        func: "dot_product".to_string(),
        line: 1,
        percent: 80.0,
        reason: "integration test".to_string(),
    }];
    let tmp = tempdir().expect("tempdir");
    let result = generate(&source, &targets, tmp.path(), false).expect("generate failed");

    assert_eq!(result.generated_functions.len(), 1);
    assert_eq!(
        result.fallback_functions, 0,
        "dot_product should not fallback"
    );
}

// ── normalize_pixels (image_preprocess) ──────────────────────────────────────

#[test]
fn integration_normalize_pixels_generates() {
    let (_, source) = load_example("image_preprocess.py");
    let targets = vec![TargetSpec {
        func: "normalize_pixels".to_string(),
        line: 1,
        percent: 90.0,
        reason: "integration test".to_string(),
    }];
    let tmp = tempdir().expect("tempdir");
    let result = generate(&source, &targets, tmp.path(), false).expect("generate failed");

    assert_eq!(result.generated_functions.len(), 1);
    let lib_rs =
        std::fs::read_to_string(tmp.path().join("rustify_ml_ext/src/lib.rs")).expect("read lib.rs");
    assert!(
        lib_rs.contains("fn normalize_pixels"),
        "missing fn normalize_pixels"
    );
}

// ── dry_run writes files ──────────────────────────────────────────────────────

#[test]
fn integration_dry_run_still_writes_files() {
    let (_, source) = load_example("euclidean.py");
    let targets = vec![TargetSpec {
        func: "euclidean".to_string(),
        line: 1,
        percent: 100.0,
        reason: "dry-run test".to_string(),
    }];
    let tmp = tempdir().expect("tempdir");
    let result = generate(&source, &targets, tmp.path(), true).expect("generate failed");

    // dry_run=true still writes files (for inspection), just skips maturin
    assert_eq!(result.generated_functions.len(), 1);
    assert!(
        tmp.path().join("rustify_ml_ext/src/lib.rs").exists(),
        "dry-run should still write lib.rs"
    );
}

// ── multiple targets ──────────────────────────────────────────────────────────

#[test]
fn integration_multiple_targets_same_file() {
    let (_, source) = load_example("matrix_ops.py");
    let targets = vec![
        TargetSpec {
            func: "dot_product".to_string(),
            line: 1,
            percent: 60.0,
            reason: "integration test".to_string(),
        },
        TargetSpec {
            func: "matmul".to_string(),
            line: 1,
            percent: 40.0,
            reason: "integration test".to_string(),
        },
    ];
    let tmp = tempdir().expect("tempdir");
    let result = generate(&source, &targets, tmp.path(), false).expect("generate failed");

    assert_eq!(result.generated_functions.len(), 2, "expected 2 functions");
    let lib_rs =
        std::fs::read_to_string(tmp.path().join("rustify_ml_ext/src/lib.rs")).expect("read lib.rs");
    assert!(lib_rs.contains("fn dot_product"), "missing dot_product");
    assert!(lib_rs.contains("fn matmul"), "missing matmul");
}

// ── pow translation ───────────────────────────────────────────────────────────

#[test]
fn integration_euclidean_pow_translated() {
    let (_, source) = load_example("euclidean.py");
    let targets = vec![TargetSpec {
        func: "euclidean".to_string(),
        line: 1,
        percent: 100.0,
        reason: "pow test".to_string(),
    }];
    let tmp = tempdir().expect("tempdir");
    generate(&source, &targets, tmp.path(), false).expect("generate failed");

    let lib_rs =
        std::fs::read_to_string(tmp.path().join("rustify_ml_ext/src/lib.rs")).expect("read lib.rs");

    // total ** 0.5 should translate to (total).powf(0.5)
    assert!(
        lib_rs.contains("powf"),
        "expected powf translation for ** operator"
    );
}
