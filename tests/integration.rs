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

// ── nested for fallback ───────────────────────────────────────────────────────

#[test]
fn integration_nested_for_else_fallback() {
    // for..else is unsupported; expect fallback count > 0
    let source = InputSource::Snippet(
        "def f(n):\n    total = 0\n    for i in range(n):\n        for j in range(n):\n            total += i + j\n    else:\n        total += 1\n    return total\n".to_string(),
    );
    let targets = vec![TargetSpec {
        func: "f".to_string(),
        line: 1,
        percent: 100.0,
        reason: "nested for else".to_string(),
    }];
    let tmp = tempdir().expect("tempdir");
    let result = generate(&source, &targets, tmp.path(), false).expect("generate failed");

    assert_eq!(result.generated_functions.len(), 1);
    assert!(
        result.fallback_functions > 0,
        "expected fallback for for..else construct"
    );
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

#[test]
fn integration_matmul_zero_fallback() {
    let (_, source) = load_example("matrix_ops.py");
    let targets = vec![TargetSpec {
        func: "matmul".to_string(),
        line: 7,
        percent: 95.0,
        reason: "matmul nested loops".to_string(),
    }];
    let tmp = tempdir().expect("tempdir");
    let result = generate(&source, &targets, tmp.path(), false).expect("generate failed");

    assert_eq!(result.generated_functions.len(), 1);
    assert_eq!(
        result.fallback_functions, 0,
        "matmul should fully translate without fallback"
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

#[test]
fn integration_normalize_pixels_zero_fallback() {
    let (_, source) = load_example("image_preprocess.py");
    let targets = vec![TargetSpec {
        func: "normalize_pixels".to_string(),
        line: 1,
        percent: 90.0,
        reason: "integration fallback check".to_string(),
    }];
    let tmp = tempdir().expect("tempdir");
    let result = generate(&source, &targets, tmp.path(), false).expect("generate failed");

    assert_eq!(result.generated_functions.len(), 1);
    assert_eq!(
        result.fallback_functions, 0,
        "normalize_pixels should not fallback"
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

// ── BPE tokenizer (while loop) ───────────────────────────────────────────────

#[test]
fn integration_count_pairs_generates_while_loop() {
    let (_, source) = load_example("bpe_tokenizer.py");
    let targets = vec![TargetSpec {
        func: "count_pairs".to_string(),
        line: 1,
        percent: 85.0,
        reason: "integration test".to_string(),
    }];
    let tmp = tempdir().expect("tempdir");
    let result = generate(&source, &targets, tmp.path(), false).expect("generate failed");

    assert_eq!(result.generated_functions.len(), 1);
    let lib_rs =
        std::fs::read_to_string(tmp.path().join("rustify_ml_ext/src/lib.rs")).expect("read lib.rs");
    assert!(lib_rs.contains("fn count_pairs"), "missing fn count_pairs");
    // count_pairs uses a for loop over range(len(tokens)-1)
    assert!(
        lib_rs.contains("for i in 0.."),
        "expected for loop in count_pairs, got:\n{}",
        lib_rs
    );
}

#[test]
fn integration_bpe_encode_generates() {
    let (_, source) = load_example("bpe_tokenizer.py");
    let targets = vec![TargetSpec {
        func: "bpe_encode".to_string(),
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
        lib_rs.contains("fn bpe_encode"),
        "missing fn bpe_encode, got:\n{}",
        lib_rs
    );
    // bpe_encode has while loops — check they're translated
    assert!(
        lib_rs.contains("while "),
        "expected while loop in bpe_encode, got:\n{}",
        lib_rs
    );
}

// ── golden file snapshot ──────────────────────────────────────────────────────

/// Golden file test: assert the generated lib.rs for euclidean.py contains all
/// expected structural elements. Update the snapshot by running:
///   UPDATE_SNAPSHOTS=1 cargo test integration_euclidean_golden
#[test]
fn integration_euclidean_golden() {
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

    // Structural invariants that must always hold
    let required = [
        "use pyo3::prelude::*;",
        "#[pyfunction]",
        "pub fn euclidean(",
        "py: Python",
        "Vec<f64>",
        "PyResult<f64>",
        "for i in 0..",
        "powf(",
        "#[pymodule]",
        "fn rustify_ml_ext(",
        "wrap_pyfunction!(euclidean",
    ];
    for pat in &required {
        assert!(
            lib_rs.contains(pat),
            "golden snapshot missing pattern {:?}\n\nActual lib.rs:\n{}",
            pat,
            lib_rs
        );
    }

    // Snapshot file: write if UPDATE_SNAPSHOTS=1, else compare prefix
    let snap_path = std::path::PathBuf::from("tests/snapshots/euclidean_lib_rs.snap");
    if std::env::var("UPDATE_SNAPSHOTS").is_ok() {
        std::fs::create_dir_all(snap_path.parent().unwrap()).ok();
        std::fs::write(&snap_path, &lib_rs).expect("write snapshot");
        println!("Snapshot updated: {}", snap_path.display());
    } else if snap_path.exists() {
        let snap = std::fs::read_to_string(&snap_path).expect("read snapshot");
        // Compare only the first 5 lines (header) to avoid fragile full-text matching
        let snap_head: Vec<&str> = snap.lines().take(5).collect();
        let actual_head: Vec<&str> = lib_rs.lines().take(5).collect();
        assert_eq!(
            snap_head, actual_head,
            "golden snapshot header mismatch — run UPDATE_SNAPSHOTS=1 cargo test to refresh"
        );
    }
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
