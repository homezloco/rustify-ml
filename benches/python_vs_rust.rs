use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

#[derive(Clone, Copy)]
struct Row {
    a: f64,
    b: f64,
    c: f64,
}

#[derive(Clone, Copy)]
struct Features {
    log_a: f64,
    log_b: f64,
    log_c: f64,
    scaled_a: f64,
    scaled_b: f64,
    scaled_c: f64,
    centered_a: f64,
    centered_b: f64,
    centered_c: f64,
}

fn rust_apply_rows(rows: &[Row]) -> Vec<Features> {
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let Row { a, b, c } = *row;
        let log_a = if a > 0.0 { (1.0 + a).ln() } else { 0.0 };
        let log_b = if b > 0.0 { (1.0 + b).ln() } else { 0.0 };
        let log_c = if c > 0.0 { (1.0 + c).ln() } else { 0.0 };
        let scaled_a = a * 1.5 + 2.0;
        let scaled_b = b * 1.5 + 2.0;
        let scaled_c = c * 1.5 + 2.0;
        out.push(Features {
            log_a,
            log_b,
            log_c,
            scaled_a,
            scaled_b,
            scaled_c,
            centered_a: a - 0.5,
            centered_b: b - 0.5,
            centered_c: c - 0.5,
        });
    }
    out
}

fn python_apply_rows(rows: usize) -> anyhow::Result<Duration> {
    // Minimal Python baseline to compare against Rust.
    // Uses the same data pattern as rust_apply_rows.
    let script = format!(
        r#"from math import log1p
rows = [{{"a": float(i % 7), "b": float(i % 5), "c": float(i % 3)}} for i in range({rows})]

def featurize_row(row):
    out = {{}}
    for key, value in row.items():
        scaled = value * 1.5 + 2.0
        if value > 0:
            out[f"log_{{key}}"] = log1p(value)
        out[f"scaled_{{key}}"] = scaled
        out[f"centered_{{key}}"] = value - 0.5
    return out

def apply_rows(rows):
    return [featurize_row(r) for r in rows]

if __name__ == "__main__":
    import time
    start = time.perf_counter()
    apply_rows(rows)
    elapsed = time.perf_counter() - start
    print(f"{{elapsed:.6f}}")"#
    );

    let child = Command::new("python3")
        .arg("-c")
        .arg(script)
        .stdout(Stdio::piped())
        .spawn()?;

    let output = child.wait_with_output()?;
    if !output.status.success() {
        anyhow::bail!("python baseline failed: status {:?}", output.status);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout
        .lines()
        .last()
        .ok_or_else(|| anyhow::anyhow!("no output from python baseline"))?;
    let seconds: f64 = line
        .trim()
        .parse()
        .map_err(|e| anyhow::anyhow!("parse baseline seconds: {e}"))?;
    Ok(Duration::from_secs_f64(seconds))
}

fn make_rows(n: usize) -> Vec<Row> {
    (0..n)
        .map(|i| Row {
            a: (i % 7) as f64,
            b: (i % 5) as f64,
            c: (i % 3) as f64,
        })
        .collect()
}

fn bench_python_vs_rust(c: &mut Criterion) {
    let rows = make_rows(50_000);
    let mut group = c.benchmark_group("apply_rows_python_vs_rust");
    group.sample_size(20); // avoid excessive python invocations

    group.bench_function(BenchmarkId::new("python", rows.len()), |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let start = Instant::now();
                let duration = python_apply_rows(rows.len())
                    .expect("python baseline should run (requires python3)");
                // Include subprocess spawn + work in the measurement.
                total += start.elapsed().max(duration);
            }
            total
        })
    });

    group.bench_function(BenchmarkId::new("rust", rows.len()), |b| {
        b.iter(|| {
            let out = rust_apply_rows(black_box(&rows));
            black_box(out);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_python_vs_rust);
criterion_main!(benches);
