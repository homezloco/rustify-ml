use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        sum += d * d;
    }
    sum.sqrt()
}

fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    let mut total = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        total += x * y;
    }
    total
}

fn moving_average(signal: &[f64], window: usize) -> Vec<f64> {
    let n = signal.len();
    if window == 0 || n < window {
        return Vec::new();
    }
    let mut out = vec![0.0f64; n - window + 1];
    for i in 0..=n - window {
        let mut total = 0.0;
        for j in 0..window {
            total += signal[i + j];
        }
        out[i] = total / window as f64;
    }
    out
}

fn make_vec(len: usize, start: f64) -> Vec<f64> {
    (0..len).map(|i| start + i as f64).collect()
}

fn bench_speedups(c: &mut Criterion) {
    let n = 1_000;
    let window = 10;
    let a = make_vec(n, 0.0);
    let b = make_vec(n, 1.0);

    c.bench_function("euclidean n=1000", |bch| {
        bch.iter(|| euclidean(black_box(&a), black_box(&b)))
    });

    c.bench_function("dot_product n=1000", |bch| {
        bch.iter(|| dot_product(black_box(&a), black_box(&b)))
    });

    c.bench_function("moving_average n=1000 w=10", |bch| {
        bch.iter(|| moving_average(black_box(&a), black_box(window)))
    });
}

criterion_group!(benches, bench_speedups);
criterion_main!(benches);
