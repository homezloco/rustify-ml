use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustify_stdlib::{bpe_encode, convolve1d, dot_product, euclidean, moving_average};

fn make_vec(len: usize, start: f64) -> Vec<f64> {
    (0..len).map(|i| start + i as f64).collect()
}

fn bench_speed(c: &mut Criterion) {
    let n = 1_000;
    let window = 10;
    let a = make_vec(n, 0.0);
    let b = make_vec(n, 1.0);
    let kernel = vec![1.0, 0.0, -1.0, 0.5, -0.5];

    c.bench_function("euclidean n=1000", |bch| {
        bch.iter(|| euclidean(black_box(a.clone()), black_box(b.clone())))
    });

    c.bench_function("dot_product n=1000", |bch| {
        bch.iter(|| dot_product(black_box(a.clone()), black_box(b.clone())))
    });

    c.bench_function("moving_average n=1000 w=10", |bch| {
        bch.iter(|| moving_average(black_box(a.clone()), black_box(window)))
    });

    c.bench_function("convolve1d n=1000 k=5", |bch| {
        bch.iter(|| convolve1d(black_box(a.clone()), black_box(kernel.clone())))
    });

    c.bench_function("bpe_encode len=100", |bch| {
        bch.iter(|| bpe_encode(black_box("hello world".repeat(10)), black_box(vec![])))
    });
}

criterion_group!(benches, bench_speed);
criterion_main!(benches);
