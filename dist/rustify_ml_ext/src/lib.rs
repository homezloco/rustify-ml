#![allow(unsafe_op_in_unsafe_fn)]
use pyo3::prelude::*;
use pyo3::Bound;

#[pyfunction]
pub fn euclidean(_py: Python, p1: Vec<f64>, p2: Vec<f64>) -> PyResult<f64> {
    let mut total: f64 = 0.0;
    for i in 0..p1.len() {
        let diff = p1[i] - p2[i];
        total += diff * diff;
    }
    Ok(total.sqrt())
}

#[pyfunction]
pub fn dot_product(_py: Python, a: Vec<f64>, b: Vec<f64>) -> PyResult<f64> {
    let mut total: f64 = 0.0;
    for i in 0..a.len() {
        total += a[i] * b[i];
    }
    Ok(total)
}

#[pyfunction]
pub fn normalize_pixels(_py: Python, pixels: Vec<f64>, mean: f64, std: f64) -> PyResult<Vec<f64>> {
    let mut result: Vec<f64> = vec![0.0f64; pixels.len()];
    for i in 0..pixels.len() {
        result[i] = (pixels[i] - mean) / std;
    }
    Ok(result)
}

#[pyfunction]
pub fn standard_scale(_py: Python, data: Vec<f64>, mean: f64, std: f64) -> PyResult<Vec<f64>> {
    let mut result: Vec<f64> = vec![0.0f64; data.len()];
    for i in 0..data.len() {
        result[i] = (data[i] - mean) / std;
    }
    Ok(result)
}

#[pyfunction]
pub fn min_max_scale(_py: Python, data: Vec<f64>, min_val: f64, max_val: f64) -> PyResult<Vec<f64>> {
    let range_val = max_val - min_val;
    let mut result: Vec<f64> = vec![0.0f64; data.len()];
    for i in 0..data.len() {
        result[i] = (data[i] - min_val) / range_val;
    }
    Ok(result)
}

#[pyfunction]
pub fn l2_normalize(_py: Python, data: Vec<f64>) -> PyResult<Vec<f64>> {
    let mut total: f64 = 0.0;
    for i in 0..data.len() {
        total += data[i] * data[i];
    }
    let norm = total.sqrt();
    let mut result: Vec<f64> = vec![0.0f64; data.len()];
    for i in 0..data.len() {
        result[i] = data[i] / norm;
    }
    Ok(result)
}

#[pyfunction]
pub fn running_mean(_py: Python, values: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    let mut result: Vec<f64> = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        let start = if i + 1 >= window { i + 1 - window } else { 0 };
        let mut total: f64 = 0.0;
        let mut count: usize = 0;
        for j in start..=i {
            total += values[j];
            count += 1;
        }
        result.push(if count > 0 { total / count as f64 } else { 0.0 });
    }
    Ok(result)
}

#[pyfunction]
pub fn convolve1d(_py: Python, signal: Vec<f64>, kernel: Vec<f64>) -> PyResult<Vec<f64>> {
    let n = signal.len();
    let k = kernel.len();
    let out_len = if n >= k { n - k + 1 } else { 0 };
    let mut result: Vec<f64> = vec![0.0f64; out_len];
    for i in 0..out_len {
        let mut total: f64 = 0.0;
        for j in 0..k {
            total += signal[i + j] * kernel[j];
        }
        result[i] = total;
    }
    Ok(result)
}

#[pyfunction]
pub fn moving_average(_py: Python, signal: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    let n = signal.len();
    let out_len = if n >= window { n - window + 1 } else { 0 };
    let mut result: Vec<f64> = vec![0.0f64; out_len];
    for i in 0..out_len {
        let mut total: f64 = 0.0;
        for j in 0..window {
            total += signal[i + j];
        }
        result[i] = total / window as f64;
    }
    Ok(result)
}

#[pyfunction]
pub fn diff(_py: Python, signal: Vec<f64>) -> PyResult<Vec<f64>> {
    let n = signal.len();
    let mut result: Vec<f64> = vec![0.0f64; if n > 0 { n - 1 } else { 0 }];
    for i in 0..result.len() {
        result[i] = signal[i + 1] - signal[i];
    }
    Ok(result)
}

#[pyfunction]
pub fn cumsum(_py: Python, signal: Vec<f64>) -> PyResult<Vec<f64>> {
    let n = signal.len();
    let mut result: Vec<f64> = vec![0.0f64; n];
    let mut total: f64 = 0.0;
    for i in 0..n {
        total += signal[i];
        result[i] = total;
    }
    Ok(result)
}

#[pyfunction]
pub fn bpe_encode(_py: Python, text: String, merges: Vec<(i64, i64)>) -> PyResult<Vec<i64>> {
    let mut tokens: Vec<i64> = text.into_bytes().into_iter().map(|b| b as i64).collect();
    let mut merge_rank: std::collections::HashMap<(i64, i64), usize> = std::collections::HashMap::new();
    for (rank, pair) in merges.into_iter().enumerate() {
        merge_rank.insert((pair.0, pair.1), rank);
    }
    let mut changed = true;
    while changed {
        changed = false;
        let mut i: usize = 0;
        while i + 1 < tokens.len() {
            let pair = (tokens[i], tokens[i + 1]);
            if let Some(rank) = merge_rank.get(&pair) {
                let new_id = 256 + (*rank as i64);
                tokens[i] = new_id;
                tokens.remove(i + 1);
                changed = true;
            } else {
                i += 1;
            }
        }
    }
    Ok(tokens)
}

#[pyfunction]
pub fn count_pairs(_py: Python, tokens: Vec<i64>) -> PyResult<std::collections::HashMap<(i64, i64), i64>> {
    let mut counts: std::collections::HashMap<(i64, i64), i64> = std::collections::HashMap::new();
    for i in 0..tokens.len().saturating_sub(1) {
        let a = tokens[i];
        let b = tokens[i + 1];
        let entry = counts.entry((a, b)).or_insert(0);
        *entry += 1;
    }
    Ok(counts)
}

#[pymodule]
fn rustify_ml_ext(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(euclidean, m)?)?;
    m.add_function(wrap_pyfunction!(dot_product, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_pixels, m)?)?;
    m.add_function(wrap_pyfunction!(standard_scale, m)?)?;
    m.add_function(wrap_pyfunction!(min_max_scale, m)?)?;
    m.add_function(wrap_pyfunction!(l2_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(running_mean, m)?)?;
    m.add_function(wrap_pyfunction!(convolve1d, m)?)?;
    m.add_function(wrap_pyfunction!(moving_average, m)?)?;
    m.add_function(wrap_pyfunction!(diff, m)?)?;
    m.add_function(wrap_pyfunction!(cumsum, m)?)?;
    m.add_function(wrap_pyfunction!(bpe_encode, m)?)?;
    m.add_function(wrap_pyfunction!(count_pairs, m)?)?;
    Ok(())
}
