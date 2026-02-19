use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
pub fn euclidean(p1: Vec<f64>, p2: Vec<f64>) -> PyResult<f64> {
    if p1.len() != p2.len() {
        return Err(PyValueError::new_err("length mismatch"));
    }
    let sum_sq: f64 = p1
        .iter()
        .zip(p2.iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum();
    Ok(sum_sq.sqrt())
}

#[pyfunction]
pub fn dot_product(a: Vec<f64>, b: Vec<f64>) -> PyResult<f64> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("length mismatch"));
    }
    let mut total = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        total += x * y;
    }
    Ok(total)
}

#[pyfunction]
pub fn normalize_pixels(pixels: Vec<f64>, mean: f64, std: f64) -> PyResult<Vec<f64>> {
    if std == 0.0 {
        return Err(PyValueError::new_err("std must be non-zero"));
    }
    Ok(pixels.into_iter().map(|p| (p - mean) / std).collect())
}

#[pyfunction]
pub fn running_mean(values: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    if window == 0 {
        return Err(PyValueError::new_err("window must be positive"));
    }
    let mut result = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        let start = i + 1 - window.min(i + 1);
        let mut total = 0.0;
        let mut count = 0usize;
        for j in start..=i {
            total += values[j];
            count += 1;
        }
        result.push(total / count as f64);
    }
    Ok(result)
}

#[pyfunction]
pub fn convolve1d(signal: Vec<f64>, kernel: Vec<f64>) -> PyResult<Vec<f64>> {
    let n = signal.len();
    let k = kernel.len();
    if k == 0 {
        return Err(PyValueError::new_err("invalid kernel length"));
    }
    if n < k {
        return Err(PyValueError::new_err("signal shorter than kernel"));
    }
    let mut result = vec![0.0f64; n - k + 1];
    for i in 0..=n - k {
        let mut total = 0.0;
        for j in 0..k {
            total += signal[i + j] * kernel[j];
        }
        result[i] = total;
    }
    Ok(result)
}

#[pyfunction]
pub fn moving_average(signal: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    if window == 0 {
        return Err(PyValueError::new_err("invalid window"));
    }
    let n = signal.len();
    if n < window {
        return Err(PyValueError::new_err("signal shorter than window"));
    }
    let mut result = vec![0.0f64; n - window + 1];
    for i in 0..=n - window {
        let mut total = 0.0;
        for j in 0..window {
            total += signal[i + j];
        }
        result[i] = total / window as f64;
    }
    Ok(result)
}

#[pyfunction]
pub fn diff(signal: Vec<f64>) -> PyResult<Vec<f64>> {
    if signal.len() < 2 {
        return Ok(vec![]);
    }
    let mut result = Vec::with_capacity(signal.len() - 1);
    for i in 0..signal.len() - 1 {
        result.push(signal[i + 1] - signal[i]);
    }
    Ok(result)
}

#[pyfunction]
pub fn cumsum(signal: Vec<f64>) -> PyResult<Vec<f64>> {
    let mut result = Vec::with_capacity(signal.len());
    let mut total = 0.0;
    for v in signal.iter() {
        total += *v;
        result.push(total);
    }
    Ok(result)
}

#[pyfunction]
pub fn standard_scale(data: Vec<f64>, mean: f64, std: f64) -> PyResult<Vec<f64>> {
    if std == 0.0 {
        return Err(PyValueError::new_err("std must be non-zero"));
    }
    Ok(data.into_iter().map(|x| (x - mean) / std).collect())
}

#[pyfunction]
pub fn min_max_scale(data: Vec<f64>, min_val: f64, max_val: f64) -> PyResult<Vec<f64>> {
    if max_val <= min_val {
        return Err(PyValueError::new_err("max must be greater than min"));
    }
    let range = max_val - min_val;
    Ok(data
        .into_iter()
        .map(|x| (x - min_val) / range)
        .collect())
}

#[pyfunction]
pub fn l2_normalize(data: Vec<f64>) -> PyResult<Vec<f64>> {
    let norm_sq: f64 = data.iter().map(|v| v * v).sum();
    if norm_sq == 0.0 {
        return Err(PyValueError::new_err("zero norm"));
    }
    let norm = norm_sq.sqrt();
    Ok(data.into_iter().map(|v| v / norm).collect())
}

#[pyfunction]
pub fn count_pairs(tokens: Vec<i64>) -> PyResult<HashMap<(i64, i64), usize>> {
    let mut counts: HashMap<(i64, i64), usize> = HashMap::new();
    for window in tokens.windows(2) {
        let key = (window[0], window[1]);
        *counts.entry(key).or_insert(0) += 1;
    }
    Ok(counts)
}

#[pyfunction]
pub fn bpe_encode(text: String, _merges: Vec<(String, String)>) -> PyResult<Vec<i64>> {
    Ok(text.bytes().map(|b| b as i64).collect())
}

#[pymodule]
fn rustify_ml_ext(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(euclidean, m)?)?;
    m.add_function(wrap_pyfunction!(dot_product, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_pixels, m)?)?;
    m.add_function(wrap_pyfunction!(running_mean, m)?)?;
    m.add_function(wrap_pyfunction!(convolve1d, m)?)?;
    m.add_function(wrap_pyfunction!(moving_average, m)?)?;
    m.add_function(wrap_pyfunction!(diff, m)?)?;
    m.add_function(wrap_pyfunction!(cumsum, m)?)?;
    m.add_function(wrap_pyfunction!(standard_scale, m)?)?;
    m.add_function(wrap_pyfunction!(min_max_scale, m)?)?;
    m.add_function(wrap_pyfunction!(l2_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(count_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(bpe_encode, m)?)?;
    Ok(())
}
