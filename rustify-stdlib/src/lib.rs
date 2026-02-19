#![allow(unsafe_op_in_unsafe_fn)]
use pyo3::prelude::Bound;
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
pub fn bpe_encode(text: String, _merges: Vec<(String, String)>) -> PyResult<Vec<i64>> {
    Ok(text.bytes().map(|b| b as i64).collect())
}

#[pymodule]
fn rustify_stdlib(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(euclidean, m)?)?;
    m.add_function(wrap_pyfunction!(dot_product, m)?)?;
    m.add_function(wrap_pyfunction!(moving_average, m)?)?;
    m.add_function(wrap_pyfunction!(convolve1d, m)?)?;
    m.add_function(wrap_pyfunction!(bpe_encode, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean() {
        let v = euclidean(vec![0.0, 3.0, 4.0], vec![0.0, 0.0, 0.0]).unwrap();
        assert!((v - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_dot_product() {
        let v = dot_product(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]).unwrap();
        assert!((v - 32.0).abs() < 1e-9);
    }

    #[test]
    fn test_moving_average_valid() {
        let v = moving_average(vec![1.0, 2.0, 3.0, 4.0, 5.0], 3).unwrap();
        assert_eq!(v, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_convolve1d() {
        let v = convolve1d(vec![1.0, 2.0, 3.0, 4.0], vec![1.0, 0.0, -1.0]).unwrap();
        assert_eq!(v, vec![-2.0, -2.0]);
    }

    #[test]
    fn test_bpe_encode() {
        let v = bpe_encode("ab".to_string(), vec![]).unwrap();
        assert_eq!(v, vec![97, 98]);
    }
}
