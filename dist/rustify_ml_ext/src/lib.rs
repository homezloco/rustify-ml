use pyo3::prelude::*;

#[pyfunction]
/// Auto-generated from Python hotspot `normalize_pixels` at line 1 (100.00%): --function flag
pub fn normalize_pixels(py: Python, pixels: Vec<f64>, mean: Vec<f64>, std: Vec<f64>) -> PyResult<Vec<f64>> {
    let _ = py; // reserved for future GIL use
        if pixels.len() != mean.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("length mismatch"));
    }
    if pixels.len() != std.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("length mismatch"));
    }

        // docstring omitted
    let mut result = vec![0.0f64; pixels.len()];
    for i in 0..pixels.len() {
        result[i] = ((pixels[i] - mean) / std);

    }
    return Ok(result);

}

#[pymodule]
fn rustify_ml_ext(_py: Python, m: &PyModule) -> PyResult<()> {
m.add_function(wrap_pyfunction!(normalize_pixels, m)?)?;
Ok(())
}
