use pyo3::prelude::*;

#[pyfunction]
/// Auto-generated from Python hotspot `euclidean` at line 1 (100.00%): --function flag
pub fn euclidean(py: Python, p1: Vec<f64>, p2: Vec<f64>) -> PyResult<f64> {
    let _ = py; // reserved for future GIL use
        if p1.len() != p2.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("length mismatch"));
    }

        let mut total: f64 = 0.0;
    for i in 0..p1.len() {
        let mut diff = (p1[i] - p2[i]);
        total += (diff * diff);

    }
    return Ok((total).powf(0.5));

}

#[pymodule]
fn rustify_ml_ext(_py: Python, m: &PyModule) -> PyResult<()> {
m.add_function(wrap_pyfunction!(euclidean, m)?)?;
Ok(())
}
