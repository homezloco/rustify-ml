/// rustify-ml library crate.
///
/// Exposes the core pipeline modules as a public API so that
/// integration tests in tests/ can import them via `rustify_ml::`.
///
/// The binary entry point (src/main.rs) uses these same modules.
pub mod analyzer;
pub mod builder;
pub mod generator;
pub mod input;
pub mod profiler;
pub mod utils;
