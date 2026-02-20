use std::env;

fn main() {
    let lib_dir = env::var("PYO3_LIB_DIR")
        .or_else(|_| env::var("PYTHON_LIB_DIR"))
        .unwrap_or_else(|_| "/usr/lib/x86_64-linux-gnu".to_string());
    let lib_name = env::var("PYO3_LIB_NAME").unwrap_or_else(|_| "python3.12".to_string());

    println!("cargo:rustc-link-search=native={lib_dir}");
    println!("cargo:rustc-link-lib={lib_name}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");

    println!("cargo:rerun-if-env-changed=PYO3_LIB_DIR");
    println!("cargo:rerun-if-env-changed=PYTHON_LIB_DIR");
    println!("cargo:rerun-if-env-changed=PYO3_LIB_NAME");
}
