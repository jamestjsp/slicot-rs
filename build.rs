fn main() {
    // On macOS, link against the Accelerate framework for BLAS/LAPACK
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
