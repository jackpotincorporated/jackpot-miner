//! Build script for the CUDA backend.
//!
//! The `cuda` feature links the CUDA driver API (`cuda`) and NVML (for
//! clock/power control + readout). The backend drives miniZ's embedded fatbin
//! (`src/miniz/equihash192_7.fatbin`) via the driver API, so no nvcc / kernel
//! compilation is needed at build time. (The default OpenCL backend needs no
//! build-script support — `ocl` links `OpenCL` itself, cross-platform.)
//!
//! Linking is target-aware so the `cuda` feature builds on both Linux and
//! Windows:
//!   - Linux: `libcuda.so` + `libnvidia-ml.so` from the system / toolkit dirs.
//!   - Windows: `cuda.lib` + `nvml.lib` from `%CUDA_PATH%\lib\x64`.

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        // Use the *target* OS (correct for cross-compilation too), not the host.
        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        link_cuda_driver(&target_os);
    }
}

/// Link the CUDA driver library plus NVML. The backend loads the embedded miniZ
/// fatbin at runtime, so there is nothing to compile here.
fn link_cuda_driver(target_os: &str) {
    if target_os == "windows" {
        // CUDA Toolkit import libraries (cuda.lib, nvml.lib).
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            println!("cargo:rustc-link-search=native={cuda_path}\\lib\\x64");
        }
        // Driver API import lib is `cuda.lib`; NVML is `nvml.lib` (nvml.dll
        // ships with the NVIDIA driver).
        println!("cargo:rustc-link-lib=dylib=cuda");
        println!("cargo:rustc-link-lib=dylib=nvml");
    } else {
        for dir in ["/usr/lib64", "/usr/lib", "/opt/cuda/lib64"] {
            if Path::new(dir).exists() {
                println!("cargo:rustc-link-search=native={dir}");
            }
        }
        // GNU ld: embed an rpath so libcuda is found at runtime (Linux only —
        // MSVC's linker rejects `-Wl,...`).
        if target_os == "linux" {
            println!("cargo:rustc-link-arg=-Wl,-rpath,/opt/cuda/lib64");
        }
        println!("cargo:rustc-link-lib=dylib=cuda");
        println!("cargo:rustc-link-lib=dylib=nvidia-ml");
    }
}
