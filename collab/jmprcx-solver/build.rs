// Link against the system CUDA driver (libcuda.so -> libcuda.so.1).
// Falls back to the CUDA toolkit stub if the driver symlink isn't in /usr/lib.
fn main() {
    println!("cargo:rustc-link-search=native=/usr/lib");
    println!("cargo:rustc-link-search=native=/usr/lib64");
    println!("cargo:rustc-link-search=native=/opt/cuda/targets/x86_64-linux/lib/stubs");
    println!("cargo:rustc-link-lib=dylib=cuda");
}
