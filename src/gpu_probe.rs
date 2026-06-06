//! GPU device probing for the config tool (`jackpotminer-config` only — this is
//! not compiled into the miner, so there is no duplicate FFI).
//!
//! With the `gpu`/`cuda` features the OpenCL/CUDA SDKs are linked in (build.rs
//! links `cuda`/`nvml`; the `ocl` crate links `OpenCL`), and the tool enumerates
//! devices directly — handy on Windows where you may not want to shell out to
//! the miner. Without those features the functions return empty lists and the
//! tool falls back to spawning `jackpotminer --devices-json`.

/// True when at least one GPU SDK is compiled in, so direct probing works.
pub const HAS_SDK: bool = cfg!(feature = "gpu") || cfg!(feature = "cuda");

/// OpenCL devices as `"[i] <platform> / <device>"` (empty without the SDK or on
/// any enumeration error).
#[cfg(feature = "gpu")]
pub fn opencl() -> Vec<String> {
    use ocl::{Device, Platform};
    let mut names = Vec::new();
    let mut idx = 0;
    for platform in Platform::list() {
        let pname = platform.name().unwrap_or_else(|_| "?".into());
        for device in Device::list_all(platform).unwrap_or_default() {
            let dname = device.name().unwrap_or_else(|_| "?".into());
            names.push(format!("[{idx}] {pname} / {dname}"));
            idx += 1;
        }
    }
    names
}

#[cfg(not(feature = "gpu"))]
pub fn opencl() -> Vec<String> {
    Vec::new()
}

/// CUDA devices as `"[i] <name>"` via the driver API (empty without the SDK, no
/// driver, or any error). Uses a tiny self-contained FFI subset.
#[cfg(feature = "cuda")]
pub fn cuda() -> Vec<String> {
    use std::ffi::CStr;
    use std::os::raw::{c_char, c_int, c_uint};

    // Linked via build.rs (`cuda`), matching src/cuda.rs's declarations.
    extern "C" {
        fn cuInit(flags: c_uint) -> c_int;
        fn cuDeviceGetCount(count: *mut c_int) -> c_int;
        fn cuDeviceGet(device: *mut c_int, ordinal: c_int) -> c_int;
        fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: c_int) -> c_int;
    }

    let mut out = Vec::new();
    unsafe {
        if cuInit(0) != 0 {
            return out;
        }
        let mut n: c_int = 0;
        if cuDeviceGetCount(&mut n) != 0 {
            return out;
        }
        for i in 0..n {
            let mut dev: c_int = 0;
            let name = if cuDeviceGet(&mut dev, i) == 0 {
                let mut buf = [0i8; 128];
                if cuDeviceGetName(buf.as_mut_ptr() as *mut c_char, 128, dev) == 0 {
                    CStr::from_ptr(buf.as_ptr() as *const c_char).to_string_lossy().into_owned()
                } else {
                    format!("CUDA device {i}")
                }
            } else {
                format!("CUDA device {i}")
            };
            out.push(format!("[{i}] {name}"));
        }
    }
    out
}

#[cfg(not(feature = "cuda"))]
pub fn cuda() -> Vec<String> {
    Vec::new()
}
