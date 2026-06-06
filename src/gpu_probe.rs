//! GPU device probing for the config tool (`jackpotminer-config` only — this is
//! not compiled into the miner, so there is no duplicate FFI).
//!
//! With the `gpu`/`cuda` features the tool enumerates devices directly — handy
//! on Windows where you may not want to shell out to the miner. OpenCL goes
//! through the `ocl` crate; CUDA is `dlopen`'d at runtime (so this binary, like
//! the miner, has no build- or load-time dependency on libcuda). Without those
//! features the functions return empty lists and the tool falls back to spawning
//! `jackpotminer --devices-json`.

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
/// driver, or any error). The CUDA driver is `dlopen`'d at runtime — a tiny
/// self-contained subset of the FFI in `src/cuda.rs` — so this binary needs no
/// link- or load-time libcuda.
#[cfg(feature = "cuda")]
pub fn cuda() -> Vec<String> {
    use std::ffi::CStr;
    use std::os::raw::{c_char, c_int, c_uint};

    type CuInit = unsafe extern "C" fn(c_uint) -> c_int;
    type CuCount = unsafe extern "C" fn(*mut c_int) -> c_int;
    type CuGet = unsafe extern "C" fn(*mut c_int, c_int) -> c_int;
    type CuName = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int;

    let mut out = Vec::new();
    unsafe {
        // libcuda.so.1 ships with the NVIDIA driver; absent on AMD-only hosts.
        let lib = match ["libcuda.so.1", "libcuda.so", "nvcuda.dll"]
            .iter()
            .find_map(|n| libloading::Library::new(n).ok())
        {
            Some(l) => l,
            None => return out,
        };
        let sym = |name: &[u8]| -> Option<*mut std::ffi::c_void> {
            lib.get::<*mut std::ffi::c_void>(name).ok().map(|s| *s)
        };
        let (Some(init), Some(count), Some(get), Some(getname)) = (
            sym(b"cuInit\0"),
            sym(b"cuDeviceGetCount\0"),
            sym(b"cuDeviceGet\0"),
            sym(b"cuDeviceGetName\0"),
        ) else {
            return out;
        };
        let cu_init: CuInit = std::mem::transmute(init);
        let cu_count: CuCount = std::mem::transmute(count);
        let cu_get: CuGet = std::mem::transmute(get);
        let cu_name: CuName = std::mem::transmute(getname);

        if cu_init(0) != 0 {
            return out;
        }
        let mut n: c_int = 0;
        if cu_count(&mut n) != 0 {
            return out;
        }
        for i in 0..n {
            let mut dev: c_int = 0;
            let name = if cu_get(&mut dev, i) == 0 {
                let mut buf = [0i8; 128];
                if cu_name(buf.as_mut_ptr() as *mut c_char, 128, dev) == 0 {
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
