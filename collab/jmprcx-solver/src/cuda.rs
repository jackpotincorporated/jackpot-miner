//! Minimal raw FFI bindings to the CUDA Driver API (libcuda) — only what we need
//! to load Equihash 192,7 fatbin and drive its kernels.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::{c_char, c_int, c_uint, c_void, CStr};

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUdeviceptr = u64;

pub const CUDA_SUCCESS: CUresult = 0;

// CUfunction_attribute values (cuda.h)
pub const CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 0;
pub const CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: c_int = 1;
pub const CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES: c_int = 2;
pub const CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: c_int = 3;
pub const CU_FUNC_ATTRIBUTE_NUM_REGS: c_int = 4;
pub const CU_FUNC_ATTRIBUTE_PTX_VERSION: c_int = 5;
pub const CU_FUNC_ATTRIBUTE_BINARY_VERSION: c_int = 6;
pub const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;

// cuLaunchKernel `extra` directives
pub const CU_LAUNCH_PARAM_END: usize = 0x00;
pub const CU_LAUNCH_PARAM_BUFFER_POINTER: usize = 0x01;
pub const CU_LAUNCH_PARAM_BUFFER_SIZE: usize = 0x02;

extern "C" {
    pub fn cuInit(flags: c_uint) -> CUresult;
    pub fn cuDriverGetVersion(version: *mut c_int) -> CUresult;
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    pub fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
    pub fn cuCtxCreate_v2(pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult;
    pub fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;
    pub fn cuCtxSynchronize() -> CUresult;

    // Module / kernel loading + introspection
    pub fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    pub fn cuModuleUnload(module: CUmodule) -> CUresult;
    pub fn cuModuleGetFunction(func: *mut CUfunction, module: CUmodule, name: *const c_char) -> CUresult;
    pub fn cuModuleGetFunctionCount(count: *mut c_uint, module: CUmodule) -> CUresult;
    pub fn cuModuleEnumerateFunctions(functions: *mut CUfunction, num: c_uint, module: CUmodule) -> CUresult;
    pub fn cuFuncGetName(name: *mut *const c_char, func: CUfunction) -> CUresult;
    pub fn cuFuncGetAttribute(pi: *mut c_int, attrib: c_int, func: CUfunction) -> CUresult;
    pub fn cuFuncSetAttribute(func: CUfunction, attrib: c_int, value: c_int) -> CUresult;

    // Memory + launch
    pub fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    pub fn cuMemsetD8_v2(dptr: CUdeviceptr, uc: u8, n: usize) -> CUresult;
    pub fn cuMemsetD32_v2(dptr: CUdeviceptr, ui: u32, n: usize) -> CUresult;
    pub fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    pub fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, n: usize) -> CUresult;
    pub fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;
    pub fn cuLaunchKernel(
        f: CUfunction,
        gx: c_uint, gy: c_uint, gz: c_uint,
        bx: c_uint, by: c_uint, bz: c_uint,
        shared_mem_bytes: c_uint,
        stream: CUstream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;

    pub fn cuGetErrorName(error: CUresult, s: *mut *const c_char) -> CUresult;
    pub fn cuGetErrorString(error: CUresult, s: *mut *const c_char) -> CUresult;
}

/// Human-readable "NAME: description" for a CUresult.
pub fn err_str(code: CUresult) -> String {
    unsafe {
        let mut name: *const c_char = std::ptr::null();
        let mut desc: *const c_char = std::ptr::null();
        cuGetErrorName(code, &mut name);
        cuGetErrorString(code, &mut desc);
        let n = if name.is_null() { "?".into() } else { CStr::from_ptr(name).to_string_lossy().into_owned() };
        let d = if desc.is_null() { "".into() } else { CStr::from_ptr(desc).to_string_lossy().into_owned() };
        format!("{n} ({code}): {d}")
    }
}

/// Turn a CUresult into a Result, with context.
pub fn check(code: CUresult, what: &str) -> Result<(), String> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{what} failed: {}", err_str(code)))
    }
}
