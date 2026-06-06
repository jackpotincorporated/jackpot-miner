//! NVML backend for [`crate::gpu_tune::GpuTuner`] — the GPU control surface
//! (clocks, power, V/F offsets, readout). Cross-platform: `libnvidia-ml.so` on
//! Linux, `nvml.dll` on Windows; the C API is identical, so this one module
//! serves both (see `build.rs` for the per-target link name).
//!
//! The CUDA Driver API can't control clocks/power; that lives in NVML. Each
//! [`NvmlTuner`] wraps one device handle (resolved by PCI bus id, so it matches
//! the physical card regardless of CUDA-vs-NVML index ordering) and holds an NVML
//! init refcount for its lifetime. All setters are privileged: without elevated
//! rights (root / Administrator) they return `NO_PERMISSION` (and some are
//! `NOT_SUPPORTED`, e.g. persistence mode on laptop GPUs) — surfaced as
//! [`SetOutcome`] and only logged.

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::ptr;

use crate::gpu_tune::{GpuTuner, SetOutcome};

type nvmlReturn_t = c_int;
type nvmlDevice_t = *mut c_void;

const NVML_SUCCESS: nvmlReturn_t = 0;
const NVML_ERROR_NOT_SUPPORTED: nvmlReturn_t = 3;
const NVML_ERROR_NO_PERMISSION: nvmlReturn_t = 4;

const NVML_FEATURE_ENABLED: c_uint = 1;
const NVML_FEATURE_DISABLED: c_uint = 0;
// nvmlClockType_t
const NVML_CLOCK_SM: c_int = 1;
const NVML_CLOCK_MEM: c_int = 2;
// nvmlTemperatureSensors_t
const NVML_TEMPERATURE_GPU: c_int = 0;

// NVML, loaded at runtime via dlopen (see `crate::dylib`) rather than linked at
// build time — it ships with the NVIDIA driver (`libnvidia-ml.so.1`;
// `nvml.dll` on Windows) and is absent on driver-less / AMD-only hosts.
// `nvml_lib()` is `None` when it can't be opened; `open()` checks it first and
// returns `None` (no tuning) so the rest of the program is unaffected.
crate::dylib::dynamic_library! {
    lib_struct: NvmlLib,
    loader: nvml_lib,
    names: ["libnvidia-ml.so.1", "libnvidia-ml.so", "nvml.dll"],
    fn nvmlInit_v2() -> nvmlReturn_t;
    fn nvmlShutdown() -> nvmlReturn_t;
    fn nvmlDeviceGetName(device: nvmlDevice_t, name: *mut c_char, length: c_uint) -> nvmlReturn_t;
    fn nvmlDeviceGetMaxClockInfo(device: nvmlDevice_t, clock_type: c_int, clock: *mut c_uint) -> nvmlReturn_t;
    fn nvmlDeviceGetClockInfo(device: nvmlDevice_t, clock_type: c_int, clock: *mut c_uint) -> nvmlReturn_t;
    fn nvmlDeviceSetPersistenceMode(device: nvmlDevice_t, mode: c_uint) -> nvmlReturn_t;
    fn nvmlDeviceSetGpuLockedClocks(device: nvmlDevice_t, min_mhz: c_uint, max_mhz: c_uint) -> nvmlReturn_t;
    fn nvmlDeviceSetMemoryLockedClocks(device: nvmlDevice_t, min_mhz: c_uint, max_mhz: c_uint) -> nvmlReturn_t;
    fn nvmlDeviceGetPowerManagementLimitConstraints(device: nvmlDevice_t, min_mw: *mut c_uint, max_mw: *mut c_uint) -> nvmlReturn_t;
    fn nvmlDeviceGetPowerManagementDefaultLimit(device: nvmlDevice_t, default_mw: *mut c_uint) -> nvmlReturn_t;
    fn nvmlDeviceSetPowerManagementLimit(device: nvmlDevice_t, limit_mw: c_uint) -> nvmlReturn_t;
    fn nvmlDeviceGetPowerManagementLimit(device: nvmlDevice_t, limit_mw: *mut c_uint) -> nvmlReturn_t;
    fn nvmlDeviceResetGpuLockedClocks(device: nvmlDevice_t) -> nvmlReturn_t;
    fn nvmlDeviceResetMemoryLockedClocks(device: nvmlDevice_t) -> nvmlReturn_t;
    fn nvmlDeviceGetHandleByPciBusId_v2(pci_bus_id: *const c_char, device: *mut nvmlDevice_t) -> nvmlReturn_t;
    fn nvmlDeviceGetPowerUsage(device: nvmlDevice_t, milliwatts: *mut c_uint) -> nvmlReturn_t;
    fn nvmlDeviceGetTemperature(device: nvmlDevice_t, sensor: c_int, temp: *mut c_uint) -> nvmlReturn_t;
    fn nvmlDeviceSetGpcClkVfOffset(device: nvmlDevice_t, offset: c_int) -> nvmlReturn_t;
    fn nvmlDeviceSetMemClkVfOffset(device: nvmlDevice_t, offset: c_int) -> nvmlReturn_t;
    fn nvmlDeviceGetGpcClkMinMaxVfOffset(device: nvmlDevice_t, min_off: *mut c_int, max_off: *mut c_int) -> nvmlReturn_t;
    fn nvmlDeviceGetMemClkMinMaxVfOffset(device: nvmlDevice_t, min_off: *mut c_int, max_off: *mut c_int) -> nvmlReturn_t;
    fn nvmlErrorString(result: nvmlReturn_t) -> *const c_char;
}

/// One NVML device handle. Holds an NVML init refcount until dropped.
pub struct NvmlTuner {
    dev: nvmlDevice_t,
}

// NVML device handles are process-global and queries are thread-safe, so a tuner
// can travel to the worker thread that owns its solver.
unsafe impl Send for NvmlTuner {}

/// Open an NVML control handle for the GPU at `pci_bus_id` (e.g. "0000:01:00.0").
pub fn open(pci_bus_id: &str) -> Option<Box<dyn GpuTuner>> {
    nvml_lib()?; // NVML not installed → no tuning
    let cstr = CString::new(pci_bus_id).ok()?;
    unsafe {
        if nvmlInit_v2() != NVML_SUCCESS {
            return None;
        }
        let mut dev: nvmlDevice_t = ptr::null_mut();
        if nvmlDeviceGetHandleByPciBusId_v2(cstr.as_ptr(), &mut dev) != NVML_SUCCESS {
            nvmlShutdown();
            return None;
        }
        Some(Box::new(NvmlTuner { dev }))
    }
}

/// Map an NVML return code to a [`SetOutcome`].
fn outcome(rc: nvmlReturn_t) -> SetOutcome {
    match rc {
        NVML_SUCCESS => SetOutcome::Applied,
        NVML_ERROR_NO_PERMISSION => SetOutcome::Denied,
        NVML_ERROR_NOT_SUPPORTED => SetOutcome::Unsupported,
        other => SetOutcome::Failed(errstr(other)),
    }
}

impl NvmlTuner {
    fn max_clock(&self, kind: c_int) -> Option<u32> {
        unsafe {
            let mut c: c_uint = 0;
            if nvmlDeviceGetMaxClockInfo(self.dev, kind, &mut c) == NVML_SUCCESS && c > 0 {
                Some(c)
            } else {
                None
            }
        }
    }

    fn current_clock(&self, kind: c_int) -> Option<u32> {
        unsafe {
            let mut c: c_uint = 0;
            if nvmlDeviceGetClockInfo(self.dev, kind, &mut c) == NVML_SUCCESS && c > 0 {
                Some(c)
            } else {
                None
            }
        }
    }
}

impl GpuTuner for NvmlTuner {
    fn name(&self) -> String {
        unsafe {
            let mut buf = [0 as c_char; 96];
            if nvmlDeviceGetName(self.dev, buf.as_mut_ptr(), buf.len() as c_uint) == NVML_SUCCESS {
                CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned()
            } else {
                "GPU".to_string()
            }
        }
    }

    fn watts(&self) -> Option<f64> {
        unsafe {
            let mut mw: c_uint = 0;
            if nvmlDeviceGetPowerUsage(self.dev, &mut mw) == NVML_SUCCESS {
                Some(mw as f64 / 1000.0)
            } else {
                None
            }
        }
    }

    fn temperature_c(&self) -> Option<u32> {
        unsafe {
            let mut t: c_uint = 0;
            if nvmlDeviceGetTemperature(self.dev, NVML_TEMPERATURE_GPU, &mut t) == NVML_SUCCESS {
                Some(t)
            } else {
                None
            }
        }
    }

    fn current_power_limit_w(&self) -> Option<u32> {
        unsafe {
            let mut mw: c_uint = 0;
            if nvmlDeviceGetPowerManagementLimit(self.dev, &mut mw) == NVML_SUCCESS && mw > 0 {
                Some(mw / 1000)
            } else {
                None
            }
        }
    }

    fn core_clock_mhz(&self) -> Option<u32> {
        self.current_clock(NVML_CLOCK_SM)
    }

    fn mem_clock_mhz(&self) -> Option<u32> {
        self.current_clock(NVML_CLOCK_MEM)
    }

    fn max_core_clock_mhz(&self) -> Option<u32> {
        self.max_clock(NVML_CLOCK_SM)
    }

    fn max_mem_clock_mhz(&self) -> Option<u32> {
        self.max_clock(NVML_CLOCK_MEM)
    }

    fn power_limit_range_w(&self) -> Option<(u32, u32)> {
        unsafe {
            let (mut lo, mut hi): (c_uint, c_uint) = (0, 0);
            if nvmlDeviceGetPowerManagementLimitConstraints(self.dev, &mut lo, &mut hi) == NVML_SUCCESS && hi > 0 {
                Some((lo / 1000, hi / 1000))
            } else {
                None
            }
        }
    }

    fn set_persistence(&self, on: bool) -> SetOutcome {
        let mode = if on { NVML_FEATURE_ENABLED } else { NVML_FEATURE_DISABLED };
        unsafe { outcome(nvmlDeviceSetPersistenceMode(self.dev, mode)) }
    }

    fn lock_core_clock_mhz(&self, mhz: u32) -> SetOutcome {
        let c = self.max_core_clock_mhz().map_or(mhz, |mx| mhz.min(mx));
        unsafe { outcome(nvmlDeviceSetGpuLockedClocks(self.dev, c, c)) }
    }

    fn lock_mem_clock_mhz(&self, mhz: u32) -> SetOutcome {
        let c = self.max_mem_clock_mhz().map_or(mhz, |mx| mhz.min(mx));
        unsafe { outcome(nvmlDeviceSetMemoryLockedClocks(self.dev, c, c)) }
    }

    fn set_power_limit_w(&self, watts: u32) -> SetOutcome {
        let mw = match self.power_limit_range_w() {
            Some((lo, hi)) => (watts * 1000).clamp(lo * 1000, hi * 1000),
            None => watts * 1000,
        };
        unsafe { outcome(nvmlDeviceSetPowerManagementLimit(self.dev, mw)) }
    }

    fn set_core_offset_mhz(&self, mhz: i32) -> SetOutcome {
        unsafe {
            let (mut lo, mut hi): (c_int, c_int) = (0, 0);
            let off = if nvmlDeviceGetGpcClkMinMaxVfOffset(self.dev, &mut lo, &mut hi) == NVML_SUCCESS {
                mhz.clamp(lo, hi)
            } else {
                mhz
            };
            outcome(nvmlDeviceSetGpcClkVfOffset(self.dev, off))
        }
    }

    fn set_mem_offset_mhz(&self, mhz: i32) -> SetOutcome {
        unsafe {
            let (mut lo, mut hi): (c_int, c_int) = (0, 0);
            let off = if nvmlDeviceGetMemClkMinMaxVfOffset(self.dev, &mut lo, &mut hi) == NVML_SUCCESS {
                mhz.clamp(lo, hi)
            } else {
                mhz
            };
            outcome(nvmlDeviceSetMemClkVfOffset(self.dev, off))
        }
    }

    fn reset(&self) {
        unsafe {
            let mut restored = false;
            restored |= nvmlDeviceResetGpuLockedClocks(self.dev) == NVML_SUCCESS;
            restored |= nvmlDeviceResetMemoryLockedClocks(self.dev) == NVML_SUCCESS;
            let mut def: c_uint = 0;
            if nvmlDeviceGetPowerManagementDefaultLimit(self.dev, &mut def) == NVML_SUCCESS && def > 0 {
                restored |= nvmlDeviceSetPowerManagementLimit(self.dev, def) == NVML_SUCCESS;
            }
            restored |= nvmlDeviceSetGpcClkVfOffset(self.dev, 0) == NVML_SUCCESS;
            restored |= nvmlDeviceSetMemClkVfOffset(self.dev, 0) == NVML_SUCCESS;
            if restored {
                log::info!("{}: restored default clocks/power", self.name());
            }
        }
    }
}

impl Drop for NvmlTuner {
    fn drop(&mut self) {
        unsafe { nvmlShutdown() };
    }
}

fn errstr(rc: nvmlReturn_t) -> String {
    unsafe {
        let p = nvmlErrorString(rc);
        if p.is_null() {
            format!("NVML error {rc}")
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}
