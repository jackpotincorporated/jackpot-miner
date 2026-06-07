//! AMD GPU telemetry for [`crate::gpu_tune::GpuTuner`] via the Linux amdgpu
//! sysfs interface (`/sys/class/drm/cardN/device/...`).
//!
//! Telemetry only: board power, edge temperature, and core/memory clocks come
//! from the world-readable hwmon + DPM nodes, so it works unprivileged. The
//! control nodes (`pp_od_clk_voltage`, power cap, performance level) are
//! root-write-only and carry GPU-hang risk, so every setter returns
//! [`SetOutcome::Unsupported`] — this backend never writes. The handle is matched
//! to the physical card by PCI bus id (e.g. from OpenCL's `cl_khr_pci_bus_info`),
//! so it lines up with whichever device the solver actually opened.

use std::path::{Path, PathBuf};

use crate::gpu_tune::{GpuTuner, SetOutcome};

/// One amdgpu card's sysfs telemetry handle.
pub struct AmdTuner {
    /// Canonical `/sys/.../<PCI BDF>` device directory (holds `pp_dpm_*`).
    device_dir: PathBuf,
    /// `device_dir/hwmon/hwmonM` (the index M is not stable — resolved by glob).
    hwmon_dir: PathBuf,
    name: String,
}

// Only `PathBuf`/`String` — `AmdTuner` is `Send` automatically; no `unsafe impl`.

/// Open a telemetry handle for the amdgpu card at `pci_bus_id` (e.g.
/// "0000:03:00.0"). Returns `None` off Linux, or when no amdgpu card with
/// telemetry nodes matches the bus id.
pub fn open(pci_bus_id: &str) -> Option<Box<dyn GpuTuner>> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = pci_bus_id;
        None
    }
    #[cfg(target_os = "linux")]
    {
        let want = bdf_tail(pci_bus_id)?;
        for entry in std::fs::read_dir("/sys/class/drm").ok()?.flatten() {
            let fname = entry.file_name();
            let fname = fname.to_string_lossy();
            // Match the GPU nodes ("card0", "card1", …), not the per-connector
            // dirs ("card1-DP-1") or render nodes.
            if !fname.starts_with("card") || fname.contains('-') {
                continue;
            }
            // `cardN/device` symlinks to the PCI device dir `…/<BDF>`.
            let Ok(device_dir) = std::fs::canonicalize(entry.path().join("device")) else {
                continue;
            };
            let matches = device_dir
                .file_name()
                .and_then(|s| s.to_str())
                .and_then(bdf_tail)
                .map(|t| t == want)
                .unwrap_or(false);
            if !matches {
                continue;
            }
            let Some(hwmon_dir) = find_hwmon(&device_dir) else { continue };
            // Require a telemetry node so we don't attach to a card without
            // sensors (e.g. some virtual/headless devices).
            if !hwmon_dir.join("temp1_input").exists()
                && !hwmon_dir.join("power1_average").exists()
            {
                continue;
            }
            let name = read_name(&device_dir);
            return Some(Box::new(AmdTuner { device_dir, hwmon_dir, name }));
        }
        None
    }
}

/// First `hwmon/hwmon*` subdir of a device dir (the index isn't stable).
fn find_hwmon(device_dir: &Path) -> Option<PathBuf> {
    for e in std::fs::read_dir(device_dir.join("hwmon")).ok()?.flatten() {
        if e.file_name().to_string_lossy().starts_with("hwmon") {
            return Some(e.path());
        }
    }
    None
}

/// A display name for the card; amdgpu sysfs rarely exposes a marketing name, so
/// fall back to a generic label.
fn read_name(device_dir: &Path) -> String {
    read_trim(&device_dir.join("product_name"))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "AMD GPU".to_string())
}

/// Normalize a PCI BDF to its `bus:device.function` tail (drop the domain), so
/// e.g. "0000:03:00.0" and "03:00.0" compare equal.
fn bdf_tail(bdf: &str) -> Option<String> {
    let bdf = bdf.trim().to_ascii_lowercase();
    match bdf.split(':').collect::<Vec<_>>().as_slice() {
        [_domain, bus, devfunc] => Some(format!("{bus}:{devfunc}")),
        [bus, devfunc] => Some(format!("{bus}:{devfunc}")),
        _ => None,
    }
}

fn read_trim(p: &Path) -> Option<String> {
    std::fs::read_to_string(p).ok().map(|s| s.trim().to_string())
}

fn read_u64(p: &Path) -> Option<u64> {
    read_trim(p)?.parse().ok()
}

/// Parse the MHz value out of a `pp_dpm_*` line like `"2: 2700Mhz *"`.
fn parse_mhz(line: &str) -> Option<u32> {
    let after_colon = line.split(':').nth(1)?;
    let token = after_colon.trim().split_whitespace().next()?;
    let digits: String = token.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

impl AmdTuner {
    /// MHz of the currently active DPM level (the line marked `*`) in `file`.
    fn dpm_active_mhz(&self, file: &str) -> Option<u32> {
        let s = read_trim(&self.device_dir.join(file))?;
        s.lines().find(|l| l.contains('*')).and_then(parse_mhz)
    }
    /// Highest DPM level (MHz) listed in `file`.
    fn dpm_max_mhz(&self, file: &str) -> Option<u32> {
        let s = read_trim(&self.device_dir.join(file))?;
        s.lines().filter_map(parse_mhz).max()
    }
}

impl GpuTuner for AmdTuner {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn watts(&self) -> Option<f64> {
        // power1_average is µW; fall back to the instantaneous power1_input.
        let uw = read_u64(&self.hwmon_dir.join("power1_average"))
            .or_else(|| read_u64(&self.hwmon_dir.join("power1_input")))?;
        Some(uw as f64 / 1_000_000.0)
    }

    fn temperature_c(&self) -> Option<u32> {
        // temp1 = edge (m°C).
        let mc = read_u64(&self.hwmon_dir.join("temp1_input"))?;
        Some(((mc + 500) / 1000) as u32)
    }

    fn current_power_limit_w(&self) -> Option<u32> {
        // Absent on Navi 44 (RX 9060 XT); best-effort for cards that expose it.
        read_u64(&self.hwmon_dir.join("power1_cap")).map(|uw| (uw / 1_000_000) as u32)
    }

    fn core_clock_mhz(&self) -> Option<u32> {
        // freq1_input (Hz) is the live sclk; the DPM active level is the fallback.
        if let Some(hz) = read_u64(&self.hwmon_dir.join("freq1_input")) {
            if hz > 0 {
                return Some((hz / 1_000_000) as u32);
            }
        }
        self.dpm_active_mhz("pp_dpm_sclk")
    }

    fn mem_clock_mhz(&self) -> Option<u32> {
        self.dpm_active_mhz("pp_dpm_mclk")
    }

    fn max_core_clock_mhz(&self) -> Option<u32> {
        self.dpm_max_mhz("pp_dpm_sclk")
    }

    fn max_mem_clock_mhz(&self) -> Option<u32> {
        self.dpm_max_mhz("pp_dpm_mclk")
    }

    fn power_limit_range_w(&self) -> Option<(u32, u32)> {
        let mn = read_u64(&self.hwmon_dir.join("power1_cap_min"))?;
        let mx = read_u64(&self.hwmon_dir.join("power1_cap_max"))?;
        Some(((mn / 1_000_000) as u32, (mx / 1_000_000) as u32))
    }

    // Telemetry-only backend: never writes the root-only control nodes.
    fn set_persistence(&self, _on: bool) -> SetOutcome {
        SetOutcome::Unsupported
    }
    fn lock_core_clock_mhz(&self, _mhz: u32) -> SetOutcome {
        SetOutcome::Unsupported
    }
    fn lock_mem_clock_mhz(&self, _mhz: u32) -> SetOutcome {
        SetOutcome::Unsupported
    }
    fn set_power_limit_w(&self, _watts: u32) -> SetOutcome {
        SetOutcome::Unsupported
    }
    fn set_core_offset_mhz(&self, _mhz: i32) -> SetOutcome {
        SetOutcome::Unsupported
    }
    fn set_mem_offset_mhz(&self, _mhz: i32) -> SetOutcome {
        SetOutcome::Unsupported
    }
    fn reset(&self) {}
}
