//! Platform-agnostic GPU tuning.
//!
//! [`GpuTuner`] is the per-card control surface (clocks, power, offsets, power
//! readout). The Linux backend is NVML ([`crate::nvml`]); a Windows NVAPI backend
//! can implement the same trait without touching the policy in [`apply`] or any
//! call site. [`open`] picks the backend for the running platform.
//!
//! All setters are best-effort and report a [`SetOutcome`] (most need elevated
//! privileges); the miner never fails because a knob couldn't be turned.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use log::{debug, info, warn};

/// Tuning configuration from the CLI flags, set once at startup via [`configure`].
#[derive(Clone, Copy, Default)]
pub struct TuneConfig {
    /// Master switch (false ⇒ touch nothing; from `--no-gpu-tune`).
    pub enabled: bool,
    pub power_limit_w: Option<u32>,
    pub gpu_clock_mhz: Option<u32>,
    pub mem_clock_mhz: Option<u32>,
    pub gpu_offset_mhz: Option<i32>,
    pub mem_offset_mhz: Option<i32>,
    /// Sweep the core offset for the fastest stable solve rate.
    pub auto_tune: bool,
    /// Live dashboard controls are unlocked: keep clocks free to boost (not hard
    /// locked) so the user's VF offsets actually take effect.
    pub unlock_controls: bool,
}

static CONFIG: OnceLock<TuneConfig> = OnceLock::new();

/// Install the tuning configuration (call once, before any solver is built).
pub fn configure(cfg: TuneConfig) {
    let _ = CONFIG.set(cfg);
}

fn config() -> TuneConfig {
    CONFIG.get().copied().unwrap_or_default()
}

/// Per-device tuning overrides (from a config file's `[[gpu]]` tables). Any field
/// set here overrides the global [`TuneConfig`] for that device index.
#[derive(Clone, Copy, Default)]
pub struct DeviceTune {
    pub power_limit_w: Option<u32>,
    pub gpu_clock_mhz: Option<u32>,
    pub mem_clock_mhz: Option<u32>,
    pub gpu_offset_mhz: Option<i32>,
    pub mem_offset_mhz: Option<i32>,
}

static DEVICE_CONFIG: OnceLock<Vec<(usize, DeviceTune)>> = OnceLock::new();

/// Install per-device tuning overrides (call once, before any solver is built).
pub fn configure_devices(overrides: Vec<(usize, DeviceTune)>) {
    let _ = DEVICE_CONFIG.set(overrides);
}

/// The effective config for `device_index`: the global config with any matching
/// per-device override folded in.
fn device_config(device_index: usize) -> TuneConfig {
    let mut cfg = config();
    if let Some(d) = DEVICE_CONFIG
        .get()
        .and_then(|v| v.iter().find(|(i, _)| *i == device_index))
        .map(|(_, d)| *d)
    {
        cfg.power_limit_w = d.power_limit_w.or(cfg.power_limit_w);
        cfg.gpu_clock_mhz = d.gpu_clock_mhz.or(cfg.gpu_clock_mhz);
        cfg.mem_clock_mhz = d.mem_clock_mhz.or(cfg.mem_clock_mhz);
        cfg.gpu_offset_mhz = d.gpu_offset_mhz.or(cfg.gpu_offset_mhz);
        cfg.mem_offset_mhz = d.mem_offset_mhz.or(cfg.mem_offset_mhz);
    }
    cfg
}

/// Whether auto-tuning is enabled (and tuning isn't disabled outright).
pub fn auto_tune_enabled() -> bool {
    let c = config();
    c.enabled && c.auto_tune
}

/// Result of a privileged set call, for logging and change tracking.
pub enum SetOutcome {
    Applied,
    Denied,
    Unsupported,
    Failed(String),
}

impl SetOutcome {
    /// Did the change actually take effect?
    pub fn applied(&self) -> bool {
        matches!(self, SetOutcome::Applied)
    }
}

/// Per-GPU control surface. One implementation per platform (NVML on Linux,
/// NVAPI on Windows). Setters clamp to the card's own limits.
pub trait GpuTuner: Send {
    fn name(&self) -> String;
    /// Current board power draw in watts, or `None` if unavailable.
    fn watts(&self) -> Option<f64>;
    /// Current GPU core temperature in °C, or `None` if unavailable.
    fn temperature_c(&self) -> Option<u32>;
    /// Currently enforced power limit in watts, or `None` if unavailable.
    fn current_power_limit_w(&self) -> Option<u32>;
    /// Current SM (core) clock in MHz, or `None` if unavailable.
    fn core_clock_mhz(&self) -> Option<u32>;
    /// Current memory clock in MHz, or `None` if unavailable.
    fn mem_clock_mhz(&self) -> Option<u32>;
    fn max_core_clock_mhz(&self) -> Option<u32>;
    fn max_mem_clock_mhz(&self) -> Option<u32>;
    fn power_limit_range_w(&self) -> Option<(u32, u32)>;
    fn set_persistence(&self, on: bool) -> SetOutcome;
    fn lock_core_clock_mhz(&self, mhz: u32) -> SetOutcome;
    fn lock_mem_clock_mhz(&self, mhz: u32) -> SetOutcome;
    fn set_power_limit_w(&self, watts: u32) -> SetOutcome;
    fn set_core_offset_mhz(&self, mhz: i32) -> SetOutcome;
    fn set_mem_offset_mhz(&self, mhz: i32) -> SetOutcome;
    /// Restore default clocks / power / offsets.
    fn reset(&self);
}

/// Open a control handle for the GPU at `pci_bus_id` (matches the physical card
/// regardless of CUDA-vs-driver index ordering). `None` if unavailable.
///
/// NVML is the backend on both Linux (`libnvidia-ml`) and Windows (`nvml.dll`);
/// the C API is identical, so the same [`crate::nvml`] code serves both.
pub fn open(pci_bus_id: &str) -> Option<Box<dyn GpuTuner>> {
    #[cfg(any(unix, windows))]
    {
        crate::nvml::open(pci_bus_id)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = pci_bus_id;
        None
    }
}

static WARNED_PRIVS: AtomicBool = AtomicBool::new(false);

fn log_set(out: SetOutcome, name: &str, what: &str, denied: &mut bool) -> bool {
    match out {
        SetOutcome::Applied => {
            info!("{name}: {what} ✓");
            true
        }
        SetOutcome::Denied => {
            *denied = true;
            debug!("{name}: {what} — needs elevated privileges");
            false
        }
        SetOutcome::Unsupported => {
            debug!("{name}: {what} — not supported");
            false
        }
        SetOutcome::Failed(e) => {
            warn!("{name}: {what} failed ({e})");
            false
        }
    }
}

/// Apply the configured tuning policy to one card. Returns true if anything
/// changed (so the caller restores defaults on exit).
///
/// With no flags it locks clocks + power to the card maximum (peak hashrate).
/// Any flag — including `--auto-tune` and `--unlock-controls` — switches to
/// manual mode: only requested knobs are applied, and clocks are left free to
/// boost so a clock offset (from auto-tune or the live dashboard controls) isn't
/// defeated by a hard lock. Auto-tune and unlocked controls still force power to
/// max for headroom.
pub fn apply(t: &dyn GpuTuner, device_index: usize) -> bool {
    let cfg = device_config(device_index);
    if !cfg.enabled {
        return false;
    }
    let name = t.name();
    let mut denied = false;
    let mut changed = false;

    // Keep the driver resident so clocks don't drop between solves (harmless to
    // leave on, so not counted as a change to undo).
    log_set(t.set_persistence(true), &name, "persistence mode", &mut denied);

    let manual = cfg.auto_tune
        || cfg.unlock_controls
        || cfg.power_limit_w.is_some()
        || cfg.gpu_clock_mhz.is_some()
        || cfg.mem_clock_mhz.is_some()
        || cfg.gpu_offset_mhz.is_some()
        || cfg.mem_offset_mhz.is_some();

    // Core / memory clock: lock to the requested value, or to max in default mode.
    if let Some(mhz) = cfg.gpu_clock_mhz.or((!manual).then(|| t.max_core_clock_mhz()).flatten()) {
        changed |= log_set(t.lock_core_clock_mhz(mhz), &name, &format!("lock SM clock @ {mhz} MHz"), &mut denied);
    }
    if let Some(mhz) = cfg.mem_clock_mhz.or((!manual).then(|| t.max_mem_clock_mhz()).flatten()) {
        changed |= log_set(t.lock_mem_clock_mhz(mhz), &name, &format!("lock memory clock @ {mhz} MHz"), &mut denied);
    }

    // Power: requested cap, or max in default mode / for auto-tune headroom /
    // when controls are unlocked (so freeing the clocks doesn't cost performance).
    let want_max_power = !manual || cfg.auto_tune || cfg.unlock_controls;
    if let Some(w) = cfg.power_limit_w.or(want_max_power.then(|| t.power_limit_range_w().map(|(_, mx)| mx)).flatten()) {
        changed |= log_set(t.set_power_limit_w(w), &name, &format!("power limit @ {w} W"), &mut denied);
    }

    // Clock offsets. Auto-tune owns both the core and memory offsets (it sweeps
    // them), so skip the static offsets in that case.
    if !cfg.auto_tune {
        if let Some(off) = cfg.gpu_offset_mhz {
            changed |= log_set(t.set_core_offset_mhz(off), &name, &format!("core clock offset {off:+} MHz"), &mut denied);
        }
        if let Some(off) = cfg.mem_offset_mhz {
            changed |= log_set(t.set_mem_offset_mhz(off), &name, &format!("memory clock offset {off:+} MHz"), &mut denied);
        }
    }

    if denied && !WARNED_PRIVS.swap(true, Ordering::Relaxed) {
        warn!("some GPU performance settings need elevated privileges — run the miner as root (Linux) or Administrator (Windows) for full effect (clocks/power stay at defaults otherwise)");
    }
    changed
}
