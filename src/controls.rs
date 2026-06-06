//! Live, user-adjustable hardware controls, shared between the TUI (which
//! mutates them on key presses) and the worker threads (which apply them to each
//! card's GPU tuner). Pure atomics — no backend dependency, so it compiles for
//! every feature set even though only the CUDA backend acts on the values.
//!
//! `core_off`/`mem_off` are absolute VF clock offsets in MHz; `power_w` is an
//! absolute power-limit target in watts (0 = leave at the card default). `seq`
//! bumps on every change so workers re-apply lazily, and only after the user
//! actually touches a control.

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// Per-GPU hardware controls plus the TUI's currently-selected device. Workers
/// each drive `device(worker_id)`; the dashboard adjusts `device(selected())`
/// and cycles `selected` with Tab.
pub struct Controls {
    devices: Vec<HwControls>,
    /// Number of CPU-group rows that follow the GPU devices in the selection
    /// order (Tab cycles GPUs first, then groups). Dynamic: changing the CPU
    /// group size from the dashboard changes how many rows there are, so the TUI
    /// keeps this in sync via [`Controls::set_group_rows`]. The control blocks
    /// above are GPU-only; groups carry their own state in [`crate::cpu_groups`].
    groups: AtomicUsize,
    selected: AtomicUsize,
    /// Whether the live control keys may actually change clocks/power. Locked by
    /// default (set via `--unlock-controls`) so stray key presses can't retune.
    unlocked: bool,
}

impl Controls {
    /// One control block per GPU device (each seeded from the CLI tuning flags),
    /// plus `groups` selectable CPU-group rows after them.
    pub fn new(devices: usize, groups: usize, core_off: i32, mem_off: i32, power_w: u32, unlocked: bool) -> Arc<Self> {
        Arc::new(Self {
            devices: (0..devices.max(1)).map(|_| HwControls::new(core_off, mem_off, power_w)).collect(),
            groups: AtomicUsize::new(groups),
            selected: AtomicUsize::new(0),
            unlocked,
        })
    }

    /// Update the number of selectable CPU-group rows (after a regroup).
    pub fn set_group_rows(&self, n: usize) {
        self.groups.store(n, Ordering::Relaxed);
    }

    /// Whether live hardware controls are enabled.
    pub fn unlocked(&self) -> bool {
        self.unlocked
    }

    /// The control block for device `i` (clamped to the valid range).
    pub fn device(&self, i: usize) -> &HwControls {
        &self.devices[i.min(self.devices.len() - 1)]
    }

    /// Number of GPU device control blocks.
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Total selectable rows: GPU devices followed by CPU groups.
    fn total(&self) -> usize {
        self.devices.len() + self.groups.load(Ordering::Relaxed)
    }

    /// The unified selection index over [GPU devices.., CPU groups..].
    pub fn selected(&self) -> usize {
        self.selected.load(Ordering::Relaxed).min(self.total() - 1)
    }

    /// The selected GPU device index, or `None` when a CPU group is selected.
    pub fn selected_device(&self) -> Option<usize> {
        let s = self.selected();
        (s < self.devices.len()).then_some(s)
    }

    /// The selected CPU-group index, or `None` when a GPU device is selected.
    pub fn selected_group(&self) -> Option<usize> {
        let s = self.selected();
        (s >= self.devices.len()).then(|| s - self.devices.len())
    }

    /// Advance the dashboard's selection (GPUs then groups; wraps around).
    pub fn select_next(&self) {
        let n = self.total();
        self.selected.store((self.selected() + 1) % n, Ordering::Relaxed);
    }
}

/// Fixed bounds for the clock-offset controls. Generous on purpose — the NVML
/// tuner additionally clamps to the card's real VF-offset range at apply time,
/// so these just stop the displayed value from running away.
const CORE_OFF_MIN: i32 = -1000;
const CORE_OFF_MAX: i32 = 1500;
const MEM_OFF_MIN: i32 = -2000;
const MEM_OFF_MAX: i32 = 6000;

pub struct HwControls {
    core_off: AtomicI32,
    mem_off: AtomicI32,
    power_w: AtomicU32,
    /// Power-limit clamp range (watts), learned from the card at startup.
    pwr_min: AtomicU32,
    pwr_max: AtomicU32,
    pwr_limits_set: AtomicBool,
    /// Whether this device should mine. Toggled from the dashboard (Backspace);
    /// the worker drains and idles while false. Devices start enabled.
    enabled: AtomicBool,
    seq: AtomicU64,
}

impl HwControls {
    /// Seed initial offsets/power from the CLI tuning flags so the panel opens
    /// showing what's already applied (0 ⇒ "default"). Initial offsets are
    /// clamped to the fixed bounds.
    fn new(core_off: i32, mem_off: i32, power_w: u32) -> Self {
        Self {
            core_off: AtomicI32::new(core_off.clamp(CORE_OFF_MIN, CORE_OFF_MAX)),
            mem_off: AtomicI32::new(mem_off.clamp(MEM_OFF_MIN, MEM_OFF_MAX)),
            power_w: AtomicU32::new(power_w),
            pwr_min: AtomicU32::new(0),
            pwr_max: AtomicU32::new(u32::MAX),
            pwr_limits_set: AtomicBool::new(false),
            enabled: AtomicBool::new(true),
            seq: AtomicU64::new(0),
        }
    }

    /// Whether this device is currently set to mine.
    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Flip the device between mining and idle (dashboard Backspace).
    pub fn toggle_enabled(&self) {
        self.enabled.fetch_xor(true, Ordering::Relaxed);
    }

    /// Set the device's enabled state directly (control server).
    pub fn set_enabled(&self, on: bool) {
        self.enabled.store(on, Ordering::Relaxed);
    }

    pub fn seq(&self) -> u64 {
        self.seq.load(Ordering::Relaxed)
    }
    pub fn core_off(&self) -> i32 {
        self.core_off.load(Ordering::Relaxed)
    }
    pub fn mem_off(&self) -> i32 {
        self.mem_off.load(Ordering::Relaxed)
    }
    pub fn power_w(&self) -> u32 {
        self.power_w.load(Ordering::Relaxed)
    }

    // Only the TUI key-handler thread mutates these, so a load/clamp/store is
    // race-free; workers only read.
    pub fn adjust_core(&self, delta: i32) {
        let n = (self.core_off.load(Ordering::Relaxed) + delta).clamp(CORE_OFF_MIN, CORE_OFF_MAX);
        self.core_off.store(n, Ordering::Relaxed);
        self.bump();
    }
    /// Set the displayed core offset without triggering a re-apply — used to
    /// reflect a value already pushed to the card (e.g. by `--auto-tune`).
    pub fn set_core_display(&self, v: i32) {
        self.core_off.store(v.clamp(CORE_OFF_MIN, CORE_OFF_MAX), Ordering::Relaxed);
    }
    /// As [`Self::set_core_display`] but for the memory offset.
    pub fn set_mem_display(&self, v: i32) {
        self.mem_off.store(v.clamp(MEM_OFF_MIN, MEM_OFF_MAX), Ordering::Relaxed);
    }
    pub fn adjust_mem(&self, delta: i32) {
        let n = (self.mem_off.load(Ordering::Relaxed) + delta).clamp(MEM_OFF_MIN, MEM_OFF_MAX);
        self.mem_off.store(n, Ordering::Relaxed);
        self.bump();
    }
    /// Set absolute clock offsets / power directly (control server), clamped and
    /// triggering a re-apply (bump) like the dashboard adjust keys.
    pub fn set_core_off(&self, v: i32) {
        self.core_off.store(v.clamp(CORE_OFF_MIN, CORE_OFF_MAX), Ordering::Relaxed);
        self.bump();
    }
    pub fn set_mem_off(&self, v: i32) {
        self.mem_off.store(v.clamp(MEM_OFF_MIN, MEM_OFF_MAX), Ordering::Relaxed);
        self.bump();
    }
    pub fn set_power_w(&self, v: u32) {
        let min = self.pwr_min.load(Ordering::Relaxed);
        let max = self.pwr_max.load(Ordering::Relaxed);
        self.power_w.store(v.clamp(min, max), Ordering::Relaxed);
        self.bump();
    }

    /// Adjust the power-limit target, clamped to the card's reported range
    /// (defaults to a 0 floor until the card publishes its limits).
    pub fn adjust_power(&self, delta: i32) {
        let min = self.pwr_min.load(Ordering::Relaxed) as i64;
        let max = self.pwr_max.load(Ordering::Relaxed) as i64;
        let cur = self.power_w.load(Ordering::Relaxed) as i64;
        let next = (cur + delta as i64).clamp(min, max) as u32;
        self.power_w.store(next, Ordering::Relaxed);
        self.bump();
    }

    /// Publish the card's power-limit range (watts) and seed the baseline from
    /// its current limit, once. Doesn't trigger a re-apply.
    pub fn init_power(&self, current: u32, min: u32, max: u32) {
        if !self.pwr_limits_set.swap(true, Ordering::Relaxed) {
            self.pwr_min.store(min, Ordering::Relaxed);
            self.pwr_max.store(max, Ordering::Relaxed);
            // Seed baseline unless the CLI already set an explicit limit.
            let _ = self
                .power_w
                .compare_exchange(0, current, Ordering::Relaxed, Ordering::Relaxed);
        }
    }

    fn bump(&self) {
        self.seq.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_spans_devices_then_groups() {
        // 2 GPUs followed by 3 CPU groups.
        let c = Controls::new(2, 3, 0, 0, 0, false);
        // Starts on the first GPU.
        assert_eq!(c.selected_device(), Some(0));
        assert_eq!(c.selected_group(), None);
        // Tab walks GPU0, GPU1, group0, group1, group2, then wraps to GPU0.
        c.select_next();
        assert_eq!(c.selected_device(), Some(1));
        c.select_next();
        assert_eq!(c.selected_device(), None);
        assert_eq!(c.selected_group(), Some(0));
        c.select_next();
        assert_eq!(c.selected_group(), Some(1));
        c.select_next();
        assert_eq!(c.selected_group(), Some(2));
        c.select_next();
        assert_eq!(c.selected_device(), Some(0));
    }

    #[test]
    fn selection_with_no_groups() {
        let c = Controls::new(1, 0, 0, 0, 0, false);
        assert_eq!(c.selected_device(), Some(0));
        assert_eq!(c.selected_group(), None);
        c.select_next(); // single row: wraps back to itself
        assert_eq!(c.selected_device(), Some(0));
    }
}
