//! CPU mining split into groups of cores that the dashboard can toggle on/off.
//!
//! The logical CPUs are partitioned into fixed-size groups (see
//! [`CpuGroups::new`]). Each group is mined by its own worker thread running on a
//! dedicated rayon pool sized to the group's core count, so enabling a group adds
//! roughly that many mining threads and disabling it stops them. Groups draw from
//! the same shared nonce counter and submit through the same pool client as the
//! GPU workers, so CPU mining runs alongside whatever backend is active.
//!
//! State here is pure atomics shared between the worker threads (which read
//! `enabled` and bump the counters) and the TUI (which toggles `enabled` and
//! reads the counters for the per-group Sol/s rows).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// One toggleable group of specific CPU cores. The group's worker pins its
/// mining threads to exactly these logical cores (see `miner::cpu_group_worker`).
pub struct CpuGroup {
    cores: Vec<usize>,
    enabled: AtomicBool,
    solutions: AtomicU64,
    shares: AtomicU64,
}

impl CpuGroup {
    /// The logical cores this group pins its threads to.
    pub fn cores(&self) -> &[usize] {
        &self.cores
    }

    /// Number of cores (and therefore mining threads) this group runs.
    pub fn ncores(&self) -> usize {
        self.cores.len()
    }

    /// Human-readable core list, e.g. "CPU cores 0-3", "CPU core 7", or
    /// "CPU cores 0,2,4,6" for a non-contiguous selection.
    pub fn label(&self) -> String {
        format!("CPU {}", fmt_cores(&self.cores))
    }

    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Flip the group between mining and idle (called from the TUI key handler).
    pub fn toggle(&self) {
        self.enabled.fetch_xor(true, Ordering::Relaxed);
    }

    /// Set the enabled state directly (used when rebuilding groups for a new
    /// group size, to restore the previously-enabled cores).
    pub fn set_enabled(&self, on: bool) {
        self.enabled.store(on, Ordering::Relaxed);
    }

    pub fn solutions(&self) -> u64 {
        self.solutions.load(Ordering::Relaxed)
    }

    pub fn shares(&self) -> u64 {
        self.shares.load(Ordering::Relaxed)
    }

    /// Counters the group's worker bumps as it mines (see `miner::submit_solutions`).
    pub(crate) fn solutions_atomic(&self) -> &AtomicU64 {
        &self.solutions
    }
    pub(crate) fn shares_atomic(&self) -> &AtomicU64 {
        &self.shares
    }
}

/// All CPU core groups, shared between the worker threads and the dashboard.
pub struct CpuGroups {
    groups: Vec<CpuGroup>,
}

impl CpuGroups {
    /// Group the selected `cores` into mining groups of up to `cores_per_group`,
    /// **aligned to core-index blocks**: a core belongs to block
    /// `core / cores_per_group`, and each non-empty block becomes one group. So a
    /// group never straddles a `cores_per_group`-aligned boundary (keeping its
    /// threads within one cache/NUMA-friendly range), and a partial selection
    /// splits at the boundary rather than being sliced into raw chunks — e.g.
    /// cores 2..=9 with size 4 give `[2,3] | [4,5,6,7] | [8,9]`, not
    /// `[2,3,4,5] | [6,7,8,9]`. `cores` is the set to mine on (`--cpu-cores`,
    /// default all); each group starts enabled iff `start_enabled`
    /// (`--cpu-mining`). Groups (rows) are ordered by ascending core index.
    pub fn new(cores: Vec<usize>, cores_per_group: usize, start_enabled: bool) -> Arc<Self> {
        let cores_per_group = cores_per_group.max(1);
        let mut blocks: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for c in cores {
            blocks.entry(c / cores_per_group).or_default().push(c);
        }
        let groups = blocks
            .into_values()
            .map(|mut cores| {
                cores.sort_unstable();
                CpuGroup {
                    cores,
                    enabled: AtomicBool::new(start_enabled),
                    solutions: AtomicU64::new(0),
                    shares: AtomicU64::new(0),
                }
            })
            .collect();
        Arc::new(Self { groups })
    }

    pub fn len(&self) -> usize {
        self.groups.len()
    }

    #[allow(dead_code)] // paired with len() for clippy::len_without_is_empty
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    pub fn group(&self, i: usize) -> &CpuGroup {
        &self.groups[i]
    }

    pub fn iter(&self) -> impl Iterator<Item = &CpuGroup> {
        self.groups.iter()
    }
}

/// Format a sorted core list compactly, collapsing contiguous runs into ranges:
/// `[5]` -> "core 5", `[0,1,2,3]` -> "cores 0-3", `[0,2,4]` -> "cores 0,2,4",
/// `[0,1,2,8,9]` -> "cores 0-2,8-9".
fn fmt_cores(cores: &[usize]) -> String {
    if cores.is_empty() {
        return "cores (none)".to_string();
    }
    let mut parts: Vec<String> = Vec::new();
    let mut i = 0;
    while i < cores.len() {
        let start = cores[i];
        let mut j = i;
        while j + 1 < cores.len() && cores[j + 1] == cores[j] + 1 {
            j += 1;
        }
        parts.push(if j == i {
            format!("{start}")
        } else {
            format!("{start}-{}", cores[j])
        });
        i = j + 1;
    }
    let noun = if cores.len() == 1 { "core" } else { "cores" };
    format!("{noun} {}", parts.join(","))
}

/// Build a [`CpuGroups`] for `cores`/`size`, enabling each group whose cores are
/// all in `enabled` (so the previously-enabled cores survive a regroup).
fn grouped(cores: &[usize], size: usize, enabled: &BTreeSet<usize>) -> Arc<CpuGroups> {
    let groups = CpuGroups::new(cores.to_vec(), size, false);
    for g in groups.iter() {
        let on = !g.cores().is_empty() && g.cores().iter().all(|c| enabled.contains(c));
        g.set_enabled(on);
    }
    groups
}

/// Live controller for CPU mining. Owns the selected cores plus the current
/// group size, and rebuilds the [`CpuGroups`] when the size is cycled from the
/// dashboard (the worker supervisor watches `group_size()` and respawns to
/// match). Which *cores* are enabled is tracked here so it survives a resize:
/// e.g. enabling cores 0-3 at size 4 keeps them on after switching to size 2.
pub struct CpuMining {
    cores: Vec<usize>,
    /// Group sizes the dashboard cycles through (sorted, includes the CLI value).
    sizes: Vec<usize>,
    size_idx: AtomicUsize,
    /// The set of cores currently enabled for mining (drives each group's state).
    enabled: Mutex<BTreeSet<usize>>,
    /// The current grouping, swapped on resize; read by the TUI and supervisor.
    groups: Mutex<Arc<CpuGroups>>,
    /// Bumps on every regroup so the TUI can resync its per-group state.
    generation: AtomicU64,
}

impl CpuMining {
    /// `cores` are the logical cores to mine on (`--cpu-cores`), `initial_size`
    /// the starting group size (`--cpu-group-size`), and `start_enabled` whether
    /// mining begins on (`--cpu-mining`).
    pub fn new(cores: Vec<usize>, initial_size: usize, start_enabled: bool) -> Arc<Self> {
        let initial_size = initial_size.max(1);
        let enabled: BTreeSet<usize> = if start_enabled { cores.iter().copied().collect() } else { BTreeSet::new() };
        let groups = grouped(&cores, initial_size, &enabled);

        // Cycle list: the usual powers of two plus the requested size, capped so a
        // group never exceeds the core count (unless the user explicitly asked for
        // a larger size), sorted and de-duplicated.
        let cap = cores.len().max(initial_size).max(1);
        let mut sizes: Vec<usize> = [1usize, 2, 4, 8]
            .into_iter()
            .chain([initial_size])
            .filter(|&s| s >= 1 && s <= cap)
            .collect();
        sizes.sort_unstable();
        sizes.dedup();
        let size_idx = sizes.iter().position(|&s| s == initial_size).unwrap_or(0);

        Arc::new(Self {
            cores,
            sizes,
            size_idx: AtomicUsize::new(size_idx),
            enabled: Mutex::new(enabled),
            groups: Mutex::new(groups),
            generation: AtomicU64::new(0),
        })
    }

    /// The current cores-per-group.
    pub fn group_size(&self) -> usize {
        self.sizes[self.size_idx.load(Ordering::Relaxed).min(self.sizes.len() - 1)]
    }

    /// Advance to the next group size in the cycle (dashboard 'g' key). The
    /// supervisor notices the change and regroups.
    pub fn cycle_group_size(&self) {
        let n = self.sizes.len();
        if n > 1 {
            let next = (self.size_idx.load(Ordering::Relaxed) + 1) % n;
            self.size_idx.store(next, Ordering::Relaxed);
        }
    }

    /// Set the group size to the nearest available value (control server). The
    /// supervisor notices the change and regroups.
    pub fn set_group_size(&self, n: usize) {
        if let Some((idx, _)) = self
            .sizes
            .iter()
            .enumerate()
            .min_by_key(|(_, &s)| (s as i64 - n as i64).unsigned_abs())
        {
            self.size_idx.store(idx, Ordering::Relaxed);
        }
    }

    /// A handle to the current grouping (cheap Arc clone).
    pub fn groups(&self) -> Arc<CpuGroups> {
        self.groups.lock().unwrap().clone()
    }

    /// Regroup counter; the TUI resets its per-group stats when this changes.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }

    /// Toggle mining on the current grouping's `gi`-th group, updating the
    /// persistent set of enabled cores so the choice survives a later resize.
    pub fn toggle_group(&self, gi: usize) {
        let groups = self.groups();
        if gi >= groups.len() {
            return;
        }
        let g = groups.group(gi);
        g.toggle();
        let on = g.enabled();
        let mut set = self.enabled.lock().unwrap();
        for &c in g.cores() {
            if on {
                set.insert(c);
            } else {
                set.remove(&c);
            }
        }
    }

    /// Rebuild the grouping for the current size, preserving the enabled cores,
    /// publish it, and bump the generation. Returns the new grouping.
    pub fn rebuild(&self) -> Arc<CpuGroups> {
        let size = self.group_size();
        let set = self.enabled.lock().unwrap().clone();
        let groups = grouped(&self.cores, size, &set);
        *self.groups.lock().unwrap() = groups.clone();
        self.generation.fetch_add(1, Ordering::Relaxed);
        groups
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partitions_cores_into_groups() {
        // Even split.
        let g = CpuGroups::new((0..24).collect(), 4, false);
        assert_eq!(g.len(), 6);
        assert!(g.iter().all(|x| x.ncores() == 4));

        // Remainder lands in a smaller final group.
        let g = CpuGroups::new((0..10).collect(), 4, false);
        assert_eq!(g.len(), 3);
        assert_eq!(g.group(0).ncores(), 4);
        assert_eq!(g.group(2).ncores(), 2);
        assert_eq!(g.group(2).cores(), &[8, 9]);
        assert_eq!(g.group(2).label(), "CPU cores 8-9");

        // Degenerate inputs don't panic or divide by zero.
        assert!(CpuGroups::new(vec![], 4, false).is_empty());
        assert_eq!(CpuGroups::new((0..3).collect(), 0, false).len(), 3); // per_group clamped to 1
    }

    #[test]
    fn groups_use_the_selected_cores() {
        // Cores bucket by aligned 4-core block (core/4 -> 0:cores0-3, 1:cores4-7,
        // 2:cores8-11), so [1,3,5,7,8] becomes [1,3] | [5,7] | [8].
        let g = CpuGroups::new(vec![1, 3, 5, 7, 8], 4, true);
        assert_eq!(g.len(), 3);
        assert_eq!(g.group(0).cores(), &[1, 3]);
        assert_eq!(g.group(0).label(), "CPU cores 1,3");
        assert_eq!(g.group(1).cores(), &[5, 7]);
        assert_eq!(g.group(2).cores(), &[8]);
        assert_eq!(g.group(2).label(), "CPU core 8");
        assert!(g.iter().all(|x| x.enabled())); // start_enabled = true
    }

    #[test]
    fn groups_align_to_blocks() {
        // A selection crossing 4-core block boundaries splits at the boundary,
        // not into raw chunks of 4: cores 2..=9 -> [2,3] | [4,5,6,7] | [8,9].
        let g = CpuGroups::new((2..10).collect(), 4, false);
        assert_eq!(g.len(), 3);
        assert_eq!(g.group(0).cores(), &[2, 3]);
        assert_eq!(g.group(1).cores(), &[4, 5, 6, 7]);
        assert_eq!(g.group(2).cores(), &[8, 9]);

        // Group size 1 is one row per core regardless of selection.
        assert_eq!(CpuGroups::new((0..4).collect(), 1, false).len(), 4);
    }

    #[test]
    fn start_enabled_controls_initial_state() {
        // Default (--cpu-mining absent): all groups start disabled.
        assert!(CpuGroups::new((0..8).collect(), 4, false).iter().all(|g| !g.enabled()));
        // --cpu-mining: all groups start enabled.
        assert!(CpuGroups::new((0..8).collect(), 4, true).iter().all(|g| g.enabled()));
    }

    #[test]
    fn cpu_mining_cycles_size_and_preserves_enabled_cores() {
        // 8 cores, start at size 4 fully enabled -> two groups [0-3],[4-7], both on.
        let m = CpuMining::new((0..8).collect(), 4, true);
        assert_eq!(m.group_size(), 4);
        assert_eq!(m.groups().len(), 2);
        assert!(m.groups().iter().all(|g| g.enabled()));

        // Disable the second group (cores 4-7): now only cores 0-3 are enabled.
        m.toggle_group(1);
        assert!(m.groups().group(0).enabled());
        assert!(!m.groups().group(1).enabled());

        // Cycle to size 8 and rebuild: the single [0-7] group is off, because not
        // all of its cores were enabled.
        while m.group_size() != 8 {
            m.cycle_group_size();
        }
        let g = m.rebuild();
        assert_eq!(g.len(), 1);
        assert!(!g.group(0).enabled());

        // Cycle to size 2 and rebuild: cores 0-3 are still tracked as enabled, so
        // [0,1] and [2,3] come back on while [4,5] and [6,7] stay off — the choice
        // survived two regroups (not derived from the all-off size-8 grouping).
        while m.group_size() != 2 {
            m.cycle_group_size();
        }
        let g = m.rebuild();
        assert_eq!(g.len(), 4);
        assert!(g.group(0).enabled()); // [0,1]
        assert!(g.group(1).enabled()); // [2,3]
        assert!(!g.group(2).enabled()); // [4,5]
        assert!(!g.group(3).enabled()); // [6,7]
        assert!(m.generation() >= 2);
    }

    #[test]
    fn toggle_flips_enabled() {
        let g = CpuGroups::new((0..8).collect(), 4, false);
        assert!(!g.group(0).enabled());
        g.group(0).toggle();
        assert!(g.group(0).enabled());
        assert!(!g.group(1).enabled()); // independent
        g.group(0).toggle();
        assert!(!g.group(0).enabled());
    }
}
