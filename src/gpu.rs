//! OpenCL Equihash 192,7 solver.
//!
//! Runs the entire solve on the GPU: BLAKE2b generation, all seven collision
//! rounds over a bucketed hash table, and back-reference recovery of the leaf
//! indices. Only a handful of candidate solutions cross back to the host, which
//! canonicalises, verifies and de-duplicates them. Building with
//! `--no-default-features` drops OpenCL entirely and the miner uses the CPU
//! solver.
//!
//! Memory layout: `NR_ROWS` buckets keyed by the high `ROW_BITS` bits of the
//! current 24-bit collision block, `NR_SLOTS` entries per bucket. Each slot is
//! u32 words — one leading word (leaf index for table 0, packed back-reference
//! otherwise) followed by blocks `r..7`, one 24-bit block per word so the
//! kernels do coalesced word-aligned loads/XORs.
//!
//! Sizing: `ROW_BITS=21` gives mean bucket occupancy 16 with a cap of 32 — the
//! 2x headroom that keeps overflow rare enough to find solutions. With the
//! split back-reference / ping-pong-block layout that costs ~6 GB (down from
//! ~11 GB when all seven block tables were resident), so 8 GB cards work.
//! Halving the buckets to save more memory makes mean = cap (heavy overflow →
//! ~zero solutions, measured), so `ROW_BITS=21` is the floor; `pick_geom` keeps
//! it and warns when a card is too small. `ZCL_OPENCL_ROWBITS` forces a value
//! for experimentation only (below 21 typically finds no solutions).

use anyhow::{anyhow, Result};
use ocl::{Buffer, MemFlags, ProQue};

use crate::blake;
use crate::equihash;
use crate::params::{BLAKE_CALLS, HASH_OUTPUT, HEADER_LEN, SOLUTION_INDICES};

/// Entries per bucket. Mean occupancy is 2^25 / NR_ROWS; overflow is dropped.
/// Must be a power of two. Kept fixed across VRAM profiles so the kernels' local
/// memory use is unchanged — only the bucket count (`ROW_BITS`) is scaled. At
/// fixed memory a higher mean has lower relative variance, so fewer buckets
/// overflow and more solutions survive.
const NR_SLOTS: usize = 32;
/// Bits needed to index a slot (log2 of NR_SLOTS).
const SLOT_BITS: usize = NR_SLOTS.trailing_zeros() as usize;
/// Maximum candidate solutions captured per nonce.
const MAX_SOLS: usize = 16384;
/// Number of collision tables (rounds 0..6) — i.e. back-reference arrays.
const NUM_TABLES: usize = 7;
/// Blocks per slot in a block working buffer (table 0 carries all 8). The two
/// ping-pong buffers are sized for this maximum.
const MAXB: usize = 8;

/// Table geometry, chosen to fit the device's VRAM. Only the bucket count
/// (`row_bits`) varies; `NR_SLOTS` is fixed.
#[derive(Clone, Copy)]
struct Geom {
    row_bits: u32,
    nr_rows: usize,
}

impl Geom {
    fn new(row_bits: u32) -> Self {
        Self { row_bits, nr_rows: 1usize << row_bits }
    }
    /// Total device memory used: 7 back-reference arrays (1 u32/slot, kept for
    /// recovery) + 2 block working buffers (MAXB u32/slot) + per-table counters.
    fn bytes(&self) -> u64 {
        let slots = (self.nr_rows * NR_SLOTS) as u64;
        let backrefs = NUM_TABLES as u64 * slots * 4;
        let blocks = 2 * slots * MAXB as u64 * 4;
        let counts = NUM_TABLES as u64 * self.nr_rows as u64 * 4;
        backrefs + blocks + counts
    }
    /// Size of the largest single buffer (a block working buffer) — must fit the
    /// device's `CL_DEVICE_MAX_MEM_ALLOC_SIZE`.
    fn largest_alloc(&self) -> u64 {
        (self.nr_rows * NR_SLOTS * MAXB * 4) as u64
    }
}

/// The only geometry with enough bucket headroom (2x: mean 16, cap 32) to find
/// solutions. Smaller tables don't yield, so this is the floor.
const VIABLE_ROW_BITS: u32 = 21;

/// Choose the table geometry. There's effectively one viable size (~6 GB with
/// the split layout); we detect the device's VRAM and warn clearly if it's too
/// small (rather than failing later with a cryptic allocation error).
/// `ZCL_OPENCL_ROWBITS` forces a value for experimentation — note that anything
/// below 21 typically finds *no* solutions, so it won't actually mine.
fn pick_geom(device: &ocl::Device) -> Geom {
    use ocl::enums::{DeviceInfo, DeviceInfoResult};

    if let Ok(v) = std::env::var("ZCL_OPENCL_ROWBITS") {
        if let Ok(rb) = v.parse::<u32>() {
            let g = Geom::new(rb.clamp(16, 24));
            log::warn!(
                "OpenCL: ROW_BITS={} forced via ZCL_OPENCL_ROWBITS (~{:.1} GB tables); values below {VIABLE_ROW_BITS} usually find no solutions",
                g.row_bits,
                g.bytes() as f64 / 1e9,
            );
            return g;
        }
    }

    let g = Geom::new(VIABLE_ROW_BITS);
    let q = |k| device.info(k).ok();
    let global = match q(DeviceInfo::GlobalMemSize) {
        Some(DeviceInfoResult::GlobalMemSize(b)) => b,
        _ => 0,
    };
    let max_alloc = match q(DeviceInfo::MaxMemAllocSize) {
        Some(DeviceInfoResult::MaxMemAllocSize(b)) => b,
        _ => 0,
    };
    if global > 0 {
        log::info!(
            "OpenCL: {:.1} GB VRAM (max alloc {:.1} GB); 192,7 tables need ~{:.1} GB",
            global as f64 / 1e9,
            max_alloc as f64 / 1e9,
            g.bytes() as f64 / 1e9,
        );
        let fits = (global as f64 * 0.85) as u64 >= g.bytes()
            && (max_alloc == 0 || max_alloc >= g.largest_alloc());
        if !fits {
            log::warn!(
                "OpenCL: this GPU has too little usable VRAM for the 192,7 tables (~{:.1} GB needed) — allocation will likely fail. Use a card with more VRAM, or the CUDA backend (--backend cuda) which needs only ~3.7 GB.",
                g.bytes() as f64 / 1e9,
            );
        }
    }
    g
}

const KERNEL_BODY: &str = include_str!("../kernels/equihash.cl");

/// Build the kernel source with the host-side constants prepended.
fn kernel_source(geom: &Geom) -> String {
    format!(
        "#define NR_ROWS {nr_rows}u\n\
         #define NR_SLOTS {nr_slots}u\n\
         #define ROW_BITS {row_bits}u\n\
         #define SLOT_BITS {slot_bits}u\n\
         #define SLOT_MASK {slot_mask}u\n\
         #define CBL 24u\n\
         #define MAXB {maxb}u\n\
         #define MAX_SOLS {max_sols}u\n\
         {body}",
        nr_rows = geom.nr_rows,
        nr_slots = NR_SLOTS,
        row_bits = geom.row_bits,
        slot_bits = SLOT_BITS,
        slot_mask = NR_SLOTS - 1,
        maxb = MAXB,
        max_sols = MAX_SOLS,
        body = KERNEL_BODY,
    )
}

/// The default (project-native) OpenCL solver, bound to one device. Wrapped by
/// [`GpuSolver`], which selects it for non-AMD devices.
struct LegacySolver {
    pq: ProQue,
    header: Buffer<u8>,
    /// Per-table back-reference arrays (1 u32/slot), kept resident for recovery.
    backref: Vec<Buffer<u32>>,
    /// Two ping-pong block working buffers (MAXB u32/slot) reused across rounds.
    blk: [Buffer<u32>; 2],
    counts: Buffer<u32>,
    sols: Buffer<u32>,
    solcnt: Buffer<u32>,
    out: Buffer<u32>,
    /// Bucket count chosen for this device (kernel global work size).
    nr_rows: usize,
}

impl LegacySolver {
    /// This device's product name (e.g. "NVIDIA GeForce RTX 5080"), if available.
    pub fn device_name(&self) -> Option<String> {
        self.pq.device().name().ok()
    }

    /// Initialise the solver and allocate all device buffers.
    pub fn new(platform: ocl::Platform, device: ocl::Device) -> Result<Self> {
        let geom = pick_geom(&device);
        // The device's platform must be set explicitly: ProQue otherwise builds
        // the context against `Platform::default()` (the first platform), which
        // fails clCreateContext with CL_INVALID_VALUE when the chosen device
        // lives on a different platform (e.g. an AMD card alongside NVIDIA).
        let pq = ProQue::builder()
            .src(kernel_source(&geom))
            .platform(platform)
            .device(device)
            .dims(geom.nr_rows)
            .build()
            .map_err(|e| anyhow!("OpenCL build failed: {e}"))?;

        let header = Buffer::<u8>::builder()
            .queue(pq.queue().clone())
            .flags(MemFlags::new().read_only())
            .len(HEADER_LEN)
            .build()?;
        // One back-reference array per table (1 word/slot), kept for recovery.
        let mut backref = Vec::with_capacity(NUM_TABLES);
        for _ in 0..NUM_TABLES {
            backref.push(
                Buffer::<u32>::builder()
                    .queue(pq.queue().clone())
                    .flags(MemFlags::new().read_write())
                    .len(geom.nr_rows * NR_SLOTS)
                    .build()?,
            );
        }
        // Two ping-pong block working buffers (MAXB words/slot).
        let mk_blk = || -> Result<Buffer<u32>> {
            Ok(Buffer::<u32>::builder()
                .queue(pq.queue().clone())
                .flags(MemFlags::new().read_write())
                .len(geom.nr_rows * NR_SLOTS * MAXB)
                .build()?)
        };
        let blk = [mk_blk()?, mk_blk()?];
        let counts = Buffer::<u32>::builder()
            .queue(pq.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(NUM_TABLES * geom.nr_rows)
            .build()?;
        let sols = Buffer::<u32>::builder()
            .queue(pq.queue().clone())
            .len(MAX_SOLS)
            .build()?;
        let solcnt = Buffer::<u32>::builder()
            .queue(pq.queue().clone())
            .len(1)
            .build()?;
        let out = Buffer::<u32>::builder()
            .queue(pq.queue().clone())
            .len(MAX_SOLS * SOLUTION_INDICES)
            .build()?;

        Ok(Self { pq, header, backref, blk, counts, sols, solcnt, out, nr_rows: geom.nr_rows })
    }

    /// Run gen + collision rounds + final + recovery on the GPU. Returns the
    /// raw candidate count and the recovered leaf indices (`n * 128`, where
    /// `n = min(raw, MAX_SOLS)`).
    fn run_pipeline(&self, header: &[u8]) -> Result<(usize, Vec<u32>)> {
        self.header.write(header).enq()?;
        self.counts.cmd().fill(0u32, None).enq()?;
        self.solcnt.cmd().fill(0u32, None).enq()?;

        // Round 0: hash + bucket into backref[0] + block buffer blk[0].
        let gen = self
            .pq
            .kernel_builder("gen")
            .arg(&self.header)
            .arg(&self.backref[0])
            .arg(&self.blk[0])
            .arg(&self.counts)
            .build()?;
        unsafe { gen.cmd().global_work_size(BLAKE_CALLS).enq()?; }

        // Intermediate collision rounds 0..5. Table r's blocks live in blk[r%2];
        // each round reads that and writes table r+1's backref + blk[(r+1)%2].
        for r in 0..(NUM_TABLES - 1) as u32 {
            let src_blk = &self.blk[(r % 2) as usize];
            let dst_blk = &self.blk[((r + 1) % 2) as usize];
            let k = self
                .pq
                .kernel_builder("round_collide")
                .arg(src_blk)
                .arg(&self.backref[r as usize + 1])
                .arg(dst_blk)
                .arg(&self.counts)
                .arg(r)
                .build()?;
            unsafe { k.cmd().global_work_size(self.nr_rows).enq()?; }
        }

        // Final round: table 6's blocks are in blk[6 % 2].
        let fin = self
            .pq
            .kernel_builder("final_round")
            .arg(&self.blk[(NUM_TABLES - 1) % 2])
            .arg(&self.counts)
            .arg(&self.sols)
            .arg(&self.solcnt)
            .build()?;
        unsafe { fin.cmd().global_work_size(self.nr_rows).enq()?; }

        let mut sc = [0u32; 1];
        self.solcnt.read(&mut sc[..]).enq()?;
        let raw = sc[0] as usize;
        let n = raw.min(MAX_SOLS);
        if n == 0 {
            return Ok((raw, Vec::new()));
        }

        // Recover leaf indices for each candidate (walks the back-ref arrays).
        let mut rec = self.pq.kernel_builder("recover");
        for b in &self.backref {
            rec.arg(b);
        }
        let rec = rec.arg(&self.sols).arg(&self.out).build()?;
        unsafe { rec.cmd().global_work_size(n).enq()?; }

        let mut out = vec![0u32; n * SOLUTION_INDICES];
        self.out.read(&mut out[..]).enq()?;
        Ok((raw, out))
    }

    /// Solve the puzzle for `header` (140 bytes). Returns valid, canonical,
    /// de-duplicated solutions as leaf-index lists.
    pub fn solve(&self, header: &[u8]) -> Result<Vec<Vec<u32>>> {
        assert_eq!(header.len(), HEADER_LEN);
        let base = blake::base_state(header);
        let (_raw, out) = self.run_pipeline(header)?;
        Ok(equihash::filter_candidates(&base, &out))
    }

    /// Time each GPU stage individually (forcing a sync between them).
    pub fn profile(&self, header: &[u8]) -> Result<()> {
        use log::info;
        use std::time::Instant;

        let q = self.pq.queue();
        let stage = |q: &ocl::Queue, label: &str, t: Instant| -> Result<()> {
            q.finish().map_err(|e| anyhow!("{label} failed: {e}"))?;
            info!("  {label:14} {:6.1} ms", t.elapsed().as_secs_f64() * 1000.0);
            Ok(())
        };

        self.header.write(header).enq()?;
        self.counts.cmd().fill(0u32, None).enq()?;
        self.solcnt.cmd().fill(0u32, None).enq()?;
        q.finish().ok();

        let t = Instant::now();
        let gen = self
            .pq
            .kernel_builder("gen")
            .arg(&self.header)
            .arg(&self.backref[0])
            .arg(&self.blk[0])
            .arg(&self.counts)
            .build()?;
        unsafe { gen.cmd().global_work_size(BLAKE_CALLS).enq()?; }
        stage(q, "gen", t)?;

        for r in 0..(NUM_TABLES - 1) as u32 {
            let t = Instant::now();
            let src_blk = &self.blk[(r % 2) as usize];
            let dst_blk = &self.blk[((r + 1) % 2) as usize];
            let k = self
                .pq
                .kernel_builder("round_collide")
                .arg(src_blk)
                .arg(&self.backref[r as usize + 1])
                .arg(dst_blk)
                .arg(&self.counts)
                .arg(r)
                .build()?;
            unsafe { k.cmd().global_work_size(self.nr_rows).enq()?; }
            stage(q, &format!("round {r}"), t)?;
        }

        let t = Instant::now();
        let fin = self
            .pq
            .kernel_builder("final_round")
            .arg(&self.blk[(NUM_TABLES - 1) % 2])
            .arg(&self.counts)
            .arg(&self.sols)
            .arg(&self.solcnt)
            .build()?;
        unsafe { fin.cmd().global_work_size(self.nr_rows).enq()?; }
        stage(q, "final", t)?;
        Ok(())
    }

    /// Solve and also return the raw GPU candidate count (for diagnostics).
    pub fn solve_with_stats(&self, header: &[u8]) -> Result<(usize, Vec<Vec<u32>>)> {
        assert_eq!(header.len(), HEADER_LEN);
        let base = blake::base_state(header);
        let (raw, out) = self.run_pipeline(header)?;
        Ok((raw, equihash::filter_candidates(&base, &out)))
    }

    /// Compute every first-round BLAKE2b output (one 48-byte digest per index).
    /// Used by the self-test to compare the GPU kernel against the CPU.
    pub fn hash_all(&self, header: &[u8]) -> Result<Vec<u8>> {
        assert_eq!(header.len(), HEADER_LEN);
        self.header.write(header).enq()?;

        let out = Buffer::<u8>::builder()
            .queue(self.pq.queue().clone())
            .flags(MemFlags::new().write_only())
            .len(BLAKE_CALLS * HASH_OUTPUT)
            .build()?;

        let kernel = self
            .pq
            .kernel_builder("equihash_hash")
            .arg(&self.header)
            .arg(&out)
            .build()?;
        unsafe { kernel.cmd().global_work_size(BLAKE_CALLS).enq()?; }

        let mut buf = vec![0u8; BLAKE_CALLS * HASH_OUTPUT];
        out.read(&mut buf).enq()?;
        Ok(buf)
    }
}

/// OpenCL solver for one device. Dispatches to the AMD-tuned kernel
/// (`equihash192_7.cl`) on AMD-vendor devices and the default project kernel
/// (`equihash.cl`) everywhere else. Forceable with `ZCL_OPENCL_KERNEL=amd|legacy`.
pub struct GpuSolver {
    inner: SolverInner,
    /// Per-card telemetry handle (AMD amdgpu sysfs / NVML), matched to the device
    /// by PCI bus id. `None` when no telemetry backend matches.
    tuner: Option<Box<dyn crate::gpu_tune::GpuTuner>>,
}

enum SolverInner {
    Legacy(LegacySolver),
    Amd(crate::gpu_amd::AmdSolver),
}

impl GpuSolver {
    /// Initialise the solver for a flat device index, choosing the kernel by
    /// device vendor (AMD → `equihash192_7.cl`).
    pub fn new(device_index: usize) -> Result<Self> {
        let (platform, device) = pick_device(device_index)?;
        // Resolve a telemetry handle (AMD sysfs / NVML) from the device's PCI bus
        // before `device` is consumed by the inner solver.
        let tuner = device_pci_bus_id(&device).and_then(|bus| crate::gpu_tune::open(&bus));
        let inner = if use_amd_kernel(&device) {
            log::info!("OpenCL: AMD device — using the equihash192_7 kernel");
            SolverInner::Amd(crate::gpu_amd::AmdSolver::new(platform, device)?)
        } else {
            SolverInner::Legacy(LegacySolver::new(platform, device)?)
        };
        Ok(Self { inner, tuner })
    }

    /// This device's product name, if available.
    pub fn device_name(&self) -> Option<String> {
        match &self.inner {
            SolverInner::Legacy(s) => s.device_name(),
            SolverInner::Amd(s) => s.device_name(),
        }
    }

    /// Current board power draw in watts (telemetry handle), if available.
    pub fn power_watts(&self) -> Option<f64> {
        self.tuner.as_ref().and_then(|t| t.watts())
    }

    /// Current GPU temperature in °C, if available.
    pub fn temperature_c(&self) -> Option<u32> {
        self.tuner.as_ref().and_then(|t| t.temperature_c())
    }

    /// Currently enforced power limit in watts, if available.
    pub fn current_power_limit_w(&self) -> Option<u32> {
        self.tuner.as_ref().and_then(|t| t.current_power_limit_w())
    }

    /// (min, max) settable power limit in watts, if available.
    pub fn power_limit_range_w(&self) -> Option<(u32, u32)> {
        self.tuner.as_ref().and_then(|t| t.power_limit_range_w())
    }

    /// Current (core, memory) clocks in MHz, each `None` if unavailable.
    pub fn current_clocks_mhz(&self) -> (Option<u32>, Option<u32>) {
        match &self.tuner {
            Some(t) => (t.core_clock_mhz(), t.mem_clock_mhz()),
            None => (None, None),
        }
    }

    /// Solve the puzzle for `header` (140 bytes).
    pub fn solve(&self, header: &[u8]) -> Result<Vec<Vec<u32>>> {
        match &self.inner {
            SolverInner::Legacy(s) => s.solve(header),
            SolverInner::Amd(s) => s.solve(header),
        }
    }

    /// Solve and also return the raw GPU candidate count (for diagnostics).
    pub fn solve_with_stats(&self, header: &[u8]) -> Result<(usize, Vec<Vec<u32>>)> {
        match &self.inner {
            SolverInner::Legacy(s) => s.solve_with_stats(header),
            SolverInner::Amd(s) => s.solve_with_stats(header),
        }
    }

    /// Time each GPU stage individually.
    pub fn profile(&self, header: &[u8]) -> Result<()> {
        match &self.inner {
            SolverInner::Legacy(s) => s.profile(header),
            SolverInner::Amd(s) => s.profile(header),
        }
    }

    /// Whether the per-index BLAKE2b probe ([`Self::hash_all`]) is available.
    /// Only the default kernel exposes a linear digest layout; the AMD kernel
    /// buckets in round 0, so the self-test skips the probe there.
    pub fn supports_blake_probe(&self) -> bool {
        matches!(self.inner, SolverInner::Legacy(_))
    }

    /// Compute every first-round BLAKE2b output (default kernel only).
    pub fn hash_all(&self, header: &[u8]) -> Result<Vec<u8>> {
        match &self.inner {
            SolverInner::Legacy(s) => s.hash_all(header),
            SolverInner::Amd(_) => Err(anyhow!("hash_all is not supported by the AMD kernel")),
        }
    }
}

/// The device's PCI address as `"DDDD:BB:DD.F"` (lowercase), for matching the
/// physical card to a telemetry backend. Prefers `cl_khr_pci_bus_info`; falls
/// back to `cl_device_topology_amd`. `None` if the device exposes neither.
fn device_pci_bus_id(device: &ocl::Device) -> Option<String> {
    const CL_DEVICE_PCI_BUS_INFO_KHR: u32 = 0x10F2;
    const CL_DEVICE_TOPOLOGY_AMD: u32 = 0x4037;
    // cl_device_pci_bus_info_khr = { u32 pci_domain, pci_bus, pci_device, pci_function }.
    if let Ok(b) = device.info_raw(CL_DEVICE_PCI_BUS_INFO_KHR) {
        if b.len() >= 16 {
            let rd = |i: usize| u32::from_ne_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
            return Some(format!("{:04x}:{:02x}:{:02x}.{:x}", rd(0), rd(4), rd(8), rd(12)));
        }
    }
    // cl_device_topology_amd (PCIE branch): 24-byte struct, bus/device/function
    // are the last three bytes; domain isn't exposed (assume 0000).
    if let Ok(b) = device.info_raw(CL_DEVICE_TOPOLOGY_AMD) {
        if b.len() >= 24 {
            return Some(format!("0000:{:02x}:{:02x}.{:x}", b[21], b[22], b[23]));
        }
    }
    None
}

/// Decide whether to drive `device` with the AMD `equihash192_7.cl` kernel.
/// `ZCL_OPENCL_KERNEL` forces the choice (`amd` or `legacy`); otherwise it's by
/// device vendor.
fn use_amd_kernel(device: &ocl::Device) -> bool {
    use ocl::enums::{DeviceInfo, DeviceInfoResult};
    match std::env::var("ZCL_OPENCL_KERNEL").ok().as_deref() {
        Some(v) if v.eq_ignore_ascii_case("amd") => return true,
        Some(v) if v.eq_ignore_ascii_case("legacy") => return false,
        _ => {}
    }
    match device.info(DeviceInfo::Vendor) {
        Ok(DeviceInfoResult::Vendor(v)) => {
            let v = v.to_ascii_lowercase();
            v.contains("advanced micro devices") || v.contains("amd")
        }
        _ => false,
    }
}

/// List `(platform, device)` names so the user can choose `--device`.
pub fn list_devices() -> Result<Vec<String>> {
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
    Ok(names)
}

/// For each OpenCL device — in the same flat order as [`list_devices`] and
/// `--devices` — whether its vendor is NVIDIA. The mixed backend uses this to
/// hand NVIDIA cards to CUDA (and mine only the non-NVIDIA OpenCL devices).
pub fn device_is_nvidia() -> Vec<bool> {
    use ocl::enums::{DeviceInfo, DeviceInfoResult};
    use ocl::{Device, Platform};
    let mut out = Vec::new();
    for platform in Platform::list() {
        for device in Device::list_all(platform).unwrap_or_default() {
            let is_nv = matches!(
                device.info(DeviceInfo::Vendor),
                Ok(DeviceInfoResult::Vendor(v)) if v.to_ascii_lowercase().contains("nvidia")
            );
            out.push(is_nv);
        }
    }
    out
}

/// The flat OpenCL device index of the first CPU-type device (e.g. PoCL), if any.
/// Lets CPU mining run through the OpenCL backend on the CPU. The index matches
/// [`list_devices`] / `--devices`.
pub fn cpu_device_index() -> Option<usize> {
    use ocl::enums::{DeviceInfo, DeviceInfoResult};
    use ocl::{Device, Platform};
    let mut idx = 0;
    for platform in Platform::list() {
        for device in Device::list_all(platform).unwrap_or_default() {
            let is_cpu = matches!(
                device.info(DeviceInfo::Type).ok(),
                Some(DeviceInfoResult::Type(t)) if t.contains(ocl::flags::DeviceType::CPU)
            );
            if is_cpu {
                return Some(idx);
            }
            idx += 1;
        }
    }
    None
}

/// Resolve a flat device index across all platforms, returning the device along
/// with the platform it belongs to (needed to build the context against the
/// right platform).
pub(crate) fn pick_device(index: usize) -> Result<(ocl::Platform, ocl::Device)> {
    use ocl::{Device, Platform};
    let mut idx = 0;
    for platform in Platform::list() {
        for device in Device::list_all(platform).unwrap_or_default() {
            if idx == index {
                return Ok((platform, device));
            }
            idx += 1;
        }
    }
    Err(anyhow!("no OpenCL device with index {index}"))
}
