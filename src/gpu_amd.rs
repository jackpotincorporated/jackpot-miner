//! AMD OpenCL Equihash 192,7 solver (`kernels/equihash192_7.cl`).
//!
//! A second OpenCL backend, selected for AMD-vendor devices by
//! [`crate::gpu::GpuSolver`]. Where the default [`crate::gpu`] driver runs the
//! project's own `gen`/`round_collide`/`recover` kernel, this one drives a
//! self-contained, GCN-tuned kernel with a fixed table geometry and a different
//! host ABI: a `clearCounter` → `blake` → `round1..round7` → `combine`
//! pipeline.
//!
//! ## Geometry (hard-coded in the kernel, mirrored here)
//!
//! 2^25 initial entries are bucketed into `NR_ROWS = 8192` rows. Round 0 and the
//! early rounds keep `SLOTS_R0 = 4592` `uint8` slots per row (`buffer0`/
//! `buffer1` ping-pong, ~1.2 GB each); rounds 4–5 widen to `SLOTS_R45 = 8688`
//! `uint4` slots in `buffer2` (~1.1 GB). `buffer1` is additionally reinterpreted
//! as `uint4`/`uint2` for the late-round R46/R57 outputs at fixed offsets. A
//! flat counter array of 8 banks × 16384 tracks per-row occupancy; the
//! round-7/R5 survivor count lives at index `R5_COUNTER_IDX = 114688` and sizes
//! the `combine` launch.
//!
//! ## Hashing ABI
//!
//! `blake` takes the BLAKE2b first-block midstate as a by-value `ulong8`
//! (`hashState`, from [`BatchHasher::midstate`]) plus a `nonce` whose low 32
//! bits become message word `m[1]`'s low half (= header bytes [136..140]); the
//! kernel hard-codes `m[0] = 0`, so the header's bytes [128..136] must be zero
//! (the same `cuda_compatible` rule the CUDA backend uses). Each work item hashes
//! index `tId`, emitting the two leaf entries `2*tId` and `2*tId+1` — exactly the
//! canonical leaf-index/sub-block split [`crate::equihash`] verifies against.
//!
//! `combine` writes recovered solutions to the `output0`/`res` buffer:
//! `output0[0].s0` is the solution count and each solution is 32 `uint4`
//! (`SOLUTION_INDICES = 128` pre-sorted 25-bit leaf indices) at
//! `output0[1 + 32*i ..]`. Those flatten straight into
//! [`equihash::filter_candidates`], which canonicalises, verifies and
//! de-duplicates them — the same contract as the default driver.

use anyhow::{anyhow, Result};
use ocl::prm::Ulong8;
use ocl::{Buffer, MemFlags, ProQue};

use crate::blake::BatchHasher;
use crate::equihash;
use crate::params::{BLAKE_CALLS, HEADER_LEN, SOLUTION_INDICES};

/// Buckets ("rows") the 2^25 entries are hashed into (kernel: `& 0x1FFF`).
const NR_ROWS: usize = 8192;
/// `uint8` slots per row in the round-0/early `buffer0`/`buffer1` tables.
const SLOTS_R0: usize = 4592;
/// `uint4` slots per row in the round-4/5 `buffer2` table.
const SLOTS_R45: usize = 8688;
/// `uint8` entries in `buffer0`/`buffer1` (the kernel's `37617664` bound).
const BUF01_U8ENTRIES: usize = NR_ROWS * SLOTS_R0;
/// `uint4` entries in `buffer2`.
const BUF2_U4ENTRIES: usize = NR_ROWS * SLOTS_R45;
/// Flat counter array: 8 banks × 16384 (one bank consumed per round).
const COUNTERS_U32: usize = 8 * 16384;
/// Counter index holding the round-7 / R5 survivor count (sizes `combine`).
const R5_COUNTER_IDX: usize = 114688;
/// `combine` only emits solutions for `addr < this` (matches the kernel cap);
/// far above the ~2 solutions a 192,7 nonce yields. Reads are capped to match.
const MAX_WRITTEN_SOLS: usize = 16;
/// Solution buffer capacity in solutions (`output0` = 1 + 32*cap `uint4`).
const SOL_CAP: usize = 16;
/// `reqd_work_group_size` of `blake`/`combine`.
const WG_BLAKE: usize = 64;
/// `reqd_work_group_size` of `round1..round7`.
const WG_ROUND: usize = 256;
/// Input rows (buckets) each collision round reads. The table is keyed on 13
/// bits (8192) through round 4, then narrows to 12 bits (4096) — a round that
/// reads `b` rows must launch exactly `b * 4` work-groups (kernel:
/// `bucket = grp >> 2`), or it processes uninitialised rows and explodes.
const ROUND_BUCKETS: [usize; 7] = [8192, 8192, 8192, 8192, 4096, 4096, 4096];
/// Extra rows of slack appended to each big table. The kernel's per-row
/// `atomic_inc` writes are uncapped, so a row that overflows its slot count
/// spills into the next row and the top row spills past the nominal table end;
/// this slack absorbs that (mean occupancy ~4096 sits well under the 4592/8688
/// slot counts, so realistic overflow is a few rows at most).
const ROW_SLACK: usize = 64;

const KERNEL_SRC: &str = include_str!("../kernels/equihash192_7.cl");

/// A persistent AMD OpenCL solver bound to one device.
pub struct AmdSolver {
    pq: ProQue,
    /// Round-0/early ping-pong tables (`uint8`), reinterpreted at narrower
    /// widths in late rounds.
    buffer0: Buffer<u32>,
    buffer1: Buffer<u32>,
    /// Round-4/5 wide table (`uint4`).
    buffer2: Buffer<u32>,
    /// Per-row occupancy counters (8 banks).
    counters: Buffer<u32>,
    /// `res` / `output0`: `[count, then 32 uint4 per solution]`.
    sols: Buffer<u32>,
}

unsafe impl Send for AmdSolver {}

impl AmdSolver {
    /// This device's product name, if available.
    pub fn device_name(&self) -> Option<String> {
        self.pq.device().name().ok()
    }

    /// Build the solver on `(platform, device)` and allocate all device buffers
    /// (~3.5 GB total).
    pub fn new(platform: ocl::Platform, device: ocl::Device) -> Result<Self> {
        // Build the program via an explicit ProgramBuilder so OpenCL compiler
        // options can be injected. `ZCL_CL_OPTS` (space-separated) is appended to
        // the build — e.g. `-mwavefrontsize64` to force wave64 on RDNA, or other
        // flags — for RDNA4 tuning experiments without recompiling the host.
        let mut prog = ocl::Program::builder();
        prog.src(KERNEL_SRC);
        if let Ok(opts) = std::env::var("ZCL_CL_OPTS") {
            for opt in opts.split_whitespace() {
                prog.cmplr_opt(opt.to_string());
            }
            log::info!("AMD OpenCL: extra build options: {opts}");
        }
        let pq = ProQue::builder()
            .prog_bldr(prog)
            .platform(platform)
            .device(device)
            .dims(1) // placeholder; every launch sets its own work size
            .build()
            .map_err(|e| anyhow!("AMD OpenCL build failed: {e}"))?;

        let alloc = |len: usize| -> Result<Buffer<u32>> {
            Ok(Buffer::<u32>::builder()
                .queue(pq.queue().clone())
                .flags(MemFlags::new().read_write())
                .len(len)
                .build()?)
        };
        // uint8 entries → 8 u32 each; uint4 entries → 4 u32 each. Each table gets
        // ROW_SLACK extra rows of write headroom (see ROW_SLACK). buffer1's
        // nominal uint8 size (75.2M uint4) already covers the late-round R46/R57
        // regions at fixed offsets 48496640 / 67305472.
        let buffer0 = alloc((BUF01_U8ENTRIES + ROW_SLACK * SLOTS_R0) * 8)?;
        let buffer1 = alloc((BUF01_U8ENTRIES + ROW_SLACK * SLOTS_R0) * 8)?;
        let buffer2 = alloc((BUF2_U4ENTRIES + ROW_SLACK * SLOTS_R45) * 4)?;
        let counters = alloc(COUNTERS_U32)?;
        let sols = alloc((1 + 32 * SOL_CAP) * 4)?;

        Ok(Self { pq, buffer0, buffer1, buffer2, counters, sols })
    }

    /// Set the eight by-value args shared by every kernel in the pipeline:
    /// `(buffer0, buffer1, buffer2, counters, res, extra, hashState, nonce)`.
    fn build_kernel(&self, name: &str, hash_state: Ulong8, nonce: u64) -> Result<ocl::Kernel> {
        Ok(self
            .pq
            .kernel_builder(name)
            .arg(&self.buffer0)
            .arg(&self.buffer1)
            .arg(&self.buffer2)
            .arg(&self.counters)
            .arg(&self.sols)
            .arg(0u32) // extra (unused by the pipeline)
            .arg(hash_state)
            .arg(nonce)
            .build()?)
    }

    /// Run the full pipeline for `header` and return the flat recovered leaf
    /// indices (`n * SOLUTION_INDICES`), ready for [`equihash::filter_candidates`].
    fn run_pipeline(&self, header: &[u8]) -> Result<Vec<u32>> {
        let mid = BatchHasher::new(header).midstate();
        let hash_state = Ulong8::new(
            mid[0], mid[1], mid[2], mid[3], mid[4], mid[5], mid[6], mid[7],
        );
        // Kernel's gId = nonce & 0xFFFFFFFF = message word m[1] low = header[136..140].
        let nonce = u32::from_le_bytes(header[136..140].try_into().unwrap()) as u64;

        // Clear counters + solution header (global = counter uint4 count).
        let clear = self.build_kernel("clearCounter", hash_state, nonce)?;
        unsafe {
            clear.cmd().global_work_size(COUNTERS_U32 / 4).enq()?;
        }

        // Round 0: BLAKE2b + bucket. One work item per blake call (2^24); each
        // emits two leaf entries.
        let blake = self.build_kernel("blake", hash_state, nonce)?;
        unsafe {
            blake
                .cmd()
                .global_work_size(BLAKE_CALLS)
                .local_work_size(WG_BLAKE)
                .enq()?;
        }

        // Collision rounds 1..7 (4 groups per input row, 256 work items each).
        for r in 1..=7 {
            let k = self.build_kernel(&format!("round{r}"), hash_state, nonce)?;
            unsafe {
                k.cmd()
                    .global_work_size(ROUND_BUCKETS[r - 1] * 4 * WG_ROUND)
                    .local_work_size(WG_ROUND)
                    .enq()?;
            }
        }

        // Size `combine` from the round-7 survivor count (one group per candidate).
        let mut r5 = [0u32; 1];
        self.counters
            .read(&mut r5[..])
            .offset(R5_COUNTER_IDX)
            .enq()?;
        let groups = r5[0] as usize;
        if groups == 0 {
            return Ok(Vec::new());
        }
        let combine = self.build_kernel("combine", hash_state, nonce)?;
        unsafe {
            combine
                .cmd()
                .global_work_size(groups * WG_BLAKE)
                .local_work_size(WG_BLAKE)
                .enq()?;
        }

        // output0[0].s0 = solution count; each solution is 128 u32 (32 uint4)
        // starting at uint4 index 1 (= u32 offset 4).
        let mut head = [0u32; 1];
        self.sols.read(&mut head[..]).enq()?;
        let nsols = (head[0] as usize).min(MAX_WRITTEN_SOLS);
        if nsols == 0 {
            return Ok(Vec::new());
        }
        let mut data = vec![0u32; nsols * SOLUTION_INDICES];
        self.sols.read(&mut data[..]).offset(4).enq()?;
        Ok(data)
    }

    /// Solve for `header` (140 bytes): returns valid, canonical, de-duplicated
    /// solutions as leaf-index lists.
    pub fn solve(&self, header: &[u8]) -> Result<Vec<Vec<u32>>> {
        assert_eq!(header.len(), HEADER_LEN);
        let base = crate::blake::base_state(header);
        let out = self.run_pipeline(header)?;
        Ok(equihash::filter_candidates(&base, &out))
    }

    /// Solve and also return the raw recovered-candidate count (for diagnostics).
    pub fn solve_with_stats(&self, header: &[u8]) -> Result<(usize, Vec<Vec<u32>>)> {
        assert_eq!(header.len(), HEADER_LEN);
        let base = crate::blake::base_state(header);
        let out = self.run_pipeline(header)?;
        let raw = out.len() / SOLUTION_INDICES;
        Ok((raw, equihash::filter_candidates(&base, &out)))
    }

    /// Time each pipeline stage individually (forces a sync between stages).
    pub fn profile(&self, header: &[u8]) -> Result<()> {
        use log::info;
        use std::time::Instant;

        let mid = BatchHasher::new(header).midstate();
        let hash_state = Ulong8::new(
            mid[0], mid[1], mid[2], mid[3], mid[4], mid[5], mid[6], mid[7],
        );
        let nonce = u32::from_le_bytes(header[136..140].try_into().unwrap()) as u64;
        let q = self.pq.queue();
        let stage = |label: &str, t: Instant| -> Result<()> {
            q.finish().map_err(|e| anyhow!("{label} failed: {e}"))?;
            info!("  {label:14} {:6.1} ms", t.elapsed().as_secs_f64() * 1000.0);
            Ok(())
        };

        let t = Instant::now();
        let clear = self.build_kernel("clearCounter", hash_state, nonce)?;
        unsafe {
            clear.cmd().global_work_size(COUNTERS_U32 / 4).enq()?;
        }
        stage("clear", t)?;

        let t = Instant::now();
        let blake = self.build_kernel("blake", hash_state, nonce)?;
        unsafe {
            blake.cmd().global_work_size(BLAKE_CALLS).local_work_size(WG_BLAKE).enq()?;
        }
        stage("blake", t)?;

        for r in 1..=7 {
            let t = Instant::now();
            let k = self.build_kernel(&format!("round{r}"), hash_state, nonce)?;
            unsafe {
                k.cmd()
                    .global_work_size(ROUND_BUCKETS[r - 1] * 4 * WG_ROUND)
                    .local_work_size(WG_ROUND)
                    .enq()?;
            }
            stage(&format!("round {r}"), t)?;
        }
        Ok(())
    }
}
