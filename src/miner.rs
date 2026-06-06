//! Mining loop: build headers from pool work, run the solver, check candidate
//! solutions against the share target, and submit winners.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use log::{info, warn};
use sha2::{Digest, Sha256};

use crate::cpu_groups::{CpuGroups, CpuMining};
use crate::equihash;
use crate::params::{HEADER_LEN, SOLUTION_BYTES};
use crate::stratum::{StratumClient, Work};

/// Double SHA-256, as used for the Zcash/ZClassic block PoW hash.
fn sha256d(data: &[u8]) -> [u8; 32] {
    let first = Sha256::digest(data);
    let second = Sha256::digest(first);
    let mut out = [0u8; 32];
    out.copy_from_slice(&second);
    out
}

/// Does this header hash meet the target? The block hash is the byte-reversed
/// SHA-256d digest, compared big-endian against the target.
fn meets_target(header_with_solution: &[u8], target: &[u8; 32]) -> bool {
    let h = sha256d(header_with_solution);
    // Reverse to big-endian for an ordinary lexicographic comparison.
    let mut be = h;
    be.reverse();
    be <= *target
}

/// Compose the 140-byte header from the job prefix and a full 32-byte nonce.
fn build_header(work: &Work, nonce: &[u8; 32]) -> Vec<u8> {
    let mut header = Vec::with_capacity(HEADER_LEN);
    header.extend_from_slice(&work.prefix);
    header.extend_from_slice(nonce);
    debug_assert_eq!(header.len(), HEADER_LEN);
    header
}

/// Serialise a solution with its Bitcoin compact-size length prefix (0xfd +
/// uint16 LE for 400 bytes).
fn encode_solution_field(solution: &[u8]) -> Vec<u8> {
    debug_assert_eq!(solution.len(), SOLUTION_BYTES);
    let mut out = Vec::with_capacity(3 + SOLUTION_BYTES);
    out.push(0xfd);
    out.extend_from_slice(&(SOLUTION_BYTES as u16).to_le_bytes());
    out.extend_from_slice(solution);
    out
}

/// Which solver runs the puzzle.
pub enum Backend {
    /// CPU solver, with an optional per-group clamp for the fast (probabilistic)
    /// mining path; `None` runs the exact solver.
    Cpu(Option<usize>),
    #[cfg(feature = "gpu")]
    Gpu(crate::gpu::GpuSolver),
    #[cfg(feature = "cuda")]
    Cuda(crate::cuda::CudaSolver),
}

impl Backend {
    pub fn name(&self) -> String {
        match self {
            Backend::Cpu(_) => "CPU".to_string(),
            #[cfg(feature = "gpu")]
            Backend::Gpu(_) => "GPU (OpenCL)".to_string(),
            #[cfg(feature = "cuda")]
            Backend::Cuda(_) => "GPU (CUDA)".to_string(),
        }
    }

    /// Run the full solver for one header, returning valid solutions
    /// (synchronous; used by the benchmark/self-test paths).
    pub fn solve(&self, header: &[u8]) -> Result<Vec<Vec<u32>>> {
        match self {
            Backend::Cpu(clamp) => Ok(equihash::solve_with(header, *clamp)),
            #[cfg(feature = "gpu")]
            Backend::Gpu(solver) => solver.solve(header),
            #[cfg(feature = "cuda")]
            Backend::Cuda(solver) => solver.solve(header),
        }
    }

    /// How many `enqueue` calls happen before that header's results come back.
    /// 0 = synchronous (CPU/OpenCL); the CUDA backend pipelines one pass deep so
    /// the GPU keeps working while the host verifies the previous pass.
    pub fn pipeline_depth(&self) -> usize {
        match self {
            #[cfg(feature = "cuda")]
            Backend::Cuda(_) => 1,
            _ => 0,
        }
    }

    /// Submit `header` for solving. Returns solutions for the header passed
    /// `pipeline_depth()` calls ago (current header when depth is 0).
    pub fn enqueue(&mut self, header: &[u8]) -> Result<Vec<Vec<u32>>> {
        match self {
            Backend::Cpu(clamp) => Ok(equihash::solve_with(header, *clamp)),
            #[cfg(feature = "gpu")]
            Backend::Gpu(solver) => solver.solve(header),
            #[cfg(feature = "cuda")]
            Backend::Cuda(solver) => solver.enqueue(header),
        }
    }

    /// Flush the final in-flight pass(es) at shutdown (no-op when depth is 0).
    pub fn drain(&mut self) -> Result<Vec<Vec<u32>>> {
        match self {
            #[cfg(feature = "cuda")]
            Backend::Cuda(solver) => solver.drain(),
            _ => Ok(Vec::new()),
        }
    }

    /// The GPU's product name (e.g. "NVIDIA GeForce RTX 5080"), if the backend
    /// can report it.
    pub fn device_name(&self) -> Option<String> {
        match self {
            Backend::Cpu(_) => None,
            #[cfg(feature = "gpu")]
            Backend::Gpu(solver) => solver.device_name(),
            #[cfg(feature = "cuda")]
            Backend::Cuda(solver) => solver.device_name(),
        }
    }

    /// Current board power draw in watts, if the backend can report it.
    pub fn power_watts(&self) -> Option<f64> {
        match self {
            #[cfg(feature = "cuda")]
            Backend::Cuda(solver) => solver.power_watts(),
            _ => None,
        }
    }

    /// Current GPU core temperature in °C, if the backend can report it.
    pub fn temperature_c(&self) -> Option<u32> {
        match self {
            #[cfg(feature = "cuda")]
            Backend::Cuda(solver) => solver.temperature_c(),
            _ => None,
        }
    }

    /// Currently enforced power limit in watts, if the backend can report it.
    pub fn current_power_limit_w(&self) -> Option<u32> {
        match self {
            #[cfg(feature = "cuda")]
            Backend::Cuda(solver) => solver.current_power_limit_w(),
            _ => None,
        }
    }

    /// This card's (min, max) settable power limit in watts, if reportable.
    pub fn power_limit_range_w(&self) -> Option<(u32, u32)> {
        match self {
            #[cfg(feature = "cuda")]
            Backend::Cuda(solver) => solver.power_limit_range_w(),
            _ => None,
        }
    }

    /// The core clock offset (MHz) `--auto-tune` settled on, if it ran.
    pub fn tuned_core_offset(&self) -> Option<i32> {
        match self {
            #[cfg(feature = "cuda")]
            Backend::Cuda(solver) => solver.tuned_core_offset(),
            _ => None,
        }
    }

    /// The memory clock offset (MHz) `--auto-tune` settled on, if it ran.
    pub fn tuned_mem_offset(&self) -> Option<i32> {
        match self {
            #[cfg(feature = "cuda")]
            Backend::Cuda(solver) => solver.tuned_mem_offset(),
            _ => None,
        }
    }

    /// Current (SM core, memory) clock in MHz, each `None` if unavailable.
    pub fn current_clocks_mhz(&self) -> (Option<u32>, Option<u32>) {
        match self {
            #[cfg(feature = "cuda")]
            Backend::Cuda(solver) => solver.current_clocks_mhz(),
            _ => (None, None),
        }
    }

    /// Apply live dashboard hardware controls (core/mem VF offsets in MHz, power
    /// target in watts) to this backend's card. No-op for non-GPU backends.
    #[allow(unused_variables)]
    pub fn apply_hw_controls(&self, core_off: i32, mem_off: i32, power_w: u32) {
        match self {
            #[cfg(feature = "cuda")]
            Backend::Cuda(solver) => solver.apply_hw_controls(core_off, mem_off, power_w),
            _ => {}
        }
    }
}

/// Describes a solver to launch on its own thread. Construction happens inside
/// the worker thread so each device initialises its own GPU context.
#[derive(Clone, Copy)]
pub enum BackendSpec {
    Cpu(Option<usize>),
    #[cfg(feature = "gpu")]
    Gpu(usize),
    #[cfg(feature = "cuda")]
    Cuda(usize),
}

impl BackendSpec {
    pub fn build(&self) -> Result<Backend> {
        match self {
            BackendSpec::Cpu(clamp) => Ok(Backend::Cpu(*clamp)),
            #[cfg(feature = "gpu")]
            BackendSpec::Gpu(device) => Ok(Backend::Gpu(crate::gpu::GpuSolver::new(*device)?)),
            #[cfg(feature = "cuda")]
            BackendSpec::Cuda(device) => Ok(Backend::Cuda(crate::cuda::CudaSolver::new(*device)?)),
        }
    }

    fn label(&self) -> String {
        match self {
            BackendSpec::Cpu(_) => "CPU".to_string(),
            #[cfg(feature = "gpu")]
            BackendSpec::Gpu(d) => format!("GPU {d} (OpenCL)"),
            #[cfg(feature = "cuda")]
            BackendSpec::Cuda(d) => format!("GPU {d} (CUDA)"),
        }
    }
}

/// Progress counters for one worker (card).
struct WorkerStats {
    /// Valid solutions produced (the comparable "Sol/s" metric pools report).
    solutions: AtomicU64,
    shares: AtomicU64,
    /// Most recent board power sample in milliwatts (0 = unknown).
    power_mw: AtomicU32,
    /// Most recent GPU core temperature in °C (0 = unknown).
    temp_c: AtomicU32,
    /// Most recent SM core / memory clocks in MHz (0 = unknown).
    core_clock_mhz: AtomicU32,
    mem_clock_mhz: AtomicU32,
    /// The card's product name, published once the backend is built.
    name: OnceLock<String>,
}

/// Per-worker progress counters reported periodically, plus each worker's label.
pub(crate) struct Stats {
    workers: Vec<WorkerStats>,
    labels: Vec<String>,
}

/// A point-in-time copy of one card's counters, for the TUI / reporter.
pub(crate) struct CardSnapshot {
    /// Index + backend label, e.g. "GPU 0 (CUDA)".
    pub label: String,
    /// Product name, e.g. "NVIDIA GeForce RTX 5080" (once the backend is built).
    pub name: Option<String>,
    pub solutions: u64,
    pub shares: u64,
    pub power_mw: u32,
    pub temp_c: u32,
    pub core_clock_mhz: u32,
    pub mem_clock_mhz: u32,
}

impl Stats {
    /// Snapshot every card's current counters (cheap atomic loads).
    pub(crate) fn cards(&self) -> Vec<CardSnapshot> {
        self.workers
            .iter()
            .enumerate()
            .map(|(i, w)| CardSnapshot {
                label: self.labels[i].clone(),
                name: w.name.get().cloned(),
                solutions: w.solutions.load(Ordering::Relaxed),
                shares: w.shares.load(Ordering::Relaxed),
                power_mw: w.power_mw.load(Ordering::Relaxed),
                temp_c: w.temp_c.load(Ordering::Relaxed),
                core_clock_mhz: w.core_clock_mhz.load(Ordering::Relaxed),
                mem_clock_mhz: w.mem_clock_mhz.load(Ordering::Relaxed),
            })
            .collect()
    }

    /// Construct an all-zero `Stats` with the given worker labels (tests only).
    #[cfg(test)]
    pub(crate) fn for_test(labels: Vec<String>) -> Self {
        let workers = labels
            .iter()
            .map(|_| WorkerStats {
                solutions: AtomicU64::new(0),
                shares: AtomicU64::new(0),
                power_mw: AtomicU32::new(0),
                temp_c: AtomicU32::new(0),
                core_clock_mhz: AtomicU32::new(0),
                mem_clock_mhz: AtomicU32::new(0),
                name: OnceLock::new(),
            })
            .collect();
        Self { workers, labels }
    }
}

/// Mine until interrupted. One worker thread per backend draws nonces from a
/// shared counter and submits shares through the shared pool client. If
/// `job_timeout` is set, a worker pauses when no new job has arrived within it
/// (stale work) and resumes when fresh work comes in.
#[allow(clippy::too_many_arguments)]
pub fn run(
    client: Arc<StratumClient>,
    specs: Vec<BackendSpec>,
    running: Arc<AtomicBool>,
    job_timeout: Option<Duration>,
    tui: bool,
    pool: String,
    controls: Arc<crate::controls::Controls>,
    cpu_mining: Arc<CpuMining>,
    cpu_clamp: Option<usize>,
    control_port: Option<u16>,
) -> Result<()> {
    let nonce_counter = Arc::new(AtomicU64::new(0));
    let labels: Vec<String> = specs.iter().map(|s| s.label()).collect();
    let stats = Arc::new(Stats {
        workers: (0..specs.len())
            .map(|_| WorkerStats {
                solutions: AtomicU64::new(0),
                shares: AtomicU64::new(0),
                power_mw: AtomicU32::new(0),
                temp_c: AtomicU32::new(0),
                core_clock_mhz: AtomicU32::new(0),
                mem_clock_mhz: AtomicU32::new(0),
                name: OnceLock::new(),
            })
            .collect(),
        labels,
    });

    let mut handles = Vec::new();
    for (id, spec) in specs.into_iter().enumerate() {
        let client = client.clone();
        let running = running.clone();
        let nonce_counter = nonce_counter.clone();
        let stats = stats.clone();
        let controls = controls.clone();
        handles.push(std::thread::spawn(move || {
            if let Err(e) = worker(id, spec, client, running, nonce_counter, stats, job_timeout, controls) {
                warn!("worker {id} ({}) stopped: {e}", spec.label());
            }
        }));
    }

    // One supervisor thread for CPU mining: it spawns a worker per CPU group and
    // rebuilds them whenever the group size is changed from the dashboard. Each
    // group worker idles until the user enables it.
    {
        let client = client.clone();
        let running = running.clone();
        let nonce_counter = nonce_counter.clone();
        let cpu_mining = cpu_mining.clone();
        handles.push(std::thread::spawn(move || {
            cpu_mining_supervisor(cpu_mining, cpu_clamp, client, running, nonce_counter, job_timeout);
        }));
    }

    // Optional local control server for the GUI config tool (live retrieve/adjust).
    if let Some(port) = control_port {
        let controls = controls.clone();
        let cpu_mining = cpu_mining.clone();
        let stats = stats.clone();
        let running = running.clone();
        handles.push(std::thread::spawn(move || {
            crate::control::serve(port, controls, cpu_mining, stats, running);
        }));
    }

    if tui {
        // Foreground dashboard (replaces the periodic log reporter). Blocks until
        // Ctrl-C clears `running`; workers then wind down and we join below.
        crate::tui::run(&stats, &pool, &running, Instant::now(), &client, &controls, &cpu_mining);
    } else {
        // Reporter thread: per-card throughput (plus a total when multi-card).
        let running = running.clone();
        let stats = stats.clone();
        handles.push(std::thread::spawn(move || {
            let n = stats.workers.len();
            let mut last = Instant::now();
            let mut last_sol = vec![0u64; n];
            while running.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(500));
                if last.elapsed() < Duration::from_secs(10) {
                    continue;
                }
                let dt = last.elapsed().as_secs_f64();
                let mut total_sol = 0.0;
                let mut total_shares = 0u64;
                let mut total_w = 0.0;
                for i in 0..n {
                    let sol = stats.workers[i].solutions.load(Ordering::Relaxed);
                    let shares = stats.workers[i].shares.load(Ordering::Relaxed);
                    let watts = stats.workers[i].power_mw.load(Ordering::Relaxed) as f64 / 1000.0;
                    let sol_s = (sol - last_sol[i]) as f64 / dt;
                    // Append power + efficiency only when the card reports power.
                    let power = if watts > 0.0 {
                        total_w += watts;
                        format!(", {watts:.0} W, {:.2} Sol/W", sol_s / watts)
                    } else {
                        String::new()
                    };
                    let temp = match stats.workers[i].temp_c.load(Ordering::Relaxed) {
                        0 => String::new(),
                        t => format!(", {t}°C"),
                    };
                    let dev = stats.workers[i].name.get().map(|n| format!(" — {n}")).unwrap_or_default();
                    info!("{}{dev}: {sol_s:.1} Sol/s ({shares} shares{power}{temp})", stats.labels[i]);
                    last_sol[i] = sol;
                    total_sol += sol_s;
                    total_shares += shares;
                }
                if n > 1 {
                    let power = if total_w > 0.0 {
                        format!(", {total_w:.0} W, {:.2} Sol/W", total_sol / total_w)
                    } else {
                        String::new()
                    };
                    info!("total: {total_sol:.1} Sol/s ({total_shares} shares{power})");
                }
                last = Instant::now();
            }
        }));
    }

    for h in handles {
        let _ = h.join();
    }
    Ok(())
}

/// The context needed to check and submit one in-flight pass's results: the job
/// it was built for and the nonce that produced it. Kept in a queue because GPU
/// backends return results lagged by `pipeline_depth` passes.
struct Pending {
    header: Vec<u8>,
    nonce: [u8; 32],
    nonce1_len: usize,
    job_id: String,
    time_hex: String,
    target: [u8; 32],
}

/// Per-device mining loop. Keeps the pipeline `depth` passes deep so a GPU stays
/// busy while the host verifies the previous pass's results.
fn worker(
    id: usize,
    spec: BackendSpec,
    client: Arc<StratumClient>,
    running: Arc<AtomicBool>,
    nonce_counter: Arc<AtomicU64>,
    stats: Arc<Stats>,
    job_timeout: Option<Duration>,
    controls: Arc<crate::controls::Controls>,
) -> Result<()> {
    let mut backend = spec.build()?;
    let depth = backend.pipeline_depth();
    // Publish the card's product name for the dashboard/reporter.
    if let Some(name) = backend.device_name() {
        let _ = stats.workers[id].name.set(name);
    }
    // This worker drives its own device's control block.
    let dev_ctrl = controls.device(id);
    // Reflect the core/memory offsets --auto-tune locked in (already applied to
    // the card) in the dashboard panel.
    if let Some(core) = backend.tuned_core_offset() {
        dev_ctrl.set_core_display(core);
        dev_ctrl.set_mem_display(backend.tuned_mem_offset().unwrap_or(0));
    }
    // Seed the TUI's TDP control + clamp range from the card, and only start
    // applying dashboard controls once the user actually changes one.
    if let Some(cur) = backend.current_power_limit_w() {
        let (min, max) = backend.power_limit_range_w().unwrap_or((0, u32::MAX));
        dev_ctrl.init_power(cur, min, max);
    }
    let mut last_ctrl_seq = dev_ctrl.seq();
    let dev = stats.workers[id].name.get().map(|n| format!(" — {n}")).unwrap_or_default();
    info!("worker {id}: {} backend ready ({}{dev})", backend.name(), spec.label());
    let work_handle = client.work();

    // Wait for the first job.
    let mut current = loop {
        if !running.load(Ordering::Relaxed) {
            return Ok(());
        }
        if let Some(w) = work_handle.get() {
            break w;
        }
        std::thread::sleep(Duration::from_millis(200));
    };

    // Outstanding passes whose results haven't come back yet (FIFO).
    let mut inflight: std::collections::VecDeque<Pending> = std::collections::VecDeque::new();
    let mut pass = 0u64;
    let mut last_job = Instant::now();
    let mut paused = false;
    let mut disabled_pause = false;

    while running.load(Ordering::Relaxed) {
        if work_handle.epoch() != current.epoch {
            if let Some(w) = work_handle.get() {
                current = w;
            }
            last_job = Instant::now();
            if paused {
                info!("worker {id}: new job received, resuming");
                paused = false;
            }
        }

        // Dashboard enable/disable for this device (Backspace in the TUI). A
        // disabled device drains its pipeline and idles, clearing its readouts,
        // until it is re-enabled.
        if !dev_ctrl.enabled() {
            if !disabled_pause {
                info!("worker {id}: disabled from dashboard, idling");
                drain_pipeline(id, &mut backend, &mut inflight, &client, &stats)?;
                stats.workers[id].power_mw.store(0, Ordering::Relaxed);
                stats.workers[id].temp_c.store(0, Ordering::Relaxed);
                stats.workers[id].core_clock_mhz.store(0, Ordering::Relaxed);
                stats.workers[id].mem_clock_mhz.store(0, Ordering::Relaxed);
                disabled_pause = true;
            }
            std::thread::sleep(Duration::from_millis(200));
            continue;
        }
        if disabled_pause {
            info!("worker {id}: re-enabled from dashboard, resuming");
            disabled_pause = false;
            last_job = Instant::now(); // don't trip the stale guard on resume
        }

        // Stale-work guard: if no fresh job has arrived within `job_timeout`,
        // stop solving (drain the pipeline, idle) until new work comes in.
        if let Some(timeout) = job_timeout {
            if last_job.elapsed() >= timeout {
                if !paused {
                    warn!("worker {id}: no new job in {}s — pausing (work likely stale)", timeout.as_secs());
                    drain_pipeline(id, &mut backend, &mut inflight, &client, &stats)?;
                    stats.workers[id].power_mw.store(0, Ordering::Relaxed);
                    stats.workers[id].temp_c.store(0, Ordering::Relaxed);
                    stats.workers[id].core_clock_mhz.store(0, Ordering::Relaxed);
                    stats.workers[id].mem_clock_mhz.store(0, Ordering::Relaxed);
                    paused = true;
                }
                std::thread::sleep(Duration::from_millis(200));
                continue;
            }
        }

        // Apply any hardware-control changes made for this device from the
        // dashboard (only when the user unlocked live controls).
        let seq = dev_ctrl.seq();
        if controls.unlocked() && seq != last_ctrl_seq {
            backend.apply_hw_controls(dev_ctrl.core_off(), dev_ctrl.mem_off(), dev_ctrl.power_w());
            last_ctrl_seq = seq;
        }

        // Sample board power + temperature occasionally (cheap NVML calls, off
        // the GPU path).
        pass = pass.wrapping_add(1);
        if pass % 16 == 0 {
            if let Some(w) = backend.power_watts() {
                stats.workers[id].power_mw.store((w * 1000.0) as u32, Ordering::Relaxed);
            }
            if let Some(t) = backend.temperature_c() {
                stats.workers[id].temp_c.store(t, Ordering::Relaxed);
            }
            let (core_mhz, mem_mhz) = backend.current_clocks_mhz();
            if let Some(c) = core_mhz {
                stats.workers[id].core_clock_mhz.store(c, Ordering::Relaxed);
            }
            if let Some(m) = mem_mhz {
                stats.workers[id].mem_clock_mhz.store(m, Ordering::Relaxed);
            }
        }

        // Each pass explores a unique nonce drawn from the shared counter.
        let counter = nonce_counter.fetch_add(1, Ordering::Relaxed);
        let nonce = build_nonce(&current.nonce1, counter)?;
        let header = build_header(&current, &nonce);

        // Returns the results of the pass `depth` calls ago (current when depth 0).
        let solutions = backend.enqueue(&header)?;
        inflight.push_back(Pending {
            header,
            nonce,
            nonce1_len: current.nonce1.len(),
            job_id: current.job_id.clone(),
            time_hex: current.time_hex.clone(),
            target: current.target,
        });

        // Once the pipeline is primed, each enqueue completes the oldest pass.
        if inflight.len() > depth {
            let ctx = inflight.pop_front().unwrap();
            process_results(id, &client, &ctx, &solutions, &stats);
        }
    }

    drain_pipeline(id, &mut backend, &mut inflight, &client, &stats)?;
    Ok(())
}

/// Supervise CPU mining: spawn one worker per group for the current group size,
/// and whenever the dashboard cycles the size, stop those workers (`gen_running`),
/// rebuild the grouping, and respawn. Runs until `running` is cleared.
fn cpu_mining_supervisor(
    cpu_mining: Arc<CpuMining>,
    clamp: Option<usize>,
    client: Arc<StratumClient>,
    running: Arc<AtomicBool>,
    nonce_counter: Arc<AtomicU64>,
    job_timeout: Option<Duration>,
) {
    let mut size = cpu_mining.group_size();
    let mut groups = cpu_mining.groups();
    while running.load(Ordering::Relaxed) {
        // Spawn this generation's group workers.
        let gen_running = Arc::new(AtomicBool::new(true));
        let mut handles = Vec::new();
        for idx in 0..groups.len() {
            let groups = groups.clone();
            let client = client.clone();
            let running = running.clone();
            let gen_running = gen_running.clone();
            let nonce_counter = nonce_counter.clone();
            handles.push(std::thread::spawn(move || {
                if let Err(e) = cpu_group_worker(idx, groups, clamp, client, running, gen_running, nonce_counter, job_timeout) {
                    warn!("cpu group {idx} stopped: {e}");
                }
            }));
        }

        // Run until the dashboard changes the group size (or we're shutting down).
        while running.load(Ordering::Relaxed) && cpu_mining.group_size() == size {
            std::thread::sleep(Duration::from_millis(150));
        }

        // Wind down this generation (each worker exits after its current solve).
        gen_running.store(false, Ordering::Relaxed);
        for h in handles {
            let _ = h.join();
        }
        if !running.load(Ordering::Relaxed) {
            break;
        }

        // Size changed: rebuild the grouping (preserving enabled cores) and loop.
        size = cpu_mining.group_size();
        groups = cpu_mining.rebuild();
        info!("CPU mining regrouped to {size} core(s) per row ({} row(s))", groups.len());
    }
}

/// Mining loop for one CPU core group. Idles while the group is disabled or while
/// `gen_running` is cleared (a regroup); otherwise runs a clamped CPU solve on
/// the group's own rayon pool (sized to its core count) for each shared nonce and
/// submits any winners. Synchronous (no pipeline), drawing from the same nonce
/// counter / pool client as the backend workers so it adds throughput without
/// duplicating work.
#[allow(clippy::too_many_arguments)]
fn cpu_group_worker(
    idx: usize,
    cpu_groups: Arc<CpuGroups>,
    clamp: Option<usize>,
    client: Arc<StratumClient>,
    running: Arc<AtomicBool>,
    gen_running: Arc<AtomicBool>,
    nonce_counter: Arc<AtomicU64>,
    job_timeout: Option<Duration>,
) -> Result<()> {
    let group = cpu_groups.group(idx);
    let cores: Vec<usize> = group.cores().to_vec();
    let nthreads = cores.len();
    // A dedicated pool so this group's solves use exactly its cores and stay
    // independent of the global pool (and of the other groups). Each worker
    // thread is pinned (CPU affinity) to one of the group's logical cores, so
    // `--cpu-cores` actually controls *which* cores run.
    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    let pin = cores.clone();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(nthreads)
        .thread_name(move |t| format!("cpu-g{idx}-{t}"))
        .start_handler(move |t| {
            if let Some(&core) = pin.get(t) {
                if let Some(id) = core_ids.get(core) {
                    core_affinity::set_for_current(*id);
                }
            }
        })
        .build()
        .map_err(|e| anyhow!("cpu group {idx}: failed to build thread pool: {e}"))?;

    let work_handle = client.work();

    // Wait for the first job (or shutdown / regroup).
    let mut current = loop {
        if !running.load(Ordering::Relaxed) || !gen_running.load(Ordering::Relaxed) {
            return Ok(());
        }
        if let Some(w) = work_handle.get() {
            break w;
        }
        std::thread::sleep(Duration::from_millis(200));
    };
    let mut last_job = Instant::now();

    while running.load(Ordering::Relaxed) && gen_running.load(Ordering::Relaxed) {
        if work_handle.epoch() != current.epoch {
            if let Some(w) = work_handle.get() {
                current = w;
            }
            last_job = Instant::now();
        }

        // Idle when disabled, or when work has gone stale (same guard the backend
        // workers use). Sleep briefly so toggling/fresh work is picked up quickly.
        let stale = job_timeout.map_or(false, |t| last_job.elapsed() >= t);
        if !group.enabled() || stale {
            std::thread::sleep(Duration::from_millis(150));
            continue;
        }

        let counter = nonce_counter.fetch_add(1, Ordering::Relaxed);
        let nonce = build_nonce(&current.nonce1, counter)?;
        let header = build_header(&current, &nonce);
        let solutions = pool.install(|| equihash::solve_with(&header, clamp));

        let ctx = Pending {
            header,
            nonce,
            nonce1_len: current.nonce1.len(),
            job_id: current.job_id.clone(),
            time_hex: current.time_hex.clone(),
            target: current.target,
        };
        submit_solutions(
            &group.label(),
            &client,
            &ctx,
            &solutions,
            group.solutions_atomic(),
            group.shares_atomic(),
        );
    }
    Ok(())
}

/// Flush every in-flight pass, checking/submitting its results against the job
/// it was built for. Used at shutdown and when pausing on stale work.
fn drain_pipeline(
    id: usize,
    backend: &mut Backend,
    inflight: &mut std::collections::VecDeque<Pending>,
    client: &StratumClient,
    stats: &Stats,
) -> Result<()> {
    while let Some(ctx) = inflight.pop_front() {
        let solutions = backend.drain()?;
        process_results(id, client, &ctx, &solutions, stats);
    }
    Ok(())
}

/// Check one pass's solutions against its share target and submit winners.
fn process_results(
    id: usize,
    client: &StratumClient,
    ctx: &Pending,
    solutions: &[Vec<u32>],
    stats: &Stats,
) {
    submit_solutions(
        &stats.labels[id],
        client,
        ctx,
        solutions,
        &stats.workers[id].solutions,
        &stats.workers[id].shares,
    );
}

/// Count `solutions` toward `sol_counter`, then submit any that clear the share
/// target, bumping `share_counter` per accepted share. Shared by the GPU workers
/// (via [`process_results`]) and the CPU-group workers. `who` labels log lines.
fn submit_solutions(
    who: &str,
    client: &StratumClient,
    ctx: &Pending,
    solutions: &[Vec<u32>],
    sol_counter: &AtomicU64,
    share_counter: &AtomicU64,
) {
    sol_counter.fetch_add(solutions.len() as u64, Ordering::Relaxed);

    for sol in solutions {
        let packed = equihash::indices_to_solution(sol);
        let mut full = ctx.header.clone();
        full.extend_from_slice(&encode_solution_field(&packed));

        if meets_target(&full, &ctx.target) {
            let nonce2 = &ctx.nonce[ctx.nonce1_len..];
            match client.submit(who, &ctx.job_id, &ctx.time_hex, nonce2, &encode_solution_field(&packed)) {
                Ok(()) => {
                    share_counter.fetch_add(1, Ordering::Relaxed);
                }
                Err(e) => warn!("{who}: submit failed: {e}"),
            }
        }
    }
}

/// Build a 32-byte nonce: the pool's nonce1 prefix followed by a worker nonce2
/// tail whose first 8 bytes hold the little-endian counter.
fn build_nonce(nonce1: &[u8], counter: u64) -> Result<[u8; 32]> {
    if nonce1.len() > 32 {
        return Err(anyhow!("nonce1 is {} bytes, exceeds 32", nonce1.len()));
    }
    let mut nonce = [0u8; 32];
    nonce[..nonce1.len()].copy_from_slice(nonce1);

    let tail = &mut nonce[nonce1.len()..];
    let n = tail.len().min(8);
    tail[..n].copy_from_slice(&counter.to_le_bytes()[..n]);
    Ok(nonce)
}
