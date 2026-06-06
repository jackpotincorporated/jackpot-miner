//! jackpotminer — a GPU-accelerated Equihash 192,7 miner for ZClassic and other
//! coins using the same proof-of-work.

mod blake;
mod control;
mod controls;
mod cpu_groups;
mod equihash;
mod miner;
mod params;
mod stratum;
mod tui;

#[cfg(feature = "gpu")]
mod gpu;

// AMD-tuned OpenCL kernel driver (selected by GpuSolver for AMD-vendor devices).
#[cfg(feature = "gpu")]
mod gpu_amd;

// Runtime dynamic-library loader (dlopen) for the CUDA driver + NVML.
#[cfg(feature = "cuda")]
mod dylib;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
mod nvml;

#[cfg(feature = "cuda")]
mod gpu_tune;

use std::io::IsTerminal;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use clap::{CommandFactory, FromArgMatches, Parser};
use log::{info, warn};

use crate::miner::BackendSpec;
use crate::stratum::StratumClient;

/// Command-line options.
#[derive(Parser, Debug)]
#[command(name = "jackpotminer", version, about = "equihash 192,7 miner")]
struct Args {
    /// Load options from a TOML config file. Values in the file are applied
    /// unless the same option is also given on the command line (CLI overrides
    /// the file overrides defaults). Keys mirror the long flag names without
    /// "--"; see mine.example.toml.
    #[arg(long, value_name = "FILE")]
    config: Option<String>,

    /// Pool URL, e.g. stratum+tcp://zcl.pool.example:3032
    #[arg(long)]
    url: Option<String>,

    /// Pool port, used when --url has no ":port" (e.g. --url pool.example --port 3032).
    #[arg(long)]
    port: Option<u16>,

    /// Worker / wallet login (e.g. address.worker).
    #[arg(short = 'u', long = "user", default_value = "")]
    user: String,

    /// Worker password / pool mode selector. Set to "no-jackpot" for PPLNS;
    /// any other value (the default) mines the jackpot. (--solo and --jackpot
    /// take precedence.)
    #[arg(short, long, default_value = "jackpot")]
    pass: String,

    /// Use PPLNS: set the pool password to "no-jackpot" (overrides --pass;
    /// opts out of jackpot mining).
    #[arg(long)]
    no_jackpot: bool,

    /// Set the pool password to "solo" (overrides --pass; for solo mining on
    /// pools that use this convention).
    #[arg(long, conflicts_with = "no_jackpot")]
    solo: bool,

    /// Jackpot participation as a whole percent, 3 (3%) to 100 (100%); sets the
    /// pool password to "jackpot.<percent>" (e.g. jackpot.50), overriding --pass.
    #[arg(long, value_name = "PERCENT", value_parser = clap::value_parser!(u32).range(3..=100), conflicts_with_all = ["no_jackpot", "solo"])]
    jackpot: Option<u32>,

    /// Pause mining if no new job arrives within this many seconds (stale work
    /// guard); resumes automatically when fresh work arrives. Default 600 (10
    /// minutes). 0 disables.
    #[arg(long, value_name = "SECS", default_value_t = 600)]
    job_timeout: u64,

    /// Open a local control server on 127.0.0.1:<PORT> so the GUI config tool can
    /// retrieve and adjust live settings (device enable, clocks/power, CPU group
    /// size/rows) on the fly. Off by default; localhost-only, no auth.
    #[arg(long, value_name = "PORT")]
    control_port: Option<u16>,

    /// CPU threads for the solver (defaults to all cores).
    #[arg(short, long)]
    threads: Option<usize>,

    /// Force the CPU hashing backend even when GPU support is compiled in.
    #[arg(long)]
    cpu: bool,

    /// Enable CPU mining at startup: the CPU mining rows (below the device table)
    /// begin enabled and mine alongside the selected backend. Off by default;
    /// rows can also be toggled live in the dashboard with Backspace. Note: each
    /// enabled row runs one full solve (~4 GB RAM) across its cores.
    #[arg(long)]
    cpu_mining: bool,

    /// Which logical CPU cores to use for CPU mining, e.g. "0-7", "0,2,4,6", or
    /// "0-3,8-11" (default: "all"). Cores are grouped into toggleable rows of
    /// --cpu-group-size (each row runs one solve with its threads pinned to its
    /// cores). Combine with --cpu-mining to start immediately (e.g. headless:
    /// --cpu-mining --cpu-cores 0-7).
    #[arg(long, value_name = "SPEC")]
    cpu_cores: Option<String>,

    /// Cores per CPU mining row (default 4). Each row runs one shared solve
    /// across its cores; larger groups cut memory sharply: total RAM is ~4 GB ×
    /// (enabled cores / this size). Use 1 for one row (and one solve) per core.
    /// Rows are aligned to core-index blocks of this size, so a row never
    /// straddles a boundary. Cycle it live in the dashboard with 'g'.
    #[arg(long, value_name = "N", default_value_t = 4)]
    cpu_group_size: usize,

    /// CPU solver bucket clamp: cap each exact-collision group at N entries.
    /// This bounds the naive Wagner algorithm's degenerate-collision blow-up
    /// (the same bound the GPU enforces via fixed bucket slots) and is required
    /// for the CPU solver to terminate on dense headers. Default 32 finds the
    /// ~2 real solutions per nonce; lower is faster but may drop solutions
    /// (≤8 drops real ones). `--cpu-clamp 0` runs the exact, unclamped solver
    /// (WARNING: can consume tens of GB and OOM on dense headers).
    #[arg(long, value_name = "N", default_value_t = 32)]
    cpu_clamp: usize,

    /// OpenCL device index for single-device modes (benchmark, gpu-debug).
    #[arg(short, long, default_value_t = 0)]
    device: usize,

    /// GPU devices to mine on: comma-separated indices (e.g. "0,1") or "all".
    /// Defaults to all detected devices.
    #[arg(long, default_value = "all")]
    devices: String,

    /// GPU backend: "mixed" (default — each card on its native backend: NVIDIA
    /// on CUDA, AMD/Intel on OpenCL), "opencl" (every card via OpenCL), or
    /// "cuda" (NVIDIA only). In mixed mode `--devices` indexes the combined list
    /// shown by --list-devices.
    #[arg(long, default_value = "mixed")]
    backend: String,

    /// Force the OpenCL backend, disabling CUDA (overrides --backend).
    #[arg(long)]
    force_opencl: bool,

    /// List available OpenCL devices and exit.
    #[arg(long)]
    list_devices: bool,

    /// Print detected OpenCL/CUDA devices as JSON and exit. Used by the GUI
    /// config tool to populate card-specific options.
    #[arg(long)]
    devices_json: bool,

    /// Run internal correctness self-tests and exit.
    #[arg(long)]
    selftest: bool,

    /// Run only the GPU solver on a fixed header with diagnostics, and exit.
    #[arg(long)]
    gpu_debug: bool,

    /// Benchmark the selected backend over N solves (no pool) and exit.
    #[arg(long, value_name = "N")]
    benchmark: Option<usize>,

    /// Disable the live dashboard and use periodic log lines instead. (The
    /// dashboard is on by default when mining in a terminal.)
    #[arg(long)]
    no_tui: bool,

    /// Don't put GPUs into maximum-performance mode (skip CUDA clock/power tuning).
    #[arg(long)]
    no_gpu_tune: bool,

    /// Allow the TUI's live hardware-control keys (core/mem offset, TDP) to
    /// change clocks/power. Locked by default so stray key presses can't retune.
    #[arg(long)]
    unlock_controls: bool,

    /// Auto-tune each GPU at startup: sweep the core clock offset up to find the
    /// fastest stable solve rate (overclock-for-speed; needs root). Takes ~30 s.
    #[arg(long)]
    auto_tune: bool,

    /// Efficiency: cap each GPU's power limit in watts (default: card max).
    /// Lower power trades a little hashrate for much better Sol/W.
    #[arg(long, value_name = "WATTS")]
    power_limit: Option<u32>,

    /// Efficiency: lock each GPU's core/SM clock in MHz (default: card max).
    #[arg(long, value_name = "MHZ")]
    gpu_clock: Option<u32>,

    /// Efficiency: lock each GPU's memory clock in MHz (default: card max).
    #[arg(long, value_name = "MHZ")]
    mem_clock: Option<u32>,

    /// Core clock V/F offset in MHz (LACT-style, e.g. 200 or -150). Combine with
    /// --power-limit for undervolt-style efficiency. Signed.
    #[arg(long, value_name = "MHZ", allow_hyphen_values = true)]
    gpu_clock_offset: Option<i32>,

    /// Memory clock V/F offset in MHz (LACT-style, signed).
    #[arg(long, value_name = "MHZ", allow_hyphen_values = true)]
    mem_clock_offset: Option<i32>,
}

/// The options loadable from a `--config` TOML file. Every field is optional; a
/// present value is applied to [`Args`] unless that option was also given on the
/// command line. Keys are the kebab-case long flag names (e.g. `cpu-group-size`).
#[derive(Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
struct FileConfig {
    url: Option<String>,
    port: Option<u16>,
    user: Option<String>,
    pass: Option<String>,
    no_jackpot: Option<bool>,
    solo: Option<bool>,
    jackpot: Option<u32>,
    job_timeout: Option<u64>,
    control_port: Option<u16>,
    threads: Option<usize>,
    cpu: Option<bool>,
    cpu_mining: Option<bool>,
    cpu_cores: Option<String>,
    cpu_group_size: Option<usize>,
    cpu_clamp: Option<usize>,
    device: Option<usize>,
    devices: Option<String>,
    backend: Option<String>,
    force_opencl: Option<bool>,
    no_tui: Option<bool>,
    no_gpu_tune: Option<bool>,
    unlock_controls: Option<bool>,
    auto_tune: Option<bool>,
    power_limit: Option<u32>,
    gpu_clock: Option<u32>,
    mem_clock: Option<u32>,
    gpu_clock_offset: Option<i32>,
    mem_clock_offset: Option<i32>,
    /// Per-device GPU tuning overrides (`[[gpu]]` tables); config-file only.
    #[serde(default)]
    gpu: Vec<GpuDeviceCfg>,
}

/// One `[[gpu]]` config table: per-device backend selection plus tuning that
/// overrides the global tuning flags for that device index (tuning is CUDA/NVML
/// only). `backend` ("cuda" or "opencl") lets individual cards run on a
/// different backend than the global `--backend`; when unset the card uses the
/// global default.
#[derive(Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
struct GpuDeviceCfg {
    index: usize,
    backend: Option<String>,
    power_limit: Option<u32>,
    gpu_clock: Option<u32>,
    mem_clock: Option<u32>,
    gpu_clock_offset: Option<i32>,
    mem_clock_offset: Option<i32>,
}

/// Read `--config` (if given) and fold its values into `args`: a file value is
/// applied only when that option was *not* passed explicitly on the command line
/// (so the CLI always wins). `matches` is used to tell explicit flags from
/// defaults.
fn apply_config(args: &mut Args, matches: &clap::ArgMatches) -> Result<Vec<GpuDeviceCfg>> {
    let Some(path) = args.config.clone() else {
        return Ok(Vec::new());
    };
    let text = std::fs::read_to_string(&path).with_context(|| format!("reading config file '{path}'"))?;
    let file: FileConfig = toml::from_str(&text).with_context(|| format!("parsing config file '{path}'"))?;

    let explicit = |name: &str| matches.value_source(name) == Some(clap::parser::ValueSource::CommandLine);
    // Scalar/bool options: take the file value (unwrapped) when not on the CLI.
    macro_rules! merge {
        ($($f:ident),* $(,)?) => {$(
            if !explicit(stringify!($f)) {
                if let Some(v) = file.$f { args.$f = v; }
            }
        )*};
    }
    // Optional options: copy the file's Option directly when not on the CLI.
    macro_rules! merge_opt {
        ($($f:ident),* $(,)?) => {$(
            if !explicit(stringify!($f)) && file.$f.is_some() {
                args.$f = file.$f;
            }
        )*};
    }

    merge!(
        user, pass, no_jackpot, solo, job_timeout, cpu, cpu_mining, cpu_group_size, cpu_clamp,
        device, devices, backend, force_opencl, no_tui, no_gpu_tune, unlock_controls, auto_tune,
    );
    merge_opt!(url, port, jackpot, control_port, threads, cpu_cores, power_limit, gpu_clock, mem_clock, gpu_clock_offset, mem_clock_offset);
    Ok(file.gpu)
}

/// Look for a default `mine.toml` (next to the binary, then the working dir), so
/// a double-clicked binary with a config beside it just works.
fn find_default_config() -> Option<String> {
    let mut cands: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            cands.push(dir.join("mine.toml"));
        }
    }
    cands.push(std::path::PathBuf::from("mine.toml"));
    cands.into_iter().find(|p| p.exists()).map(|p| p.to_string_lossy().into_owned())
}

/// When launched from a GUI (no controlling terminal) for an interactive mining
/// run, relaunch ourselves inside a terminal emulator so the dashboard is
/// visible. Returns true if a window was launched (the caller should then exit).
/// Falls through (returns false) on headless hosts / when no terminal is found,
/// and is disabled by `JACKPOTMINER_NO_WINDOW=1`.
fn relaunch_in_terminal(args: &Args) -> bool {
    use std::io::IsTerminal;
    // Skip when already interactive, told to go headless, already relaunched, or
    // explicitly disabled. Require *no* tty on either std stream (the GUI case),
    // so piping output from a real shell doesn't spawn a window.
    if std::io::stdout().is_terminal()
        || std::io::stdin().is_terminal()
        || args.no_tui
        || std::env::var_os("JACKPOTMINER_IN_TERMINAL").is_some()
        || std::env::var_os("JACKPOTMINER_NO_WINDOW").is_some()
    {
        return false;
    }
    // One-shot/diagnostic modes don't need a window, and there's nothing to mine
    // without a pool.
    if args.list_devices || args.devices_json || args.selftest || args.gpu_debug || args.benchmark.is_some() || args.url.is_none() {
        return false;
    }
    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(_) => return false,
    };
    let fwd: Vec<String> = std::env::args().skip(1).collect();

    // (program, args before the command). The command (exe + forwarded args) is
    // appended. Try $TERMINAL first, then common emulators.
    let mut tries: Vec<(String, Vec<&str>)> = Vec::new();
    let term_env = std::env::var("TERMINAL").ok();
    if let Some(t) = &term_env {
        tries.push((t.clone(), vec!["-e"]));
    }
    for (p, pre) in [
        ("x-terminal-emulator", vec!["-e"]),
        ("gnome-terminal", vec!["--"]),
        ("konsole", vec!["-e"]),
        ("xfce4-terminal", vec!["-x"]),
        ("alacritty", vec!["-e"]),
        ("wezterm", vec!["start", "--"]),
        ("kitty", vec![]),
        ("foot", vec![]),
        ("ghostty", vec!["-e"]),
        ("xterm", vec!["-e"]),
    ] {
        tries.push((p.to_string(), pre));
    }

    for (prog, pre) in tries {
        let ok = std::process::Command::new(&prog)
            .args(&pre)
            .arg(&exe)
            .args(&fwd)
            .env("JACKPOTMINER_IN_TERMINAL", "1")
            .spawn()
            .is_ok();
        if ok {
            return true;
        }
    }
    false
}

fn main() -> Result<()> {
    // Parse the CLI, but keep the matches so a `--config` file can fill in any
    // option that wasn't passed explicitly.
    let matches = Args::command().get_matches();
    let mut args = Args::from_arg_matches(&matches).expect("clap matches convert to Args");
    // Double-click convenience: with nothing specified, pick up a mine.toml.
    if args.config.is_none()
        && args.url.is_none()
        && !args.selftest
        && !args.list_devices
        && !args.devices_json
        && !args.gpu_debug
        && args.benchmark.is_none()
    {
        args.config = find_default_config();
    }
    let gpu_devices = apply_config(&mut args, &matches)?;
    // If started from a GUI, reopen in a terminal window so the dashboard shows.
    if relaunch_in_terminal(&args) {
        return Ok(());
    }

    // The dashboard is on by default, but only for the actual mining run (not for
    // one-shot modes like --selftest/--benchmark) and only on a real terminal.
    // Anything else falls back to ordinary log output.
    let mining_mode =
        !args.list_devices && !args.selftest && !args.gpu_debug && args.benchmark.is_none();
    let tui = !args.no_tui && mining_mode && std::io::stdout().is_terminal();
    if tui {
        // Capture logs into the dashboard's pane instead of the screen.
        tui::install_logger();
    } else {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
        if !args.no_tui && mining_mode {
            info!("no terminal detected — using log output (the dashboard needs a TTY)");
        }
    }

    // Install the GPU tuning policy before any solver is built. Each CUDA solver
    // applies it to its own card and restores defaults when dropped (covers the
    // Ctrl-C shutdown path, since workers drop their solvers on exit).
    #[cfg(feature = "cuda")]
    {
        gpu_tune::configure(gpu_tune::TuneConfig {
            enabled: !args.no_gpu_tune,
            power_limit_w: args.power_limit,
            gpu_clock_mhz: args.gpu_clock,
            mem_clock_mhz: args.mem_clock,
            gpu_offset_mhz: args.gpu_clock_offset,
            mem_offset_mhz: args.mem_clock_offset,
            auto_tune: args.auto_tune,
            unlock_controls: args.unlock_controls,
        });
        // Per-device overrides from `[[gpu]]` config tables.
        gpu_tune::configure_devices(
            gpu_devices
                .iter()
                .map(|g| {
                    (
                        g.index,
                        gpu_tune::DeviceTune {
                            power_limit_w: g.power_limit,
                            gpu_clock_mhz: g.gpu_clock,
                            mem_clock_mhz: g.mem_clock,
                            gpu_offset_mhz: g.gpu_clock_offset,
                            mem_offset_mhz: g.mem_clock_offset,
                        },
                    )
                })
                .collect(),
        );
    }

    if args.list_devices {
        list_devices();
        return Ok(());
    }

    if args.devices_json {
        println!("{}", devices_json());
        return Ok(());
    }

    if let Some(n) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .ok();
    }

    if args.selftest {
        return selftest(args.device);
    }

    #[cfg(feature = "cuda")]
    if args.gpu_debug && args.backend.eq_ignore_ascii_case("cuda") {
        let solver = cuda::CudaSolver::new(args.device)?;
        let mut warm = vec![0x42u8; params::HEADER_LEN];
        cuda_compatible(&mut warm); // CUDA needs header[128..135] == 0
        solver.solve(&warm)?; // warm up
        info!("CUDA per-stage timing (warm):");
        solver.profile(&warm)?;
        let runs = 16u32;
        let start = std::time::Instant::now();
        let mut valid = 0usize;
        for nonce in 0..runs {
            let mut header = vec![0x42u8; params::HEADER_LEN];
            header[108..112].copy_from_slice(&nonce.to_le_bytes());
            cuda_compatible(&mut header);
            valid += solver.solve(&header)?.len();
        }
        let dt = start.elapsed().as_secs_f64();
        info!("CUDA: {:.0} ms/solve, {:.2} valid/solve", dt * 1000.0 / runs as f64, valid as f64 / runs as f64);
        return Ok(());
    }

    #[cfg(feature = "gpu")]
    if args.gpu_debug {
        let solver = gpu::GpuSolver::new(args.device)?;
        info!("per-stage timing:");
        solver.profile(&vec![0x42u8; params::HEADER_LEN])?;
        let runs = 32u32;
        let mut max_raw = 0usize;
        let mut total = std::time::Duration::ZERO;
        for nonce in 0..runs {
            // Vary the nonce region of an otherwise fixed header.
            let mut header = vec![0x42u8; params::HEADER_LEN];
            header[108..112].copy_from_slice(&nonce.to_le_bytes());
            let t = std::time::Instant::now();
            let (raw, sols) = solver.solve_with_stats(&header)?;
            let dt = t.elapsed();
            total += dt;
            max_raw = max_raw.max(raw);
            info!(
                "nonce {nonce:2}: raw_candidates={raw:6}, valid={}, {:.0} ms",
                sols.len(),
                dt.as_secs_f64() * 1000.0
            );
        }
        info!(
            "summary: {runs} solves, avg {:.0} ms/solve, max raw candidates={max_raw} (MAX_SOLS cap is plenty)",
            total.as_secs_f64() * 1000.0 / runs as f64
        );
        return Ok(());
    }

    if let Some(runs) = args.benchmark {
        let specs = backend_specs(&args, &gpu_devices)?;
        return benchmark(specs, runs.max(1));
    }

    let url = args
        .url
        .as_deref()
        .ok_or_else(|| anyhow!("--url is required (e.g. stratum+tcp://host:port)"))?;
    let (host, port) = parse_url(url, args.port)?;

    // Password = pool mode. --solo / --jackpot <pct> take precedence; otherwise
    // PPLNS only when explicitly requested (--no-jackpot or `-p no-jackpot`),
    // and jackpot for anything else.
    let pass = if args.solo {
        "solo".to_string()
    } else if let Some(pct) = args.jackpot {
        format!("jackpot.{pct}")
    } else if args.no_jackpot || args.pass == "no-jackpot" {
        "no-jackpot".to_string()
    } else {
        "jackpot".to_string()
    };
    info!("connecting to {host}:{port} as '{}'", args.user);
    let client = Arc::new(StratumClient::connect(&host, port, &args.user, &pass)?);

    #[allow(unused_mut)]
    let mut specs = backend_specs(&args, &gpu_devices)?;
    // CPU mining via OpenCL: if requested and an OpenCL CPU device (e.g. PoCL) is
    // available, run the CPU solve through the OpenCL backend on that device
    // (one worker) instead of the native AVX2 per-core groups.
    #[cfg(feature = "gpu")]
    let cpu_opencl = args.cpu_mining.then(gpu::cpu_device_index).flatten();
    #[cfg(not(feature = "gpu"))]
    let cpu_opencl: Option<usize> = None;
    #[cfg(feature = "gpu")]
    if let Some(idx) = cpu_opencl {
        info!("CPU mining via OpenCL device {idx} (CPU); native AVX2 CPU groups stay off");
        specs.push(BackendSpec::Gpu(idx));
    }
    info!("launching {} worker(s)", specs.len());

    let running = Arc::new(AtomicBool::new(true));
    {
        let r = running.clone();
        ctrlc::set_handler(move || {
            info!("interrupt received, shutting down...");
            r.store(false, Ordering::Relaxed);
        })
        .context("failed to install Ctrl-C handler")?;
    }

    let job_timeout = (args.job_timeout > 0).then(|| std::time::Duration::from_secs(args.job_timeout));
    // CPU mining as toggleable rows of --cpu-group-size cores (over the cores
    // selected by --cpu-cores, default all), shown below the device table and
    // mined alongside the backend. Starts enabled only with --cpu-mining.
    let logical_cpus = num_cpus::get();
    let cpu_cores = match args.cpu_cores.as_deref() {
        Some(spec) => parse_core_spec(spec, logical_cpus)?,
        None => (0..logical_cpus).collect(),
    };
    // AVX2 per-core groups start enabled only when CPU mining is on AND we're not
    // already running CPU work through OpenCL.
    let cpu_mining = cpu_groups::CpuMining::new(cpu_cores, args.cpu_group_size, args.cpu_mining && cpu_opencl.is_none());
    // 0 selects the exact (unclamped) CPU solver; any other value clamps.
    let cpu_clamp = (args.cpu_clamp != 0).then_some(args.cpu_clamp);
    // Shared per-GPU live hardware controls (adjusted from the TUI, applied by
    // each device's worker); the selection also spans the CPU-group rows.
    let controls = controls::Controls::new(
        specs.len(),
        cpu_mining.groups().len(),
        args.gpu_clock_offset.unwrap_or(0),
        args.mem_clock_offset.unwrap_or(0),
        args.power_limit.unwrap_or(0),
        args.unlock_controls,
    );
    miner::run(client, specs, running, job_timeout, tui, format!("{host}:{port}"), controls, cpu_mining, cpu_clamp, args.control_port)
}

/// Which GPU backend the user selected.
enum BackendKind {
    Cpu,
    /// Each physical card on its native backend (NVIDIA→CUDA, others→OpenCL).
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    Mixed,
    #[cfg(feature = "gpu")]
    OpenCl,
    #[cfg(feature = "cuda")]
    Cuda,
}

/// Resolve the backend kind from `--cpu` / `--backend` and compiled features.
fn backend_kind(args: &Args) -> Result<BackendKind> {
    if args.cpu {
        return Ok(BackendKind::Cpu);
    }
    // --force-opencl disables CUDA regardless of --backend.
    if args.force_opencl {
        #[cfg(feature = "gpu")]
        {
            return Ok(BackendKind::OpenCl);
        }
        #[cfg(not(feature = "gpu"))]
        {
            return Err(anyhow!("--force-opencl needs the OpenCL backend compiled in (build with --features gpu)"));
        }
    }
    match args.backend.to_ascii_lowercase().as_str() {
        "mixed" => {
            // Each card on its native backend; falls back to whatever single GPU
            // backend is compiled, or to CPU when none is.
            #[cfg(any(feature = "gpu", feature = "cuda"))]
            {
                Ok(BackendKind::Mixed)
            }
            #[cfg(not(any(feature = "gpu", feature = "cuda")))]
            Ok(BackendKind::Cpu)
        }
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                Ok(BackendKind::Cuda)
            }
            #[cfg(not(feature = "cuda"))]
            Err(anyhow!("CUDA backend not compiled in (build with --features cuda)"))
        }
        "opencl" | "" => {
            #[cfg(feature = "gpu")]
            {
                Ok(BackendKind::OpenCl)
            }
            #[cfg(not(feature = "gpu"))]
            Ok(BackendKind::Cpu)
        }
        other => Err(anyhow!("unknown --backend '{other}' (expected mixed, opencl, or cuda)")),
    }
}

/// Parse a `--cpu-cores` spec into a sorted, de-duplicated list of logical core
/// indices. Accepts "all", single indices, and inclusive ranges, comma-joined:
/// "0-7", "0,2,4,6", "0-3,8,10-11". Every index must be `< available`.
fn parse_core_spec(spec: &str, available: usize) -> Result<Vec<usize>> {
    let spec = spec.trim();
    if spec.eq_ignore_ascii_case("all") {
        return Ok((0..available).collect());
    }
    let mut set = std::collections::BTreeSet::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        match part.split_once('-') {
            Some((a, b)) => {
                let a: usize = a.trim().parse().map_err(|_| anyhow!("bad core range '{part}'"))?;
                let b: usize = b.trim().parse().map_err(|_| anyhow!("bad core range '{part}'"))?;
                if a > b {
                    return Err(anyhow!("core range '{part}' is reversed"));
                }
                set.extend(a..=b);
            }
            None => {
                set.insert(part.parse::<usize>().map_err(|_| anyhow!("bad core index '{part}'"))?);
            }
        }
    }
    let max = *set.iter().next_back().ok_or_else(|| anyhow!("--cpu-cores selected no cores"))?;
    if max >= available {
        return Err(anyhow!(
            "--cpu-cores includes core {max}, but only {available} logical CPUs are available (0..={})",
            available - 1
        ));
    }
    Ok(set.into_iter().collect())
}

/// Determine the solver workers to launch from the CLI flags. `gpu_devices` are
/// the `[[gpu]]` config tables; a card whose table sets `backend` runs on that
/// backend instead of the global `--backend` default, so a single run can mix
/// CUDA and OpenCL cards.
#[allow(unused_variables)]
fn backend_specs(args: &Args, gpu_devices: &[GpuDeviceCfg]) -> Result<Vec<BackendSpec>> {
    let default = backend_kind(args)?;
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        // The default backend fixes the device enumeration `--devices` indexes
        // into; per-card overrides then flip individual cards.
        let (available, default_cuda) = match default {
            BackendKind::Cpu => {
                let clamp = (args.cpu_clamp != 0).then_some(args.cpu_clamp);
                return Ok(vec![BackendSpec::Cpu(clamp)]);
            }
            // Mixed builds its own unified list (each card on its native backend).
            BackendKind::Mixed => return mixed_specs(args),
            #[cfg(feature = "cuda")]
            BackendKind::Cuda => (cuda::device_count()?, true),
            #[cfg(feature = "gpu")]
            BackendKind::OpenCl => (gpu::list_devices()?.len(), false),
        };
        let devices = parse_devices(&args.devices, available)?;
        let mut specs = Vec::with_capacity(devices.len());
        for idx in devices {
            let cuda = match gpu_devices.iter().find(|g| g.index == idx).and_then(|g| g.backend.as_deref()) {
                Some(b) if b.eq_ignore_ascii_case("cuda") => true,
                Some(b) if b.eq_ignore_ascii_case("opencl") => false,
                Some(other) => {
                    return Err(anyhow!("device {idx}: unknown backend '{other}' (expected cuda or opencl)"))
                }
                None => default_cuda,
            };
            specs.push(gpu_spec(idx, cuda)?);
        }
        Ok(specs)
    }
    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
    {
        // 0 selects the exact (unclamped) solver; any other value clamps.
        let clamp = (args.cpu_clamp != 0).then_some(args.cpu_clamp);
        Ok(vec![BackendSpec::Cpu(clamp)])
    }
}

/// The unified device list for the `mixed` backend, as `(label, spec)`: each
/// physical GPU on its native backend, with no card mined twice. NVIDIA cards go
/// to CUDA (listed first); the remaining OpenCL devices (AMD/Intel, plus NVIDIA
/// when CUDA is unavailable) go to OpenCL. Shared by [`mixed_specs`] and
/// [`list_devices`]; `--devices` indexes into this list.
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn mixed_plan() -> Vec<(String, BackendSpec)> {
    /// Drop a leading `"[<n>] "` index prefix from a backend's device label, so
    /// the mixed list shows its own single index instead of two.
    fn strip_index(label: &str) -> &str {
        label
            .strip_prefix('[')
            .and_then(|s| s.split_once("] "))
            .map(|(_, rest)| rest)
            .unwrap_or(label)
    }

    #[allow(unused_mut)]
    let mut plan: Vec<(String, BackendSpec)> = Vec::new();

    // NVIDIA cards via CUDA, when the backend is compiled and the driver loads.
    #[cfg(feature = "cuda")]
    let cuda_has_nvidia = {
        let names = cuda::list_devices().unwrap_or_default();
        for (i, label) in names.iter().enumerate() {
            plan.push((format!("{} (CUDA)", strip_index(label)), BackendSpec::Cuda(i)));
        }
        !names.is_empty()
    };
    #[cfg(not(feature = "cuda"))]
    let cuda_has_nvidia = false;

    // Remaining OpenCL cards via OpenCL; skip NVIDIA ones already on CUDA.
    #[cfg(feature = "gpu")]
    {
        let names = gpu::list_devices().unwrap_or_default();
        let nvidia = gpu::device_is_nvidia();
        for (j, label) in names.iter().enumerate() {
            if nvidia.get(j).copied().unwrap_or(false) && cuda_has_nvidia {
                continue;
            }
            plan.push((format!("{} (OpenCL)", strip_index(label)), BackendSpec::Gpu(j)));
        }
    }
    // `cuda_has_nvidia` is only consumed by the OpenCL branch above.
    #[cfg(not(feature = "gpu"))]
    let _ = cuda_has_nvidia;

    plan
}

/// Build the worker list for `--backend mixed`: each card on its native backend.
/// `--devices` selects into [`mixed_plan`]'s unified list.
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn mixed_specs(args: &Args) -> Result<Vec<BackendSpec>> {
    let plan = mixed_plan();
    if plan.is_empty() {
        return Err(anyhow!(
            "no GPUs found for the mixed backend — none detected via CUDA or OpenCL"
        ));
    }
    let selected = parse_devices(&args.devices, plan.len())?;
    Ok(selected.into_iter().map(|i| plan[i].1).collect())
}

/// Build a single GPU worker spec for `idx`, choosing CUDA or OpenCL, erroring if
/// the requested backend wasn't compiled in.
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn gpu_spec(idx: usize, cuda: bool) -> Result<BackendSpec> {
    if cuda {
        #[cfg(feature = "cuda")]
        {
            Ok(BackendSpec::Cuda(idx))
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(anyhow!("device {idx} requests the CUDA backend, but it isn't compiled in (build with --features cuda)"))
        }
    } else {
        #[cfg(feature = "gpu")]
        {
            Ok(BackendSpec::Gpu(idx))
        }
        #[cfg(not(feature = "gpu"))]
        {
            Err(anyhow!("device {idx} requests the OpenCL backend, but it isn't compiled in (build with --features gpu)"))
        }
    }
}

/// Parse a `--devices` value: "all" (use `available`) or a comma-separated list.
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn parse_devices(spec: &str, available: usize) -> Result<Vec<usize>> {
    if spec.trim().eq_ignore_ascii_case("all") {
        return Ok((0..available).collect());
    }
    let devices: Vec<usize> = spec
        .split(',')
        .map(|s| s.trim().parse::<usize>().map_err(|_| anyhow!("bad device index '{s}'")))
        .collect::<Result<_>>()?;
    if devices.is_empty() {
        return Err(anyhow!("no devices selected"));
    }
    Ok(devices)
}


/// Build a JSON description of the detected devices for the GUI config tool:
/// `{"opencl":["<name>",...],"cuda":["<name>",...]}`. A backend not compiled in,
/// or with no runtime/devices, yields an empty list (so the tool can warn).
fn devices_json() -> String {
    #[cfg(feature = "gpu")]
    let opencl: Vec<String> = gpu::list_devices().unwrap_or_default();
    #[cfg(not(feature = "gpu"))]
    let opencl: Vec<String> = Vec::new();
    #[cfg(feature = "cuda")]
    let cuda: Vec<String> = cuda::list_devices().unwrap_or_default();
    #[cfg(not(feature = "cuda"))]
    let cuda: Vec<String> = Vec::new();
    // Flat OpenCL index of a CPU device (e.g. PoCL), used for CPU-via-OpenCL.
    #[cfg(feature = "gpu")]
    let opencl_cpu: Option<usize> = gpu::cpu_device_index();
    #[cfg(not(feature = "gpu"))]
    let opencl_cpu: Option<usize> = None;
    serde_json::json!({ "opencl": opencl, "cuda": cuda, "opencl_cpu_index": opencl_cpu }).to_string()
}

/// Print the GPU devices visible to each compiled backend.
fn list_devices() {
    #[cfg(feature = "gpu")]
    match gpu::list_devices() {
        Ok(devs) if !devs.is_empty() => {
            println!("OpenCL devices (--backend opencl):");
            for d in devs {
                println!("  {d}");
            }
        }
        Ok(_) => println!("no OpenCL devices found"),
        Err(e) => println!("error listing OpenCL devices: {e}"),
    }
    #[cfg(feature = "cuda")]
    match cuda::list_devices() {
        Ok(devs) if !devs.is_empty() => {
            println!("CUDA devices (--backend cuda):");
            for d in devs {
                println!("  {d}");
            }
        }
        Ok(_) => println!("no CUDA devices found"),
        Err(e) => println!("error listing CUDA devices: {e}"),
    }
    // What the default `mixed` backend will mine, and the indices `--devices`
    // selects from in that mode.
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        let plan = mixed_plan();
        if !plan.is_empty() {
            println!("\nMixed backend (--backend mixed, the default) — `--devices` indexes this list:");
            for (i, (label, _)) in plan.iter().enumerate() {
                println!("  [{i}] {label}");
            }
        }
    }
    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
    println!("built without GPU support (rebuild with the `gpu` or `cuda` feature)");
}

/// Parse `stratum+tcp://host:port`, `tcp://host:port`, or `host:port`. When the
/// URL omits `:port`, fall back to `default_port` (from `--port`).
fn parse_url(url: &str, default_port: Option<u16>) -> Result<(String, u16)> {
    let trimmed = url
        .strip_prefix("stratum+tcp://")
        .or_else(|| url.strip_prefix("tcp://"))
        .unwrap_or(url);
    match trimmed.rsplit_once(':') {
        Some((host, port)) => {
            let port: u16 = port.parse().with_context(|| format!("bad port in {url}"))?;
            Ok((host.to_string(), port))
        }
        None => {
            let port = default_port.ok_or_else(|| {
                anyhow!("URL '{url}' has no port; include one (host:port) or pass --port")
            })?;
            Ok((trimmed.to_string(), port))
        }
    }
}

/// Run quick correctness checks: encode/decode round-trip and an end-to-end
/// solve-then-verify against a fixed header.
fn selftest(gpu_device: usize) -> Result<()> {
    info!("running self-tests (this performs one full solve and may take a while)...");

    // A deterministic header full of a fixed byte pattern, with the nonce tail
    // zeroed so the CUDA backend (which assumes header[128..135] == 0) is
    // actually exercised by the GPU/CUDA comparison below.
    let mut header = vec![0x42u8; params::HEADER_LEN];
    cuda_compatible(&mut header);
    let base = blake::base_state(&header);

    // Use the clamped solver (clamp 32, matching the GPU's fixed bucket slots):
    // the unclamped path explodes on dense 192,7 headers, and the GPU it is
    // compared against also bounds its buckets.
    let solutions = equihash::solve_with(&header, Some(32));
    info!("CPU found {} solution(s) for the test header", solutions.len());

    for (i, sol) in solutions.iter().enumerate() {
        let ok = equihash::is_valid_solution(&base, sol);
        let packed = equihash::indices_to_solution(sol);
        let back = equihash::solution_to_indices(&packed);
        let roundtrip = &back == sol;
        info!(
            "  CPU solution {i}: valid={ok}, encode_roundtrip={roundtrip}, bytes={}",
            packed.len()
        );
        if !ok || !roundtrip {
            return Err(anyhow!("self-test failed on solution {i}"));
        }
    }

    // Validate the GPU solver against the CPU: it must produce only valid
    // solutions, all of which are a subset of the CPU's exhaustive set.
    #[cfg(feature = "gpu")]
    {
        info!("initialising GPU solver for comparison (OpenCL device {gpu_device})...");
        let solver = gpu::GpuSolver::new(gpu_device)
            .with_context(|| format!("init OpenCL device {gpu_device}"))?;

        // Spot-check the BLAKE2b kernel against the CPU reference. The AMD kernel
        // buckets its round-0 output instead of exposing per-index digests, so
        // the probe is skipped there (the solve-vs-CPU check below still runs).
        if solver.supports_blake_probe() {
            let outputs = solver.hash_all(&header)?;
            let step = params::BLAKE_CALLS / 64;
            for k in 0..64 {
                let g = (k * step) as u32;
                let cpu = blake::generate_hash(&base, g);
                let off = g as usize * params::HASH_OUTPUT;
                if cpu != outputs[off..off + params::HASH_OUTPUT] {
                    return Err(anyhow!("GPU BLAKE2b mismatch at g={g}"));
                }
            }
            info!("GPU BLAKE2b kernel matches CPU");
        } else {
            info!("skipping BLAKE2b kernel probe (AMD kernel buckets round-0 output)");
        }

        let gpu_solutions = solver.solve(&header)?;
        info!("GPU found {} valid solution(s)", gpu_solutions.len());

        let cpu_set: std::collections::HashSet<Vec<u32>> =
            solutions.iter().map(|s| sorted(s)).collect();
        for sol in &gpu_solutions {
            if !equihash::is_valid_solution(&base, sol) {
                return Err(anyhow!("GPU returned an invalid solution"));
            }
            if !cpu_set.contains(&sorted(sol)) {
                return Err(anyhow!("GPU solution not found by the CPU reference"));
            }
        }
        info!(
            "GPU solver verified: {}/{} of the CPU solutions recovered",
            gpu_solutions.len(),
            solutions.len()
        );
    }

    // Validate the CUDA solver the same way (subset of the CPU's solutions).
    #[cfg(feature = "cuda")]
    {
        info!("initialising CUDA solver for comparison...");
        let solver = cuda::CudaSolver::new(0).context("init CUDA device 0")?;
        let cuda_solutions = solver.solve(&header)?;
        info!("CUDA found {} valid solution(s)", cuda_solutions.len());

        let cpu_set: std::collections::HashSet<Vec<u32>> =
            solutions.iter().map(|s| sorted(s)).collect();
        for sol in &cuda_solutions {
            if !equihash::is_valid_solution(&base, sol) {
                return Err(anyhow!("CUDA returned an invalid solution"));
            }
            if !cpu_set.contains(&sorted(sol)) {
                return Err(anyhow!("CUDA solution not found by the CPU reference"));
            }
        }
        info!(
            "CUDA solver verified: {}/{} of the CPU solutions recovered",
            cuda_solutions.len(),
            solutions.len()
        );
    }

    info!("self-tests passed");
    Ok(())
}

/// Sorted copy of an index list, for set comparison.
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn sorted(v: &[u32]) -> Vec<u32> {
    let mut s = v.to_vec();
    s.sort_unstable();
    s
}

/// Benchmark the configured backends concurrently (one thread each), reporting
/// per-worker and aggregate throughput. With multiple GPUs this measures real
/// concurrent multi-device performance.
fn benchmark(specs: Vec<BackendSpec>, runs: usize) -> Result<()> {
    use std::time::Instant;
    info!("benchmarking {runs} solve(s) per worker across {} worker(s)", specs.len());

    let start = Instant::now();
    let mut handles = Vec::new();
    for (id, spec) in specs.into_iter().enumerate() {
        handles.push(std::thread::spawn(move || -> Result<(usize, f64)> {
            let backend = spec.build()?;
            backend.solve(&pseudo_header(id as u64))?; // warm up (excluded)
            let t = Instant::now();
            let mut sols = 0usize;
            for i in 0..runs {
                // Distinct nonce space per worker.
                let seed = ((id as u64) << 40) | (i as u64 + 1);
                sols += backend.solve(&pseudo_header(seed))?.len();
            }
            Ok((sols, t.elapsed().as_secs_f64()))
        }));
    }

    // Aggregate by summing per-worker steady-state rates (excludes warm-up).
    let mut agg_sols = 0.0;
    let mut workers = 0usize;
    for h in handles {
        match h.join().unwrap() {
            Ok((sols, dt)) => {
                let sol_s = sols as f64 / dt;
                info!(
                    "  worker {workers}: {sol_s:.2} Sol/s ({:.0} ms/solve), {sols} solutions",
                    1000.0 * dt / runs as f64
                );
                agg_sols += sol_s;
                workers += 1;
            }
            Err(e) => warn!("  worker failed: {e}"),
        }
    }

    let _ = start;
    info!("aggregate: {agg_sols:.1} Sol/s across {workers} worker(s)");
    Ok(())
}

/// A deterministic pseudo-random 140-byte header for benchmarking.
fn pseudo_header(seed: u64) -> Vec<u8> {
    let mut header = vec![0u8; params::HEADER_LEN];
    let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    for b in header.iter_mut() {
        // xorshift64*
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        *b = (x.wrapping_mul(0x2545F4914F6CDD1D) >> 33) as u8;
    }
    cuda_compatible(&mut header);
    header
}

/// Zero header bytes [128..135] (nonce[20..27]). The CUDA fatbin replay injects
/// only the midstate over header[0..128] and the 4 tail bytes [136..139]; it
/// hard-codes [128..135] = 0, so a header with those bytes set makes the CUDA
/// backend find nothing. Real pool nonces keep them zero (nonce = nonce1 ||
/// counter || zeros); synthetic test headers must do the same to exercise CUDA.
fn cuda_compatible(header: &mut [u8]) {
    for b in &mut header[128..136] {
        *b = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_core_spec, parse_url};

    #[test]
    fn core_spec_parsing() {
        // "all" expands to every core.
        assert_eq!(parse_core_spec("all", 4).unwrap(), vec![0, 1, 2, 3]);
        // Ranges, lists, and a mix; result is sorted + de-duplicated.
        assert_eq!(parse_core_spec("0-3", 8).unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(parse_core_spec("0,2,4,6", 8).unwrap(), vec![0, 2, 4, 6]);
        assert_eq!(parse_core_spec("0-2,8,10-11", 12).unwrap(), vec![0, 1, 2, 8, 10, 11]);
        assert_eq!(parse_core_spec("3, 3 , 1-2", 8).unwrap(), vec![1, 2, 3]);
        // Out-of-range, reversed, empty, and garbage all error.
        assert!(parse_core_spec("0-8", 8).is_err()); // core 8 >= 8
        assert!(parse_core_spec("5-1", 8).is_err());
        assert!(parse_core_spec("", 8).is_err());
        assert!(parse_core_spec("x", 8).is_err());
    }

    #[test]
    fn url_port_parsing() {
        // Explicit port in the URL is used as-is (scheme stripped).
        assert_eq!(parse_url("stratum+tcp://pool.example:3032", None).unwrap(), ("pool.example".to_string(), 3032));
        // URL port wins over --port when both are present.
        assert_eq!(parse_url("tcp://1.2.3.4:1234", Some(9999)).unwrap(), ("1.2.3.4".to_string(), 1234));
        // No port in URL -> fall back to --port.
        assert_eq!(parse_url("pool.example", Some(3032)).unwrap(), ("pool.example".to_string(), 3032));
        // No port and no --port -> error.
        assert!(parse_url("pool.example", None).is_err());
        // A colon with a non-numeric suffix is still an error.
        assert!(parse_url("host:notaport", Some(3032)).is_err());
    }
}
