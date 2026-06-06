//! A small native GUI for building a jackpotminer TOML config (loadable with
//! `--config`). It's an optional, standalone binary so the miner never depends
//! on a GUI toolkit.
//!
//! Build & run:
//!   cargo run --release --features config-gui --bin jackpotminer-config
//!
//! Edit the fields, Save to a `.toml`, then run:  `jackpotminer --config mine.toml`.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::time::Duration;

use eframe::egui;
use serde_json::{json, Value};

// Direct OpenCL/CUDA probing, compiled in when the SDK features are enabled.
#[path = "gpu_probe.rs"]
mod gpu_probe;

fn main() -> eframe::Result<()> {
    // Headless self-check: `jackpotminer-config --probe` prints what detection
    // sees and exits (no window) — handy for diagnosing "no devices populated".
    if std::env::args().any(|a| a == "--probe") {
        println!("built-in GPU SDK: {}", gpu_probe::HAS_SDK);
        println!("CUDA devices:   {:?}", gpu_probe::cuda());
        println!("OpenCL devices: {:?}", gpu_probe::opencl());
        println!(
            "logical CPUs:   {}",
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0)
        );
        std::process::exit(0);
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([560.0, 760.0]),
        ..Default::default()
    };
    eframe::run_native(
        "jackpotminer config",
        options,
        Box::new(|_cc| Ok(Box::new(ConfigApp::default()) as Box<dyn eframe::App>)),
    )
}

#[derive(PartialEq, Clone, Copy, Default)]
enum Payout {
    #[default]
    Jackpot,
    JackpotPercent,
    Pplns,
    Solo,
}

#[derive(PartialEq, Clone, Copy, Default)]
enum Backend {
    #[default]
    Cuda,
    OpenCl,
}

/// The TOML shape `--config` reads (kebab-case keys, only set fields serialized).
#[derive(serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "kebab-case", default, deny_unknown_fields)]
struct Conf {
    #[serde(skip_serializing_if = "Option::is_none")]
    url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    port: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pass: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    no_jackpot: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    solo: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    jackpot: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    job_timeout: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    control_port: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    threads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cpu: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cpu_mining: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cpu_cores: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cpu_group_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cpu_clamp: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    devices: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    force_opencl: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    no_tui: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    no_gpu_tune: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    unlock_controls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    auto_tune: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    power_limit: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_clock: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mem_clock: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_clock_offset: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mem_clock_offset: Option<i32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    gpu: Vec<GpuDeviceCfg>,
}

/// One `[[gpu]]` per-device table in the config file: which backend drives this
/// card plus any tuning overrides.
#[derive(serde::Serialize, serde::Deserialize, Default, Clone)]
#[serde(rename_all = "kebab-case", default, deny_unknown_fields)]
struct GpuDeviceCfg {
    index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    power_limit: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_clock: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mem_clock: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_clock_offset: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mem_clock_offset: Option<i32>,
}

/// Editable per-card row in the GUI: the backend that drives the card plus
/// tuning overrides (numbers as strings; blank = unset). Tuning applies only to
/// CUDA cards (NVML).
#[derive(Default, Clone)]
struct DeviceTuneForm {
    label: String,
    backend: Backend,
    power_limit: String,
    gpu_clock_offset: String,
    mem_clock_offset: String,
    // Carried through from a loaded config so a save round-trips them.
    keep_gpu_clock: Option<u32>,
    keep_mem_clock: Option<u32>,
}

/// The editor state: widget-friendly types (numbers as strings so a field can be
/// left blank = "unset"), converted to/from [`Conf`] on save/load.
struct ConfigApp {
    path: String,
    // pool
    url: String,
    port: String,
    user: String,
    // payout
    payout: Payout,
    jackpot_percent: String,
    // gpu — the backend is chosen per-card; this is the default a card uses
    // until it's overridden in the per-card table, and the enumeration the
    // miner's `--devices` indexes into.
    default_backend: Backend,
    devices: String,
    // detected devices (from `jackpotminer --devices-json`)
    detected_cuda: Vec<String>,
    detected_opencl: Vec<String>,
    device_sel: Vec<bool>,
    gpu_status: String,
    gpu_status_err: bool,
    dev_tune: Vec<DeviceTuneForm>,
    // misc
    threads: String,
    job_timeout: String,
    control_port: String,
    no_tui: bool,
    // ui state
    status: String,
    status_err: bool,
    preview: String,
    // live control of a running miner
    live_host: String,
    live_port: String,
    live_status: String,
    live_err: bool,
    live_connected: bool,
    live_unlocked: bool,
    live_group_size: String,
    live_devices: Vec<LiveDevice>,
    live_rows: Vec<LiveRow>,
    // startup detection
    first_frame: bool,
}

impl Default for ConfigApp {
    fn default() -> Self {
        Self {
            path: "mine.toml".into(),
            url: "stratum+tcp://zcl.pool.example:3032".into(),
            port: String::new(),
            user: String::new(),
            payout: Payout::Jackpot,
            jackpot_percent: "50".into(),
            default_backend: Backend::Cuda,
            devices: "all".into(),
            detected_cuda: Vec::new(),
            detected_opencl: Vec::new(),
            device_sel: Vec::new(),
            gpu_status: String::new(),
            gpu_status_err: false,
            dev_tune: Vec::new(),
            threads: String::new(),
            job_timeout: "300".into(),
            control_port: String::new(),
            no_tui: false,
            status: String::new(),
            status_err: false,
            preview: String::new(),
            live_host: "127.0.0.1".into(),
            live_port: String::new(),
            live_status: String::new(),
            live_err: false,
            live_connected: false,
            live_unlocked: false,
            live_group_size: String::new(),
            live_devices: Vec::new(),
            live_rows: Vec::new(),
            first_frame: true,
        }
    }
}

/// Parse a possibly-empty field into `Option<T>` (blank = `None`), with a
/// human-readable error naming the field.
fn parse_opt<T: std::str::FromStr>(s: &str, field: &str) -> Result<Option<T>, String> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(None);
    }
    s.parse::<T>().map(Some).map_err(|_| format!("invalid {field}: '{s}'"))
}

fn num_str<T: ToString>(o: Option<T>) -> String {
    o.map(|v| v.to_string()).unwrap_or_default()
}

/// Truncate a string to `max` chars with an ellipsis.
fn trunc(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max.saturating_sub(1)).collect::<String>() + "…"
    }
}

/// Parse a `--devices` spec into a per-card checkbox vector of length `n`
/// (blank/"all" selects everything; out-of-range indices are ignored).
fn parse_devices_spec(spec: &str, n: usize) -> Vec<bool> {
    let s = spec.trim();
    if s.is_empty() || s.eq_ignore_ascii_case("all") {
        return vec![true; n];
    }
    let mut sel = vec![false; n];
    for part in s.split(',') {
        if let Ok(c) = part.trim().parse::<usize>() {
            if c < n {
                sel[c] = true;
            }
        }
    }
    sel
}

/// Devices reported by `jackpotminer --devices-json`.
#[derive(serde::Deserialize, Default)]
struct DevList {
    #[serde(default)]
    opencl: Vec<String>,
    #[serde(default)]
    cuda: Vec<String>,
}

/// One device row retrieved from a running miner's control server, with the
/// editable clock/power fields kept as strings.
#[derive(Default)]
struct LiveDevice {
    index: u64,
    label: String,
    enabled: bool,
    power_w: String,
    core_off: String,
    mem_off: String,
    watts: f64,
    temp_c: u64,
    shares: u64,
}

/// One CPU-row from a running miner's control server.
#[derive(Default)]
struct LiveRow {
    index: u64,
    label: String,
    enabled: bool,
    shares: u64,
}

/// Send one request to a control server and read its single-line JSON reply.
fn control_request(addr: &str, req: &Value) -> Result<Value, String> {
    let mut stream = TcpStream::connect(addr).map_err(|e| format!("connect {addr}: {e}"))?;
    stream.set_read_timeout(Some(Duration::from_secs(3))).ok();
    stream.set_write_timeout(Some(Duration::from_secs(3))).ok();
    let mut msg = req.to_string();
    msg.push('\n');
    stream.write_all(msg.as_bytes()).map_err(|e| e.to_string())?;
    let mut reader = BufReader::new(stream);
    let mut resp = String::new();
    reader.read_line(&mut resp).map_err(|e| e.to_string())?;
    serde_json::from_str::<Value>(resp.trim()).map_err(|e| format!("bad reply: {e}"))
}

/// A numeric JSON field rendered as an editable string (handles signed/unsigned).
fn num_field(v: &Value, key: &str) -> String {
    v.get(key)
        .and_then(|n| n.as_i64().map(|x| x.to_string()).or_else(|| n.as_u64().map(|x| x.to_string())))
        .unwrap_or_default()
}

fn parse_live_device(v: &Value) -> LiveDevice {
    LiveDevice {
        index: v.get("index").and_then(Value::as_u64).unwrap_or(0),
        label: v.get("label").and_then(Value::as_str).unwrap_or("").to_string(),
        enabled: v.get("enabled").and_then(Value::as_bool).unwrap_or(false),
        power_w: num_field(v, "power_w"),
        core_off: num_field(v, "core_off"),
        mem_off: num_field(v, "mem_off"),
        watts: v.get("watts").and_then(Value::as_f64).unwrap_or(0.0),
        temp_c: v.get("temp_c").and_then(Value::as_u64).unwrap_or(0),
        shares: v.get("shares").and_then(Value::as_u64).unwrap_or(0),
    }
}

fn parse_live_row(v: &Value) -> LiveRow {
    LiveRow {
        index: v.get("index").and_then(Value::as_u64).unwrap_or(0),
        label: v.get("label").and_then(Value::as_str).unwrap_or("").to_string(),
        enabled: v.get("enabled").and_then(Value::as_bool).unwrap_or(false),
        shares: v.get("shares").and_then(Value::as_u64).unwrap_or(0),
    }
}

/// Locate the miner binary: prefer a `jackpotminer` next to this config tool,
/// otherwise fall back to one on `PATH`.
fn miner_binary() -> std::path::PathBuf {
    let name = if cfg!(windows) { "jackpotminer.exe" } else { "jackpotminer" };
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let cand = dir.join(name);
            if cand.exists() {
                return cand;
            }
        }
    }
    std::path::PathBuf::from(name)
}

fn opt_str(s: &str) -> Option<String> {
    let s = s.trim();
    (!s.is_empty()).then(|| s.to_string())
}

/// The config-file spelling of a backend.
fn backend_name(b: Backend) -> &'static str {
    match b {
        Backend::Cuda => "cuda",
        Backend::OpenCl => "opencl",
    }
}

impl ConfigApp {
    /// Build the serializable config from the form (validating numbers).
    fn to_conf(&self) -> Result<Conf, String> {
        let default_backend = self.default_backend;
        let mut c = Conf {
            url: opt_str(&self.url),
            port: parse_opt(&self.port, "port")?,
            user: opt_str(&self.user),
            // The default backend fixes the device enumeration `devices` indexes
            // into; per-card `[[gpu]] backend = …` overrides flip individual cards.
            backend: Some(backend_name(default_backend).into()),
            devices: opt_str(&self.devices),
            threads: parse_opt(&self.threads, "threads")?,
            job_timeout: parse_opt(&self.job_timeout, "job-timeout")?,
            control_port: parse_opt(&self.control_port, "control-port")?,
            ..Default::default()
        };
        match self.payout {
            Payout::Jackpot => {}
            Payout::JackpotPercent => c.jackpot = parse_opt(&self.jackpot_percent, "jackpot")?,
            Payout::Pplns => c.no_jackpot = Some(true),
            Payout::Solo => c.solo = Some(true),
        }
        // Booleans only serialize when true (omitted = the default).
        c.no_tui = self.no_tui.then_some(true);

        // Per-card tables -> [[gpu]]: emit one when a card's backend differs from
        // the default or it carries CUDA tuning. Only selected (checked) cards;
        // tuning is NVML-only, so it's dropped for OpenCL cards.
        for (i, f) in self.dev_tune.iter().enumerate() {
            if !self.device_sel.get(i).copied().unwrap_or(true) {
                continue;
            }
            let cuda_card = f.backend == Backend::Cuda;
            let power_limit =
                cuda_card.then(|| parse_opt::<u32>(&f.power_limit, "device power-limit")).transpose()?.flatten();
            let gpu_clock_offset =
                cuda_card.then(|| parse_opt::<i32>(&f.gpu_clock_offset, "device core offset")).transpose()?.flatten();
            let mem_clock_offset =
                cuda_card.then(|| parse_opt::<i32>(&f.mem_clock_offset, "device mem offset")).transpose()?.flatten();
            let keep_gpu_clock = cuda_card.then_some(f.keep_gpu_clock).flatten();
            let keep_mem_clock = cuda_card.then_some(f.keep_mem_clock).flatten();
            let backend_differs = f.backend != default_backend;
            let has_tune = power_limit.is_some()
                || gpu_clock_offset.is_some()
                || mem_clock_offset.is_some()
                || keep_gpu_clock.is_some()
                || keep_mem_clock.is_some();
            if backend_differs || has_tune {
                c.gpu.push(GpuDeviceCfg {
                    index: i,
                    backend: backend_differs.then(|| backend_name(f.backend).to_string()),
                    power_limit,
                    gpu_clock: keep_gpu_clock,
                    mem_clock: keep_mem_clock,
                    gpu_clock_offset,
                    mem_clock_offset,
                });
            }
        }
        Ok(c)
    }

    /// Populate the form from a loaded config.
    fn from_conf(&mut self, c: Conf) {
        self.url = c.url.unwrap_or_default();
        self.port = num_str(c.port);
        self.user = c.user.unwrap_or_default();
        self.payout = if c.solo == Some(true) {
            Payout::Solo
        } else if let Some(j) = c.jackpot {
            self.jackpot_percent = j.to_string();
            Payout::JackpotPercent
        } else if c.no_jackpot == Some(true) {
            Payout::Pplns
        } else {
            Payout::Jackpot
        };
        self.default_backend = match c.backend.as_deref() {
            Some("opencl") => Backend::OpenCl,
            _ => Backend::Cuda,
        };
        self.devices = c.devices.unwrap_or_else(|| "all".into());
        // Reset every card to the default backend with no tuning, then apply the
        // per-card overrides from the [[gpu]] tables.
        for f in &mut self.dev_tune {
            f.backend = self.default_backend;
            f.power_limit.clear();
            f.gpu_clock_offset.clear();
            f.mem_clock_offset.clear();
            f.keep_gpu_clock = None;
            f.keep_mem_clock = None;
        }
        if let Some(max) = c.gpu.iter().map(|g| g.index).max() {
            if self.dev_tune.len() <= max {
                self.dev_tune
                    .resize(max + 1, DeviceTuneForm { backend: self.default_backend, ..Default::default() });
            }
        }
        for g in &c.gpu {
            if let Some(f) = self.dev_tune.get_mut(g.index) {
                if let Some(b) = g.backend.as_deref() {
                    f.backend = if b.eq_ignore_ascii_case("opencl") { Backend::OpenCl } else { Backend::Cuda };
                }
                f.power_limit = num_str(g.power_limit);
                f.gpu_clock_offset = num_str(g.gpu_clock_offset);
                f.mem_clock_offset = num_str(g.mem_clock_offset);
                f.keep_gpu_clock = g.gpu_clock;
                f.keep_mem_clock = g.mem_clock;
            }
        }
        // Mirror the devices spec onto the per-card on/off checkboxes.
        self.device_sel = parse_devices_spec(&self.devices, self.cards().len());
        self.threads = num_str(c.threads);
        self.job_timeout = c.job_timeout.map(|v| v.to_string()).unwrap_or_else(|| "300".into());
        self.control_port = num_str(c.control_port);
        if self.live_port.trim().is_empty() {
            self.live_port = num_str(c.control_port);
        }
        self.no_tui = c.no_tui.unwrap_or(false);
    }

    fn ok(&mut self, msg: impl Into<String>) {
        self.status = msg.into();
        self.status_err = false;
    }
    fn err(&mut self, msg: impl Into<String>) {
        self.status = msg.into();
        self.status_err = true;
    }

    fn save(&mut self) {
        let conf = match self.to_conf() {
            Ok(c) => c,
            Err(e) => return self.err(e),
        };
        match toml::to_string_pretty(&conf) {
            Ok(text) => match std::fs::write(&self.path, &text) {
                Ok(()) => {
                    self.preview = text;
                    self.ok(format!("Saved {}", self.path));
                }
                Err(e) => self.err(format!("write {}: {e}", self.path)),
            },
            Err(e) => self.err(format!("serialize: {e}")),
        }
    }

    fn load(&mut self) {
        let text = match std::fs::read_to_string(&self.path) {
            Ok(t) => t,
            Err(e) => return self.err(format!("read {}: {e}", self.path)),
        };
        match toml::from_str::<Conf>(&text) {
            Ok(c) => {
                self.from_conf(c);
                self.preview = text;
                self.ok(format!("Loaded {}", self.path));
            }
            Err(e) => self.err(format!("parse {}: {e}", self.path)),
        }
    }

    /// Populate the detected-device lists. Uses the built-in SDK when this tool
    /// was compiled with the `gpu`/`cuda` features; otherwise shells out to
    /// `jackpotminer --devices-json`.
    fn detect_gpus(&mut self) {
        if gpu_probe::HAS_SDK {
            self.detected_cuda = gpu_probe::cuda();
            self.detected_opencl = gpu_probe::opencl();
            self.after_detect("built-in SDK");
        } else {
            self.detect_via_subprocess();
        }
    }

    /// Fallback: ask the miner binary to enumerate devices for us.
    fn detect_via_subprocess(&mut self) {
        let bin = miner_binary();
        match std::process::Command::new(&bin).arg("--devices-json").output() {
            Ok(o) if o.status.success() => {
                let stdout = String::from_utf8_lossy(&o.stdout);
                // The JSON is the last object-looking line (any logs precede it).
                let line = stdout.lines().rev().find(|l| l.trim_start().starts_with('{')).unwrap_or("");
                match serde_json::from_str::<DevList>(line) {
                    Ok(d) => {
                        self.detected_cuda = d.cuda;
                        self.detected_opencl = d.opencl;
                        self.after_detect("miner probe");
                    }
                    Err(e) => {
                        self.gpu_status_err = true;
                        self.gpu_status = format!("Could not parse device list: {e}");
                    }
                }
            }
            Ok(o) => {
                self.gpu_status_err = true;
                let err = String::from_utf8_lossy(&o.stderr);
                self.gpu_status = format!("Device probe failed: {}", err.lines().next().unwrap_or("non-zero exit"));
            }
            Err(e) => {
                self.gpu_status_err = true;
                self.gpu_status = format!(
                    "Could not run '{}': {e} — is the miner built and OpenCL/CUDA installed?",
                    bin.display()
                );
            }
        }
    }

    /// Common post-detection: warn if nothing was found, else set the status and
    /// mirror the cards onto the form. `via` names the detection source.
    fn after_detect(&mut self, via: &str) {
        if self.detected_cuda.is_empty() && self.detected_opencl.is_empty() {
            self.gpu_status_err = true;
            self.gpu_status = "OpenCL or CUDA is not available — no GPU devices detected.".into();
            self.dev_tune.clear();
            self.device_sel.clear();
            return;
        }
        self.gpu_status_err = false;
        self.gpu_status = format!(
            "Detected {} CUDA, {} OpenCL via {via}.",
            self.detected_cuda.len(),
            self.detected_opencl.len()
        );
        // Default backend = CUDA when NVIDIA cards are present, else OpenCL.
        self.default_backend = if !self.detected_cuda.is_empty() { Backend::Cuda } else { Backend::OpenCl };
        // One row per canonical card, preserving any values already entered
        // (by index); new rows start on the default backend.
        let cards: Vec<String> = self.cards().to_vec();
        let mut rows = Vec::with_capacity(cards.len());
        for (i, label) in cards.iter().enumerate() {
            let mut f = self
                .dev_tune
                .get(i)
                .cloned()
                .unwrap_or_else(|| DeviceTuneForm { backend: self.default_backend, ..Default::default() });
            f.label = label.clone();
            rows.push(f);
        }
        self.dev_tune = rows;
        if self.device_sel.len() != cards.len() {
            self.device_sel = vec![true; cards.len()];
        }
        self.rebuild_devices_field();
    }

    /// The canonical per-card list: the CUDA cards when any NVIDIA card is
    /// present (the default backend and the enumeration `devices` indexes into),
    /// otherwise the OpenCL list. A card can still be flipped to OpenCL per-row.
    fn cards(&self) -> &[String] {
        if !self.detected_cuda.is_empty() {
            &self.detected_cuda
        } else {
            &self.detected_opencl
        }
    }

    /// Rebuild the `devices` field from the per-card checkboxes (all = "all").
    fn rebuild_devices_field(&mut self) {
        let n = self.cards().len();
        if self.device_sel.len() != n {
            self.device_sel = vec![true; n];
        }
        if n == 0 {
            return;
        }
        if self.device_sel.iter().all(|&b| b) {
            self.devices = "all".into();
        } else {
            let list: Vec<String> =
                self.device_sel.iter().enumerate().filter(|(_, &b)| b).map(|(i, _)| i.to_string()).collect();
            self.devices = if list.is_empty() { "0".into() } else { list.join(",") };
        }
    }

    fn live_addr(&self) -> String {
        format!("{}:{}", self.live_host.trim(), self.live_port.trim())
    }

    /// Fetch a live snapshot from the running miner's control server.
    fn live_connect(&mut self) {
        match control_request(&self.live_addr(), &json!({"op": "get"})) {
            Ok(v) => {
                self.live_unlocked = v.get("unlocked").and_then(Value::as_bool).unwrap_or(false);
                self.live_group_size =
                    v.pointer("/cpu/group_size").and_then(Value::as_u64).map(|n| n.to_string()).unwrap_or_default();
                self.live_devices = v
                    .get("devices")
                    .and_then(Value::as_array)
                    .map(|a| a.iter().map(parse_live_device).collect())
                    .unwrap_or_default();
                self.live_rows = v
                    .pointer("/cpu/rows")
                    .and_then(Value::as_array)
                    .map(|a| a.iter().map(parse_live_row).collect())
                    .unwrap_or_default();
                self.live_connected = true;
                self.live_err = false;
                self.live_status =
                    format!("Connected — {} GPU(s), {} CPU row(s)", self.live_devices.len(), self.live_rows.len());
            }
            Err(e) => {
                self.live_connected = false;
                self.live_err = true;
                self.live_status = e;
            }
        }
    }

    /// Send a set-request and report ok / error in the live status line.
    fn live_send(&mut self, req: Value, ok_msg: String) {
        match control_request(&self.live_addr(), &req) {
            Ok(v) if v.get("ok").and_then(Value::as_bool).unwrap_or(false) => {
                self.live_err = false;
                self.live_status = ok_msg;
            }
            Ok(v) => {
                self.live_err = true;
                self.live_status = v.get("error").and_then(Value::as_str).unwrap_or("rejected").to_string();
            }
            Err(e) => {
                self.live_err = true;
                self.live_status = e;
            }
        }
    }

    fn live_apply_device(&mut self, i: usize) {
        let (req, label) = {
            let d = &self.live_devices[i];
            let mut r = json!({"op": "set_device", "index": d.index, "enabled": d.enabled});
            if let Ok(p) = d.power_w.trim().parse::<u32>() {
                r["power_w"] = json!(p);
            }
            if let Ok(c) = d.core_off.trim().parse::<i32>() {
                r["core_off"] = json!(c);
            }
            if let Ok(m) = d.mem_off.trim().parse::<i32>() {
                r["mem_off"] = json!(m);
            }
            (r, d.label.clone())
        };
        self.live_send(req, format!("Applied to {label}"));
    }

    fn live_set_group_size(&mut self) {
        if let Ok(n) = self.live_group_size.trim().parse::<usize>() {
            self.live_send(json!({"op": "set_cpu_group_size", "value": n}), format!("CPU group size -> {n}"));
        }
    }

    fn live_toggle_row(&mut self, index: u64, enabled: bool) {
        self.live_send(
            json!({"op": "set_cpu_row", "index": index, "enabled": enabled}),
            format!("CPU row {index} {}", if enabled { "enabled" } else { "disabled" }),
        );
    }

    fn do_preview(&mut self) {
        match self.to_conf().and_then(|c| toml::to_string_pretty(&c).map_err(|e| e.to_string())) {
            Ok(text) => {
                self.preview = text;
                self.ok("Preview updated");
            }
            Err(e) => self.err(e),
        }
    }
}

/// A labelled single-line text field row inside a grid, returning the field's
/// response (so callers can react to edits).
fn row_resp(ui: &mut egui::Ui, label: &str, value: &mut String, hint: &str) -> egui::Response {
    ui.label(label);
    let r = ui.add(egui::TextEdit::singleline(value).desired_width(280.0).hint_text(hint));
    ui.end_row();
    r
}

/// As [`row_resp`] but discarding the response.
fn row(ui: &mut egui::Ui, label: &str, value: &mut String, hint: &str) {
    row_resp(ui, label, value, hint);
}

impl eframe::App for ConfigApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Auto-detect GPUs on the first frame so the form is pre-populated.
        if self.first_frame {
            self.first_frame = false;
            self.detect_gpus();
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("Config file:");
                ui.add(egui::TextEdit::singleline(&mut self.path).desired_width(240.0));
                if ui.button("Load").clicked() {
                    self.load();
                }
                if ui.button("Save").clicked() {
                    self.save();
                }
                if ui.button("Preview").clicked() {
                    self.do_preview();
                }
            });
            if !self.status.is_empty() {
                let color = if self.status_err { egui::Color32::LIGHT_RED } else { egui::Color32::LIGHT_GREEN };
                ui.colored_label(color, &self.status);
            }
            ui.add_space(4.0);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                egui::CollapsingHeader::new("Pool").default_open(true).show(ui, |ui| {
                    egui::Grid::new("pool").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                        row(ui, "Pool URL", &mut self.url, "stratum+tcp://host:port");
                        row(ui, "Port (optional)", &mut self.port, "only if URL has no :port");
                        row(ui, "Wallet / user", &mut self.user, "address.worker");
                    });
                });

                egui::CollapsingHeader::new("Payout").default_open(true).show(ui, |ui| {
                    ui.radio_value(&mut self.payout, Payout::Jackpot, "Jackpot (default)");
                    ui.radio_value(&mut self.payout, Payout::JackpotPercent, "Jackpot percent");
                    ui.radio_value(&mut self.payout, Payout::Pplns, "PPLNS (no-jackpot)");
                    ui.radio_value(&mut self.payout, Payout::Solo, "Solo");
                    // Intended lottery %, applied in "Jackpot percent" mode.
                    ui.horizontal(|ui| {
                        ui.label("Jackpot percent (intended lottery %, 3-100)");
                        let r = ui.add(
                            egui::TextEdit::singleline(&mut self.jackpot_percent)
                                .desired_width(60.0)
                                .hint_text("e.g. 50"),
                        );
                        // Typing a percent implies the jackpot-percent payout mode.
                        if r.changed() && !self.jackpot_percent.trim().is_empty() {
                            self.payout = Payout::JackpotPercent;
                        }
                    });
                });

                egui::CollapsingHeader::new("GPU (per-card)").default_open(true).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        if ui.button("Detect GPUs").clicked() {
                            self.detect_gpus();
                        }
                        if !self.gpu_status.is_empty() {
                            let c = if self.gpu_status_err { egui::Color32::LIGHT_RED } else { egui::Color32::LIGHT_GREEN };
                            ui.colored_label(c, &self.gpu_status);
                        }
                    });

                    // Per-card: enable to mine, choose its backend, and (CUDA
                    // cards only) set per-card tuning. Cards default to CUDA when
                    // NVIDIA cards are detected; switch any card to OpenCL here.
                    let names: Vec<String> = self.cards().to_vec();
                    if names.is_empty() {
                        ui.label("No devices detected — click \"Detect GPUs\".");
                    } else {
                        if self.device_sel.len() != names.len() {
                            self.device_sel = vec![true; names.len()];
                        }
                        if self.dev_tune.len() != names.len() {
                            self.dev_tune.resize(
                                names.len(),
                                DeviceTuneForm { backend: self.default_backend, ..Default::default() },
                            );
                        }
                        ui.label("Each card: check to mine, pick its backend; tuning (CUDA only) blank = leave alone.");
                        let mut dev_changed = false;
                        egui::Grid::new("cards").num_columns(6).spacing([6.0, 4.0]).show(ui, |ui| {
                            ui.label("On");
                            ui.label("Card");
                            ui.label("Backend");
                            ui.label("Power W");
                            ui.label("Core off");
                            ui.label("Mem off");
                            ui.end_row();
                            for i in 0..names.len() {
                                dev_changed |= ui.checkbox(&mut self.device_sel[i], "").changed();
                                ui.label(trunc(&names[i], 24));
                                egui::ComboBox::from_id_source(("backend", i))
                                    .selected_text(backend_name(self.dev_tune[i].backend))
                                    .width(72.0)
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(&mut self.dev_tune[i].backend, Backend::Cuda, "cuda");
                                        ui.selectable_value(&mut self.dev_tune[i].backend, Backend::OpenCl, "opencl");
                                    });
                                // Tuning is NVML-only; show the fields for CUDA cards.
                                if self.dev_tune[i].backend == Backend::Cuda {
                                    ui.add(egui::TextEdit::singleline(&mut self.dev_tune[i].power_limit).desired_width(52.0));
                                    ui.add(egui::TextEdit::singleline(&mut self.dev_tune[i].gpu_clock_offset).desired_width(52.0));
                                    ui.add(egui::TextEdit::singleline(&mut self.dev_tune[i].mem_clock_offset).desired_width(52.0));
                                } else {
                                    ui.label("—");
                                    ui.label("—");
                                    ui.label("—");
                                }
                                ui.end_row();
                            }
                        });
                        if dev_changed {
                            self.rebuild_devices_field();
                        }
                    }
                    egui::Grid::new("gpu").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                        row(ui, "Devices", &mut self.devices, "all  or  0,1");
                    });
                });

                egui::CollapsingHeader::new("Misc").default_open(true).show(ui, |ui| {
                    egui::Grid::new("misc").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                        row(ui, "Solver threads", &mut self.threads, "default: all cores");
                        row(ui, "Job timeout (s)", &mut self.job_timeout, "0 = off");
                        row(ui, "Control port", &mut self.control_port, "enables live control, e.g. 4068");
                    });
                    ui.checkbox(&mut self.no_tui, "Disable dashboard (--no-tui)");
                });

                egui::CollapsingHeader::new("Live control (running miner)").default_open(true).show(ui, |ui| {
                    ui.label("Connect to a miner started with --control-port to read and adjust it on the fly.");
                    ui.horizontal(|ui| {
                        ui.label("Host");
                        ui.add(egui::TextEdit::singleline(&mut self.live_host).desired_width(90.0));
                        ui.label("Port");
                        ui.add(egui::TextEdit::singleline(&mut self.live_port).desired_width(60.0));
                        if ui.button("Connect / Refresh").clicked() {
                            self.live_connect();
                        }
                    });
                    if !self.live_status.is_empty() {
                        let c = if self.live_err { egui::Color32::LIGHT_RED } else { egui::Color32::LIGHT_GREEN };
                        ui.colored_label(c, &self.live_status);
                    }
                    if self.live_connected {
                        if !self.live_unlocked {
                            ui.colored_label(
                                egui::Color32::YELLOW,
                                "clock/power changes apply only if the miner has --unlock-controls",
                            );
                        }
                        for i in 0..self.live_devices.len() {
                            ui.separator();
                            ui.label(format!(
                                "{}   {:.0} W · {} °C · {} shares",
                                self.live_devices[i].label,
                                self.live_devices[i].watts,
                                self.live_devices[i].temp_c,
                                self.live_devices[i].shares
                            ));
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.live_devices[i].enabled, "mining");
                                ui.label("W");
                                ui.add(egui::TextEdit::singleline(&mut self.live_devices[i].power_w).desired_width(46.0));
                                ui.label("core");
                                ui.add(egui::TextEdit::singleline(&mut self.live_devices[i].core_off).desired_width(46.0));
                                ui.label("mem");
                                ui.add(egui::TextEdit::singleline(&mut self.live_devices[i].mem_off).desired_width(46.0));
                                if ui.button("Apply").clicked() {
                                    self.live_apply_device(i);
                                }
                            });
                        }
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label("CPU group size");
                            ui.add(egui::TextEdit::singleline(&mut self.live_group_size).desired_width(40.0));
                            if ui.button("Set").clicked() {
                                self.live_set_group_size();
                            }
                        });
                        for i in 0..self.live_rows.len() {
                            let label = format!("{}  ({} shares)", self.live_rows[i].label, self.live_rows[i].shares);
                            if ui.checkbox(&mut self.live_rows[i].enabled, label).changed() {
                                let (idx, en) = (self.live_rows[i].index, self.live_rows[i].enabled);
                                self.live_toggle_row(idx, en);
                            }
                        }
                    }
                });

                ui.separator();
                let cmd = format!("jackpotminer --config {}", self.path);
                ui.horizontal(|ui| {
                    ui.label("Run:");
                    ui.code(&cmd);
                    if ui.button("Copy").clicked() {
                        ui.output_mut(|o| o.copied_text = cmd.clone());
                        self.ok("Command copied to clipboard");
                    }
                });

                if !self.preview.is_empty() {
                    egui::CollapsingHeader::new("Generated TOML").default_open(true).show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(&mut self.preview)
                                .code_editor()
                                .desired_rows(8)
                                .desired_width(f32::INFINITY)
                                .interactive(false),
                        );
                    });
                }
            });
        });
    }
}
