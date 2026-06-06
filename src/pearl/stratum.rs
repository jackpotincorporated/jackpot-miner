//! Pearl (PRL) pool client + mining loop — wire protocol pinned from a live
//! capture of `prl.kryptex.network` (see `pearl-dump/pearl-stratum.log`).
//!
//! Handshake:
//!   C> mining.configure  [["pearl/v1"], {}]   S> {pearl/v1:true, share_format:"base64"}
//!   C> mining.subscribe  ["<ua>"]             S> [[["mining.set_difficulty",id],["mining.notify",id]], "", 0]
//!   C> mining.authorize  ["<address>", "<pw>"]
//! Server notifications:
//!   pearl.set_mining_params  [{m,n,k,rank,rows_pattern,cols_pattern,mma_type}]
//!   mining.set_difficulty    [<share_diff>]
//!   mining.notify  [job_id, prev_hash, header_hex(76B = IncompleteBlockHeader),
//!                   height, ntime, nbits, clean]
//! Submit (TODO: confirm against an accepted share):
//!   mining.submit / submitPlainProof with base64 proof (share_format=base64).
//!
//! Reality from the capture: m=n=131072, k=4096 → each attempt is a ~512MB×512MB
//! int8 GEMM (GPU-scale). The CPU path here parses/validates and grinds only at
//! small sizes; real mining needs the GPU pipeline.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};

use primitive_types::U256;

use super::pearlhash::{
    build_proof, encode_proof, share_bound, try_mine_one, verify, IncompleteBlockHeader, MMAType, MiningConfiguration,
    PeriodicPattern, Solution,
};

/// Largest CPU-grindable problem (elements per operand). Real pool configs are
/// far bigger (131072×4096) and need the GPU; above this we parse+log only.
const CPU_GRIND_LIMIT: usize = 4096 * 256;

/// Upper bounds on what the GPU path (feature `pearl-cuda`) will attempt. The
/// whole attempt is on-GPU now (rng + commitment + noise + IMMA sweep + harvest),
/// so the only real limit is VRAM: the full-GPU path holds 4 operand buffers ≈
/// `2·(m+n)·k` bytes, so `GPU_MAX_ELEMS` ≈ half the VRAM budget. The **live
/// config (m=n=131072 ⇒ 134M tiles, ~2.15 GB) now FITS and runs** (~50 s/attempt
/// on a 16 GB 5080 via the tensor-core sweep — functional but latency-bound).
/// `GPU_MAX_TILES` is a backstop against absurd grids.
#[cfg(feature = "pearl-cuda")]
const GPU_MAX_TILES: u128 = 256 << 20; // up to ~268M tiles per attempt (live = 134M)
#[cfg(feature = "pearl-cuda")]
const GPU_MAX_ELEMS: usize = 1280 << 20; // (m+n)·k ≤ 1.34e9 ⇒ ≲2.7 GB resident
/// Tiles per GPU sweep batch — bounds the per-batch index upload + jackpot
/// readback (~64 B/tile); the resident matrices are untouched between batches.
#[cfg(feature = "pearl-cuda")]
const GPU_BATCH_TILES: usize = 1 << 16;

#[derive(Clone)]
pub struct Job {
    pub id: String,
    pub header: IncompleteBlockHeader,
    pub config: MiningConfiguration,
    pub m: usize,
    pub n: usize,
}

impl Job {
    fn k(&self) -> usize {
        self.config.common_dim as usize
    }
}

pub struct PearlStratum {
    stream: TcpStream,
    rx: std::sync::mpsc::Receiver<Value>,
    pending: std::collections::VecDeque<Value>,
    next_id: u64,
    closed: bool,
}

impl PearlStratum {
    pub fn connect(host: &str, port: u16, user: &str, pass: &str) -> Result<Self> {
        let stream = TcpStream::connect((host, port)).with_context(|| format!("connect {host}:{port}"))?;
        // A dedicated reader thread does blocking line reads (no partial-line /
        // non-blocking hazards) and forwards parsed messages over a channel, so
        // the mine loop can poll for newer jobs without ever blocking on I/O.
        let read_stream = stream.try_clone()?;
        let (tx, rx) = std::sync::mpsc::channel::<Value>();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(read_stream);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) | Err(_) => break, // closed / fatal
                    Ok(_) => {
                        let t = line.trim();
                        if t.is_empty() {
                            continue;
                        }
                        match serde_json::from_str::<Value>(t) {
                            Ok(v) => {
                                if tx.send(v).is_err() {
                                    break; // receiver gone
                                }
                            }
                            Err(_) => {} // ignore malformed lines
                        }
                    }
                }
            }
        });
        let mut s = Self { stream, rx, pending: std::collections::VecDeque::new(), next_id: 1, closed: false };
        s.call("mining.configure", json!([["pearl/v1"], {}]))?;
        s.call("mining.subscribe", json!(["jackpotminer/0.1"]))?;
        s.call("mining.authorize", json!([user, pass]))?;
        Ok(s)
    }

    fn call(&mut self, method: &str, params: Value) -> Result<u64> {
        let id = self.next_id;
        self.next_id += 1;
        let mut line = serde_json::to_string(&json!({ "id": id, "method": method, "params": params }))?;
        line.push('\n');
        self.stream.write_all(line.as_bytes())?;
        Ok(id)
    }

    /// Drain the reader channel into the pending buffer (non-blocking).
    fn pump(&mut self) {
        loop {
            match self.rx.try_recv() {
                Ok(v) => self.pending.push_back(v),
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.closed = true;
                    break;
                }
            }
        }
    }

    fn try_recv(&mut self) -> Result<Option<Value>> {
        self.pump();
        if let Some(v) = self.pending.pop_front() {
            return Ok(Some(v));
        }
        if self.closed {
            return Err(anyhow!("pool closed the connection"));
        }
        Ok(None)
    }

    /// Whether a `mining.notify` for a job other than `current_id` is buffered —
    /// i.e. newer work has arrived and the current attempt should be abandoned.
    /// Non-consuming: the message stays queued for the main loop to apply.
    pub fn has_newer_job(&mut self, current_id: &str) -> bool {
        self.pump();
        self.pending.iter().any(|m| is_newer_notify(m, current_id))
    }

    /// Submit a base64 proof for `job_id`. TODO(confirm): exact method/params +
    /// any nonce/ntime fields, from an accepted-share capture.
    pub fn submit(&mut self, user: &str, job_id: &str, proof: &[u8]) -> Result<()> {
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(proof);
        self.call("mining.submit", json!([user, job_id, b64]))?;
        Ok(())
    }
}

/// A buffered message is "newer work" iff it's a `mining.notify` whose job id
/// (params[0]) differs from the one currently being mined.
fn is_newer_notify(m: &Value, current_id: &str) -> bool {
    m.get("method").and_then(Value::as_str) == Some("mining.notify")
        && m["params"].get(0).and_then(Value::as_str).is_some_and(|id| id != current_id)
}

fn unhex(s: &str) -> Vec<u8> {
    let s = s.trim_start_matches("0x");
    (0..s.len() / 2).filter_map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()).collect()
}

/// Parse `pearl.set_mining_params` → (config, m, n).
fn parse_mining_params(p: &Value) -> Option<(MiningConfiguration, usize, usize)> {
    let o = p.get(0)?;
    let m = o.get("m")?.as_u64()? as usize;
    let n = o.get("n")?.as_u64()? as usize;
    let k = o.get("k")?.as_u64()? as u32;
    let rank = o.get("rank")?.as_u64()? as u16;
    let to_list = |key: &str| -> Option<Vec<u32>> {
        Some(o.get(key)?.as_array()?.iter().filter_map(|v| v.as_u64().map(|x| x as u32)).collect())
    };
    let rows = PeriodicPattern::from_list(&to_list("rows_pattern")?)?;
    let cols = PeriodicPattern::from_list(&to_list("cols_pattern")?)?;
    let config = MiningConfiguration {
        common_dim: k,
        rank,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: rows,
        cols_pattern: cols,
        reserved: [0u8; 32],
    };
    Some((config, m, n))
}

/// Parse a `mining.notify` array using the current matmul config.
fn parse_notify(p: &Value, config: &MiningConfiguration, m: usize, n: usize) -> Option<Job> {
    let arr = p.as_array()?;
    let id = arr.first()?.as_str()?.to_string();
    let header_hex = arr.get(2)?.as_str()?;
    let header = IncompleteBlockHeader::from_bytes(&unhex(header_hex))?;
    Some(Job { id, header, config: *config, m, n })
}

/// The Pearl mining loop. Connects, tracks config/difficulty, parses jobs.
/// Grinds on CPU only for small problems; real (131072×4096) configs are logged
/// for the GPU pipeline.
pub fn run(host: &str, port: u16, user: &str, pass: &str, running: Arc<AtomicBool>) -> Result<()> {
    let mut pool = PearlStratum::connect(host, port, user, pass)?;
    let threads = num_cpus::get().max(1);
    log::info!("pearl: connected to {host}:{port} ({threads} CPU threads)");

    #[cfg(feature = "pearl-cuda")]
    let mut gpu_nonce: u64 = 0;
    #[cfg(feature = "pearl-cuda")]
    let gpu = match super::gpu::PearlGpu::new(0) {
        Ok(g) => {
            log::info!("pearl: GPU 0 ready (int8 dp4a tile kernel)");
            Some(g)
        }
        Err(e) => {
            log::warn!("pearl: GPU init failed ({e}); CPU grind only");
            None
        }
    };

    let mut config: Option<MiningConfiguration> = None;
    let (mut m, mut n) = (0usize, 0usize);
    let mut difficulty = 1u64;
    let mut job: Option<Job> = None;

    while running.load(Ordering::Relaxed) {
        while let Some(msg) = pool.try_recv()? {
            log::debug!("pearl: recv {}", serde_json::to_string(&msg).unwrap_or_default());
            match msg.get("method").and_then(Value::as_str).unwrap_or("") {
                "pearl.set_mining_params" => {
                    if let Some((c, mm, nn)) = parse_mining_params(&msg["params"]) {
                        log::info!(
                            "pearl: params m={mm} n={nn} k={} rank={} h={} w={}",
                            c.common_dim, c.rank, c.rows_pattern.size(), c.cols_pattern.size()
                        );
                        config = Some(c);
                        m = mm;
                        n = nn;
                    }
                }
                "mining.set_difficulty" => {
                    if let Some(d) = msg["params"].get(0).and_then(Value::as_f64) {
                        difficulty = d.max(1.0) as u64;
                        log::info!("pearl: share difficulty {difficulty}");
                    }
                }
                "mining.notify" => {
                    if let Some(c) = &config {
                        if let Some(j) = parse_notify(&msg["params"], c, m, n) {
                            log::info!("pearl: job {} (nbits {:#010x})", j.id, j.header.nbits);
                            job = Some(j);
                        }
                    }
                }
                _ => {}
            }
        }

        let Some(j) = job.clone() else {
            std::thread::sleep(Duration::from_millis(200));
            continue;
        };

        let bound = share_bound(difficulty, j.config.rows_pattern.size() as usize, j.config.cols_pattern.size() as usize, j.k());
        let deadline = Instant::now() + Duration::from_millis(500);
        let cpu_ok = j.m.saturating_mul(j.k()) <= CPU_GRIND_LIMIT;

        // Pick a miner: GPU for tractable configs (pearl-cuda), else CPU for
        // small problems, else skip. A GPU attempt can open several tiles — each
        // is a separate share — so the miner returns ALL winners for the job.
        let sols: Vec<Solution> = {
            #[cfg(feature = "pearl-cuda")]
            {
                if let (Some(g), true) = (gpu.as_ref(), gpu_suitable(&j)) {
                    gpu_mine(g, &j, bound, &mut pool, &running, &mut gpu_nonce)
                } else if cpu_ok {
                    mine_batch(&j, bound, deadline, &running, threads).into_iter().collect()
                } else {
                    log::warn!(
                        "pearl: job {} too large for the GPU VRAM budget (m={} n={} k={}, {} tiles)",
                        j.id, j.m, j.n, j.k(), tile_count(&j)
                    );
                    std::thread::sleep(Duration::from_millis(500));
                    Vec::new()
                }
            }
            #[cfg(not(feature = "pearl-cuda"))]
            {
                if cpu_ok {
                    mine_batch(&j, bound, deadline, &running, threads).into_iter().collect()
                } else {
                    log::warn!(
                        "pearl: job {} is GPU-scale (m={} k={}); CPU grind skipped — build with --features pearl-cuda",
                        j.id, j.m, j.k()
                    );
                    std::thread::sleep(Duration::from_millis(500));
                    Vec::new()
                }
            }
        };

        for sol in &sols {
            let proof = build_proof(&j.header, &j.config, sol);
            pool.submit(user, &j.id, &encode_proof(&proof))?;
        }
        if !sols.is_empty() {
            log::info!("pearl: submitted {} share(s) for job {}", sols.len(), j.id);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Validate parsing against the real prl.kryptex.network capture
    /// (pearl-dump/pearl-stratum.log).
    #[test]
    fn parses_live_capture() {
        let cols: Vec<u32> = (0..64).collect();
        let params = json!([{
            "m": 131072, "n": 131072, "k": 4096, "rank": 128,
            "rows_pattern": [0, 32], "cols_pattern": cols, "mma_type": "Int7xInt7ToInt32"
        }]);
        let (c, m, n) = parse_mining_params(&params).expect("params");
        assert_eq!((m, n, c.common_dim, c.rank), (131072, 131072, 4096, 128));
        assert_eq!(c.rows_pattern.size(), 2);
        assert_eq!(c.cols_pattern.size(), 64);

        let header_hex = "0000402089571bf45a9e6d9f88d4e48aa1d44fe0060dbb1b871002d21a58b56a36e0793958eec088936f1453e3f3903402e691f23ab996f73ca0fa38146d70cb1656b2ee2a4e226a86350118";
        let prev = "3979e0366ab5581ad20210871bbb0d06e04fd4a18ae4d4889f6d9e5af41b5789";
        let notify = json!(["0001099c-fad2b0b3-1978c7701d91dda9", prev, header_hex, 67996, "6a224e2a", "18013586", true]);
        let j = parse_notify(&notify, &c, m, n).expect("notify");
        assert_eq!(j.id, "0001099c-fad2b0b3-1978c7701d91dda9");
        assert_eq!(j.header.nbits, 0x1801_3586);
        assert_eq!(j.header.timestamp, 0x6a22_4e2a);
        // prev_block recovered (header stores it reversed); job_key uses to_bytes.
        assert_eq!(j.header.prev_block.to_vec(), unhex(prev));
        // header round-trips: to_bytes == the notify header hex (so commitment matches).
        assert_eq!(j.header.to_bytes().to_vec(), unhex(header_hex));
    }

    /// `is_newer_notify` drives the abandon-stale-work check: it fires only for a
    /// `mining.notify` carrying a job id different from the one being mined.
    #[test]
    fn detects_newer_job() {
        let cur = "0001099c-fad2b0b3-1978c7701d91dda9";
        let same = json!({"method": "mining.notify", "params": [cur, "prev", "00", 1, "t", "n", true]});
        let newer = json!({"method": "mining.notify", "params": ["0001099d-aaaa-bbbb", "prev", "00", 1, "t", "n", true]});
        let other = json!({"method": "mining.set_difficulty", "params": [30]});
        assert!(!is_newer_notify(&same, cur), "same job id is not newer");
        assert!(is_newer_notify(&newer, cur), "different job id is newer work");
        assert!(!is_newer_notify(&other, cur), "non-notify is not newer work");
    }
}

/// Grind `try_mine_one` across `threads` cores until a tile opens or the deadline
/// passes. First verified solution wins.
fn mine_batch(j: &Job, bound: U256, deadline: Instant, running: &AtomicBool, threads: usize) -> Option<Solution> {
    let found: Mutex<Option<Solution>> = Mutex::new(None);
    let stop = AtomicBool::new(false);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let mut rng = rand::rng();
                while !stop.load(Ordering::Relaxed) && running.load(Ordering::Relaxed) && Instant::now() < deadline {
                    if let Some(sol) = try_mine_one(&mut rng, j.m, j.n, j.k(), &j.header, &j.config, bound) {
                        if verify(&j.header, &j.config, &sol, bound) {
                            *found.lock().unwrap() = Some(sol);
                            stop.store(true, Ordering::Relaxed);
                            return;
                        }
                    }
                }
            });
        }
    });
    found.into_inner().unwrap()
}

/// Tiles swept per attempt for the periodic patterns: `(m/h)·(n/w)`.
#[cfg(feature = "pearl-cuda")]
fn tile_count(j: &Job) -> u128 {
    let h = j.config.rows_pattern.size() as usize;
    let w = j.config.cols_pattern.size() as usize;
    (j.m / h.max(1)) as u128 * (j.n / w.max(1)) as u128
}

/// Whether the GPU host-gather path can attempt this job (operands + tile count
/// within [`GPU_MAX_ELEMS`]/[`GPU_MAX_TILES`]). The live config exceeds both.
#[cfg(feature = "pearl-cuda")]
fn gpu_suitable(j: &Job) -> bool {
    let elems = j.m.saturating_add(j.n).saturating_mul(j.k());
    elems <= GPU_MAX_ELEMS && tile_count(j) <= GPU_MAX_TILES
}

/// GPU mining: runs ONE **fully-on-GPU** attempt (A,Bᵀ generated, committed,
/// noised, and swept on the GPU; see [`super::gpu::PearlGpu::mine_full_gpu_all`])
/// over every tile with the next nonce, returning ALL winning tiles (each a
/// separate share). The sweep checks `running` between row-tile chunks, so a long
/// live-scale attempt stays interruptible. The chunk-aligned fast path needs
/// `(m·k)%1024==0`; otherwise it falls back to CPU-generated operands (first win).
/// The caller loops (refreshing the job between attempts).
#[cfg(feature = "pearl-cuda")]
fn gpu_mine(gpu: &super::gpu::PearlGpu, j: &Job, bound: U256, pool: &mut PearlStratum, running: &AtomicBool, nonce: &mut u64) -> Vec<Solution> {
    let (m, n, k) = (j.m, j.n, j.k());
    let batch = GPU_BATCH_TILES;
    let aligned = (m * k) % 1024 == 0 && (n * k) % 1024 == 0;
    *nonce = nonce.wrapping_add(1);
    let attempt = if aligned {
        // Abandon the attempt on shutdown or when the pool pushes a newer job.
        let jid = j.id.clone();
        gpu.mine_full_gpu_all(&j.header, &j.config, m, n, bound, *nonce, batch, || {
            running.load(Ordering::Relaxed) && !pool.has_newer_job(&jid)
        })
    } else {
        use rand::Rng;
        let mut rng = rand::rng();
        let a: Vec<Vec<i8>> = (0..m).map(|_| (0..k).map(|_| rng.random_range(-64..=64i8)).collect()).collect();
        let b: Vec<Vec<i8>> = (0..k).map(|_| (0..n).map(|_| rng.random_range(-64..=64i8)).collect()).collect();
        let bt: Vec<Vec<i8>> = (0..n).map(|i| (0..k).map(|jj| b[jj][i]).collect()).collect();
        gpu.try_mine_one_gpu(&j.header, &j.config, a, bt, m, n, bound, batch).map(|o| o.into_iter().collect())
    };
    match attempt {
        Ok(v) => v,
        Err(e) => {
            log::error!("pearl: GPU mine error: {e}");
            Vec::new()
        }
    }
}
