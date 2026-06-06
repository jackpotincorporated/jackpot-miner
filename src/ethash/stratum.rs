//! Ethash stratum client + mining loop.
//!
//! Ethash uses a different stratum dialect than the Zcash/Equihash `StratumClient`
//! in `src/stratum.rs`. This implements the common **EthereumStratum/1.0.0**
//! (NiceHash) shape — `mining.subscribe` / `mining.set_difficulty` /
//! `mining.notify [jobId, seedhash, headerhash, cleanJobs]` / `mining.submit
//! [user, jobId, nonce]`. The older getwork/ethproxy variant differs slightly;
//! `TODO(pool)` marks the framing to confirm against a live pool.
//!
//! Loop: job → GPU search (WIP, see ../ethash.rs) → **CPU verify** (`super::verify`)
//! → submit verified shares only.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};

use super::verify::{self, epoch_from_seedhash};
use super::EthashSolver;

/// One Ethash job.
#[derive(Clone)]
pub struct Job {
    pub id: String,
    pub seedhash: [u8; 32],
    pub header_hash: [u8; 32],
    pub target: [u8; 32],
    pub epoch: u64,
}

pub struct EthashStratum {
    stream: TcpStream,
    reader: BufReader<TcpStream>,
    next_id: u64,
    /// Current difficulty → target boundary (set by mining.set_difficulty).
    target: [u8; 32],
    extranonce: String,
}

impl EthashStratum {
    pub fn connect(host: &str, port: u16, user: &str, pass: &str) -> Result<Self> {
        let stream = TcpStream::connect((host, port)).with_context(|| format!("connect {host}:{port}"))?;
        let reader = BufReader::new(stream.try_clone()?);
        let mut s = Self { stream, reader, next_id: 1, target: [0xff; 32], extranonce: String::new() };
        // TODO(pool): some pools want ["miner/ver", "EthereumStratum/1.0.0"].
        s.call("mining.subscribe", json!(["jackpotminer-ethash/0.1", "EthereumStratum/1.0.0"]))?;
        s.call("mining.authorize", json!([user, pass]))?;
        Ok(s)
    }

    fn send(&mut self, v: &Value) -> Result<()> {
        let mut line = serde_json::to_string(v)?;
        line.push('\n');
        self.stream.write_all(line.as_bytes())?;
        Ok(())
    }

    fn call(&mut self, method: &str, params: Value) -> Result<u64> {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({ "id": id, "method": method, "params": params }))?;
        Ok(id)
    }

    fn recv(&mut self) -> Result<Value> {
        let mut line = String::new();
        if self.reader.read_line(&mut line)? == 0 {
            return Err(anyhow!("pool closed the connection"));
        }
        Ok(serde_json::from_str(line.trim())?)
    }

    /// Block until the next job, applying difficulty/extranonce updates en route.
    pub fn next_job(&mut self) -> Result<Job> {
        loop {
            let msg = self.recv()?;
            match msg.get("method").and_then(Value::as_str).unwrap_or("") {
                "mining.set_difficulty" => {
                    if let Some(d) = msg["params"].get(0).and_then(Value::as_f64) {
                        self.target = target_from_difficulty(d.max(1.0) as u64);
                    }
                }
                "mining.set_extranonce" => {
                    if let Some(x) = msg["params"].get(0).and_then(Value::as_str) {
                        self.extranonce = x.to_string();
                    }
                }
                "mining.notify" => {
                    // params: [jobId, seedhash, headerhash, cleanJobs]  TODO(pool): order varies.
                    let p = &msg["params"];
                    let id = p.get(0).and_then(Value::as_str).unwrap_or("").to_string();
                    let seedhash = hex32(p.get(1));
                    let header_hash = hex32(p.get(2));
                    let epoch = epoch_from_seedhash(&seedhash)
                        .ok_or_else(|| anyhow!("unknown seedhash (epoch scan failed)"))?;
                    return Ok(Job { id, seedhash, header_hash, target: self.target, epoch });
                }
                _ => {}
            }
        }
    }

    /// Submit a verified nonce. NiceHash: nonce hex is extranonce-prefixed.
    /// TODO(pool): some pools also want the mix hash as a 4th param.
    pub fn submit(&mut self, user: &str, job_id: &str, nonce: u64) -> Result<()> {
        let nonce_hex = format!("0x{}{:016x}", self.extranonce, nonce);
        self.call("mining.submit", json!([user, job_id, nonce_hex]))?;
        Ok(())
    }
}

fn hex32(v: Option<&Value>) -> [u8; 32] {
    let s = v.and_then(Value::as_str).unwrap_or("").trim_start_matches("0x");
    let mut o = [0u8; 32];
    for i in 0..32.min(s.len() / 2) {
        o[i] = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap_or(0);
    }
    o
}

/// Ethash boundary = floor(2^256 / difficulty), big-endian 32 bytes.
pub fn target_from_difficulty(diff: u64) -> [u8; 32] {
    if diff <= 1 {
        return [0xff; 32];
    }
    let d = diff as u128;
    let mut q = [0u8; 32];
    let mut rem: u128 = 0;
    // Dividend = 1 followed by 256 zero bits (257 bits); keep the low 256 quotient bits.
    for bitpos in (0..=256usize).rev() {
        rem = (rem << 1) | if bitpos == 256 { 1 } else { 0 };
        if rem >= d {
            rem -= d;
            if bitpos < 256 {
                let from_msb = 255 - bitpos;
                q[from_msb / 8] |= 1 << (7 - (from_msb % 8));
            }
        }
    }
    q
}

/// The Ethash mining loop (single device). Mirrors `pearl_main::run`. The GPU
/// search is still WIP (see ../ethash.rs `TODO(capture)`); the verify+submit
/// path is real, so once `search` returns candidates this is end-to-end.
pub fn run(
    host: &str,
    port: u16,
    user: &str,
    pass: &str,
    device: usize,
    recording_path: &str,
    running: Arc<AtomicBool>,
) -> Result<()> {
    let _ = recording_path; // the recording is bundled; DAG is built in new()
    let solver = EthashSolver::new(device).context("init Ethash solver")?;
    // The GPU replay built the DAG from the captured (epoch-pinned) recording, so
    // it verifies against that exact epoch. Cross-check on the CPU with the same
    // light cache the GPU used — independent of the coin's epoch convention.
    let light = solver.cpu_light();
    let mut pool = EthashStratum::connect(host, port, user, pass)?;
    log::info!("ethash: mining on device {device}");

    let mut start_nonce: u64 = 0;
    while running.load(Ordering::Relaxed) {
        let job = pool.next_job()?;
        match solver.search(&job.header_hash, &job.target, start_nonce) {
            Ok(res) => {
                for f in &res.found {
                    // CPU verify (same cache as the GPU): never submit a bad share.
                    if verify::verify(&light, &job.header_hash, f.nonce, &job.target, Some(&f.mix)) {
                        pool.submit(user, &job.id, f.nonce)?;
                        log::info!("ethash: submitted nonce {:#x} for job {}", f.nonce, job.id);
                    } else {
                        log::debug!("ethash: GPU nonce {:#x} failed CPU verify — dropped", f.nonce);
                    }
                }
            }
            Err(e) => log::debug!("ethash: search error: {e}"),
        }
        start_nonce = start_nonce.wrapping_add(25_559_040); // 399360 * 64 nonces per batch
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn difficulty_target_basic() {
        // diff 1 -> max boundary; larger diff -> smaller boundary.
        assert_eq!(target_from_difficulty(1), [0xff; 32]);
        let t2 = target_from_difficulty(2);
        // 2^256/2 = 2^255 -> 0x80 00 .. 00
        assert_eq!(t2[0], 0x80);
        assert!(t2[1..].iter().all(|&b| b == 0));
        // monotonic: higher difficulty => numerically smaller target
        let a = target_from_difficulty(1000);
        let b = target_from_difficulty(2000);
        assert!(b < a); // lexicographic on big-endian == numeric
    }
}
