//! Minimal Zcash-style Stratum client (the protocol used by ZClassic and other
//! Equihash pools, as implemented by nheqminer/EWBF).
//!
//! A supervisor thread owns the socket lifecycle: it performs the
//! subscribe/authorize handshake, reads newline-delimited JSON-RPC messages, and
//! publishes the latest job and target into shared state for the mining threads.
//! Share submissions are written back through the same socket. If the pool drops
//! the connection, the supervisor reconnects automatically (capped exponential
//! backoff) and re-handshakes; TCP keepalive reduces idle disconnects.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use log::{debug, info, warn};
use serde_json::{json, Value};
use socket2::{SockRef, TcpKeepalive};

/// Backoff bounds for reconnecting after the pool drops us.
const RECONNECT_MIN: Duration = Duration::from_secs(1);
const RECONNECT_MAX: Duration = Duration::from_secs(30);
/// Idle time before TCP keepalive probes start.
const KEEPALIVE: Duration = Duration::from_secs(30);
/// JSON-RPC id for our custom `get_jackpot` request.
const JACKPOT_ID: u64 = 4;

/// Everything a mining thread needs to build headers and check shares.
#[derive(Clone)]
pub struct Work {
    /// Monotonic id; bumped whenever a fresh job arrives so workers can restart.
    pub epoch: u64,
    pub job_id: String,
    /// The 108 header bytes before the nonce (version..bits), ready to concat.
    pub prefix: Vec<u8>,
    /// Time field as sent by the pool (hex), echoed back on submit.
    pub time_hex: String,
    /// Pool-assigned nonce prefix; the worker fills the remaining nonce bytes.
    pub nonce1: Vec<u8>,
    /// 256-bit share target, big-endian.
    pub target: [u8; 32],
}

/// Shared, lock-protected current work plus an epoch counter for cheap polling.
#[derive(Clone)]
pub struct WorkHandle {
    inner: Arc<Mutex<Option<Work>>>,
    epoch: Arc<AtomicU64>,
}

impl WorkHandle {
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(None)),
            epoch: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Current epoch (changes when a new job is published).
    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    /// Snapshot the current work, if any.
    pub fn get(&self) -> Option<Work> {
        self.inner.lock().unwrap().clone()
    }
}

/// A Stratum session that reconnects automatically when the pool drops it.
///
/// `writer` holds the live socket, or `None` while disconnected; the supervisor
/// thread swaps it as connections come and go.
#[derive(Clone)]
pub struct StratumClient {
    host: String,
    port: u16,
    user: String,
    pass: String,
    writer: Arc<Mutex<Option<TcpStream>>>,
    work: WorkHandle,
    submit_id: Arc<AtomicU64>,
    /// In-flight share submissions: submit id -> the worker that sent it (e.g.
    /// "GPU 0 (CUDA)" or "CPU cores 0-3"), so the pool's accept/reject reply can
    /// name the submitter. Entries are removed when the reply arrives, and the
    /// map is cleared on reconnect (those replies will never come).
    pending: Arc<Mutex<HashMap<u64, String>>>,
    /// Latest `get_jackpot` reply: a JSON object with current jackpot + pool-share
    /// info for the current and last 10 lotteries (None until the pool replies).
    jackpot: Arc<Mutex<Option<Value>>>,
}

impl StratumClient {
    /// Connect, handshake, and start the supervisor thread that keeps the
    /// connection alive (reconnecting on drop). The first connection must succeed
    /// so a bad URL or unreachable pool fails fast.
    pub fn connect(host: &str, port: u16, user: &str, pass: &str) -> Result<Self> {
        let client = Self {
            host: host.to_string(),
            port,
            user: user.to_string(),
            pass: pass.to_string(),
            writer: Arc::new(Mutex::new(None)),
            work: WorkHandle::new(),
            submit_id: Arc::new(AtomicU64::new(100)),
            pending: Arc::new(Mutex::new(HashMap::new())),
            jackpot: Arc::new(Mutex::new(None)),
        };

        let stream = client.dial_and_handshake()?;

        // Supervisor: read until the pool drops us, then reconnect forever.
        let supervisor = client.clone();
        thread::spawn(move || supervisor.run_sessions(stream));

        Ok(client)
    }

    /// Handle for mining threads to read jobs from.
    pub fn work(&self) -> WorkHandle {
        self.work.clone()
    }

    /// The pool payout mode implied by the worker password: `no-jackpot` →
    /// "pplns", `solo` → "solo", `jackpot`/`jackpot.<pct>` (the default) →
    /// "jackpot".
    pub fn mining_mode(&self) -> &'static str {
        if self.pass == "solo" {
            "solo"
        } else if self.pass == "no-jackpot" {
            "pplns"
        } else {
            // "jackpot" or "jackpot.<pct>"
            "jackpot"
        }
    }

    /// The pool's last `get_jackpot` reply: a JSON object with current jackpot and
    /// pool-share info for the current and last 10 lotteries. `None` until the
    /// pool responds. (Accessor for consumers/UI; the raw object is preserved.)
    #[allow(dead_code)]
    pub fn jackpot(&self) -> Option<Value> {
        self.jackpot.lock().unwrap().clone()
    }

    /// Open a fresh socket, enable keepalive, install it as the writer, and run
    /// the subscribe/authorize handshake. Returns the stream for the reader loop.
    fn dial_and_handshake(&self) -> Result<TcpStream> {
        let stream = TcpStream::connect((self.host.as_str(), self.port))
            .with_context(|| format!("connecting to {}:{}", self.host, self.port))?;
        stream.set_nodelay(true).ok();
        if let Err(e) = SockRef::from(&stream).set_tcp_keepalive(&TcpKeepalive::new().with_time(KEEPALIVE)) {
            debug!("could not enable TCP keepalive: {e}");
        }

        // Install the writer before sending so `send` can use it.
        *self.writer.lock().unwrap() = Some(stream.try_clone()?);

        self.send(&json!({
            "id": 1,
            "method": "mining.subscribe",
            "params": ["jackpotminer/0.1.0", Value::Null, self.host, self.port],
        }))?;
        self.send(&json!({
            "id": 2,
            "method": "mining.authorize",
            "params": [self.user, self.pass],
        }))?;
        // Custom call: current jackpot + pool-share info for the current and last
        // 10 lotteries. Best-effort (pools without it just error/ignore).
        self.send(&json!({
            "id": JACKPOT_ID,
            "method": "get_jackpot",
            "params": [],
        }))?;

        Ok(stream)
    }

    /// Own the connection lifecycle: read until disconnect, then reconnect with
    /// capped exponential backoff. Runs until the process exits.
    fn run_sessions(&self, first: TcpStream) {
        let mut stream = first;
        let mut backoff = RECONNECT_MIN;
        loop {
            // Blocks until the pool closes the connection or errors.
            match reader_loop(stream, self.work.clone(), self.jackpot.clone(), self.pending.clone()) {
                Ok(()) => warn!("stratum connection closed; reconnecting..."),
                Err(e) => warn!("stratum disconnected: {e}; reconnecting..."),
            }
            // The socket is dead; stop workers from writing to it, and drop any
            // in-flight submissions — their replies will never arrive.
            *self.writer.lock().unwrap() = None;
            self.pending.lock().unwrap().clear();

            // Retry until reconnected, backing off between failed attempts.
            stream = loop {
                match self.dial_and_handshake() {
                    Ok(s) => {
                        info!("reconnected to {}:{}", self.host, self.port);
                        backoff = RECONNECT_MIN;
                        break s;
                    }
                    Err(e) => {
                        warn!("reconnect to {}:{} failed: {e}; retrying in {:?}", self.host, self.port, backoff);
                        thread::sleep(backoff);
                        backoff = (backoff * 2).min(RECONNECT_MAX);
                    }
                }
            };
        }
    }

    /// Submit a found share. `nonce2` is the worker-chosen nonce tail; `solution`
    /// is the full serialised solution including its compact-size prefix. Fails
    /// (without crashing the worker) if currently disconnected — the supervisor
    /// reconnects in the background for the next share.
    pub fn submit(&self, who: &str, job_id: &str, time_hex: &str, nonce2: &[u8], solution: &[u8]) -> Result<()> {
        let id = self.submit_id.fetch_add(1, Ordering::Relaxed);
        let msg = json!({
            "id": id,
            "method": "mining.submit",
            "params": [
                self.user,
                job_id,
                time_hex,
                hex::encode(nonce2),
                hex::encode(solution),
            ],
        });
        // Record the submitter so the pool's reply (matched by `id`) can name it.
        self.pending.lock().unwrap().insert(id, who.to_string());
        info!("submitting share for job {job_id} from {who}");
        self.send(&msg)
    }

    /// Serialise a JSON-RPC message and write it (newline-terminated) to the
    /// current socket. Errors if disconnected or the write fails (broken pipe).
    fn send(&self, msg: &Value) -> Result<()> {
        let mut line = serde_json::to_string(msg)?;
        line.push('\n');
        let mut guard = self.writer.lock().unwrap();
        let stream = guard.as_mut().ok_or_else(|| anyhow!("not connected to pool"))?;
        stream.write_all(line.as_bytes())?;
        stream.flush()?;
        debug!("-> {}", line.trim_end());
        Ok(())
    }
}

/// Read and dispatch messages until the connection closes.
fn reader_loop(
    stream: TcpStream,
    work: WorkHandle,
    jackpot: Arc<Mutex<Option<Value>>>,
    pending: Arc<Mutex<HashMap<u64, String>>>,
) -> Result<()> {
    // Pool-assigned nonce prefix, learned from the subscribe response.
    let mut nonce1: Vec<u8> = Vec::new();
    // Latest target; defaults to "anything" until the pool sets one.
    let mut target = [0xffu8; 32];

    let reader = BufReader::new(stream);
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        debug!("<- {line}");
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                warn!("bad json from pool: {e}");
                continue;
            }
        };

        // Subscribe response: result[1] carries the nonce1 prefix.
        if v.get("id").and_then(Value::as_u64) == Some(1) {
            if let Some(arr) = v.get("result").and_then(Value::as_array) {
                if let Some(n1) = arr.get(1).and_then(Value::as_str) {
                    nonce1 = hex::decode(n1).unwrap_or_default();
                    info!("subscribed; nonce1 = {} ({} bytes)", n1, nonce1.len());
                }
            }
            continue;
        }

        // get_jackpot response: a JSON object with current jackpot + pool-share
        // info for the current and last 10 lotteries. Stored raw for consumers.
        if v.get("id").and_then(Value::as_u64) == Some(JACKPOT_ID) {
            if v.get("error").map(|e| !e.is_null()).unwrap_or(false) {
                debug!("pool has no get_jackpot: {}", v.get("error").unwrap());
            } else if let Some(result) = v.get("result").filter(|r| r.is_object()) {
                info!("jackpot info received");
                *jackpot.lock().unwrap() = Some(result.clone());
            }
            continue;
        }

        match v.get("method").and_then(Value::as_str) {
            Some("mining.set_target") => {
                if let Some(t) = v
                    .get("params")
                    .and_then(Value::as_array)
                    .and_then(|p| p.first())
                    .and_then(Value::as_str)
                {
                    target = parse_target(t)?;
                    info!("new target {t}");
                }
            }
            Some("mining.notify") => {
                if let Some(params) = v.get("params").and_then(Value::as_array) {
                    match build_work(params, &nonce1, target) {
                        Ok(w) => {
                            let epoch = work.epoch.fetch_add(1, Ordering::AcqRel) + 1;
                            let w = Work { epoch, ..w };
                            info!("new job {} (epoch {epoch})", w.job_id);
                            *work.inner.lock().unwrap() = Some(w);
                        }
                        Err(e) => warn!("ignoring malformed job: {e}"),
                    }
                }
            }
            _ => {
                // Submit acks land here; surface rejects. Name the submitter by
                // matching the reply's id against the pending-submit map.
                let id = v.get("id").and_then(Value::as_u64);
                let who = id.and_then(|i| pending.lock().unwrap().remove(&i));
                let from = who.as_deref().unwrap_or("?");
                if v.get("error").map(|e| !e.is_null()).unwrap_or(false) {
                    let reason = reject_reason(v.get("error").unwrap());
                    // Flag address rejections distinctly so the dashboard can
                    // highlight them (a misconfigured payout address rejects
                    // every share — worth shouting about).
                    if reason.to_lowercase().contains("address") {
                        warn!("share rejected [{from}]: invalid address — {reason}");
                    } else {
                        warn!("share rejected [{from}]: {reason}");
                    }
                } else if id.map(|i| i >= 100).unwrap_or(false) {
                    info!("share accepted [{from}]");
                }
            }
        }
    }
    Err(anyhow!("connection closed by pool"))
}

/// Extract a human-readable reason from a JSON-RPC error value. Stratum errors
/// are usually `[code, "message", data]`, but some pools send an object with a
/// `message` field or a bare string; fall back to the raw JSON otherwise.
fn reject_reason(err: &Value) -> String {
    if let Some(msg) = err.as_array().and_then(|a| a.get(1)).and_then(Value::as_str) {
        return msg.to_string();
    }
    if let Some(msg) = err.get("message").and_then(Value::as_str) {
        return msg.to_string();
    }
    if let Some(s) = err.as_str() {
        return s.to_string();
    }
    err.to_string()
}

/// Parse a `mining.notify` params array into [`Work`] (epoch filled in later).
fn build_work(params: &[Value], nonce1: &[u8], target: [u8; 32]) -> Result<Work> {
    let s = |i: usize| -> Result<&str> {
        params
            .get(i)
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("notify param {i} missing"))
    };

    let job_id = s(0)?.to_string();
    let version = hex::decode(s(1)?)?;
    let prevhash = hex::decode(s(2)?)?;
    let merkle = hex::decode(s(3)?)?;
    let reserved = hex::decode(s(4)?)?;
    let time_hex = s(5)?.to_string();
    let time = hex::decode(&time_hex)?;
    let bits = hex::decode(s(6)?)?;

    let mut prefix =
        Vec::with_capacity(version.len() + prevhash.len() + merkle.len() + reserved.len() + 8);
    prefix.extend_from_slice(&version);
    prefix.extend_from_slice(&prevhash);
    prefix.extend_from_slice(&merkle);
    prefix.extend_from_slice(&reserved);
    prefix.extend_from_slice(&time);
    prefix.extend_from_slice(&bits);

    if prefix.len() != 108 {
        return Err(anyhow!("header prefix is {} bytes, expected 108", prefix.len()));
    }

    Ok(Work {
        epoch: 0,
        job_id,
        prefix,
        time_hex,
        nonce1: nonce1.to_vec(),
        target,
    })
}

/// Decode a 256-bit big-endian target hex string into a fixed array.
fn parse_target(hexstr: &str) -> Result<[u8; 32]> {
    let bytes = hex::decode(hexstr)?;
    if bytes.len() != 32 {
        return Err(anyhow!("target must be 32 bytes, got {}", bytes.len()));
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::time::{Duration, Instant};

    /// Drive a full handshake against a mock pool and validate that the client
    /// publishes correct work and submits a well-formed share.
    #[test]
    fn handshake_job_and_submit() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        // Mock pool: handshake, push a target + job, then capture one submit.
        let server = std::thread::spawn(move || {
            let (sock, _) = listener.accept().unwrap();
            let mut writer = sock.try_clone().unwrap();
            let mut lines = BufReader::new(sock).lines();

            // subscribe (id 1) -> nonce1 in result[1]; authorize (id 2) -> ok.
            let sub = lines.next().unwrap().unwrap();
            assert!(sub.contains("mining.subscribe"));
            writeln!(writer, r#"{{"id":1,"result":[null,"deadbeef"],"error":null}}"#).unwrap();
            let auth = lines.next().unwrap().unwrap();
            assert!(auth.contains("mining.authorize"));
            writeln!(writer, r#"{{"id":2,"result":true,"error":null}}"#).unwrap();
            // Client also sends our custom get_jackpot request.
            let jp = lines.next().unwrap().unwrap();
            assert!(jp.contains("get_jackpot"));
            writeln!(writer, r#"{{"id":4,"result":{{"current":{{"jackpot":12.5,"pool_shares":40}},"history":[]}},"error":null}}"#).unwrap();

            // Push target and a job. prevhash/merkle/reserved are 32-byte hex.
            let h32 = "11".repeat(32);
            let target = "ff".repeat(32);
            writeln!(writer, r#"{{"id":null,"method":"mining.set_target","params":["{target}"]}}"#).unwrap();
            writeln!(
                writer,
                r#"{{"id":null,"method":"mining.notify","params":["job1","04000000","{h32}","{h32}","{h32}","abcdef00","1d00ffff",true]}}"#
            ).unwrap();

            // Capture the share the client submits.
            let submit = lines.next().unwrap().unwrap();
            submit
        });

        let client = StratumClient::connect("127.0.0.1", port, "wallet.worker", "x").unwrap();
        let work = client.work();

        // Wait for the job to propagate.
        let deadline = Instant::now() + Duration::from_secs(5);
        let w = loop {
            if let Some(w) = work.get() {
                break w;
            }
            assert!(Instant::now() < deadline, "no job received");
            std::thread::sleep(Duration::from_millis(20));
        };

        assert_eq!(w.job_id, "job1");
        assert_eq!(w.prefix.len(), 108, "header prefix must be 108 bytes");
        assert_eq!(w.nonce1, hex::decode("deadbeef").unwrap());
        assert_eq!(w.time_hex, "abcdef00");
        assert_eq!(w.target, [0xffu8; 32]);
        // Prefix begins with the version field we sent.
        assert_eq!(&w.prefix[..4], &hex::decode("04000000").unwrap()[..]);

        // The get_jackpot object was intaken and preserved.
        let jp = client.jackpot().expect("jackpot info");
        assert_eq!(jp["current"]["jackpot"].as_f64(), Some(12.5));
        assert_eq!(jp["current"]["pool_shares"].as_i64(), Some(40));

        // Submit a dummy share and verify the wire format.
        let nonce2 = [0xAAu8; 28];
        let solution = vec![0x01u8; 3 + crate::params::SOLUTION_BYTES];
        client.submit("test", &w.job_id, &w.time_hex, &nonce2, &solution).unwrap();

        let submit = server.join().unwrap();
        let v: Value = serde_json::from_str(&submit).unwrap();
        assert_eq!(v["method"], "mining.submit");
        let params = v["params"].as_array().unwrap();
        assert_eq!(params[0], "wallet.worker");
        assert_eq!(params[1], "job1");
        assert_eq!(params[2], "abcdef00");
        assert_eq!(params[3], hex::encode(nonce2));
        assert_eq!(params[4], hex::encode(&solution));
    }

    #[test]
    fn target_parsing() {
        let t = parse_target(&"00".repeat(32)).unwrap();
        assert_eq!(t, [0u8; 32]);
        assert!(parse_target("abcd").is_err());
    }

    /// After the pool drops the socket, the client should reconnect and run the
    /// handshake again (rather than failing every later submit with broken pipe).
    #[test]
    fn reconnects_after_drop() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        let server = std::thread::spawn(move || {
            // First connection: handshake, then drop the socket (simulated drop).
            {
                let (sock, _) = listener.accept().unwrap();
                let mut w = sock.try_clone().unwrap();
                let mut lines = BufReader::new(sock).lines();
                assert!(lines.next().unwrap().unwrap().contains("mining.subscribe"));
                writeln!(w, r#"{{"id":1,"result":[null,"deadbeef"],"error":null}}"#).unwrap();
                assert!(lines.next().unwrap().unwrap().contains("mining.authorize"));
                writeln!(w, r#"{{"id":2,"result":true,"error":null}}"#).unwrap();
                // Consume the custom handshake request before dropping so the
                // client's handshake writes all land before the socket closes.
                assert!(lines.next().unwrap().unwrap().contains("get_jackpot"));
                // sock + w drop here -> client sees EOF and must reconnect.
            }

            // Second connection: the client should reconnect and re-subscribe.
            let (sock2, _) = listener.accept().unwrap();
            let resub = BufReader::new(sock2).lines().next().unwrap().unwrap();
            assert!(resub.contains("mining.subscribe"), "client did not re-subscribe after drop");
        });

        let _client = StratumClient::connect("127.0.0.1", port, "wallet.worker", "x").unwrap();
        // The server thread asserts the reconnect happened; join surfaces failures.
        server.join().unwrap();
    }
}
