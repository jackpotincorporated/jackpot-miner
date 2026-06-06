//! Optional local control server (`--control-port`): a `127.0.0.1` JSON-line
//! service that lets the GUI config tool retrieve and adjust a running miner's
//! *live* controls on the fly — per-device enable + clock/power, and the CPU
//! group size + per-row enable. Bound to localhost only (no auth); intended for
//! a tool running on the same machine.
//!
//! Protocol: one JSON request object per line, one JSON reply per line.
//!   {"op":"get"}                                              -> snapshot
//!   {"op":"set_device","index":0,"enabled":true,"power_w":250,
//!                       "core_off":150,"mem_off":0}            -> {"ok":true}
//!   {"op":"set_cpu_group_size","value":4}                     -> {"ok":true}
//!   {"op":"set_cpu_row","index":0,"enabled":true}             -> {"ok":true}

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};

use crate::controls::Controls;
use crate::cpu_groups::CpuMining;
use crate::miner::Stats;

/// Serve until `running` is cleared. Non-blocking accept so shutdown is prompt.
pub fn serve(port: u16, controls: Arc<Controls>, cpu_mining: Arc<CpuMining>, stats: Arc<Stats>, running: Arc<AtomicBool>) {
    let listener = match TcpListener::bind(("127.0.0.1", port)) {
        Ok(l) => l,
        Err(e) => {
            log::warn!("control server: cannot bind 127.0.0.1:{port}: {e}");
            return;
        }
    };
    if listener.set_nonblocking(true).is_err() {
        log::warn!("control server: set_nonblocking failed; control disabled");
        return;
    }
    log::info!("control server on 127.0.0.1:{port} (live retrieve/adjust)");
    while running.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _)) => handle(stream, &controls, &cpu_mining, &stats),
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(150));
            }
            Err(e) => {
                log::debug!("control accept error: {e}");
                std::thread::sleep(Duration::from_millis(150));
            }
        }
    }
}

/// Serve one connection: read request lines (with a read timeout so a stalled
/// client can't pin the thread past shutdown) and reply to each.
fn handle(stream: TcpStream, controls: &Controls, cpu_mining: &CpuMining, stats: &Stats) {
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let mut writer = match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    };
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => return,
            Ok(_) => {}
            Err(_) => return,
        }
        if line.trim().is_empty() {
            continue;
        }
        let resp = process(line.trim(), controls, cpu_mining, stats);
        if writer.write_all(resp.to_string().as_bytes()).is_err()
            || writer.write_all(b"\n").is_err()
            || writer.flush().is_err()
        {
            return;
        }
    }
}

fn process(line: &str, controls: &Controls, cpu_mining: &CpuMining, stats: &Stats) -> Value {
    let req: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => return json!({"ok": false, "error": format!("bad json: {e}")}),
    };
    match req.get("op").and_then(Value::as_str).unwrap_or("") {
        "get" => snapshot(controls, cpu_mining, stats),
        "set_device" => set_device(&req, controls),
        "set_cpu_group_size" => {
            if let Some(n) = req.get("value").and_then(Value::as_u64) {
                cpu_mining.set_group_size(n as usize);
            }
            json!({"ok": true})
        }
        "set_cpu_row" => {
            let i = req.get("index").and_then(Value::as_u64).unwrap_or(u64::MAX) as usize;
            if let Some(e) = req.get("enabled").and_then(Value::as_bool) {
                let groups = cpu_mining.groups();
                if i < groups.len() && groups.group(i).enabled() != e {
                    cpu_mining.toggle_group(i);
                }
            }
            json!({"ok": true})
        }
        other => json!({"ok": false, "error": format!("unknown op '{other}'")}),
    }
}

fn set_device(req: &Value, controls: &Controls) -> Value {
    let i = req.get("index").and_then(Value::as_u64).unwrap_or(u64::MAX) as usize;
    if i >= controls.device_count() {
        return json!({"ok": false, "error": "device index out of range"});
    }
    let d = controls.device(i);
    if let Some(e) = req.get("enabled").and_then(Value::as_bool) {
        d.set_enabled(e);
    }
    if let Some(p) = req.get("power_w").and_then(Value::as_u64) {
        d.set_power_w(p as u32);
    }
    if let Some(c) = req.get("core_off").and_then(Value::as_i64) {
        d.set_core_off(c as i32);
    }
    if let Some(m) = req.get("mem_off").and_then(Value::as_i64) {
        d.set_mem_off(m as i32);
    }
    json!({"ok": true})
}

fn snapshot(controls: &Controls, cpu_mining: &CpuMining, stats: &Stats) -> Value {
    let cards = stats.cards();
    let devices: Vec<Value> = cards
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let d = controls.device(i);
            json!({
                "index": i,
                "label": c.label,
                "name": c.name,
                "enabled": d.enabled(),
                "power_w": d.power_w(),
                "core_off": d.core_off(),
                "mem_off": d.mem_off(),
                "watts": c.power_mw as f64 / 1000.0,
                "temp_c": c.temp_c,
                "shares": c.shares,
                "solutions": c.solutions,
            })
        })
        .collect();
    let groups = cpu_mining.groups();
    let rows: Vec<Value> = groups
        .iter()
        .enumerate()
        .map(|(i, g)| json!({"index": i, "label": g.label(), "enabled": g.enabled(), "shares": g.shares()}))
        .collect();
    json!({
        "ok": true,
        "unlocked": controls.unlocked(),
        "devices": devices,
        "cpu": {"group_size": cpu_mining.group_size(), "rows": rows},
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::miner::Stats;

    #[test]
    fn protocol_get_and_set() {
        let controls = Controls::new(1, 0, 0, 0, 0, true);
        let cpu = CpuMining::new((0..16).collect(), 4, false); // 16 cores -> tier cap 4, sizes [1,2,4]
        let stats = Stats::for_test(vec!["GPU 0".into()]);

        // get: one device, cpu rows reflect the grouping (16 cores / 4 = 4 rows).
        let v = process("{\"op\":\"get\"}", &controls, &cpu, &stats);
        assert_eq!(v["ok"], true);
        assert_eq!(v["devices"].as_array().unwrap().len(), 1);
        assert_eq!(v["cpu"]["group_size"], 4);
        assert_eq!(v["cpu"]["rows"].as_array().unwrap().len(), 4);

        // set device 0 disabled.
        assert!(controls.device(0).enabled());
        process("{\"op\":\"set_device\",\"index\":0,\"enabled\":false,\"power_w\":250}", &controls, &cpu, &stats);
        assert!(!controls.device(0).enabled());

        // out-of-range device index is rejected.
        let e = process("{\"op\":\"set_device\",\"index\":9}", &controls, &cpu, &stats);
        assert_eq!(e["ok"], false);

        // set CPU group size to the nearest available (2).
        process("{\"op\":\"set_cpu_group_size\",\"value\":2}", &controls, &cpu, &stats);
        assert_eq!(cpu.group_size(), 2);

        // toggle a CPU row on.
        assert!(!cpu.groups().group(0).enabled());
        process("{\"op\":\"set_cpu_row\",\"index\":0,\"enabled\":true}", &controls, &cpu, &stats);
        assert!(cpu.groups().group(0).enabled());

        // garbage and unknown ops are reported, not panicked.
        assert_eq!(process("not json", &controls, &cpu, &stats)["ok"], false);
        assert_eq!(process("{\"op\":\"nope\"}", &controls, &cpu, &stats)["ok"], false);
    }
}
