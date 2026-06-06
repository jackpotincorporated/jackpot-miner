//! CUDA Ethash backend (feature `ethash`, default-off) — drives miniZ's extracted
//! Ethash/etchash solver by replaying a captured launch trace, the same idea as
//! `src/cuda.rs` for Equihash 192,7 but adapted to Ethash's DAG+search shape.
//!
//! ## What was reverse-engineered (see `src/ethash/README.md`, `pearl-dump/`)
//!
//! miniZ resolves the driver API via `dlopen("libcuda.so.1")` + `dlsym`, so the
//! launch trace was captured with `pearl-dump/dlhook.so` (redirects that dlopen to
//! `cuhook.so`) + `pearl-dump/cuhook.so` (logs allocs, `cuModuleGetGlobal`
//! constants, HtoD, and — packing the `kernelParams` array by the demangled
//! signatures — every kernel's args). The recovered ABI:
//!
//!   * **DAG build** — `digitf<…>(u32 start, hash64* dag, u64 dag_items,
//!     hash64* cache, u32 cache_items, …)` launched 539× with grid 512 / block
//!     256, `start` stepping by 512*256 to cover the DAG. Only `start` varies.
//!   * **Search** — `equihash<…>(u64 start_nonce, Search_results* out)` (miniZ's
//!     name; it's Dagger-Hashimoto). The DAG pointer, header and target are NOT
//!     args — they live in `__constant__` symbols: `d_dag` (DAG base), `d_header`
//!     (32 B), `d_target` (8 B boundary), plus `d_dag_size`/`i_dag_size`/`d_maxmem`.
//!   * The **light cache** miniZ uploads (≈67 MB) is captured verbatim to a
//!     sidecar so the DAG can be rebuilt and CPU verification uses the same cache.
//!
//! ## How a solve works here
//!
//! `new()` allocates our own DAG/cache/results buffers, uploads the captured
//! cache, writes the constants (rebasing `d_dag` to our DAG), and replays the 539
//! `digitf` launches to build the DAG. `search()` writes `d_header`/`d_target`,
//! launches the `equihash` kernel over a nonce batch, and reads back the
//! `Search_results`. Every GPU nonce is re-checked on the CPU (`verify`) before it
//! could ever be submitted, so a wrong replay can only yield nothing, never a bad
//! share. `selftest()` proves the path end-to-end without a pool.

use std::collections::HashMap;
use std::ffi::{c_int, c_uint, c_void, CStr, CString};
use std::ptr;

use anyhow::{anyhow, Context, Result};

/// CPU light verifier (real Ethash; gates shares before submit).
pub mod verify;
/// Ethash stratum dialect + mining loop.
pub mod stratum;

/// miniZ's Ethash solver, 8 arches (sm_50/60/61/70/75/80/86/120). Same fatbin
/// also covers etchash/ubqhash/reth.
static FATBIN: &[u8] = include_bytes!("ethash/ethash.fatbin");

/// The captured launch trace (alloc table, constant writes, digitf + equihash
/// launches). The ≈67 MB light-cache sidecar it references (`file=…`) is read at
/// runtime from the path recorded in the trace.
static RECORDING: &str = include_str!("ethash/recording.log");

// ---- CUDA Driver API FFI (subset; mirrors src/cuda.rs) ----

type CUresult = c_int;
type CUdevice = c_int;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUdeviceptr = u64;

const CUDA_SUCCESS: CUresult = 0;
const CU_LAUNCH_PARAM_END: usize = 0x00;
const CU_LAUNCH_PARAM_BUFFER_POINTER: usize = 0x01;
const CU_LAUNCH_PARAM_BUFFER_SIZE: usize = 0x02;

extern "C" {
    fn cuInit(flags: c_uint) -> CUresult;
    fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    fn cuCtxCreate_v2(pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult;
    fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;
    fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    fn cuModuleUnload(module: CUmodule) -> CUresult;
    fn cuModuleGetFunction(f: *mut CUfunction, m: CUmodule, name: *const i8) -> CUresult;
    fn cuModuleGetGlobal_v2(dptr: *mut CUdeviceptr, bytes: *mut usize, m: CUmodule, name: *const i8) -> CUresult;
    fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    fn cuMemsetD8_v2(dptr: CUdeviceptr, uc: u8, n: usize) -> CUresult;
    fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, n: usize) -> CUresult;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, n: usize) -> CUresult;
    fn cuLaunchKernel(
        f: CUfunction, gx: c_uint, gy: c_uint, gz: c_uint,
        bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, stream: *mut c_void,
        params: *mut *mut c_void, extra: *mut *mut c_void,
    ) -> CUresult;
    fn cuCtxSynchronize() -> CUresult;
    fn cuGetErrorName(error: CUresult, s: *mut *const i8) -> CUresult;
}

fn check(code: CUresult, what: &str) -> Result<()> {
    if code == CUDA_SUCCESS {
        return Ok(());
    }
    let name = unsafe {
        let mut p: *const i8 = ptr::null();
        if cuGetErrorName(code, &mut p) == CUDA_SUCCESS && !p.is_null() {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        } else {
            format!("CUDA error {code}")
        }
    };
    Err(anyhow!("{what} failed: {name}"))
}

// ---- recording parser ----

fn hex_to_bytes(h: &str) -> Vec<u8> {
    (0..h.len() / 2)
        .map(|i| u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap_or(0))
        .collect()
}

fn triplet(s: &str) -> (u32, u32, u32) {
    let v: Vec<u32> = s.split(',').filter_map(|x| x.parse().ok()).collect();
    (v.first().copied().unwrap_or(1), v.get(1).copied().unwrap_or(1), v.get(2).copied().unwrap_or(1))
}

struct Launch {
    name: String,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared: u32,
    arg: Vec<u8>,
}

/// Everything the replay needs, distilled from the trace.
struct Recording {
    allocs: Vec<(u64, u64)>,              // (base, size)
    consts: Vec<(String, Vec<u8>)>,      // constant-symbol writes (sym -> bytes)
    cache_base: u64,
    cache_size: u64,
    cache_file: String,
    digitf: Vec<Launch>,                 // the 539 DAG-build launches
    search: Launch,                      // the first equihash launch (main pool)
    dag_base: u64,                       // d_dag value (captured DAG base)
    results_base: u64,                   // equihash arg[8..16] (captured results ptr)
}

fn field_u64(arg: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(arg[off..off + 8].try_into().unwrap())
}

fn parse_recording(text: &str) -> Result<Recording> {
    let mut allocs = Vec::new();
    let mut consts: Vec<(String, Vec<u8>)> = Vec::new();
    let (mut cache_base, mut cache_size, mut cache_file) = (0u64, 0u64, String::new());
    let mut digitf = Vec::new();
    let mut search: Option<Launch> = None;
    let mut dag_base = 0u64;

    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("[alloc] ") {
            // "<size> bytes @ 0x<base>"
            let p: Vec<&str> = rest.split_whitespace().collect();
            if let (Ok(size), Some(hex)) = (p[0].parse::<u64>(), p.get(3).and_then(|s| s.strip_prefix("0x"))) {
                if let Ok(base) = u64::from_str_radix(hex, 16) {
                    allocs.push((base, size));
                }
            }
        } else if let Some(rest) = line.strip_prefix("[htod] ") {
            // "dst=0x.. size=N [(async)] [sym=X] [file=P] first=HEX"
            let mut dst = 0u64;
            let mut size = 0u64;
            let mut sym = None;
            let mut file = None;
            let mut first = "";
            for tok in rest.split_whitespace() {
                if let Some(v) = tok.strip_prefix("dst=0x") { dst = u64::from_str_radix(v, 16).unwrap_or(0); }
                else if let Some(v) = tok.strip_prefix("size=") { size = v.parse().unwrap_or(0); }
                else if let Some(v) = tok.strip_prefix("sym=") { sym = Some(v.to_string()); }
                else if let Some(v) = tok.strip_prefix("file=") { file = Some(v.to_string()); }
                else if let Some(v) = tok.strip_prefix("first=") { first = v; }
            }
            if let Some(f) = file {
                cache_base = dst;
                cache_size = size;
                cache_file = f;
            }
            if let Some(s) = sym {
                // `first=` carries up to 32 bytes — full value for every constant here.
                let bytes = hex_to_bytes(first);
                let n = (size as usize).min(bytes.len());
                let val = bytes[..n].to_vec();
                if s == "d_dag" && val.len() >= 8 {
                    dag_base = u64::from_le_bytes(val[..8].try_into().unwrap());
                }
                // keep the last write per symbol (header/target get rewritten)
                if let Some(e) = consts.iter_mut().find(|(k, _)| *k == s) { e.1 = val; }
                else { consts.push((s, val)); }
            }
        } else if let Some(rest) = line.strip_prefix("[REC] ") {
            let mut name = "";
            let (mut g, mut b, mut sh, mut arg) = ("", "", 0u32, "");
            for (i, tok) in rest.split_whitespace().enumerate() {
                if i == 0 { name = tok; }
                else if let Some(v) = tok.strip_prefix("g=") { g = v; }
                else if let Some(v) = tok.strip_prefix("b=") { b = v; }
                else if let Some(v) = tok.strip_prefix("sh=") { sh = v.parse().unwrap_or(0); }
                else if let Some(v) = tok.strip_prefix("arg=") { arg = v; }
            }
            let l = Launch {
                name: name.to_string(),
                grid: triplet(g),
                block: triplet(b),
                shared: sh,
                arg: hex_to_bytes(arg),
            };
            if name.starts_with("_Z6digitf") {
                digitf.push(l);
            } else if name.starts_with("_Z8equihash") && search.is_none() {
                search = Some(l); // first equihash after the DAG = main-pool search kernel
            }
        }
    }

    let search = search.ok_or_else(|| anyhow!("no equihash (search) launch in recording"))?;
    if digitf.is_empty() {
        return Err(anyhow!("no digitf (DAG-build) launches in recording"));
    }
    if cache_file.is_empty() {
        return Err(anyhow!("recording has no light-cache sidecar (file=…) line"));
    }
    if dag_base == 0 {
        return Err(anyhow!("recording has no d_dag constant write"));
    }
    let results_base = if search.arg.len() >= 16 { field_u64(&search.arg, 8) } else { 0 };
    Ok(Recording { allocs, consts, cache_base, cache_size, cache_file, digitf, search, dag_base, results_base })
}

// ---- Search_results readback ----
//
// Layout recovered empirically (see `selftest`): the kernel keeps up to
// MAX_RESULTS match slots — `mix[i]` (32 B) at offset `i*32`, and the matched
// nonce as a u64 at `NONCE_BASE + i*8`. The mix array fills 0x00..0x100, so there
// are 8 slots. We zero the buffer before each launch, so an unused slot reads as
// nonce 0 / zero mix and is skipped; the CPU verifier is the real gate regardless.
const RESULTS_BYTES: usize = 1 << 20;
const MAX_RESULTS: usize = 8;
const NONCE_BASE: usize = 0x100;

#[derive(Clone, Debug, Default)]
pub struct Found {
    pub nonce: u64,
    pub mix: [u8; 32],
}

/// Convert a 256-bit big-endian boundary to miniZ's `d_target`: the top 64 bits
/// as a little-endian u64 (the kernel compares `be_top64(result) <= d_target`).
fn target8(target: &[u8; 32]) -> [u8; 8] {
    u64::from_be_bytes(target[0..8].try_into().unwrap()).to_le_bytes()
}

pub struct SearchResult {
    pub found: Vec<Found>,
}

pub struct EthashSolver {
    ctx: CUcontext,
    module: CUmodule,
    bufs: Vec<CUdeviceptr>,
    my_dag: CUdeviceptr,
    my_cache: CUdeviceptr,
    my_results: CUdeviceptr,
    /// captured device ptr -> our device ptr (DAG / cache / results)
    remap: HashMap<u64, u64>,
    digitf: Vec<(CUfunction, (u32, u32, u32), (u32, u32, u32), u32, Vec<u8>)>,
    search_fn: CUfunction,
    search_block: (u32, u32, u32),
    search_arg: Vec<u8>,
    /// raw cache bytes (kept for CPU cross-check via verify::Light::from_raw)
    cache_bytes: Vec<u8>,
    full_size: usize,
}

unsafe impl Send for EthashSolver {}

impl EthashSolver {
    /// Load the fatbin, build the DAG by replaying the captured trace. Uses the
    /// bundled recording; the light-cache sidecar is read from the path the
    /// recording names (override the recording text via `recording_override`).
    pub fn new(device_index: usize) -> Result<Self> {
        Self::with_recording(device_index, RECORDING)
    }

    pub fn with_recording(device_index: usize, recording: &str) -> Result<Self> {
        let rec = parse_recording(recording)?;
        let cache_bytes = std::fs::read(&rec.cache_file)
            .with_context(|| format!("read ethash light-cache sidecar '{}'", rec.cache_file))?;
        if cache_bytes.len() as u64 != rec.cache_size {
            return Err(anyhow!("cache sidecar size {} != recorded {}", cache_bytes.len(), rec.cache_size));
        }
        // DAG buffer size = the alloc whose base == d_dag (padded). The dataset
        // size the kernels actually index by is `d_dag_size` *128-byte* rows, so
        // the hashimoto full_size (bytes) = d_dag_size * 128 (≠ the padded alloc).
        let dag_size = rec.allocs.iter().find(|&&(b, _)| b == rec.dag_base).map(|&(_, s)| s)
            .ok_or_else(|| anyhow!("d_dag base not found in alloc table"))?;
        let dag_rows_128 = rec.consts.iter().find(|(k, _)| k == "d_dag_size")
            .and_then(|(_, v)| v.get(..4)).map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
            .ok_or_else(|| anyhow!("recording has no d_dag_size constant"))?;
        let full_size = dag_rows_128 * 128;

        unsafe {
            check(cuInit(0), "cuInit")?;
            let mut dev: CUdevice = 0;
            check(cuDeviceGet(&mut dev, device_index as c_int), "cuDeviceGet")?;
            let mut ctx: CUcontext = ptr::null_mut();
            check(cuCtxCreate_v2(&mut ctx, 0, dev), "cuCtxCreate")?;
            check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent")?;
            let mut module: CUmodule = ptr::null_mut();
            check(cuModuleLoadData(&mut module, FATBIN.as_ptr() as *const c_void), "cuModuleLoadData (ethash)")?;

            let mut bufs = Vec::new();
            let alloc = |sz: usize, bufs: &mut Vec<CUdeviceptr>| -> Result<CUdeviceptr> {
                let mut p: CUdeviceptr = 0;
                check(cuMemAlloc_v2(&mut p, sz), "cuMemAlloc")?;
                check(cuMemsetD8_v2(p, 0, sz), "cuMemsetD8")?;
                bufs.push(p);
                Ok(p)
            };
            let my_dag = alloc(dag_size as usize, &mut bufs)?;
            let my_cache = alloc(cache_bytes.len(), &mut bufs)?;
            let my_results = alloc(RESULTS_BYTES, &mut bufs)?;

            // Upload the captured light cache.
            check(cuMemcpyHtoD_v2(my_cache, cache_bytes.as_ptr() as *const c_void, cache_bytes.len()), "cuMemcpyHtoD(cache)")?;

            let mut remap = HashMap::new();
            remap.insert(rec.dag_base, my_dag);
            remap.insert(rec.cache_base, my_cache);
            if rec.results_base != 0 { remap.insert(rec.results_base, my_results); }

            // Write the constants (rebasing d_dag to our DAG buffer).
            for (sym, bytes) in &rec.consts {
                if sym == "d_header" || sym == "d_target" {
                    continue; // set per-search
                }
                let mut v = bytes.clone();
                if sym == "d_dag" && v.len() >= 8 {
                    v[..8].copy_from_slice(&my_dag.to_le_bytes());
                }
                set_global(module, sym, &v)?;
            }

            // Resolve + rebase the digitf launches, then replay them to build the DAG.
            let mut digitf = Vec::with_capacity(rec.digitf.len());
            for l in &rec.digitf {
                let f = get_function(module, &l.name)?;
                let arg = rebase(&l.arg, &remap);
                digitf.push((f, l.grid, l.block, l.shared, arg));
            }
            let search_fn = get_function(module, &rec.search.name)?;
            let search_arg = rebase(&rec.search.arg, &remap);

            let solver = Self {
                ctx, module, bufs, my_dag, my_cache, my_results, remap,
                digitf, search_fn, search_block: rec.search.block, search_arg,
                cache_bytes, full_size,
            };
            solver.build_dag()?;
            Ok(solver)
        }
    }

    /// Replay the 539 digitf launches to build the DAG (once per epoch).
    fn build_dag(&self) -> Result<()> {
        unsafe {
            check(cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")?;
            for (f, grid, block, shared, arg) in &self.digitf {
                launch(*f, *grid, *block, *shared, arg)?;
            }
            check(cuCtxSynchronize(), "cuCtxSynchronize(dag)")?;
        }
        let _ = (self.my_dag, self.my_cache, &self.remap);
        Ok(())
    }

    /// Run one nonce-batch search. `target8` is the 8-byte boundary miniZ compares
    /// (`d_target`). `grid_x` threads-blocks (each block is `search_block` threads)
    /// = `grid_x * block` nonces tested starting at `start_nonce`.
    fn search_raw(&self, header: &[u8; 32], target8: &[u8; 8], start_nonce: u64, grid_x: u32) -> Result<Vec<u8>> {
        unsafe {
            check(cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")?;
            set_global(self.module, "d_header", header)?;
            set_global(self.module, "d_target", target8)?;
            // zero a generous prefix so leftover data can't masquerade as results.
            check(cuMemsetD8_v2(self.my_results, 0, 64 * 1024), "cuMemsetD8(results)")?;

            let mut arg = self.search_arg.clone();
            arg[0..8].copy_from_slice(&start_nonce.to_le_bytes()); // start_nonce
            // arg[8..16] already rebased to my_results in new()
            launch(self.search_fn, (grid_x, 1, 1), self.search_block, 0, &arg)?;
            check(cuCtxSynchronize(), "cuCtxSynchronize(search)")?;

            let mut out = vec![0u8; 64 * 1024];
            check(cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, self.my_results, out.len()), "cuMemcpyDtoH(results)")?;
            Ok(out)
        }
    }

    /// Decode a `Search_results` buffer: mix[i] @ i*32, nonce[i] (u64) @ 0x100+i*8.
    fn decode(buf: &[u8]) -> Vec<Found> {
        let mut found = Vec::new();
        for i in 0..MAX_RESULTS {
            let no = NONCE_BASE + i * 8;
            if no + 8 > buf.len() { break; }
            let nonce = u64::from_le_bytes(buf[no..no + 8].try_into().unwrap());
            if nonce == 0 { continue; } // empty slot
            let mut mix = [0u8; 32];
            mix.copy_from_slice(&buf[i * 32..i * 32 + 32]);
            found.push(Found { nonce, mix });
        }
        found
    }

    /// Public search: full 256-bit `target`, returns decoded matches.
    pub fn search(&self, header: &[u8; 32], target: &[u8; 32], start_nonce: u64) -> Result<SearchResult> {
        let buf = self.search_raw(header, &target8(target), start_nonce, 399360)?;
        Ok(SearchResult { found: Self::decode(&buf) })
    }

    /// A CPU `Light` over the exact cache the GPU used — for cross-checking.
    pub fn cpu_light(&self) -> verify::Light {
        verify::Light::from_raw(&self.cache_bytes, self.full_size)
    }
}

impl Drop for EthashSolver {
    fn drop(&mut self) {
        unsafe {
            cuCtxSetCurrent(self.ctx);
            for &b in &self.bufs {
                cuMemFree_v2(b);
            }
            cuModuleUnload(self.module);
            cuCtxDestroy_v2(self.ctx);
        }
    }
}

/// Rebase any 8-byte field that exactly equals a captured base pointer.
fn rebase(arg: &[u8], remap: &HashMap<u64, u64>) -> Vec<u8> {
    let mut a = arg.to_vec();
    let mut off = 0;
    while off + 8 <= a.len() {
        let v = u64::from_le_bytes(a[off..off + 8].try_into().unwrap());
        if let Some(&nv) = remap.get(&v) {
            a[off..off + 8].copy_from_slice(&nv.to_le_bytes());
        }
        off += 8;
    }
    a
}

fn get_function(m: CUmodule, name: &str) -> Result<CUfunction> {
    let cname = CString::new(name).map_err(|_| anyhow!("kernel name NUL"))?;
    let mut f: CUfunction = ptr::null_mut();
    unsafe { check(cuModuleGetFunction(&mut f, m, cname.as_ptr()), "cuModuleGetFunction")? };
    Ok(f)
}

/// Write `bytes` into a `__constant__` symbol resolved by name.
fn set_global(m: CUmodule, symbol: &str, bytes: &[u8]) -> Result<()> {
    let cname = CString::new(symbol)?;
    unsafe {
        let mut dptr: CUdeviceptr = 0;
        let mut sz: usize = 0;
        check(cuModuleGetGlobal_v2(&mut dptr, &mut sz, m, cname.as_ptr()), "cuModuleGetGlobal")?;
        check(cuMemcpyHtoD_v2(dptr, bytes.as_ptr() as *const c_void, bytes.len()), "cuMemcpyHtoD(const)")?;
    }
    Ok(())
}

/// Launch via the `extra`/BUFFER_POINTER path — identical to src/cuda.rs::launch.
unsafe fn launch(f: CUfunction, grid: (u32, u32, u32), block: (u32, u32, u32), shared: u32, arg: &[u8]) -> Result<()> {
    let mut argsz = arg.len();
    let mut extra: [*mut c_void; 5] = [
        CU_LAUNCH_PARAM_BUFFER_POINTER as *mut c_void,
        arg.as_ptr() as *mut c_void,
        CU_LAUNCH_PARAM_BUFFER_SIZE as *mut c_void,
        &mut argsz as *mut usize as *mut c_void,
        CU_LAUNCH_PARAM_END as *mut c_void,
    ];
    check(
        cuLaunchKernel(f, grid.0, grid.1, grid.2, block.0, block.1, block.2, shared, ptr::null_mut(), ptr::null_mut(), extra.as_mut_ptr()),
        "cuLaunchKernel",
    )
}

/// End-to-end GPU self-test (no pool needed): build the DAG by replaying the
/// captured trace, choose a hard target on the CPU so one nonce in a batch is the
/// winner, run the GPU search, decode the `Search_results`, and confirm the GPU
/// returns that exact nonce with a mix that passes the full CPU verifier. Because
/// the CPU `Light` uses the same captured cache the GPU DAG was built from, a PASS
/// proves the GPU is computing real, correct Dagger-Hashimoto. Returns a report.
pub fn selftest(device: usize) -> Result<String> {
    let mut r = String::new();
    let solver = EthashSolver::with_recording(device, RECORDING)?;
    let light = solver.cpu_light();
    r.push_str(&format!(
        "DAG built (replayed {} digitf launches): full_size={} bytes ({} 64B rows)\n",
        solver.digitf.len(), solver.full_size, solver.full_size / 64
    ));

    let header: [u8; 32] = *b"jackpotminer-ethash-selftest!!!\0";

    // Evaluate a batch on the CPU; the lowest result becomes a hard target so the
    // GPU should return exactly that nonce.
    let n_probe = 512u64;
    let base = 0x2000_0000u64;
    let mut best = (u64::MAX, base, [0u8; 32], [0u8; 32]); // (be_top64, nonce, mix, res)
    for i in 0..n_probe {
        let (mix, res) = light.hashimoto(&header, base + i);
        let top = u64::from_be_bytes(res[0..8].try_into().unwrap());
        if top < best.0 { best = (top, base + i, mix, res); }
    }
    let (best_top, best_nonce, best_mix, best_res) = best;

    let grid = (n_probe as u32).div_ceil(solver.search_block.0);
    let buf = solver.search_raw(&header, &best_top.to_le_bytes(), base, grid)?;
    let found = EthashSolver::decode(&buf);
    r.push_str(&format!(
        "GPU search ({} nonces, target=top64 {:#018x}): {} match(es); CPU winner={:#x}\n",
        grid as u64 * solver.search_block.0 as u64, best_top, found.len(), best_nonce
    ));

    // The GPU must return the CPU winner with the matching mix, and it must pass
    // the full 256-bit CPU verifier (the real submit gate). `best_res` is the
    // winner's own result, so `result <= target` holds for it.
    let mut pass = false;
    for f in &found {
        let mix_ok = f.mix == best_mix;
        let verified = verify::verify(&light, &header, f.nonce, &best_res, Some(&f.mix));
        r.push_str(&format!("   nonce={:#x} mix_match={mix_ok} cpu_verify={verified}\n", f.nonce));
        if f.nonce == best_nonce && mix_ok && verified { pass = true; }
    }
    r.push_str(if pass {
        "\n[result] PASS \u{2713} — GPU built the DAG, found the nonce, and it CPU-verifies\n"
    } else {
        "\n[result] FAIL \u{2717}\n"
    });
    if !pass {
        return Err(anyhow!("ethash GPU search self-test did not verify:\n{r}"));
    }
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Requires a CUDA GPU + the captured cache sidecar; run explicitly:
    /// `cargo test --features ethash ethash::tests::gpu_selftest -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn gpu_selftest() {
        match selftest(0) {
            Ok(report) => println!("{report}"),
            Err(e) => panic!("ethash selftest failed: {e:?}"),
        }
    }
}
