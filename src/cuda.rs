//! CUDA Equihash 192,7 backend driving miniZ's extracted GPU solver.
//!
//! Unlike a hand-written solver, this backend loads miniZ's captured CUDA fatbin
//! (`miniz/equihash192_7.fatbin`) through the CUDA Driver API and replays its
//! exact 10-kernel Wagner pipeline. The launch sequence — grid/block/shared-mem
//! config and the pre-packed argument buffers for every kernel — was recorded at
//! the `libcuda` boundary from a live mining run and is embedded as
//! `miniz/recording.log`. See `collab/jmprcx-solver/` and
//! `/home/access/code/miniz-dump/solver_192_7/ORCHESTRATION.md` for the
//! reverse-engineering work behind this.
//!
//! ## How a solve works
//!
//! The whole pass addresses a single ~15 GB arena. At init we allocate our own
//! arena and rebase every recorded device pointer into it
//! (`mine = arena + (recorded - recorded_arena_base)`). To solve a header we:
//!   1. compute the 64-byte BLAKE2b midstate = compress(header[0..128]) and the
//!      4 varying tail bytes header[136..140] on the CPU,
//!   2. inject them into `digit_f`'s argument buffer (arg[0..64] and arg[96..100]),
//!   3. replay `cleanup → digit_f → digit_1..3 → digit_4w/5w/6w → digit_l →
//!      sort_and_compress`,
//!   4. read `digit_l`'s solution counter and container (128 consecutive u32
//!      indices per solution at offset 0) back to the host,
//!   5. hand the recovered indices to [`equihash::filter_candidates`], which
//!      canonicalises and fully verifies each candidate against the real header.
//!
//! Step 5 is the correctness guarantee: only solutions that genuinely verify for
//! this exact header are ever returned, so the backend can never yield a bad
//! share. The kernel reconstructs the 8 header bytes [128..135] (= nonce[20..27])
//! as zero, matching miniZ's nonce layout; the standard miner nonce layout keeps
//! those bytes zero, so solutions verify. Any header whose bytes [128..135] are
//! non-zero simply yields nothing (the verifier rejects the mismatched set)
//! rather than a wrong result.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::ptr;

use anyhow::{anyhow, Result};

use crate::blake;
use crate::equihash;
use crate::params::HEADER_LEN;

/// miniZ's captured Equihash 192,7 solver (sm_50/60/70/75/80/86/120 cubins; the
/// driver picks the one matching the active GPU). sm_80/86/120 carry the full
/// kernel set (all bucket configs); sm_50/60 also full; sm_70/75 carry a reduced
/// set, so on those arches only a config whose kernels are present will replay.
static FATBIN: &[u8] = include_bytes!("miniz/equihash192_7.fatbin");

/// One bundled solver configuration. miniZ ships several bucket geometries with
/// different memory footprints; we pick the highest-capacity one that fits the
/// card's free VRAM (see [`select_config`]). `table_capacity` is the number of
/// table slots (higher ⇒ fewer dropped collisions ⇒ better solution yield).
struct ConfigDef {
    name: &'static str,
    table_capacity: u64,
    recording: &'static str,
}

/// Bundled configs, captured from live miniZ runs (see `miniz/configs/README.md`).
/// Ordered low→high capacity; selection scans for the best that fits.
static CONFIGS: &[ConfigDef] = &[
    ConfigDef {
        name: "2048x16960",
        table_capacity: 34_734_080,
        recording: include_str!("miniz/configs/config_2048x16960.log"),
    },
    ConfigDef {
        name: "10000x4032",
        table_capacity: 40_325_000,
        recording: include_str!("miniz/configs/config_10000x32.log"),
    },
    ConfigDef {
        name: "12288x3392",
        table_capacity: 41_713_664,
        recording: include_str!("miniz/configs/config_12288x32.log"),
    },
];

/// VRAM held back for the CUDA context / driver and other processes.
const VRAM_HEADROOM: usize = 1_500_000_000;
/// Extra space past the highest pointer offset in a single over-allocated arena,
/// to cover the buffer that lives at that offset.
const ARENA_MARGIN: usize = 2 << 30; // 2 GiB

/// Cap on solutions read back from the container per solve. The recorded
/// container alloc is 1.5 MB = 3072 * 128 * 4 bytes, so this stays in-bounds.
const MAX_SOLS: usize = 3072;

// ---- CUDA Driver API FFI ----

type CUresult = c_int;
type CUdevice = c_int;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUstream = *mut c_void;
type CUevent = *mut c_void;
type CUdeviceptr = u64;

const CUDA_SUCCESS: CUresult = 0;

// CUfunction_attribute: opt in to >48 KB dynamic shared memory.
const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;

// cuLaunchKernel `extra` directives (miniZ passes a single pre-packed arg buffer).
const CU_LAUNCH_PARAM_END: usize = 0x00;
const CU_LAUNCH_PARAM_BUFFER_POINTER: usize = 0x01;
const CU_LAUNCH_PARAM_BUFFER_SIZE: usize = 0x02;

// The CUDA driver API, loaded at runtime via dlopen (see `crate::dylib`) rather
// than linked at build time: the SONAME `libcuda.so.1` ships with the NVIDIA
// driver (`nvcuda.dll` on Windows) and is absent on driver-less / AMD-only
// hosts. `cuda_lib()` returns `None` when it can't be opened; the public entry
// points below turn that into a clear error / empty device list, so the binary
// still builds and starts everywhere.
crate::dylib::dynamic_library! {
    lib_struct: CudaLib,
    loader: cuda_lib,
    names: ["libcuda.so.1", "libcuda.so", "nvcuda.dll"],
    fn cuInit(flags: c_uint) -> CUresult;
    fn cuDeviceGetCount(count: *mut c_int) -> CUresult;
    fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
    fn cuDeviceGetPCIBusId(pci_bus_id: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
    fn cuCtxCreate_v2(pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult;
    fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;
    fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    fn cuModuleUnload(module: CUmodule) -> CUresult;
    fn cuModuleGetFunction(hfunc: *mut CUfunction, hmod: CUmodule, name: *const c_char) -> CUresult;
    fn cuFuncSetAttribute(func: CUfunction, attrib: c_int, value: c_int) -> CUresult;
    fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    fn cuMemsetD8_v2(dptr: CUdeviceptr, uc: u8, n: usize) -> CUresult;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, byte_count: usize) -> CUresult;
    fn cuMemcpyDtoHAsync_v2(dst: *mut c_void, src: CUdeviceptr, byte_count: usize, stream: CUstream) -> CUresult;
    fn cuMemAllocHost_v2(pp: *mut *mut c_void, bytesize: usize) -> CUresult;
    fn cuMemFreeHost(p: *mut c_void) -> CUresult;
    fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;
    fn cuStreamCreate(stream: *mut CUstream, flags: c_uint) -> CUresult;
    fn cuStreamDestroy_v2(stream: CUstream) -> CUresult;
    fn cuEventCreate(event: *mut CUevent, flags: c_uint) -> CUresult;
    fn cuEventRecord(event: CUevent, stream: CUstream) -> CUresult;
    fn cuEventSynchronize(event: CUevent) -> CUresult;
    fn cuEventDestroy_v2(event: CUevent) -> CUresult;
    fn cuLaunchKernel(
        f: CUfunction,
        gx: c_uint, gy: c_uint, gz: c_uint,
        bx: c_uint, by: c_uint, bz: c_uint,
        shared_mem: c_uint,
        stream: CUstream,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
    fn cuCtxSynchronize() -> CUresult;
    fn cuGetErrorName(error: CUresult, str: *mut *const c_char) -> CUresult;
}

/// Error returned when the CUDA driver library isn't present on the host.
fn cuda_unavailable() -> anyhow::Error {
    anyhow!("CUDA driver library (libcuda.so.1) not found — is the NVIDIA driver installed?")
}

/// Turn a non-success `CUresult` into an error with the driver's symbolic name.
fn check(code: CUresult, what: &str) -> Result<()> {
    if code == CUDA_SUCCESS {
        return Ok(());
    }
    let name = unsafe {
        let mut p: *const c_char = ptr::null();
        if cuGetErrorName(code, &mut p) == CUDA_SUCCESS && !p.is_null() {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        } else {
            format!("CUDA error {code}")
        }
    };
    Err(anyhow!("{what} failed: {name}"))
}

/// Number of CUDA devices (initialises the driver as a side effect). Returns an
/// error if the CUDA driver library isn't installed.
pub fn device_count() -> Result<usize> {
    cuda_lib().ok_or_else(cuda_unavailable)?;
    unsafe {
        check(cuInit(0), "cuInit")?;
        let mut n: c_int = 0;
        check(cuDeviceGetCount(&mut n), "cuDeviceGetCount")?;
        Ok(n as usize)
    }
}

/// List CUDA devices as human-readable strings.
pub fn list_devices() -> Result<Vec<String>> {
    let n = device_count()?;
    let mut out = Vec::with_capacity(n);
    unsafe {
        for i in 0..n {
            let mut dev: CUdevice = 0;
            let name = if cuDeviceGet(&mut dev, i as c_int) == CUDA_SUCCESS {
                let mut buf = [0i8; 128];
                if cuDeviceGetName(buf.as_mut_ptr() as *mut c_char, 128, dev) == CUDA_SUCCESS {
                    CStr::from_ptr(buf.as_ptr() as *const c_char).to_string_lossy().into_owned()
                } else {
                    format!("CUDA device {i}")
                }
            } else {
                format!("CUDA device {i}")
            };
            out.push(format!("[{i}] {name}"));
        }
    }
    Ok(out)
}

// ---- BLAKE2b midstate (Equihash 192,7 personalisation) ----
//
// digit_f wants the 64-byte BLAKE2b state after compressing the first 128-byte
// header block (not finalised); blake2b_simd doesn't expose that intermediate
// state, so we compute it directly here.

const BLAKE_IV: [u64; 8] = [
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
];

const SIGMA: [[usize; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

#[inline]
#[allow(clippy::too_many_arguments)]
fn mix(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

/// One BLAKE2b compression of a 128-byte block into state `h`.
fn compress(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
    let mut m = [0u64; 16];
    for i in 0..16 {
        m[i] = u64::from_le_bytes(block[i * 8..i * 8 + 8].try_into().unwrap());
    }
    let mut v = [0u64; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&BLAKE_IV);
    v[12] ^= t as u64;
    v[13] ^= (t >> 64) as u64;
    if last {
        v[14] = !v[14];
    }
    for r in 0..12 {
        let s = &SIGMA[r];
        mix(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        mix(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        mix(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        mix(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        mix(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        mix(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        mix(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        mix(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

/// The 64-byte midstate `digit_f` expects: the BLAKE2b(192,7) state after
/// compressing header[0..128] (personalisation `"ZcashPoW"||LE32(192)||LE32(7)`,
/// digest length 48).
fn midstate(header: &[u8]) -> [u8; 64] {
    let mut personal = [0u8; 16];
    personal[..8].copy_from_slice(b"ZcashPoW");
    personal[8..12].copy_from_slice(&192u32.to_le_bytes());
    personal[12..16].copy_from_slice(&7u32.to_le_bytes());

    let mut h = BLAKE_IV;
    h[0] ^= 0x0101_0000 ^ 48; // digest_length=48, fanout=1, depth=1
    h[6] ^= u64::from_le_bytes(personal[0..8].try_into().unwrap());
    h[7] ^= u64::from_le_bytes(personal[8..16].try_into().unwrap());

    let mut block = [0u8; 128];
    block.copy_from_slice(&header[0..128]);
    compress(&mut h, &block, 128, false);

    let mut out = [0u8; 64];
    for i in 0..8 {
        out[i * 8..i * 8 + 8].copy_from_slice(&h[i].to_le_bytes());
    }
    out
}

// ---- Recording (driver-boundary launch trace) ----

/// One recorded `cuLaunchKernel`: kernel name, dims, dynamic shared bytes, and
/// the pre-packed argument buffer (device pointers already rebased into our
/// arena by [`Recording::resolve`]).
struct Launch {
    name: String,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared: u32,
    arg: Vec<u8>,
}

/// Parsed recording: device allocations plus the first full 10-kernel pass.
struct Recording {
    allocs: Vec<(u64, u64)>, // (base, size)
    pass: Vec<Launch>,       // cleanup .. sort_and_compress
}

fn triplet(s: &str) -> (u32, u32, u32) {
    let v: Vec<u32> = s.split(',').filter_map(|x| x.parse().ok()).collect();
    (
        v.first().copied().unwrap_or(1),
        v.get(1).copied().unwrap_or(1),
        v.get(2).copied().unwrap_or(1),
    )
}

fn parse_recording(text: &str) -> Result<Recording> {
    let mut allocs = Vec::new();
    let mut launches = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("[alloc] ") {
            // "<size> bytes @ 0x<base>"
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if parts.len() >= 4 {
                if let (Ok(size), Some(hex)) = (parts[0].parse::<u64>(), parts[3].strip_prefix("0x")) {
                    if let Ok(base) = u64::from_str_radix(hex, 16) {
                        allocs.push((base, size));
                    }
                }
            }
        } else if let Some(rest) = line.strip_prefix("[REC] ") {
            // "<name> g=.. b=.. sh=N sz=N arg=<hex>"
            let mut name = "";
            let (mut g, mut b, mut sh, mut arg) = ("", "", 0u32, "");
            for (i, tok) in rest.split_whitespace().enumerate() {
                if i == 0 {
                    name = tok;
                } else if let Some(v) = tok.strip_prefix("g=") {
                    g = v;
                } else if let Some(v) = tok.strip_prefix("b=") {
                    b = v;
                } else if let Some(v) = tok.strip_prefix("sh=") {
                    sh = v.parse().unwrap_or(0);
                } else if let Some(v) = tok.strip_prefix("arg=") {
                    arg = v;
                }
            }
            let bytes = (0..arg.len() / 2)
                .map(|i| u8::from_str_radix(&arg[2 * i..2 * i + 2], 16).unwrap_or(0))
                .collect();
            launches.push(Launch {
                name: name.to_string(),
                grid: triplet(g),
                block: triplet(b),
                shared: sh,
                arg: bytes,
            });
        }
    }

    // Take the first full pass: cleanup .. sort_and_compress.
    let start = launches
        .iter()
        .position(|l| l.name.contains("7cleanup"))
        .ok_or_else(|| anyhow!("no cleanup launch in recording"))?;
    let end = start
        + launches[start..]
            .iter()
            .position(|l| l.name.contains("sort_and_compress"))
            .ok_or_else(|| anyhow!("no sort_and_compress in recording"))?;
    let pass: Vec<Launch> = launches.drain(start..=end).collect();
    Ok(Recording { allocs, pass })
}

/// Bytes at the start of a kernel's arg buffer that are by-value (not device
/// pointers) and must NOT be rebased.
fn byval_prefix(name: &str) -> usize {
    if name.contains("7digit_f") {
        64 // two ulonglong4 (the BLAKE2b midstate)
    } else if name.contains("sort_and_compress") {
        112 // SHA256_CTX by value
    } else {
        0
    }
}

/// Recorded device-address range used by miniZ's arena allocations.
fn in_dev(v: u64) -> bool {
    (0x7000_0000_0000..0x8000_0000_0000).contains(&v)
}

/// A device buffer the pass references, with the highest offset dereferenced
/// into it (`base`/`size` are the recorded allocation; `high_water` is the
/// largest `ptr - base` seen).
#[derive(Clone)]
struct RefBuf {
    base: u64,
    size: u64,
    high_water: u64,
}

impl Recording {
    /// The distinct device buffers this pass references — the small configs use
    /// several separate allocations, the 12288 config one big arena. For each we
    /// track the highest offset dereferenced into it (its owner is the first
    /// recorded alloc that contains the pointer; reused addresses share a base,
    /// so the choice is unambiguous for rebasing).
    fn referenced_buffers(&self) -> Vec<RefBuf> {
        let mut refs: Vec<RefBuf> = Vec::new();
        for l in &self.pass {
            let mut off = byval_prefix(&l.name);
            while off + 8 <= l.arg.len() {
                let v = u64::from_le_bytes(l.arg[off..off + 8].try_into().unwrap());
                if in_dev(v) {
                    if let Some(&(b, s)) = self.allocs.iter().find(|&&(b, s)| v >= b && v < b + s) {
                        let hw = v - b;
                        match refs.iter_mut().find(|r| r.base == b) {
                            Some(r) => r.high_water = r.high_water.max(hw),
                            None => refs.push(RefBuf { base: b, size: s, high_water: hw }),
                        }
                    }
                }
                off += 8;
            }
        }
        refs
    }
}

/// Minimum device memory a config needs.
///
/// - One over-allocated arena (the 12288 config): only the region up to the
///   highest pointer plus the buffer there is touched, so we can cap it.
/// - Several dedicated buffers (the small configs): each is fully indexed by its
///   kernels, so all must be allocated at full size.
fn required_bytes(refs: &[RefBuf]) -> usize {
    if refs.len() == 1 {
        (refs[0].high_water as usize + ARENA_MARGIN).min(refs[0].size as usize)
    } else {
        refs.iter().map(|r| r.size as usize).sum()
    }
}

/// A chosen, parsed config ready to allocate and replay.
struct Chosen {
    name: &'static str,
    rec: Recording,
    refs: Vec<RefBuf>,
    required: usize,
}

/// Configs that fit `budget` bytes of usable VRAM, **highest table capacity
/// first**. `ZCL_CUDA_CONFIG=<name>` forces exactly one (even if it doesn't fit).
///
/// The caller picks the first of these whose kernels are actually present in the
/// active GPU's cubin (see [`config_present`]) — the legacy-arch cubins (sm_70/75)
/// ship a reduced kernel set, so the highest-capacity VRAM-fitting config may not
/// exist there and we fall through to one that does.
fn candidate_configs(budget: usize) -> Result<Vec<Chosen>> {
    let forced = std::env::var("ZCL_CUDA_CONFIG").ok();
    let mut cands: Vec<(u64, Chosen)> = Vec::new();
    let mut min_required = usize::MAX;

    for def in CONFIGS {
        let rec = parse_recording(def.recording)?;
        let refs = rec.referenced_buffers();
        let required = required_bytes(&refs);
        min_required = min_required.min(required);

        if forced.as_deref() == Some(def.name) {
            return Ok(vec![Chosen { name: def.name, rec, refs, required }]);
        }
        if forced.is_none() && required <= budget {
            cands.push((def.table_capacity, Chosen { name: def.name, rec, refs, required }));
        }
    }

    if let Some(f) = forced {
        return Err(anyhow!("ZCL_CUDA_CONFIG='{f}' is not a known config"));
    }
    // Highest table capacity first.
    cands.sort_by(|a, b| b.0.cmp(&a.0));
    if cands.is_empty() {
        return Err(anyhow!(
            "insufficient VRAM: ~{:.1} GB usable, but the smallest solver config needs ~{:.1} GB",
            budget as f64 / 1e9,
            min_required as f64 / 1e9
        ));
    }
    Ok(cands.into_iter().map(|(_, c)| c).collect())
}

/// Whether every kernel a config replays is present in `module` (this GPU's
/// cubin). Probing `cuModuleGetFunction` only resolves a handle — it allocates
/// nothing — so this is a cheap pre-flight before committing to a config.
unsafe fn config_present(module: CUmodule, rec: &Recording) -> bool {
    let mut seen = std::collections::HashSet::new();
    for l in &rec.pass {
        if !seen.insert(l.name.as_str()) {
            continue;
        }
        let Ok(cname) = CString::new(l.name.as_str()) else {
            return false;
        };
        let mut f: CUfunction = ptr::null_mut();
        if cuModuleGetFunction(&mut f, module, cname.as_ptr()) != CUDA_SUCCESS {
            return false;
        }
    }
    true
}

/// From VRAM-fitting candidates (best first), pick the first fully present in the
/// loaded `module`. On full arches (sm_50/60/80/86/120) this is the highest-
/// capacity config; on reduced arches (sm_70/75) it falls through to a present one.
fn pick_present_config(candidates: Vec<Chosen>, module: CUmodule) -> Result<Chosen> {
    let n = candidates.len();
    for c in candidates {
        if unsafe { config_present(module, &c.rec) } {
            return Ok(c);
        }
    }
    Err(anyhow!(
        "none of the {n} VRAM-fitting solver config(s) is fully present in this GPU's cubin \
         (reduced legacy-arch kernel set — try a smaller config or a fuller-arch GPU)"
    ))
}

// ---- Solver ----

/// A persistent CUDA solver bound to one device + context, holding the loaded
/// fatbin, its device buffers, and the resolved/rebased launch sequence.
pub struct CudaSolver {
    ctx: CUcontext,
    _module: CUmodule,
    /// Device buffers we allocated for the selected config (freed in `Drop`).
    bufs: Vec<CUdeviceptr>,
    /// Resolved launches: (function, grid, block, shared, rebased arg buffer).
    launches: Vec<(CUfunction, (u32, u32, u32), (u32, u32, u32), u32, Vec<u8>)>,
    /// Index of `digit_f` within `launches` (where we inject midstate + tail).
    digit_f: usize,
    /// Device pointers for `digit_l`'s solution counter and container.
    counter_ptr: CUdeviceptr,
    container_ptr: CUdeviceptr,

    // --- pipelining (used by enqueue/drain) ---
    /// Stream all pipelined launches + async copies run on.
    stream: CUstream,
    /// Double-buffered pinned host memory for the counter and container readback.
    host_counter: [*mut u32; 2],
    host_container: [*mut u32; 2],
    /// Completion event per buffer slot.
    event: [CUevent; 2],
    /// Buffer slot the next `enqueue` will use.
    slot: usize,
    /// The pass currently in flight: (its buffer slot, the header that produced it).
    pending: Option<(usize, Vec<u8>)>,
    /// Per-card GPU control (clocks/power/readout); None if unavailable.
    tuner: Option<Box<dyn crate::gpu_tune::GpuTuner>>,
    /// Whether tuning changed clock/power state (so we restore it on Drop).
    tuned: bool,
    /// Core / memory clock offsets (MHz) chosen by `--auto-tune`, for the
    /// dashboard to reflect; `None` if auto-tune didn't run or was skipped.
    auto_core_off: Option<i32>,
    auto_mem_off: Option<i32>,
}

// The context is created on, and only used from, the worker thread that owns the
// solver, so it is safe to move the solver to that thread.
unsafe impl Send for CudaSolver {}

impl CudaSolver {
    /// Initialise the driver, create a context on `device_index`, load the miniZ
    /// fatbin, select the config that fits free VRAM, allocate its buffers, and
    /// rebase the recorded launch sequence.
    pub fn new(device_index: usize) -> Result<Self> {
        cuda_lib().ok_or_else(cuda_unavailable)?;
        unsafe {
            check(cuInit(0), "cuInit")?;
            let mut dev: CUdevice = 0;
            check(cuDeviceGet(&mut dev, device_index as c_int), "cuDeviceGet")?;

            // Resolve this card's PCI bus id so we can open a GPU control handle
            // (clocks/power/readout) for the matching physical GPU — CUDA and the
            // driver's NVML index orderings can differ — then apply the tuning
            // policy to that card.
            let tuner = {
                let mut buf = [0 as c_char; 32];
                if cuDeviceGetPCIBusId(buf.as_mut_ptr(), buf.len() as c_int, dev) == CUDA_SUCCESS {
                    let bus = CStr::from_ptr(buf.as_ptr()).to_string_lossy();
                    crate::gpu_tune::open(&bus)
                } else {
                    None
                }
            };
            let tuned = match &tuner {
                Some(t) => crate::gpu_tune::apply(t.as_ref(), device_index),
                None => false,
            };

            let mut ctx: CUcontext = ptr::null_mut();
            check(cuCtxCreate_v2(&mut ctx, 0, dev), "cuCtxCreate")?;
            check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent")?;

            // Choose the config by available VRAM (highest capacity that fits).
            let mut free = 0usize;
            let mut total = 0usize;
            check(cuMemGetInfo_v2(&mut free, &mut total), "cuMemGetInfo")?;
            let budget = free.saturating_sub(VRAM_HEADROOM);
            let candidates = candidate_configs(budget)?;

            // Load the module first so we can pick a config whose kernels this
            // GPU's cubin actually contains (legacy arches ship a reduced set).
            let mut module: CUmodule = ptr::null_mut();
            check(
                cuModuleLoadData(&mut module, FATBIN.as_ptr() as *const c_void),
                "cuModuleLoadData (does this GPU's arch match the fatbin's sm_50/60/70/75/80/86/120 cubins?)",
            )?;

            let chosen = pick_present_config(candidates, module)?;
            log::info!(
                "CUDA device {device_index}: config '{}' ({} ref buffer(s), ~{:.1} GB; {:.1} GB free)",
                chosen.name,
                chosen.refs.len(),
                chosen.required as f64 / 1e9,
                free as f64 / 1e9,
            );

            // Allocate a device buffer per referenced recording buffer. A lone
            // over-allocated arena is capped to what the pipeline actually
            // touches; dedicated buffers are allocated at full size. `rebase`
            // maps each recorded pointer to its owning new buffer.
            let single_arena = chosen.refs.len() == 1;
            let mut bufs = Vec::with_capacity(chosen.refs.len());           // (orig_base, orig_size, my_base)
            let mut device_bufs = Vec::with_capacity(chosen.refs.len());
            let mut remaining = budget;
            for r in &chosen.refs {
                let want = if single_arena {
                    (r.high_water as usize + ARENA_MARGIN).min(r.size as usize).min(remaining)
                } else {
                    (r.size as usize).min(remaining)
                };
                let mut p: CUdeviceptr = 0;
                check(cuMemAlloc_v2(&mut p, want), "cuMemAlloc")?;
                check(cuMemsetD8_v2(p, 0, want), "cuMemsetD8")?;
                remaining = remaining.saturating_sub(want);
                bufs.push((r.base, r.size, p));
                device_bufs.push(p);
            }

            let rebase = |v: u64| -> Option<u64> {
                bufs.iter()
                    .find(|&&(b, s, _)| v >= b && v < b + s)
                    .map(|&(b, _, mb)| mb + (v - b))
            };

            // Resolve every kernel, opt into large dynamic shared memory, and
            // rebase the device pointers in each arg buffer once.
            let mut launches = Vec::with_capacity(chosen.rec.pass.len());
            let mut digit_f = None;
            for (idx, l) in chosen.rec.pass.iter().enumerate() {
                let cname = CString::new(l.name.as_str()).map_err(|_| anyhow!("kernel name has NUL"))?;
                let mut f: CUfunction = ptr::null_mut();
                check(
                    cuModuleGetFunction(&mut f, module, cname.as_ptr()),
                    "cuModuleGetFunction",
                )?;
                if l.shared > 0 {
                    cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, l.shared as c_int);
                }

                let mut arg = l.arg.clone();
                let mut off = byval_prefix(&l.name);
                while off + 8 <= arg.len() {
                    let v = u64::from_le_bytes(arg[off..off + 8].try_into().unwrap());
                    if in_dev(v) {
                        if let Some(nv) = rebase(v) {
                            arg[off..off + 8].copy_from_slice(&nv.to_le_bytes());
                        }
                    }
                    off += 8;
                }
                if l.name.contains("7digit_f") {
                    digit_f = Some(idx);
                }
                launches.push((f, l.grid, l.block, l.shared, arg));
            }

            let digit_f = digit_f.ok_or_else(|| anyhow!("no digit_f in recorded pass"))?;

            // digit_l: arg[8..16] = solution counter*, arg[16..24] = container*
            // (already rebased above).
            let dl = chosen
                .rec
                .pass
                .iter()
                .position(|l| l.name.contains("7digit_l"))
                .ok_or_else(|| anyhow!("no digit_l in recorded pass"))?;
            let dl_arg = &launches[dl].4;
            if dl_arg.len() < 24 {
                return Err(anyhow!("digit_l arg buffer too short"));
            }
            let counter_ptr = u64::from_le_bytes(dl_arg[8..16].try_into().unwrap());
            let container_ptr = u64::from_le_bytes(dl_arg[16..24].try_into().unwrap());

            // Pipelining resources: a stream + per-slot completion event and
            // pinned host buffers for async readback (overlaps the next pass's
            // GPU work with this pass's host-side verification).
            let mut stream: CUstream = ptr::null_mut();
            check(cuStreamCreate(&mut stream, 0), "cuStreamCreate")?;
            let alloc_host = |bytes: usize| -> Result<*mut u32> {
                let mut p: *mut c_void = ptr::null_mut();
                check(cuMemAllocHost_v2(&mut p, bytes), "cuMemAllocHost")?;
                Ok(p as *mut u32)
            };
            let mut event = [ptr::null_mut(); 2];
            let mut host_counter = [ptr::null_mut(); 2];
            let mut host_container = [ptr::null_mut(); 2];
            for s in 0..2 {
                check(cuEventCreate(&mut event[s], 0), "cuEventCreate")?;
                host_counter[s] = alloc_host(64)?;
                host_container[s] = alloc_host(MAX_SOLS * 128 * 4)?;
            }

            let mut solver = Self {
                ctx,
                _module: module,
                bufs: device_bufs,
                launches,
                digit_f,
                counter_ptr,
                container_ptr,
                stream,
                host_counter,
                host_container,
                event,
                slot: 0,
                pending: None,
                tuner,
                tuned,
                auto_core_off: None,
                auto_mem_off: None,
            };

            // Optionally find this card's fastest stable clock offsets.
            if crate::gpu_tune::auto_tune_enabled() {
                if let Some((core, mem)) = solver.auto_tune_speed() {
                    solver.auto_core_off = Some(core);
                    solver.auto_mem_off = Some(mem);
                }
            }

            Ok(solver)
        }
    }

    /// Current board power draw in watts, or `None` if unavailable.
    pub fn power_watts(&self) -> Option<f64> {
        self.tuner.as_ref().and_then(|t| t.watts())
    }

    /// Current GPU core temperature in °C, or `None` if unavailable.
    pub fn temperature_c(&self) -> Option<u32> {
        self.tuner.as_ref().and_then(|t| t.temperature_c())
    }

    /// This card's product name (e.g. "NVIDIA GeForce RTX 5080"), if available.
    pub fn device_name(&self) -> Option<String> {
        self.tuner.as_ref().map(|t| t.name())
    }

    /// Currently enforced power limit in watts, or `None` if unavailable.
    pub fn current_power_limit_w(&self) -> Option<u32> {
        self.tuner.as_ref().and_then(|t| t.current_power_limit_w())
    }

    /// This card's (min, max) settable power limit in watts, or `None`.
    pub fn power_limit_range_w(&self) -> Option<(u32, u32)> {
        self.tuner.as_ref().and_then(|t| t.power_limit_range_w())
    }

    /// Apply live hardware controls from the dashboard: absolute core/memory VF
    /// offsets (MHz) and an absolute power-limit target (watts; 0 = leave the
    /// power limit alone). Best-effort — needs elevated privileges, and the
    /// tuner clamps each value to the card's allowed range.
    pub fn apply_hw_controls(&self, core_off: i32, mem_off: i32, power_w: u32) {
        if let Some(t) = &self.tuner {
            t.set_core_offset_mhz(core_off);
            t.set_mem_offset_mhz(mem_off);
            if power_w > 0 {
                t.set_power_limit_w(power_w);
            }
        }
    }

    /// Sweep this card's core clock offset, then its memory clock offset, each
    /// upward to maximise solve throughput, locking in the best stable value of
    /// each. Best-effort: needs root to change clocks, and stops a sweep at the
    /// first sign of instability (a kernel error or no valid solutions). Records
    /// every step for the dashboard. Runs once at startup.
    pub fn auto_tune_speed(&self) -> Option<(i32, i32)> {
        use log::info;
        use std::time::Instant;

        // Per-step sample window. Longer windows average out per-pass jitter and
        // power fluctuation, so the efficiency/throughput numbers are stable.
        const SAMPLE_SECS: f64 = 5.0;

        let ctrl = self.tuner.as_ref()?;

        // A header whose nonce bytes [128..136) are zero, so the GPU yields valid
        // solutions we can both rate and use as a stability check.
        let mut header = vec![0x42u8; HEADER_LEN];
        for b in &mut header[128..136] {
            *b = 0;
        }

        // Throughput (passes/s), solution rate (Sol/s) and average board power
        // (W) over `secs`. `None` on a kernel fault (⇒ unstable).
        let measure = |secs: f64| -> Option<(f64, f64, f64)> {
            let t = Instant::now();
            let (mut passes, mut sols) = (0u64, 0u64);
            let (mut wsum, mut wcnt) = (0.0f64, 0u64);
            while t.elapsed().as_secs_f64() < secs {
                match self.solve(&header) {
                    Ok(s) => {
                        passes += 1;
                        sols += s.len() as u64;
                    }
                    Err(_) => return None,
                }
                if let Some(w) = self.power_watts() {
                    wsum += w;
                    wcnt += 1;
                }
            }
            let el = t.elapsed().as_secs_f64().max(1e-9);
            let watts = if wcnt > 0 { wsum / wcnt as f64 } else { 0.0 };
            Some((passes as f64 / el, sols as f64 / el, watts))
        };

        if !ctrl.set_core_offset_mhz(0).applied() {
            info!("auto-tune: cannot set GPU clock offset (needs root/Administrator) — skipping");
            return None;
        }
        ctrl.set_mem_offset_mhz(0);

        info!("auto-tune: sampling solve rate ({SAMPLE_SECS:.0}s/step) before sweeping clock offsets...");

        let base_rate = match measure(SAMPLE_SECS) {
            Some((rate, sol_s, _)) if sol_s > 0.0 => rate,
            _ => {
                info!("auto-tune: baseline produced no solutions — skipping");
                return None;
            }
        };

        // Sweep one knob upward via `set` until throughput stops improving (two
        // stale steps) or the card goes unstable, returning the best offset.
        // `set_*_offset` clamps to the driver's allowed range.
        let mut best_rate = base_rate;
        let mut sweep = |set: &dyn Fn(i32) -> bool, step: i32, cap: i32| -> i32 {
            let mut best = 0i32;
            let mut stale = 0;
            let mut off = 0;
            while off < cap {
                off += step;
                if !set(off) {
                    break;
                }
                match measure(SAMPLE_SECS) {
                    Some((rate, sol_s, _)) if sol_s > 0.0 => {
                        if rate > best_rate * 1.005 {
                            best_rate = rate;
                            best = off;
                            stale = 0;
                        } else {
                            stale += 1;
                            if stale >= 2 {
                                break;
                            }
                        }
                    }
                    _ => break, // error or zero solutions ⇒ unstable; stop here
                }
            }
            best
        };

        // Phase 1: core offset (memory held at 0).
        let best_core = sweep(&|o| ctrl.set_core_offset_mhz(o).applied(), 45, 450);
        // Lock the winning core offset before sweeping memory.
        ctrl.set_core_offset_mhz(best_core);

        // Phase 2: memory offset (core held at the winner).
        let best_mem = sweep(&|o| ctrl.set_mem_offset_mhz(o).applied(), 200, 1600);
        ctrl.set_mem_offset_mhz(best_mem);

        info!(
            "auto-tune: core {best_core:+} MHz, memory {best_mem:+} MHz ({:+.1}% solve rate)",
            (best_rate / base_rate - 1.0) * 100.0
        );
        Some((best_core, best_mem))
    }

    /// The core clock offset (MHz) `--auto-tune` settled on, if it ran.
    pub fn tuned_core_offset(&self) -> Option<i32> {
        self.auto_core_off
    }

    /// The memory clock offset (MHz) `--auto-tune` settled on, if it ran.
    pub fn tuned_mem_offset(&self) -> Option<i32> {
        self.auto_mem_off
    }

    /// Current (SM core, memory) clock in MHz, each `None` if unavailable.
    pub fn current_clocks_mhz(&self) -> (Option<u32>, Option<u32>) {
        match &self.tuner {
            Some(t) => (t.core_clock_mhz(), t.mem_clock_mhz()),
            None => (None, None),
        }
    }

    /// Inject the per-header midstate + tail into `digit_f` and launch all 10
    /// kernels back-to-back on `stream`. They have strict data dependencies and
    /// the stream is in-order, so no host sync is needed between them.
    fn launch_pass(&self, header: &[u8], stream: CUstream) -> Result<()> {
        let mut digit_f_arg = self.launches[self.digit_f].4.clone();
        digit_f_arg[0..64].copy_from_slice(&midstate(header));
        digit_f_arg[96..100].copy_from_slice(&[header[136], header[137], header[138], header[139]]);
        unsafe {
            for (idx, (f, grid, block, shared, arg)) in self.launches.iter().enumerate() {
                let buf = if idx == self.digit_f { &digit_f_arg } else { arg };
                launch(*f, *grid, *block, *shared, buf, stream)?;
            }
        }
        Ok(())
    }

    /// Synchronous one-shot replay: launch the pipeline and block on the readback.
    /// Used by `--benchmark`/`--gpu-debug`/`--selftest`; the mining loop uses the
    /// pipelined [`enqueue`](Self::enqueue) path instead.
    fn run_pipeline(&self, header: &[u8]) -> Result<Vec<u32>> {
        unsafe {
            check(cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")?;
            self.launch_pass(header, ptr::null_mut())?;

            // Synchronous DtoH on the default stream blocks until the pipeline
            // finishes.
            let mut counter = [0u32; 1];
            check(
                cuMemcpyDtoH_v2(counter.as_mut_ptr() as *mut c_void, self.counter_ptr, 4),
                "cuMemcpyDtoH(counter)",
            )?;
            let n = (counter[0] as usize).min(MAX_SOLS);
            if n == 0 {
                return Ok(Vec::new());
            }
            let mut out = vec![0u32; n * 128];
            check(
                cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, self.container_ptr, n * 128 * 4),
                "cuMemcpyDtoH(container)",
            )?;
            Ok(out)
        }
    }

    /// Solve the puzzle for `header` (140 bytes), returning verified solutions.
    /// One-shot/synchronous; for mining throughput use [`enqueue`](Self::enqueue).
    pub fn solve(&self, header: &[u8]) -> Result<Vec<Vec<u32>>> {
        assert_eq!(header.len(), HEADER_LEN);
        let recovered = self.run_pipeline(header)?;
        if recovered.is_empty() {
            return Ok(Vec::new());
        }
        let base = blake::base_state(header);
        Ok(equihash::filter_candidates(&base, &recovered))
    }

    /// Pipelined solve: launch `header`'s pass and asynchronously copy its results
    /// back, then return the solutions of the pass enqueued one call ago (or empty
    /// on the first call). The GPU runs the new pass while the host verifies the
    /// previous one, keeping the device ~100% busy. Drain the final in-flight pass
    /// with [`drain`](Self::drain). Returns solutions for the header passed to the
    /// *previous* `enqueue`.
    pub fn enqueue(&mut self, header: &[u8]) -> Result<Vec<Vec<u32>>> {
        assert_eq!(header.len(), HEADER_LEN);
        let slot = self.slot;
        unsafe {
            check(cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")?;
            self.launch_pass(header, self.stream)?;
            // Queue the readback after the pass on the same stream, then mark it.
            check(
                cuMemcpyDtoHAsync_v2(self.host_counter[slot] as *mut c_void, self.counter_ptr, 4, self.stream),
                "cuMemcpyDtoHAsync(counter)",
            )?;
            check(
                cuMemcpyDtoHAsync_v2(self.host_container[slot] as *mut c_void, self.container_ptr, MAX_SOLS * 128 * 4, self.stream),
                "cuMemcpyDtoHAsync(container)",
            )?;
            check(cuEventRecord(self.event[slot], self.stream), "cuEventRecord")?;
        }

        // While the GPU runs the pass just queued, verify the previous one.
        let result = match self.pending.take() {
            Some((prev, prev_header)) => self.read_slot(prev, &prev_header)?,
            None => Vec::new(),
        };
        self.pending = Some((slot, header.to_vec()));
        self.slot = 1 - slot;
        Ok(result)
    }

    /// Wait for and verify the last in-flight pass (after the final `enqueue`).
    pub fn drain(&mut self) -> Result<Vec<Vec<u32>>> {
        match self.pending.take() {
            Some((prev, prev_header)) => self.read_slot(prev, &prev_header),
            None => Ok(Vec::new()),
        }
    }

    /// Block on slot `s`'s completion event, then verify its recovered indices
    /// against `header`. Reads pinned host memory filled by `enqueue`.
    fn read_slot(&self, s: usize, header: &[u8]) -> Result<Vec<Vec<u32>>> {
        unsafe {
            check(cuEventSynchronize(self.event[s]), "cuEventSynchronize")?;
            let n = (*self.host_counter[s] as usize).min(MAX_SOLS);
            if n == 0 {
                return Ok(Vec::new());
            }
            let recovered = std::slice::from_raw_parts(self.host_container[s], n * 128);
            let base = blake::base_state(header);
            Ok(equihash::filter_candidates(&base, recovered))
        }
    }

    /// Time each GPU kernel individually (sync between launches).
    pub fn profile(&self, header: &[u8]) -> Result<()> {
        use log::info;
        use std::time::Instant;

        let mid = midstate(header);
        let tail4 = [header[136], header[137], header[138], header[139]];
        unsafe {
            check(cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")?;
            check(cuCtxSynchronize(), "cuCtxSynchronize")?;
            for (idx, (f, grid, block, shared, arg)) in self.launches.iter().enumerate() {
                let mut a = arg.clone();
                if idx == self.digit_f {
                    a[0..64].copy_from_slice(&mid);
                    a[96..100].copy_from_slice(&tail4);
                }
                let t = Instant::now();
                launch(*f, *grid, *block, *shared, &a, ptr::null_mut())?;
                check(cuCtxSynchronize(), "cuCtxSynchronize")?;
                info!("  kernel {idx:2} {:>6.1} ms", t.elapsed().as_secs_f64() * 1000.0);
            }
        }
        Ok(())
    }
}

impl Drop for CudaSolver {
    fn drop(&mut self) {
        // Restore default clocks/power if we changed them.
        if self.tuned {
            if let Some(t) = &self.tuner {
                t.reset();
            }
        }
        unsafe {
            cuCtxSetCurrent(self.ctx);
            for s in 0..2 {
                cuEventDestroy_v2(self.event[s]);
                cuMemFreeHost(self.host_counter[s] as *mut c_void);
                cuMemFreeHost(self.host_container[s] as *mut c_void);
            }
            cuStreamDestroy_v2(self.stream);
            for &b in &self.bufs {
                cuMemFree_v2(b);
            }
            cuModuleUnload(self._module);
            cuCtxDestroy_v2(self.ctx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KNOWN_HEADER: &str = "040000002ba84c97ffc202b55a5843d55837d256fdc32410390b8e95502bd8f648040000cb560c7083a13e06273570350805668e83c3e2362e39e131612fead6f4ea9937a19ceba5b597e2217d7e0c53ba24de3d36b92cf97743550c2745c9464f4dc847ba9e1e6a34cf101e80032bb40ae5118877fccacf8d961e648f6a228d0000000000000000ce856809";

    fn known_header() -> Vec<u8> {
        let h: Vec<u8> = (0..KNOWN_HEADER.len() / 2)
            .map(|i| u8::from_str_radix(&KNOWN_HEADER[2 * i..2 * i + 2], 16).unwrap())
            .collect();
        assert_eq!(h.len(), HEADER_LEN);
        h
    }

    /// End-to-end GPU harvest on a real, pool-accepted header (job 19ae0) for the
    /// auto-selected (default) config: drive the full pipeline and confirm at
    /// least one solution verifies. Ignored by default — needs an NVIDIA GPU with
    /// ~10 GB free whose arch matches the fatbin (sm_50/60/70/75/80/86/120). Run with:
    ///   cargo test --no-default-features --features cuda -- --ignored --nocapture
    #[test]
    #[ignore]
    fn harvests_known_solution() {
        let header = known_header();
        let solver = CudaSolver::new(0).expect("init CUDA device 0");
        let sols = solver.solve(&header).expect("solve");
        assert!(!sols.is_empty(), "expected at least one harvested solution");
        let base = blake::base_state(&header);
        for s in &sols {
            assert!(equihash::is_valid_solution(&base, s), "harvested solution must verify");
        }
        eprintln!("harvested {} valid solution(s) from the GPU", sols.len());
    }

    /// Drive every bundled config (forced via `ZCL_CUDA_CONFIG`) on the known
    /// header and confirm each replays cleanly (no OOB from per-alloc rebasing)
    /// and returns only valid solutions. Ignored by default (needs a GPU).
    #[test]
    #[ignore]
    fn all_configs_replay_cleanly() {
        let header = known_header();
        let base = blake::base_state(&header);
        for cfg in CONFIGS {
            std::env::set_var("ZCL_CUDA_CONFIG", cfg.name);
            let solver = CudaSolver::new(0).unwrap_or_else(|e| panic!("init config {}: {e}", cfg.name));
            let sols = solver.solve(&header).unwrap_or_else(|e| panic!("solve {}: {e}", cfg.name));
            for s in &sols {
                assert!(equihash::is_valid_solution(&base, s), "config {} produced an invalid solution", cfg.name);
            }
            eprintln!("config {:<12} -> {} valid solution(s)", cfg.name, sols.len());
        }
        std::env::remove_var("ZCL_CUDA_CONFIG");
    }

    /// The pipelined `enqueue`/`drain` path must produce the same valid solutions
    /// as the synchronous `solve`. Auto-selects a config that fits free VRAM.
    /// Ignored by default (needs a GPU).
    #[test]
    #[ignore]
    fn pipelined_matches_known() {
        let header = known_header();
        let base = blake::base_state(&header);
        let mut solver = CudaSolver::new(0).expect("init CUDA device 0");

        let r0 = solver.enqueue(&header).expect("enqueue 1"); // priming -> empty
        assert!(r0.is_empty(), "first enqueue should return no results yet");
        let r1 = solver.enqueue(&header).expect("enqueue 2"); // results of enqueue 1
        let r2 = solver.drain().expect("drain"); // results of enqueue 2

        for (label, sols) in [("enqueue", &r1), ("drain", &r2)] {
            assert!(!sols.is_empty(), "pipelined {label} harvested no solutions");
            for s in sols {
                assert!(equihash::is_valid_solution(&base, s), "pipelined {label} invalid solution");
            }
        }
        eprintln!("pipelined: enqueue={} drain={} valid solution(s)", r1.len(), r2.len());
    }
}

/// Launch a kernel via the `extra` / `BUFFER_POINTER` path (a single pre-packed
/// argument buffer), matching how miniZ drives these kernels. The driver
/// marshals the argument bytes during this call, so a shared `&[u8]` is fine.
unsafe fn launch(
    f: CUfunction,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared: u32,
    arg: &[u8],
    stream: CUstream,
) -> Result<()> {
    let mut argsz = arg.len();
    let mut extra: [*mut c_void; 5] = [
        CU_LAUNCH_PARAM_BUFFER_POINTER as *mut c_void,
        arg.as_ptr() as *mut c_void,
        CU_LAUNCH_PARAM_BUFFER_SIZE as *mut c_void,
        &mut argsz as *mut usize as *mut c_void,
        CU_LAUNCH_PARAM_END as *mut c_void,
    ];
    check(
        cuLaunchKernel(
            f,
            grid.0, grid.1, grid.2,
            block.0, block.1, block.2,
            shared,
            stream,
            ptr::null_mut(),
            extra.as_mut_ptr(),
        ),
        "cuLaunchKernel",
    )
}
