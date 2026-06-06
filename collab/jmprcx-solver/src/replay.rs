//! Full-pipeline replay of an Equihash 192,7 solve.
//!
//! The whole pipeline addresses a single ~16 GB arena, so here we:
//!   1. allocate our own arena,
//!   2. for each recorded launch, rebase every device pointer in its arg buffer
//!      (arena_base + (ptr - recorded_arena_base)),
//!   3. launch the same kernel with the same grid/block/shared via the
//!      `extra`/BUFFER_POINTER mechanism,
//!   4. run cleanup -> digit_f -> digit_1..6 -> digit_l -> sort_and_compress.
//!
//! `inject_midstate` (Some 64 bytes) overrides digit_f's midstate so a caller
//! can mint a new job from a header (see blake2b.rs).

use crate::cuda::*;
use std::ffi::{c_void, CString};
use std::ptr;

pub struct Launch {
    pub name: String,
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared: u32,
    pub arg: Vec<u8>,
}

pub struct Recording {
    pub allocs: Vec<(u64, u64)>, // (base, size)
    pub pass: Vec<Launch>,       // first full 10-kernel pass
}

fn triplet(s: &str) -> (u32, u32, u32) {
    let v: Vec<u32> = s.split(',').filter_map(|x| x.parse().ok()).collect();
    (v[0], v[1], v[2])
}

pub fn parse_recording(path: &str) -> Result<Recording, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
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
                if i == 0 { name = tok; }
                else if let Some(v) = tok.strip_prefix("g=") { g = v; }
                else if let Some(v) = tok.strip_prefix("b=") { b = v; }
                else if let Some(v) = tok.strip_prefix("sh=") { sh = v.parse().unwrap_or(0); }
                else if let Some(v) = tok.strip_prefix("arg=") { arg = v; }
            }
            let bytes = (0..arg.len() / 2)
                .map(|i| u8::from_str_radix(&arg[2 * i..2 * i + 2], 16).unwrap_or(0))
                .collect();
            launches.push(Launch { name: name.to_string(), grid: triplet(g), block: triplet(b), shared: sh, arg: bytes });
        }
    }
    // dedup consecutive duplicate allocs, take the first full pass (cleanup .. sort_and_compress)
    let start = launches.iter().position(|l| l.name.contains("7cleanup")).ok_or("no cleanup launch in recording")?;
    let end = launches[start..].iter().position(|l| l.name.contains("sort_and_compress")).ok_or("no sort_and_compress in recording")? + start;
    let pass: Vec<Launch> = launches.drain(start..=end).collect();
    Ok(Recording { allocs, pass })
}

/// number of bytes at the start of a kernel's arg buffer that are by-value
/// (not device pointers) and must NOT be rebased.
fn byval_prefix(name: &str) -> usize {
    if name.contains("7digit_f") { 64 }            // two ulonglong4 (BLAKE2b midstate)
    else if name.contains("sort_and_compress") { 112 } // SHA256_CTX by value
    else { 0 }
}

/// Optional injection to make the GPU solve a header we know:
/// the 64-byte BLAKE2b midstate (= compress(header[0..128])) and the 4 header
/// tail bytes header[136..140] (digit_f's trailing `uint` arg; header[128..135]
/// are constant zero.
pub struct Inject {
    pub midstate: [u8; 64],
    pub tail4: [u8; 4],
}

pub unsafe fn run(module: CUmodule, rec: &Recording, inject: Option<Inject>) -> Result<(Vec<u32>, [u8; 64], Vec<u8>), String> {
    println!("\n== full-pipeline replay ({} kernels) ==", rec.pass.len());

    // identify the arena: the alloc that the most pass pointers fall into
    let in_dev = |v: u64| (0x7000_0000_0000..0x8000_0000_0000).contains(&v);
    let mut votes = vec![0u32; rec.allocs.len()];
    for l in &rec.pass {
        let skip = byval_prefix(&l.name);
        let mut off = skip;
        while off + 8 <= l.arg.len() {
            let v = u64::from_le_bytes(l.arg[off..off + 8].try_into().unwrap());
            if in_dev(v) {
                if let Some(i) = rec.allocs.iter().position(|&(b, s)| v >= b && v < b + s) {
                    votes[i] += 1;
                }
            }
            off += 8;
        }
    }
    let ai = votes.iter().enumerate().max_by_key(|(_, &v)| v).map(|(i, _)| i).ok_or("no arena found")?;
    let (arena_base, arena_size) = rec.allocs[ai];
    println!("arena  : recorded base=0x{arena_base:x} size={} ({:.2} GB), {} ptrs", arena_size, arena_size as f64 / 1e9, votes[ai]);

    // allocate our arena: as much as fits (pipeline only touches the low ~7 GB)
    let mut free = 0usize; let mut total = 0usize;
    cuMemGetInfo_v2(&mut free, &mut total);
    let alloc_size = (arena_size as usize).min(free.saturating_sub(1_500_000_000));
    let mut arena: CUdeviceptr = 0;
    check(cuMemAlloc_v2(&mut arena, alloc_size), "alloc arena")?;
    cuMemsetD8_v2(arena, 0, alloc_size);
    println!("arena  : allocated {:.2} GB at 0x{arena:x} (vram free {:.2} GB)", alloc_size as f64 / 1e9, free as f64 / 1e9);

    let rebase = |v: u64| -> u64 { arena + (v - arena_base) };

    // replay every kernel
    for (idx, l) in rec.pass.iter().enumerate() {
        let cname = CString::new(l.name.clone()).unwrap();
        let mut f: CUfunction = ptr::null_mut();
        check(cuModuleGetFunction(&mut f, module, cname.as_ptr()), &format!("get {}", short(&l.name)))?;

        if l.shared > 0 {
            // opt in to large dynamic shared memory (>48 KB)
            cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, l.shared as i32);
        }

        // rebase pointers in a copy of the arg buffer
        let mut arg = l.arg.clone();
        if l.name.contains("7digit_f") {
            if let Some(inj) = &inject {
                arg[0..64].copy_from_slice(&inj.midstate); // midstate
                arg[96..100].copy_from_slice(&inj.tail4);   // trailing uint = header[136..140]
            }
        }
        let skip = byval_prefix(&l.name);
        let mut off = skip;
        let mut rebased = 0;
        while off + 8 <= arg.len() {
            let v = u64::from_le_bytes(arg[off..off + 8].try_into().unwrap());
            if in_dev(v) && v >= arena_base && v < arena_base + arena_size {
                arg[off..off + 8].copy_from_slice(&rebase(v).to_le_bytes());
                rebased += 1;
            }
            off += 8;
        }

        // launch via the extra / BUFFER_POINTER mechanism
        let mut argsz = arg.len();
        let mut extra: [*mut c_void; 5] = [
            CU_LAUNCH_PARAM_BUFFER_POINTER as *mut c_void,
            arg.as_mut_ptr() as *mut c_void,
            CU_LAUNCH_PARAM_BUFFER_SIZE as *mut c_void,
            &mut argsz as *mut _ as *mut c_void,
            CU_LAUNCH_PARAM_END as *mut c_void,
        ];
        let rc = cuLaunchKernel(
            f, l.grid.0, l.grid.1, l.grid.2, l.block.0, l.block.1, l.block.2,
            l.shared, ptr::null_mut(), ptr::null_mut(), extra.as_mut_ptr(),
        );
        if rc != CUDA_SUCCESS {
            cuMemFree_v2(arena);
            return Err(format!("launch #{idx} {} failed: {}", short(&l.name), err_str(rc)));
        }
        let s = cuCtxSynchronize();
        if s != CUDA_SUCCESS {
            cuMemFree_v2(arena);
            return Err(format!("kernel #{idx} {} sync error: {}", short(&l.name), err_str(s)));
        }
        println!("  [{idx}] {:<18} grid={:<6} block={:<5} shmem={:<6} rebased {rebased} ptr(s)  OK",
                 short(&l.name), l.grid.0, l.block.0, l.shared);
    }

    // dump digit_l's container (+ first candidate) for oracle scanning
    println!("\nreading digit_l container:");
    let mut sol: Vec<u32> = Vec::new();
    let mut container_bytes: Vec<u8> = Vec::new();
    if let Some(dl) = rec.pass.iter().find(|l| l.name.contains("7digit_l")) {
        let p = |off: usize| u64::from_le_bytes(dl.arg[off..off + 8].try_into().unwrap());
        let counter = rebase(p(8));
        let container = rebase(p(16));
        let mut cnt = [0u32; 8];
        cuMemcpyDtoH_v2(cnt.as_mut_ptr() as *mut c_void, counter, 32);
        let dump = 32 * 1024 * 1024usize; // 32 MB window of the container
        container_bytes = vec![0u8; dump];
        cuMemcpyDtoH_v2(container_bytes.as_mut_ptr() as *mut c_void, container, dump);
        sol = container_bytes[..512].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        println!("  counter[0]={}  container[0..4]={:?}  (dumped {} MB)", cnt[0], &sol[..4], dump / 1048576);
    }

    // the midstate actually used by digit_f (injected, or from the recording)
    let mut midstate = [0u8; 64];
    if let Some(df) = rec.pass.iter().find(|l| l.name.contains("7digit_f")) {
        midstate.copy_from_slice(&df.arg[0..64]);
    }
    if let Some(inj) = &inject { midstate = inj.midstate; }

    cuMemFree_v2(arena);
    Ok((sol, midstate, container_bytes))
}

fn short(name: &str) -> String {
    name.split(['I', 'E']).next().unwrap_or(name).trim_start_matches('_').trim_start_matches("Z7").trim_start_matches("Z8").trim_start_matches("Z17").to_string()
}
