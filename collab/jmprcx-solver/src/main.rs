//! jmprcx-solver — load the Equihash 192,7 GPU solver fatbin
//! and drive its kernels through the CUDA Driver API.
//!
//! What this does (always, safely):
//!   * cuInit + create a context on the chosen GPU
//!   * load the captured fatbin (auto-selects the cubin matching your GPU arch)
//!   * enumerate every kernel, print attributes, and label the Wagner pipeline
//!
//! What this does with `--launch` (experimental):
//!   * actually launches one real solver kernel (`cleanup<64>(void*, uint)`),
//!     which has a simple, known signature, and synchronizes.
//!

mod cuda;
mod round0;
mod replay;
mod blake2b;
mod verify;
use cuda::*;
use std::ffi::{c_void, CStr, CString};
use std::ptr;

const DEFAULT_FATBIN: &str =
    "/home/access/RustroverProjects/zclminer/collab/jmprcx-solver/src/equihash192_7.fatbin";
const CLEANUP_MANGLED: &str = "_Z7cleanupILj64EEvPvj"; // void cleanup<64u>(void*, unsigned)

/// Best-effort role label from the (mangled) kernel name.
fn role(name: &str) -> &'static str {
    if name.contains("7digit_f") { "round 0  : BLAKE2b hash + initial bucketing" }
    else if name.contains("7digit_1") { "round 1  : Wagner collision" }
    else if name.contains("7digit_2") { "round 2  : Wagner collision" }
    else if name.contains("7digit_3") { "round 3  : Wagner collision" }
    else if name.contains("8digit_4w") { "round 4  : Wagner collision (wide)" }
    else if name.contains("8digit_5w") { "round 5  : Wagner collision (wide)" }
    else if name.contains("8digit_6w") { "round 6  : Wagner collision (wide)" }
    else if name.contains("7digit_l") { "round 7  : final collision + solution recovery" }
    else if name.contains("sort_and_compress") { "post     : sort + compress solutions" }
    else if name.contains("7cleanup") { "util     : buffer cleanup" }
    else { "other" }
}

fn family(name: &str) -> &'static str {
    for k in ["7digit_f","7digit_1","7digit_2","7digit_3","8digit_4w","8digit_5w",
              "8digit_6w","7digit_l","sort_and_compress","7cleanup"] {
        if name.contains(k) { return k; }
    }
    "other"
}

/// A real, pool-accepted 192,7 block header (job 19ae0) captured from the wire.
/// Used by `--solve` as a known-good header so the GPU output can be verified.
const KNOWN_HEADER: &str = "040000002ba84c97ffc202b55a5843d55837d256fdc32410390b8e95502bd8f648040000cb560c7083a13e06273570350805668e83c3e2362e39e131612fead6f4ea9937a19ceba5b597e2217d7e0c53ba24de3d36b92cf97743550c2745c9464f4dc847ba9e1e6a34cf101e80032bb40ae5118877fccacf8d961e648f6a228d0000000000000000ce856809";

/// Scan a container dump for a 128-index group the verifier accepts, using the
/// proven per-index hash as an oracle. The range filter (128 consecutive u32 all
/// in (0, 2^25)) is effectively impossible for random GPU memory, so the
/// expensive XOR check runs only on real solution-shaped windows.
fn scan_container(header: &[u8], bytes: &[u8]) -> Option<Vec<u32>> {
    let u: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    if u.len() < 128 { return None; }
    let mut checked = 0u64;
    for start in 0..=u.len() - 128 {
        let w = &u[start..start + 128];
        if !w.iter().all(|&x| x > 0 && x < (1 << 25)) { continue; }
        let mut d = w.to_vec(); d.sort_unstable(); d.dedup();
        if d.len() != 128 { continue; }
        checked += 1;
        if verify::top_xor_zero_bits(w, |i| blake2b::index_hash(header, i)) >= 168 {
            let (ok, _) = verify::verify(w, |i| blake2b::index_hash(header, i));
            if ok {
                println!("  found at u32 offset {start} (after {checked} solution-shaped windows)");
                return Some(w.to_vec());
            }
        }
    }
    println!("  {checked} solution-shaped windows checked, none verified");
    None
}

/// Decode an Equihash 192,7 stratum solution (varint length + 128 x 25-bit
/// big-endian indices) into 128 indices.
fn decode_solution(hex: &str) -> Vec<u32> {
    let raw = parse_hex(hex);
    // strip the compactsize/varint length prefix (0xfd => 2-byte LE length)
    let body = if raw.first() == Some(&0xfd) { &raw[3..] } else { &raw[1..] };
    let (mut acc, mut bits, mut out) = (0u64, 0u32, Vec::with_capacity(128));
    for &b in body {
        acc = (acc << 8) | b as u64;
        bits += 8;
        while bits >= 25 {
            bits -= 25;
            out.push(((acc >> bits) & 0x1ff_ffff) as u32);
        }
    }
    out.truncate(128);
    out
}

fn verify_share() {
    const SOLUTION: &str = "fd900101420199f2d450c74cdec6d8f3437c5bb217e1e37cb50bacf43cb332bb3ded21346edbc173c868e724d1496f04f3f38bab5705abbb7b168e947bc16b75d4043ce7fb16c10f417c6de5ce8306b1aa5dcd02b7c9e49e6001193aae954c3a733f4f55ce5a9703692af8dea5014a587a1ba2d3a0cf03902cfd212fe5846bc9096bdc615a22e4c1f232d9b945de079c2f29aa3a9c87d0681612d8804a8ccf24c752df1837d4c31bb61b5266328dafeb46af26f96ecc74f2d59ad96c9bff231b4a5e7d87aa33bd916270e703c1d6f090ad8ad02cb86c0550f37585042135ae202f5848bb0b0e695cfe638dfdf89c325833a98125c0f765c6d535e886c915cc01f775b9a35a5972c4ecc40afeb4ff083a7493ab8c238f188b2231218771810cb907f02506020d8f2525a627573126d20955d552328cd1557e34e225b4a2f09c411377055c039163df1c499a4e92a011bf71fc4e58839d23f5822d0a200f65ef194d0a3cf0919b35091b681db6db5293d49e2e12960994436d15300bef5f53799ba98e9e752af7842374f4abc6b5eecd5775de07";
    let header = parse_hex(KNOWN_HEADER);
    let sol = decode_solution(SOLUTION);
    println!("known-answer share (job 19ae0): header {} B, {} indices, {} distinct",
             header.len(), sol.len(), { let mut s = sol.clone(); s.sort_unstable(); s.dedup(); s.len() });
    let zb = verify::top_xor_zero_bits(&sol, |i| blake2b::index_hash(&header, i));
    let (ok, msg) = verify::verify(&sol, |i| blake2b::index_hash(&header, i));
    println!("  full 128-leaf XOR leading zero bits = {zb} / 192");
    println!("  verify: {} — {msg}", if ok { "VALID ✓ (matches pool)" } else { "INVALID" });
}

fn parse_hex(s: &str) -> Vec<u8> {
    let s: String = s.chars().filter(|c| c.is_ascii_hexdigit()).collect();
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap_or(0)).collect()
}

fn main() {
    if let Err(e) = run() {
        eprintln!("\nerror: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let do_launch = args.iter().any(|a| a == "--launch");
    let do_round0 = args.iter().any(|a| a == "--round0");
    let do_replay = args.iter().any(|a| a == "--replay");
    if args.iter().any(|a| a == "--selftest") {
        println!("BLAKE2b-512 known-answer self-test: {}",
                 if blake2b::selftest() { "PASS" } else { "FAIL" });
        return Ok(());
    }
    if args.iter().any(|a| a == "--verify-share") {
        verify_share();
        return Ok(());
    }
    let fatbin_path = args.iter().skip(1)
        .find(|a| a.ends_with(".fatbin"))
        .cloned()
        .unwrap_or_else(|| DEFAULT_FATBIN.to_string());

    // --- read the captured solver fatbin ---
    let image = std::fs::read(&fatbin_path)
        .map_err(|e| format!("reading fatbin {fatbin_path}: {e}"))?;
    if image.len() < 4 || &image[0..4] != [0x50, 0xed, 0x55, 0xba] {
        eprintln!("warning: {fatbin_path} does not start with the fatbin magic 0xBA55ED50");
    }
    println!("== jmprcx Equihash 192,7 solver loader ==");
    println!("fatbin : {fatbin_path} ({} bytes)", image.len());

    unsafe {
        // --- init driver + device + context ---
        check(cuInit(0), "cuInit")?;
        let mut ver = 0;
        cuDriverGetVersion(&mut ver);
        println!("driver : CUDA {}.{}", ver / 1000, (ver % 1000) / 10);

        let mut dev: CUdevice = 0;
        check(cuDeviceGet(&mut dev, 0), "cuDeviceGet")?;
        let mut name = [0i8; 128];
        cuDeviceGetName(name.as_mut_ptr() as *mut _, 128, dev);
        let gpu = CStr::from_ptr(name.as_ptr() as *const _).to_string_lossy().into_owned();
        println!("device : GPU#0 {gpu}");

        let mut ctx: CUcontext = ptr::null_mut();
        check(cuCtxCreate_v2(&mut ctx, 0, dev), "cuCtxCreate")?;

        // --- load the fatbin (driver picks the cubin matching this GPU's arch) ---
        let mut module: CUmodule = ptr::null_mut();
        check(cuModuleLoadData(&mut module, image.as_ptr() as *const c_void),
              "cuModuleLoadData")
            .map_err(|e| format!("{e}\n(the fatbin has sm_80/sm_86/sm_120; the driver needs the cubin matching this GPU)"))?;
        println!("module : loaded OK\n");

        // --- enumerate every kernel in the solver ---
        let mut count: u32 = 0;
        check(cuModuleGetFunctionCount(&mut count, module), "cuModuleGetFunctionCount")?;
        let mut funcs: Vec<CUfunction> = vec![ptr::null_mut(); count as usize];
        check(cuModuleEnumerateFunctions(funcs.as_mut_ptr(), count, module),
              "cuModuleEnumerateFunctions")?;
        println!("solver exposes {count} device kernels:\n");
        println!("  {:<22} {:>5} {:>7} {:>7} {:>6}  role", "name", "regs", "shared", "local", "maxT");
        println!("  {}", "-".repeat(86));

        use std::collections::BTreeMap;
        let mut by_family: BTreeMap<&str, u32> = BTreeMap::new();

        for &f in &funcs {
            let mut np: *const std::ffi::c_char = ptr::null();
            let fname = if cuFuncGetName(&mut np, f) == CUDA_SUCCESS && !np.is_null() {
                CStr::from_ptr(np).to_string_lossy().into_owned()
            } else { "<unknown>".into() };

            let attr = |a: i32| -> i32 { let mut v = 0; cuFuncGetAttribute(&mut v, a, f); v };
            let regs = attr(CU_FUNC_ATTRIBUTE_NUM_REGS);
            let shared = attr(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
            let local = attr(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES);
            let maxt = attr(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);

            *by_family.entry(family(&fname)).or_insert(0) += 1;

            // show a short, stable handle: the C++ template head up to the first '<'/param block
            let short: String = fname.chars().take(22).collect();
            println!("  {:<22} {:>5} {:>7} {:>7} {:>6}  {}", short, regs, shared, local, maxt, role(&fname));
        }

        println!("\nkernel families (Wagner n=192, k=7 pipeline):");
        for (fam, n) in &by_family {
            println!("  {:<20} x{:<3} {}", fam.trim_start_matches(char::is_numeric), n, role(fam));
        }

        // --- optional: actually launch one real solver kernel ---
        if do_launch {
            println!("\n--launch: running cleanup<64>(void*, uint) on the GPU ...");
            let cname = CString::new(CLEANUP_MANGLED).unwrap();
            let mut cf: CUfunction = ptr::null_mut();
            match check(cuModuleGetFunction(&mut cf, module, cname.as_ptr()), "cuModuleGetFunction(cleanup)") {
                Err(e) => println!("  skipped: {e}"),
                Ok(()) => {
                    let bytes: usize = 64 * 1024 * 1024;
                    let mut dptr: CUdeviceptr = 0;
                    check(cuMemAlloc_v2(&mut dptr, bytes), "cuMemAlloc")?;
                    check(cuMemsetD8_v2(dptr, 0xCC, bytes), "cuMemset")?; // poison so we can see it run

                    let n: u32 = 1024;
                    let block: u32 = 64;
                    let grid: u32 = (n + block - 1) / block;
                    let mut p_buf: CUdeviceptr = dptr;
                    let mut p_n: u32 = n;
                    let mut params: [*mut c_void; 2] = [
                        &mut p_buf as *mut _ as *mut c_void,
                        &mut p_n as *mut _ as *mut c_void,
                    ];
                    let rc = cuLaunchKernel(cf, grid, 1, 1, block, 1, 1, 0,
                                            ptr::null_mut(), params.as_mut_ptr(), ptr::null_mut());
                    if rc != CUDA_SUCCESS {
                        println!("  launch returned: {}", err_str(rc));
                    } else {
                        let sync = cuCtxSynchronize();
                        if sync == CUDA_SUCCESS {
                            println!("  launch OK: grid={grid} block={block} — kernel executed and synchronized.");
                        } else {
                            println!("  launched, but sync error: {}", err_str(sync));
                            println!("  (expected-ish: exact element count/indexing for cleanup is unverified)");
                        }
                    }
                    cuMemFree_v2(dptr);
                }
            }
        } else if !do_round0 {
            println!("\n(tip: `--launch` runs cleanup<64>; `--round0` replays digit_f round 0)");
        }

        // --- optional: drive the real round-0 (digit_f) pipeline stage ---
        if do_round0 {
            if let Err(e) = round0::run(module) {
                println!("round 0: {e}");
            }
        }

        // --- replay the pipeline; optionally solve a known header via the verifier oracle ---
        let header_hex = args.iter().position(|a| a == "--header").and_then(|i| args.get(i + 1)).cloned();
        let do_solve = args.iter().any(|a| a == "--solve");
        if do_replay || do_solve || header_hex.is_some() {
            let rec_path = args.iter().skip(1).find(|a| a.ends_with(".log")).cloned()
                .unwrap_or_else(|| "recording.log".to_string());
            match replay::parse_recording(&rec_path) {
                Err(e) => println!("replay: {e}"),
                Ok(rec) => {
                    // header to solve: --solve uses the captured known-good job; --header is user-supplied
                    let header: Option<Vec<u8>> = if do_solve {
                        Some(parse_hex(KNOWN_HEADER))
                    } else {
                        header_hex.as_ref().map(|h| parse_hex(h)).filter(|h| h.len() >= 140)
                    };
                    let inject = header.as_ref().map(|h| {
                        let mid = blake2b::midstate(h);
                        replay::Inject { midstate: mid, tail4: [h[136], h[137], h[138], h[139]] }
                    });
                    if let Some(h) = &header {
                        println!("solving header ({} B); midstate=compress(header[0..128]), tail={:02x?}",
                                 h.len(), &h[136..140]);
                    }
                    match replay::run(module, &rec, inject) {
                        Err(e) => println!("replay: {e}"),
                        Ok((_first, _mid, container)) => match &header {
                            None => println!("pipeline ran (no header to verify against)"),
                            Some(h) => {
                                println!("\nscanning container ({} MB) with the proven verifier as oracle...", container.len() / 1048576);
                                match scan_container(h, &container) {
                                    Some(sol) => {
                                        let (ok, msg) = verify::verify(&sol, |i| blake2b::index_hash(h, i));
                                        println!("\n*** SOLUTION HARVESTED FROM GPU — {} ***", if ok { "VALID ✓" } else { "?" });
                                        println!("  {msg}");
                                        println!("  indices: {:?}{}", &sol[..8], " ...");
                                    }
                                    None => println!("  no verifying 128-index group in the dumped window"),
                                }
                            }
                        },
                    }
                }
            }
        }

        cuModuleUnload(module);
        cuCtxDestroy_v2(ctx);
    }
    Ok(())
}
