//! Approach A: drive alpha-miner's *extracted* CUDA kernels directly.
//!
//! alpha-miner's anti-analysis (dlmopen + /proc/self/maps pointer checks) lives in
//! its **host** code; the cubin itself is plain GPU code. So we `cuModuleLoadData`
//! the extracted `alpha-miner.35.sm_120.cubin` in *our* process and call its
//! kernels — no anti-analysis involved. `headless_mine_kernel` is the fused
//! production kernel (SASS shows IMMA GEMM + `ATOMS.XOR` transcript + the BLAKE3
//! IV + an atomic winner dump), so one launch does GEMM→transcript→jackpot→dump.
//!
//! The remaining RE (done experimentally with the validated CPU oracle in
//! `pearlhash`) is the *input contract*: the 23-param ABI is mapped (see
//! `pearl-dump/INTEGRATION.md §3`), but the grid mapping, the two seed buffers,
//! where the difficulty bound lives, the dump-entry layout, and whether operands
//! are pre-noised must be pinned by feeding known inputs and matching the dump to
//! the CPU jackpot. This module is the loader/launcher foundation for that.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::ptr;

use anyhow::{anyhow, Result};

// Mangled names from pearl-dump/functions.mangled.txt (sm_120 / pearl_blackwell).
const MINE_LB0: &str = "_ZN15pearl_blackwell20headless_mine_kernelILb0ELb0EN4cute5tupleIJiiiEEENS2_IJNS1_1CILi128EEENS4_ILi256EEES5_EEEaNS2_IJiNS4_ILi1EEEEEENS1_14ComposedLayoutINS1_7SwizzleILi2ELi4ELi3EEENS4_ILi0EEENS1_6LayoutINS2_IJNS2_IJNS4_ILi16EEENS4_ILi8EEEEEENS2_IJS5_S8_EEENS2_IJS8_NS4_ILi2EEEEEEEEENS2_IJNS2_IJS5_NS4_ILi2048EEEEEENS2_IJS8_SD_EEENS2_IJSD_NS4_ILi16384EEEEEEEEEEEEENS1_9TiledCopyINS1_9Copy_AtomIJNS1_25SM80_CP_ASYNC_CACHEGLOBALIN7cutlass9uint128_tESY_EEaEEENSE_INS2_IJNS2_IJSG_NS4_ILi32EEEEEESF_EEENS2_IJNS2_IJNS4_ILi512EEES8_EEES11_EEEEENS2_IJS11_S5_EEEEENSV_IJNS1_17SM75_U32x4_LDSM_NEaEEEaS9_NSA_ISC_SD_NSE_INS2_IJNS2_IJSF_SF_EEESI_SK_EEENS2_IJSN_SO_NS2_IJSD_NS4_ILi32768EEEEEEEEEEEEES19_NSV_IJNS1_17SM75_U32x2_LDSM_NEaEEENS1_8TiledMMAINS1_8MMA_AtomIJNS1_26SM80_16x8x32_S32S8S8S32_TNEEEENSE_INS2_IJSJ_NS4_ILi4EEES8_EEENS2_IJS8_SJ_SD_EEEEENS2_IJS11_S11_S11_EEEEEEEvT1_T2_PKT3_T4_T5_T6_T7_PKT8_T9_T10_T11_T12_P16HostSignalHeaderP14HostSignalSyncPKjS2G_T13_iP29PearlHeadlessJackpotDumpEntryPiPjiS2L_";
const MINE_LB1: &str = "_ZN15pearl_blackwell20headless_mine_kernelILb1ELb0EN4cute5tupleIJiiiEEENS2_IJNS1_1CILi128EEENS4_ILi256EEES5_EEEaNS2_IJiNS4_ILi1EEEEEENS1_14ComposedLayoutINS1_7SwizzleILi2ELi4ELi3EEENS4_ILi0EEENS1_6LayoutINS2_IJNS2_IJNS4_ILi16EEENS4_ILi8EEEEEENS2_IJS5_S8_EEENS2_IJS8_NS4_ILi2EEEEEEEEENS2_IJNS2_IJS5_NS4_ILi2048EEEEEENS2_IJS8_SD_EEENS2_IJSD_NS4_ILi16384EEEEEEEEEEEEENS1_9TiledCopyINS1_9Copy_AtomIJNS1_25SM80_CP_ASYNC_CACHEGLOBALIN7cutlass9uint128_tESY_EEaEEENSE_INS2_IJNS2_IJSG_NS4_ILi32EEEEEESF_EEENS2_IJNS2_IJNS4_ILi512EEES8_EEES11_EEEEENS2_IJS11_S5_EEEEENSV_IJNS1_17SM75_U32x4_LDSM_NEaEEEaS9_NSA_ISC_SD_NSE_INS2_IJNS2_IJSF_SF_EEESI_SK_EEENS2_IJSN_SO_NS2_IJSD_NS4_ILi32768EEEEEEEEEEEEES19_NSV_IJNS1_17SM75_U32x2_LDSM_NEaEEENS1_8TiledMMAINS1_8MMA_AtomIJNS1_26SM80_16x8x32_S32S8S8S32_TNEEEENSE_INS2_IJSJ_NS4_ILi4EEES8_EEENS2_IJS8_SJ_SD_EEEEENS2_IJS11_S11_S11_EEEEEEEvT1_T2_PKT3_T4_T5_T6_T7_PKT8_T9_T10_T11_T12_P16HostSignalHeaderP14HostSignalSyncPKjS2G_T13_iP29PearlHeadlessJackpotDumpEntryPiPjiS2L_";

type CUresult = c_int;
type CUdevice = c_int;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUdeviceptr = u64;
const CUDA_SUCCESS: CUresult = 0;
const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;
const CU_LAUNCH_PARAM_END: usize = 0x00;
const CU_LAUNCH_PARAM_BUFFER_POINTER: usize = 0x01;
const CU_LAUNCH_PARAM_BUFFER_SIZE: usize = 0x02;
extern "C" {
    fn cuInit(f: c_uint) -> CUresult;
    fn cuDeviceGet(d: *mut CUdevice, o: c_int) -> CUresult;
    fn cuCtxCreate_v2(c: *mut CUcontext, f: c_uint, d: CUdevice) -> CUresult;
    fn cuCtxDestroy_v2(c: CUcontext) -> CUresult;
    fn cuModuleLoadData(m: *mut CUmodule, image: *const c_void) -> CUresult;
    fn cuModuleUnload(m: CUmodule) -> CUresult;
    fn cuModuleGetFunction(f: *mut CUfunction, m: CUmodule, name: *const c_char) -> CUresult;
    fn cuFuncSetAttribute(f: CUfunction, a: c_int, v: c_int) -> CUresult;
    fn cuMemAlloc_v2(p: *mut CUdeviceptr, n: usize) -> CUresult;
    fn cuMemFree_v2(p: CUdeviceptr) -> CUresult;
    fn cuMemsetD8_v2(p: CUdeviceptr, uc: c_uint, n: usize) -> CUresult;
    fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, n: usize) -> CUresult;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, n: usize) -> CUresult;
    fn cuLaunchKernel(f: CUfunction, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, sh: c_uint, st: *mut c_void, params: *mut *mut c_void, extra: *mut *mut c_void) -> CUresult;
    fn cuCtxSynchronize() -> CUresult;
    fn cuGetErrorName(e: CUresult, s: *mut *const c_char) -> CUresult;
}

fn cu(r: CUresult, what: &str) -> Result<()> {
    if r == CUDA_SUCCESS {
        return Ok(());
    }
    let name = unsafe {
        let mut p: *const c_char = ptr::null();
        if cuGetErrorName(r, &mut p) == CUDA_SUCCESS && !p.is_null() {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        } else {
            format!("CUresult {r}")
        }
    };
    Err(anyhow!("{what}: {name}"))
}

fn func(module: CUmodule, mangled: &str) -> Result<CUfunction> {
    let cname = CString::new(mangled)?;
    let mut f: CUfunction = ptr::null_mut();
    cu(unsafe { cuModuleGetFunction(&mut f, module, cname.as_ptr()) }, "cuModuleGetFunction")?;
    Ok(f)
}

/// alpha-miner's loaded kernels (in our process).
pub struct AlphaKernels {
    ctx: CUcontext,
    module: CUmodule,
    /// `headless_mine_kernel<false,false,...>` (no jackpot dump).
    pub mine_lb0: CUfunction,
    /// `headless_mine_kernel<true,false,...>` (jackpot-dump enabled).
    pub mine_lb1: CUfunction,
}

unsafe impl Send for AlphaKernels {}

impl AlphaKernels {
    /// Load the extracted cubin (raw bytes) and resolve the mine kernels.
    pub fn load(cubin: &[u8], device_index: usize) -> Result<Self> {
        unsafe {
            cu(cuInit(0), "cuInit")?;
            let mut dev: CUdevice = 0;
            cu(cuDeviceGet(&mut dev, device_index as c_int), "cuDeviceGet")?;
            let mut ctx: CUcontext = ptr::null_mut();
            cu(cuCtxCreate_v2(&mut ctx, 0, dev), "cuCtxCreate")?;
            let mut module: CUmodule = ptr::null_mut();
            cu(
                cuModuleLoadData(&mut module, cubin.as_ptr() as *const c_void),
                "cuModuleLoadData (alpha-miner cubin; arch must match GPU)",
            )?;
            let mine_lb0 = func(module, MINE_LB0)?;
            let mine_lb1 = func(module, MINE_LB1)?;
            Ok(Self { ctx, module, mine_lb0, mine_lb1 })
        }
    }
}

/// What the kernel wrote after an exploratory launch.
pub struct ExploreReport {
    pub launched: bool,
    pub dump_count: u32,
    pub dump_head: Vec<u8>,
    pub header_head: Vec<u8>,
    /// (name, first 64 bytes) for every output buffer, to localize what changed.
    pub buffers: Vec<(&'static str, Vec<u8>)>,
}

impl AlphaKernels {
    /// Exploratory launch of `headless_mine_kernel<true,..>` on a tiny shape, to
    /// reverse the input contract by inspecting what it writes. Sets the run-gate,
    /// permissive header (0xFF), known operands; reads back the dump counter +
    /// dump/header heads. The 23-param packed buffer follows INTEGRATION.md §3.
    pub fn explore(&self, m: i32, n: i32, k: i32, _rank: i32, grid: (u32, u32, u32), smem: u32) -> Result<ExploreReport> {
        unsafe {
            let alloc = |bytes: usize, fill: u8| -> Result<CUdeviceptr> {
                let mut p: CUdeviceptr = 0;
                cu(cuMemAlloc_v2(&mut p, bytes.max(16)), "cuMemAlloc")?;
                cu(cuMemsetD8_v2(p, fill as c_uint, bytes.max(16)), "cuMemset")?;
                Ok(p)
            };
            // Over-allocate every buffer (8 MiB) so the CUTLASS kernel's vectorized
            // cp.async / swizzled tile reads can't run off the end — isolates an
            // out-of-bounds read (sizing) from a genuinely wrong pointer.
            let big = 8usize << 20;
            // PEARL_A / PEARL_B set a uniform int8 fill for A / B (default 1), to test
            // whether the GEMM-derived dump fields scale with operand magnitude.
            let af = std::env::var("PEARL_A").ok().and_then(|s| s.parse().ok()).unwrap_or(1u8);
            let bf = std::env::var("PEARL_B").ok().and_then(|s| s.parse().ok()).unwrap_or(1u8);
            let a = alloc(big, af)?; // A' int8
            let b = alloc(big, bf)?; // B'ᵀ int8
            // Optionally fill A/B with a VARIED int8 pattern (small magnitudes so the
            // int32 GEMM stays bounded) so the transcript M is non-trivial — this
            // pins the transcript field's exact offset in the dump entry. PEARL_VARIED=1.
            // PEARL_VARIED: fill A and B with full-range pseudo-random int8 (two
            // different LCGs) so the GEMM accumulators are effectively random and the
            // per-tile XOR can't cancel — this is what real pearl operands (random
            // from the commitment) look like. If M becomes non-zero, operands genuinely
            // drive the transcript.
            if std::env::var("PEARL_VARIED").map(|v| v == "1").unwrap_or(false) {
                let mut sa: u32 = 0x1234_5678;
                let mut sb: u32 = 0x9E37_79B1;
                let mut pa = vec![0u8; big];
                let mut pb = vec![0u8; big];
                for i in 0..big {
                    sa = sa.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    sb = sb.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    pa[i] = (sa >> 24) as u8;
                    pb[i] = (sb >> 24) as u8;
                }
                cu(cuMemcpyHtoD_v2(a, pa.as_ptr() as *const c_void, big), "htod A")?;
                cu(cuMemcpyHtoD_v2(b, pb.as_ptr() as *const c_void, big), "htod B")?;
            }
            let hdr = alloc(4096, 0xFF)?; // permissive
            // HostSignalSync: [0] = first-finder CAS latch, [+4] = abort/stop flag.
            // Prologue gate is `EXIT if sync[+4]==1`, so sync[+4] must be 0 (≠1) to run
            // (we had this inverted — setting it to 1 forced instant exit). Zeroed.
            let sync = alloc(256, 0)?;
            // CONFIRMED via SASS + live dump: ord14 (c[0x3c0], PKj) = the 256-bit
            // difficulty BOUND (hash≤bound wins; word-by-word MSW→LSW compare), and
            // ord15 (c[0x3c8], S2G_) = the BLAKE3 KEY sA (8 words folded into the hash
            // state at 0x2e00). Here both 0xFF: bound=max ⇒ every hash wins (proves the
            // trigger); key=0xFF is just a placeholder for the exploratory launch.
            let bound = alloc(big, 0xFF)?; // ord14
            let key = alloc(big, 0xFF)?; // ord15
            // PEARL_KEY=1: upload a known non-degenerate 32-byte key/commitment seed
            // [0,1,..,31] so we can test whether the seed drives the transcript M
            // (dumped hash would diverge from blake3(zeros, key)).
            if std::env::var("PEARL_KEY").map(|v| v == "1").unwrap_or(false) {
                let kb: [u8; 32] = std::array::from_fn(|i| i as u8);
                cu(cuMemcpyHtoD_v2(key, kb.as_ptr() as *const c_void, 32), "htod key")?;
            }
            let dump = alloc(big, 0)?;
            let dcount = alloc(big, 0)?;
            let out20 = alloc(big, 0)?;
            let out22 = alloc(big, 0)?;

            // Pack the 128-byte argument buffer.
            let mut arg = [0u8; 128];
            let put_u32 = |a: &mut [u8], off: usize, v: u32| a[off..off + 4].copy_from_slice(&v.to_le_bytes());
            let put_ptr = |a: &mut [u8], off: usize, v: u64| a[off..off + 8].copy_from_slice(&v.to_le_bytes());
            put_u32(&mut arg, 0x00, m as u32);
            put_u32(&mut arg, 0x04, n as u32);
            put_u32(&mut arg, 0x08, k as u32);
            put_ptr(&mut arg, 0x10, a);
            // Stride scalars (cute Layout). PEARL_LDA/PEARL_LDB override (default k).
            let envu = |name: &str, d: u32| std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
            put_u32(&mut arg, 0x18, envu("PEARL_LDA", k as u32)); // ldA (ord3)
            put_ptr(&mut arg, 0x20, b);
            put_u32(&mut arg, 0x28, envu("PEARL_LDB", k as u32)); // ldB (ord8)
            put_ptr(&mut arg, 0x30, hdr);
            put_ptr(&mut arg, 0x38, sync);
            put_ptr(&mut arg, 0x40, bound); // ord14 = 256-bit difficulty bound
            put_ptr(&mut arg, 0x48, key); // ord15 = BLAKE3 key sA
            put_u32(&mut arg, 0x54, envu("PEARL_O17", 1)); // ord17 = rank cadence divisor (chunk%ord17==0 folds)
            put_ptr(&mut arg, 0x58, dump);
            put_ptr(&mut arg, 0x60, dcount);
            put_ptr(&mut arg, 0x68, out20);
            put_u32(&mut arg, 0x70, envu("PEARL_O21", 1)); // ord21
            put_ptr(&mut arg, 0x78, out22);

            // <Lb0,Lb0> (mine_lb0) is the variant carrying the bound-compare +
            // CAS first-finder latch + STG jackpot dump (SASS in /tmp/mk.sass).
            // <Lb1,Lb0> (mine_lb1) is the atomic-accumulate variant (283 ATOMG.ADD)
            // that updates global transcripts but never dumps. Pick via PEARL_LB1=1.
            let use_lb1 = std::env::var("PEARL_LB1").map(|v| v == "1").unwrap_or(false);
            let fnc = if use_lb1 { self.mine_lb1 } else { self.mine_lb0 };
            let set = cuFuncSetAttribute(fnc, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as c_int);
            if set != CUDA_SUCCESS {
                return Ok(ExploreReport { launched: false, dump_count: 0, dump_head: vec![set as u8], header_head: vec![0xEE], buffers: vec![] });
            }
            let mut argsz = arg.len();
            let mut extra: [*mut c_void; 5] = [
                CU_LAUNCH_PARAM_BUFFER_POINTER as *mut c_void,
                arg.as_ptr() as *mut c_void,
                CU_LAUNCH_PARAM_BUFFER_SIZE as *mut c_void,
                &mut argsz as *mut usize as *mut c_void,
                CU_LAUNCH_PARAM_END as *mut c_void,
            ];
            let lr = cuLaunchKernel(fnc, grid.0, grid.1, grid.2, 256, 1, 1, smem, ptr::null_mut(), ptr::null_mut(), extra.as_mut_ptr());
            let launched = lr == CUDA_SUCCESS && cuCtxSynchronize() == CUDA_SUCCESS;

            // The found-count is HostSignalSync[+4] (ord19/c[0x3e0] is unused by the
            // <Lb0,Lb0> dump variant). Read sync[1] (the u32 at byte offset 4).
            let mut sync_words = [0u32; 2];
            cuMemcpyDtoH_v2(sync_words.as_mut_ptr() as *mut c_void, sync, 8);
            let count = [sync_words[1]];
            let _ = dcount;
            let mut dump_head = vec![0u8; 256];
            cuMemcpyDtoH_v2(dump_head.as_mut_ptr() as *mut c_void, dump, 256);
            // The jackpot dump lands in ord12 (HostSignalHeader, c[0x3b0]) — the SASS
            // dump-finalize block writes the PearlHeadlessJackpotDumpEntry to R52 =
            // LDC.64 c[0x0][0x3b0]. Read the full ~0x2d8-byte entry.
            let mut header_head = vec![0u8; 768];
            cuMemcpyDtoH_v2(header_head.as_mut_ptr() as *mut c_void, hdr, 768);

            // Snapshot the first 64 bytes of every output buffer to localize writes.
            let snap = |p: CUdeviceptr| -> Vec<u8> {
                let mut v = vec![0u8; 64];
                cuMemcpyDtoH_v2(v.as_mut_ptr() as *mut c_void, p, 64);
                v
            };
            let buffers = vec![
                ("sync(ord13)", snap(sync)),
                ("counter(ord19/Pi)", snap(dcount)),
                ("dump(ord18)", snap(dump)),
                ("ord20(Pj)", snap(out20)),
                ("ord22(S2L)", snap(out22)),
                ("hdr(ord12)", snap(hdr)),
            ];

            for p in [a, b, hdr, sync, bound, key, dump, dcount, out20, out22] {
                cuMemFree_v2(p);
            }
            Ok(ExploreReport { launched, dump_count: count[0], dump_head, header_head, buffers })
        }
    }
}

impl Drop for AlphaKernels {
    fn drop(&mut self) {
        unsafe {
            cuModuleUnload(self.module);
            cuCtxDestroy_v2(self.ctx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Proves we can load alpha-miner's extracted cubin and resolve its mining
    /// kernels in our own process (no anti-analysis). Ignored — needs the cubin +
    /// an sm_120 GPU. Run:
    ///   cargo test --no-default-features --features pearl-cuda -- --ignored --nocapture
    #[test]
    #[ignore]
    fn loads_alpha_cubin() {
        let path = "pearl-dump/cubins/alpha-miner.35.sm_120.cubin";
        let cubin = match std::fs::read(path) {
            Ok(c) => c,
            Err(_) => { eprintln!("no {path}; skipping"); return; }
        };
        let k = AlphaKernels::load(&cubin, 0).expect("load+resolve alpha-miner kernels");
        assert!(!k.mine_lb0.is_null() && !k.mine_lb1.is_null());
        eprintln!("loaded alpha-miner cubin; resolved headless_mine_kernel<false,..> and <true,..>");
    }

    /// Exploratory launch — observe what the fused kernel writes (no crash =
    /// param ABI structurally valid). Ignored — needs the cubin + sm_120 GPU.
    #[test]
    #[ignore]
    fn explore_mine_kernel() {
        let path = "pearl-dump/cubins/alpha-miner.35.sm_120.cubin";
        let cubin = match std::fs::read(path) {
            Ok(c) => c,
            Err(_) => { eprintln!("no {path}; skipping"); return; }
        };
        // ONE launch per process — an ILLEGAL_ADDRESS fault poisons the whole CUDA
        // context state, so sweep smem from bash with PEARL_SMEM env, fresh process
        // each. Baked CTA tile is 128x256; smem is a multistage pipeline
        // (per-stage A 0x4000 + B 0x8000 = 49152).
        let smem: u32 = std::env::var("PEARL_SMEM").ok().and_then(|s| s.parse().ok()).unwrap_or(98304);
        let k = AlphaKernels::load(&cubin, 0).expect("load");
        match k.explore(128, 256, 128, 128, (1, 1, 1), smem) {
            Ok(r) => {
                eprintln!("smem={smem} launched={} dump_count={}", r.launched, r.dump_count);
                if r.launched {
                    eprintln!("  >>> JACKPOT DUMP ENTRY (ord12/HostSignalHeader), non-zero 16B rows:");
                    for (i, chunk) in r.header_head.chunks(16).enumerate() {
                        if chunk.iter().any(|&b| b != 0) {
                            eprintln!("    +0x{:03x}: {:02x?}", i * 16, chunk);
                        }
                    }
                    for (name, bytes) in &r.buffers {
                        let nz = bytes.iter().any(|&b| b != 0);
                        eprintln!("  {name:20} {}: {:02x?}", if nz { "CHANGED" } else { "zero   " }, &bytes[..16]);
                    }
                } else {
                    eprintln!("  (setattr/launch failed: dump_head[0]={:#x})", r.dump_head.first().copied().unwrap_or(0));
                }
            }
            Err(e) => eprintln!("smem={smem} error: {e}"),
        }
    }

    /// Prove we can REPRODUCE the kernel's dumped BLAKE3 jackpot hash with our own
    /// CPU blake3 — i.e. we understand the hash function + the entry's transcript
    /// field. The kernel writes the 256-bit hash at entry+0x29c; we passed key
    /// (ord15) = all-0xFF. So some 64-byte window of the entry is the transcript M,
    /// and `blake3_keyed(M, [0xFF;32]) == hash`. Brute-force the window to locate M
    /// and confirm the match. Ignored — needs the cubin + sm_120 GPU.
    #[test]
    #[ignore]
    fn reproduce_dumped_hash() {
        use crate::pearl::pearlhash::blake3_digest;
        let path = "pearl-dump/cubins/alpha-miner.35.sm_120.cubin";
        let cubin = match std::fs::read(path) {
            Ok(c) => c,
            Err(_) => { eprintln!("no {path}; skipping"); return; }
        };
        let smem: u32 = std::env::var("PEARL_SMEM").ok().and_then(|s| s.parse().ok()).unwrap_or(98304);
        let kdim: i32 = std::env::var("PEARL_K").ok().and_then(|s| s.parse().ok()).unwrap_or(128);
        let k = AlphaKernels::load(&cubin, 0).expect("load");
        eprintln!("launching with K={kdim} (UR7=ceil(K/128)={} chunks; transcript mainloop runs while UR7>1)", (kdim + 127) / 128);
        let r = k.explore(128, 256, kdim, 128, (1, 1, 1), smem).expect("explore");
        assert!(r.launched, "kernel must launch");
        let e = &r.header_head; // the dump entry (in ord12)
        let hash: [u8; 32] = e[0x29c..0x29c + 32].try_into().unwrap();
        eprintln!("dumped hash @+0x29c = {:02x?}", hash);

        // Candidate keys: what we passed. PEARL_KEY=1 uploads [0..31] to ord15.
        let key_used: [u8; 32] = if std::env::var("PEARL_KEY").map(|v| v == "1").unwrap_or(false) {
            std::array::from_fn(|i| i as u8)
        } else {
            [0xFFu8; 32]
        };
        // Decisive: if the dumped hash != blake3(zeros, key_used), then M != 0, i.e.
        // the seed/commitment drives the transcript (operands don't).
        let zeros_hash = blake3_digest(&[0u8; 64], Some(key_used));
        eprintln!("blake3(zeros, key_used) == dumped_hash ? {}  (true => M is still zero)",
                  zeros_hash == hash);
        let key_candidates: [(&str, Option<[u8; 32]>); 2] =
            [("keyed[key_used]", Some(key_used)), ("unkeyed", None)];

        let mut found = false;
        let mut found_nonzero = false;
        for (kname, key) in key_candidates {
            for off in 0..=(e.len().saturating_sub(64)) {
                let window = &e[off..off + 64];
                if blake3_digest(window, key) == hash {
                    let nonzero = window.iter().any(|&b| b != 0);
                    if !found || nonzero {
                        eprintln!("MATCH: transcript M @ entry+0x{off:03x}, key={kname}, nonzero={nonzero}");
                    }
                    if nonzero {
                        eprintln!("  M bytes = {:02x?}", window);
                        found_nonzero = true;
                    }
                    found = true;
                    if found_nonzero { break; }
                }
            }
            if found_nonzero { break; }
        }
        eprintln!("=> hash reproduced: {found}; non-degenerate transcript reached: {found_nonzero}");
        assert!(found, "could not reproduce the dumped hash from any 64-byte window — \
                        transcript layout or key interpretation still unknown");
    }
}
