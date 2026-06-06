//! Round 0 (`digit_f`) standalone driver for the Equihash 192,7 solver.
//!
//!   * launch config: grid=65536, block=256, shmem=0
//!   * argument layout: (ulonglong4 mid0, ulonglong4 mid1, uint4* A, uint4* B,
//!                       uchar* C, uint* counters, uint nonce)
//!   * a real 64-byte BLAKE2b midstate + nonce captured from one job
//!   * buffer sizes derived from the kernel template array dims
//!
//! We replay that exact job's round 0: hash + bucket on the GPU, then read back
//! the per-bucket counters to prove the round executed and distributed entries.

use crate::cuda::*;
use std::ffi::{c_void, CString};
use std::ptr;

// Exact runtime variant (from the fatbin); demangled:
// void digit_f<656825858919744ul,2u,14u,12288u,3392u,1u,5498900316166ul,
//   uint4[106][12288][32], uint4[106][12288][32], unsigned char[53][12288][64]>
//   (ulonglong4, ulonglong4, uint4(*)[106][12288][32], uint4(*)[106][12288][32],
//    unsigned char(*)[53][12288][64], unsigned int*, unsigned int)
const DIGIT_F: &str = "_Z7digit_fILm656825858919744ELj2ELj14ELj12288ELj3392ELj1ELm5498900316166EA106_A12288_A32_5uint4S3_A53_A12288_A64_hEv10ulonglong4S7_PT6_PT7_PT8_Pjj";

// 64-byte BLAKE2b midstate (8x u64 state) captured from a live job, passed as
// two ulonglong4 by value.
const MIDSTATE0: [u8; 32] = [
    0x2d, 0xc6, 0x4e, 0x32, 0xef, 0x89, 0x19, 0x16, 0x30, 0xe1, 0x2d, 0x16, 0x17, 0xb9, 0xeb, 0xee,
    0x33, 0x8a, 0x63, 0xc6, 0xbb, 0xb3, 0x96, 0x33, 0xf1, 0x79, 0x25, 0x9a, 0x7a, 0x26, 0xae, 0x67,
];
const MIDSTATE1: [u8; 32] = [
    0x37, 0x5f, 0x85, 0x39, 0x46, 0x27, 0x08, 0xc0, 0xad, 0x3c, 0x08, 0xe3, 0xda, 0x65, 0xdf, 0xdd,
    0x27, 0x73, 0x1f, 0x13, 0x4d, 0x6f, 0xea, 0x58, 0x96, 0x0d, 0x8b, 0xf3, 0x7c, 0x29, 0x29, 0x9a,
];
const NONCE_ARG: u32 = 1_508_556_231;

// Buffer sizes from the template array dimensions.
const BUF_A: usize = 106 * 12288 * 32 * 16; // uint4[106][12288][32] ≈ 636 MB
const BUF_C: usize = 53 * 12288 * 64; //         uchar[53][12288][64] ≈ 40 MB
const COUNTERS: usize = 64 * 1024 * 1024; //     generous (observed array ≈ 1.5 MB)
const COUNT_READBACK: usize = 12288 * 32; // per-bucket-slot counters to inspect

pub unsafe fn run(module: CUmodule) -> Result<(), String> {
    println!("\n== round 0 (digit_f) standalone replay ==");

    let mut free: usize = 0;
    let mut total: usize = 0;
    cuMemGetInfo_v2(&mut free, &mut total);
    println!(
        "vram   : {} MB free / {} MB total; need ~{} MB",
        free / 1048576, total / 1048576, (2 * BUF_A + BUF_C + COUNTERS) / 1048576
    );

    let cname = CString::new(DIGIT_F).unwrap();
    let mut f: CUfunction = ptr::null_mut();
    check(cuModuleGetFunction(&mut f, module, cname.as_ptr()), "cuModuleGetFunction(digit_f)")?;
    println!("kernel : digit_f<...12288...> resolved, launching grid=65536 block=256");

    // allocate the four device buffers
    let (mut a, mut b, mut c, mut cnt): (CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr) = (0, 0, 0, 0);
    check(cuMemAlloc_v2(&mut a, BUF_A), "alloc bufA")?;
    check(cuMemAlloc_v2(&mut b, BUF_A), "alloc bufB")?;
    check(cuMemAlloc_v2(&mut c, BUF_C), "alloc bufC")?;
    check(cuMemAlloc_v2(&mut cnt, COUNTERS), "alloc counters")?;
    cuMemsetD8_v2(a, 0, BUF_A);
    cuMemsetD8_v2(b, 0, BUF_A);
    cuMemsetD8_v2(c, 0, BUF_C);
    cuMemsetD32_v2(cnt, 0, COUNTERS / 4); // cleanup<64> does this in the real pipeline

    let mut mid0 = MIDSTATE0;
    let mut mid1 = MIDSTATE1;
    let (mut pa, mut pb, mut pc, mut pcnt) = (a, b, c, cnt);
    let mut nonce = NONCE_ARG;
    let mut params: [*mut c_void; 7] = [
        mid0.as_mut_ptr() as *mut c_void,
        mid1.as_mut_ptr() as *mut c_void,
        &mut pa as *mut _ as *mut c_void,
        &mut pb as *mut _ as *mut c_void,
        &mut pc as *mut _ as *mut c_void,
        &mut pcnt as *mut _ as *mut c_void,
        &mut nonce as *mut _ as *mut c_void,
    ];

    let rc = cuLaunchKernel(f, 65536, 1, 1, 256, 1, 1, 0, ptr::null_mut(), params.as_mut_ptr(), ptr::null_mut());
    let result = if rc != CUDA_SUCCESS {
        Err(format!("launch failed: {}", err_str(rc)))
    } else {
        let s = cuCtxSynchronize();
        if s != CUDA_SUCCESS {
            Err(format!("kernel sync error: {}", err_str(s)))
        } else {
            // read back the bucket counters and summarize
            let mut host = vec![0u32; COUNT_READBACK];
            cuMemcpyDtoH_v2(host.as_mut_ptr() as *mut c_void, cnt, COUNT_READBACK * 4);
            let nz = host.iter().filter(|&&x| x != 0).count();
            let sum: u64 = host.iter().map(|&x| x as u64).sum();
            let mx = host.iter().copied().max().unwrap_or(0);
            println!("result : round 0 executed OK");
            println!("         {nz}/{COUNT_READBACK} counter slots non-zero");
            println!("         total bucketed entries = {sum}  (max per slot = {mx})");
            println!("         (2^24 = {} threads each hashed; ~2^25 entries expected)", 1u64 << 24);
            Ok(())
        }
    };

    cuMemFree_v2(a);
    cuMemFree_v2(b);
    cuMemFree_v2(c);
    cuMemFree_v2(cnt);
    result
}
