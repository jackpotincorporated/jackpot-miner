//! Equihash 192,7 solver (Wagner's algorithm) plus solution
//! encoding/decoding and verification.
//!
//! The solver follows the reference "basic" algorithm from the Zcash
//! specification, generalised over the (n, k) constants in [`crate::params`].
//! It is correctness-oriented; heavy lifting of the initial hashing stage can
//! be offloaded to the GPU (see [`crate::gpu`]).

use std::collections::HashSet;

use blake2b_simd::State;
use rayon::prelude::*;

use crate::blake::{base_state, generate_hash, BatchHasher};
use crate::params::*;

/// A partially-collided list entry: the remaining (trimmed) hash bytes plus the
/// trail of leaf indices that produced it.
#[derive(Clone)]
struct Row {
    hash: Vec<u8>,
    indices: Vec<u32>,
}

/// Expand a packed little-endian-on-the-wire bit array into one byte group per
/// `bit_len`-bit element (each element right-aligned in `(bit_len+7)/8 +
/// byte_pad` bytes). Mirrors Zcash's `ExpandArray`.
pub fn expand_array(input: &[u8], bit_len: usize, byte_pad: usize) -> Vec<u8> {
    assert!(bit_len >= 8 && bit_len + 7 <= 32);
    let out_width = (bit_len + 7) / 8 + byte_pad;
    let elements = input.len() * 8 / bit_len;
    let mut out = vec![0u8; elements * out_width];

    let bit_len_mask: u32 = (1u32 << bit_len) - 1;
    let mut acc_bits = 0usize;
    let mut acc_value: u64 = 0;
    let mut j = 0usize;

    for &b in input {
        acc_value = (acc_value << 8) | b as u64;
        acc_bits += 8;
        if acc_bits >= bit_len {
            acc_bits -= bit_len;
            for x in byte_pad..out_width {
                let shift = acc_bits + 8 * (out_width - x - 1);
                let mask = (bit_len_mask >> (8 * (out_width - x - 1))) & 0xFF;
                out[j + x] = ((acc_value >> shift) as u32 & mask) as u8;
            }
            j += out_width;
        }
    }
    out
}

/// Inverse of [`expand_array`]; mirrors Zcash's `CompressArray`.
pub fn compress_array(input: &[u8], out_len: usize, bit_len: usize, byte_pad: usize) -> Vec<u8> {
    let in_width = (bit_len + 7) / 8 + byte_pad;
    let bit_len_mask: u32 = (1u32 << bit_len) - 1;
    let mut out = vec![0u8; out_len];

    let mut acc_bits = 0usize;
    let mut acc_value: u64 = 0;
    let mut j = 0usize;

    for o in out.iter_mut() {
        if acc_bits < 8 {
            acc_value <<= bit_len;
            for x in byte_pad..in_width {
                let g = (bit_len_mask >> (8 * (in_width - x - 1))) & 0xFF;
                acc_value |= ((input[j + x] as u32 & g) as u64) << (8 * (in_width - x - 1));
            }
            j += in_width;
            acc_bits += bit_len;
        }
        acc_bits -= 8;
        *o = ((acc_value >> acc_bits) & 0xFF) as u8;
    }
    out
}

/// Encode a solution's indices into the compact 400-byte wire form.
pub fn indices_to_solution(indices: &[u32]) -> Vec<u8> {
    let cbl = COLLISION_BIT_LENGTH;
    let byte_pad = 4 - ((cbl + 1) + 7) / 8;
    let mut bytes = Vec::with_capacity(indices.len() * 4);
    for &idx in indices {
        bytes.extend_from_slice(&idx.to_be_bytes());
    }
    let min_len = (cbl + 1) * indices.len() / 8;
    compress_array(&bytes, min_len, cbl + 1, byte_pad)
}

/// Decode the compact wire form back into solution indices.
pub fn solution_to_indices(minimal: &[u8]) -> Vec<u32> {
    let cbl = COLLISION_BIT_LENGTH;
    let byte_pad = 4 - ((cbl + 1) + 7) / 8;
    let expanded = expand_array(minimal, cbl + 1, byte_pad);
    expanded
        .chunks_exact(4)
        .map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Build the expanded, byte-padded solver hash for a single leaf index.
fn leaf_row(base: &State, idx: u32) -> Row {
    let g = idx / INDICES_PER_HASH_OUTPUT as u32;
    let i = (idx % INDICES_PER_HASH_OUTPUT as u32) as usize;
    let h = generate_hash(base, g);
    let start = i * HASH_BYTES;
    Row {
        hash: expand_array(&h[start..start + HASH_BYTES], COLLISION_BIT_LENGTH, 0),
        indices: vec![idx],
    }
}

// ---------------------------------------------------------------------------
// Bucketed solver (Wagner's algorithm with back-references) — the same
// structure the GPU kernels use, on the CPU. Each round groups entries by their
// leading 24-bit collision block (which, since blocks are byte-aligned, groups
// exact collisions), emits one XOR child per colliding pair, and records the
// pair as a back-reference. After the rounds the references are walked to
// recover each candidate's 128 leaf indices, then verified + de-duplicated.
// Distinctness isn't enforced during the rounds — the few degenerate
// (index-repeating) candidates are discarded cheaply at the end, as in the GPU
// pipeline.
//
// Grouping uses xenoncat's two-level split rather than a single 2^24-wide
// counting sort: the high `TOP_BITS` of the leading block select one of
// `TOP_BUCKETS` partitions, and the remaining `LOW_BITS` are resolved by a small
// per-bucket histogram that stays resident in L1/L2 (vs. a 2^24-entry / 64 MiB
// histogram that has to be zeroed, scatter-filled, and cloned every round). Two
// entries can collide only when their *whole* leading block matches, so they
// always land in the same partition — making the partitions independent
// collision domains that are processed in parallel across the rayon pool the
// single CPU worker otherwise leaves idle.
//
// The data layout follows xenoncat's packed slots and targets AVX2 throughput:
//   * Entries live in fixed 32-byte `SLOT`s (one `__m256i`); producing a child
//     is a single load/xor/permute/store (`xor_child_avx2`).
//   * A dense parallel `keys[]` array mirrors each entry's leading block, so the
//     `partition_top`/`count_pairs`/`low_group` histogram sweeps stream over
//     4 bytes/entry instead of striding the slots.
//   * `collide` runs two passes — count pairs, then emit children directly into
//     one pre-sized arena at per-partition offsets — so there is no per-bucket
//     allocation and no final concatenation copy; the `keys`/`slots` buffers
//     ping-pong between rounds.
//   * Round 0 repacks BLAKE2b digests into 24-bit blocks with `pshufb`.
// A further bandwidth step (not yet taken) is xenoncat's full radix scatter of
// the payload into bucket-contiguous storage, turning the inner-loop slot
// gathers into sequential reads.
// ---------------------------------------------------------------------------

/// 24-bit collision blocks carried by a round-0 entry (= 8 for 192,7).
const NBLK0: usize = HASH_LENGTH / COLLISION_BYTE_LENGTH;

/// High bits of the leading block used to pick a partition (12 for 192,7).
const TOP_BITS: usize = COLLISION_BIT_LENGTH / 2;
/// Low bits resolved by the per-bucket histogram (12 for 192,7).
const LOW_BITS: usize = COLLISION_BIT_LENGTH - TOP_BITS;
/// Number of top-level partitions (4096).
const TOP_BUCKETS: usize = 1 << TOP_BITS;
/// Histogram width within a partition (4096 entries → 16 KiB).
const LOW_BUCKETS: usize = 1 << LOW_BITS;
/// Mask isolating the low resolved bits.
const LOW_MASK: u32 = (LOW_BUCKETS - 1) as u32;

/// Number of u32 words in a padded entry slot (32 bytes = one AVX2 register).
/// Every round stores its residual hash words in a fixed 8-word slot so the XOR
/// that produces a child is a single 256-bit load/xor/permute/store and every
/// slot access is naturally aligned — xenoncat's packed-slot trick (and the same
/// `uint4`-aligned-slot idea the CUDA backend already uses), on the CPU. The
/// leading collision word lives in lane 0 and is mirrored into a dense parallel
/// `keys[]` array so the histogram passes stream over 4 bytes/entry instead of
/// striding the 32-byte slots.
const SLOT: usize = 8;

/// Scalar child producer: `out[0..8] = (a XOR b)` rotated left one lane; returns
/// the child's new leading word (lane 1 of the XOR). Lane 0 of the XOR is the
/// just-collided block (always zero) and is rotated out.
#[inline]
unsafe fn xor_child_scalar(out: *mut u32, a: *const u32, b: *const u32) -> u32 {
    let mut x = [0u32; SLOT];
    for t in 0..SLOT {
        x[t] = *a.add(t) ^ *b.add(t);
    }
    for t in 0..SLOT {
        *out.add(t) = x[(t + 1) % SLOT];
    }
    x[1]
}

/// AVX2 child producer: one `vpxor` + one `vpermd` (rotate the 8 lanes left by
/// one) + one store. Replaces the per-word scalar XOR loop.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn xor_child_avx2(out: *mut u32, a: *const u32, b: *const u32) -> u32 {
    use core::arch::x86_64::*;
    let x = _mm256_xor_si256(
        _mm256_loadu_si256(a as *const __m256i),
        _mm256_loadu_si256(b as *const __m256i),
    );
    // rotate left by one 32-bit lane: out[i] = x[(i + 1) % 8]
    let p = _mm256_permutevar8x32_epi32(x, _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0));
    _mm256_storeu_si256(out as *mut __m256i, p);
    _mm_cvtsi128_si32(_mm256_castsi256_si128(p)) as u32
}

/// Scalar round-0 repack: 24 digest bytes -> eight big-endian 24-bit blocks.
#[inline]
fn repack_index_scalar(src: &[u8], dst: &mut [u32]) {
    for b in 0..NBLK0 {
        let p = 3 * b;
        dst[b] = ((src[p] as u32) << 16) | ((src[p + 1] as u32) << 8) | src[p + 2] as u32;
    }
}

/// AVX2 round-0 repack via two `pshufb`s: gathers the eight 3-byte big-endian
/// groups straight into eight little-endian u32 lanes (the top byte of each is
/// zeroed by the 0x80 mask bytes). `src` must hold this index's 24 digest bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn repack_index_avx2(src: &[u8], dst: *mut u32) {
    use core::arch::x86_64::*;
    // lo half: u32 j <- src[3j..3j+3] big-endian -> LE bytes [3j+2, 3j+1, 3j, 0].
    let mask_lo = _mm_setr_epi8(2, 1, 0, -128, 5, 4, 3, -128, 8, 7, 6, -128, 11, 10, 9, -128);
    // hi half loads from src+8, so its byte indices are the above minus 8.
    let mask_hi = _mm_setr_epi8(6, 5, 4, -128, 9, 8, 7, -128, 12, 11, 10, -128, 15, 14, 13, -128);
    let a = _mm_loadu_si128(src.as_ptr() as *const __m128i); // src[0..16]
    let b = _mm_loadu_si128(src.as_ptr().add(8) as *const __m128i); // src[8..24]
    _mm_storeu_si128(dst as *mut __m128i, _mm_shuffle_epi8(a, mask_lo));
    _mm_storeu_si128((dst as *mut __m128i).add(1), _mm_shuffle_epi8(b, mask_hi));
}

/// Repack one index's digest into its eight-word slot. Dispatches to AVX2.
#[inline]
fn repack_index(src: &[u8], dst: &mut [u32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { repack_index_avx2(src, dst.as_mut_ptr()) };
            return;
        }
    }
    repack_index_scalar(src, dst);
}

/// Partition the `n` entries into `TOP_BUCKETS` runs by the high `TOP_BITS` of
/// their (dense) leading block. Returns `(starts, order)`, where partition `v`
/// owns the input indices `order[starts[v]..starts[v + 1]]`. The histogram
/// passes stream over `keys[]` (4 bytes/entry) instead of striding the slots.
fn partition_top(keys: &[u32], n: usize) -> (Vec<u32>, Vec<u32>) {
    let mut starts = vec![0u32; TOP_BUCKETS + 1];
    for k in 0..n {
        starts[(keys[k] >> LOW_BITS) as usize + 1] += 1;
    }
    for i in 0..TOP_BUCKETS {
        starts[i + 1] += starts[i];
    }
    let mut order = vec![0u32; n];
    let mut cur = starts.clone(); // small: TOP_BUCKETS + 1 entries
    for k in 0..n {
        let b = (keys[k] >> LOW_BITS) as usize;
        order[cur[b] as usize] = k as u32;
        cur[b] += 1;
    }
    (starts, order)
}

/// Count the colliding pairs a partition will emit, from the low-bit histogram
/// alone (no reordering). `clamp` caps each exact-collision group, matching the
/// emit pass so the output offsets line up. `hist` is reusable `LOW_BUCKETS`
/// scratch.
fn count_pairs(keys: &[u32], run: &[u32], hist: &mut [u32], clamp: usize) -> usize {
    for h in hist.iter_mut() {
        *h = 0;
    }
    for &k in run {
        hist[(keys[k as usize] & LOW_MASK) as usize] += 1;
    }
    let mut pairs = 0usize;
    for i in 0..LOW_BUCKETS {
        let c = (hist[i] as usize).min(clamp);
        pairs += c * (c - 1) / 2;
    }
    pairs
}

/// Within one partition, group `run`'s entries by the low bits of their leading
/// block, writing the grouped indices into `sorted`. `hist` is reusable
/// `LOW_BUCKETS + 1` scratch. After this call `sorted` lists the run's indices
/// with equal low keys contiguous, so callers recover each exact-collision
/// group by walking adjacent equal keys.
fn low_group(keys: &[u32], run: &[u32], hist: &mut [u32], sorted: &mut Vec<u32>) {
    let m = run.len();
    sorted.clear();
    if m == 0 {
        return;
    }
    for h in hist.iter_mut() {
        *h = 0;
    }
    for &k in run {
        hist[(keys[k as usize] & LOW_MASK) as usize + 1] += 1;
    }
    for i in 0..LOW_BUCKETS {
        hist[i + 1] += hist[i];
    }
    sorted.resize(m, 0);
    // hist[low] now holds the run-start offset; reuse it as the live cursor.
    for &k in run {
        let low = (keys[k as usize] & LOW_MASK) as usize;
        sorted[hist[low] as usize] = k;
        hist[low] += 1;
    }
}

/// Walk a partition's grouped entries (`sorted`) and emit one child per
/// colliding pair into the partition's output sub-slices, returning the count
/// written. `clamp` bounds each exact-collision group (`usize::MAX` = exact).
/// Monomorphised over the XOR producer so the AVX2 intrinsics inline cleanly
/// inside a `target_feature` wrapper while sharing one source of truth.
macro_rules! emit_bucket_body {
    ($keys:expr, $slots:expr, $sorted:expr, $kout:expr, $sout:expr, $pout:expr, $clamp:expr, $xor:path) => {{
        let s = $sorted;
        let m = s.len();
        let mut w = 0usize;
        let mut i = 0;
        while i < m {
            let key = $keys[s[i] as usize] & LOW_MASK;
            let mut j = i + 1;
            while j < m && ($keys[s[j] as usize] & LOW_MASK) == key {
                j += 1;
            }
            let hi = j.min(i.saturating_add($clamp));
            for a in i..hi {
                let l = s[a] as usize;
                for b in (a + 1)..hi {
                    let mr = s[b] as usize;
                    let nk = $xor(
                        $sout.as_mut_ptr().add(w * SLOT),
                        $slots.as_ptr().add(l * SLOT),
                        $slots.as_ptr().add(mr * SLOT),
                    );
                    $kout[w] = nk;
                    $pout[w] = ((l as u64) << 32) | mr as u64;
                    w += 1;
                }
            }
            i = j;
        }
        w
    }};
}

unsafe fn emit_bucket_scalar(
    keys: &[u32],
    slots: &[u32],
    sorted: &[u32],
    kout: &mut [u32],
    sout: &mut [u32],
    pout: &mut [u64],
    clamp: usize,
) -> usize {
    emit_bucket_body!(keys, slots, sorted, kout, sout, pout, clamp, xor_child_scalar)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn emit_bucket_avx2(
    keys: &[u32],
    slots: &[u32],
    sorted: &[u32],
    kout: &mut [u32],
    sout: &mut [u32],
    pout: &mut [u64],
    clamp: usize,
) -> usize {
    emit_bucket_body!(keys, slots, sorted, kout, sout, pout, clamp, xor_child_avx2)
}

/// Emit a partition's children, dispatching to the AVX2 producer when available.
unsafe fn emit_bucket(
    keys: &[u32],
    slots: &[u32],
    sorted: &[u32],
    kout: &mut [u32],
    sout: &mut [u32],
    pout: &mut [u64],
    clamp: usize,
) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return emit_bucket_avx2(keys, slots, sorted, kout, sout, pout, clamp);
        }
    }
    emit_bucket_scalar(keys, slots, sorted, kout, sout, pout, clamp)
}

/// Group `n` entries by their leading block, then emit one child per colliding
/// pair: the XOR of the residual blocks (rotated into a fresh 8-word slot) plus
/// a packed `(l << 32) | mr` back-reference. Two passes — count, then emit
/// directly into one pre-sized arena — so there is no per-partition allocation
/// or final concatenation copy. Returns `(keys_out, slots_out, parents)`.
fn collide(keys: &[u32], slots: &[u32], n: usize, clamp: usize) -> (Vec<u32>, Vec<u32>, Vec<u64>) {
    let (starts, order) = partition_top(keys, n);

    // Pass 1: per-partition child counts (histogram-derived, no reordering).
    let counts: Vec<usize> = (0..TOP_BUCKETS)
        .into_par_iter()
        .map_init(
            || vec![0u32; LOW_BUCKETS],
            |hist, v| {
                let run = &order[starts[v] as usize..starts[v + 1] as usize];
                count_pairs(keys, run, hist, clamp)
            },
        )
        .collect();

    let mut out_starts = vec![0usize; TOP_BUCKETS + 1];
    for v in 0..TOP_BUCKETS {
        out_starts[v + 1] = out_starts[v] + counts[v];
    }
    let total = out_starts[TOP_BUCKETS];
    let mut keys_out = vec![0u32; total];
    let mut slots_out = vec![0u32; total * SLOT];
    let mut parents = vec![0u64; total];

    // Carve the output arena into disjoint per-partition sub-slices so workers
    // write their children directly into place (no concatenation).
    let mut kparts: Vec<&mut [u32]> = Vec::with_capacity(TOP_BUCKETS);
    let mut sparts: Vec<&mut [u32]> = Vec::with_capacity(TOP_BUCKETS);
    let mut pparts: Vec<&mut [u64]> = Vec::with_capacity(TOP_BUCKETS);
    {
        let (mut kr, mut sr, mut pr) = (&mut keys_out[..], &mut slots_out[..], &mut parents[..]);
        for &c in &counts {
            let (kh, kt) = kr.split_at_mut(c);
            let (sh, st) = sr.split_at_mut(c * SLOT);
            let (ph, pt) = pr.split_at_mut(c);
            kparts.push(kh);
            sparts.push(sh);
            pparts.push(ph);
            kr = kt;
            sr = st;
            pr = pt;
        }
    }

    // Pass 2: group each partition and emit its colliding pairs in place.
    kparts
        .into_par_iter()
        .zip(sparts)
        .zip(pparts)
        .enumerate()
        .for_each(|(v, ((kout, sout), pout))| {
            let run = &order[starts[v] as usize..starts[v + 1] as usize];
            let mut hist = vec![0u32; LOW_BUCKETS + 1];
            let mut sorted = Vec::new();
            low_group(keys, run, &mut hist, &mut sorted);
            let w = unsafe { emit_bucket(keys, slots, &sorted, kout, sout, pout, clamp) };
            debug_assert_eq!(w, kout.len());
        });

    (keys_out, slots_out, parents)
}

/// Final round (slots hold `[w0, w1, …]`): among entries sharing leading block
/// `w0`, a pair whose `w1` also matches XORs the last two blocks to zero — a
/// candidate. Returns the `(l, mr)` parents of each candidate.
fn collide_final(keys: &[u32], slots: &[u32], n: usize, clamp: usize) -> Vec<(u32, u32)> {
    let (starts, order) = partition_top(keys, n);

    (0..TOP_BUCKETS)
        .into_par_iter()
        .map_init(
            || (vec![0u32; LOW_BUCKETS + 1], Vec::<u32>::new()),
            |(hist, sorted), v| {
                let run = &order[starts[v] as usize..starts[v + 1] as usize];
                low_group(keys, run, hist, sorted);
                let m = sorted.len();
                let mut local = Vec::new();
                let mut i = 0;
                while i < m {
                    let key = keys[sorted[i] as usize] & LOW_MASK;
                    let mut j = i + 1;
                    while j < m && (keys[sorted[j] as usize] & LOW_MASK) == key {
                        j += 1;
                    }
                    let hi = j.min(i.saturating_add(clamp));
                    for a in i..hi {
                        let l = sorted[a] as usize;
                        for b in (a + 1)..hi {
                            let mr = sorted[b] as usize;
                            if slots[l * SLOT + 1] == slots[mr * SLOT + 1] {
                                local.push((l as u32, mr as u32));
                            }
                        }
                    }
                    i = j;
                }
                local
            },
        )
        .reduce(Vec::new, |mut acc, mut v| {
            acc.append(&mut v);
            acc
        })
}

/// CPU solver, exact (no clamp): hash the first round with BLAKE2b, run the
/// bucketed collision rounds, recover each candidate's leaves, then verify +
/// de-duplicate.
///
/// WARNING: with no clamp this runs the naive (basic-spec) Wagner algorithm,
/// which does not track index distinctness during the rounds. For dense
/// Equihash 192,7 headers the degenerate (duplicate-index) collisions cascade in
/// the last rounds and the candidate count explodes — easily tens of GB of
/// recovery buffers. Mining and `--selftest` use [`solve_with`] with a clamp
/// instead (the same bound the GPU enforces via fixed-size bucket slots). Use
/// this only when you specifically want unclamped semantics and have bounded the
/// header.
#[allow(dead_code)] // exact public API; callers use solve_with (clamp 0 == exact)
pub fn solve(header: &[u8]) -> Vec<Vec<u32>> {
    solve_with(header, None)
}

/// As [`solve`], but optionally clamps each exact-collision group to `clamp`
/// entries. Clamping bounds the naive algorithm's degenerate-collision blow-up
/// (xenoncat's / the GPU's bounded buckets), trading a small, probabilistic loss
/// of solutions for a working, fast solve. `None` is exact (see the warning on
/// [`solve`]); a clamp around 16–64 keeps the ~2 real solutions per nonce.
pub fn solve_with(header: &[u8], clamp: Option<usize>) -> Vec<Vec<u32>> {
    let clamp = clamp.unwrap_or(usize::MAX);

    // ---- round 0: hash every index into NBLK0 big-endian 24-bit blocks, stored
    // in padded 8-word slots with the leading block mirrored into `keys`. Entry
    // k corresponds to leaf index k (the gen order), so no leaf table is needed.
    // The midstate hasher emits four `g` digests per call (AVX2), each yielding
    // INDICES_PER_HASH_OUTPUT entries; one chunk = four `g` values = 8 entries.
    let n0 = INIT_SIZE;
    let hasher = BatchHasher::new(header);
    let mut keys: Vec<u32> = vec![0u32; n0];
    let mut slots: Vec<u32> = vec![0u32; n0 * SLOT];
    let kgroup = 4 * INDICES_PER_HASH_OUTPUT; // eight entries
    let sgroup = kgroup * SLOT;
    debug_assert_eq!(n0 % kgroup, 0, "round-0 buffer must split into whole 4-g groups");
    slots
        .par_chunks_mut(sgroup)
        .zip(keys.par_chunks_mut(kgroup))
        .enumerate()
        .for_each(|(c, (schunk, kchunk))| {
            let g0 = (c * 4) as u32;
            let mut hs = [[0u8; HASH_OUTPUT]; 4];
            hasher.hash4(g0, &mut hs);
            for j in 0..4 {
                for i in 0..INDICES_PER_HASH_OUTPUT {
                    let e = j * INDICES_PER_HASH_OUTPUT + i;
                    let src = &hs[j][i * HASH_BYTES..i * HASH_BYTES + HASH_BYTES];
                    let dst = &mut schunk[e * SLOT..e * SLOT + SLOT];
                    repack_index(src, dst);
                    kchunk[e] = dst[0];
                }
            }
        });

    // ---- collision rounds: 6 intermediate rounds (tables 1..6), then final.
    // parents[t] (t = 0..K-2) maps a round-(t+1) entry to its two round-t parents.
    // `keys`/`slots` ping-pong between rounds (the previous buffers are freed as
    // the new ones replace them).
    let mut parents: Vec<Vec<u64>> = Vec::with_capacity(K - 1);
    let mut n = n0;
    for _ in 0..(K - 1) {
        let (ok, os, op) = collide(&keys, &slots, n, clamp);
        n = op.len();
        parents.push(op);
        keys = ok;
        slots = os;
        if n == 0 {
            return Vec::new();
        }
    }

    let candidates = collide_final(&keys, &slots, n, clamp);
    if candidates.is_empty() {
        return Vec::new();
    }

    // ---- recover 128 leaf indices per candidate by walking back-references.
    let mut recovered: Vec<u32> = Vec::with_capacity(candidates.len() * SOLUTION_INDICES);
    let mut refs: Vec<u32> = Vec::with_capacity(SOLUTION_INDICES);
    let mut next: Vec<u32> = Vec::with_capacity(SOLUTION_INDICES);
    for (a, b) in candidates {
        refs.clear();
        refs.push(a);
        refs.push(b);
        for par in parents.iter().rev() {
            next.clear();
            for &x in &refs {
                let p = par[x as usize];
                next.push((p >> 32) as u32);
                next.push((p & 0xffff_ffff) as u32);
            }
            std::mem::swap(&mut refs, &mut next);
        }
        // refs are now round-0 indices == leaf indices.
        recovered.extend_from_slice(&refs);
    }

    filter_candidates(&base_state(header), &recovered)
}

/// Re-order recovered leaf indices into the canonical solution ordering: at
/// every tree node the subtree with the smaller leading index comes first.
///
/// Both the CPU and GPU solvers recover the 128 leaves in tree in-order but do
/// not order sibling subtrees; this restores the ordering the spec (and
/// verifier) require.
pub fn canonicalize(leaves: &[u32]) -> Vec<u32> {
    if leaves.len() <= 1 {
        return leaves.to_vec();
    }
    let half = leaves.len() / 2;
    let left = canonicalize(&leaves[..half]);
    let right = canonicalize(&leaves[half..]);
    let mut out = Vec::with_capacity(leaves.len());
    if left[0] < right[0] {
        out.extend_from_slice(&left);
        out.extend_from_slice(&right);
    } else {
        out.extend_from_slice(&right);
        out.extend_from_slice(&left);
    }
    out
}

/// Filter raw GPU-recovered candidates (`n * SOLUTION_INDICES` leaf indices)
/// into valid, canonical, de-duplicated solutions. A cheap distinctness check
/// rejects the many degenerate (duplicate-index) candidates before the costlier
/// full verification. Shared by the CPU and GPU solvers.
pub fn filter_candidates(base: &State, recovered: &[u32]) -> Vec<Vec<u32>> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for chunk in recovered.chunks_exact(SOLUTION_INDICES) {
        let unique: HashSet<u32> = chunk.iter().copied().collect();
        if unique.len() != SOLUTION_INDICES {
            continue; // degenerate: repeated leaf indices
        }
        let canon = canonicalize(chunk);
        if !is_valid_solution(base, &canon) {
            continue;
        }
        let mut key = canon.clone();
        key.sort_unstable();
        if seen.insert(key) {
            result.push(canon);
        }
    }
    result
}

/// Verify a candidate solution against the spec: correct length, globally
/// distinct indices, the proper collisions at every tree level, the canonical
/// index ordering, and an all-zero final XOR.
pub fn is_valid_solution(base: &State, indices: &[u32]) -> bool {
    if indices.len() != SOLUTION_INDICES {
        return false;
    }
    // All 2^k indices must be globally distinct.
    let unique: HashSet<u32> = indices.iter().copied().collect();
    if unique.len() != indices.len() {
        return false;
    }

    let mut rows: Vec<Row> = indices.iter().map(|&idx| leaf_row(base, idx)).collect();
    let cb = COLLISION_BYTE_LENGTH;

    while rows.len() > 1 {
        let mut next = Vec::with_capacity(rows.len() / 2);
        for pair in rows.chunks(2) {
            let (a, b) = (&pair[0], &pair[1]);
            // Leading collision block must match.
            if a.hash[..cb] != b.hash[..cb] {
                return false;
            }
            // Canonical ordering: smaller leading index on the left.
            if a.indices[0] >= b.indices[0] {
                return false;
            }
            let len = a.hash.len();
            let mut nh = Vec::with_capacity(len - cb);
            for x in cb..len {
                nh.push(a.hash[x] ^ b.hash[x]);
            }
            let mut ni = a.indices.clone();
            ni.extend_from_slice(&b.indices);
            next.push(Row { hash: nh, indices: ni });
        }
        rows = next;
    }

    rows[0].hash.iter().all(|&x| x == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_compress_roundtrip() {
        // 128 indices of 25 bits each round-trip through compress/expand.
        let indices: Vec<u32> = (0..SOLUTION_INDICES as u32).map(|i| i * 7919 % INIT_SIZE as u32).collect();
        let packed = indices_to_solution(&indices);
        assert_eq!(packed.len(), SOLUTION_BYTES);
        let back = solution_to_indices(&packed);
        assert_eq!(back, indices);
    }

    #[test]
    fn collision_block_sizes() {
        assert_eq!(COLLISION_BIT_LENGTH, 24);
        assert_eq!(HASH_LENGTH, 24);
        assert_eq!(SOLUTION_INDICES, 128);
        assert_eq!(SOLUTION_BYTES, 400);
        assert_eq!(BLAKE_CALLS, 1 << 24);
    }

    // AVX2 round-0 repack must match the scalar reference for arbitrary digest
    // bytes (24 valid bytes per index).
    #[test]
    fn repack_matches_scalar() {
        let src: Vec<u8> = (0..24u32).map(|i| (i.wrapping_mul(37).wrapping_add(11)) as u8).collect();
        let mut a = [0u32; NBLK0];
        let mut b = [0u32; NBLK0];
        repack_index_scalar(&src, &mut a);
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            unsafe { repack_index_avx2(&src, b.as_mut_ptr()) };
            assert_eq!(a, b, "avx2 repack != scalar");
        }
        // Block 0 is the big-endian 24-bit value of the first three bytes.
        assert_eq!(a[0], (src[0] as u32) << 16 | (src[1] as u32) << 8 | src[2] as u32);
    }

    // AVX2 XOR-child (xor + rotate-left-one-lane) must match the scalar version.
    #[test]
    fn xor_child_matches_scalar() {
        let pa: [u32; SLOT] = [9, 8, 7, 6, 5, 4, 3, 2];
        let pb: [u32; SLOT] = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut o1 = [0u32; SLOT];
        let mut o2 = [0u32; SLOT];
        let k1 = unsafe { xor_child_scalar(o1.as_mut_ptr(), pa.as_ptr(), pb.as_ptr()) };
        // Reference: rotate-left-one of the lane-wise XOR; new key = lane 1.
        let x: Vec<u32> = (0..SLOT).map(|i| pa[i] ^ pb[i]).collect();
        let expect: Vec<u32> = (0..SLOT).map(|i| x[(i + 1) % SLOT]).collect();
        assert_eq!(&o1[..], &expect[..]);
        assert_eq!(k1, x[1]);
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            let k2 = unsafe { xor_child_avx2(o2.as_mut_ptr(), pa.as_ptr(), pb.as_ptr()) };
            assert_eq!(o1, o2, "avx2 xor_child != scalar");
            assert_eq!(k1, k2);
        }
    }

    // Clamped solve over a fixed header must find valid solutions (Equihash
    // 192,7 yields ~2 per nonce). The clamp bounds the naive algorithm's
    // degenerate-collision blow-up, mirroring the GPU's fixed bucket slots.
    #[test]
    #[ignore]
    fn full_solve_finds_valid_solutions() {
        let header = vec![0x42u8; HEADER_LEN];
        let base = base_state(&header);
        for clamp in [8usize, 16, 32, 64] {
            let t = std::time::Instant::now();
            let sols = solve_with(&header, Some(clamp));
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            eprintln!("clamp={clamp}: {} solution(s) in {ms:.0} ms", sols.len());
            for s in &sols {
                assert!(is_valid_solution(&base, s), "solver returned an invalid solution");
            }
        }
    }
}
