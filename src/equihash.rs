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
// collision domains processed in parallel across the rayon pool.
//
// The data layout targets AVX2 throughput; the round is memory-bandwidth bound,
// so every choice below exists to cut DRAM traffic / random access:
//   * `partition_top` is a parallel counting sort (per-chunk histograms ->
//     per-chunk base offsets -> disjoint-region scatter), not a serial pass.
//   * It also emits `keys_part` (the leading keys in partition order) so the
//     per-partition `count_pairs`/`low_group` sweeps read keys sequentially
//     instead of gathering `keys[order[..]]`; `low_group` likewise emits
//     `keys_sorted` so the emit group walk streams a dense local copy.
//   * Slots hold residual hash words in a *narrowing* packed slot: round 0 is
//     `SLOT` words; each round consumes the leading block, so round r packs at
//     `SLOT - r` words. The XOR child is one 256-bit load/xor/permute that
//     masked-stores the live lanes (`xor_child_avx2`); buffers carry a
//     `SLOT_SLACK` pad for the 256-bit over-read of a narrow tail slot.
//   * `collide` runs two passes — count pairs, then emit children directly into
//     one pre-sized arena at per-partition offsets — so there is no per-bucket
//     allocation and no final concatenation copy; the `keys`/`slots` buffers
//     ping-pong between rounds.
//   * Round 0 repacks BLAKE2b digests into 24-bit blocks with `pshufb`.
// The remaining bottleneck is the random slot gather in `emit_bucket` over the
// ~1 GiB round buffer — the Equihash memory-hardness floor. A full radix scatter
// of the payload into bucket-contiguous storage (to make that gather sequential)
// was evaluated and loses: the wide-slot scatter costs more than the gather it
// saves on the early, wide rounds.
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

/// Number of u32 words in a round-0 entry slot (32 bytes = one AVX2 register).
/// Round 0 stores all eight 24-bit blocks; the XOR that produces a child is a
/// single 256-bit load/xor/permute — xenoncat's packed-slot trick (and the same
/// `uint4`-aligned-slot idea the CUDA backend uses), on the CPU. The leading
/// collision word lives in lane 0 and is mirrored into a dense parallel `keys[]`
/// array so the histogram passes stream over 4 bytes/entry instead of striding
/// the slots.
///
/// Later rounds use a *narrower* pitch: each collision round consumes the leading
/// block, so a round-`r` output entry only carries `8 - r` meaningful words. The
/// solver packs each round's slots at that width (see `collide`'s `w_out`),
/// cutting the per-round slot-buffer DRAM traffic that bounds the collision rounds.
/// The XOR producer still loads a full 256-bit register (over-reading up to
/// `SLOT` words past a narrow tail slot, hence the `SLOT_SLACK` pad) but
/// masked-stores only the `w_out` live lanes so packed neighbours aren't touched.
const SLOT: usize = 8;

/// Trailing pad (in u32 words) on every slot buffer so the XOR producer's 256-bit
/// load over a narrow tail slot stays in bounds. The over-read reaches at most
/// `(n-1)*w + SLOT` words for pitch `w`, i.e. `SLOT - w ≤ SLOT` words past the
/// `n*w` payload; `SLOT` words always covers it.
const SLOT_SLACK: usize = SLOT;

/// Scalar child producer: writes the `w_out` live words of `(a XOR b)` rotated
/// left one lane into `out`, and returns the child's new leading word (lane 1 of
/// the XOR). Lane 0 of the XOR is the just-collided block (always zero) and is
/// rotated out. `a`/`b` are read a full `SLOT` words wide (the caller pads each
/// slot buffer by `SLOT_SLACK`); only `out[0..w_out]` is written.
#[inline]
unsafe fn xor_child_scalar(out: *mut u32, a: *const u32, b: *const u32, w_out: usize) -> u32 {
    let mut x = [0u32; SLOT];
    for t in 0..SLOT {
        x[t] = *a.add(t) ^ *b.add(t);
    }
    // out[t] = x[(t + 1) % SLOT]; for t < w_out <= SLOT-1 the modulo is a no-op.
    for t in 0..w_out {
        *out.add(t) = x[t + 1];
    }
    x[1]
}

/// AVX2 child producer: one `vpxor` + one `vpermd` (rotate the 8 lanes left by
/// one), then a masked store of the low `w_out` lanes so the packed, `w_out`-
/// pitched output never clobbers the next slot.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn xor_child_avx2(out: *mut u32, a: *const u32, b: *const u32, w_out: usize) -> u32 {
    use core::arch::x86_64::*;
    let x = _mm256_xor_si256(
        _mm256_loadu_si256(a as *const __m256i),
        _mm256_loadu_si256(b as *const __m256i),
    );
    // rotate left by one 32-bit lane: out[i] = x[(i + 1) % 8]
    let p = _mm256_permutevar8x32_epi32(x, _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0));
    // mask = lanes [0, w_out) -> all-ones; maskstore writes only those.
    let mask = _mm256_cmpgt_epi32(
        _mm256_set1_epi32(w_out as i32),
        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
    );
    _mm256_maskstore_epi32(out as *mut i32, mask, p);
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

/// A raw `*mut u32` shared across rayon workers for the parallel scatter below.
/// Sound only because the workers write provably disjoint position sets.
#[derive(Clone, Copy)]
struct OrderPtr(*mut u32);
unsafe impl Send for OrderPtr {}
unsafe impl Sync for OrderPtr {}

/// Partition the `n` entries into `TOP_BUCKETS` runs by the high `TOP_BITS` of
/// their (dense) leading block. Returns `(starts, order, keys_part)`: partition
/// `v` owns the input indices `order[starts[v]..starts[v + 1]]`, and `keys_part`
/// is the entries' leading words in the same partition-contiguous order
/// (`keys_part[p] == keys[order[p]]`). Carrying `keys_part` lets the per-partition
/// `count_pairs`/`low_group` sweeps read keys sequentially instead of gathering
/// `keys[order[..]]` over the whole array — at the cost of one extra 4-byte/entry
/// scatter here, folded into the (already parallel) phase-3 pass.
///
/// Parallel counting sort: the input is split into one contiguous chunk per
/// rayon worker. Each chunk histograms its slice (phase 1), a small serial pass
/// turns those into per-chunk base offsets within each bucket's output region
/// (phase 2), and each chunk scatters its entries into `order`/`keys_part`
/// (phase 3). Chunk `c`'s bucket-`b` writes land in `[off[c][b], off[c+1][b])`,
/// disjoint from every other chunk and bucket, so the concurrent writes never
/// alias. Entries within a bucket end up chunk-major rather than index-major;
/// that reordering is immaterial — `count_pairs`/`low_group` depend only on the
/// multiset of low keys, and final solutions are canonicalised, de-duplicated,
/// and verified.
fn partition_top(keys: &[u32], n: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut starts = vec![0u32; TOP_BUCKETS + 1];
    let mut order = vec![0u32; n];
    let mut keys_part = vec![0u32; n];
    if n == 0 {
        return (starts, order, keys_part);
    }

    let nthreads = rayon::current_num_threads().max(1);
    let chunk = n.div_ceil(nthreads);
    let nchunks = n.div_ceil(chunk);

    // Phase 1 (parallel): per-chunk top-bucket histograms.
    let local_hists: Vec<Vec<u32>> = (0..nchunks)
        .into_par_iter()
        .map(|c| {
            let lo = c * chunk;
            let hi = ((c + 1) * chunk).min(n);
            let mut h = vec![0u32; TOP_BUCKETS];
            for &key in &keys[lo..hi] {
                h[(key >> LOW_BITS) as usize] += 1;
            }
            h
        })
        .collect();

    // Phase 2 (serial, nchunks * TOP_BUCKETS work): bucket starts, then each
    // chunk's per-bucket base offset within its bucket's output region.
    let mut totals = vec![0u32; TOP_BUCKETS];
    for h in &local_hists {
        for b in 0..TOP_BUCKETS {
            totals[b] += h[b];
        }
    }
    for b in 0..TOP_BUCKETS {
        starts[b + 1] = starts[b] + totals[b];
    }
    let mut offsets = vec![vec![0u32; TOP_BUCKETS]; nchunks];
    let mut running = starts[..TOP_BUCKETS].to_vec(); // running[b] starts at starts[b]
    for c in 0..nchunks {
        for b in 0..TOP_BUCKETS {
            offsets[c][b] = running[b];
            running[b] += local_hists[c][b];
        }
    }

    // Phase 3 (parallel): each chunk scatters into its disjoint sub-ranges.
    let optr = OrderPtr(order.as_mut_ptr());
    let kptr = OrderPtr(keys_part.as_mut_ptr());
    offsets.into_par_iter().enumerate().for_each(|(c, mut cur)| {
        let (optr, kptr) = (optr, kptr); // capture whole (Sync) wrappers
        let lo = c * chunk;
        let hi = ((c + 1) * chunk).min(n);
        let (obase, kbase) = (optr.0, kptr.0);
        for k in lo..hi {
            let key = keys[k];
            let b = (key >> LOW_BITS) as usize;
            let pos = cur[b] as usize;
            // SAFETY: `pos` ranges over `[off[c][b], off[c+1][b])`, a range owned
            // exclusively by chunk `c` and within `order`/`keys_part` bounds.
            unsafe {
                *obase.add(pos) = k as u32;
                *kbase.add(pos) = key;
            }
            cur[b] += 1;
        }
    });

    (starts, order, keys_part)
}

/// Count the colliding pairs a partition will emit, from the low-bit histogram
/// of its (partition-contiguous) leading keys `keys_run`. `clamp` caps each
/// exact-collision group, matching the emit pass so the output offsets line up.
/// `hist` is reusable `LOW_BUCKETS` scratch.
fn count_pairs(keys_run: &[u32], hist: &mut [u32], clamp: usize) -> usize {
    for h in hist.iter_mut() {
        *h = 0;
    }
    for &key in keys_run {
        hist[(key & LOW_MASK) as usize] += 1;
    }
    let mut pairs = 0usize;
    for i in 0..LOW_BUCKETS {
        let c = (hist[i] as usize).min(clamp);
        pairs += c * (c - 1) / 2;
    }
    pairs
}

/// Within one partition, group its entries by the low bits of their leading
/// block. Inputs are the partition-contiguous slabs `keys_run` (leading keys) and
/// `order_run` (matching global indices). Outputs, in low-key-sorted order:
/// `sorted` (the global indices, for the emit slot gather + back-refs) and
/// `keys_sorted` (the leading keys, so the emit group walk streams a dense local
/// array instead of gathering `keys[sorted[i]]`). `hist` is reusable
/// `LOW_BUCKETS + 1` scratch. Both reads are sequential over the slabs.
fn low_group(
    keys_run: &[u32],
    order_run: &[u32],
    hist: &mut [u32],
    sorted: &mut Vec<u32>,
    keys_sorted: &mut Vec<u32>,
) {
    let m = order_run.len();
    sorted.clear();
    keys_sorted.clear();
    if m == 0 {
        return;
    }
    for h in hist.iter_mut() {
        *h = 0;
    }
    for &key in keys_run {
        hist[(key & LOW_MASK) as usize + 1] += 1;
    }
    for i in 0..LOW_BUCKETS {
        hist[i + 1] += hist[i];
    }
    sorted.resize(m, 0);
    keys_sorted.resize(m, 0);
    // hist[low] now holds the run-start offset; reuse it as the live cursor.
    for i in 0..m {
        let key = keys_run[i];
        let low = (key & LOW_MASK) as usize;
        let pos = hist[low] as usize;
        sorted[pos] = order_run[i];
        keys_sorted[pos] = key;
        hist[low] += 1;
    }
}

/// Walk a partition's grouped entries (`sorted`) and emit one child per
/// colliding pair into the partition's output sub-slices, returning the count
/// written. `clamp` bounds each exact-collision group (`usize::MAX` = exact).
/// Monomorphised over the XOR producer so the AVX2 intrinsics inline cleanly
/// inside a `target_feature` wrapper while sharing one source of truth.
macro_rules! emit_bucket_body {
    ($keys_sorted:expr, $slots:expr, $sorted:expr, $kout:expr, $sout:expr, $pout:expr, $clamp:expr, $w_in:expr, $w_out:expr, $xor:path) => {{
        let s = $sorted;
        let ks = $keys_sorted; // leading keys in `s`-order; group walk streams it
        let m = s.len();
        let w_in = $w_in;
        let w_out = $w_out;
        let mut w = 0usize;
        let mut i = 0;
        while i < m {
            let key = ks[i] & LOW_MASK;
            let mut j = i + 1;
            while j < m && (ks[j] & LOW_MASK) == key {
                j += 1;
            }
            let hi = j.min(i.saturating_add($clamp));
            for a in i..hi {
                let l = s[a] as usize;
                for b in (a + 1)..hi {
                    let mr = s[b] as usize;
                    let nk = $xor(
                        $sout.as_mut_ptr().add(w * w_out),
                        $slots.as_ptr().add(l * w_in),
                        $slots.as_ptr().add(mr * w_in),
                        w_out,
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
    keys_sorted: &[u32],
    slots: &[u32],
    sorted: &[u32],
    kout: &mut [u32],
    sout: &mut [u32],
    pout: &mut [u64],
    clamp: usize,
    w_in: usize,
    w_out: usize,
) -> usize {
    emit_bucket_body!(keys_sorted, slots, sorted, kout, sout, pout, clamp, w_in, w_out, xor_child_scalar)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn emit_bucket_avx2(
    keys_sorted: &[u32],
    slots: &[u32],
    sorted: &[u32],
    kout: &mut [u32],
    sout: &mut [u32],
    pout: &mut [u64],
    clamp: usize,
    w_in: usize,
    w_out: usize,
) -> usize {
    emit_bucket_body!(keys_sorted, slots, sorted, kout, sout, pout, clamp, w_in, w_out, xor_child_avx2)
}

/// Emit a partition's children, dispatching to the AVX2 producer when available.
/// `w_in`/`w_out` are the input/output slot pitches (`w_out == w_in - 1`).
unsafe fn emit_bucket(
    keys_sorted: &[u32],
    slots: &[u32],
    sorted: &[u32],
    kout: &mut [u32],
    sout: &mut [u32],
    pout: &mut [u64],
    clamp: usize,
    w_in: usize,
    w_out: usize,
) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return emit_bucket_avx2(keys_sorted, slots, sorted, kout, sout, pout, clamp, w_in, w_out);
        }
    }
    emit_bucket_scalar(keys_sorted, slots, sorted, kout, sout, pout, clamp, w_in, w_out)
}

/// Group `n` entries by their leading block, then emit one child per colliding
/// pair: the XOR of the residual blocks (rotated into a fresh slot) plus a packed
/// `(l << 32) | mr` back-reference. Two passes — count, then emit directly into
/// one pre-sized arena — so there is no per-partition allocation or final
/// concatenation copy. Returns `(keys_out, slots_out, parents)`.
///
/// `w_in` is the input slot pitch; the round consumes one block, so the output is
/// packed at `w_out = w_in - 1` words/slot. `slots_out` carries a `SLOT_SLACK`
/// trailing pad so the next round's 256-bit over-read stays in bounds.
fn collide(
    keys: &[u32],
    slots: &[u32],
    n: usize,
    clamp: usize,
    w_in: usize,
) -> (Vec<u32>, Vec<u32>, Vec<u64>) {
    let w_out = w_in - 1;
    // Sub-phase timing, gated on `EQ_PROFILE`. Prints partition / count / alloc /
    // emit splits so we can see which part of the round dominates.
    let prof = std::env::var_os("EQ_PROFILE").is_some();
    let t0 = std::time::Instant::now();

    let (starts, order, keys_part) = partition_top(keys, n);
    let t_part = std::time::Instant::now();

    // Pass 1: per-partition child counts from each partition's contiguous keys.
    let counts: Vec<usize> = (0..TOP_BUCKETS)
        .into_par_iter()
        .map_init(
            || vec![0u32; LOW_BUCKETS],
            |hist, v| {
                let keys_run = &keys_part[starts[v] as usize..starts[v + 1] as usize];
                count_pairs(keys_run, hist, clamp)
            },
        )
        .collect();
    let t_count = std::time::Instant::now();

    let mut out_starts = vec![0usize; TOP_BUCKETS + 1];
    for v in 0..TOP_BUCKETS {
        out_starts[v + 1] = out_starts[v] + counts[v];
    }
    let total = out_starts[TOP_BUCKETS];
    let mut keys_out = vec![0u32; total];
    // Packed at `w_out` words/slot, plus a slack pad for the next round's over-read.
    let mut slots_out = vec![0u32; total * w_out + SLOT_SLACK];
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
            let (sh, st) = sr.split_at_mut(c * w_out);
            let (ph, pt) = pr.split_at_mut(c);
            kparts.push(kh);
            sparts.push(sh);
            pparts.push(ph);
            kr = kt;
            sr = st;
            pr = pt;
        }
    }

    let t_alloc = std::time::Instant::now();

    // Pass 2: group each partition and emit its colliding pairs in place.
    kparts
        .into_par_iter()
        .zip(sparts)
        .zip(pparts)
        .enumerate()
        .for_each(|(v, ((kout, sout), pout))| {
            let lo = starts[v] as usize;
            let hi = starts[v + 1] as usize;
            let mut hist = vec![0u32; LOW_BUCKETS + 1];
            let mut sorted = Vec::new();
            let mut keys_sorted = Vec::new();
            low_group(&keys_part[lo..hi], &order[lo..hi], &mut hist, &mut sorted, &mut keys_sorted);
            let w =
                unsafe { emit_bucket(&keys_sorted, slots, &sorted, kout, sout, pout, clamp, w_in, w_out) };
            debug_assert_eq!(w, kout.len());
        });

    if prof {
        let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1000.0;
        eprintln!(
            "    [collide n={n}] partition {:6.1}  count {:6.1}  alloc {:6.1}  emit(group+xor) {:6.1} ms",
            ms(t0, t_part),
            ms(t_part, t_count),
            ms(t_count, t_alloc),
            ms(t_alloc, std::time::Instant::now()),
        );
    }

    (keys_out, slots_out, parents)
}

/// Final round (slots hold `[w0, w1, …]` at pitch `w_in`): among entries sharing
/// leading block `w0`, a pair whose `w1` also matches XORs the last two blocks to
/// zero — a candidate. Returns the `(l, mr)` parents of each candidate.
fn collide_final(keys: &[u32], slots: &[u32], n: usize, clamp: usize, w_in: usize) -> Vec<(u32, u32)> {
    let (starts, order, keys_part) = partition_top(keys, n);

    (0..TOP_BUCKETS)
        .into_par_iter()
        .map_init(
            || (vec![0u32; LOW_BUCKETS + 1], Vec::<u32>::new(), Vec::<u32>::new()),
            |(hist, sorted, keys_sorted), v| {
                let lo = starts[v] as usize;
                let hi = starts[v + 1] as usize;
                low_group(&keys_part[lo..hi], &order[lo..hi], hist, sorted, keys_sorted);
                let m = sorted.len();
                let mut local = Vec::new();
                let mut i = 0;
                while i < m {
                    let key = keys_sorted[i] & LOW_MASK;
                    let mut j = i + 1;
                    while j < m && (keys_sorted[j] & LOW_MASK) == key {
                        j += 1;
                    }
                    let hi = j.min(i.saturating_add(clamp));
                    for a in i..hi {
                        let l = sorted[a] as usize;
                        for b in (a + 1)..hi {
                            let mr = sorted[b] as usize;
                            if slots[l * w_in + 1] == slots[mr * w_in + 1] {
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

    // Optional per-phase timing, gated on `EQ_PROFILE` (any value). Zero cost
    // when unset — the only overhead is one env lookup. Each `phase(label, n)`
    // prints the wall time since the previous call and the live entry count.
    let prof = std::env::var_os("EQ_PROFILE").is_some();
    let mut t_last = std::time::Instant::now();
    let mut phase = |label: &str, n: usize| {
        if prof {
            let now = std::time::Instant::now();
            eprintln!(
                "  [profile] {label:<13} {:8.1} ms  (n={n})",
                (now - t_last).as_secs_f64() * 1000.0
            );
            t_last = now;
        }
    };

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
    phase("round0-hash", n0);
    let mut parents: Vec<Vec<u64>> = Vec::with_capacity(K - 1);
    let mut n = n0;
    // Round-0 slots carry all SLOT words; each collision round consumes the
    // leading block, so the next round's slots are one word narrower.
    let mut width = SLOT;
    for r in 0..(K - 1) {
        let (ok, os, op) = collide(&keys, &slots, n, clamp, width);
        n = op.len();
        parents.push(op);
        keys = ok;
        slots = os;
        width -= 1;
        phase(&format!("collide r{}", r + 1), n);
        if n == 0 {
            return Vec::new();
        }
    }

    let candidates = collide_final(&keys, &slots, n, clamp, width);
    phase("collide-final", candidates.len());
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
    phase("recover", recovered.len() / SOLUTION_INDICES);

    let result = filter_candidates(&base_state(header), &recovered);
    phase("filter+verify", result.len());
    result
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

    // The XOR-child producers (scalar + AVX2) must agree for every output width,
    // write exactly `w_out` words (rotate-left-one of the lane-wise XOR), and
    // leave the rest of the packed buffer untouched (the AVX2 path masked-stores).
    #[test]
    fn xor_child_matches_scalar() {
        const SENT: u32 = 0xDEAD_BEEF;
        let pa: [u32; SLOT] = [9, 8, 7, 6, 5, 4, 3, 2];
        let pb: [u32; SLOT] = [1, 2, 3, 4, 5, 6, 7, 8];
        let x: Vec<u32> = (0..SLOT).map(|i| pa[i] ^ pb[i]).collect();
        for w_out in 1..SLOT {
            let mut o1 = [SENT; SLOT];
            let k1 = unsafe { xor_child_scalar(o1.as_mut_ptr(), pa.as_ptr(), pb.as_ptr(), w_out) };
            for t in 0..w_out {
                assert_eq!(o1[t], x[(t + 1) % SLOT], "scalar word {t} (w_out={w_out})");
            }
            for t in w_out..SLOT {
                assert_eq!(o1[t], SENT, "scalar wrote past w_out at {t} (w_out={w_out})");
            }
            assert_eq!(k1, x[1]);
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") {
                let mut o2 = [SENT; SLOT];
                let k2 = unsafe { xor_child_avx2(o2.as_mut_ptr(), pa.as_ptr(), pb.as_ptr(), w_out) };
                assert_eq!(o1, o2, "avx2 xor_child != scalar (w_out={w_out})");
                assert_eq!(k1, k2);
            }
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

    /// A deterministic dense 140-byte header for benchmarking, filled by an
    /// xorshift64* stream keyed on `seed` (the same scheme as `main::pseudo_header`)
    /// so each nonce exercises a realistic, fully-populated header rather than the
    /// degenerate all-`0x42` one.
    #[cfg(test)]
    fn bench_header(seed: u64) -> Vec<u8> {
        let mut header = vec![0u8; HEADER_LEN];
        let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        for b in header.iter_mut() {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            *b = (x.wrapping_mul(0x2545F4914F6CDD1D) >> 33) as u8;
        }
        header
    }

    /// Steady-state throughput baseline for the CPU solver. Ignored by default;
    /// run it in release (the only meaningful configuration — `debug_assert`s in
    /// the collision hot path are off and `opt-level=3`/LTO is on) with output
    /// visible:
    ///
    /// ```text
    /// cargo test --release full_solve_baseline -- --ignored --nocapture
    /// ```
    ///
    /// Tunables via env:
    ///   * `EQ_BENCH_ITERS`  — timed solves per clamp (default 8).
    ///   * `EQ_BENCH_CLAMPS` — comma-separated clamp sweep (default `32`, the
    ///     value the miner and `--selftest` use).
    ///
    /// Each timed iteration uses a distinct nonce so the figures average over
    /// per-header variance; one warm-up solve per clamp (pages in the ~GB of
    /// round buffers, warms caches) is excluded. This measures one solve at a
    /// time across the whole rayon pool — i.e. solver latency and its memory
    /// behaviour, which is exactly what the planned radix-scatter change targets
    /// — not the aggregate multi-group mining rate `--bench` reports.
    #[test]
    #[ignore]
    fn full_solve_baseline() {
        use std::time::Instant;

        let iters: usize = std::env::var("EQ_BENCH_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        let clamps: Vec<usize> = std::env::var("EQ_BENCH_CLAMPS")
            .ok()
            .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
            .filter(|v: &Vec<usize>| !v.is_empty())
            .unwrap_or_else(|| vec![32]);

        eprintln!(
            "CPU solver baseline — {iters} timed solve(s)/clamp, clamps={clamps:?}, \
             rayon threads={}",
            rayon::current_num_threads()
        );

        for &clamp in &clamps {
            // Warm-up (excluded): fault in buffers, warm caches.
            let _ = solve_with(&bench_header(0), Some(clamp));

            let mut times = Vec::with_capacity(iters);
            let mut total_sols = 0usize;
            for i in 0..iters {
                let header = bench_header(i as u64 + 1);
                let base = base_state(&header);
                let t = Instant::now();
                let sols = solve_with(&header, Some(clamp));
                times.push(t.elapsed().as_secs_f64());
                for s in &sols {
                    assert!(is_valid_solution(&base, s), "solver returned an invalid solution");
                }
                total_sols += sols.len();
            }

            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = times.len();
            let sum: f64 = times.iter().sum();
            let mean = 1000.0 * sum / n as f64;
            let median = 1000.0 * times[n / 2];
            let min = 1000.0 * times[0];
            let max = 1000.0 * times[n - 1];
            let solves_per_s = n as f64 / sum;
            eprintln!(
                "clamp={clamp:>3}: {solves_per_s:6.2} solve/s | mean {mean:6.0} ms  \
                 median {median:6.0} ms  min {min:6.0} ms  max {max:6.0} ms | \
                 {total_sols} sol over {n} solve(s) ({:.2}/solve)",
                total_sols as f64 / n as f64,
            );
        }
    }
}
