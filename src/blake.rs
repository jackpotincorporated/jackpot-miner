//! Personalised BLAKE2b state used by Equihash.
//!
//! The personalisation string is `"ZcashPoW" || LE32(n) || LE32(k)` and the
//! digest length is `HASH_OUTPUT` bytes. The base state absorbs the 140-byte
//! block header; each index `g` is then hashed by cloning the base state and
//! appending `LE32(g)`.
//!
//! [`generate_hash`] (one hash at a time, via `blake2b_simd`) is used by the
//! verification paths. The solver's hot round-0 loop instead uses
//! [`BatchHasher`], which precomputes the midstate of the shared first 128-byte
//! header block once and then compresses only the final 16-byte block — four
//! `g` values at a time with AVX2 — mirroring xenoncat's hashing strategy.

use crate::params::{HASH_OUTPUT, HEADER_LEN, K, N};
use blake2b_simd::{Params, State};

/// Build the Equihash personalisation block: `"ZcashPoW" || LE32(n) || LE32(k)`.
pub fn personalization() -> [u8; 16] {
    let mut personal = [0u8; 16];
    personal[..8].copy_from_slice(b"ZcashPoW");
    personal[8..12].copy_from_slice(&(N as u32).to_le_bytes());
    personal[12..16].copy_from_slice(&(K as u32).to_le_bytes());
    personal
}

/// Create the base BLAKE2b state with the header absorbed.
///
/// `header` must be exactly the 140 pre-solution header bytes.
pub fn base_state(header: &[u8]) -> State {
    let personal = personalization();
    let mut state = Params::new()
        .hash_length(HASH_OUTPUT)
        .personal(&personal)
        .to_state();
    state.update(header);
    state
}

/// Compute the BLAKE2b output for index `g` from a base state.
#[inline]
pub fn generate_hash(base: &State, g: u32) -> [u8; HASH_OUTPUT] {
    let mut s = base.clone();
    s.update(&g.to_le_bytes());
    let hash = s.finalize();
    let mut out = [0u8; HASH_OUTPUT];
    out.copy_from_slice(hash.as_bytes());
    out
}

// ---------------------------------------------------------------------------
// Batched midstate hasher (round-0 hot path).
//
// The 140-byte header plus the 4-byte index `g` is 144 bytes = two BLAKE2b
// blocks. The first 128-byte block is identical for every `g`, so we compress
// it once into a midstate and, per `g`, compress only the final block (the 12
// trailing header bytes + `LE32(g)`, zero-padded). With AVX2 four such final
// blocks are compressed in parallel lanes.
// ---------------------------------------------------------------------------

/// BLAKE2b IV.
const IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

/// BLAKE2b message schedule.
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

/// Number of trailing header bytes that land in the final block (140 - 128).
const TAIL_LEN: usize = HEADER_LEN - 128;
/// Cumulative byte count for the (last) final-block compression: 140 + 4.
const FINAL_COUNT: u64 = (HEADER_LEN + 4) as u64;

/// Initial BLAKE2b state words: IV xor'd with the parameter block (digest
/// length 48, fanout/depth 1, our personalisation in bytes 48..64).
fn init_words() -> [u64; 8] {
    let mut param = [0u8; 64];
    param[0] = HASH_OUTPUT as u8; // digest length
    param[2] = 1; // fanout
    param[3] = 1; // depth
    param[48..64].copy_from_slice(&personalization());
    let mut h = IV;
    for i in 0..8 {
        h[i] ^= u64::from_le_bytes(param[i * 8..i * 8 + 8].try_into().unwrap());
    }
    h
}

#[inline(always)]
fn scalar_g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

/// Scalar BLAKE2b compression of one 128-byte block into `h`.
fn scalar_compress(h: &mut [u64; 8], block: &[u8; 128], count: u64, last: bool) {
    let mut m = [0u64; 16];
    for i in 0..16 {
        m[i] = u64::from_le_bytes(block[i * 8..i * 8 + 8].try_into().unwrap());
    }
    let mut v = [0u64; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&IV);
    v[12] ^= count;
    if last {
        v[14] ^= !0u64;
    }
    for s in &SIGMA {
        scalar_g(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        scalar_g(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        scalar_g(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        scalar_g(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        scalar_g(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        scalar_g(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        scalar_g(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        scalar_g(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

/// Precomputed midstate for a fixed header, hashing many `g` indices.
#[derive(Clone)]
pub struct BatchHasher {
    /// State words after compressing the shared first 128-byte block.
    mid: [u64; 8],
    /// The trailing header bytes (140 - 128) that begin the final block.
    tail: [u8; TAIL_LEN],
}

impl BatchHasher {
    pub fn new(header: &[u8]) -> Self {
        assert_eq!(header.len(), HEADER_LEN, "header must be {HEADER_LEN} bytes");
        let mut mid = init_words();
        let mut blk = [0u8; 128];
        blk.copy_from_slice(&header[..128]);
        scalar_compress(&mut mid, &blk, 128, false);
        let mut tail = [0u8; TAIL_LEN];
        tail.copy_from_slice(&header[128..HEADER_LEN]);
        Self { mid, tail }
    }

    /// Assemble the zero-padded final block for index `g`.
    #[inline]
    fn final_block(&self, g: u32) -> [u8; 128] {
        let mut blk = [0u8; 128];
        blk[..TAIL_LEN].copy_from_slice(&self.tail);
        blk[TAIL_LEN..TAIL_LEN + 4].copy_from_slice(&g.to_le_bytes());
        blk
    }

    /// Hash a single index `g` (scalar; used as the AVX2 fallback and in tests).
    pub fn hash1(&self, g: u32) -> [u8; HASH_OUTPUT] {
        let mut h = self.mid;
        let blk = self.final_block(g);
        scalar_compress(&mut h, &blk, FINAL_COUNT, true);
        let mut out = [0u8; HASH_OUTPUT];
        for i in 0..HASH_OUTPUT / 8 {
            out[i * 8..i * 8 + 8].copy_from_slice(&h[i].to_le_bytes());
        }
        out
    }

    /// Hash four consecutive indices `g0, g0+1, g0+2, g0+3`, writing each
    /// 48-byte digest into `out[0..4]`. Uses AVX2 when available.
    #[inline]
    pub fn hash4(&self, g0: u32, out: &mut [[u8; HASH_OUTPUT]; 4]) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { self.hash4_avx2(g0, out) };
                return;
            }
        }
        for (l, o) in out.iter_mut().enumerate() {
            *o = self.hash1(g0 + l as u32);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn hash4_avx2(&self, g0: u32, out: &mut [[u8; HASH_OUTPUT]; 4]) {
        use core::arch::x86_64::*;

        #[inline(always)]
        unsafe fn rotr32(x: __m256i) -> __m256i {
            _mm256_or_si256(_mm256_srli_epi64(x, 32), _mm256_slli_epi64(x, 32))
        }
        #[inline(always)]
        unsafe fn rotr24(x: __m256i) -> __m256i {
            _mm256_or_si256(_mm256_srli_epi64(x, 24), _mm256_slli_epi64(x, 40))
        }
        #[inline(always)]
        unsafe fn rotr16(x: __m256i) -> __m256i {
            _mm256_or_si256(_mm256_srli_epi64(x, 16), _mm256_slli_epi64(x, 48))
        }
        #[inline(always)]
        unsafe fn rotr63(x: __m256i) -> __m256i {
            _mm256_or_si256(_mm256_srli_epi64(x, 63), _mm256_slli_epi64(x, 1))
        }
        #[inline(always)]
        unsafe fn g4(v: &mut [__m256i; 16], a: usize, b: usize, c: usize, d: usize, x: __m256i, y: __m256i) {
            v[a] = _mm256_add_epi64(_mm256_add_epi64(v[a], v[b]), x);
            v[d] = rotr32(_mm256_xor_si256(v[d], v[a]));
            v[c] = _mm256_add_epi64(v[c], v[d]);
            v[b] = rotr24(_mm256_xor_si256(v[b], v[c]));
            v[a] = _mm256_add_epi64(_mm256_add_epi64(v[a], v[b]), y);
            v[d] = rotr16(_mm256_xor_si256(v[d], v[a]));
            v[c] = _mm256_add_epi64(v[c], v[d]);
            v[b] = rotr63(_mm256_xor_si256(v[b], v[c]));
        }

        // Message words. Only m0 and m1 are nonzero; m1's high 32 bits hold the
        // per-lane `g`. (TAIL_LEN is 12 for 192,7: 8 bytes -> m0, 4 -> m1 low.)
        let tail0 = u64::from_le_bytes(self.tail[0..8].try_into().unwrap());
        let tail_hi = u32::from_le_bytes(self.tail[8..12].try_into().unwrap()) as u64;
        let m1_lane = |g: u32| (tail_hi | ((g as u64) << 32)) as i64;
        let mut m = [_mm256_setzero_si256(); 16];
        m[0] = _mm256_set1_epi64x(tail0 as i64);
        m[1] = _mm256_set_epi64x(m1_lane(g0 + 3), m1_lane(g0 + 2), m1_lane(g0 + 1), m1_lane(g0));

        let mut v = [_mm256_setzero_si256(); 16];
        for i in 0..8 {
            v[i] = _mm256_set1_epi64x(self.mid[i] as i64);
            v[i + 8] = _mm256_set1_epi64x(IV[i] as i64);
        }
        v[12] = _mm256_xor_si256(v[12], _mm256_set1_epi64x(FINAL_COUNT as i64));
        v[14] = _mm256_xor_si256(v[14], _mm256_set1_epi64x(-1)); // last-block flag

        for s in &SIGMA {
            g4(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
            g4(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
            g4(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
            g4(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
            g4(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
            g4(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
            g4(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
            g4(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
        }

        // Output: digest is h[i] = mid[i] ^ v[i] ^ v[i+8]; we need the first
        // HASH_OUTPUT/8 words (48 bytes -> 6 words) per lane, little-endian.
        let mut tmp = [0u64; 4];
        for i in 0..HASH_OUTPUT / 8 {
            let o = _mm256_xor_si256(
                _mm256_xor_si256(_mm256_set1_epi64x(self.mid[i] as i64), v[i]),
                v[i + 8],
            );
            _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, o);
            for l in 0..4 {
                out[l][i * 8..i * 8 + 8].copy_from_slice(&tmp[l].to_le_bytes());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_matches_reference() {
        // Deterministic 140-byte header.
        let header: Vec<u8> = (0..HEADER_LEN as u32).map(|i| (i.wrapping_mul(2654435761) >> 24) as u8).collect();
        let base = base_state(&header);
        let hasher = BatchHasher::new(&header);

        // hash1 (scalar) must match the reference for assorted g.
        for &g in &[0u32, 1, 2, 3, 4, 7, 255, 1024, 1_000_003, u32::MAX] {
            assert_eq!(hasher.hash1(g), generate_hash(&base, g), "hash1 mismatch at g={g}");
        }

        // hash4 (AVX2 or scalar fallback) must match for several aligned batches.
        for base_g in [0u32, 4, 100, 1_000_000] {
            let mut out = [[0u8; HASH_OUTPUT]; 4];
            hasher.hash4(base_g, &mut out);
            for l in 0..4 {
                assert_eq!(out[l], generate_hash(&base, base_g + l as u32), "hash4 mismatch at g={}", base_g + l as u32);
            }
        }
    }
}
