//! BLAKE2b-512 with Equihash (ZcashPoW, n=192, k=7) personalization, plus the
//! "midstate" client hands to `digit_f`: the BLAKE2b state after compressing the
//! first 128-byte block of the block header.
//!
//! Equihash params for 192,7:
//!   personalization = "ZcashPoW" || LE32(192) || LE32(7)
//!   digest_length   = (512/192)*192/8 = 48 bytes (2 indices x 24 bytes)
//!
//! The 140-byte header + 4-byte index = 144 bytes hashed per index. The first
//! 128 bytes are header-independent of the index, so the client precompresses them on
//! the CPU into the 64-byte (8x u64) midstate; the GPU finishes per index.

const IV: [u64; 8] = [
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
fn g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
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
    v[8..].copy_from_slice(&IV);
    v[12] ^= t as u64;
    v[13] ^= (t >> 64) as u64;
    if last {
        v[14] = !v[14];
    }
    for r in 0..12 {
        let s = &SIGMA[r];
        g(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        g(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        g(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        g(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        g(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        g(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        g(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        g(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

/// Known-answer self-test of the core compression: standard BLAKE2b-512("abc").
pub fn selftest() -> bool {
    let mut h = IV;
    h[0] ^= 0x0101_0000 ^ 64; // digest_length=64, fanout=1, depth=1, no personalization
    let msg = b"abc";
    let mut block = [0u8; 128];
    block[..3].copy_from_slice(msg);
    compress(&mut h, &block, 3, true);
    let mut out = [0u8; 64];
    for i in 0..8 {
        out[i * 8..i * 8 + 8].copy_from_slice(&h[i].to_le_bytes());
    }
    // RFC 7693 test vector for BLAKE2b-512("abc")
    let expect: [u8; 64] = [
        0xba, 0x80, 0xa5, 0x3f, 0x98, 0x1c, 0x4d, 0x0d, 0x6a, 0x27, 0x97, 0xb6, 0x9f, 0x12, 0xf6, 0xe9,
        0x4c, 0x21, 0x2f, 0x14, 0x68, 0x5a, 0xc4, 0xb7, 0x4b, 0x12, 0xbb, 0x6f, 0xdb, 0xff, 0xa2, 0xd1,
        0x7d, 0x87, 0xc5, 0x39, 0x2a, 0xab, 0x79, 0x2d, 0xc2, 0x52, 0xd5, 0xde, 0x45, 0x33, 0xcc, 0x95,
        0x18, 0xd3, 0x8a, 0xa8, 0xdb, 0xf1, 0x92, 0x5a, 0xb9, 0x23, 0x86, 0xed, 0xd4, 0x00, 0x99, 0x23,
    ];
    out == expect
}

/// Initial BLAKE2b state for Equihash(192,7).
pub fn init_state() -> [u64; 8] {
    let mut personal = [0u8; 16];
    personal[..8].copy_from_slice(b"ZcashPoW");
    personal[8..12].copy_from_slice(&192u32.to_le_bytes());
    personal[12..16].copy_from_slice(&7u32.to_le_bytes());

    let mut h = IV;
    // param block: digest_length=48, key=0, fanout=1, depth=1
    h[0] ^= 0x0101_0000 ^ 48;
    // words 6,7 hold the 16-byte personalization
    h[6] ^= u64::from_le_bytes(personal[0..8].try_into().unwrap());
    h[7] ^= u64::from_le_bytes(personal[8..16].try_into().unwrap());
    h
}

/// The 64-byte midstate digit_f expects: state after compressing header[0..128].
/// `header` must be the 140-byte block header.
pub fn midstate(header: &[u8]) -> [u8; 64] {
    assert!(header.len() >= 128, "header must be >= 128 bytes");
    let mut h = init_state();
    let mut block = [0u8; 128];
    block.copy_from_slice(&header[0..128]);
    compress(&mut h, &block, 128, false);
    let mut out = [0u8; 64];
    for i in 0..8 {
        out[i * 8..i * 8 + 8].copy_from_slice(&h[i].to_le_bytes());
    }
    out
}

/// Finalize a hash directly from a 64-byte midstate (h[0..8]) plus a final block
/// whose first `idx_len` bytes are LE(idx_word) and the rest zero, with the
/// total byte counter `t_total`. Returns the 48-byte digest (h[0..6]).
/// Used to test the GPU's per-index hash construction (midstate + index, no tail).
pub fn digest_from_midstate(mid: &[u8; 64], idx_word: u32, idx_len: usize, t_total: u128) -> [u8; 48] {
    let mut h = [0u64; 8];
    for i in 0..8 {
        h[i] = u64::from_le_bytes(mid[i * 8..i * 8 + 8].try_into().unwrap());
    }
    let mut block = [0u8; 128];
    let w = idx_word.to_le_bytes();
    block[..idx_len.min(4)].copy_from_slice(&w[..idx_len.min(4)]);
    compress(&mut h, &block, t_total, true);
    let mut out = [0u8; 48];
    for i in 0..6 {
        out[i * 8..i * 8 + 8].copy_from_slice(&h[i].to_le_bytes());
    }
    out
}

/// Full Equihash(192,7) per-index hash: BLAKE2b(header || LE32(g)) -> 48 bytes,
/// where g = index / 2. Returns the 24-byte half selected by (index & 1).
/// Used for solution verification (reference path, midstate not required).
pub fn index_hash(header: &[u8], index: u32) -> [u8; 24] {
    let mut h = init_state();
    // header is 140 bytes; append LE32(g) -> 144 bytes total = one full block + 16
    let g_word = (index / 2).to_le_bytes();
    let mut input = Vec::with_capacity(144);
    input.extend_from_slice(&header[..140]);
    input.extend_from_slice(&g_word);
    // block 1
    let mut b0 = [0u8; 128];
    b0.copy_from_slice(&input[0..128]);
    compress(&mut h, &b0, 128, false);
    // final block (16 bytes used, rest zero), t = 144, last
    let mut b1 = [0u8; 128];
    b1[..16].copy_from_slice(&input[128..144]);
    compress(&mut h, &b1, 144, true);
    let mut full = [0u8; 64];
    for i in 0..8 {
        full[i * 8..i * 8 + 8].copy_from_slice(&h[i].to_le_bytes());
    }
    // 48-byte digest -> two 24-byte index hashes
    let half = (index & 1) as usize * 24;
    full[half..half + 24].try_into().unwrap()
}
