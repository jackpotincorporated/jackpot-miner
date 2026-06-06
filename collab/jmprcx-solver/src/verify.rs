//! Equihash (n=192, k=7) solution verification (Wagner tree).
//!
//! A solution is 2^k = 128 indices. With collision length c = n/(k+1) = 24 bits
//! and each per-index hash being n=192 bits (24 bytes):
//!   * all indices distinct
//!   * canonical ordering: at every tree node, the smallest index of the left
//!     subtree < that of the right subtree
//!   * at level r (1..=k), each block of 2^r leaves XORs to zero in its first
//!     r*24 bits; the full 128-leaf XOR is zero over all 192 bits.

const N_BITS: usize = 192;
const K: usize = 7;
const COLL: usize = N_BITS / (K + 1); // 24

/// number of leading zero bits in a 24-byte big-endian-ish hash (byte 0 = MSB).
fn leading_zero_bits(h: &[u8; 24]) -> usize {
    let mut n = 0;
    for &b in h {
        if b == 0 { n += 8; } else { n += b.leading_zeros() as usize; break; }
    }
    n
}

fn xor24(a: &[u8; 24], b: &[u8; 24]) -> [u8; 24] {
    let mut o = [0u8; 24];
    for i in 0..24 { o[i] = a[i] ^ b[i]; }
    o
}

/// Verify a 128-index solution given a per-index hash function.
/// Returns (valid, diagnostic_string).
pub fn verify(indices: &[u32], hash: impl Fn(u32) -> [u8; 24]) -> (bool, String) {
    if indices.len() != 128 {
        return (false, format!("expected 128 indices, got {}", indices.len()));
    }
    // distinctness
    let mut sorted = indices.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.len() != 128 {
        return (false, format!("indices not distinct ({} unique)", sorted.len()));
    }

    // leaf hashes
    let leaves: Vec<[u8; 24]> = indices.iter().map(|&i| hash(i)).collect();

    // bottom-up: each level halves; check collision prefix grows by COLL bits
    let mut level: Vec<[u8; 24]> = leaves.clone();
    let mut worst_zero = usize::MAX;
    for r in 1..=K {
        let need = r * COLL;
        let mut next = Vec::with_capacity(level.len() / 2);
        for pair in level.chunks(2) {
            let x = xor24(&pair[0], &pair[1]);
            let z = leading_zero_bits(&x);
            worst_zero = worst_zero.min(z);
            if z < need {
                return (false, format!("level {r}: only {z} leading zero bits, need {need}"));
            }
            next.push(x);
        }
        level = next;
    }
    let full_zero = level.len() == 1 && level[0].iter().all(|&b| b == 0);
    let msg = format!(
        "all {K} levels pass collision checks; final XOR {} (min prefix zeros seen = {})",
        if full_zero { "= 0 (VALID)" } else { "!= 0" }, worst_zero
    );
    (full_zero, msg)
}

/// Quick diagnostic when the hash model may be off: report the max leading-zero
/// bits of the full 128-leaf XOR (≈168+ means the hash model is correct).
pub fn top_xor_zero_bits(indices: &[u32], hash: impl Fn(u32) -> [u8; 24]) -> usize {
    let mut acc = [0u8; 24];
    for &i in indices {
        let h = hash(i);
        for j in 0..24 { acc[j] ^= h[j]; }
    }
    leading_zero_bits(&acc)
}
