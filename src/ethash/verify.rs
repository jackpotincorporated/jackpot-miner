//! CPU Ethash light verifier — recompute a candidate share on the CPU and accept
//! it only if it genuinely meets target, so the (reverse-engineered) GPU replay
//! can never submit a bad share. Mirrors the role of `equihash::is_valid_solution`.
//!
//! Unlike pearlhash, Ethash is a public spec, so this is a *correct* light-client
//! implementation (no DAG needed — dataset items are derived from the cache on the
//! fly), not a stub. The full pipeline (mkcache → dataset_item → hashimoto → result)
//! is pinned by `hashimoto_geth_kat` below against go-ethereum's canonical
//! `TestHashimoto` vector, and the Keccak core has its own KATs.
//!
//! Spec: <https://eth.wiki/concepts/ethash/ethash> (Dagger-Hashimoto).

// ---------- Keccak-f[1600] (original Keccak padding 0x01, as Ethash uses) ----------

const RC: [u64; 24] = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
    0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
];
const RHO: [u32; 24] = [
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
];
const PI: [usize; 24] = [
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
];

fn keccakf(a: &mut [u64; 25]) {
    for round in 0..24 {
        // theta
        let mut c = [0u64; 5];
        for x in 0..5 {
            c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        }
        for x in 0..5 {
            let d = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            for y in 0..5 {
                a[x + 5 * y] ^= d;
            }
        }
        // rho + pi
        let mut last = a[1];
        for i in 0..24 {
            let j = PI[i];
            let tmp = a[j];
            a[j] = last.rotate_left(RHO[i]);
            last = tmp;
        }
        // chi
        for y in 0..5 {
            let t: [u64; 5] = [a[5 * y], a[5 * y + 1], a[5 * y + 2], a[5 * y + 3], a[5 * y + 4]];
            for x in 0..5 {
                a[x + 5 * y] = t[x] ^ ((!t[(x + 1) % 5]) & t[(x + 2) % 5]);
            }
        }
        // iota
        a[0] ^= RC[round];
    }
}

/// Generic Keccak sponge with original-Keccak padding (delimiter 0x01).
fn keccak(rate: usize, input: &[u8], out: &mut [u8]) {
    debug_assert!(rate % 8 == 0 && out.len() <= rate);
    let mut st = [0u64; 25];
    let absorb = |st: &mut [u64; 25], block: &[u8]| {
        for i in 0..rate / 8 {
            st[i] ^= u64::from_le_bytes(block[i * 8..i * 8 + 8].try_into().unwrap());
        }
    };
    let mut off = 0;
    while input.len() - off >= rate {
        absorb(&mut st, &input[off..off + rate]);
        keccakf(&mut st);
        off += rate;
    }
    let rem = &input[off..];
    let mut block = vec![0u8; rate];
    block[..rem.len()].copy_from_slice(rem);
    block[rem.len()] ^= 0x01;
    block[rate - 1] ^= 0x80;
    absorb(&mut st, &block);
    keccakf(&mut st);
    let mut bytes = [0u8; 200];
    for i in 0..25 {
        bytes[i * 8..i * 8 + 8].copy_from_slice(&st[i].to_le_bytes());
    }
    out.copy_from_slice(&bytes[..out.len()]);
}

fn keccak256(input: &[u8]) -> [u8; 32] {
    let mut o = [0u8; 32];
    keccak(136, input, &mut o);
    o
}
fn keccak512(input: &[u8]) -> [u8; 64] {
    let mut o = [0u8; 64];
    keccak(72, input, &mut o);
    o
}

// ---------- Ethash parameters ----------

const HASH_BYTES: usize = 64;
const MIX_BYTES: usize = 128;
const EPOCH_LENGTH: u64 = 30_000;
const CACHE_BYTES_INIT: usize = 1 << 24;
const CACHE_BYTES_GROWTH: usize = 1 << 17;
const DATASET_BYTES_INIT: usize = 1 << 30;
const DATASET_BYTES_GROWTH: usize = 1 << 23;
const DATASET_PARENTS: u32 = 256;
const CACHE_ROUNDS: usize = 3;
const ACCESSES: usize = 64;
const FNV_PRIME: u32 = 0x0100_0193;

#[inline]
fn fnv(a: u32, b: u32) -> u32 {
    a.wrapping_mul(FNV_PRIME) ^ b
}

fn is_prime(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }
    let mut i = 3usize;
    while i * i <= n {
        if n % i == 0 {
            return false;
        }
        i += 2;
    }
    true
}

pub fn epoch(block_number: u64) -> u64 {
    block_number / EPOCH_LENGTH
}

/// Seed hash for an epoch (Keccak-256 chained from zero, `epoch` times).
pub fn seedhash(epoch: u64) -> [u8; 32] {
    let mut s = [0u8; 32];
    for _ in 0..epoch {
        s = keccak256(&s);
    }
    s
}

/// Recover the epoch from a pool-supplied seedhash (bounded scan).
pub fn epoch_from_seedhash(seed: &[u8; 32]) -> Option<u64> {
    let mut s = [0u8; 32];
    for e in 0..(1u64 << 20) {
        if &s == seed {
            return Some(e);
        }
        s = keccak256(&s);
    }
    None
}

fn cache_size(epoch: u64) -> usize {
    let mut sz = CACHE_BYTES_INIT + CACHE_BYTES_GROWTH * epoch as usize - HASH_BYTES;
    while !is_prime(sz / HASH_BYTES) {
        sz -= 2 * HASH_BYTES;
    }
    sz
}

fn full_size(epoch: u64) -> usize {
    let mut sz = DATASET_BYTES_INIT + DATASET_BYTES_GROWTH * epoch as usize - MIX_BYTES;
    while !is_prime(sz / MIX_BYTES) {
        sz -= 2 * MIX_BYTES;
    }
    sz
}

#[inline]
fn words(row: &[u8; 64]) -> [u32; 16] {
    let mut w = [0u32; 16];
    for i in 0..16 {
        w[i] = u32::from_le_bytes(row[i * 4..i * 4 + 4].try_into().unwrap());
    }
    w
}
#[inline]
fn bytes(w: &[u32; 16]) -> [u8; 64] {
    let mut b = [0u8; 64];
    for i in 0..16 {
        b[i * 4..i * 4 + 4].copy_from_slice(&w[i].to_le_bytes());
    }
    b
}

/// Light cache for one epoch (`mkcache`). Held so verification of many shares in
/// an epoch pays generation once.
pub struct Light {
    pub epoch: u64,
    full_size: usize,
    cache: Vec<[u8; 64]>,
}

/// `mkcache` — build the `n`-row light cache from a 32-byte seed (Keccak-512
/// seed chain + `CACHE_ROUNDS` of RandMemoHash). Shared by `Light::new` and the
/// known-answer tests so both exercise the exact same generator.
fn mkcache(n: usize, seed: &[u8; 32]) -> Vec<[u8; 64]> {
    let mut cache: Vec<[u8; 64]> = Vec::with_capacity(n);
    cache.push(keccak512(seed));
    for i in 1..n {
        cache.push(keccak512(&cache[i - 1]));
    }
    // CACHE_ROUNDS of RandMemoHash.
    for _ in 0..CACHE_ROUNDS {
        for i in 0..n {
            let v = u32::from_le_bytes(cache[i][0..4].try_into().unwrap()) as usize % n;
            let mut x = [0u8; 64];
            for k in 0..64 {
                x[k] = cache[(i + n - 1) % n][k] ^ cache[v][k];
            }
            cache[i] = keccak512(&x);
        }
    }
    cache
}

impl Light {
    pub fn new(epoch: u64) -> Self {
        let n = cache_size(epoch) / HASH_BYTES;
        let cache = mkcache(n, &seedhash(epoch));
        Self { epoch, full_size: full_size(epoch), cache }
    }

    /// Build a `Light` directly from raw cache bytes + the full dataset size, with
    /// no epoch math. Used to cross-check the GPU replay (`src/ethash.rs`): the
    /// exact light cache miniZ uploaded is captured, so CPU hashimoto here uses the
    /// *same* cache the kernels did — making verification independent of the
    /// coin's epoch-length convention (ethash vs ECIP-1099 etchash).
    pub fn from_raw(cache_bytes: &[u8], full_size: usize) -> Self {
        assert!(cache_bytes.len() % HASH_BYTES == 0, "cache not a multiple of 64");
        let cache = cache_bytes
            .chunks_exact(HASH_BYTES)
            .map(|c| {
                let mut r = [0u8; 64];
                r.copy_from_slice(c);
                r
            })
            .collect();
        Self { epoch: 0, full_size, cache }
    }

    /// `calc_dataset_item` — derive dataset row `i` from the cache.
    fn dataset_item(&self, i: u32) -> [u32; 16] {
        let n = self.cache.len() as u32;
        let mut mix = words(&self.cache[(i % n) as usize]);
        mix[0] ^= i;
        mix = words(&keccak512(&bytes(&mix)));
        for j in 0..DATASET_PARENTS {
            let parent = fnv(i ^ j, mix[(j % 16) as usize]) % n;
            let p = words(&self.cache[parent as usize]);
            for k in 0..16 {
                mix[k] = fnv(mix[k], p[k]);
            }
        }
        words(&keccak512(&bytes(&mix)))
    }

    /// `hashimoto_light` — returns (mix_hash[32], result[32]) for a header+nonce.
    pub fn hashimoto(&self, header_hash: &[u8; 32], nonce: u64) -> ([u8; 32], [u8; 32]) {
        let n = (self.full_size / HASH_BYTES) as u32; // dataset rows
        let mut seed_in = [0u8; 40];
        seed_in[..32].copy_from_slice(header_hash);
        seed_in[32..].copy_from_slice(&nonce.to_le_bytes());
        let s = keccak512(&seed_in);
        let sw = words(&s);

        let mut mix = [0u32; 32]; // MIX_BYTES/4
        mix[..16].copy_from_slice(&sw);
        mix[16..].copy_from_slice(&sw);

        let mixhashes = (MIX_BYTES / HASH_BYTES) as u32; // 2
        for i in 0..ACCESSES as u32 {
            let p = fnv(i ^ sw[0], mix[(i as usize) % 32]) % (n / mixhashes) * mixhashes;
            let mut newdata = [0u32; 32];
            for j in 0..mixhashes {
                let item = self.dataset_item(p + j);
                newdata[(j * 16) as usize..(j * 16 + 16) as usize].copy_from_slice(&item);
            }
            for k in 0..32 {
                mix[k] = fnv(mix[k], newdata[k]);
            }
        }

        // compress mix (32 -> 8 words)
        let mut cmix = [0u32; 8];
        for i in (0..32).step_by(4) {
            cmix[i / 4] = fnv(fnv(fnv(mix[i], mix[i + 1]), mix[i + 2]), mix[i + 3]);
        }
        let mut mix_hash = [0u8; 32];
        for i in 0..8 {
            mix_hash[i * 4..i * 4 + 4].copy_from_slice(&cmix[i].to_le_bytes());
        }
        let mut final_in = [0u8; 96];
        final_in[..64].copy_from_slice(&s);
        final_in[64..].copy_from_slice(&mix_hash);
        let result = keccak256(&final_in);
        (mix_hash, result)
    }
}

/// Big-endian 256-bit compare: `result <= target` (the Ethash boundary check).
fn le_or_eq_be(result: &[u8; 32], target: &[u8; 32]) -> bool {
    for i in 0..32 {
        if result[i] != target[i] {
            return result[i] < target[i];
        }
    }
    true
}

/// Verify a candidate share. `light` must be for the job's epoch. Returns true
/// only if the recomputed result meets `target`; if `mix_hash` is provided
/// (from the GPU/submit), it must also match — catching a bad mix.
pub fn verify(
    light: &Light,
    header_hash: &[u8; 32],
    nonce: u64,
    target: &[u8; 32],
    mix_hash: Option<&[u8; 32]>,
) -> bool {
    let (mix, result) = light.hashimoto(header_hash, nonce);
    if let Some(expected) = mix_hash {
        if &mix != expected {
            return false;
        }
    }
    le_or_eq_be(&result, target)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex32(s: &str) -> [u8; 32] {
        let mut o = [0u8; 32];
        for i in 0..32 {
            o[i] = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap();
        }
        o
    }

    /// KAT: Keccak-256("") — the canonical Ethereum empty-string hash. Validates
    /// the whole Keccak core (permutation, padding, rate).
    #[test]
    fn keccak256_empty_kat() {
        assert_eq!(
            keccak256(b""),
            hex32("c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470")
        );
    }

    /// KAT: Keccak-256("abc").
    #[test]
    fn keccak256_abc_kat() {
        assert_eq!(
            keccak256(b"abc"),
            hex32("4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45")
        );
    }

    #[test]
    fn fnv_matches_spec() {
        // fnv(a,b) = a*FNV_PRIME ^ b, mod 2^32
        assert_eq!(fnv(1, 0), FNV_PRIME);
        assert_eq!(fnv(0, 7), 7);
    }

    #[test]
    fn seedhash_epoch_roundtrip() {
        let s = seedhash(3);
        assert_eq!(epoch_from_seedhash(&s), Some(3));
        assert_eq!(seedhash(0), [0u8; 32]);
    }

    /// Cache sizes are prime in HASH_BYTES units and shrink-search is stable.
    #[test]
    fn cache_size_is_prime_units() {
        for e in [0u64, 1, 10] {
            assert!(is_prime(cache_size(e) / HASH_BYTES));
            assert!(is_prime(full_size(e) / MIX_BYTES));
        }
    }

    /// Full-pipeline KAT for `hashimoto` (mkcache → dataset_item → mix → result),
    /// using go-ethereum's canonical `TestHashimoto` vector
    /// (`consensus/ethash/algorithm_test.go`): epoch-0 seed (32 zero bytes), a
    /// 1024-byte / 16-row cache, dataset size 32*1024 bytes, header hash
    /// `c9149cc0…`, nonce 0. This pins the *entire* light-verify path against a
    /// trusted external implementation, not just the Keccak core.
    #[test]
    fn hashimoto_geth_kat() {
        // 16-row cache from the epoch-0 seed (all zeros), and a 32768-byte
        // dataset — the same tiny sizes geth's TestHashimoto uses.
        let light = Light { epoch: 0, full_size: 32 * 1024, cache: mkcache(16, &[0u8; 32]) };

        let header = hex32("c9149cc0386e689d789a1c2f3d5d169a61a6218ed30e74414dc736e442ef3d1f");
        let (mix, result) = light.hashimoto(&header, 0);

        assert_eq!(
            mix,
            hex32("e4073cffaef931d37117cefd9afd27ea0f1cad6a981dd2605c4a1ac97c519800"),
            "mix digest mismatch vs geth TestHashimoto"
        );
        assert_eq!(
            result,
            hex32("d3539235ee2e6f8db665c0a72169f55b7f6c605712330b778ec3944f0eb5a557"),
            "result hash mismatch vs geth TestHashimoto"
        );

        // verify() must accept when result <= target and reject just below it.
        let mut target = result;
        assert!(verify(&light, &header, 0, &target, Some(&mix)));
        // A wrong mix is rejected even when the result meets target.
        assert!(!verify(&light, &header, 0, &target, Some(&[0u8; 32])));
        // Tightening target one ulp below the result rejects the share.
        for b in target.iter_mut().rev() {
            if *b > 0 { *b -= 1; break; } else { *b = 0xff; }
        }
        assert!(!verify(&light, &header, 0, &target, None));
    }
}
