//! pearlhash / Pearl (PRL) Proof-of-Useful-Work — native CPU implementation.
//!
//! A faithful port of the official reference (github.com/pearl-research-labs/pearl,
//! ISC license): `zk-pow/src/ffi/mine.rs` + `circuit/pearl_noise.rs` +
//! `api/proof{,_utils}.rs`, using the stock `blake3` crate. Confirmed against the
//! whitepaper (pearlresearch.ai) and the alpha-miner GPU SASS we dumped — see
//! `pearl-dump/SPEC.md`.
//!
//! Pipeline: keyed-BLAKE3 commitment → low-rank int8 noise (A′=A+E, B′=B+F) →
//! tiled int8·int8→int32 matmul → 16×u32 transcript `M[ℓ%16]=(M⋘13)⊕X` →
//! win when `BLAKE3(M, key=sA) ≤ difficulty·h·w·k`.

use primitive_types::U256;
use rand::Rng;

use super::merkle::{MerkleProof, MerkleTree};

// ---- constants (from pearl_program.rs / pearl_noise.rs) ----
pub const JACKPOT_SIZE: usize = 16;
pub const LROT_PER_TILE: u32 = 13;
const BLAKE3_DIGEST_SIZE: usize = 32;
const CHUNK_LEN: usize = 1024; // blake3 chunk
const SIGNAL_MIN: i8 = -64;
const SIGNAL_MAX: i8 = 64;

const NOISE_RANGE: usize = 128;
const IDXS_PER_COL: usize = 2;
const UNIFORM_NOISE_RANGE: usize = NOISE_RANGE / IDXS_PER_COL; // 64
const ZERO_POINT_TRANSLATION: i8 = (UNIFORM_NOISE_RANGE / 2) as i8; // 32
const RANGE_MASK: u8 = (UNIFORM_NOISE_RANGE - 1) as u8; // 63

const fn padded_label(label: [u8; 8]) -> [u8; 32] {
    let mut r = [0u8; 32];
    let mut i = 0;
    while i < 8 {
        r[i] = label[i];
        i += 1;
    }
    r
}
/// BLAKE3 `seed` label for the A-side noise PRNG (goes in message bytes 32..64).
pub const SEED_LABEL_A: [u8; 32] = padded_label(*b"A_tensor");
/// BLAKE3 `seed` label for the B-side noise PRNG.
pub const SEED_LABEL_B: [u8; 32] = padded_label(*b"B_tensor");

/// Keyed (or unkeyed) BLAKE3, matching the reference `blake3_digest`.
pub fn blake3_digest(data: &[u8], key: Option<[u8; 32]>) -> [u8; 32] {
    let mut h = match key {
        Some(k) => blake3::Hasher::new_keyed(&k),
        None => blake3::Hasher::new(),
    };
    h.update(data);
    *h.finalize().as_bytes()
}

fn pad_to_chunk_boundary(data: &[u8]) -> Vec<u8> {
    let mut v = data.to_vec();
    let rem = v.len() % CHUNK_LEN;
    if rem != 0 {
        v.resize(v.len() + (CHUNK_LEN - rem), 0);
    }
    v
}

// ---- block header + mining configuration (wire-exact serialization) ----

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MMAType {
    Int7xInt7ToInt32 = 0,
}

#[derive(Clone, Copy, Debug)]
pub struct IncompleteBlockHeader {
    pub version: u32,
    pub prev_block: [u8; 32],
    pub merkle_root: [u8; 32],
    pub timestamp: u32,
    pub nbits: u32,
}

impl IncompleteBlockHeader {
    pub const SERIALIZED_SIZE: usize = 76;

    /// Parse the 76-byte serialized header (e.g. the `mining.notify` header hex).
    /// Inverse of [`to_bytes`](Self::to_bytes).
    pub fn from_bytes(d: &[u8]) -> Option<Self> {
        if d.len() < 76 {
            return None;
        }
        let mut prev_block: [u8; 32] = d[4..36].try_into().ok()?;
        prev_block.reverse();
        let mut merkle_root: [u8; 32] = d[36..68].try_into().ok()?;
        merkle_root.reverse();
        Some(Self {
            version: u32::from_le_bytes(d[0..4].try_into().ok()?),
            prev_block,
            merkle_root,
            timestamp: u32::from_le_bytes(d[68..72].try_into().ok()?),
            nbits: u32::from_le_bytes(d[72..76].try_into().ok()?),
        })
    }

    /// version LE | prev_block reversed | merkle_root reversed | timestamp LE | nbits LE
    pub fn to_bytes(&self) -> [u8; 76] {
        let mut b = Vec::with_capacity(76);
        b.extend_from_slice(&self.version.to_le_bytes());
        b.extend(self.prev_block.iter().rev().copied());
        b.extend(self.merkle_root.iter().rev().copied());
        b.extend_from_slice(&self.timestamp.to_le_bytes());
        b.extend_from_slice(&self.nbits.to_le_bytes());
        b.try_into().unwrap()
    }
}

/// 3-dimensional periodic partition of matrix rows/cols (the "thread pattern").
#[derive(Clone, Copy, Debug)]
pub struct PeriodicPattern {
    pub shape: [(u32, u32); 3], // (stride, length) per dim
}

impl PeriodicPattern {
    pub const NUM_DIMS: usize = 3;
    /// Size-1 pattern ({0}) — one row/col per tile.
    pub fn single() -> Self {
        Self { shape: [(1, 1); 3] }
    }
    pub fn to_list(&self) -> Vec<u32> {
        let mut res = vec![0u32];
        for &(stride, length) in &self.shape {
            let mut next = Vec::with_capacity(res.len() * length as usize);
            for i in 0..length {
                for &r in &res {
                    next.push(r + i * stride);
                }
            }
            res = next;
        }
        res
    }
    pub fn offset_is_valid(&self, mut offset: u32) -> bool {
        for &(stride, length) in self.shape.iter().rev() {
            offset %= stride * length;
            if offset >= stride {
                return false;
            }
        }
        true
    }
    pub fn period(&self) -> u32 {
        let &(stride, length) = self.shape.last().unwrap();
        stride * length
    }
    pub fn size(&self) -> u32 {
        self.shape.iter().map(|&(_, l)| l).product()
    }
    /// 6-byte serialization: per dim (factor-1, length-1).
    pub fn to_bytes(&self) -> [u8; 6] {
        let mut d = [0u8; 6];
        let mut min_stride = 1u32;
        for (i, &(stride, length)) in self.shape.iter().enumerate() {
            let factor = stride / min_stride;
            d[2 * i] = (factor - 1) as u8;
            d[2 * i + 1] = (length - 1) as u8;
            min_stride = stride * length;
        }
        d
    }

    /// Inverse of [`to_bytes`](Self::to_bytes): reconstruct shape from 6 bytes.
    pub fn from_bytes(d: &[u8; 6]) -> Self {
        let mut shape = [(0u32, 0u32); 3];
        let mut min_stride = 1u32;
        for i in 0..3 {
            let factor = d[2 * i] as u32 + 1;
            let length = d[2 * i + 1] as u32 + 1;
            let stride = factor * min_stride;
            shape[i] = (stride, length);
            min_stride = stride * length;
        }
        Self { shape }
    }

    /// Build a pattern from a sorted index list starting at 0 (port of the
    /// reference `PeriodicPattern::from_list`). Returns None if not periodic.
    pub fn from_list(pattern: &[u32]) -> Option<Self> {
        if pattern.is_empty() || pattern[0] != 0 || !pattern.windows(2).all(|w| w[0] < w[1]) {
            return None;
        }
        let mut p = pattern.to_vec();
        let mut shape_vec: Vec<(u32, u32)> = Vec::new();
        while p.len() > 1 {
            let mut found = false;
            for period in 1..p.len() {
                if p.len() % period == 0 {
                    let s = p[period];
                    if (0..p.len() - period).all(|i| p[i] + s == p[i + period]) {
                        shape_vec.push((s, (p.len() / period) as u32));
                        p.truncate(period);
                        found = true;
                        break;
                    }
                }
            }
            found.then_some(())?;
        }
        shape_vec.reverse();
        let period = shape_vec.last().map_or(1, |&(s, l)| s * l);
        while shape_vec.len() < 3 {
            shape_vec.push((period, 1));
        }
        Some(Self { shape: shape_vec.try_into().ok()? })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MiningConfiguration {
    pub common_dim: u32, // k
    pub rank: u16,       // r
    pub mma_type: MMAType,
    pub rows_pattern: PeriodicPattern,
    pub cols_pattern: PeriodicPattern,
    pub reserved: [u8; 32],
}

impl MiningConfiguration {
    pub const SERIALIZED_SIZE: usize = 52;
    pub fn to_bytes(&self) -> [u8; 52] {
        let mut b = Vec::with_capacity(52);
        b.extend_from_slice(&self.common_dim.to_le_bytes());
        b.extend_from_slice(&self.rank.to_le_bytes());
        b.extend_from_slice(&(self.mma_type as u16).to_le_bytes());
        b.extend_from_slice(&self.rows_pattern.to_bytes());
        b.extend_from_slice(&self.cols_pattern.to_bytes());
        b.extend_from_slice(&self.reserved);
        b.try_into().unwrap()
    }
    pub fn dot_product_length(&self) -> usize {
        let k = self.common_dim as usize;
        let r = self.rank as usize;
        k - k % r
    }
}

// ---- commitment + difficulty ----

fn compute_job_key(header: &IncompleteBlockHeader, config: &MiningConfiguration) -> [u8; 32] {
    let mut d = Vec::with_capacity(128);
    d.extend_from_slice(&header.to_bytes());
    d.extend_from_slice(&config.to_bytes());
    blake3_digest(&d, None)
}

/// Returns (b_noise_seed = sB, a_noise_seed = sA).
fn compute_commitment_hash(job_key: &[u8; 32], a_row_major: &[u8], b_col_major: &[u8]) -> ([u8; 32], [u8; 32]) {
    let hash_a = blake3_digest(a_row_major, Some(*job_key));
    let hash_b = blake3_digest(b_col_major, Some(*job_key));
    let mut bi = [0u8; 64];
    bi[..32].copy_from_slice(job_key);
    bi[32..].copy_from_slice(&hash_b);
    let b_noise_seed = blake3_digest(&bi, None);
    let mut ai = [0u8; 64];
    ai[..32].copy_from_slice(&b_noise_seed);
    ai[32..].copy_from_slice(&hash_a);
    let a_noise_seed = blake3_digest(&ai, None);
    (b_noise_seed, a_noise_seed)
}

pub fn compute_jackpot_hash(jackpot: &[u32; 16], commitment_hash: [u8; 32]) -> [u8; 32] {
    let msg: [u8; 64] = std::array::from_fn(|i| jackpot[i / 4].to_le_bytes()[i % 4]);
    blake3_digest(&msg, Some(commitment_hash))
}

/// Bitcoin-style compact target decode.
pub fn nbits_to_difficulty(nbits: u32) -> U256 {
    let exponent = (nbits >> 24) as usize;
    let mantissa = nbits & 0x00ff_ffff;
    if mantissa == 0 || exponent == 0 || mantissa & 0x0080_0000 != 0 {
        return U256::zero();
    }
    let mut t = U256::from(mantissa);
    if exponent <= 3 {
        t >>= 8 * (3 - exponent);
    } else {
        t <<= 8 * (exponent - 3);
    }
    t
}

/// Pool **share** acceptance bound for stratum difficulty `D`: a candidate tile
/// is a valid share iff `jackpot_hash (LE u256) <= share_bound(D, h, w, k)`.
///
/// Derived empirically by recomputing 5 accepted prl.kryptex.network shares
/// (all at `mining.set_difficulty 30`): they satisfy
/// `hash <= 2^256 / (D · h·w·k · C)` with `C ≈ 32` — the `∝ 1/(D·h·w·k)` form
/// matches the reference block bound, and the weakest accepted share is within
/// ~1% of the `C=32` threshold. We use `C=32` (slightly strict) so every share
/// we submit clears the pool's threshold; re-derive `C` if a new config rejects.
pub fn share_bound(difficulty: u64, h: usize, w: usize, k: usize) -> U256 {
    let denom = U256::from(difficulty) * U256::from(h * w * k) * U256::from(32u64);
    if denom.is_zero() { U256::MAX } else { U256::MAX / denom }
}

/// Block-level acceptance bound (full network difficulty from `nbits`):
/// `hash <= nbits_to_difficulty(nbits) · h·w·k`. Pool mining uses [`share_bound`].
pub fn extract_difficulty_bound(nbits: u32, config: &MiningConfiguration) -> U256 {
    let diff = nbits_to_difficulty(nbits);
    let h = config.rows_pattern.size() as usize;
    let w = config.cols_pattern.size() as usize;
    let factor = U256::from(h * w * config.dot_product_length());
    if factor.is_zero() || diff > U256::MAX / factor {
        U256::MAX
    } else {
        diff * factor
    }
}

// ---- noise generation (port of pearl_noise.rs) ----

fn get_random_hash(index: usize, seed: &[u8; 32], key: &[u8; 32], prepend_index: usize) -> [u8; 32] {
    let mut m = [0u8; 64];
    let v = (1 + index) as i32;
    m[prepend_index * 4..prepend_index * 4 + 4].copy_from_slice(&v.to_le_bytes());
    m[32..64].copy_from_slice(seed);
    blake3_digest(&m, Some(*key))
}

fn generate_uniform_random_matrix(seed: &[u8; 32], key: &[u8; 32], row_indices: &[usize], num_cols: usize) -> Vec<Vec<i8>> {
    row_indices
        .iter()
        .map(|&row| {
            let start = row * num_cols;
            (start / BLAKE3_DIGEST_SIZE..(start + num_cols).div_ceil(BLAKE3_DIGEST_SIZE))
                .flat_map(|block| {
                    get_random_hash(block, seed, key, 0).into_iter().enumerate().filter_map(move |(k, byte)| {
                        let idx = block * BLAKE3_DIGEST_SIZE + k;
                        (idx >= start && idx < start + num_cols).then(|| (byte & RANGE_MASK) as i8 - ZERO_POINT_TRANSLATION)
                    })
                })
                .collect()
        })
        .collect()
}

fn mul_hi_u32(a: u32, b: u32) -> u32 {
    ((a as u64 * b as u64) >> 32) as u32
}

fn generate_permutation_matrix(seed: &[u8; 32], key: &[u8; 32], k: usize, noise_rank: usize) -> Vec<[u32; 2]> {
    const LINES_PER_HASH: usize = BLAKE3_DIGEST_SIZE / 4;
    let rank_mask = (noise_rank - 1) as u32;
    let mut res = vec![[0u32; 2]; k];
    for (i, chunk) in res.chunks_mut(LINES_PER_HASH).enumerate() {
        let rh = get_random_hash(i, seed, key, 1);
        for (j, slot) in chunk.iter_mut().enumerate() {
            let r = u32::from_le_bytes([rh[j * 4], rh[j * 4 + 1], rh[j * 4 + 2], rh[j * 4 + 3]]);
            let first = r & rank_mask;
            let second = first ^ (1 + mul_hi_u32((noise_rank - 1) as u32, r));
            *slot = [first, second];
        }
    }
    res
}

fn matvec_sparse_perm(perm: &[[u32; 2]], vec: &[i8]) -> Vec<i8> {
    perm.iter()
        .map(|&[a, b]| (vec[a as usize] as i32 - vec[b as usize] as i32) as i8)
        .collect()
}

struct Noise {
    a: Vec<Vec<i8>>,   // m×k (selected rows)
    b: Vec<Vec<i8>>,   // n×k (selected cols of Bᵀ)
}

fn compute_noise_for_indices(
    k: usize,
    noise_rank: usize,
    (b_noise_seed, a_noise_seed): ([u8; 32], [u8; 32]),
    a_rows: &[usize],
    b_cols: &[usize],
) -> Noise {
    let e_al = generate_uniform_random_matrix(&SEED_LABEL_A, &a_noise_seed, a_rows, noise_rank);
    let e_ar_t = generate_permutation_matrix(&SEED_LABEL_A, &a_noise_seed, k, noise_rank);
    let e_bl = generate_permutation_matrix(&SEED_LABEL_B, &b_noise_seed, k, noise_rank);
    let e_br_t = generate_uniform_random_matrix(&SEED_LABEL_B, &b_noise_seed, b_cols, noise_rank);
    Noise {
        a: e_al.iter().map(|row| matvec_sparse_perm(&e_ar_t, row)).collect(),
        b: e_br_t.iter().map(|col| matvec_sparse_perm(&e_bl, col)).collect(),
    }
}

// ---- mining ----

fn flatten(matrix: &[Vec<i8>]) -> Vec<u8> {
    matrix.iter().flatten().map(|&x| x as u8).collect()
}

pub fn threads_partition(pattern: &PeriodicPattern, total: usize) -> Vec<Vec<usize>> {
    let period = pattern.period() as usize;
    assert!(total % period == 0, "dimension must be divisible by pattern period");
    let base: Vec<usize> = pattern.to_list().iter().map(|&i| i as usize).collect();
    (0..total)
        .filter(|&i| pattern.offset_is_valid(i as u32))
        .map(|off| base.iter().map(|&d| off + d).collect())
        .collect()
}

/// A winning tile: the matrices + the row/col indices that open it. Enough to
/// rebuild the Merkle proof for submission (TODO: proof serialization).
pub struct Solution {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub rank: usize,
    pub a: Vec<Vec<i8>>,
    pub bt: Vec<Vec<i8>>, // Bᵀ
    pub a_rows: Vec<usize>,
    pub b_cols: Vec<usize>,
    pub jackpot_hash: [u8; 32],
}

/// Noised int8 operands for one mining attempt, shared by the CPU and GPU
/// miners so they cannot drift. `A′ = A + E` and `B′ᵀ = (B + F)ᵀ` both fit int8
/// (A,B∈[-64,64], noise∈[-63,62] ⇒ [-127,126]), stored row-major and flat.
pub struct AttemptOperands {
    pub a_noised: Vec<i8>,      // m×k row-major, A′ = A + E
    pub bt_noised: Vec<i8>,     // n×k row-major, B′ᵀ = (B + F)ᵀ
    pub a_noise_seed: [u8; 32], // sA — the keyed-BLAKE3 jackpot-hash key
}

/// Commitment + noise for an attempt over random `a` (m×k) and `bt` (n×k = Bᵀ),
/// producing the flat int8 noised operands and the jackpot key. This is the work
/// every tile sweep (CPU or GPU) shares; the per-tile GEMM is [`tile_jackpot`].
pub fn prepare_attempt(header: &IncompleteBlockHeader, config: &MiningConfiguration, a: &[Vec<i8>], bt: &[Vec<i8>]) -> AttemptOperands {
    let m = a.len();
    let n = bt.len();
    let k = config.common_dim as usize;
    let rank = config.rank as usize;

    let job_key = compute_job_key(header, config);
    let a_rm = pad_to_chunk_boundary(&flatten(a));
    let b_cm = pad_to_chunk_boundary(&flatten(bt));
    let seeds = compute_commitment_hash(&job_key, &a_rm, &b_cm);
    let a_noise_seed = seeds.1;

    let noise = compute_noise_for_indices(k, rank, seeds, &(0..m).collect::<Vec<_>>(), &(0..n).collect::<Vec<_>>());
    let mut a_noised = vec![0i8; m * k];
    for (row, (ar, nr)) in a.iter().zip(&noise.a).enumerate() {
        for (l, (&x, &e)) in ar.iter().zip(nr).enumerate() {
            a_noised[row * k + l] = (x as i32 + e as i32) as i8;
        }
    }
    // noise.b is indexed [col][l] (per selected column of Bᵀ).
    let mut bt_noised = vec![0i8; n * k];
    for (col, (bc, nc)) in bt.iter().zip(&noise.b).enumerate() {
        for (l, (&x, &e)) in bc.iter().zip(nc).enumerate() {
            bt_noised[col * k + l] = (x as i32 + e as i32) as i8;
        }
    }
    AttemptOperands { a_noised, bt_noised, a_noise_seed }
}

/// The job key = `BLAKE3(header‖config)` (no key). Drives the commitment.
pub fn job_key(header: &IncompleteBlockHeader, config: &MiningConfiguration) -> [u8; 32] {
    compute_job_key(header, config)
}

/// Derive `(b_noise_seed, a_noise_seed)` from the job key and the two matrix
/// commitments `HA`/`HB` (e.g. computed on the GPU). Same as the seed-mixing
/// inside [`attempt_seeds`], but for externally-computed `HA`/`HB`.
pub fn seeds_from_commitment(job_key: &[u8; 32], hash_a: [u8; 32], hash_b: [u8; 32]) -> ([u8; 32], [u8; 32]) {
    let mut bi = [0u8; 64];
    bi[..32].copy_from_slice(job_key);
    bi[32..].copy_from_slice(&hash_b);
    let b_noise_seed = blake3_digest(&bi, None);
    let mut ai = [0u8; 64];
    ai[..32].copy_from_slice(&b_noise_seed);
    ai[32..].copy_from_slice(&hash_a);
    let a_noise_seed = blake3_digest(&ai, None);
    (b_noise_seed, a_noise_seed)
}

/// Commitment seeds for an attempt, without computing the noise: returns
/// `(b_noise_seed, a_noise_seed)` (= sB, sA). `a_noise_seed` is also the jackpot
/// hash key. Used by the GPU noise path, which generates A′/B′ᵀ on-device from
/// these seeds. The CPU [`prepare_attempt`] computes the same seeds internally.
pub fn attempt_seeds(header: &IncompleteBlockHeader, config: &MiningConfiguration, a: &[Vec<i8>], bt: &[Vec<i8>]) -> ([u8; 32], [u8; 32]) {
    let job_key = compute_job_key(header, config);
    let a_rm = pad_to_chunk_boundary(&flatten(a));
    let b_cm = pad_to_chunk_boundary(&flatten(bt));
    compute_commitment_hash(&job_key, &a_rm, &b_cm)
}

/// Cumulative tiled int8 GEMM + transcript fold for one tile, over the flat
/// noised operands from [`prepare_attempt`]. `Cblk` accumulates across rank
/// chunks; each chunk XOR-reduces the tile and folds it into the 16×u32
/// transcript `M[chunk%16] = (M⋘13) ⊕ X`. Matches the GPU `pearl_tile_dp4a`.
pub fn tile_jackpot(a_noised: &[i8], bt_noised: &[i8], a_rows: &[usize], b_cols: &[usize], k: usize, rank: usize) -> [u32; 16] {
    let (th, tw) = (a_rows.len(), b_cols.len());
    let mut tile = vec![0i32; th * tw];
    let mut jackpot = [0u32; 16];
    let mut chunk = 0usize;
    let mut ll = rank;
    while ll <= k {
        for (u, &ai) in a_rows.iter().enumerate() {
            for (v, &bi) in b_cols.iter().enumerate() {
                let mut s = tile[u * tw + v];
                for l in ll - rank..ll {
                    s += a_noised[ai * k + l] as i32 * bt_noised[bi * k + l] as i32;
                }
                tile[u * tw + v] = s;
            }
        }
        let xored = tile.iter().fold(0u32, |acc, &x| acc ^ x as u32);
        jackpot[chunk % JACKPOT_SIZE] = jackpot[chunk % JACKPOT_SIZE].rotate_left(LROT_PER_TILE) ^ xored;
        chunk += 1;
        ll += rank;
    }
    jackpot
}

/// One mining attempt over random A,B. Returns a solution if any tile opens.
pub fn try_mine_one<R: Rng>(
    rng: &mut R,
    m: usize,
    n: usize,
    k: usize,
    header: &IncompleteBlockHeader,
    config: &MiningConfiguration,
    bound: U256,
) -> Option<Solution> {
    let rank = config.rank as usize;
    let a: Vec<Vec<i8>> = (0..m).map(|_| (0..k).map(|_| rng.random_range(SIGNAL_MIN..=SIGNAL_MAX)).collect()).collect();
    let b: Vec<Vec<i8>> = (0..k).map(|_| (0..n).map(|_| rng.random_range(SIGNAL_MIN..=SIGNAL_MAX)).collect()).collect();
    let bt: Vec<Vec<i8>> = (0..n).map(|i| (0..k).map(|j| b[j][i]).collect()).collect();

    let op = prepare_attempt(header, config, &a, &bt);
    for a_rows in threads_partition(&config.rows_pattern, m) {
        for b_cols in threads_partition(&config.cols_pattern, n) {
            let jackpot = tile_jackpot(&op.a_noised, &op.bt_noised, &a_rows, &b_cols, k, rank);
            let jackpot_hash = compute_jackpot_hash(&jackpot, op.a_noise_seed);
            if U256::from_little_endian(&jackpot_hash) <= bound {
                return Some(Solution { m, n, k, rank, a, bt, a_rows, b_cols, jackpot_hash });
            }
        }
    }
    None
}

/// Recompute a solution's jackpot hash and check it meets target — the CPU
/// share verifier (gate before submit). Self-consistent with the miner.
pub fn verify(header: &IncompleteBlockHeader, config: &MiningConfiguration, sol: &Solution, bound: U256) -> bool {
    let (_m, n, k, rank) = (sol.m, sol.n, sol.k, sol.rank);
    let job_key = compute_job_key(header, config);
    let a_rm = pad_to_chunk_boundary(&flatten(&sol.a));
    let b_cm = pad_to_chunk_boundary(&flatten(&sol.bt));
    let seeds = compute_commitment_hash(&job_key, &a_rm, &b_cm);
    let a_noise_seed = seeds.1;
    let noise = compute_noise_for_indices(k, rank, seeds, &sol.a_rows, &sol.b_cols);

    let mut jackpot = [0u32; 16];
    let mut tile = vec![vec![0i32; sol.b_cols.len()]; sol.a_rows.len()];
    for (li, ll) in (rank..=k).step_by(rank).enumerate() {
        for (u, _ai) in sol.a_rows.iter().enumerate() {
            for (v, _bi) in sol.b_cols.iter().enumerate() {
                for l in ll - rank..ll {
                    // A′ = A + E (noise rows are indexed 0..rows for the selected set)
                    let a_v = sol.a[sol.a_rows[u]][l] as i32 + noise.a[u][l] as i32;
                    let b_v = sol.bt[sol.b_cols[v]][l] as i32 + noise.b[v][l] as i32;
                    tile[u][v] += a_v * b_v;
                }
            }
        }
        let xored = tile.iter().flatten().fold(0u32, |acc, &x| acc ^ x as u32);
        let tid = li % JACKPOT_SIZE;
        jackpot[tid] = jackpot[tid].rotate_left(LROT_PER_TILE) ^ xored;
    }
    let jh = compute_jackpot_hash(&jackpot, a_noise_seed);
    let _ = n;
    jh == sol.jackpot_hash && U256::from_little_endian(&jh) <= bound
}

// ---- submission proof (Merkle authentication of the used A-rows / B-cols) ----

/// Merkle proof for one matrix + which rows it authenticates.
pub struct MatrixMerkleProof {
    pub proof: MerkleProof,
    pub row_indices: Vec<usize>,
}

/// The block-opening proof submitted to the pool (pre-ZK "PlainProof").
pub struct PlainProof {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub noise_rank: usize,
    pub a: MatrixMerkleProof,
    pub bt: MatrixMerkleProof,
}

/// Build a matrix's Merkle proof for `row_indices` (port of reference
/// `build_matrix_proof`). The tree root equals `blake3(padded, key=job_key)`,
/// i.e. the commitment HA/HB.
fn build_matrix_proof(matrix: &[Vec<i8>], job_key: &[u8; 32], row_indices: &[usize], num_cols: usize) -> MatrixMerkleProof {
    let padded = super::merkle::pad_to_chunk_boundary(&flatten(matrix));
    let tree = MerkleTree::new(&padded, *job_key);
    let leaf_indices = MerkleTree::compute_leaf_indices_from_rows(row_indices, (matrix.len(), num_cols));
    MatrixMerkleProof {
        proof: tree.get_multileaf_proof(&leaf_indices),
        row_indices: row_indices.to_vec(),
    }
}

/// Assemble the submittable proof for a found [`Solution`].
pub fn build_proof(header: &IncompleteBlockHeader, config: &MiningConfiguration, sol: &Solution) -> PlainProof {
    let job_key = compute_job_key(header, config);
    PlainProof {
        m: sol.m,
        n: sol.n,
        k: sol.k,
        noise_rank: sol.rank,
        a: build_matrix_proof(&sol.a, &job_key, &sol.a_rows, sol.k),
        bt: build_matrix_proof(&sol.bt, &job_key, &sol.b_cols, sol.k),
    }
}

/// Wire encoding of the proof = `bincode(PlainProof)`, matching the reference
/// exactly (the pool accepts `base64(this)`; confirmed from the live capture —
/// `mining.submit [addr, job_id, base64(proof)]`, `share_format:"base64"`).
///
/// bincode layout (little-endian fixint, `usize→u64`, `Vec→u64 count+elems`):
///   PlainProof { m, n, k, noise_rank, a, bt }
///   MatrixMerkleProof { proof, row_indices: Vec<u64> }
///   MerkleProof { leaf_data, leaf_indices, total_leaves, root[32], siblings }
///   leaf_data (serde_chunk_vec = `Vec<&[u8]>`): count, then per leaf `len(1024)+bytes`.
pub fn encode_proof(p: &PlainProof) -> Vec<u8> {
    let mut out = Vec::new();
    let mut u64 = |o: &mut Vec<u8>, v: usize| o.extend_from_slice(&(v as u64).to_le_bytes());
    u64(&mut out, p.m);
    u64(&mut out, p.n);
    u64(&mut out, p.k);
    u64(&mut out, p.noise_rank);
    for mp in [&p.a, &p.bt] {
        // MerkleProof.leaf_data : Vec<&[u8]>  (count, then per-leaf len + bytes)
        u64(&mut out, mp.proof.leaf_data.len());
        for leaf in &mp.proof.leaf_data {
            u64(&mut out, leaf.len()); // 1024
            out.extend_from_slice(leaf);
        }
        // leaf_indices : Vec<u64>
        u64(&mut out, mp.proof.leaf_indices.len());
        for &li in &mp.proof.leaf_indices {
            u64(&mut out, li);
        }
        // total_leaves : u64
        u64(&mut out, mp.proof.total_leaves);
        // root : [u8;32]
        out.extend_from_slice(&mp.proof.root);
        // siblings : Vec<[u8;32]>
        u64(&mut out, mp.proof.siblings.len());
        for sib in &mp.proof.siblings {
            out.extend_from_slice(sib);
        }
        // MatrixMerkleProof.row_indices : Vec<u64>
        u64(&mut out, mp.row_indices.len());
        for &r in &mp.row_indices {
            u64(&mut out, r);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn easy_config(k: u32, rank: u16) -> MiningConfiguration {
        MiningConfiguration {
            common_dim: k,
            rank,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::single(),
            cols_pattern: PeriodicPattern::single(),
            reserved: [0u8; 32],
        }
    }

    #[test]
    fn serialization_sizes() {
        let h = IncompleteBlockHeader { version: 1, prev_block: [3; 32], merkle_root: [4; 32], timestamp: 9, nbits: 0x207fffff };
        assert_eq!(h.to_bytes().len(), 76);
        let c = easy_config(64, 32);
        assert_eq!(c.to_bytes().len(), 52);
        assert_eq!(c.rows_pattern.to_list(), vec![0]); // size-1 pattern
    }

    /// Recompute the jackpot hash from a real ACCEPTED pool proof
    /// (/tmp/pearl_proof.bin) to derive the difficulty-30 share-target formula.
    /// Ignored — needs the captured proof file. Run:
    ///   cargo test --no-default-features --features pearl --bin jackpotminer derive_share_target -- --ignored --nocapture
    #[test]
    #[ignore]
    fn derive_share_target() {
        let hx = |s: &str| -> Vec<u8> { (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2*i..2*i+2], 16).unwrap()).collect() };
        let hdr_hex = match std::fs::read_to_string("/tmp/pearl_headers.txt") {
            Ok(s) => s.lines().find(|l| !l.trim().is_empty()).unwrap_or("").to_string(),
            Err(_) => { eprintln!("no /tmp/pearl_headers.txt; skipping"); return; }
        };
        let header = IncompleteBlockHeader::from_bytes(&hx(&hdr_hex)).unwrap();
        let config = MiningConfiguration {
            common_dim: 4096, rank: 128, mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 32]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&(0..64).collect::<Vec<_>>()).unwrap(),
            reserved: [0u8; 32],
        };
        let job_key = compute_job_key(&header, &config);
        let mut quals: Vec<U256> = vec![];
        let mut hwk = U256::one();
        for idx in 0..16usize {
            let raw = match std::fs::read(format!("/tmp/pearl_proof_{idx}.bin")) {
                Ok(r) => r,
                Err(_) => break,
            };
            let mut o = 0usize;
            let rd = |o: &mut usize| { let v = u64::from_le_bytes(raw[*o..*o + 8].try_into().unwrap()); *o += 8; v as usize };
            let (_m, _n, k, rank) = (rd(&mut o), rd(&mut o), rd(&mut o), rd(&mut o));
            let mut mats = vec![];
            for _ in 0..2 {
                let ld = rd(&mut o);
                let mut chunks = std::collections::HashMap::new();
                let mut idxs = vec![];
                let mut leaves = vec![];
                for _ in 0..ld { let l = rd(&mut o); leaves.push(raw[o..o + l].to_vec()); o += l; }
                let li = rd(&mut o);
                for _ in 0..li { idxs.push(rd(&mut o)); }
                let _t = rd(&mut o);
                let root: [u8; 32] = raw[o..o + 32].try_into().unwrap(); o += 32;
                let sib = rd(&mut o); o += sib * 32;
                let rows = rd(&mut o);
                let mut row_indices = vec![];
                for _ in 0..rows { row_indices.push(rd(&mut o)); }
                for (c, leaf) in idxs.iter().zip(leaves) { chunks.insert(*c, leaf); }
                mats.push((chunks, row_indices, root));
            }
            let cpr = k / 1024;
            let extract = |chunks: &std::collections::HashMap<usize, Vec<u8>>, row: usize| -> Vec<i8> {
                let mut v = Vec::with_capacity(k);
                for c in 0..cpr { v.extend(chunks[&(row * cpr + c)].iter().map(|&b| b as i8)); }
                v
            };
            let (hash_a, hash_b) = (mats[0].2, mats[1].2);
            let mut bi = [0u8; 64]; bi[..32].copy_from_slice(&job_key); bi[32..].copy_from_slice(&hash_b);
            let b_seed = blake3_digest(&bi, None);
            let mut ai = [0u8; 64]; ai[..32].copy_from_slice(&b_seed); ai[32..].copy_from_slice(&hash_a);
            let a_seed = blake3_digest(&ai, None);
            let a_rows = mats[0].1.clone();
            let b_cols = mats[1].1.clone();
            let noise = compute_noise_for_indices(k, rank, (b_seed, a_seed), &a_rows, &b_cols);
            let a_strips: Vec<Vec<i8>> = a_rows.iter().map(|&r| extract(&mats[0].0, r)).collect();
            let b_strips: Vec<Vec<i8>> = b_cols.iter().map(|&c| extract(&mats[1].0, c)).collect();
            let (h, w) = (a_rows.len(), b_cols.len());
            hwk = U256::from(h * w * k);
            let mut jackpot = [0u32; 16];
            let mut tile = vec![vec![0i32; w]; h]; // Cblk ACCUMULATES across chunks
            for (ci, ll) in (rank..=k).step_by(rank).enumerate() {
                for u in 0..h {
                    for vv in 0..w {
                        for l in ll - rank..ll {
                            let a = a_strips[u][l] as i32 + noise.a[u][l] as i32;
                            let b = b_strips[vv][l] as i32 + noise.b[vv][l] as i32;
                            tile[u][vv] = tile[u][vv].wrapping_add(a * b);
                        }
                    }
                }
                let x = tile.iter().flatten().fold(0u32, |acc, &e| acc ^ e as u32);
                jackpot[ci % 16] = jackpot[ci % 16].rotate_left(13) ^ x;
            }
            let hash = U256::from_little_endian(&compute_jackpot_hash(&jackpot, a_seed));
            let q = U256::MAX / hash;
            eprintln!("proof[{idx}] h={h} w={w} hwk={} hash=0x{:x} MAX/hash={}", h * w * k, hash, q);
            quals.push(q);
        }
        if quals.is_empty() { eprintln!("no /tmp/pearl_proof_*.bin"); return; }
        let minq = *quals.iter().min().unwrap();
        eprintln!("--- pool share difficulty was 30 ---");
        eprintln!("min(MAX/hash)            = {}", minq);
        eprintln!("min(MAX/hash) / 30       = {}", minq / U256::from(30u64));
        eprintln!("min(MAX/hash) / hwk      = {}", minq / hwk);
        eprintln!("min(MAX/hash) / (30*hwk) = {}", minq / (U256::from(30u64) * hwk));
    }

    #[test]
    fn pattern_from_list_and_bytes_roundtrip() {
        let p = PeriodicPattern::from_list(&[0, 1, 8, 9]).expect("periodic");
        let mut got = p.to_list();
        got.sort_unstable();
        assert_eq!(got, vec![0, 1, 8, 9]);
        assert_eq!(p.size(), 4);
        let p2 = PeriodicPattern::from_bytes(&p.to_bytes());
        assert_eq!(p2.shape, p.shape, "to_bytes/from_bytes roundtrip");
        assert!(PeriodicPattern::from_list(&[0, 1, 3]).is_none(), "non-periodic rejected");
    }

    /// Mine on a very easy target, then verify the found solution round-trips,
    /// and a tampered nonce/jackpot is rejected.
    #[test]
    fn mine_then_verify_roundtrip() {
        let header = IncompleteBlockHeader { version: 0, prev_block: [1; 32], merkle_root: [2; 32], timestamp: 0x66666666, nbits: 0x207fffff };
        let config = easy_config(64, 32); // k=64, r=32
        let bound = extract_difficulty_bound(header.nbits, &config); // easy nbits
        let mut rng = rand::rng();
        // Easy target => a tile opens within a few attempts.
        let mut sol = None;
        for _ in 0..64 {
            if let Some(s) = try_mine_one(&mut rng, 2, 2, 64, &header, &config, bound) {
                sol = Some(s);
                break;
            }
        }
        let sol = sol.expect("found a solution on easy difficulty");
        assert!(verify(&header, &config, &sol, bound), "found solution must verify");

        // Build the submission proof; its Merkle root must equal the commitment HA.
        let proof = build_proof(&header, &config, &sol);
        let jk = compute_job_key(&header, &config);
        let pa = crate::pearl::merkle::pad_to_chunk_boundary(&flatten(&sol.a));
        assert_eq!(proof.a.proof.root, blake3_digest(&pa, Some(jk)), "merkle root == commitment HA");
        assert!(!encode_proof(&proof).is_empty());

        // Tamper: corrupt the recorded jackpot hash -> verify fails.
        let mut bad = Solution { jackpot_hash: sol.jackpot_hash, ..sol_clone(&sol) };
        bad.jackpot_hash[0] ^= 1;
        assert!(!verify(&header, &config, &bad, bound), "tampered solution must be rejected");
    }

    fn sol_clone(s: &Solution) -> Solution {
        Solution {
            m: s.m, n: s.n, k: s.k, rank: s.rank,
            a: s.a.clone(), bt: s.bt.clone(),
            a_rows: s.a_rows.clone(), b_cols: s.b_cols.clone(),
            jackpot_hash: s.jackpot_hash,
        }
    }
}
