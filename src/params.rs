//! Equihash parameters for the (n=192, k=7) variant used by ZClassic and
//! several other coins.
//!
//! The relationships below follow the Zcash Equihash specification. With
//! n = 192 and k = 7 the puzzle has eight 24-bit "collision blocks", solutions
//! contain 2^k = 128 indices and serialise to 400 bytes.

/// Equihash `n` parameter (hash length in bits).
pub const N: usize = 192;
/// Equihash `k` parameter (number of collision rounds + 1 blocks).
pub const K: usize = 7;

/// Bits that must collide in each round: n / (k + 1) = 24.
pub const COLLISION_BIT_LENGTH: usize = N / (K + 1);
/// Collision length rounded up to whole bytes: 3.
pub const COLLISION_BYTE_LENGTH: usize = (COLLISION_BIT_LENGTH + 7) / 8;
/// Byte length of the expanded, byte-padded hash used by the solver: 24.
pub const HASH_LENGTH: usize = (K + 1) * COLLISION_BYTE_LENGTH;
/// How many n-bit indices a single BLAKE2b output yields: 512 / n = 2.
pub const INDICES_PER_HASH_OUTPUT: usize = 512 / N;
/// BLAKE2b digest length in bytes: (512 / n) * (n / 8) = 48.
pub const HASH_OUTPUT: usize = INDICES_PER_HASH_OUTPUT * (N / 8);
/// Raw n-bit hash length in bytes: 24.
pub const HASH_BYTES: usize = N / 8;

/// Number of indices in a full solution: 2^k = 128.
pub const SOLUTION_INDICES: usize = 1 << K;
/// Serialised solution size in bytes: (collision_bit_length + 1) * 2^k / 8 = 400.
pub const SOLUTION_BYTES: usize = (COLLISION_BIT_LENGTH + 1) * SOLUTION_INDICES / 8;

/// Number of entries generated for the first round: 2^(collision_bit_length + 1).
pub const INIT_SIZE: usize = 1 << (COLLISION_BIT_LENGTH + 1);
/// Number of BLAKE2b invocations needed to fill the first round. (Used by the
/// OpenCL/CUDA gen kernels; the CPU solver derives it implicitly.)
#[allow(dead_code)]
pub const BLAKE_CALLS: usize = INIT_SIZE / INDICES_PER_HASH_OUTPUT;

/// Bytes of the block header that feed the Equihash hash (everything before the
/// variable-length solution): version + prevhash + merkleroot + reserved +
/// time + bits + nonce.
pub const HEADER_LEN: usize = 4 + 32 + 32 + 32 + 4 + 4 + 32;

// For 192,7 the collision length is byte-aligned, so the expanded solver hash
// equals the raw n-bit hash length. The solver relies on this.
const _: () = assert!(HASH_LENGTH == HASH_BYTES);
