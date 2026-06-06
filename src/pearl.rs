//! Pearl (PRL) "pearlhash" Proof-of-Useful-Work backend (feature `pearl`).
//!
//! Native CPU port of the official open-source reference
//! (github.com/pearl-research-labs/pearl, ISC), confirmed against the whitepaper
//! and the alpha-miner GPU SASS. See `pearl-dump/SPEC.md`. GPU acceleration and
//! the Pearl stratum client build on top of this core.
pub mod merkle;
pub mod pearlhash;
pub mod stratum;

/// GPU acceleration (NVRTC int8 GEMM+transcript kernel), feature `pearl-cuda`.
#[cfg(feature = "pearl-cuda")]
pub mod gpu;

/// Approach A: drive alpha-miner's extracted cubin directly, feature `pearl-cuda`.
#[cfg(feature = "pearl-cuda")]
pub mod alpha_kernel;
