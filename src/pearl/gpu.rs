//! Pearl GPU acceleration (feature `pearl-cuda`) — int8 **tensor-core** (IMMA)
//! kernel via NVRTC (no build-time nvcc; links libcuda + libnvrtc).
//!
//! Each warp computes one 16×16 output tile's jackpot using the WMMA API
//! (`nvcuda::wmma`, s8×s8→s32 — the IMMA tensor-core path). Per rank-chunk it
//! does the int8 GEMM on tensor cores, XOR-reduces the 16×16 int32 tile, and
//! folds it into the 16×u32 transcript `M[tid]=(M⋘13)⊕X`. The host then
//! BLAKE3-hashes M and compares to target.
//!
//! Operands are the noised int8 matrices A′ (tile_h×k) and B′ᵀ (tile_w×k); they
//! fit int8 (A,B∈[-64,64], noise∈[-63,62] ⇒ A′∈[-127,126]). Validated
//! bit-for-bit against the CPU transcript in `tests::gpu_wmma_matches_cpu`.
//!
//! Tiles here are 16×16 (one warp). Wider tile shapes = multiple WMMA blocks per
//! tile, and gathering the row/col *pattern* into a dense tile, are the next step.

use std::ffi::{c_char, c_int, c_uint, c_void, CString};
use std::ptr;

use anyhow::{anyhow, Result};
use primitive_types::U256;

use super::pearlhash::{
    attempt_seeds, job_key, seeds_from_commitment, threads_partition, IncompleteBlockHeader,
    MiningConfiguration, Solution, SEED_LABEL_A, SEED_LABEL_B,
};

/// One warp per 16×16 tile; int8 IMMA via WMMA, per-chunk XOR + transcript.
const KERNEL_SRC: &str = r#"
#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void pearl_tile_wmma(
    const signed char* a, const signed char* b, int k, int rank, unsigned* out)
{
    int tile = blockIdx.x;
    const signed char* A = a + (long)tile * 16 * k;  // 16 x k (A', row-major)
    const signed char* B = b + (long)tile * 16 * k;  // 16 x k (B'^T, row-major)
    __shared__ int ctile[256];
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc;
    unsigned M[16];
    #pragma unroll
    for (int t = 0; t < 16; t++) M[t] = 0u;
    int chunk = 0;
    wmma::fill_fragment(acc, 0);   // Cblk ACCUMULATES across all rank-chunks
    for (int ll = rank; ll <= k; ll += rank) {
        for (int kk = ll - rank; kk < ll; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> bf;
            wmma::load_matrix_sync(af, A + kk, k);  // A'[u][kk+l]
            wmma::load_matrix_sync(bf, B + kk, k);  // col_major over B'^T => B'[kk+l][v]
            wmma::mma_sync(acc, af, bf, acc);
        }
        wmma::store_matrix_sync(ctile, acc, 16, wmma::mem_row_major);
        __syncwarp();
        if ((threadIdx.x & 31) == 0) {
            unsigned x = 0;
            for (int t = 0; t < 256; t++) x ^= (unsigned)ctile[t];
            int tid = chunk & 15;
            M[tid] = ((M[tid] << 13) | (M[tid] >> 19)) ^ x;   // rotate_left(13)
        }
        chunk++;
    }
    if ((threadIdx.x & 31) == 0) {
        for (int t = 0; t < 16; t++) out[(long)tile * 16 + t] = M[t];
    }
}

__device__ __forceinline__ int dp4(int a, int b, int c) {
    int d; asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d;
}

// DP4A int8 path for tall-skinny tiles (the live h=2,w=64 config). One warp per
// tile; each lane owns hw/32 of the h*w accumulators. Operands are noised int8,
// packed 4-per-int (a: tiles*h*(k/4) ints, b: tiles*w*(k/4) ints). Cumulative.
extern "C" __global__ void pearl_tile_dp4a(
    const int* a, const int* b, int h, int w, int k, int rank, unsigned* out)
{
    int tile = blockIdx.x;
    int lane = threadIdx.x & 31;
    int hw = h * w;
    int k4 = k / 4;
    const int* A = a + (long)tile * h * k4;
    const int* B = b + (long)tile * w * k4;
    int acc[8];
    int ne = 0;
    for (int e = lane; e < hw; e += 32) { acc[ne++] = 0; }
    unsigned M[16];
    for (int t = 0; t < 16; t++) M[t] = 0u;
    int chunk = 0;
    for (int ll = rank; ll <= k; ll += rank) {
        int ni = 0;
        for (int e = lane; e < hw; e += 32) {
            int u = e / w, v = e % w;
            const int* ar = A + (long)u * k4;
            const int* br = B + (long)v * k4;
            int s = acc[ni];
            for (int l4 = (ll - rank) / 4; l4 < ll / 4; l4++) s = dp4(ar[l4], br[l4], s);
            acc[ni++] = s;
        }
        unsigned x = 0;
        for (int i = 0; i < ne; i++) x ^= (unsigned)acc[i];
        for (int o = 16; o > 0; o >>= 1) x ^= __shfl_xor_sync(0xffffffffu, x, o);
        if (lane == 0) { int tid = chunk & 15; M[tid] = ((M[tid] << 13) | (M[tid] >> 19)) ^ x; }
        chunk++;
    }
    if (lane == 0) { for (int t = 0; t < 16; t++) out[(long)tile * 16 + t] = M[t]; }
}

// Gather-free DP4A: the FULL noised matrices A'(m×k) and B'^T(n×k) stay resident
// on the GPU; each tile is described by its row/col INDICES (a_rows[tile*h+u],
// b_cols[tile*w+v]) rather than host-gathered strips — the kernel indexes rows
// directly. Identical math to pearl_tile_dp4a, but per-tile host→device traffic
// drops from (h+w)*k bytes to (h+w)*4 bytes. This is the live-scale sweep path.
extern "C" __global__ void pearl_tile_dp4a_idx(
    const int* A, const int* B, const int* a_rows, const int* b_cols,
    int h, int w, int k, int rank, unsigned* out)
{
    int tile = blockIdx.x;
    int lane = threadIdx.x & 31;
    int hw = h * w;
    int k4 = k / 4;
    const int* ar = a_rows + (long)tile * h;
    const int* bc = b_cols + (long)tile * w;
    int acc[8];
    int ne = 0;
    for (int e = lane; e < hw; e += 32) { acc[ne++] = 0; }
    unsigned M[16];
    for (int t = 0; t < 16; t++) M[t] = 0u;
    int chunk = 0;
    for (int ll = rank; ll <= k; ll += rank) {
        int ni = 0;
        for (int e = lane; e < hw; e += 32) {
            int u = e / w, v = e % w;
            const int* arow = A + (long)ar[u] * k4;
            const int* brow = B + (long)bc[v] * k4;
            int s = acc[ni];
            for (int l4 = (ll - rank) / 4; l4 < ll / 4; l4++) s = dp4(arow[l4], brow[l4], s);
            acc[ni++] = s;
        }
        unsigned x = 0;
        for (int i = 0; i < ne; i++) x ^= (unsigned)acc[i];
        for (int o = 16; o > 0; o >>= 1) x ^= __shfl_xor_sync(0xffffffffu, x, o);
        if (lane == 0) { int tid = chunk & 15; M[tid] = ((M[tid] << 13) | (M[tid] >> 19)) ^ x; }
        chunk++;
    }
    if (lane == 0) { for (int t = 0; t < 16; t++) out[(long)tile * 16 + t] = M[t]; }
}

// ---- BLAKE3 keyed single-block compression (one 64-byte block, root) ----
// Matches the `blake3` crate's new_keyed(key).update(msg64).finalize() for a
// 64-byte input: cv = key words, flags = CHUNK_START|CHUNK_END|ROOT|KEYED_HASH.
__device__ __forceinline__ unsigned b3rotr(unsigned x, int n) { return (x >> n) | (x << (32 - n)); }

#define B3G(a,b,c,d,x,y) { \
    v[a]=v[a]+v[b]+(x); v[d]=b3rotr(v[d]^v[a],16); v[c]=v[c]+v[d]; v[b]=b3rotr(v[b]^v[c],12); \
    v[a]=v[a]+v[b]+(y); v[d]=b3rotr(v[d]^v[a],8);  v[c]=v[c]+v[d]; v[b]=b3rotr(v[b]^v[c],7); }

// General compression: cv[8], block m[16] (LE words), 64-bit counter, block_len,
// flags -> chaining value out[8] = state[i]^state[i+8].
__device__ void blake3_compress(const unsigned* cv, const unsigned* msg, unsigned counter_lo, unsigned counter_hi, unsigned block_len, unsigned flags, unsigned* out) {
    unsigned v[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) v[i] = cv[i];
    v[8] = 0x6A09E667u; v[9] = 0xBB67AE85u; v[10] = 0x3C6EF372u; v[11] = 0xA54FF53Au;
    v[12] = counter_lo; v[13] = counter_hi; v[14] = block_len; v[15] = flags;
    unsigned m[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) m[i] = msg[i];
    const int P[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
    #pragma unroll
    for (int r = 0; r < 7; r++) {
        B3G(0, 4, 8, 12, m[0], m[1]);
        B3G(1, 5, 9, 13, m[2], m[3]);
        B3G(2, 6, 10, 14, m[4], m[5]);
        B3G(3, 7, 11, 15, m[6], m[7]);
        B3G(0, 5, 10, 15, m[8], m[9]);
        B3G(1, 6, 11, 12, m[10], m[11]);
        B3G(2, 7, 8, 13, m[12], m[13]);
        B3G(3, 4, 9, 14, m[14], m[15]);
        if (r < 6) {
            unsigned t[16];
            #pragma unroll
            for (int i = 0; i < 16; i++) t[i] = m[P[i]];
            #pragma unroll
            for (int i = 0; i < 16; i++) m[i] = t[i];
        }
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) out[i] = v[i] ^ v[i + 8];
}

// key[8], msg[16] (LE u32 words of the 64-byte block) -> out[8]: keyed single
// 64-byte block, flags = START|END|ROOT|KEYED (0x1B), counter 0.
__device__ void blake3_keyed64(const unsigned* key, const unsigned* msg, unsigned* out) {
    blake3_compress(key, msg, 0u, 0u, 64u, 0x1Bu, out);
}

// Keyed-BLAKE3 chunk chaining value for ONE full 1024-byte chunk (16 blocks),
// `finalize_non_root` style: cv = key; each block compresses with counter =
// chunk index, START on first, END on last, KEYED throughout. data = 256 LE u32.
__device__ void blake3_chunk_cv(const unsigned* key, const unsigned* data256, unsigned chunk_idx, unsigned* out) {
    unsigned cv[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = key[i];
    #pragma unroll
    for (int b = 0; b < 16; b++) {
        unsigned flags = 0x10u; // KEYED_HASH
        if (b == 0) flags |= 0x01u;   // CHUNK_START
        if (b == 15) flags |= 0x02u;  // CHUNK_END
        unsigned tmp[8];
        blake3_compress(cv, data256 + b * 16, chunk_idx, 0u, 64u, flags, tmp);
        #pragma unroll
        for (int i = 0; i < 8; i++) cv[i] = tmp[i];
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) out[i] = cv[i];
}

// One thread per 1024-byte chunk: chunk CV of the matrix `data` (bytes, padded to
// a 1024 multiple) -> cvs[chunk*8 .. +8]. Host merges CVs to the root (hazmat).
extern "C" __global__ void pearl_chunk_cv(const unsigned* key, const unsigned* data, unsigned* cvs, long nchunks) {
    long c = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nchunks) return;
    blake3_chunk_cv(key, data + c * 256, (unsigned)c, cvs + c * 8);
}

// Counter-PRNG operand generation: out[idx] = signal int8 in [-64,64] from
// (nonce, salt, idx). A,B are the miner's free choice, so any reproducible
// generator is valid; on a win the resident buffer is read back for the proof.
extern "C" __global__ void pearl_rng(signed char* out, long total, unsigned long long nonce, unsigned long long salt) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    unsigned long long h = (unsigned long long)idx * 0x9E3779B97F4A7C15ULL + nonce * 0xD1B54A32D192ED03ULL + salt;
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL; h ^= h >> 33;
    out[idx] = (signed char)((int)(h % 129ULL) - 64);
}

// Validation kernel: hash `count` (key,msg) pairs. keys: count*8 u32, msgs:
// count*16 u32, outs: count*8 u32. One thread per pair.
extern "C" __global__ void pearl_blake3_test(const unsigned* keys, const unsigned* msgs, unsigned* outs, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    blake3_keyed64(keys + (long)i * 8, msgs + (long)i * 16, outs + (long)i * 8);
}

// ---- on-GPU low-rank noise generation (port of pearl_noise.rs) ----
// Build a keyed-BLAKE3 message: word[prepend] = index+1, words[8..16] = label.
__device__ __forceinline__ void noise_msg(unsigned* msg, const unsigned* label, unsigned index_plus1, int prepend) {
    #pragma unroll
    for (int i = 0; i < 16; i++) msg[i] = 0u;
    msg[prepend] = index_plus1;
    #pragma unroll
    for (int i = 0; i < 8; i++) msg[8 + i] = label[i];
}

// Uniform signed-6-bit stream: e[b*32 .. b*32+32] from hash block `b`
// (prepend_index=0). One thread per 32-byte block; value = (byte & 63) - 32.
extern "C" __global__ void pearl_uniform(const unsigned* key, const unsigned* label, signed char* e, long n_blocks) {
    long b = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    unsigned msg[16];
    noise_msg(msg, label, (unsigned)(b + 1), 0);
    unsigned h[8];
    blake3_keyed64(key, msg, h);
    #pragma unroll
    for (int w = 0; w < 8; w++) {
        unsigned word = h[w];
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            int val = (int)((word >> (8 * s)) & 63u) - 32;
            e[b * 32 + w * 4 + s] = (signed char)val;
        }
    }
}

// Sparse ±1 permutation: per column l (i=l/8, j=l%8, prepend_index=1),
// first = r & (rank-1), second = first ^ (1 + mulhi(rank-1, r)). One thread per
// hash (8 columns). Requires k % 8 == 0.
extern "C" __global__ void pearl_perm(const unsigned* key, const unsigned* label, int* first, int* second, int k, int rank) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k / 8) return;
    unsigned msg[16];
    noise_msg(msg, label, (unsigned)(i + 1), 1);
    unsigned h[8];
    blake3_keyed64(key, msg, h);
    unsigned rank_mask = (unsigned)(rank - 1);
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        unsigned r = h[j];
        unsigned f = r & rank_mask;
        unsigned mulhi = (unsigned)(((unsigned long long)(unsigned)(rank - 1) * (unsigned long long)r) >> 32);
        first[i * 8 + j] = (int)f;
        second[i * 8 + j] = (int)(f ^ (1u + mulhi));
    }
}

// Combine: out[row*k+l] = sig[row*k+l] + e[row*rank + first[l]] - e[row*rank + second[l]].
// One thread per element (rows*k). e is the per-row uniform stream (rank wide).
extern "C" __global__ void pearl_combine(
    const signed char* sig, const signed char* e, const int* first, const int* second,
    signed char* out, long rows, int k, int rank)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * (long)k) return;
    long row = idx / k;
    int l = (int)(idx % k);
    const signed char* erow = e + row * (long)rank;
    int noise = (int)erow[first[l]] - (int)erow[second[l]];
    out[idx] = (signed char)((int)sig[idx] + noise);
}

// Fused harvest sweep: one block per tile, ONE THREAD PER OUTPUT ELEMENT
// (blockDim = h*w, ≤256). Each thread accumulates its own h×w cell's int8 GEMM
// (cumulative across rank-chunks via dp4a); per chunk the block XOR-reduces the
// cumulative tile (shared mem) and thread 0 folds it into the transcript M. After
// the k-loop thread 0 hashes M with keyed BLAKE3, compares to `bound` (8 LE u32
// limbs, MSB-first), and atomicAdd-emits ONLY winners {tile, hash[0..8]} (capped).
// More parallelism than 1 warp/tile (was ~0.58 TOPS): 4 warps/tile for h=2,w=64.
extern "C" __global__ void pearl_tile_dp4a_idx_harvest(
    const int* A, const int* B, const int* a_rows, const int* b_cols,
    int h, int w, int k, int rank,
    const unsigned* key, const unsigned* bound,
    unsigned* wcount, unsigned* winners, int cap)
{
    int tile = blockIdx.x;
    int e = threadIdx.x;          // one output element per thread
    int hw = h * w;
    int k4 = k / 4;
    const int* ar = a_rows + (long)tile * h;
    const int* bc = b_cols + (long)tile * w;
    int u = e / w, v = e % w;
    int r4 = rank / 4;

    extern __shared__ int smem[];  // dynamic: h*r4 ints (A) + w*r4 ints (B)
    int* As = smem;
    int* Bs = smem + h * r4;
    __shared__ int red[256];       // cumulative C per element, XOR-reduced/chunk
    __shared__ unsigned M[16];
    if (e < 16) M[e] = 0u;

    int acc = 0;
    int chunk = 0;
    for (int c = 0; c < k4; c += r4) {  // c = base packed-int index of this rank chunk
        // Coalesced staging of this chunk's A rows (h×r4) and B rows (w×r4).
        for (int i = e; i < h * r4; i += hw) As[i] = A[(long)ar[i / r4] * k4 + c + (i % r4)];
        for (int i = e; i < w * r4; i += hw) Bs[i] = B[(long)bc[i / r4] * k4 + c + (i % r4)];
        __syncthreads();
        const int* as_ = As + u * r4;
        const int* bs_ = Bs + v * r4;
        for (int ll = 0; ll < r4; ll++) acc = dp4(as_[ll], bs_[ll], acc);
        red[e] = acc;
        __syncthreads();
        // parallel XOR tree reduction over the hw elements (handles non-pow2)
        for (int s = 1; s < hw; s <<= 1) {
            if ((e % (s << 1)) == 0 && e + s < hw) red[e] ^= red[e + s];
            __syncthreads();
        }
        if (e == 0) {
            int tid = chunk & 15;
            M[tid] = ((M[tid] << 13) | (M[tid] >> 19)) ^ (unsigned)red[0];
        }
        __syncthreads();
        chunk++;
    }

    if (e == 0) {
        unsigned hsh[8];
        blake3_keyed64(key, M, hsh);            // M[16] = 64-byte LE message
        bool le = true;
        #pragma unroll
        for (int i = 7; i >= 0; i--) {          // compare from most-significant limb
            if (hsh[i] < bound[i]) { le = true; break; }
            if (hsh[i] > bound[i]) { le = false; break; }
        }
        if (le) {
            unsigned slot = atomicAdd(wcount, 1u);
            if (slot < (unsigned)cap) {
                unsigned* o = winners + (long)slot * 9;
                o[0] = (unsigned)tile;
                #pragma unroll
                for (int i = 0; i < 8; i++) o[1 + i] = hsh[i];
            }
        }
    }
}

// Batched harvest with CROSS-TILE B REUSE. Grid = (ncol, ceil(nrow/G)); each
// block handles ONE col-tile × G row-tiles with blockDim = G*h*w threads (one
// output element per thread). Per rank-chunk the col-tile's B rows (w×r4) are
// staged ONCE in shared memory and reused by all G row-tiles (the dominant
// traffic, since w≫h); A rows for the G tiles (G*h×r4) are staged too. G
// independent transcripts are folded; thread (g,0,0) hashes M[g], compares
// `bound`, and emits winner {global_tile = rtile*ncol + ctile, hash[8]}.
// Dynamic smem layout (4-byte words): Bs[w*r4] As[G*h*r4] red[G*hw] M[G*16].
extern "C" __global__ void pearl_harvest_batched(
    const int* A, const int* B, const int* a_rows, const int* b_cols,
    int nrow, int ncol, int h, int w, int k, int rank, int G,
    const unsigned* key, const unsigned* bound,
    unsigned* wcount, unsigned* winners, int cap)
{
    int ctile = blockIdx.x;
    int rbase = blockIdx.y * G;
    int hw = h * w;
    int e = threadIdx.x;
    int g = e / hw, rem = e % hw, u = rem / w, v = rem % w;
    int rtile = rbase + g;
    int valid = rtile < nrow;
    int k4 = k / 4, r4 = rank / 4;

    extern __shared__ int smem[];
    int* Bs = smem;                          // w*r4
    int* As = Bs + w * r4;                    // G*h*r4
    int* red = As + G * h * r4;               // G*hw
    unsigned* M = (unsigned*)(red + G * hw);  // G*16
    const int* bc = b_cols + (long)ctile * w;

    if (e < G * 16) M[e] = 0u;
    __syncthreads();

    int acc = 0;
    int chunk = 0;
    for (int c = 0; c < k4; c += r4) {
        for (int i = e; i < w * r4; i += blockDim.x) Bs[i] = B[(long)bc[i / r4] * k4 + c + (i % r4)];
        for (int i = e; i < G * h * r4; i += blockDim.x) {
            int gg = i / (h * r4), hr = i % (h * r4), uu = hr / r4, ll = hr % r4;
            int rt = rbase + gg;
            As[i] = (rt < nrow) ? A[(long)a_rows[(long)rt * h + uu] * k4 + c + ll] : 0;
        }
        __syncthreads();
        const int* as_ = As + (long)(g * h + u) * r4;
        const int* bs_ = Bs + (long)v * r4;
        for (int ll = 0; ll < r4; ll++) acc = dp4(as_[ll], bs_[ll], acc);
        red[e] = acc;
        __syncthreads();
        if (rem == 0) {                       // one thread per row-tile in the group
            unsigned x = 0;
            for (int i = 0; i < hw; i++) x ^= (unsigned)red[g * hw + i];
            int tid = chunk & 15;
            M[g * 16 + tid] = ((M[g * 16 + tid] << 13) | (M[g * 16 + tid] >> 19)) ^ x;
        }
        __syncthreads();
        chunk++;
    }

    if (rem == 0 && valid) {
        unsigned hsh[8];
        blake3_keyed64(key, M + g * 16, hsh);
        bool le = true;
        #pragma unroll
        for (int i = 7; i >= 0; i--) {
            if (hsh[i] < bound[i]) { le = true; break; }
            if (hsh[i] > bound[i]) { le = false; break; }
        }
        if (le) {
            unsigned slot = atomicAdd(wcount, 1u);
            if (slot < (unsigned)cap) {
                unsigned* o = winners + (long)slot * 9;
                o[0] = (unsigned)(rtile * ncol + ctile);
                #pragma unroll
                for (int i = 0; i < 8; i++) o[1 + i] = hsh[i];
            }
        }
    }
}

// IMMA TENSOR-CORE batched sweep. Stacks G = 16/h row-tiles into a 16-row s8
// WMMA fragment × one col-tile (w/16 col-blocks = warps). Per rank-chunk the
// 16×rank A slice (gathered, scattered rows) and w×rank B slice (col-tile rows)
// are staged in smem; each warp accumulates its 16×16 col-block C across the
// chunk's rank/16 k-fragments (tensor cores), and C[16×w] (cumulative) is
// XOR-folded per row-tile into G transcripts. Requires G*h==16, w%16==0,
// k%16==0, rank%16==0. A,B are the noised int8 matrices (row-major, stride k).
// smem (bytes): A_sh[16*rank] B_sh[w*rank] (i8) + C_sh[16*w] (i32) + M[G*16] (u32).
extern "C" __global__ void pearl_harvest_wmma(
    const signed char* A, const signed char* B, const int* a_rows, const int* b_cols,
    int nrow, int ncol, int h, int w, int k, int rank, int G,
    const unsigned* key, const unsigned* bound,
    unsigned* wcount, unsigned* winners, int cap)
{
    int ctile = blockIdx.x;
    int rbase = blockIdx.y * G;
    int warp = threadIdx.x >> 5;
    int kf_per_chunk = rank / 16;

    extern __shared__ signed char smem_c[];
    signed char* A_sh = smem_c;                      // 16*rank
    signed char* B_sh = A_sh + 16 * rank;            // w*rank
    int* C_sh = (int*)(B_sh + w * rank);             // 16*w
    unsigned* M = (unsigned*)(C_sh + 16 * w);        // G*16
    const int* bcol = b_cols + (long)ctile * w;

    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
    wmma::fill_fragment(c_frag, 0);
    for (int i = threadIdx.x; i < G * 16; i += blockDim.x) M[i] = 0u;

    int chunk = 0;
    for (int cbase = 0; cbase < k; cbase += rank) {
        for (int i = threadIdx.x; i < 16 * rank; i += blockDim.x) {
            int r = i / rank, kk = i % rank, rt = rbase + r / h, uu = r % h;
            A_sh[i] = (rt < nrow) ? A[(long)a_rows[(long)rt * h + uu] * k + cbase + kk] : 0;
        }
        for (int i = threadIdx.x; i < w * rank; i += blockDim.x) {
            int c = i / rank, kk = i % rank;
            B_sh[i] = B[(long)bcol[c] * k + cbase + kk];
        }
        __syncthreads();
        for (int kf = 0; kf < kf_per_chunk; kf++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b_frag;
            wmma::load_matrix_sync(a_frag, A_sh + kf * 16, rank);
            wmma::load_matrix_sync(b_frag, B_sh + warp * 16 * rank + kf * 16, rank);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(C_sh + warp * 16, c_frag, w, wmma::mem_row_major);
        __syncthreads();
        if (threadIdx.x < G) {
            int g = threadIdx.x;
            unsigned x = 0;
            for (int u = 0; u < h; u++)
                for (int v = 0; v < w; v++)
                    x ^= (unsigned)C_sh[(g * h + u) * w + v];
            int tid = chunk & 15;
            M[g * 16 + tid] = ((M[g * 16 + tid] << 13) | (M[g * 16 + tid] >> 19)) ^ x;
        }
        __syncthreads();
        chunk++;
    }

    if (threadIdx.x < G) {
        int g = threadIdx.x, rtile = rbase + g;
        if (rtile < nrow) {
            unsigned hsh[8];
            blake3_keyed64(key, M + g * 16, hsh);
            bool le = true;
            #pragma unroll
            for (int i = 7; i >= 0; i--) {
                if (hsh[i] < bound[i]) { le = true; break; }
                if (hsh[i] > bound[i]) { le = false; break; }
            }
            if (le) {
                unsigned slot = atomicAdd(wcount, 1u);
                if (slot < (unsigned)cap) {
                    unsigned* o = winners + (long)slot * 9;
                    o[0] = (unsigned)(rtile * ncol + ctile);
                    #pragma unroll
                    for (int i = 0; i < 8; i++) o[1 + i] = hsh[i];
                }
            }
        }
    }
}
"#;

// ---- NVRTC FFI ----
type NvrtcProgram = *mut c_void;
extern "C" {
    fn nvrtcCreateProgram(p: *mut NvrtcProgram, src: *const c_char, name: *const c_char, nh: c_int, h: *const *const c_char, inc: *const *const c_char) -> c_int;
    fn nvrtcCompileProgram(p: NvrtcProgram, n: c_int, opts: *const *const c_char) -> c_int;
    fn nvrtcGetPTXSize(p: NvrtcProgram, sz: *mut usize) -> c_int;
    fn nvrtcGetPTX(p: NvrtcProgram, ptx: *mut c_char) -> c_int;
    fn nvrtcGetProgramLogSize(p: NvrtcProgram, sz: *mut usize) -> c_int;
    fn nvrtcGetProgramLog(p: NvrtcProgram, log: *mut c_char) -> c_int;
    fn nvrtcDestroyProgram(p: *mut NvrtcProgram) -> c_int;
}

// ---- CUDA Driver FFI (subset) ----
type CUresult = c_int;
type CUdevice = c_int;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUdeviceptr = u64;
const CUDA_SUCCESS: CUresult = 0;
extern "C" {
    fn cuInit(f: c_uint) -> CUresult;
    fn cuDeviceGet(d: *mut CUdevice, o: c_int) -> CUresult;
    fn cuCtxCreate_v2(c: *mut CUcontext, f: c_uint, d: CUdevice) -> CUresult;
    fn cuCtxDestroy_v2(c: CUcontext) -> CUresult;
    fn cuModuleLoadData(m: *mut CUmodule, image: *const c_void) -> CUresult;
    fn cuModuleUnload(m: CUmodule) -> CUresult;
    fn cuModuleGetFunction(f: *mut CUfunction, m: CUmodule, name: *const c_char) -> CUresult;
    fn cuMemAlloc_v2(p: *mut CUdeviceptr, n: usize) -> CUresult;
    fn cuMemFree_v2(p: CUdeviceptr) -> CUresult;
    fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, n: usize) -> CUresult;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, n: usize) -> CUresult;
    fn cuMemsetD32_v2(dst: CUdeviceptr, ui: c_uint, n: usize) -> CUresult;
    fn cuLaunchKernel(f: CUfunction, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, sh: c_uint, st: *mut c_void, params: *mut *mut c_void, extra: *mut *mut c_void) -> CUresult;
    fn cuCtxSynchronize() -> CUresult;
}

fn cu(r: CUresult, what: &str) -> Result<()> {
    if r == CUDA_SUCCESS { Ok(()) } else { Err(anyhow!("{what} failed: CUresult {r}")) }
}

/// 32 bytes → 8 little-endian u32 words (BLAKE3 key / label layout).
fn bytes_to_words(b: &[u8; 32]) -> [u32; 8] {
    std::array::from_fn(|i| u32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]))
}

fn compile_ptx() -> Result<Vec<u8>> {
    unsafe {
        let src = CString::new(KERNEL_SRC)?;
        let name = CString::new("pearl.cu")?;
        let mut prog: NvrtcProgram = ptr::null_mut();
        if nvrtcCreateProgram(&mut prog, src.as_ptr(), name.as_ptr(), 0, ptr::null(), ptr::null()) != 0 {
            return Err(anyhow!("nvrtcCreateProgram failed"));
        }
        // compute_80 PTX (s8 WMMA) JITs forward to Ampere..Blackwell; mma.h from the toolkit.
        let arch = CString::new("--gpu-architecture=compute_80")?;
        let inc = CString::new("--include-path=/opt/cuda/include")?;
        let opts = [arch.as_ptr(), inc.as_ptr()];
        let rc = nvrtcCompileProgram(prog, opts.len() as c_int, opts.as_ptr());
        if rc != 0 {
            let mut sz = 0usize;
            nvrtcGetProgramLogSize(prog, &mut sz);
            let mut log = vec![0u8; sz];
            nvrtcGetProgramLog(prog, log.as_mut_ptr() as *mut c_char);
            nvrtcDestroyProgram(&mut prog);
            return Err(anyhow!("nvrtc compile failed: {}", String::from_utf8_lossy(&log)));
        }
        let mut sz = 0usize;
        nvrtcGetPTXSize(prog, &mut sz);
        let mut ptx = vec![0u8; sz];
        nvrtcGetPTX(prog, ptx.as_mut_ptr() as *mut c_char);
        nvrtcDestroyProgram(&mut prog);
        Ok(ptx)
    }
}

/// GPU context holding the compiled int8 IMMA tile kernel.
pub struct PearlGpu {
    ctx: CUcontext,
    module: CUmodule,
    func: CUfunction,          // pearl_tile_wmma (16×16 IMMA)
    func_dp4a: CUfunction,     // pearl_tile_dp4a (tall-skinny h×w, gathered strips)
    func_dp4a_idx: CUfunction, // pearl_tile_dp4a_idx (gather-free, resident matrices)
    func_blake3: CUfunction,   // pearl_blake3_test (keyed single-block validation)
    func_uniform: CUfunction,  // pearl_uniform (signed-6-bit noise stream)
    func_perm: CUfunction,     // pearl_perm (sparse ±1 permutation indices)
    func_combine: CUfunction,  // pearl_combine (a_noised = a + e[first] - e[second])
    func_harvest: CUfunction,  // pearl_tile_dp4a_idx_harvest (fused hash + bound + winner emit)
    func_chunk_cv: CUfunction, // pearl_chunk_cv (keyed-BLAKE3 chunk CVs for the commitment)
    func_rng: CUfunction,      // pearl_rng (counter-PRNG operand generation)
    func_batched: CUfunction,  // pearl_harvest_batched (cross-tile B-reuse sweep)
    func_wmma_harvest: CUfunction, // pearl_harvest_wmma (IMMA tensor-core sweep)
}

unsafe impl Send for PearlGpu {}

impl PearlGpu {
    pub fn new(device_index: usize) -> Result<Self> {
        unsafe {
            cu(cuInit(0), "cuInit")?;
            let mut dev: CUdevice = 0;
            cu(cuDeviceGet(&mut dev, device_index as c_int), "cuDeviceGet")?;
            let mut ctx: CUcontext = ptr::null_mut();
            cu(cuCtxCreate_v2(&mut ctx, 0, dev), "cuCtxCreate")?;
            let ptx = compile_ptx()?;
            let mut module: CUmodule = ptr::null_mut();
            cu(cuModuleLoadData(&mut module, ptx.as_ptr() as *const c_void), "cuModuleLoadData(PTX)")?;
            let name = CString::new("pearl_tile_wmma")?;
            let mut func: CUfunction = ptr::null_mut();
            cu(cuModuleGetFunction(&mut func, module, name.as_ptr()), "cuModuleGetFunction(wmma)")?;
            let name2 = CString::new("pearl_tile_dp4a")?;
            let mut func_dp4a: CUfunction = ptr::null_mut();
            cu(cuModuleGetFunction(&mut func_dp4a, module, name2.as_ptr()), "cuModuleGetFunction(dp4a)")?;
            let name3 = CString::new("pearl_tile_dp4a_idx")?;
            let mut func_dp4a_idx: CUfunction = ptr::null_mut();
            cu(cuModuleGetFunction(&mut func_dp4a_idx, module, name3.as_ptr()), "cuModuleGetFunction(dp4a_idx)")?;
            let name4 = CString::new("pearl_blake3_test")?;
            let mut func_blake3: CUfunction = ptr::null_mut();
            cu(cuModuleGetFunction(&mut func_blake3, module, name4.as_ptr()), "cuModuleGetFunction(blake3)")?;
            let resolve = |sym: &str| -> Result<CUfunction> {
                let cname = CString::new(sym)?;
                let mut f: CUfunction = ptr::null_mut();
                cu(cuModuleGetFunction(&mut f, module, cname.as_ptr()), "cuModuleGetFunction")?;
                Ok(f)
            };
            let func_uniform = resolve("pearl_uniform")?;
            let func_perm = resolve("pearl_perm")?;
            let func_combine = resolve("pearl_combine")?;
            let func_harvest = resolve("pearl_tile_dp4a_idx_harvest")?;
            let func_chunk_cv = resolve("pearl_chunk_cv")?;
            let func_rng = resolve("pearl_rng")?;
            let func_batched = resolve("pearl_harvest_batched")?;
            let func_wmma_harvest = resolve("pearl_harvest_wmma")?;
            Ok(Self { ctx, module, func, func_dp4a, func_dp4a_idx, func_blake3, func_uniform, func_perm, func_combine, func_harvest, func_chunk_cv, func_rng, func_batched, func_wmma_harvest })
        }
    }

    /// Jackpot[16] for each 16×16 tile via tensor cores. `a`/`b` are
    /// `num_tiles` blocks of 16×k int8 (A′ and B′ᵀ). `k`,`rank` multiples of 16.
    pub fn tile_jackpots(&self, a: &[i8], b: &[i8], num_tiles: usize, k: usize, rank: usize) -> Result<Vec<[u32; 16]>> {
        assert_eq!(a.len(), num_tiles * 16 * k);
        assert_eq!(b.len(), num_tiles * 16 * k);
        assert!(k % 16 == 0 && rank % 16 == 0);
        unsafe {
            let mut da: CUdeviceptr = 0;
            let mut db: CUdeviceptr = 0;
            let mut dout: CUdeviceptr = 0;
            cu(cuMemAlloc_v2(&mut da, a.len()), "alloc a")?;
            cu(cuMemAlloc_v2(&mut db, b.len()), "alloc b")?;
            cu(cuMemAlloc_v2(&mut dout, num_tiles * 16 * 4), "alloc out")?;
            cu(cuMemcpyHtoD_v2(da, a.as_ptr() as *const c_void, a.len()), "htod a")?;
            cu(cuMemcpyHtoD_v2(db, b.as_ptr() as *const c_void, b.len()), "htod b")?;

            let (mut pa, mut pb, mut pout) = (da, db, dout);
            let (mut kk, mut rr) = (k as c_int, rank as c_int);
            let mut params: [*mut c_void; 5] = [
                &mut pa as *mut _ as *mut c_void,
                &mut pb as *mut _ as *mut c_void,
                &mut kk as *mut _ as *mut c_void,
                &mut rr as *mut _ as *mut c_void,
                &mut pout as *mut _ as *mut c_void,
            ];
            // one warp (32 threads) per tile
            cu(
                cuLaunchKernel(self.func, num_tiles as c_uint, 1, 1, 32, 1, 1, 0, ptr::null_mut(), params.as_mut_ptr(), ptr::null_mut()),
                "cuLaunchKernel",
            )?;
            cu(cuCtxSynchronize(), "cuCtxSynchronize")?;

            let mut out = vec![[0u32; 16]; num_tiles];
            cu(cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, dout, num_tiles * 16 * 4), "dtoh out")?;
            cuMemFree_v2(da);
            cuMemFree_v2(db);
            cuMemFree_v2(dout);
            Ok(out)
        }
    }

    /// DP4A int8 path: jackpot[16] per `h×w` tile (the live tall-skinny config).
    /// `a` = `tiles` blocks of `h×k` noised int8 (A′), `b` = `tiles` blocks of
    /// `w×k` noised int8 (B′ᵀ). `k`,`rank` multiples of 4; `h*w ≤ 256`.
    pub fn tile_jackpots_dp4a(&self, a: &[i8], b: &[i8], tiles: usize, h: usize, w: usize, k: usize, rank: usize) -> Result<Vec<[u32; 16]>> {
        assert_eq!(a.len(), tiles * h * k);
        assert_eq!(b.len(), tiles * w * k);
        assert!(k % 4 == 0 && rank % 4 == 0 && h * w <= 256);
        unsafe {
            let (mut da, mut db, mut dout): (CUdeviceptr, CUdeviceptr, CUdeviceptr) = (0, 0, 0);
            cu(cuMemAlloc_v2(&mut da, a.len()), "alloc a")?;
            cu(cuMemAlloc_v2(&mut db, b.len()), "alloc b")?;
            cu(cuMemAlloc_v2(&mut dout, tiles * 16 * 4), "alloc out")?;
            cu(cuMemcpyHtoD_v2(da, a.as_ptr() as *const c_void, a.len()), "htod a")?;
            cu(cuMemcpyHtoD_v2(db, b.as_ptr() as *const c_void, b.len()), "htod b")?;
            let (mut pa, mut pb, mut pout) = (da, db, dout);
            let (mut hh, mut ww, mut kk, mut rr) = (h as c_int, w as c_int, k as c_int, rank as c_int);
            let mut params: [*mut c_void; 7] = [
                &mut pa as *mut _ as *mut c_void, &mut pb as *mut _ as *mut c_void,
                &mut hh as *mut _ as *mut c_void, &mut ww as *mut _ as *mut c_void,
                &mut kk as *mut _ as *mut c_void, &mut rr as *mut _ as *mut c_void,
                &mut pout as *mut _ as *mut c_void,
            ];
            cu(cuLaunchKernel(self.func_dp4a, tiles as c_uint, 1, 1, 32, 1, 1, 0, ptr::null_mut(), params.as_mut_ptr(), ptr::null_mut()), "launch dp4a")?;
            cu(cuCtxSynchronize(), "sync")?;
            let mut out = vec![[0u32; 16]; tiles];
            cu(cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, dout, tiles * 16 * 4), "dtoh")?;
            cuMemFree_v2(da); cuMemFree_v2(db); cuMemFree_v2(dout);
            Ok(out)
        }
    }

    /// Validation helper: keyed-BLAKE3 of `count` 64-byte messages on the GPU.
    /// `keys` = count·8 u32 (8 LE words per 32-byte key), `msgs` = count·16 u32
    /// (16 LE words per 64-byte block); returns count·8 u32 digest words.
    pub fn blake3_keyed_many(&self, keys: &[u32], msgs: &[u32], count: usize) -> Result<Vec<u32>> {
        assert_eq!(keys.len(), count * 8);
        assert_eq!(msgs.len(), count * 16);
        unsafe {
            let (mut dk, mut dm, mut dout): (CUdeviceptr, CUdeviceptr, CUdeviceptr) = (0, 0, 0);
            cu(cuMemAlloc_v2(&mut dk, keys.len() * 4), "alloc keys")?;
            cu(cuMemAlloc_v2(&mut dm, msgs.len() * 4), "alloc msgs")?;
            cu(cuMemAlloc_v2(&mut dout, count * 8 * 4), "alloc outs")?;
            cu(cuMemcpyHtoD_v2(dk, keys.as_ptr() as *const c_void, keys.len() * 4), "htod keys")?;
            cu(cuMemcpyHtoD_v2(dm, msgs.as_ptr() as *const c_void, msgs.len() * 4), "htod msgs")?;
            let (mut pk, mut pm, mut pout) = (dk, dm, dout);
            let mut cnt = count as c_int;
            let mut params: [*mut c_void; 4] = [
                &mut pk as *mut _ as *mut c_void,
                &mut pm as *mut _ as *mut c_void,
                &mut pout as *mut _ as *mut c_void,
                &mut cnt as *mut _ as *mut c_void,
            ];
            let blocks = ((count + 255) / 256) as c_uint;
            cu(cuLaunchKernel(self.func_blake3, blocks.max(1), 1, 1, 256, 1, 1, 0, ptr::null_mut(), params.as_mut_ptr(), ptr::null_mut()), "launch blake3")?;
            cu(cuCtxSynchronize(), "sync")?;
            let mut out = vec![0u32; count * 8];
            cu(cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, dout, count * 8 * 4), "dtoh")?;
            cuMemFree_v2(dk);
            cuMemFree_v2(dm);
            cuMemFree_v2(dout);
            Ok(out)
        }
    }

    /// Generate one noised operand on the GPU: `out = sig + E` where `E` is the
    /// low-rank noise `e[row][first[l]] - e[row][second[l]]` keyed by `key`/`label`
    /// (mirrors `pearlhash::prepare_attempt`'s per-side noise). `sig` is the raw
    /// signal int8 (row-major, rows×k); returns the noised int8 (rows×k). Used
    /// standalone for validation and by [`Self::try_mine_one_gpu`]. `rank % 32 == 0`,
    /// `k % 8 == 0`.
    pub fn gpu_noise(&self, sig: &[i8], key: [u8; 32], label: [u8; 32], rows: usize, k: usize, rank: usize) -> Result<Vec<i8>> {
        unsafe {
            let mut dout: CUdeviceptr = 0;
            cu(cuMemAlloc_v2(&mut dout, rows * k), "alloc out")?;
            self.noise_into(dout, sig, key, label, rows, k, rank)?;
            let mut out = vec![0i8; rows * k];
            cu(cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, dout, rows * k), "dtoh out")?;
            cuMemFree_v2(dout);
            Ok(out)
        }
    }

    /// Generate the noised operand into a caller-owned device buffer `dout`
    /// (rows·k int8); allocates + frees its own scratch and synchronizes. The
    /// raw `sig` (rows·k int8) is uploaded here. See [`Self::gpu_noise`].
    #[allow(clippy::too_many_arguments)]
    unsafe fn noise_into(&self, dout: CUdeviceptr, sig: &[i8], key: [u8; 32], label: [u8; 32], rows: usize, k: usize, rank: usize) -> Result<()> {
        let mut dsig: CUdeviceptr = 0;
        cu(cuMemAlloc_v2(&mut dsig, rows * k), "alloc sig")?;
        cu(cuMemcpyHtoD_v2(dsig, sig.as_ptr() as *const c_void, rows * k), "htod sig")?;
        let r = self.noise_into_dev(dout, dsig, key, label, rows, k, rank);
        cuMemFree_v2(dsig);
        r
    }

    /// Like [`Self::noise_into`] but the raw signal `dsig` is already on the
    /// device (rows·k int8) — used by the fully-on-GPU attempt where A/Bᵀ are
    /// generated on the GPU. Allocates + frees its own scratch and syncs.
    #[allow(clippy::too_many_arguments)]
    unsafe fn noise_into_dev(&self, dout: CUdeviceptr, dsig: CUdeviceptr, key: [u8; 32], label: [u8; 32], rows: usize, k: usize, rank: usize) -> Result<()> {
        assert!(rank % 32 == 0 && k % 8 == 0);
        let key_w = bytes_to_words(&key);
        let label_w = bytes_to_words(&label);
        let n_blocks = (rows * rank / 32) as i64;
        let (mut dkey, mut dlabel): (CUdeviceptr, CUdeviceptr) = (0, 0);
        let (mut de, mut dfirst, mut dsecond): (CUdeviceptr, CUdeviceptr, CUdeviceptr) = (0, 0, 0);
        cu(cuMemAlloc_v2(&mut dkey, 32), "alloc key")?;
        cu(cuMemAlloc_v2(&mut dlabel, 32), "alloc label")?;
        cu(cuMemAlloc_v2(&mut de, rows * rank), "alloc e")?;
        cu(cuMemAlloc_v2(&mut dfirst, k * 4), "alloc first")?;
        cu(cuMemAlloc_v2(&mut dsecond, k * 4), "alloc second")?;
        cu(cuMemcpyHtoD_v2(dkey, key_w.as_ptr() as *const c_void, 32), "htod key")?;
        cu(cuMemcpyHtoD_v2(dlabel, label_w.as_ptr() as *const c_void, 32), "htod label")?;

        self.launch_noise(dkey, dlabel, de, dfirst, dsecond, dsig, dout, rows, k, rank, n_blocks)?;
        cu(cuCtxSynchronize(), "sync noise")?;

        for p in [dkey, dlabel, de, dfirst, dsecond] {
            cuMemFree_v2(p);
        }
        Ok(())
    }

    /// Generate `total` signal int8 in [-64,64] into device buffer `out` via the
    /// counter PRNG (no sync). `salt` separates independent streams (A vs Bᵀ).
    unsafe fn gpu_rng_into(&self, out: CUdeviceptr, total: usize, nonce: u64, salt: u64) -> Result<()> {
        let (mut po, mut t, mut nn, mut ss) = (out, total as i64, nonce, salt);
        let mut params: [*mut c_void; 4] = [
            &mut po as *mut _ as *mut c_void,
            &mut t as *mut _ as *mut c_void,
            &mut nn as *mut _ as *mut c_void,
            &mut ss as *mut _ as *mut c_void,
        ];
        let blocks = ((total + 255) / 256).max(1) as c_uint;
        cu(cuLaunchKernel(self.func_rng, blocks, 1, 1, 256, 1, 1, 0, ptr::null_mut(), params.as_mut_ptr(), ptr::null_mut()), "launch rng")
    }

    /// Counter-PRNG signal generation to host (validation). `[-64,64]`, deterministic.
    pub fn gpu_rng(&self, total: usize, nonce: u64, salt: u64) -> Result<Vec<i8>> {
        unsafe {
            let mut d: CUdeviceptr = 0;
            cu(cuMemAlloc_v2(&mut d, total), "alloc rng")?;
            self.gpu_rng_into(d, total, nonce, salt)?;
            cu(cuCtxSynchronize(), "sync rng")?;
            let mut out = vec![0i8; total];
            cu(cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, d, total), "dtoh rng")?;
            cuMemFree_v2(d);
            Ok(out)
        }
    }

    /// Keyed-BLAKE3 commitment of the padded `data` device buffer (`nchunks`
    /// 1024-byte chunks): chunk CVs on the GPU, merged to the root on the CPU
    /// (hazmat). Equals `blake3::keyed_hash(padded_data, key)`. `nchunks >= 2`.
    unsafe fn gpu_commitment_dev(&self, data: CUdeviceptr, nchunks: usize, key: [u8; 32]) -> Result<[u8; 32]> {
        let key_w = bytes_to_words(&key);
        let (mut dkey, mut dcvs): (CUdeviceptr, CUdeviceptr) = (0, 0);
        cu(cuMemAlloc_v2(&mut dkey, 32), "alloc key")?;
        cu(cuMemAlloc_v2(&mut dcvs, nchunks * 8 * 4), "alloc cvs")?;
        cu(cuMemcpyHtoD_v2(dkey, key_w.as_ptr() as *const c_void, 32), "htod key")?;
        let (mut pk, mut pd, mut pc) = (dkey, data, dcvs);
        let mut nc = nchunks as i64;
        let mut params: [*mut c_void; 4] = [
            &mut pk as *mut _ as *mut c_void,
            &mut pd as *mut _ as *mut c_void,
            &mut pc as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
        ];
        let blocks = ((nchunks + 255) / 256).max(1) as c_uint;
        cu(cuLaunchKernel(self.func_chunk_cv, blocks, 1, 1, 256, 1, 1, 0, ptr::null_mut(), params.as_mut_ptr(), ptr::null_mut()), "launch chunk_cv")?;
        cu(cuCtxSynchronize(), "sync chunk_cv")?;
        let mut raw = vec![0u32; nchunks * 8];
        cu(cuMemcpyDtoH_v2(raw.as_mut_ptr() as *mut c_void, dcvs, nchunks * 8 * 4), "dtoh cvs")?;
        cuMemFree_v2(dkey);
        cuMemFree_v2(dcvs);
        let cvs: Vec<[u8; 32]> = (0..nchunks)
            .map(|c| {
                let mut d = [0u8; 32];
                for j in 0..8 {
                    d[j * 4..j * 4 + 4].copy_from_slice(&raw[c * 8 + j].to_le_bytes());
                }
                d
            })
            .collect();
        Ok(super::merkle::root_from_cvs(cvs, &key))
    }

    /// Keyed-BLAKE3 commitment of host `data` (validation/utility): pads to a
    /// chunk boundary, uploads, and runs [`Self::gpu_commitment_dev`].
    pub fn gpu_commitment(&self, data: &[u8], key: [u8; 32]) -> Result<[u8; 32]> {
        let padded_len = super::merkle::padded_chunk_len(data.len());
        let nchunks = padded_len / 1024;
        unsafe {
            let mut d: CUdeviceptr = 0;
            cu(cuMemAlloc_v2(&mut d, padded_len), "alloc data")?;
            cu(cuMemsetD32_v2(d, 0, padded_len / 4), "zero data")?;
            cu(cuMemcpyHtoD_v2(d, data.as_ptr() as *const c_void, data.len()), "htod data")?;
            let root = self.gpu_commitment_dev(d, nchunks, key)?;
            cuMemFree_v2(d);
            Ok(root)
        }
    }

    /// Launch the uniform→perm→combine noise pipeline into `dout` (caller owns
    /// all buffers; no sync). `de`=rows·rank, `dfirst`/`dsecond`=k ints,
    /// `dsig`/`dout`=rows·k int8, `dkey`/`dlabel`=8 u32 each.
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_noise(
        &self,
        dkey: CUdeviceptr,
        dlabel: CUdeviceptr,
        de: CUdeviceptr,
        dfirst: CUdeviceptr,
        dsecond: CUdeviceptr,
        dsig: CUdeviceptr,
        dout: CUdeviceptr,
        rows: usize,
        k: usize,
        rank: usize,
        n_blocks: i64,
    ) -> Result<()> {
        let (mut pk, mut pl, mut pe) = (dkey, dlabel, de);
        let mut pnb = n_blocks;
        let mut up: [*mut c_void; 4] = [
            &mut pk as *mut _ as *mut c_void,
            &mut pl as *mut _ as *mut c_void,
            &mut pe as *mut _ as *mut c_void,
            &mut pnb as *mut _ as *mut c_void,
        ];
        let ublocks = ((n_blocks as usize + 255) / 256).max(1) as c_uint;
        cu(cuLaunchKernel(self.func_uniform, ublocks, 1, 1, 256, 1, 1, 0, ptr::null_mut(), up.as_mut_ptr(), ptr::null_mut()), "launch uniform")?;

        let (mut pk2, mut pl2, mut pf, mut ps) = (dkey, dlabel, dfirst, dsecond);
        let (mut kk, mut rr) = (k as c_int, rank as c_int);
        let mut pp: [*mut c_void; 6] = [
            &mut pk2 as *mut _ as *mut c_void,
            &mut pl2 as *mut _ as *mut c_void,
            &mut pf as *mut _ as *mut c_void,
            &mut ps as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void,
            &mut rr as *mut _ as *mut c_void,
        ];
        let pblocks = (((k / 8) + 255) / 256).max(1) as c_uint;
        cu(cuLaunchKernel(self.func_perm, pblocks, 1, 1, 256, 1, 1, 0, ptr::null_mut(), pp.as_mut_ptr(), ptr::null_mut()), "launch perm")?;

        let (mut psig, mut pe2, mut pf2, mut ps2, mut po) = (dsig, de, dfirst, dsecond, dout);
        let mut prows = rows as i64;
        let (mut kk2, mut rr2) = (k as c_int, rank as c_int);
        let mut cp: [*mut c_void; 8] = [
            &mut psig as *mut _ as *mut c_void,
            &mut pe2 as *mut _ as *mut c_void,
            &mut pf2 as *mut _ as *mut c_void,
            &mut ps2 as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut prows as *mut _ as *mut c_void,
            &mut kk2 as *mut _ as *mut c_void,
            &mut rr2 as *mut _ as *mut c_void,
        ];
        let total = rows * k;
        let cblocks = ((total + 255) / 256).max(1) as c_uint;
        cu(cuLaunchKernel(self.func_combine, cblocks, 1, 1, 256, 1, 1, 0, ptr::null_mut(), cp.as_mut_ptr(), ptr::null_mut()), "launch combine")?;
        Ok(())
    }

    /// One GPU mining attempt, mirroring `pearlhash::try_mine_one`: given random
    /// `a` (m×k) and `bt` (n×k = Bᵀ), generate the noised operands `A′`/`B′ᵀ`
    /// **entirely on the GPU** (resident), then sweep **every** tile with the
    /// fused harvest kernel — each tile does its GEMM+transcript, hashes the
    /// transcript with keyed BLAKE3, compares to `bound` on-device, and emits
    /// only winners. Returns the winning tile (smallest index) if any opens.
    ///
    /// No per-tile output, no host hashing, no jackpot readback: per batch only
    /// the tile indices ((h+w)·4 B/tile) go up and a tiny winners list comes
    /// back. Attempt memory is `(m+n)·k` regardless of tile count. The returned
    /// [`Solution`] carries the original A/Bᵀ + opening indices for `build_proof`.
    #[allow(clippy::too_many_arguments)]
    pub fn try_mine_one_gpu(
        &self,
        header: &IncompleteBlockHeader,
        config: &MiningConfiguration,
        a: Vec<Vec<i8>>,
        bt: Vec<Vec<i8>>,
        m: usize,
        n: usize,
        bound: U256,
        batch_tiles: usize,
    ) -> Result<Option<Solution>> {
        let k = config.common_dim as usize;
        let rank = config.rank as usize;
        let h = config.rows_pattern.size() as usize;
        let w = config.cols_pattern.size() as usize;
        assert!(k % 4 == 0 && rank % 4 == 0 && h * w <= 256);
        let batch_tiles = batch_tiles.max(1);

        // Commitment seeds on CPU (fast); the noised matrices A′/B′ᵀ are then
        // generated entirely on the GPU and kept resident for the sweep.
        let (b_seed, a_seed) = attempt_seeds(header, config, &a, &bt);
        let a_flat: Vec<i8> = a.iter().flatten().copied().collect();
        let bt_flat: Vec<i8> = bt.iter().flatten().copied().collect();
        let row_tiles = threads_partition(&config.rows_pattern, m);
        let col_tiles = threads_partition(&config.cols_pattern, n);

        unsafe {
            // Resident noised matrices, generated on-device from the seeds.
            let mut da: CUdeviceptr = 0;
            let mut db: CUdeviceptr = 0;
            cu(cuMemAlloc_v2(&mut da, m * k), "alloc A'")?;
            cu(cuMemAlloc_v2(&mut db, n * k), "alloc B'")?;
            self.noise_into(da, &a_flat, a_seed, SEED_LABEL_A, m, k, rank)?;
            self.noise_into(db, &bt_flat, b_seed, SEED_LABEL_B, n, k, rank)?;

            let win = self.harvest_sweep(da, db, a_seed, bound, &row_tiles, &col_tiles, h, w, k, rank, batch_tiles)?;
            cuMemFree_v2(da);
            cuMemFree_v2(db);
            Ok(win.map(|(a_rows, b_cols, jackpot_hash)| Solution { m, n, k, rank, a, bt, a_rows, b_cols, jackpot_hash }))
        }
    }

    /// A fully-on-GPU mining attempt (first winner only) — see
    /// [`Self::mine_full_gpu_all`]. Used by tests; mining uses the all-winners form.
    #[allow(clippy::too_many_arguments)]
    pub fn mine_full_gpu(&self, header: &IncompleteBlockHeader, config: &MiningConfiguration, m: usize, n: usize, bound: U256, nonce: u64, batch_tiles: usize) -> Result<Option<Solution>> {
        Ok(self.mine_full_gpu_all(header, config, m, n, bound, nonce, batch_tiles, || true)?.into_iter().next())
    }

    /// A fully-on-GPU mining attempt: A and Bᵀ are GENERATED on the GPU from
    /// `nonce` (counter PRNG); the commitment `HA`/`HB` and noised operands are
    /// computed on the GPU; the IMMA/dp4a harvest sweeps every tile. Returns
    /// **all** winning tiles (each a submittable share) as [`Solution`]s — a
    /// single sweep can open several. The sweep is chunked over row-tiles and
    /// checks `running` between chunks, so a long (live-scale ~50 s) attempt is
    /// interruptible. The CPU only mixes the two 32-byte commitments and (when
    /// any tile opens) reads A/Bᵀ back once. `(m·k)%1024==0`, `(n·k)%1024==0`.
    /// `should_continue` is polled between row-tile chunks; returning `false`
    /// abandons the attempt (shutdown, or newer job arrived).
    #[allow(clippy::too_many_arguments)]
    pub fn mine_full_gpu_all(
        &self,
        header: &IncompleteBlockHeader,
        config: &MiningConfiguration,
        m: usize,
        n: usize,
        bound: U256,
        nonce: u64,
        batch_tiles: usize,
        mut should_continue: impl FnMut() -> bool,
    ) -> Result<Vec<Solution>> {
        let k = config.common_dim as usize;
        let rank = config.rank as usize;
        let h = config.rows_pattern.size() as usize;
        let w = config.cols_pattern.size() as usize;
        assert!(k % 4 == 0 && rank % 4 == 0 && h * w <= 256 && rank % 32 == 0 && k % 8 == 0);
        assert!((m * k) % 1024 == 0 && (n * k) % 1024 == 0, "matrices must be chunk-aligned");
        let batch_tiles = batch_tiles.max(1);
        let jk = job_key(header, config);
        let row_tiles = threads_partition(&config.rows_pattern, m);
        let col_tiles = threads_partition(&config.cols_pattern, n);
        let ncol = col_tiles.len();
        let nrow = row_tiles.len();
        let use_wmma = 16 % h == 0 && w % 16 == 0 && k % 16 == 0 && rank % 16 == 0 && std::env::var("PEARL_NOWMMA").is_err();
        let g = std::env::var("PEARL_G").ok().and_then(|s| s.parse().ok()).unwrap_or((512 / (h * w)).clamp(1, 8));
        // Row-tiles per sweep chunk ⇒ a few M tiles/chunk so `running` is polled
        // every ~1-2 s during a long attempt (no effect when nrow is small).
        let rows_per_chunk = ((4usize << 20) / ncol.max(1)).clamp(1, nrow.max(1));

        unsafe {
            // Raw A, Bᵀ generated on the GPU (kept for the commitment + readback).
            let (mut da_raw, mut db_raw): (CUdeviceptr, CUdeviceptr) = (0, 0);
            cu(cuMemAlloc_v2(&mut da_raw, m * k), "alloc A")?;
            cu(cuMemAlloc_v2(&mut db_raw, n * k), "alloc Bt")?;
            self.gpu_rng_into(da_raw, m * k, nonce, 0)?;
            self.gpu_rng_into(db_raw, n * k, nonce, 1)?;
            cu(cuCtxSynchronize(), "sync rng")?;

            // Commitment on the GPU → seeds (mixed on the CPU, two 64-byte hashes).
            let ha = self.gpu_commitment_dev(da_raw, m * k / 1024, jk)?;
            let hb = self.gpu_commitment_dev(db_raw, n * k / 1024, jk)?;
            let (b_seed, a_seed) = seeds_from_commitment(&jk, ha, hb);

            // Noised operands on the GPU (raw signal already resident).
            let (mut da, mut db): (CUdeviceptr, CUdeviceptr) = (0, 0);
            cu(cuMemAlloc_v2(&mut da, m * k), "alloc A'")?;
            cu(cuMemAlloc_v2(&mut db, n * k), "alloc B'")?;
            self.noise_into_dev(da, da_raw, a_seed, SEED_LABEL_A, m, k, rank)?;
            self.noise_into_dev(db, db_raw, b_seed, SEED_LABEL_B, n, k, rank)?;

            // Chunked sweep over row-tiles; collect winners with GLOBAL indices.
            let mut won: Vec<(usize, [u8; 32])> = Vec::new();
            let mut start = 0usize;
            let mut err: Option<anyhow::Error> = None;
            while start < nrow {
                if !should_continue() {
                    break;
                }
                let end = (start + rows_per_chunk).min(nrow);
                let sub = &row_tiles[start..end];
                let r = if use_wmma {
                    self.harvest_all_wmma(da, db, a_seed, bound, sub, &col_tiles, h, w, k, rank, batch_tiles)
                } else {
                    self.harvest_all(da, db, a_seed, bound, sub, &col_tiles, h, w, k, rank, g, batch_tiles)
                };
                match r {
                    Ok(mut chunk_won) => won.extend(chunk_won.drain(..).map(|(idx, hash)| (idx + start * ncol, hash))),
                    Err(e) => {
                        err = Some(e);
                        break;
                    }
                }
                start = end;
            }

            // Build a Solution per winner (read raw operands back once).
            let sols = if err.is_some() || won.is_empty() {
                Vec::new()
            } else {
                let mut a_flat = vec![0i8; m * k];
                let mut bt_flat = vec![0i8; n * k];
                cu(cuMemcpyDtoH_v2(a_flat.as_mut_ptr() as *mut c_void, da_raw, m * k), "dtoh A")?;
                cu(cuMemcpyDtoH_v2(bt_flat.as_mut_ptr() as *mut c_void, db_raw, n * k), "dtoh Bt")?;
                let a: Vec<Vec<i8>> = a_flat.chunks(k).map(|c| c.to_vec()).collect();
                let bt: Vec<Vec<i8>> = bt_flat.chunks(k).map(|c| c.to_vec()).collect();
                won.into_iter()
                    .map(|(idx, jackpot_hash)| Solution {
                        m, n, k, rank,
                        a: a.clone(),
                        bt: bt.clone(),
                        a_rows: row_tiles[idx / ncol].clone(),
                        b_cols: col_tiles[idx % ncol].clone(),
                        jackpot_hash,
                    })
                    .collect()
            };
            for p in [da_raw, db_raw, da, db] {
                cuMemFree_v2(p);
            }
            match err {
                Some(e) => Err(e),
                None => Ok(sols),
            }
        }
    }

    /// Allocate the per-batch harvest buffers, sweep all tiles (row×col product)
    /// in batches of `batch_tiles` via [`Self::harvest_batch`] against resident
    /// `da`/`db`, and return the first batch's smallest-index winner (rows, cols,
    /// jackpot_hash) ≤ `bound`, or `None`. Buffers are freed before returning.
    #[allow(clippy::too_many_arguments)]
    unsafe fn harvest_sweep(
        &self,
        da: CUdeviceptr,
        db: CUdeviceptr,
        a_seed: [u8; 32],
        bound: U256,
        row_tiles: &[Vec<usize>],
        col_tiles: &[Vec<usize>],
        h: usize,
        w: usize,
        k: usize,
        rank: usize,
        batch_tiles: usize,
    ) -> Result<Option<(Vec<usize>, Vec<usize>, [u8; 32])>> {
        let key_w = bytes_to_words(&a_seed);
        let mut bound_le = [0u8; 32];
        bound.to_little_endian(&mut bound_le);
        let bound_w = bytes_to_words(&bound_le);
        let cap = batch_tiles; // ≥ max winners per batch
        let (mut d_arows, mut d_bcols): (CUdeviceptr, CUdeviceptr) = (0, 0);
        let (mut d_key, mut d_bound, mut d_wcount, mut d_winners): (CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr) = (0, 0, 0, 0);
        cu(cuMemAlloc_v2(&mut d_arows, batch_tiles * h * 4), "alloc a_rows")?;
        cu(cuMemAlloc_v2(&mut d_bcols, batch_tiles * w * 4), "alloc b_cols")?;
        cu(cuMemAlloc_v2(&mut d_key, 32), "alloc key")?;
        cu(cuMemAlloc_v2(&mut d_bound, 32), "alloc bound")?;
        cu(cuMemAlloc_v2(&mut d_wcount, 4), "alloc wcount")?;
        cu(cuMemAlloc_v2(&mut d_winners, cap * 9 * 4), "alloc winners")?;
        cu(cuMemcpyHtoD_v2(d_key, key_w.as_ptr() as *const c_void, 32), "htod key")?;
        cu(cuMemcpyHtoD_v2(d_bound, bound_w.as_ptr() as *const c_void, 32), "htod bound")?;

        let mut win: Option<(Vec<usize>, Vec<usize>, [u8; 32])> = None;
        let mut batch: Vec<(&Vec<usize>, &Vec<usize>)> = Vec::with_capacity(batch_tiles);
        let flush = |this: &Self, batch: &[(&Vec<usize>, &Vec<usize>)]| -> Result<Option<(Vec<usize>, Vec<usize>, [u8; 32])>> {
            let winners = this.harvest_batch(da, db, d_arows, d_bcols, d_key, d_bound, d_wcount, d_winners, cap, batch, h, w, k, rank)?;
            Ok(winners.into_iter().min_by_key(|(i, _)| *i).map(|(i, hash)| {
                let (ar, bc) = batch[i];
                (ar.clone(), bc.clone(), hash)
            }))
        };
        'sweep: for ar in row_tiles {
            for bc in col_tiles {
                batch.push((ar, bc));
                if batch.len() == batch_tiles {
                    win = flush(self, &batch)?;
                    batch.clear();
                    if win.is_some() {
                        break 'sweep;
                    }
                }
            }
        }
        if win.is_none() && !batch.is_empty() {
            win = flush(self, &batch)?;
        }
        for p in [d_arows, d_bcols, d_key, d_bound, d_wcount, d_winners] {
            cuMemFree_v2(p);
        }
        Ok(win)
    }

    /// Cross-tile-reuse sweep of ALL tiles in a SINGLE launch via
    /// `pearl_harvest_batched`: grid (ncol, ⌈nrow/G⌉), block G·h·w. Each block's
    /// col-tile B chunk is staged once and reused by its G row-tiles. Returns all
    /// winners as `(global tile index = rtile·ncol + ctile, jackpot_hash)`, up to
    /// `cap`. `g` is the row-tiles-per-block grouping (G·h·w ≤ 1024).
    #[allow(clippy::too_many_arguments)]
    unsafe fn harvest_all(
        &self,
        da: CUdeviceptr,
        db: CUdeviceptr,
        a_seed: [u8; 32],
        bound: U256,
        row_tiles: &[Vec<usize>],
        col_tiles: &[Vec<usize>],
        h: usize,
        w: usize,
        k: usize,
        rank: usize,
        g: usize,
        cap: usize,
    ) -> Result<Vec<(usize, [u8; 32])>> {
        let (nrow, ncol) = (row_tiles.len(), col_tiles.len());
        let r4 = rank / 4;
        let a_rows_flat: Vec<i32> = row_tiles.iter().flatten().map(|&r| r as i32).collect();
        let b_cols_flat: Vec<i32> = col_tiles.iter().flatten().map(|&c| c as i32).collect();
        let key_w = bytes_to_words(&a_seed);
        let mut bound_le = [0u8; 32];
        bound.to_little_endian(&mut bound_le);
        let bound_w = bytes_to_words(&bound_le);

        let (mut d_ar, mut d_bc, mut d_key, mut d_bound, mut d_wc, mut d_win): (CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr) = (0, 0, 0, 0, 0, 0);
        cu(cuMemAlloc_v2(&mut d_ar, a_rows_flat.len() * 4), "alloc a_rows")?;
        cu(cuMemAlloc_v2(&mut d_bc, b_cols_flat.len() * 4), "alloc b_cols")?;
        cu(cuMemAlloc_v2(&mut d_key, 32), "alloc key")?;
        cu(cuMemAlloc_v2(&mut d_bound, 32), "alloc bound")?;
        cu(cuMemAlloc_v2(&mut d_wc, 4), "alloc wcount")?;
        cu(cuMemAlloc_v2(&mut d_win, cap * 9 * 4), "alloc winners")?;
        cu(cuMemcpyHtoD_v2(d_ar, a_rows_flat.as_ptr() as *const c_void, a_rows_flat.len() * 4), "htod a_rows")?;
        cu(cuMemcpyHtoD_v2(d_bc, b_cols_flat.as_ptr() as *const c_void, b_cols_flat.len() * 4), "htod b_cols")?;
        cu(cuMemcpyHtoD_v2(d_key, key_w.as_ptr() as *const c_void, 32), "htod key")?;
        cu(cuMemcpyHtoD_v2(d_bound, bound_w.as_ptr() as *const c_void, 32), "htod bound")?;
        cu(cuMemsetD32_v2(d_wc, 0, 1), "memset wcount")?;

        let (mut pa, mut pb, mut par, mut pbc) = (da, db, d_ar, d_bc);
        let (mut nr, mut nc, mut hh, mut ww, mut kk, mut rr, mut gg) = (nrow as c_int, ncol as c_int, h as c_int, w as c_int, k as c_int, rank as c_int, g as c_int);
        let (mut pkey, mut pbound, mut pwc, mut pwin) = (d_key, d_bound, d_wc, d_win);
        let mut pcap = cap as c_int;
        let mut params: [*mut c_void; 16] = [
            &mut pa as *mut _ as *mut c_void, &mut pb as *mut _ as *mut c_void,
            &mut par as *mut _ as *mut c_void, &mut pbc as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void, &mut nc as *mut _ as *mut c_void,
            &mut hh as *mut _ as *mut c_void, &mut ww as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void, &mut rr as *mut _ as *mut c_void,
            &mut gg as *mut _ as *mut c_void,
            &mut pkey as *mut _ as *mut c_void, &mut pbound as *mut _ as *mut c_void,
            &mut pwc as *mut _ as *mut c_void, &mut pwin as *mut _ as *mut c_void,
            &mut pcap as *mut _ as *mut c_void,
        ];
        let grid_y = ((nrow + g - 1) / g) as c_uint;
        let block = (g * h * w) as c_uint;
        let smem = ((w * r4 + g * h * r4 + g * h * w + g * 16) * 4) as c_uint;
        cu(cuLaunchKernel(self.func_batched, ncol as c_uint, grid_y, 1, block, 1, 1, smem, ptr::null_mut(), params.as_mut_ptr(), ptr::null_mut()), "launch batched")?;
        cu(cuCtxSynchronize(), "sync batched")?;

        let mut wc: u32 = 0;
        cu(cuMemcpyDtoH_v2(&mut wc as *mut _ as *mut c_void, d_wc, 4), "dtoh wcount")?;
        let nw = (wc as usize).min(cap);
        let mut winners = Vec::new();
        if nw > 0 {
            let mut raw = vec![0u32; nw * 9];
            cu(cuMemcpyDtoH_v2(raw.as_mut_ptr() as *mut c_void, d_win, nw * 9 * 4), "dtoh winners")?;
            for i in 0..nw {
                let mut hash = [0u8; 32];
                for j in 0..8 {
                    hash[j * 4..j * 4 + 4].copy_from_slice(&raw[i * 9 + 1 + j].to_le_bytes());
                }
                winners.push((raw[i * 9] as usize, hash));
            }
        }
        for p in [d_ar, d_bc, d_key, d_bound, d_wc, d_win] {
            cuMemFree_v2(p);
        }
        Ok(winners)
    }

    /// IMMA tensor-core sweep of ALL tiles in a single launch via
    /// `pearl_harvest_wmma`. Stacks `G = 16/h` row-tiles into a 16-row s8 WMMA
    /// fragment. `da`/`db` are the noised int8 matrices (row-major, stride k).
    /// Requires `16 % h == 0`, `w % 16 == 0`, `k % 16 == 0`, `rank % 16 == 0`.
    /// Returns winners `(global tile index, jackpot_hash)` up to `cap`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn harvest_all_wmma(
        &self,
        da: CUdeviceptr,
        db: CUdeviceptr,
        a_seed: [u8; 32],
        bound: U256,
        row_tiles: &[Vec<usize>],
        col_tiles: &[Vec<usize>],
        h: usize,
        w: usize,
        k: usize,
        rank: usize,
        cap: usize,
    ) -> Result<Vec<(usize, [u8; 32])>> {
        assert!(16 % h == 0 && w % 16 == 0 && k % 16 == 0 && rank % 16 == 0);
        let g = 16 / h;
        let (nrow, ncol) = (row_tiles.len(), col_tiles.len());
        let a_rows_flat: Vec<i32> = row_tiles.iter().flatten().map(|&r| r as i32).collect();
        let b_cols_flat: Vec<i32> = col_tiles.iter().flatten().map(|&c| c as i32).collect();
        let key_w = bytes_to_words(&a_seed);
        let mut bound_le = [0u8; 32];
        bound.to_little_endian(&mut bound_le);
        let bound_w = bytes_to_words(&bound_le);

        let (mut d_ar, mut d_bc, mut d_key, mut d_bound, mut d_wc, mut d_win): (CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr) = (0, 0, 0, 0, 0, 0);
        cu(cuMemAlloc_v2(&mut d_ar, a_rows_flat.len() * 4), "alloc a_rows")?;
        cu(cuMemAlloc_v2(&mut d_bc, b_cols_flat.len() * 4), "alloc b_cols")?;
        cu(cuMemAlloc_v2(&mut d_key, 32), "alloc key")?;
        cu(cuMemAlloc_v2(&mut d_bound, 32), "alloc bound")?;
        cu(cuMemAlloc_v2(&mut d_wc, 4), "alloc wcount")?;
        cu(cuMemAlloc_v2(&mut d_win, cap * 9 * 4), "alloc winners")?;
        cu(cuMemcpyHtoD_v2(d_ar, a_rows_flat.as_ptr() as *const c_void, a_rows_flat.len() * 4), "htod a_rows")?;
        cu(cuMemcpyHtoD_v2(d_bc, b_cols_flat.as_ptr() as *const c_void, b_cols_flat.len() * 4), "htod b_cols")?;
        cu(cuMemcpyHtoD_v2(d_key, key_w.as_ptr() as *const c_void, 32), "htod key")?;
        cu(cuMemcpyHtoD_v2(d_bound, bound_w.as_ptr() as *const c_void, 32), "htod bound")?;
        cu(cuMemsetD32_v2(d_wc, 0, 1), "memset wcount")?;

        let (mut pa, mut pb, mut par, mut pbc) = (da, db, d_ar, d_bc);
        let (mut nr, mut nc, mut hh, mut ww, mut kk, mut rr, mut gg) = (nrow as c_int, ncol as c_int, h as c_int, w as c_int, k as c_int, rank as c_int, g as c_int);
        let (mut pkey, mut pbound, mut pwc, mut pwin) = (d_key, d_bound, d_wc, d_win);
        let mut pcap = cap as c_int;
        let mut params: [*mut c_void; 16] = [
            &mut pa as *mut _ as *mut c_void, &mut pb as *mut _ as *mut c_void,
            &mut par as *mut _ as *mut c_void, &mut pbc as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void, &mut nc as *mut _ as *mut c_void,
            &mut hh as *mut _ as *mut c_void, &mut ww as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void, &mut rr as *mut _ as *mut c_void,
            &mut gg as *mut _ as *mut c_void,
            &mut pkey as *mut _ as *mut c_void, &mut pbound as *mut _ as *mut c_void,
            &mut pwc as *mut _ as *mut c_void, &mut pwin as *mut _ as *mut c_void,
            &mut pcap as *mut _ as *mut c_void,
        ];
        let grid_y = ((nrow + g - 1) / g) as c_uint;
        let block = ((w / 16) * 32) as c_uint;
        let smem = (16 * rank + w * rank + 16 * w * 4 + g * 16 * 4) as c_uint;
        cu(cuLaunchKernel(self.func_wmma_harvest, ncol as c_uint, grid_y, 1, block, 1, 1, smem, ptr::null_mut(), params.as_mut_ptr(), ptr::null_mut()), "launch wmma")?;
        cu(cuCtxSynchronize(), "sync wmma")?;

        let mut wc: u32 = 0;
        cu(cuMemcpyDtoH_v2(&mut wc as *mut _ as *mut c_void, d_wc, 4), "dtoh wcount")?;
        let nw = (wc as usize).min(cap);
        let mut winners = Vec::new();
        if nw > 0 {
            let mut raw = vec![0u32; nw * 9];
            cu(cuMemcpyDtoH_v2(raw.as_mut_ptr() as *mut c_void, d_win, nw * 9 * 4), "dtoh winners")?;
            for i in 0..nw {
                let mut hash = [0u8; 32];
                for j in 0..8 {
                    hash[j * 4..j * 4 + 4].copy_from_slice(&raw[i * 9 + 1 + j].to_le_bytes());
                }
                winners.push((raw[i * 9] as usize, hash));
            }
        }
        for p in [d_ar, d_bc, d_key, d_bound, d_wc, d_win] {
            cuMemFree_v2(p);
        }
        Ok(winners)
    }

    /// Upload one batch's tile indices, run the fused harvest kernel against the
    /// resident `da`/`db`, and return its winners as `(within-batch tile index,
    /// jackpot_hash)`. `wcount`/`winners` are reset+filled on-device each call;
    /// `cap` caps stored winners (sized ≥ tiles, so never lost). No readback of
    /// per-tile jackpots — only the winners list (usually empty) comes back.
    #[allow(clippy::too_many_arguments)]
    unsafe fn harvest_batch(
        &self,
        da: CUdeviceptr,
        db: CUdeviceptr,
        d_arows: CUdeviceptr,
        d_bcols: CUdeviceptr,
        d_key: CUdeviceptr,
        d_bound: CUdeviceptr,
        d_wcount: CUdeviceptr,
        d_winners: CUdeviceptr,
        cap: usize,
        batch: &[(&Vec<usize>, &Vec<usize>)],
        h: usize,
        w: usize,
        k: usize,
        rank: usize,
    ) -> Result<Vec<(usize, [u8; 32])>> {
        let tiles = batch.len();
        let mut a_idx: Vec<i32> = Vec::with_capacity(tiles * h);
        let mut b_idx: Vec<i32> = Vec::with_capacity(tiles * w);
        for (ar, bc) in batch {
            a_idx.extend(ar.iter().map(|&r| r as i32));
            b_idx.extend(bc.iter().map(|&c| c as i32));
        }
        cu(cuMemcpyHtoD_v2(d_arows, a_idx.as_ptr() as *const c_void, a_idx.len() * 4), "htod a_rows")?;
        cu(cuMemcpyHtoD_v2(d_bcols, b_idx.as_ptr() as *const c_void, b_idx.len() * 4), "htod b_cols")?;
        cu(cuMemsetD32_v2(d_wcount, 0, 1), "memset wcount")?;

        let (mut pa, mut pb, mut par, mut pbc) = (da, db, d_arows, d_bcols);
        let (mut hh, mut ww, mut kk, mut rr) = (h as c_int, w as c_int, k as c_int, rank as c_int);
        let (mut pkey, mut pbound, mut pwc, mut pwin) = (d_key, d_bound, d_wcount, d_winners);
        let mut pcap = cap as c_int;
        let mut params: [*mut c_void; 13] = [
            &mut pa as *mut _ as *mut c_void,
            &mut pb as *mut _ as *mut c_void,
            &mut par as *mut _ as *mut c_void,
            &mut pbc as *mut _ as *mut c_void,
            &mut hh as *mut _ as *mut c_void,
            &mut ww as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void,
            &mut rr as *mut _ as *mut c_void,
            &mut pkey as *mut _ as *mut c_void,
            &mut pbound as *mut _ as *mut c_void,
            &mut pwc as *mut _ as *mut c_void,
            &mut pwin as *mut _ as *mut c_void,
            &mut pcap as *mut _ as *mut c_void,
        ];
        let smem = ((h + w) * (rank / 4) * 4) as c_uint; // As + Bs (ints)
        cu(cuLaunchKernel(self.func_harvest, tiles as c_uint, 1, 1, (h * w) as c_uint, 1, 1, smem, ptr::null_mut(), params.as_mut_ptr(), ptr::null_mut()), "launch harvest")?;
        cu(cuCtxSynchronize(), "sync")?;

        let mut wc: u32 = 0;
        cu(cuMemcpyDtoH_v2(&mut wc as *mut _ as *mut c_void, d_wcount, 4), "dtoh wcount")?;
        let nw = (wc as usize).min(cap);
        if nw == 0 {
            return Ok(Vec::new());
        }
        let mut raw = vec![0u32; nw * 9];
        cu(cuMemcpyDtoH_v2(raw.as_mut_ptr() as *mut c_void, d_winners, nw * 9 * 4), "dtoh winners")?;
        let winners = (0..nw)
            .map(|i| {
                let tile_idx = raw[i * 9] as usize;
                let mut hash = [0u8; 32];
                for j in 0..8 {
                    hash[j * 4..j * 4 + 4].copy_from_slice(&raw[i * 9 + 1 + j].to_le_bytes());
                }
                (tile_idx, hash)
            })
            .collect();
        Ok(winners)
    }

    /// Upload one batch's tile indices, launch the gather-free indexed kernel
    /// against the resident `da`/`db` matrices, and return the first tile
    /// (rows, cols, jackpot_hash) ≤ `bound`. Reads back per-tile jackpots and
    /// hashes on the CPU — the pre-harvest path, retained for validation
    /// (`gpu_indexed_sweep_matches_cpu`).
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn sweep_idx_batch(
        &self,
        da: CUdeviceptr,
        db: CUdeviceptr,
        d_arows: CUdeviceptr,
        d_bcols: CUdeviceptr,
        dout: CUdeviceptr,
        batch: &[(&Vec<usize>, &Vec<usize>)],
        h: usize,
        w: usize,
        k: usize,
        rank: usize,
        seed: [u8; 32],
        bound: U256,
    ) -> Result<Option<(Vec<usize>, Vec<usize>, [u8; 32])>> {
        let tiles = batch.len();
        let mut a_idx: Vec<i32> = Vec::with_capacity(tiles * h);
        let mut b_idx: Vec<i32> = Vec::with_capacity(tiles * w);
        for (ar, bc) in batch {
            a_idx.extend(ar.iter().map(|&r| r as i32));
            b_idx.extend(bc.iter().map(|&c| c as i32));
        }
        cu(cuMemcpyHtoD_v2(d_arows, a_idx.as_ptr() as *const c_void, a_idx.len() * 4), "htod a_rows")?;
        cu(cuMemcpyHtoD_v2(d_bcols, b_idx.as_ptr() as *const c_void, b_idx.len() * 4), "htod b_cols")?;

        let (mut pa, mut pb, mut par, mut pbc, mut pout) = (da, db, d_arows, d_bcols, dout);
        let (mut hh, mut ww, mut kk, mut rr) = (h as c_int, w as c_int, k as c_int, rank as c_int);
        let mut params: [*mut c_void; 9] = [
            &mut pa as *mut _ as *mut c_void,
            &mut pb as *mut _ as *mut c_void,
            &mut par as *mut _ as *mut c_void,
            &mut pbc as *mut _ as *mut c_void,
            &mut hh as *mut _ as *mut c_void,
            &mut ww as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void,
            &mut rr as *mut _ as *mut c_void,
            &mut pout as *mut _ as *mut c_void,
        ];
        cu(cuLaunchKernel(self.func_dp4a_idx, tiles as c_uint, 1, 1, 32, 1, 1, 0, ptr::null_mut(), params.as_mut_ptr(), ptr::null_mut()), "launch dp4a_idx")?;
        cu(cuCtxSynchronize(), "sync")?;
        let mut out = vec![[0u32; 16]; tiles];
        cu(cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, dout, tiles * 16 * 4), "dtoh")?;

        use super::pearlhash::compute_jackpot_hash;
        for (i, jp) in out.iter().enumerate() {
            let jh = compute_jackpot_hash(jp, seed);
            if U256::from_little_endian(&jh) <= bound {
                let (ar, bc) = batch[i];
                return Ok(Some((ar.clone(), bc.clone(), jh)));
            }
        }
        Ok(None)
    }
}

impl Drop for PearlGpu {
    fn drop(&mut self) {
        unsafe {
            cuModuleUnload(self.module);
            cuCtxDestroy_v2(self.ctx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pearl::pearlhash::{blake3_digest, prepare_attempt, tile_jackpot, verify, MMAType, PeriodicPattern};

    /// The GPU keyed-BLAKE3 single-block primitive must match the `blake3` crate
    /// (via `blake3_digest`) for random 32-byte key / 64-byte message pairs.
    /// Ignored: needs a GPU.
    #[test]
    #[ignore]
    fn gpu_blake3_matches_cpu() {
        let count = 1000usize;
        let mut keys_b = vec![0u8; count * 32];
        let mut msgs_b = vec![0u8; count * 64];
        // deterministic pseudo-random bytes
        let mut x = 0x1234_5678_9abc_def0u64;
        for b in keys_b.iter_mut().chain(msgs_b.iter_mut()) {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            *b = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 56) as u8;
        }
        let to_words = |bytes: &[u8]| -> Vec<u32> {
            bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        };
        let keys_w = to_words(&keys_b);
        let msgs_w = to_words(&msgs_b);

        let gpu = PearlGpu::new(0).expect("init GPU 0");
        let got = gpu.blake3_keyed_many(&keys_w, &msgs_w, count).expect("gpu blake3");
        for i in 0..count {
            let key: [u8; 32] = keys_b[i * 32..i * 32 + 32].try_into().unwrap();
            let want = blake3_digest(&msgs_b[i * 64..i * 64 + 64], Some(key));
            let got_bytes: Vec<u8> = got[i * 8..i * 8 + 8].iter().flat_map(|w| w.to_le_bytes()).collect();
            assert_eq!(got_bytes, want, "blake3 mismatch at pair {i}");
        }
        eprintln!("GPU keyed-BLAKE3 single-block matches CPU for {count} pairs");
    }

    /// Deterministic signed-int8 fill in the signal range [-64, 64].
    fn mk_signal(seed: u64, n: usize) -> Vec<i8> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                ((x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 56) as i64 % 65 * 2 - 64) as i8
            })
            .collect()
    }

    fn small_config() -> (MiningConfiguration, usize, usize) {
        let config = MiningConfiguration {
            common_dim: 64,
            rank: 32,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 1]).unwrap(), // h=2
            cols_pattern: PeriodicPattern::from_list(&[0, 1]).unwrap(), // w=2
            reserved: [0u8; 32],
        };
        (config, 4, 4) // m, n
    }

    /// The FULL pearl pipeline operands (commitment + low-rank noise from
    /// `prepare_attempt`) fed through the GPU dp4a tile kernel must reproduce the
    /// CPU `tile_jackpot` for every tile — bit for bit. Ignored: needs a GPU.
    ///   cargo test --no-default-features --features pearl-cuda -- --ignored --nocapture
    #[test]
    #[ignore]
    fn gpu_pipeline_tile_matches_cpu() {
        let header = IncompleteBlockHeader { version: 1, prev_block: [7; 32], merkle_root: [9; 32], timestamp: 0x6622_1100, nbits: 0x207f_ffff };
        let (config, m, n) = small_config();
        let k = config.common_dim as usize;
        let rank = config.rank as usize;
        let (h, w) = (config.rows_pattern.size() as usize, config.cols_pattern.size() as usize);

        let a: Vec<Vec<i8>> = (0..m).map(|r| mk_signal(0x100 + r as u64, k)).collect();
        let bt: Vec<Vec<i8>> = (0..n).map(|c| mk_signal(0x200 + c as u64, k)).collect();
        let op = prepare_attempt(&header, &config, &a, &bt);

        let row_tiles = threads_partition(&config.rows_pattern, m);
        let col_tiles = threads_partition(&config.cols_pattern, n);
        let mut tiles = Vec::new();
        let mut a_strip = Vec::new();
        let mut b_strip = Vec::new();
        for ar in &row_tiles {
            for bc in &col_tiles {
                for &r in ar {
                    a_strip.extend_from_slice(&op.a_noised[r * k..r * k + k]);
                }
                for &c in bc {
                    b_strip.extend_from_slice(&op.bt_noised[c * k..c * k + k]);
                }
                tiles.push((ar.clone(), bc.clone()));
            }
        }

        let gpu = PearlGpu::new(0).expect("init GPU 0");
        let got = gpu.tile_jackpots_dp4a(&a_strip, &b_strip, tiles.len(), h, w, k, rank).expect("dp4a");
        for (t, (ar, bc)) in tiles.iter().enumerate() {
            let want = tile_jackpot(&op.a_noised, &op.bt_noised, ar, bc, k, rank);
            assert_eq!(got[t], want, "tile {t} (rows {ar:?} cols {bc:?}) GPU != CPU");
        }
        eprintln!("GPU pipeline tile jackpots match CPU for {} tiles", tiles.len());
    }

    /// End-to-end fully-on-GPU attempt: A/Bᵀ generated, committed, noised, and
    /// swept entirely on the GPU. With an all-accepting bound a tile opens; the
    /// returned [`Solution`] (raw operands read back) must pass the independent
    /// CPU `verify()` — proving rng+commitment+noise+harvest agree end-to-end.
    #[test]
    #[ignore]
    fn gpu_mine_full_matches_verify() {
        let header = IncompleteBlockHeader { version: 5, prev_block: [3; 32], merkle_root: [5; 32], timestamp: 0x6677_8899, nbits: 0x207f_ffff };
        let config = MiningConfiguration {
            common_dim: 256, // m*k = 128*256 = 32768 = 32 chunks (chunk-aligned)
            rank: 64,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 32]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&(0..64).collect::<Vec<_>>()).unwrap(),
            reserved: [0u8; 32],
        };
        let (m, n) = (128usize, 128usize);
        let gpu = PearlGpu::new(0).expect("init GPU 0");
        let sol = gpu
            .mine_full_gpu(&header, &config, m, n, U256::MAX, 0xABCD, 1 << 12)
            .expect("full gpu mine")
            .expect("a tile opens under the all-accepting bound");
        assert!(verify(&header, &config, &sol, U256::MAX), "full-GPU solution must pass CPU verify()");
        // Operands must be in the valid signal range.
        assert!(sol.a.iter().flatten().all(|&v| (-64..=64).contains(&v)));
        eprintln!("full-GPU attempt verified by CPU (rows {:?} cols {:?})", sol.a_rows, sol.b_cols);
    }

    /// GPU counter-PRNG operand generation: values in [-64,64], deterministic
    /// (same nonce/salt ⇒ same bytes), and A vs Bᵀ streams (salt) differ. Ignored.
    #[test]
    #[ignore]
    fn gpu_rng_range_and_determinism() {
        let gpu = PearlGpu::new(0).expect("init GPU 0");
        let n = 1_000_000usize;
        let a1 = gpu.gpu_rng(n, 42, 0).expect("rng a1");
        let a2 = gpu.gpu_rng(n, 42, 0).expect("rng a2");
        let b = gpu.gpu_rng(n, 42, 1).expect("rng b");
        assert!(a1.iter().all(|&v| (-64..=64).contains(&v)), "values in [-64,64]");
        assert_eq!(a1, a2, "deterministic for same nonce/salt");
        assert_ne!(a1, b, "different salt ⇒ different stream");
        // sanity: roughly uniform sign (not degenerate)
        let pos = a1.iter().filter(|&&v| v > 0).count();
        assert!(pos > n / 4 && pos < 3 * n / 4, "not degenerate (pos={pos})");
        eprintln!("GPU rng: range+determinism OK ({n} samples)");
    }

    /// GPU keyed-BLAKE3 commitment (chunk CVs on GPU + CPU hazmat merge) equals
    /// `blake3::keyed_hash` over the padded data, at sizes ≥ 2 chunks. Ignored.
    #[test]
    #[ignore]
    fn gpu_commitment_matches_cpu() {
        let gpu = PearlGpu::new(0).expect("init GPU 0");
        let key = [0x5au8; 32];
        for &len in &[2048usize, 4096, 16 * 1024, 4096 * 4096] {
            let data: Vec<u8> = (0..len).map(|i| mk_signal(i as u64, 1)[0] as u8).collect();
            let padded = crate::pearl::merkle::pad_to_chunk_boundary(&data);
            let want = *blake3::keyed_hash(&key, &padded).as_bytes();
            let got = gpu.gpu_commitment(&data, key).expect("gpu commitment");
            assert_eq!(got, want, "commitment mismatch at len {len}");
        }
        eprintln!("GPU commitment matches blake3::keyed_hash (incl. 16 MiB)");
    }

    /// Live-scale feasibility: run ONE full-GPU attempt at a large square config
    /// (`PEARL_MN`, default 32768; the live pool config is 131072) and report
    /// time + tiles/s. Proves the whole on-GPU pipeline runs at scale. Ignored.
    ///   PEARL_MN=131072 cargo test ... gpu_live_scale_attempt -- --ignored --nocapture
    #[test]
    #[ignore]
    fn gpu_live_scale_attempt() {
        use std::time::Instant;
        let mn: usize = std::env::var("PEARL_MN").ok().and_then(|s| s.parse().ok()).unwrap_or(32768);
        let header = IncompleteBlockHeader { version: 1, prev_block: [1; 32], merkle_root: [2; 32], timestamp: 0x6622_3344, nbits: 0x1801_3586 };
        let config = MiningConfiguration {
            common_dim: 4096, rank: 128, mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 32]).unwrap(),                 // h=2
            cols_pattern: PeriodicPattern::from_list(&(0..64).collect::<Vec<_>>()).unwrap(), // w=64
            reserved: [0u8; 32],
        };
        let (m, n, k) = (mn, mn, 4096usize);
        let tiles = (m / 2) * (n / 64);
        let vram_gb = (4.0 * (m * k) as f64) / 1e9; // da_raw,db_raw,da,db
        eprintln!("live-scale: m=n={mn} k={k} ⇒ {tiles} tiles, ~{vram_gb:.2} GB operands");

        let gpu = PearlGpu::new(0).expect("init GPU 0");
        let t0 = Instant::now();
        let got = gpu.mine_full_gpu(&header, &config, m, n, U256::zero(), 1, 1 << 16).expect("attempt");
        let dt = t0.elapsed();
        assert!(got.is_none(), "bound 0 ⇒ no winner");
        eprintln!(
            "live-scale attempt: {dt:?} = {:.2}M tiles/s; projected to 131072² = {:.1}s/attempt",
            tiles as f64 / dt.as_secs_f64() / 1e6,
            (65536f64 * 2048.0) / (tiles as f64 / dt.as_secs_f64())
        );
    }

    /// Isolate the PURE sweep cost (harvest kernel launches only; matrices +
    /// buffers pre-allocated, noise pre-run) from alloc/noise overhead. Ignored.
    #[test]
    #[ignore]
    fn gpu_sweep_only_throughput() {
        use crate::pearl::pearlhash::{attempt_seeds, SEED_LABEL_A, SEED_LABEL_B};
        use std::time::Instant;
        let header = IncompleteBlockHeader { version: 1, prev_block: [1; 32], merkle_root: [2; 32], timestamp: 0x6622_3344, nbits: 0x207f_ffff };
        let config = MiningConfiguration {
            common_dim: 4096, rank: 128, mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 32]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&(0..64).collect::<Vec<_>>()).unwrap(),
            reserved: [0u8; 32],
        };
        let (m, n) = (4096usize, 4096usize);
        let k = config.common_dim as usize;
        let rank = config.rank as usize;
        let (h, w) = (config.rows_pattern.size() as usize, config.cols_pattern.size() as usize);
        let a: Vec<Vec<i8>> = (0..m).map(|r| mk_signal(0xE000 + r as u64, k)).collect();
        let bt: Vec<Vec<i8>> = (0..n).map(|c| mk_signal(0xF000 + c as u64, k)).collect();
        let (b_seed, a_seed) = attempt_seeds(&header, &config, &a, &bt);
        let a_flat: Vec<i8> = a.iter().flatten().copied().collect();
        let bt_flat: Vec<i8> = bt.iter().flatten().copied().collect();
        let row_tiles = threads_partition(&config.rows_pattern, m);
        let col_tiles = threads_partition(&config.cols_pattern, n);
        let mut tiles: Vec<(&Vec<usize>, &Vec<usize>)> = Vec::new();
        for ar in &row_tiles { for bc in &col_tiles { tiles.push((ar, bc)); } }
        let ntiles = tiles.len();
        let batch_tiles = 1usize << 16;

        let gpu = PearlGpu::new(0).expect("init GPU 0");
        unsafe {
            let (mut da, mut db): (CUdeviceptr, CUdeviceptr) = (0, 0);
            cu(cuMemAlloc_v2(&mut da, m * k), "a").unwrap();
            cu(cuMemAlloc_v2(&mut db, n * k), "b").unwrap();
            let tn = Instant::now();
            gpu.noise_into(da, &a_flat, a_seed, SEED_LABEL_A, m, k, rank).unwrap();
            gpu.noise_into(db, &bt_flat, b_seed, SEED_LABEL_B, n, k, rank).unwrap();
            let noise = tn.elapsed();
            let cap = batch_tiles;
            let (mut d_ar, mut d_bc, mut d_key, mut d_bound, mut d_wc, mut d_win): (CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr) = (0, 0, 0, 0, 0, 0);
            cu(cuMemAlloc_v2(&mut d_ar, cap * h * 4), "ar").unwrap();
            cu(cuMemAlloc_v2(&mut d_bc, cap * w * 4), "bc").unwrap();
            cu(cuMemAlloc_v2(&mut d_key, 32), "k").unwrap();
            cu(cuMemAlloc_v2(&mut d_bound, 32), "bd").unwrap();
            cu(cuMemAlloc_v2(&mut d_wc, 4), "wc").unwrap();
            cu(cuMemAlloc_v2(&mut d_win, cap * 9 * 4), "win").unwrap();
            let key_w = bytes_to_words(&a_seed);
            cu(cuMemcpyHtoD_v2(d_key, key_w.as_ptr() as *const c_void, 32), "hk").unwrap();
            let zero_w = [0u32; 8];
            cu(cuMemcpyHtoD_v2(d_bound, zero_w.as_ptr() as *const c_void, 32), "hb").unwrap();
            let run = || {
                for ch in tiles.chunks(batch_tiles) {
                    gpu.harvest_batch(da, db, d_ar, d_bc, d_key, d_bound, d_wc, d_win, cap, ch, h, w, k, rank).unwrap();
                }
            };
            run(); // warmup
            let t = Instant::now();
            let iters = 5;
            for _ in 0..iters { run(); }
            let sweep = t.elapsed() / iters;
            eprintln!("PURE sweep {ntiles} tiles: {sweep:?} = {:.2}M tiles/s; noise(2×) = {noise:?}", ntiles as f64 / sweep.as_secs_f64() / 1e6);
            for p in [da, db, d_ar, d_bc, d_key, d_bound, d_wc, d_win] { cuMemFree_v2(p); }
        }
    }

    /// The fused on-GPU harvest (BLAKE3 + bound compare + winner emit) must
    /// agree with the CPU: (1) with bound=MAX every tile is emitted and each
    /// emitted hash equals the CPU `compute_jackpot_hash`; (2) with a mid bound
    /// the GPU winner SET equals the CPU `{tile : hash(LE) ≤ bound}`. Ignored: GPU.
    #[test]
    #[ignore]
    fn gpu_harvest_winner_set_matches_cpu() {
        use crate::pearl::pearlhash::{attempt_seeds, compute_jackpot_hash, SEED_LABEL_A, SEED_LABEL_B};
        let header = IncompleteBlockHeader { version: 7, prev_block: [4; 32], merkle_root: [8; 32], timestamp: 0x6655_4433, nbits: 0x207f_ffff };
        let config = MiningConfiguration {
            common_dim: 256,
            rank: 64,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 32]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&(0..64).collect::<Vec<_>>()).unwrap(),
            reserved: [0u8; 32],
        };
        let (m, n) = (128usize, 128usize);
        let k = config.common_dim as usize;
        let rank = config.rank as usize;
        let (h, w) = (config.rows_pattern.size() as usize, config.cols_pattern.size() as usize);
        let a: Vec<Vec<i8>> = (0..m).map(|r| mk_signal(0x7100 + r as u64, k)).collect();
        let bt: Vec<Vec<i8>> = (0..n).map(|c| mk_signal(0x7200 + c as u64, k)).collect();
        let op = prepare_attempt(&header, &config, &a, &bt);
        let (_b_seed, a_seed) = attempt_seeds(&header, &config, &a, &bt);

        // CPU reference: every tile's hash, in batch (= global) order.
        let row_tiles = threads_partition(&config.rows_pattern, m);
        let col_tiles = threads_partition(&config.cols_pattern, n);
        let mut tiles: Vec<(&Vec<usize>, &Vec<usize>)> = Vec::new();
        for ar in &row_tiles {
            for bc in &col_tiles {
                tiles.push((ar, bc));
            }
        }
        let cpu_hash: Vec<[u8; 32]> = tiles
            .iter()
            .map(|(ar, bc)| compute_jackpot_hash(&tile_jackpot(&op.a_noised, &op.bt_noised, ar, bc, k, rank), a_seed))
            .collect();
        let mut sorted: Vec<U256> = cpu_hash.iter().map(|hh| U256::from_little_endian(hh)).collect();
        sorted.sort();
        let mid_bound = sorted[tiles.len() / 2]; // ~half the tiles qualify
        let cpu_set: std::collections::BTreeSet<usize> =
            (0..tiles.len()).filter(|&i| U256::from_little_endian(&cpu_hash[i]) <= mid_bound).collect();

        let a_flat: Vec<i8> = a.iter().flatten().copied().collect();
        let bt_flat: Vec<i8> = bt.iter().flatten().copied().collect();
        let gpu = PearlGpu::new(0).expect("init GPU 0");
        let cap = tiles.len();
        unsafe {
            let (mut da, mut db): (CUdeviceptr, CUdeviceptr) = (0, 0);
            cu(cuMemAlloc_v2(&mut da, m * k), "a").unwrap();
            cu(cuMemAlloc_v2(&mut db, n * k), "b").unwrap();
            gpu.noise_into(da, &a_flat, a_seed, SEED_LABEL_A, m, k, rank).unwrap();
            gpu.noise_into(db, &bt_flat, _b_seed, SEED_LABEL_B, n, k, rank).unwrap();
            let (mut d_ar, mut d_bc, mut d_key, mut d_bound, mut d_wc, mut d_win): (CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr) = (0, 0, 0, 0, 0, 0);
            cu(cuMemAlloc_v2(&mut d_ar, cap * h * 4), "ar").unwrap();
            cu(cuMemAlloc_v2(&mut d_bc, cap * w * 4), "bc").unwrap();
            cu(cuMemAlloc_v2(&mut d_key, 32), "k").unwrap();
            cu(cuMemAlloc_v2(&mut d_bound, 32), "bd").unwrap();
            cu(cuMemAlloc_v2(&mut d_wc, 4), "wc").unwrap();
            cu(cuMemAlloc_v2(&mut d_win, cap * 9 * 4), "win").unwrap();
            let key_w = bytes_to_words(&a_seed);
            cu(cuMemcpyHtoD_v2(d_key, key_w.as_ptr() as *const c_void, 32), "hk").unwrap();
            let upload_bound = |b: U256| {
                let mut le = [0u8; 32];
                b.to_little_endian(&mut le);
                let w = bytes_to_words(&le);
                cu(cuMemcpyHtoD_v2(d_bound, w.as_ptr() as *const c_void, 32), "hb").unwrap();
            };

            // (1) bound = MAX: all tiles win; each hash matches CPU.
            upload_bound(U256::MAX);
            let all = gpu.harvest_batch(da, db, d_ar, d_bc, d_key, d_bound, d_wc, d_win, cap, &tiles, h, w, k, rank).unwrap();
            assert_eq!(all.len(), tiles.len(), "bound=MAX should emit every tile");
            for (idx, hash) in &all {
                assert_eq!(*hash, cpu_hash[*idx], "GPU hash != CPU at tile {idx}");
            }

            // (2) mid bound: winner set matches CPU.
            upload_bound(mid_bound);
            let mid = gpu.harvest_batch(da, db, d_ar, d_bc, d_key, d_bound, d_wc, d_win, cap, &tiles, h, w, k, rank).unwrap();
            let gpu_set: std::collections::BTreeSet<usize> = mid.iter().map(|(i, _)| *i).collect();
            assert_eq!(gpu_set, cpu_set, "GPU winner set != CPU winner set");

            for p in [da, db, d_ar, d_bc, d_key, d_bound, d_wc, d_win] {
                cuMemFree_v2(p);
            }
            eprintln!("GPU harvest: all {} hashes match; mid-bound winner set ({} tiles) matches CPU", tiles.len(), cpu_set.len());
        }
    }

    /// The IMMA tensor-core sweep (`harvest_all_wmma`, stacks 16/h row-tiles into
    /// a 16-row s8 WMMA fragment) must produce the same winner SET + hashes as the
    /// CPU `{tile : hash(LE) ≤ bound}` (global index rtile·ncol + ctile). Ignored.
    #[test]
    #[ignore]
    fn gpu_wmma_winner_set_matches_cpu() {
        use crate::pearl::pearlhash::{attempt_seeds, compute_jackpot_hash, SEED_LABEL_A, SEED_LABEL_B};
        let header = IncompleteBlockHeader { version: 11, prev_block: [13; 32], merkle_root: [17; 32], timestamp: 0x6678_9abc, nbits: 0x207f_ffff };
        let config = MiningConfiguration {
            common_dim: 256, rank: 64, mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 32]).unwrap(), // h=2 ⇒ G=8
            cols_pattern: PeriodicPattern::from_list(&(0..64).collect::<Vec<_>>()).unwrap(), // w=64
            reserved: [0u8; 32],
        };
        let (m, n) = (256usize, 256usize);
        let k = config.common_dim as usize;
        let rank = config.rank as usize;
        let (h, w) = (config.rows_pattern.size() as usize, config.cols_pattern.size() as usize);
        let a: Vec<Vec<i8>> = (0..m).map(|r| mk_signal(0x5100 + r as u64, k)).collect();
        let bt: Vec<Vec<i8>> = (0..n).map(|c| mk_signal(0x5200 + c as u64, k)).collect();
        let op = prepare_attempt(&header, &config, &a, &bt);
        let (b_seed, a_seed) = attempt_seeds(&header, &config, &a, &bt);
        let row_tiles = threads_partition(&config.rows_pattern, m);
        let col_tiles = threads_partition(&config.cols_pattern, n);
        let ncol = col_tiles.len();

        let mut cpu_hash = std::collections::BTreeMap::new();
        for (ri, ar) in row_tiles.iter().enumerate() {
            for (ci, bc) in col_tiles.iter().enumerate() {
                let jp = tile_jackpot(&op.a_noised, &op.bt_noised, ar, bc, k, rank);
                cpu_hash.insert(ri * ncol + ci, compute_jackpot_hash(&jp, a_seed));
            }
        }
        let mut sorted: Vec<U256> = cpu_hash.values().map(|hh| U256::from_little_endian(&hh[..])).collect();
        sorted.sort();
        let mid = sorted[sorted.len() / 2];
        let cpu_set: std::collections::BTreeSet<usize> =
            cpu_hash.iter().filter(|(_, hh)| U256::from_little_endian(&hh[..]) <= mid).map(|(i, _)| *i).collect();

        let a_flat: Vec<i8> = a.iter().flatten().copied().collect();
        let bt_flat: Vec<i8> = bt.iter().flatten().copied().collect();
        let gpu = PearlGpu::new(0).expect("init GPU 0");
        let ntiles = row_tiles.len() * ncol;
        unsafe {
            let (mut da, mut db): (CUdeviceptr, CUdeviceptr) = (0, 0);
            cu(cuMemAlloc_v2(&mut da, m * k), "a").unwrap();
            cu(cuMemAlloc_v2(&mut db, n * k), "b").unwrap();
            gpu.noise_into(da, &a_flat, a_seed, SEED_LABEL_A, m, k, rank).unwrap();
            gpu.noise_into(db, &bt_flat, b_seed, SEED_LABEL_B, n, k, rank).unwrap();
            let won = gpu.harvest_all_wmma(da, db, a_seed, mid, &row_tiles, &col_tiles, h, w, k, rank, ntiles).unwrap();
            let gpu_set: std::collections::BTreeSet<usize> = won.iter().map(|(i, _)| *i).collect();
            assert_eq!(gpu_set, cpu_set, "WMMA winner set mismatch");
            for (i, hash) in &won {
                assert_eq!(*hash, cpu_hash[i], "WMMA hash mismatch at tile {i}");
            }
            cuMemFree_v2(da);
            cuMemFree_v2(db);
        }
        eprintln!("IMMA tensor-core sweep winner set matches CPU ({ntiles} tiles, G=8)");
    }

    /// The cross-tile-reuse batched sweep (`harvest_all`, single 2D-grid launch)
    /// must produce the same winner SET as the CPU `{tile : hash(LE) ≤ bound}`,
    /// with global tile index = rtile·ncol + ctile, for G row-tiles/block. Ignored.
    #[test]
    #[ignore]
    fn gpu_batched_winner_set_matches_cpu() {
        use crate::pearl::pearlhash::{attempt_seeds, compute_jackpot_hash, SEED_LABEL_A, SEED_LABEL_B};
        let header = IncompleteBlockHeader { version: 9, prev_block: [12; 32], merkle_root: [21; 32], timestamp: 0x6612_3456, nbits: 0x207f_ffff };
        let config = MiningConfiguration {
            common_dim: 256, rank: 64, mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 32]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&(0..64).collect::<Vec<_>>()).unwrap(),
            reserved: [0u8; 32],
        };
        let (m, n) = (128usize, 128usize);
        let k = config.common_dim as usize;
        let rank = config.rank as usize;
        let (h, w) = (config.rows_pattern.size() as usize, config.cols_pattern.size() as usize);
        let a: Vec<Vec<i8>> = (0..m).map(|r| mk_signal(0x4100 + r as u64, k)).collect();
        let bt: Vec<Vec<i8>> = (0..n).map(|c| mk_signal(0x4200 + c as u64, k)).collect();
        let op = prepare_attempt(&header, &config, &a, &bt);
        let (b_seed, a_seed) = attempt_seeds(&header, &config, &a, &bt);
        let row_tiles = threads_partition(&config.rows_pattern, m);
        let col_tiles = threads_partition(&config.cols_pattern, n);
        let ncol = col_tiles.len();

        // CPU reference, global index = rtile*ncol + ctile.
        let mut cpu_hash = std::collections::BTreeMap::new();
        for (ri, ar) in row_tiles.iter().enumerate() {
            for (ci, bc) in col_tiles.iter().enumerate() {
                let jp = tile_jackpot(&op.a_noised, &op.bt_noised, ar, bc, k, rank);
                cpu_hash.insert(ri * ncol + ci, compute_jackpot_hash(&jp, a_seed));
            }
        }
        let mut sorted: Vec<U256> = cpu_hash.values().map(|hh| U256::from_little_endian(hh)).collect();
        sorted.sort();
        let mid = sorted[sorted.len() / 2];
        let cpu_set: std::collections::BTreeSet<usize> =
            cpu_hash.iter().filter(|(_, hh)| U256::from_little_endian(&hh[..]) <= mid).map(|(i, _)| *i).collect();

        let a_flat: Vec<i8> = a.iter().flatten().copied().collect();
        let bt_flat: Vec<i8> = bt.iter().flatten().copied().collect();
        let gpu = PearlGpu::new(0).expect("init GPU 0");
        let ntiles = row_tiles.len() * ncol;
        unsafe {
            let (mut da, mut db): (CUdeviceptr, CUdeviceptr) = (0, 0);
            cu(cuMemAlloc_v2(&mut da, m * k), "a").unwrap();
            cu(cuMemAlloc_v2(&mut db, n * k), "b").unwrap();
            gpu.noise_into(da, &a_flat, a_seed, SEED_LABEL_A, m, k, rank).unwrap();
            gpu.noise_into(db, &bt_flat, b_seed, SEED_LABEL_B, n, k, rank).unwrap();
            for g in [1usize, 4, 8] {
                let won = gpu.harvest_all(da, db, a_seed, mid, &row_tiles, &col_tiles, h, w, k, rank, g, ntiles).unwrap();
                let gpu_set: std::collections::BTreeSet<usize> = won.iter().map(|(i, _)| *i).collect();
                assert_eq!(gpu_set, cpu_set, "batched winner set mismatch at G={g}");
                for (i, hash) in &won {
                    assert_eq!(*hash, cpu_hash[i], "batched hash mismatch tile {i} G={g}");
                }
            }
            cuMemFree_v2(da);
            cuMemFree_v2(db);
        }
        eprintln!("batched sweep winner set matches CPU ({} tiles, G∈{{1,4,8}})", ntiles);
    }

    /// On-GPU low-rank noise generation must reproduce the CPU `prepare_attempt`
    /// noised operands `A′`/`B′ᵀ` bit-for-bit (both sides). Ignored: needs a GPU.
    #[test]
    #[ignore]
    fn gpu_noise_matches_cpu() {
        use crate::pearl::pearlhash::{attempt_seeds, SEED_LABEL_A, SEED_LABEL_B};
        let header = IncompleteBlockHeader { version: 3, prev_block: [11; 32], merkle_root: [22; 32], timestamp: 0x6644_2200, nbits: 0x207f_ffff };
        let config = MiningConfiguration {
            common_dim: 256,
            rank: 64,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 32]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&(0..64).collect::<Vec<_>>()).unwrap(),
            reserved: [0u8; 32],
        };
        let (m, n) = (128usize, 128usize);
        let k = config.common_dim as usize;
        let rank = config.rank as usize;
        let a: Vec<Vec<i8>> = (0..m).map(|r| mk_signal(0x3100 + r as u64, k)).collect();
        let bt: Vec<Vec<i8>> = (0..n).map(|c| mk_signal(0x3200 + c as u64, k)).collect();

        let op = prepare_attempt(&header, &config, &a, &bt);
        let (b_seed, a_seed) = attempt_seeds(&header, &config, &a, &bt);
        let a_flat: Vec<i8> = a.iter().flatten().copied().collect();
        let bt_flat: Vec<i8> = bt.iter().flatten().copied().collect();

        let gpu = PearlGpu::new(0).expect("init GPU 0");
        let ga = gpu.gpu_noise(&a_flat, a_seed, SEED_LABEL_A, m, k, rank).expect("noise A");
        let gb = gpu.gpu_noise(&bt_flat, b_seed, SEED_LABEL_B, n, k, rank).expect("noise B");
        assert_eq!(ga, op.a_noised, "GPU A' != CPU A'");
        assert_eq!(gb, op.bt_noised, "GPU B'^T != CPU B'^T");
        eprintln!("GPU noise generation matches CPU for A'({m}×{k}) and B'^T({n}×{k})");
    }

    /// The gather-free indexed sweep (resident matrices, per-tile indices) must
    /// reproduce the CPU `tile_jackpot` for every tile of a multi-batch attempt
    /// at a non-trivial size (m=n=256, h=2 w=64 ⇒ 128×4 = 512 tiles). Exercises
    /// `try_mine_one_gpu`'s batch loop indirectly via an impossible bound so it
    /// sweeps all tiles, then re-checks the math. Ignored: needs a GPU.
    #[test]
    #[ignore]
    fn gpu_indexed_sweep_matches_cpu() {
        let header = IncompleteBlockHeader { version: 2, prev_block: [5; 32], merkle_root: [6; 32], timestamp: 0x6611_2233, nbits: 0x207f_ffff };
        let config = MiningConfiguration {
            common_dim: 256,
            rank: 64,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 32]).unwrap(),                 // h=2
            cols_pattern: PeriodicPattern::from_list(&(0..64).collect::<Vec<_>>()).unwrap(), // w=64
            reserved: [0u8; 32],
        };
        let (m, n) = (256usize, 256usize);
        let k = config.common_dim as usize;
        let rank = config.rank as usize;
        let (h, w) = (config.rows_pattern.size() as usize, config.cols_pattern.size() as usize);
        let a: Vec<Vec<i8>> = (0..m).map(|r| mk_signal(0xC000 + r as u64, k)).collect();
        let bt: Vec<Vec<i8>> = (0..n).map(|c| mk_signal(0xD000 + c as u64, k)).collect();
        let op = prepare_attempt(&header, &config, &a, &bt);

        // Reference: CPU jackpot per tile in the same iteration order.
        let row_tiles = threads_partition(&config.rows_pattern, m);
        let col_tiles = threads_partition(&config.cols_pattern, n);
        let mut want: Vec<[u32; 16]> = Vec::new();
        for ar in &row_tiles {
            for bc in &col_tiles {
                want.push(tile_jackpot(&op.a_noised, &op.bt_noised, ar, bc, k, rank));
            }
        }

        // GPU: drive the indexed kernel directly via the resident-matrix path.
        let gpu = PearlGpu::new(0).expect("init GPU 0");
        let mut idx = 0usize;
        let mut all_match = true;
        unsafe {
            let mut da: CUdeviceptr = 0;
            let mut db: CUdeviceptr = 0;
            cu(cuMemAlloc_v2(&mut da, op.a_noised.len()), "a").unwrap();
            cu(cuMemAlloc_v2(&mut db, op.bt_noised.len()), "b").unwrap();
            cu(cuMemcpyHtoD_v2(da, op.a_noised.as_ptr() as *const c_void, op.a_noised.len()), "ha").unwrap();
            cu(cuMemcpyHtoD_v2(db, op.bt_noised.as_ptr() as *const c_void, op.bt_noised.len()), "hb").unwrap();
            let bt_tiles = 100usize; // odd batch to exercise the remainder
            let mut d_ar: CUdeviceptr = 0;
            let mut d_bc: CUdeviceptr = 0;
            let mut dout: CUdeviceptr = 0;
            cu(cuMemAlloc_v2(&mut d_ar, bt_tiles * h * 4), "ar").unwrap();
            cu(cuMemAlloc_v2(&mut d_bc, bt_tiles * w * 4), "bc").unwrap();
            cu(cuMemAlloc_v2(&mut dout, bt_tiles * 16 * 4), "o").unwrap();
            let mut batch: Vec<(&Vec<usize>, &Vec<usize>)> = Vec::new();
            let mut flush = |gpu: &PearlGpu, batch: &mut Vec<(&Vec<usize>, &Vec<usize>)>, idx: &mut usize, all_match: &mut bool| {
                if batch.is_empty() {
                    return;
                }
                // Bound 0 ⇒ never a winner; we re-read jackpots via the public path instead.
                let _ = gpu.sweep_idx_batch(da, db, d_ar, d_bc, dout, batch, h, w, k, rank, op.a_noise_seed, U256::zero()).unwrap();
                let mut got = vec![[0u32; 16]; batch.len()];
                cu(cuMemcpyDtoH_v2(got.as_mut_ptr() as *mut c_void, dout, batch.len() * 16 * 4), "d").unwrap();
                for g in &got {
                    if *g != want[*idx] {
                        *all_match = false;
                    }
                    *idx += 1;
                }
                batch.clear();
            };
            for ar in &row_tiles {
                for bc in &col_tiles {
                    batch.push((ar, bc));
                    if batch.len() == bt_tiles {
                        flush(&gpu, &mut batch, &mut idx, &mut all_match);
                    }
                }
            }
            flush(&gpu, &mut batch, &mut idx, &mut all_match);
            cuMemFree_v2(da);
            cuMemFree_v2(db);
            cuMemFree_v2(d_ar);
            cuMemFree_v2(d_bc);
            cuMemFree_v2(dout);
        }
        assert_eq!(idx, want.len(), "swept all tiles");
        assert!(all_match, "indexed GPU sweep must match CPU for all {} tiles", want.len());
        eprintln!("gather-free indexed GPU sweep matches CPU for {} tiles", want.len());
    }

    /// Throughput probe (not a correctness check): time one full GPU attempt over
    /// a medium config (m=n=4096, k=4096, h=2 w=64 ⇒ 131072 tiles) with an
    /// impossible bound so every tile is swept. Reports tiles/s incl. CPU
    /// operand-gen + jackpot-hash. Ignored: needs a GPU + a couple seconds.
    #[test]
    #[ignore]
    fn gpu_sweep_throughput() {
        use std::time::Instant;
        let header = IncompleteBlockHeader { version: 1, prev_block: [1; 32], merkle_root: [2; 32], timestamp: 0x6622_3344, nbits: 0x207f_ffff };
        let config = MiningConfiguration {
            common_dim: 4096,
            rank: 128,
            mma_type: MMAType::Int7xInt7ToInt32,
            rows_pattern: PeriodicPattern::from_list(&[0, 32]).unwrap(),
            cols_pattern: PeriodicPattern::from_list(&(0..64).collect::<Vec<_>>()).unwrap(),
            reserved: [0u8; 32],
        };
        let (m, n) = (4096usize, 4096usize);
        let k = config.common_dim as usize;
        let tiles = (m / 2) * (n / 64);

        let gpu = PearlGpu::new(0).expect("init GPU 0");

        // Phase breakdown of one full GPU-noise attempt.
        let tg = Instant::now();
        let a: Vec<Vec<i8>> = (0..m).map(|r| mk_signal(0xE000 + r as u64, k)).collect();
        let bt: Vec<Vec<i8>> = (0..n).map(|c| mk_signal(0xF000 + c as u64, k)).collect();
        let gen = tg.elapsed();
        let (b_seed, a_seed) = crate::pearl::pearlhash::attempt_seeds(&header, &config, &a, &bt);
        let ts = Instant::now();
        let _ = crate::pearl::pearlhash::attempt_seeds(&header, &config, &a, &bt);
        let seeds = ts.elapsed();

        // Noise-only timing (incl. a small host readback).
        let a_flat: Vec<i8> = a.iter().flatten().copied().collect();
        let bt_flat: Vec<i8> = bt.iter().flatten().copied().collect();
        let rank = config.rank as usize;
        let tn = Instant::now();
        let _ = gpu.gpu_noise(&a_flat, a_seed, crate::pearl::pearlhash::SEED_LABEL_A, m, k, rank).unwrap();
        let _ = gpu.gpu_noise(&bt_flat, b_seed, crate::pearl::pearlhash::SEED_LABEL_B, n, k, rank).unwrap();
        let noise = tn.elapsed();

        let t0 = Instant::now();
        let got = gpu.try_mine_one_gpu(&header, &config, a, bt, m, n, U256::zero(), 1 << 16).expect("sweep");
        let dt = t0.elapsed();
        assert!(got.is_none(), "bound 0 ⇒ no winner");

        // Fully-on-GPU attempt (rng + commitment + noise + harvest on the GPU).
        gpu.mine_full_gpu(&header, &config, m, n, U256::zero(), 1, 1 << 16).expect("full"); // warmup
        let tf = Instant::now();
        let gf = gpu.mine_full_gpu(&header, &config, m, n, U256::zero(), 2, 1 << 16).expect("full");
        let full = tf.elapsed();
        assert!(gf.is_none());
        eprintln!(
            "{tiles} tiles:\n  CPU-operand path: rng-gen {gen:?} + commitment {seeds:?} (CPU) + GPU {dt:?}\n  \
             full-GPU path:    {full:?} = {:.2}M tiles/s (rng+commit+noise+harvest all on GPU)\n  \
             (GPU noise 2× = {noise:?})",
            tiles as f64 / full.as_secs_f64() / 1e6
        );
    }

    /// End-to-end: with an all-accepting bound the first tile opens; the GPU
    /// orchestration's [`Solution`] must be accepted by the independent CPU
    /// `verify()` (which re-derives noise + jackpot from scratch). Ignored: GPU.
    #[test]
    #[ignore]
    fn gpu_try_mine_one_matches_verify() {
        let header = IncompleteBlockHeader { version: 0, prev_block: [1; 32], merkle_root: [2; 32], timestamp: 0x6666_6666, nbits: 0x207f_ffff };
        let (config, m, n) = small_config();
        let a: Vec<Vec<i8>> = (0..m).map(|r| mk_signal(0x900 + r as u64, config.common_dim as usize)).collect();
        let bt: Vec<Vec<i8>> = (0..n).map(|c| mk_signal(0xA00 + c as u64, config.common_dim as usize)).collect();

        let gpu = PearlGpu::new(0).expect("init GPU 0");
        let sol = gpu
            .try_mine_one_gpu(&header, &config, a, bt, m, n, U256::MAX, 2)
            .expect("gpu mine")
            .expect("a tile opens under the all-accepting bound");
        assert!(verify(&header, &config, &sol, U256::MAX), "GPU solution must pass the CPU verifier");
        eprintln!("GPU try_mine_one_gpu solution verified by CPU (rows {:?} cols {:?})", sol.a_rows, sol.b_cols);
    }

    /// CPU reference: 16×16-tile jackpot (XOR over the whole tile per rank-chunk).
    fn cpu_tile_jackpot(a: &[i8], b: &[i8], k: usize, rank: usize) -> [u32; 16] {
        let mut m = [0u32; 16];
        let mut tile = [[0i32; 16]; 16]; // Cblk ACCUMULATES across rank-chunks
        let mut chunk = 0usize;
        let mut ll = rank;
        while ll <= k {
            for u in 0..16 {
                for v in 0..16 {
                    for l in ll - rank..ll {
                        tile[u][v] = tile[u][v].wrapping_add(a[u * k + l] as i32 * b[v * k + l] as i32);
                    }
                }
            }
            let mut x = 0u32;
            for u in 0..16 {
                for v in 0..16 {
                    x ^= tile[u][v] as u32;
                }
            }
            m[chunk % 16] = m[chunk % 16].rotate_left(13) ^ x;
            chunk += 1;
            ll += rank;
        }
        m
    }

    /// CPU reference for a general h×w cumulative tile jackpot.
    fn cpu_hw_jackpot(a: &[i8], b: &[i8], h: usize, w: usize, k: usize, rank: usize) -> [u32; 16] {
        let mut m = [0u32; 16];
        let mut tile = vec![0i32; h * w];
        let mut chunk = 0usize;
        let mut ll = rank;
        while ll <= k {
            for u in 0..h {
                for v in 0..w {
                    let mut s = tile[u * w + v];
                    for l in ll - rank..ll {
                        s = s.wrapping_add(a[u * k + l] as i32 * b[v * k + l] as i32);
                    }
                    tile[u * w + v] = s;
                }
            }
            let x = tile.iter().fold(0u32, |acc, &e| acc ^ e as u32);
            m[chunk % 16] = m[chunk % 16].rotate_left(13) ^ x;
            chunk += 1;
            ll += rank;
        }
        m
    }

    /// DP4A int8 path on the live tall-skinny tile shape (h=2, w=64) must equal
    /// the CPU transcript. Ignored — needs a GPU (sm_61+ for dp4a). Run:
    ///   cargo test --no-default-features --features pearl-cuda -- --ignored --nocapture
    #[test]
    #[ignore]
    fn gpu_dp4a_matches_cpu() {
        let (h, w, k, rank, tiles) = (2usize, 64usize, 256usize, 32usize, 4usize);
        let mk = |seed: u64, n: usize| -> Vec<i8> {
            let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
            (0..n).map(|_| { x ^= x >> 12; x ^= x << 25; x ^= x >> 27; (((x >> 56) as i64) % 127 - 63) as i8 }).collect()
        };
        let a = mk(1, tiles * h * k);
        let b = mk(2, tiles * w * k);
        let gpu = PearlGpu::new(0).expect("init GPU 0");
        let got = gpu.tile_jackpots_dp4a(&a, &b, tiles, h, w, k, rank).expect("dp4a run");
        for t in 0..tiles {
            let want = cpu_hw_jackpot(&a[t * h * k..(t + 1) * h * k], &b[t * w * k..(t + 1) * w * k], h, w, k, rank);
            assert_eq!(got[t], want, "dp4a tile {t} mismatch");
        }
        eprintln!("DP4A int8 tile jackpots match CPU for {tiles} tiles (h={h} w={w})");
    }

    /// int8 tensor-core (WMMA/IMMA) tile jackpot must equal the CPU transcript.
    /// Ignored by default — needs an NVIDIA GPU (sm_72+ for s8 WMMA). Run:
    ///   cargo test --no-default-features --features pearl-cuda -- --ignored --nocapture
    #[test]
    #[ignore]
    fn gpu_wmma_matches_cpu() {
        let (k, rank, tiles) = (64usize, 32usize, 3usize);
        let mk = |seed: u64, n: usize| -> Vec<i8> {
            let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
            (0..n)
                .map(|_| {
                    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
                    (((x.wrapping_mul(0x2545F4914F6CDD1D) >> 56) as i64) % 127 - 63) as i8 // [-63,63]
                })
                .collect()
        };
        let a = mk(1, tiles * 16 * k);
        let b = mk(2, tiles * 16 * k);

        let gpu = PearlGpu::new(0).expect("init GPU 0");
        let got = gpu.tile_jackpots(&a, &b, tiles, k, rank).expect("gpu run");
        for t in 0..tiles {
            let off = t * 16 * k;
            let want = cpu_tile_jackpot(&a[off..off + 16 * k], &b[off..off + 16 * k], k, rank);
            assert_eq!(got[t], want, "tile {t} jackpot mismatch (GPU IMMA vs CPU)");
        }
        eprintln!("int8 tensor-core (WMMA) tile jackpots match CPU for {tiles} tiles");
    }
}
