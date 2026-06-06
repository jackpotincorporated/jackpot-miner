// GPU Equihash 192,7 solver.
//
// Pipeline (all stages on the GPU):
//   gen          - BLAKE2b each index, bucket entries into table[0] by block 0
//   round_collide- for r=0..5: collide block r in table[r], write table[r+1]
//   final_round  - collide blocks 6 & 7 in table[6], emit candidate solutions
//   recover      - walk back-references to reconstruct 128 leaf indices/solution
//
// A "block" is one 24-bit (3-byte) chunk of the 192-bit hash. Entries are
// bucketed by the high ROW_BITS bits of the current collision block; the full
// block is re-checked within a bucket.
//
// Memory model (kept small so 8-12 GB cards work): each entry's data is split.
//   - back-reference (1 u32/slot): a leaf index in table[0], otherwise a packed
//     (row,slot,slot) parent reference. One persistent array per table (0..6),
//     because `recover` walks these — and ONLY these — to rebuild a solution.
//   - blocks (the remaining 24-bit hash words): needed only during the round
//     that consumes them, so they live in two ping-pong working buffers
//     (`src_blk`/`dst_blk`, MAXB words per slot) reused across all rounds.
// This avoids keeping seven full block tables resident (~11 GB → ~6 GB).
//
// The host prepends a prelude of #defines (NR_ROWS, NR_SLOTS, ROW_BITS, CBL,
// MAXB, MAX_SOLS), so those symbols are defined here.

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

typedef ulong u64;

__constant u64 BLAKE2B_IV[8] = {
    0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL,
    0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL,
    0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL,
    0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL
};

__constant uchar SIGMA[12][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15 },
    {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3 },
    {11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4 },
    { 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8 },
    { 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13 },
    { 2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9 },
    {12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11 },
    {13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10 },
    { 6,15,14, 9,11, 3, 0, 8,12, 2,13, 7, 1, 4,10, 5 },
    {10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13, 0 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15 },
    {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3 }
};

static inline u64 rotr64(u64 x, uint n) {
    return (x >> n) | (x << (64 - n));
}

#define G(r, i, a, b, c, d)                       \
    a = a + b + m[SIGMA[r][2*i + 0]];             \
    d = rotr64(d ^ a, 32);                        \
    c = c + d;                                    \
    b = rotr64(b ^ c, 24);                        \
    a = a + b + m[SIGMA[r][2*i + 1]];             \
    d = rotr64(d ^ a, 16);                        \
    c = c + d;                                    \
    b = rotr64(b ^ c, 63);

static void compress(u64 h[8], const u64 m[16], u64 t, int last) {
    u64 v[16];
    for (int i = 0; i < 8; i++) v[i] = h[i];
    for (int i = 0; i < 8; i++) v[8 + i] = BLAKE2B_IV[i];
    v[12] ^= t;
    if (last) v[14] = ~v[14];

    for (int r = 0; r < 12; r++) {
        G(r, 0, v[0], v[4], v[ 8], v[12]);
        G(r, 1, v[1], v[5], v[ 9], v[13]);
        G(r, 2, v[2], v[6], v[10], v[14]);
        G(r, 3, v[3], v[7], v[11], v[15]);
        G(r, 4, v[0], v[5], v[10], v[15]);
        G(r, 5, v[1], v[6], v[11], v[12]);
        G(r, 6, v[2], v[7], v[ 8], v[13]);
        G(r, 7, v[3], v[4], v[ 9], v[14]);
    }
    for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[8 + i];
}

static inline u64 load64(const uchar *p) {
    return (u64)p[0]        | ((u64)p[1] << 8)  | ((u64)p[2] << 16) | ((u64)p[3] << 24) |
           ((u64)p[4] << 32)| ((u64)p[5] << 40) | ((u64)p[6] << 48) | ((u64)p[7] << 56);
}

// Personalised BLAKE2b of header[0..140] || LE32(g); writes the 48-byte digest.
static void blake_hash(__global const uchar *header, uint g, uchar out[48]) {
    uchar msg[256];
    for (int i = 0; i < 140; i++) msg[i] = header[i];
    msg[140] = (uchar)(g & 0xff);
    msg[141] = (uchar)((g >> 8) & 0xff);
    msg[142] = (uchar)((g >> 16) & 0xff);
    msg[143] = (uchar)((g >> 24) & 0xff);
    for (int i = 144; i < 256; i++) msg[i] = 0;

    u64 h[8];
    for (int i = 0; i < 8; i++) h[i] = BLAKE2B_IV[i];
    h[0] ^= 0x0000000001010030UL;
    h[6] ^= 0x576f50687361635aUL;  // "ZcashPoW"
    h[7] ^= 0x00000007000000c0UL;  // LE32(192) || LE32(7)

    u64 m[16];
    for (int i = 0; i < 16; i++) m[i] = load64(msg + i * 8);
    compress(h, m, 128, 0);
    for (int i = 0; i < 16; i++) m[i] = load64(msg + 128 + i * 8);
    compress(h, m, 144, 1);

    for (int w = 0; w < 6; w++) {
        u64 hv = h[w];
        for (int b = 0; b < 8; b++) out[w * 8 + b] = (uchar)(hv >> (8 * b));
    }
}

// Offset of (row, slot) in a back-reference array (1 u32 per slot).
static inline size_t bref_off(uint row, uint slot) {
    return (size_t)row * NR_SLOTS + slot;
}
// Offset of (row, slot) in a block working buffer (MAXB u32 blocks per slot).
static inline size_t blk_off(uint row, uint slot) {
    return ((size_t)row * NR_SLOTS + slot) * MAXB;
}

// ---- selftest helper: raw BLAKE2b output for every index ----
__kernel void equihash_hash(__global const uchar *header, __global uchar *out) {
    uint g = get_global_id(0);
    uchar dig[48];
    blake_hash(header, g, dig);
    __global uchar *o = out + (size_t)g * 48;
    for (int i = 0; i < 48; i++) o[i] = dig[i];
}

// ---- round 0: hash and bucket into table[0] by block 0 ----
// Writes the leaf index to back-reference array `bref0` and all 8 blocks to the
// block working buffer `blk0`.
__kernel void gen(__global const uchar *header,
                  __global uint *bref0,
                  __global uint *blk0,
                  __global uint *counts) {
    uint g = get_global_id(0);
    uchar dig[48];
    blake_hash(header, g, dig);

    for (uint i = 0; i < 2; i++) {
        uchar *h = dig + i * 24;            // one 192-bit hash (blocks 0..7)
        uint blocks[8];
        for (uint b = 0; b < 8; b++)
            blocks[b] = ((uint)h[3*b] << 16) | ((uint)h[3*b + 1] << 8) | h[3*b + 2];
        uint row = blocks[0] >> (CBL - ROW_BITS);
        uint slot = atomic_inc(&counts[row]);
        if (slot < NR_SLOTS) {
            bref0[bref_off(row, slot)] = g * 2 + i;
            __global uint *bb = blk0 + blk_off(row, slot);
            for (uint b = 0; b < 8; b++) bb[b] = blocks[b];
        }
    }
}

// ---- intermediate rounds r = 0..5: collide block r, write table[r+1] ----
// Reads blocks from `src_blk` (table r), writes the parent ref to `dst_bref`
// (table r+1's back-reference array) and the carried blocks to `dst_blk`.
__kernel void round_collide(__global uint *src_blk,
                            __global uint *dst_bref,
                            __global uint *dst_blk,
                            __global uint *counts,
                            uint r) {
    uint row = get_global_id(0);
    uint carry = 7u - r;                    // blocks r+1..7 carried forward

    uint cnt = counts[r * NR_ROWS + row];
    if (cnt > NR_SLOTS) cnt = NR_SLOTS;

    for (uint i = 0; i < cnt; i++) {
        __global uint *xi = src_blk + blk_off(row, i);
        for (uint j = i + 1; j < cnt; j++) {
            __global uint *xj = src_blk + blk_off(row, j);
            // Full block r (slot block 0) must match.
            if (xi[0] == xj[0]) {
                uint nblock = xi[1] ^ xj[1];
                uint nrow = nblock >> (CBL - ROW_BITS);
                uint nslot = atomic_inc(&counts[(r + 1) * NR_ROWS + nrow]);
                if (nslot < NR_SLOTS) {
                    dst_bref[bref_off(nrow, nslot)] = (row << (2 * SLOT_BITS)) | (i << SLOT_BITS) | j;
                    __global uint *d = dst_blk + blk_off(nrow, nslot);
                    for (uint b = 0; b < carry; b++) d[b] = xi[1 + b] ^ xj[1 + b];
                }
            }
        }
    }
}

// ---- final round: blocks 6 & 7 (table 6's two blocks) must match ----
__kernel void final_round(__global uint *src_blk,
                          __global uint *counts,
                          __global uint *sols,
                          __global uint *solcnt) {
    uint row = get_global_id(0);

    uint cnt = counts[6 * NR_ROWS + row];
    if (cnt > NR_SLOTS) cnt = NR_SLOTS;

    for (uint i = 0; i < cnt; i++) {
        __global uint *xi = src_blk + blk_off(row, i);
        for (uint j = i + 1; j < cnt; j++) {
            __global uint *xj = src_blk + blk_off(row, j);
            if (xi[0] == xj[0] && xi[1] == xj[1]) {
                uint idx = atomic_inc(solcnt);
                if (idx < MAX_SOLS) sols[idx] = (row << (2 * SLOT_BITS)) | (i << SLOT_BITS) | j;
            }
        }
    }
}

// Pick back-reference array `lvl` (recover touches every level).
static __global uint *pick(uint lvl, __global uint *b0, __global uint *b1,
                           __global uint *b2, __global uint *b3,
                           __global uint *b4, __global uint *b5,
                           __global uint *b6) {
    switch (lvl) {
        case 0: return b0;
        case 1: return b1;
        case 2: return b2;
        case 3: return b3;
        case 4: return b4;
        case 5: return b5;
        default: return b6;
    }
}

// ---- recover 128 leaf indices per candidate solution ----
// Walks the per-table back-reference arrays (bN); the block buffers aren't
// needed here.
__kernel void recover(__global uint *b0, __global uint *b1, __global uint *b2,
                      __global uint *b3, __global uint *b4, __global uint *b5,
                      __global uint *b6,
                      __global uint *sols, __global uint *out) {
    uint s = get_global_id(0);
    uint packed = sols[s];
    uint row = packed >> (2 * SLOT_BITS);
    uint sa = (packed >> SLOT_BITS) & SLOT_MASK;
    uint sb = packed & SLOT_MASK;

    // refs hold (row << SLOT_BITS | slot) into the current table level.
    uint refs[128];
    uint tmp[128];
    refs[0] = (row << SLOT_BITS) | sa;
    refs[1] = (row << SLOT_BITS) | sb;
    uint cnt = 2;

    // table[6] refs -> ... -> table[0] refs (six doublings: 2 -> 128).
    for (uint lvl = 6; lvl >= 1; lvl--) {
        __global uint *b = pick(lvl, b0, b1, b2, b3, b4, b5, b6);
        uint nc = 0;
        for (uint k = 0; k < cnt; k++) {
            uint rr = refs[k] >> SLOT_BITS;
            uint sl = refs[k] & SLOT_MASK;
            uint ref = b[bref_off(rr, sl)];
            uint prow = ref >> (2 * SLOT_BITS);
            uint pa = (ref >> SLOT_BITS) & SLOT_MASK;
            uint pb = ref & SLOT_MASK;
            tmp[nc++] = (prow << SLOT_BITS) | pa;
            tmp[nc++] = (prow << SLOT_BITS) | pb;
        }
        for (uint k = 0; k < nc; k++) refs[k] = tmp[k];
        cnt = nc;
    }

    // refs now index table[0]; its back-reference array holds the leaf index.
    for (uint k = 0; k < 128; k++) {
        uint rr = refs[k] >> SLOT_BITS;
        uint sl = refs[k] & SLOT_MASK;
        out[s * 128 + k] = b0[bref_off(rr, sl)];
    }
}
