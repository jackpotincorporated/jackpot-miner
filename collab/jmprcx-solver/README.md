# miniz-solver-rs

Basic Rust program that **uses the extracted miniZ Equihash 192,7 GPU solver**.
It loads the captured CUDA fatbin (`../miniz-dump/solver_192_7/equihash192_7.fatbin`)
through the CUDA Driver API (raw FFI to `libcuda`, no external crates) and drives
its kernels on the GPU.

## Build & run

```sh
cargo build --release
./target/release/miniz-solver                 # load + enumerate all 57 kernels
./target/release/miniz-solver --launch        # also execute a real solver kernel
./target/release/miniz-solver --round0        # replay round 0 (digit_f) with a captured midstate
./target/release/miniz-solver /path/to.fatbin # use a different fatbin
```

Requires an NVIDIA GPU + driver (`/usr/lib/libcuda.so`). The fatbin contains
`sm_80`/`sm_86`/`sm_120` cubins; the driver auto-picks the one for your GPU.

## What it does

- `cuInit` → context on GPU#0
- `cuModuleLoadData` on the raw fatbin (magic `0xBA55ED50`)
- `cuModuleEnumerateFunctions` + `cuFuncGetName` + `cuFuncGetAttribute`:
  lists every kernel with regs / shared / local / max-threads and labels the
  Wagner `n=192,k=7` pipeline:
  `digit_f` (round 0: BLAKE2b + bucketing) → `digit_1..3`, `digit_4w/5w/6w`
  (rounds 1–6) → `digit_l` (round 7: solution recovery) → `sort_and_compress`.
- with `--launch`: allocates a device buffer and launches the real
  `cleanup<64>(void*, uint)` kernel, then `cuCtxSynchronize`.
- with `--round0`: drives the real **round 0** (`digit_f`) — allocates the four
  buffers at their template sizes, launches the exact runtime variant
  (grid=65536, block=256) with a BLAKE2b midstate captured from a live job, and
  reads back the bucket counters. Verified output: **33,554,432 = 2^25** entries
  bucketed into 12288 buckets (the correct 192,7 initial-entry count).
- with `--replay [rec.log]`: **runs the entire solver** — parses a recorded pass
  (`recording.log`), allocates one arena, rebases every device pointer, and
  executes all 10 kernels (`cleanup → digit_f → digit_1..6 → digit_l →
  sort_and_compress`). All kernels complete; extracts a 128-index candidate.
- with `--header <hex>`: computes a BLAKE2b(192,7) midstate from a 140-byte
  header, injects it, and runs the full pipeline (mint a new job).
- with `--selftest`: BLAKE2b-512 known-answer test (RFC 7693) — PASS.
- with `--verify-share`: verify a real pool-accepted share (BLAKE2b + Wagner) — VALID.
- with `--solve`: **the complete solver** — inject a known header's midstate+tail,
  run the GPU pipeline, and harvest a solution from the container that the verifier
  accepts. Reproducibly prints `SOLUTION HARVESTED FROM GPU — VALID ✓`.

See `../miniz-dump/solver_192_7/ORCHESTRATION.md` for the full pipeline + recovery.

### Status (honest)
- **Pipeline: complete.** All 10 kernels run standalone; round 0 verified bit-exact
  (2^25 entries). Faithful end-to-end replay of miniZ's 192,7 solver.
- **Hash model + verification: SOLVED.** Captured live stratum (plaintext) via a
  logging relay; a real pool-accepted share verifies exactly under
  `hash(i) = BLAKE2b(header‖LE32(i/2), person="ZcashPoW"+LE32(192)+LE32(7),
  digest=48)[(i%2)*24..]`. `--verify-share` reproduces VALID ✓ (192/192 zero bits,
  all 7 Wagner levels) in Rust. So `--selftest`, `blake2b.rs`, `verify.rs` and the
  solution decoder are all proven against ground truth.
- **Complete (`--solve`).** Container = 128 consecutive u32 indices at offset 0;
  the midstate is textbook BLAKE2b-after-128B and the digit_f `uint` is the 4
  varying header-tail bytes (nonce[28..31]; nonce[20..27] are constant 0). So:
  `header → midstate+tail → GPU pipeline → container[0..128] → VALID solution`,
  reproducibly. The miniZ Equihash 192,7 solver is fully reverse-engineered.

## What it does NOT do (scope)

It does **not** mine or produce valid Equihash solutions. A working solver also
needs miniZ's host orchestration, which is not part of the extracted kernels:

- exact device-buffer sizing per round (the kernels' template/array dims give the
  bucket geometry, e.g. `uint4[180][6656][32]`, but the host owns allocation)
- the precise `digit_f → digit_1..6 → digit_l → sort_and_compress` launch
  sequence with the correct grid/block dims and shared-mem config per round
- BLAKE2b midstate setup from the block header + nonce, and the `equi<...>` /
  `ScontainerReal192` struct layouts passed between kernels

That host logic lives in miniZ's encrypted blob. Reconstructing it (from the SASS
in `../miniz-dump/solver_192_7/equihash192_7.sm_120.sass` plus the kernel
signatures in `kernels_demangled.txt`) is the next step toward a standalone miner.

## Files
- `src/cuda.rs` — minimal CUDA Driver API FFI bindings
- `src/main.rs` — loader / enumerator / launch demo
- `build.rs` — links `libcuda`
