# jackpotminer

A GPU-accelerated **Equihash 192,7** miner written in Rust, for ZClassic (ZCL)
and other coins that use the same proof-of-work.

It implements the full mining pipeline:

- **Stratum client** — Zcash-style pool protocol (subscribe / authorize /
  set_target / notify / submit), as used by nheqminer / EWBF.
- **Personalised BLAKE2b** — `"ZcashPoW" || LE32(192) || LE32(7)`, 48-byte
  digest.
- **Equihash 192,7 solver** — Wagner's algorithm over eight 24-bit collision
  blocks, producing 128-index / 400-byte solutions.
- **Full GPU solver** — the entire solve runs on the GPU: BLAKE2b generation,
  all seven collision rounds over a bucketed hash table, and back-reference
  recovery of the leaf indices. Only a few candidate solutions return to the
  host for verification. Two interchangeable backends: **OpenCL** (default, our
  own kernels) and **CUDA** (driver API, replaying miniZ's reverse-engineered
  fatbin).
- **Solution verification & encoding** — full spec-compliant `IsValidSolution`
  plus compact-array encode/decode and the block PoW (double-SHA256) target
  check.

## Status

The solver, BLAKE2b personalisation, solution encoding and verification are
validated end-to-end. `--selftest` runs a full CPU solve and a full GPU solve on
the same header and confirms the GPU recovers the same verified solution set;
the OpenCL BLAKE2b kernel is also checked byte-for-byte against the CPU. The GPU
solver finds the expected ~2 solutions per nonce, matching the Equihash 192,7
theoretical rate.

The Stratum layer implements the common nheqminer-style protocol and is covered
by a mock-pool integration test; some pools differ in nonce/solution framing and
may need small adjustments (`RUST_LOG=debug` shows the raw traffic).

### Performance & memory

Measured on a single desktop RTX 5080 (default clocks, no overclock; `--benchmark
30`, ~1.8 solutions per nonce):

| Backend (`--backend`) | ms/solve | Sol/s |
|---|---:|---:|
| `cuda`   | **~20**  | **~92** |
| `opencl` | ~311     | ~5.9    |

CUDA is ~15× faster than the OpenCL backend on the same card. One worker thread
per device shares the pool connection and a global nonce counter, so a second GPU
(`--devices 0,1`) scales the aggregate Sol/s roughly linearly. (Enabling the
clock/power tuning — `--auto-tune`/`--gpu-clock-offset`, needs root — pushes both
backends higher still; the table above is untuned.)

The **CPU solver** (`--cpu`) is an AVX2-tuned Wagner implementation (xenoncat-style
packed 32-byte slots + dense collision keys, single-pass bucketed XOR, `pshufb`
round-0 repack) running ~**3 s/solve** on a 24-core host (~0.4 Sol/s), finding
the expected ~2 solutions per nonce. Like the GPU it bounds each collision bucket
(`--cpu-clamp`, default 32) — required because the naive unclamped algorithm's
degenerate collisions explode in the last rounds; `--cpu-clamp 0` selects the
exact solver (may OOM on dense headers).

The CUDA backend is profile-driven (Nsight Compute). Key optimisations over the
OpenCL backend:

- **Warp-per-bucket collision/final kernels** — each 32-lane warp cooperatively
  loads its bucket into shared memory and parallelises the pair search, fixing
  the one-thread-per-bucket version's intra-warp load imbalance (collision
  rounds ~174 ms → ~97 ms, final ~5 ms → ~1 ms).
- **16-byte-aligned `uint4` slot stores** — hash-table slots for tables 1-6 are
  padded to a multiple of 4 words so the ref+blocks are written with aligned
  `uint4` transactions. The earlier odd-sized slots (7/6/5/3 words) caused
  misaligned scalar writes; fixing this cut the collision rounds to ~68 ms
  (~120 ms → ~92 ms per solve).
- **Pinned (page-locked) host buffers** for the device→host result readback.

Things that were tried and *didn't* help on this hardware (measured, reverted):
warp-shuffle comparisons and `gen` register-capping (`__launch_bounds__`) — the
fast rounds are not compare-bound, and `gen` needs its registers. Warp-aggregated
atomics don't apply because the output-bucket atomics target data-dependent
random addresses (no within-warp sharing). The remaining cost is `gen` (~21 ms,
register/compute bound) and the scattered hash-table read/write latency that is
fundamental to bucketing. The OpenCL backend keeps the simpler kernels.

The hash table uses 2²¹ buckets × 32 slots (bucket cap 2× the mean occupancy, so
overflow is rare and essentially all solutions survive). The bulky 24-bit
collision blocks are only needed during the round that consumes them, so they
live in two ping-pong working buffers; only a small per-table back-reference
array (1 word/slot) is kept resident for solution recovery. That brings the
footprint to about **6 GB of VRAM per GPU** (down from ~11 GB when all seven
block tables were resident), so **8 GB cards work**. The backend reads the
device's VRAM at startup and warns if a card is too small; `ZCL_OPENCL_ROWBITS`
overrides the bucket count for experimentation (values below 21 usually find no
solutions). There is still optimisation headroom relative to mature miners like
lolMiner/EWBF.

The above describes the OpenCL backend (`kernels/equihash.cl`). The CUDA backend
takes a different route: instead of our own kernels it drives miniZ's
reverse-engineered Equihash 192,7 solver — see "CUDA backend" below.

## Build

Requirements: a Rust toolchain and, for the OpenCL backend, the OpenCL ICD
loader (`libOpenCL` — e.g. `ocl-icd-opencl-dev` on Debian/Ubuntu; the NVIDIA and
AMD drivers also ship it). The CUDA driver and NVML are **`dlopen`'d at runtime**
(see `src/dylib.rs`), so the `cuda` feature needs no NVIDIA toolkit or libs to
build, and a `cuda`-enabled binary still builds and starts on hosts without an
NVIDIA driver — it simply reports no CUDA devices. The fatbin and launch trace
the CUDA backend drives are embedded, so no `nvcc` is required either.

```bash
cargo build --release                          # default: OpenCL + CUDA + GUI config tool
cargo build --release --no-default-features --features gpu,cuda  # miner only, both GPU backends
cargo build --release --no-default-features --features gpu        # OpenCL only (AMD/Intel/NVIDIA)
cargo build --release --no-default-features --features cuda       # CUDA only
cargo build --release --no-default-features                       # CPU-only (no GPU)
```

### Portable / distributable builds

The miner's only runtime dependencies are the C library and the OpenCL ICD loader
(`libOpenCL.so.1`, present wherever a GPU driver is); CUDA/NVML are loaded on
demand. So the main compatibility risk when shipping a Linux binary is the
**glibc version** it was built against — not the GPU libraries. To build one that
runs on older distros, compile against an old glibc in a container:

```bash
./build-portable.sh          # → dist/jackpotminer   (Docker, or ENGINE=podman)
```

This links against Debian bullseye's glibc 2.31 (runs on most Linux from ~2020
on) and yields a single miner that drives both AMD (OpenCL) and NVIDIA (CUDA)
cards. See `Dockerfile.portable`.

A fully *static* GPU binary isn't possible: the OpenCL/CUDA driver libraries are
glibc-only and must load at runtime. For a zero-dependency binary that runs
anywhere, build the **CPU-only** miner against musl:

```bash
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl --no-default-features
```

### CUDA backend (miniZ fatbin replay)

`--features cuda` (selectable with `--backend cuda`) does **not** compile its own
kernels. It loads miniZ's reverse-engineered Equihash 192,7 solver fatbin
(embedded from `src/miniz/equihash192_7.fatbin`) through the CUDA Driver API and
replays its exact 10-kernel Wagner pipeline from an embedded launch trace,
injecting the BLAKE2b midstate + header tail into `digit_f` per solve. Recovered
indices are verified by the project's own `equihash::filter_candidates`, so only
solutions genuinely valid for the header are ever returned. Needs an NVIDIA GPU
whose arch matches the fatbin's cubins (sm_80/sm_86/sm_120). See
`collab/jmprcx-solver/` for the reverse-engineering work behind it.

**Multi-config (VRAM auto-selection).** miniZ ships several bucket geometries with
different memory footprints; `src/miniz/configs/` bundles a recorded launch trace
for each. At startup the backend reads free VRAM and picks the **highest-capacity
config that fits** (higher capacity ⇒ fewer dropped collisions ⇒ better yield):

| config | min free VRAM | table capacity |
|---|---|---|
| `12288x3392` | ~11 GB | 41.7M (best) |
| `10000x4032` | ~5.5 GB | 40.3M |
| `2048x16960` | ~5 GB | 34.7M |

So it runs on cards from ~6 GB up, using the largest config the card can hold.
Override the choice with `ZCL_CUDA_CONFIG=<name>` (e.g. for testing). If no config
fits, it fails at init with a clear message. See `src/miniz/configs/README.md`.

### GPU tuning & efficiency (NVML)

The CUDA backend tunes clocks/power via NVML (the same knobs LACT exposes). **By
default** it locks clocks and power to the card maximum (peak hashrate). Passing
**any** tuning flag switches to manual mode — only the knobs you specify are
applied, so a clock offset isn't defeated by a hard clock lock:

| flag | effect |
|---|---|
| `--power-limit <W>` | board power cap (best Sol/W lever) |
| `--gpu-clock <MHz>` / `--mem-clock <MHz>` | hard-lock the core / memory clock |
| `--gpu-clock-offset <MHz>` / `--mem-clock-offset <MHz>` | signed V/F **offsets** (LACT-style; undervolt/overclock the curve) |
| `--auto-tune` | sweep the core offset at startup for the **fastest stable solve rate** |
| `--no-gpu-tune` | touch nothing (let LACT or `nvidia-smi` own the GPU) |

**`--auto-tune`** optimizes each card for speed automatically: it raises power to
the cap, leaves clocks free to boost, then sweeps the core clock offset upward
(in +45 MHz steps), measuring solve throughput on a test header at each step. It
keeps the best and **stops at the first instability** (a kernel error or the card
no longer producing valid solutions), then locks in that offset. Takes ~30 s at
startup, runs per card, needs root, and is restored on exit. It overrides
`--gpu-clock-offset`.

A typical efficiency setup combines a power cap with a positive core offset and a
negative memory offset, letting the card boost on a shifted curve under the cap:

```bash
sudo ./jackpotminer --url pool:port -u addr --devices all \
    --power-limit 250 --gpu-clock-offset 250 --mem-clock-offset -500
```

These are **privileged** (run as root); without it you get a one-line warning and
the card free-runs. Settings (clocks, power, offsets) are **restored to defaults
on clean shutdown**. The per-card stats line shows live `Sol/s`, board `W`, and
`Sol/W` (reading power is unprivileged) so you can tune efficiency directly.

> Re: LACT — its NVIDIA support drives these same NVML offsets, but LACT has no
> stable CLI to set them (a root daemon applies a saved profile), so the miner
> sets them directly via NVML. If you'd rather LACT own the GPU, run the miner
> with `--no-gpu-tune` and configure clocks in LACT.

## Usage

```bash
# List devices (and the default "mixed" backend's combined index list)
./target/release/jackpotminer --list-devices

# Mine on one GPU
./target/release/jackpotminer \
    --url stratum+tcp://zcl.pool.example:3032 \
    --user <ZCL-address>.<worker> \
    --pass x \
    --devices 0

# Mine on multiple GPUs (one worker thread each)
./target/release/jackpotminer --url ... --user ... --devices 0,1
./target/release/jackpotminer --url ... --user ... --devices all

# Default backend is "mixed": NVIDIA cards run on CUDA, AMD/Intel on OpenCL —
# so an AMD + NVIDIA rig just works. --devices indexes the combined list from
# --list-devices. Pin a single backend for every card with:
./target/release/jackpotminer --url ... --user ... --backend opencl   # all via OpenCL
./target/release/jackpotminer --url ... --user ... --backend cuda     # NVIDIA only

# Force the CPU backend
./target/release/jackpotminer --url ... --user ... --cpu

# Benchmark concurrently across the chosen devices (no pool), 30 solves each
./target/release/jackpotminer --benchmark 30 --devices 0,1

# Correctness self-test (CPU solve vs GPU solve + kernel comparison)
./target/release/jackpotminer --selftest
```

Key flags: `--url` (defaults to `stratum+tcp://zcl.jackpot.tools:3333` when
unset), `--port` (fallback when the URL omits one), `--user` (`-u`),
`--pass` (`-p`), `--backend` (`mixed`/`opencl`/`cuda`), `--devices` (e.g. `0,1` or `all`),
`--device` (`-d`, for single-device benchmark/debug), `--threads` (`-t`), `--cpu`,
`--benchmark N`, `--list-devices`, `--selftest`. When mining in a terminal a live
ratatui **dashboard** is shown by default (per-card Sol/s, power, temperature,
Sol/W, shares; a network Sol/s panel (horizontal bars) showing the last hour /
day / 3 / 7 / 30 / 60 / 90 days; per-GPU graphs; and a log pane;
`q`/`Esc`/`Ctrl-C` to quit); it falls back to periodic log lines
when there's no TTY (headless/piped/systemd) and `--no-tui` forces log output.
`--job-timeout <secs>` pauses mining when the pool
goes silent (default 300; 0 disables). Set `RUST_LOG=debug` to see the raw Stratum
traffic (ignored under `--tui`, which captures logs into its pane at info level).

## Layout

| File | Purpose |
|------|---------|
| `src/params.rs` | Equihash 192,7 constants |
| `src/blake.rs` | Personalised BLAKE2b base state |
| `src/equihash.rs` | CPU solver, verifier, compact-array encode/decode, candidate filter |
| `src/stratum.rs` | Stratum pool client |
| `src/miner.rs` | Mining loop, nonce iteration, target check, submit, backend dispatch |
| `src/gpu.rs` | OpenCL host glue: buffers, kernel orchestration |
| `kernels/equihash.cl` | OpenCL kernels: BLAKE2b, collision rounds, recovery |
| `src/cuda.rs` | CUDA backend: drives miniZ's embedded fatbin via the driver API (FFI to `libcuda`) |
| `src/miniz/` | Embedded miniZ fatbin + recorded launch trace the CUDA backend replays |
| `build.rs` | Links `libcuda` for the `cuda` feature |

## Disclaimer

For use only on hardware and pools you are authorised to use, and where
cryptocurrency mining is legal. Mining consumes significant power.
