# `src/ethash/` — miniZ's Ethash solver (CUDA backend, in progress)

Adds the **Ethash family** (ethash · etchash · ubqhash · reth — same 15-kernel
solver) to the miner, alongside the Equihash 192,7 backend in `src/cuda.rs`.
Built from miniZ v2.5e3's extracted kernels
(`~/code/miniz-dump/solver_all/ethash/`).

> **Status: GPU search WORKING (validated end-to-end on an RTX 5080).**
> The DAG build + hashimoto search now run on the GPU by replaying a captured
> miniZ etchash trace (`recording.log`): `../ethash.rs` builds the DAG (539
> `digitf` launches), injects `d_header`/`d_target`, runs the `equihash` search
> kernel, and decodes `Search_results`. `ethash::selftest` proves it without a
> pool — it picks a hard target on the CPU, runs the GPU search, and confirms the
> GPU returns that nonce with a mix that passes the full CPU verifier (same
> captured cache). The CPU verifier (`verify.rs`) is pinned by a full **hashimoto
> KAT** vs go-ethereum, and the **stratum loop** (`stratum.rs`) wires search →
> verify → submit. `--algo ethash|etchash` is live in `main.rs`.
>
> Run the GPU self-test:
> ```sh
> CUDA_VISIBLE_DEVICES=0 cargo test --features ethash \
>   ethash::tests::gpu_selftest -- --ignored --nocapture
> ```
> It needs the ≈67 MB light-cache sidecar the recording references
> (`pearl-dump/ethash.rec.htod00.bin`) — regenerate it with a fresh capture (see
> *How the trace was captured*). The bundled `recording.log` is pinned to one
> etchash epoch; re-capture to mine a newer epoch (see *Caveats*).

## Contents

| path | role |
|---|---|
| `ethash.fatbin` | **embeddable** — 8-arch solver (sm_50/60/61/70/75/80/86/120) |
| `cubins/ethash.sm_*.cubin` | reference — per-arch cubins |
| `sass/ethash.sm_{75,80,86,120}.sass` | reference — SASS (sm_50/60/61/70 are cubin-only on CUDA 13.3) |
| `functions.demangled.txt` | reference — kernel index |
| `../ethash.rs` | the backend (feature `ethash`, default-off) — recording parser, DAG-build + search replay, `Search_results` decode, `selftest`. |
| `recording.log` | **captured miniZ etchash launch trace** (allocs, constants, 539 `digitf` + the `equihash` launch with packed args). Drives the replay. |
| `verify.rs` | **CPU light verifier** — inline Keccak (KAT-tested) + mkcache + hashimoto_light + boundary check. `Light::from_raw` builds from the captured cache. Gates shares before submit. |
| `stratum.rs` | **Ethash stratum** (EthereumStratum/1.0.0) + difficulty→target + the mining loop (search → verify → submit). |

## How Ethash differs from Equihash (why it's a new subsystem)

Equihash 192,7 replays a **fixed kernel pass per header** — stateless, all inputs
in the kernel arg buffers. Ethash is **DAG + search**:

- `digitf<…>` — builds the **DAG** for the current epoch (multi-GB, regenerated
  every ~30k blocks). Runs once per epoch, not per nonce.
- `equihash<…>` — the per-nonce **hashimoto search** kernel (the name is miniZ's;
  it's Dagger-Hashimoto, not Equihash). Signature is `(unsigned long start_nonce,
  Search_results*)` — so **the header and target are NOT kernel args**; miniZ
  pushes them into `__constant__` memory via `cuModuleGetGlobal` + `cuMemcpyHtoD`.

How the replay handles this:
1. **Two cadences** — `digitf` is replayed once at `new()` to build the DAG;
   `equihash` runs per nonce batch in `search()`.
2. **Constant-memory injection** — the trace captures every `cuModuleGetGlobal`
   symbol and the writes to them. The replay writes `d_dag` (rebased to our DAG
   buffer), `d_dag_size`/`i_dag_size`/`d_maxmem`/`d_multi` verbatim, and per-search
   `d_header` (32 B) + `d_target` (8 B = boundary top-64 bits, little-endian).
3. **`Search_results` readback** — recovered layout: `mix[i]` (32 B) at offset
   `i*32`, matched nonce (u64) at `0x100 + i*8`, 8 slots. We zero the buffer before
   each launch, decode non-empty slots, and CPU-verify each.

## Recovered ABI (from `recording.log`)

| kernel | grid / block | args (packed from kernelParams) |
|---|---|---|
| `digitf<1099511627778,256,2>` | 512 / 256, ×539 | `(u32 start, hash64* dag, u64 dag_rows128, hash64* cache, u32 cache_rows, …)` — only `start` varies (steps by 512·256) |
| `equihash<5806517278900176776,…>` | 399360 / 64 | `(u64 start_nonce, Search_results* out)` |

`full_size` (hashimoto, 64-B rows) = `d_dag_size` × 128; cache rows = `d_dag` light
buffer / 64. The DAG buffer base lives in the `d_dag` constant (rebased on replay).

## How the trace was captured

miniZ resolves the driver API by `dlopen("libcuda.so.1")` + `dlsym`, so a plain
`LD_PRELOAD` of a `cuLaunchKernel` shim is bypassed. Two preloads defeat this:
`pearl-dump/dlhook.so` redirects the libcuda `dlopen` to `pearl-dump/cuhook.so`
(linked against libcuda), so miniZ's `dlsym(handle, "cuXxx")` finds our wrappers
first and falls through to real libcuda for the rest. `cuhook.so` logs allocs,
`cuModuleGetGlobal` symbols, HtoD (tagging constant writes by symbol, dumping the
≈67 MB light cache to a sidecar), and — packing the `kernelParams` array by the
demangled signatures — every kernel's args.

```sh
cd pearl-dump && gcc -shared -fPIC -O2 -o cuhook.so cuhook.c -ldl -lcuda
gcc -shared -fPIC -O2 -o dlhook.so dlhook.c -ldl
H=$PWD; ZCL_TRACE=$H/ethash.rec CUHOOK_SO=$H/cuhook.so \
  LD_PRELOAD="$H/dlhook.so:$H/cuhook.so" CUDA_VISIBLE_DEVICES=0 \
  miniZ --par etchash --url 0x<addr>.w@etc.2miners.com:1010 --pass x --nocolour
cp ethash.rec ../src/ethash/recording.log   # the sidecar stays in pearl-dump/
```

The whole loop is end-to-end: GPU `search()` → `verify::verify` (rejects anything
that doesn't meet the full 256-bit target) → `submit`.

## Caveats

- **Epoch-pinned.** `recording.log` (and its cache sidecar) capture one etchash
  epoch's DAG-build sequence; the replay mines that epoch. Re-capture to follow a
  new epoch — the `digitf` loop isn't yet generalized to arbitrary `dag_size`.
- **Cache sidecar not committed** (67 MB). The recording names its path; regenerate
  by re-capturing on the host.
- Same fatbin covers **etchash/ubqhash/reth**; embed those the same way for distinct
  `--algo` targets.

## Kernels & arches

- Families: `digitf` (DAG build) · `equihash` (search). Many `equihash<…>`
  instantiations = per-epoch/difficulty constant sets.
- Arches in the fatbin: sm_50/60/61/70/75/80/86/120 (SASS for 75/80/86/120).
- Same solver covers **etchash/ubqhash/reth** (per `RE_INDEX.md`); embed those
  fatbins the same way if you want them as distinct `--algo` targets.

Regenerate the fatbin:
```sh
D=~/code/miniz-dump/solver_all/ethash/arches
fatbinary --create src/ethash/ethash.fatbin -64 \
  $(for a in 50 60 61 70 75 80 86 120; do echo --image3 kind=elf,sm=$a,file=$D/ethash.sm_$a.cubin; done)
```
