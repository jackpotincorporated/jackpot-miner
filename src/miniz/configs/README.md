# miniZ Equihash 192,7 — per-config recordings (for multi-config integration)

`recording.log` (in the project root) is **unchanged** — it is the single
12288-bucket config. This folder holds one **canonical recording per distinct
solver config**, plus `manifest.json`, so a parent project can do miniZ-style
runtime config selection.

## Configs (3 reachable on sm_120; the fatbin has 7 total — the other 4 are
sm_80/sm_86 tunings, not selectable on Blackwell)

| config_id | buckets × slots | table capacity | headroom¹ | min VRAM² | recording |
|---|---|---|---|---|---|
| 11489098945649464158 | 2048 × 16960 | 34.7M | 1.035 | ~3.2 GB | config_2048x16960.log |
| 8970502565599698964  | 10000 × 4032 | 40.3M | 1.202 | ~3.5 GB | config_10000x32.log |
| 10872974046970623180 | 12288 × 3392 | 41.7M | 1.242 | ~4.5 GB | config_12288x32.log |

¹ capacity ÷ 2²⁵ entries; higher = fewer dropped collisions = higher solution yield.
² smallest reported-free VRAM that selected this config (from a cuMemGetInfo sweep).

**Selection rule (what miniZ does):** pick the config with the highest
`table_capacity` whose `min_vram` fits the GPU's free VRAM. Raw pipeline speed is
~identical across configs (~14 ms/pass); the choice is a memory↔yield tradeoff.

## Recording format (line-based, easy to parse)

```
[alloc] <size> bytes @ 0x<base>                         # every cuMemAlloc (device VA + size)
[REC] <mangled_kernel> g=gx,gy,gz b=bx,by,bz sh=<bytes> sz=<n> arg=<hex>   # every cuLaunchKernel
```
One pipeline pass = the launches from the first `cleanup` through the next
`sort_and_compress` (10 kernels): `cleanup → digit_f → digit_1 → digit_2 →
digit_3 → digit_4w → digit_5w → digit_6w → digit_l → sort_and_compress`.
`arg` is the full packed kernel-argument buffer (passed by miniZ via
`cuLaunchKernel`'s `extra` / `CU_LAUNCH_PARAM_BUFFER_POINTER`).

## How to run a config (already implemented in `../src/replay.rs`)

1. Parse the recording (allocs + the 10-kernel pass).
2. Allocate a device buffer for **each alloc the pass references** (sum is small:
   3.2–3.7 GB for the small configs; the 12288 config uses one ~16 GB arena but
   only the low ~7 GB is touched).
3. For every launch, **rebase each 8-byte device pointer** in `arg` to its owning
   buffer: `mine = my_base + (ptr − orig_base)`. (Skip the by-value prefix:
   64 B for `digit_f` = BLAKE2b midstate, 112 B for `sort_and_compress`.)
4. For kernels with `sh>0`, call
   `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, sh)` before launch.
5. Launch each kernel (same grid/block/shared) via `extra`/BUFFER_POINTER.

## Solving an arbitrary header (any config)

Inject into `digit_f`'s arg buffer:
- `arg[0..64]` = BLAKE2b midstate = `compress(header[0..128])`
  (person = `"ZcashPoW" || LE32(192) || LE32(7)`, digest 48).
- `arg[96..100]` = `header[136..140]` (the 4 varying nonce tail bytes;
  nonce[20..27] are constant 0 in miniZ's layout).

Then read 128 consecutive `u32` from `digit_l`'s container (3rd pointer arg) — that
is a canonical solution. Verify: `hash(i)=BLAKE2b(header‖LE32(i/2))[(i%2)*24..]`,
XOR of all 128 = 0 over 192 bits + all 7 Wagner levels. See `../src/{blake2b,verify}.rs`.

## manifest.json
Machine-readable: per config — `config_id`, `bucket_count`, `slots_per_bucket`,
`table_capacity`, `headroom`, `min_vram_gb`, `ref_buffers`, `ref_footprint_bytes`,
and `stages[]` (each: `kernel`, full mangled `name`, `grid`, `block`, `shmem`,
`argsize`). Load it to drive launches without re-parsing the `.log`.
