# `src/miniz/` — miniZ's Equihash 192,7 solver (the blob this miner replays)

This directory holds miniZ's reverse-engineered CUDA solver and the recorded
launch traces the CUDA backend (`src/cuda.rs`) replays. The `.fatbin` + `.log`
files are **used at build time** (`include_bytes!`/`include_str!`); the `sass/`,
`cubins/`, and `functions.*` files are a **reference dump** for understanding and
maintaining the replay — they are not compiled in.

## Contents

| path | role |
|---|---|
| `equihash192_7.fatbin` | **embedded** — miniZ's solver, **7 arches** (sm_50/60/70/75/80/86/120); loaded via `cuModuleLoadData` |
| `configs/*.log` | **embedded** — driver-boundary launch recordings, one per bucket geometry |
| `configs/{manifest.json,README.md}` | recording provenance |
| `cubins/equihash192_7.<n>.<arch>.cubin` | reference — the 7 cubins extracted from the fatbin |
| `sass/all_<arch>.sass` | reference — full SASS per arch (sm_75/80/86/120; sm_50/60/70 are cubin-only) |
| `functions.{mangled,demangled}.txt` | reference — kernel symbol index |

### Architecture coverage

The fatbin was rebuilt (`fatbinary --image3 …`) from miniZ v2.5e3's per-arch
cubins (`~/code/miniz-dump/solver_all/equihash_192_7/arches/`) to widen GPU
support. The sm_80/86/120 cubins are **byte-identical** to the original embedded
fatbin, so the recorded replay is unchanged on those; the rest are additive.

| arch | kernels in cubin | replay |
|---|---|---|
| sm_50, sm_60 | 128 | full (all bucket configs) |
| sm_70, sm_75 | 18 | **reduced** — only a config whose kernels are present will replay |
| sm_80, sm_86, sm_120 | 57 | full — what the recordings were captured against |

CUDA 13.3 can only disassemble sm_75/80/86/120 (Volta/Pascal/Maxwell are
deprecated in the toolkit), so `sass/` covers those four; sm_50/60/70 ship as
cubins only but still load at runtime on their GPUs.

Regenerate the reference files:

```sh
cd src/miniz
mkdir -p cubins sass
( cd cubins && cuobjdump ../equihash192_7.fatbin --extract-elf all )
for a in sm_75 sm_80 sm_86 sm_120; do cuobjdump -sass -arch $a equihash192_7.fatbin > sass/all_$a.sass; done
grep -h "Function :" sass/all_*.sass | sed 's/^[[:space:]]*Function :[[:space:]]*//' | sort -u > functions.mangled.txt
c++filt < functions.mangled.txt > functions.demangled.txt

# rebuild the multi-arch fatbin from the dump's per-arch cubins:
D=~/code/miniz-dump/solver_all/equihash_192_7/arches
fatbinary --create equihash192_7.fatbin -64 \
  $(for a in 50 60 70 75 80 86 120; do echo --image3 kind=elf,sm=$a,file=$D/equihash_192_7.sm_$a.cubin; done)
```

## Algorithm & kernel pipeline

Equihash 192,7 (k=7): BLAKE2b expansion → Wagner shared-memory collision chaining
over 7 rounds → SHA-256 finalisation. The fatbin has **no PTX** (SASS only) and is
JIT-arch-selected by the driver at load.

`src/cuda.rs` replays one full pass in this fixed order (the 10 kernel families):

```
cleanup → digit_f → digit_1 → digit_2 → digit_3 → digit_4w → digit_5w → digit_6w
        → digit_l → sort_and_compress
```

| kernel | role | notes |
|---|---|---|
| `cleanup<64u>` | zero/reset table counters | regs≈6, no smem |
| `digit_f<…>` | BLAKE2b seed expansion → first table | **injection point** (see below) |
| `digit_1…digit_3` | early Wagner collide rounds | shared-mem `ATOMS.EXCH` chaining |
| `digit_4w/5w/6w` | wide later rounds (`w` = wide bucket) | |
| `digit_l<7u,…>` | last round → solution indices | writes the **counter + container** read back |
| `sort_and_compress` | canonicalise + SHA-256 | takes a 112-byte `SHA256_CTX` by value |

### Per-header injection (matches `cuda.rs::launch_pass`)

Only `digit_f` varies per header. The backend overwrites, in `digit_f`'s packed
arg buffer:
- `arg[0..64]` = the 64-byte BLAKE2b(192,7) midstate (`compress(header[0..128])`),
- `arg[96..100]` = header tail bytes `[136..140]`.

`digit_f`'s first two params are `ulonglong4, ulonglong4` (the 64-byte midstate),
so the first **64 bytes are by-value** and must not be pointer-rebased — this is
`cuda.rs::byval_prefix` (`digit_f` ⇒ 64, `sort_and_compress` ⇒ 112). Everything
else in every arg buffer is device pointers, rebased into our own arena at init.

### Bucket geometries (the 7 instantiations per digit)

Each `digit_*` ships **7 template instantiations** — one per bucket geometry
(table width × slots, e.g. `2048×16960`, `10000×…`, `12288×…`). They are the same
algorithm with different memory layout / occupancy constants. The repo bundles
**3** as recordings (`configs/`); `select_config` in `src/cuda.rs` picks the
highest-`table_capacity` one that fits free VRAM. The demangled signatures encode
each geometry — e.g. `digit_f<…,12288u,3392u,…>` vs `digit_f<…,2048u,16960u,…>`.
The solution readback is `digit_l` arg `[8..16]` = counter*, `[16..24]` = container*.

See `src/cuda.rs` for the full replay/rebase logic and
`../../pearl-dump/` for the same dump methodology applied to alpha-miner.
