# Building jackpotminer on Windows

The miner builds on Windows with **no external GPU SDKs**. The CUDA driver and
NVML are loaded at runtime (`dlopen`/`LoadLibrary`), and the OpenCL import
library is generated at build time from `windows/OpenCL.def` — so you don't need
the CUDA Toolkit, an OpenCL SDK, or any vendor libraries to compile. The whole
codebase (miner + the `jackpotminer-config` GUI) is verified to compile for
`x86_64-pc-windows-*`.

## Prerequisites

- **Rust** (https://rustup.rs). The default target is `x86_64-pc-windows-msvc`.
- **MSVC build tools** — Visual Studio 2019/2022 or the standalone *Build Tools
  for Visual Studio* with the **“Desktop development with C++”** workload (gives
  `link.exe`, `lib.exe`, and the Windows SDK).
  - Build from a **“x64 Native Tools Command Prompt for VS”** (or any shell where
    the MSVC environment is active) so `lib.exe` is found. `cargo` also locates
    it automatically via the registry in most cases.
- *(Alternative toolchain)* the GNU target `x86_64-pc-windows-gnu` works too,
  with **MinGW-w64** on `PATH` (provides `dlltool` and the linker).

No CUDA Toolkit and **no OpenCL SDK** are required.

## Build

```powershell
:: Default: OpenCL + CUDA backends + the GUI config tool
cargo build --release

:: Miner only (no GUI), both GPU backends
cargo build --release --no-default-features --features gpu,cuda

:: OpenCL only (AMD / Intel / NVIDIA)
cargo build --release --no-default-features --features gpu

:: NVIDIA only — needs nothing external at build time (CUDA is dlopen'd)
cargo build --release --no-default-features --features cuda
```

Outputs: `target\release\jackpotminer.exe` (and `jackpotminer-config.exe` with
the default features).

### How the OpenCL build dependency is avoided

`ocl`/`cl-sys` link `OpenCL` (`#[link(name = "OpenCL")]`), which normally needs
an `OpenCL.lib` from a vendor SDK. Instead, `build.rs` generates a
vendor-neutral import library from `windows/OpenCL.def`:

- **MSVC:** `lib.exe /def:windows\OpenCL.def /out:OpenCL.lib /machine:X64`
- **GNU:** `dlltool -d windows/OpenCL.def -l libOpenCL.a -m i386:x86-64`

and puts it on the link search path. The import library only forwards to
`OpenCL.dll`, which the GPU driver provides at runtime. If the toolchain tool
isn’t on `PATH`, `build.rs` prints a warning and the link step fails with a clear
message (compilation/`cargo check` still works).

## Distribution

Statically link the MSVC C runtime so users don’t need the VC++ redistributable:

```powershell
set RUSTFLAGS=-C target-feature=+crt-static
cargo build --release
```

## Runtime dependencies (on the mining machine)

- **`OpenCL.dll`** — the ICD loader, installed with any GPU driver (AMD/NVIDIA/
  Intel). Required for the OpenCL backend.
- **`nvcuda.dll`** + **`nvml.dll`** — installed with the NVIDIA driver. Loaded
  on demand for the CUDA backend; absent on AMD-only machines, where the miner
  simply reports no CUDA devices.
- The **VC++ runtime**, unless you built with `+crt-static`.

A `cuda`-enabled binary still starts on a machine with no NVIDIA driver.

## Building Windows binaries without a Windows machine

### GitHub Actions (recommended)

The repo ships [`.github/workflows/windows.yml`](.github/workflows/windows.yml),
which builds on `windows-latest` (MSVC + `lib.exe` already on `PATH`), runs the
tests, statically links the CRT, and uploads the `.exe`s as artifacts — so you
get Windows binaries with no Windows machine. It runs on pushes to `master`, on
PRs, and via **Run workflow** (manual dispatch).

### Cross-compile from Linux (GNU target)

```bash
rustup target add x86_64-pc-windows-gnu
sudo pacman -S mingw-w64           # or your distro's mingw-w64 (gives dlltool + linker)
cargo build --release --target x86_64-pc-windows-gnu --no-default-features --features gpu,cuda
```

The MSVC target can’t be linked from Linux. The GUI config tool (`eframe`) is
easiest to build natively on Windows.

## Status / caveats

- **Compilation for Windows is verified** here via `cargo check
  --target x86_64-pc-windows-gnu` (default features, and `gpu,cuda`).
- The OpenCL **import-library linking** uses the standard `lib.exe`/`dlltool`
  technique; validate it with an actual Windows (or MinGW cross) build, which
  needs those tools present.
- `relaunch_in_terminal` (reopen-in-a-terminal on GUI launch) is Linux-only;
  harmless on Windows, where double-clicking a console binary already opens a
  console.
