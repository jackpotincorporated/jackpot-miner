//! Build script — Windows OpenCL import library only.
//!
//! On Windows, the `ocl`/`cl-sys` crates link `OpenCL` (`#[link(name =
//! "OpenCL")]`), which normally requires an `OpenCL.lib` import library from a
//! vendor OpenCL SDK. To avoid that build dependency, we generate a
//! vendor-neutral import library ourselves from `windows/OpenCL.def` (the list
//! of OpenCL exports) using the toolchain's own tools — `lib.exe` for MSVC,
//! `dlltool` for the GNU (MinGW) toolchain — and put it on the link search path.
//! The real `OpenCL.dll` (the ICD loader) is supplied at runtime by the GPU
//! driver, exactly like `libOpenCL.so.1` on Linux.
//!
//! Nothing here is needed on non-Windows targets (the system `libOpenCL` is used
//! directly) or when the `gpu` (OpenCL) feature is off — the script no-ops. The
//! CUDA driver / NVML are loaded at runtime via dlopen (see `src/dylib.rs`), so
//! they need no build-script support.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=windows/OpenCL.def");

    // Only Windows targets with the OpenCL backend need an import library.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    if env::var_os("CARGO_FEATURE_GPU").is_none() {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let def = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("windows/OpenCL.def");
    let target = env::var("TARGET").unwrap_or_default();
    let is_msvc = env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");

    let (mut cmd, tool) = if is_msvc {
        // lib.exe /def:OpenCL.def /out:OpenCL.lib /machine:X64
        let mut c = cc::windows_registry::find(&target, "lib.exe")
            .unwrap_or_else(|| Command::new("lib.exe"));
        c.arg("/nologo")
            .arg(format!("/def:{}", def.display()))
            .arg(format!("/out:{}", out_dir.join("OpenCL.lib").display()))
            .arg("/machine:X64");
        (c, "lib.exe")
    } else {
        // GNU/MinGW: dlltool -d OpenCL.def -l libOpenCL.a -m i386:x86-64
        let dlltool = ["x86_64-w64-mingw32-dlltool", "dlltool"]
            .into_iter()
            .find(|t| Command::new(t).arg("--version").output().is_ok())
            .unwrap_or("dlltool");
        let mut c = Command::new(dlltool);
        c.arg("-d")
            .arg(&def)
            .arg("-l")
            .arg(out_dir.join("libOpenCL.a"))
            .arg("-m")
            .arg("i386:x86-64");
        (c, "dlltool")
    };

    match cmd.status() {
        Ok(s) if s.success() => {
            // Resolve `#[link(name = "OpenCL")]` against the generated import lib.
            println!("cargo:rustc-link-search=native={}", out_dir.display());
        }
        Ok(s) => panic!("{tool} failed ({s}) generating the OpenCL import library from {}", def.display()),
        Err(e) => {
            // Tool not found: let `cargo check` (which doesn't link) still succeed;
            // a real build will fail at link with a clear "cannot find OpenCL".
            println!(
                "cargo:warning=could not run {tool} to generate the OpenCL import library ({e}); \
                 ensure the toolchain tools are on PATH (MSVC dev prompt, or mingw-w64). \
                 Linking the `gpu` feature will fail until then."
            );
        }
    }
}
