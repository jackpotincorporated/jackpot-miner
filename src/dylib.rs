//! Tiny runtime dynamic-library loader for the optional GPU vendor libraries.
//!
//! The CUDA driver (`libcuda`) and NVML (`libnvidia-ml`) are vendor components
//! that ship with the NVIDIA driver — they are not installable as ordinary
//! build dependencies and are absent on AMD-only / driver-less hosts. Linking
//! them at build time would (a) make the build fail without the NVIDIA libs
//! present and (b) make the resulting binary refuse to start anywhere they are
//! missing. Instead we `dlopen` them on first use: the binary has no build-time
//! or load-time dependency on them, and the CUDA backend simply reports "no
//! devices" when the driver isn't installed.
//!
//! [`dynamic_library!`] generates, for one such library, a function-pointer
//! table plus same-named wrapper `fn`s, so the call sites in [`crate::cuda`] /
//! [`crate::nvml`] are unchanged — only the `extern "C"` block is replaced.

/// Open the first of `names` that loads (e.g. the versioned SONAME first, then
/// the unversioned dev symlink). Returns the last error if none load.
pub fn load_first(names: &[&str]) -> Result<libloading::Library, libloading::Error> {
    let mut last_err = None;
    for name in names {
        match unsafe { libloading::Library::new(name) } {
            Ok(lib) => return Ok(lib),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.expect("load_first called with an empty name list"))
}

/// Generate a runtime-loaded binding for one shared library.
///
/// Produces a hidden fn-pointer struct, a `OnceLock`-cached loader (`$loader()`
/// returns `Option<&'static _>`, `None` when the library can't be loaded), and a
/// same-named `unsafe fn` wrapper for each declared function that dispatches
/// through the table. Public entry points must check `$loader().is_some()` (or
/// `?` on the `Option`) before invoking any wrapper; the wrappers themselves
/// panic if called with the library unloaded, which the entry-point guards
/// prevent.
macro_rules! dynamic_library {
    (
        lib_struct: $Lib:ident,
        loader: $loader:ident,
        names: [$($lname:expr),+ $(,)?],
        $( fn $fname:ident($($an:ident: $at:ty),* $(,)?) -> $ret:ty; )*
    ) => {
        #[allow(non_snake_case)]
        struct $Lib {
            $( $fname: unsafe extern "C" fn($($at),*) -> $ret, )*
            // Keep the library mapped for the process lifetime; the fn pointers
            // above point into it.
            #[allow(dead_code)]
            handle: libloading::Library,
        }

        impl $Lib {
            #[allow(non_snake_case)]
            unsafe fn load() -> std::result::Result<Self, libloading::Error> {
                let handle = $crate::dylib::load_first(&[$($lname),+])?;
                $(
                    let $fname: unsafe extern "C" fn($($at),*) -> $ret =
                        *handle.get(concat!(stringify!($fname), "\0").as_bytes())?;
                )*
                Ok(Self { $($fname,)* handle })
            }
        }

        static __DYLIB: std::sync::OnceLock<Option<$Lib>> = std::sync::OnceLock::new();

        /// The loaded library, or `None` if it could not be opened.
        fn $loader() -> Option<&'static $Lib> {
            __DYLIB.get_or_init(|| unsafe { $Lib::load().ok() }).as_ref()
        }

        $(
            #[inline]
            #[allow(non_snake_case)]
            unsafe fn $fname($($an: $at),*) -> $ret {
                ($loader()
                    .expect(concat!(stringify!($fname), ": ", stringify!($Lib), " not loaded"))
                    .$fname)($($an),*)
            }
        )*
    };
}

pub(crate) use dynamic_library;
