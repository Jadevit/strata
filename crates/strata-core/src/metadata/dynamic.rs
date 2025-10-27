//! Dylib loading utilities for metadata providers.
//!
//! Safety note:
//! - This expects plugins compiled with the same Rust toolchain and compatible
//!   dependency graph as the host (or a C-ABI shim).
//! - The exported symbol **must** be named `register_plugin` and have the
//!   signature shown below.
//! - We keep the `Library` alive for the entire process lifetime via
//!   `MetadataService::_libs` to avoid dangling vtables.
//!
//! If you intend to support third-party plugins compiled out-of-tree,
//! strongly consider a C-ABI surface in `strata-abi` (repr(C) vtable) rather
//! than passing Rust trait objects across the boundary.

use std::ffi::OsStr;
use std::path::Path;

use libloading::{Library, Symbol};

use super::MetadataService;

type RegisterFn = unsafe extern "C" fn(&mut MetadataService);

#[inline]
fn is_dylib(path: &Path) -> bool {
    match path.extension().and_then(OsStr::to_str) {
        Some(ext) if cfg!(target_os = "linux") => ext == "so",
        Some(ext) if cfg!(target_os = "macos") => ext == "dylib",
        Some(ext) if cfg!(target_os = "windows") => ext == "dll",
        _ => false,
    }
}

/// Attempt to load a single dylib and call its `register_plugin` function.
/// On success, the `lib` is retained by the service to keep the plugin alive.
unsafe fn load_one(service: &mut MetadataService, path: &Path) -> Result<(), String> {
    let lib = Library::new(path).map_err(|e| format!("dlopen {}: {e}", path.display()))?;

    // Convention: each plugin exports `register_plugin`.
    let func: Symbol<RegisterFn> = lib
        .get(b"register_plugin")
        .map_err(|e| format!("dlsym(register_plugin) {}: {e}", path.display()))?;

    // Allow plugin to register one or more providers.
    func(service);

    // Keep the library alive for the process lifetime.
    service._libs.push(lib);
    Ok(())
}

/// Load all plugins from a directory. Non-existent directory is fine (no-op).
pub(super) fn load_dir_into(service: &mut MetadataService, dir: &Path) -> Result<(), String> {
    if !dir.exists() {
        return Ok(());
    }

    let entries = std::fs::read_dir(dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))?;
    for entry in entries {
        let path = entry.map_err(|e| e.to_string())?.path();
        if !is_dylib(&path) {
            continue;
        }
        unsafe {
            // Each plugin is isolated; we keep loading even if one fails.
            if let Err(e) = load_one(service, &path) {
                eprintln!(
                    "[metadata][warn] failed to load plugin {}: {}",
                    path.display(),
                    e
                );
            }
        }
    }
    Ok(())
}
