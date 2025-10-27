use std::{env, path::{Path, PathBuf}};

use crate::runtime::{default_runtime_root, runtime_current_lib_dir, runtime_is_monolith};

#[cfg(target_os = "windows")]
fn plugin_filename() -> &'static str { "StrataLlama.dll" }
#[cfg(target_os = "macos")]
fn plugin_filename() -> &'static str { "StrataLlama.dylib" }
#[cfg(all(unix, not(target_os = "macos")))]
fn plugin_filename() -> &'static str { "StrataLlama.so" }

#[cfg(target_os = "windows")]
fn runtime_llama_filename() -> &'static str { "llama.dll" }
#[cfg(target_os = "macos")]
fn runtime_llama_filename() -> &'static str { "libllama.dylib" }
#[cfg(all(unix, not(target_os = "macos")))]
fn runtime_llama_filename() -> &'static str { "libllama.so" }

pub(crate) fn locate_runtime_llama_lib(plugin_path: &Path) -> Option<PathBuf> {
    if let Ok(p) = env::var("STRATA_LLAMA_LIB_PATH") {
        let p = PathBuf::from(p);
        if p.exists() { return Some(p); }
    }

    if let Some(root) = env::var("STRATA_RUNTIME_DIR").map(PathBuf::from).ok().or_else(default_runtime_root) {
        if runtime_is_monolith(&root) {
            return None;
        }
        if let Some(dir) = runtime_current_lib_dir(&root) {
            let p = dir.join(runtime_llama_filename());
            if p.exists() { return Some(p); }
        }
        for candidate in [
            root.join(runtime_llama_filename()),
            root.join("llama_backend").join(runtime_llama_filename()),
        ] {
            if candidate.exists() { return Some(candidate); }
        }
    }

    if let Some(dir) = plugin_path.parent() {
        let cand = dir.join("resources/llama").join(runtime_llama_filename());
        if cand.exists() { return Some(cand); }
    }

    let dev = PathBuf::from("target/debug/resources/llama").join(runtime_llama_filename());
    dev.exists().then_some(dev)
}

pub(crate) fn locate_plugin_binary() -> Option<PathBuf> {
    if let Ok(p) = env::var("STRATA_PLUGIN_PATH") {
        let p = PathBuf::from(p);
        if p.exists() {
            eprintln!("[plugin] STRATA_PLUGIN_PATH = {}", p.display());
            return Some(p);
        } else {
            eprintln!("[plugin] STRATA_PLUGIN_PATH points to missing file: {}", p.display());
        }
    }

    if let Some(root) = default_runtime_root() {
        if let Some(cur) = runtime_current_lib_dir(&root) {
            let p = cur.join(plugin_filename());
            if p.exists() {
                eprintln!("[plugin] from runtime.json: {}", p.display());
                return Some(p);
            }
        }
        for variant in ["cuda", "vulkan", "metal", "cpu"] {
            let p = root.join(variant).join("llama_backend").join(plugin_filename());
            if p.exists() {
                eprintln!("[plugin] found in {variant} pack: {}", p.display());
                return Some(p);
            }
        }
        for p in [
            root.join("llama_backend").join(plugin_filename()),
            root.join("plugins").join(plugin_filename()),
        ] {
            if p.exists() { return Some(p); }
        }
    }

    for p in [
        PathBuf::from("target/debug").join(plugin_filename()),
        PathBuf::from("target/release").join(plugin_filename()),
    ] {
        if p.exists() { return Some(p); }
    }

    None
}
