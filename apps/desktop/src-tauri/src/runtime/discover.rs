use serde_json::Value as Json;
use std::path::{Path, PathBuf};

/// ~/.local/share/Strata/runtimes/<backend>
pub fn backend_runtime_root(backend_id: &str) -> Option<PathBuf> {
    dirs::data_dir().map(|p| p.join("Strata").join("runtimes").join(backend_id))
}

/// Default runtime backend (“llama” today).
pub fn default_runtime_root() -> Option<PathBuf> {
    backend_runtime_root("llama")
}

/// Read runtime.json (supports both new and legacy shapes).
fn read_runtime_json(root: &Path) -> Option<Json> {
    let p = root.join("runtime.json");
    let bytes = std::fs::read(&p).ok()?;
    serde_json::from_slice::<Json>(&bytes).ok()
}

/// Directory that contains the active plugin .so/.dll/.dylib.
/// Tries top-level "current_lib_dir", then "llama.current_lib_dir".
pub fn runtime_current_lib_dir(root: &Path) -> Option<PathBuf> {
    let j = read_runtime_json(root)?;
    if let Some(s) = j.get("current_lib_dir").and_then(|v| v.as_str()) {
        return Some(PathBuf::from(s));
    }
    j.get("llama")
        .and_then(|ll| ll.get("current_lib_dir"))
        .and_then(|v| v.as_str())
        .map(PathBuf::from)
}

/// OS-specific base names for CPU and per-GPU variants.
#[cfg(target_os = "windows")]
const CPU_BASENAME: &str = "StrataLlama.dll";
#[cfg(target_os = "linux")]
const CPU_BASENAME: &str = "libStrataLlama.so";
#[cfg(target_os = "macos")]
const CPU_BASENAME: &str = "libStrataLlama.dylib";

#[cfg(target_os = "windows")]
fn basename_for_backend(backend: Option<&str>) -> &'static str {
    match backend {
        Some("cuda") => "StrataLlama_cuda.dll",
        Some("vulkan") => "StrataLlama_vulkan.dll",
        Some("metal") => "StrataLlama_metal.dll", // rarely used on Windows, present for symmetry
        _ => CPU_BASENAME,
    }
}

#[cfg(target_os = "linux")]
fn basename_for_backend(backend: Option<&str>) -> &'static str {
    match backend {
        Some("cuda") => "libStrataLlama_cuda.so",
        Some("vulkan") => "libStrataLlama_vulkan.so",
        Some("metal") => "libStrataLlama_metal.so",
        _ => CPU_BASENAME,
    }
}

#[cfg(target_os = "macos")]
fn basename_for_backend(backend: Option<&str>) -> &'static str {
    match backend {
        Some("metal") => "libStrataLlama_metal.dylib",
        Some("cuda") => "libStrataLlama_cuda.dylib",
        Some("vulkan") => "libStrataLlama_vulkan.dylib",
        _ => CPU_BASENAME,
    }
}

/// Pick the plugin filename (.so/.dll/.dylib).
/// Precedence:
///  1) explicit "plugin_basename" (top-level or under "llama")
///  2) legacy "variants" mapping for the "active_variant"
///  3) derive from "llama.gpu_backend" (cpu/cuda/vulkan/metal)
pub fn runtime_plugin_filename(root: &Path) -> Option<String> {
    let j = read_runtime_json(root)?;

    // 1) explicit
    if let Some(s) = j.get("plugin_basename").and_then(|v| v.as_str()) {
        return Some(s.to_string());
    }
    if let Some(s) = j
        .get("llama")
        .and_then(|ll| ll.get("plugin_basename"))
        .and_then(|v| v.as_str())
    {
        return Some(s.to_string());
    }

    // 2) legacy map: active_variant -> variants[active].file
    if let Some(active) = j.get("active_variant").and_then(|v| v.as_str()) {
        if let Some(vmap) = j.get("variants").and_then(|v| v.as_object()) {
            if let Some(entry) = vmap.get(active).and_then(|e| e.as_object()) {
                if let Some(fname) = entry.get("file").and_then(|f| f.as_str()) {
                    return Some(fname.to_string());
                }
            }
        }
    }

    // 3) derive from "llama.gpu_backend" (None -> CPU)
    let backend = j
        .get("llama")
        .and_then(|ll| ll.get("gpu_backend"))
        .and_then(|v| v.as_str());

    Some(basename_for_backend(backend).to_string())
}

/// Path to the CPU variant as a fallback.
/// Prefers legacy variants["cpu"] { dir, file }, else assumes <root>/cpu/llama_backend/<cpu-basename>.
pub fn runtime_cpu_fallback_path(root: &Path) -> Option<PathBuf> {
    let j = read_runtime_json(root)?;

    // legacy variants map
    if let Some(vmap) = j.get("variants").and_then(|v| v.as_object()) {
        if let Some(entry) = vmap.get("cpu").and_then(|e| e.as_object()) {
            if let (Some(dir), Some(file)) = (
                entry.get("dir").and_then(|d| d.as_str()),
                entry.get("file").and_then(|f| f.as_str()),
            ) {
                return Some(PathBuf::from(dir).join(file));
            }
        }
    }

    // generic fallback layout
    Some(root.join("cpu").join("llama_backend").join(CPU_BASENAME))
}

/// Whether the runtime pack contains the plugin and deps together.
pub fn runtime_is_monolith(root: &Path) -> bool {
    read_runtime_json(root)
        .and_then(|j| {
            j.get("monolith")
                .cloned()
                .or_else(|| j.get("llama").and_then(|ll| ll.get("monolith")).cloned())
        })
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}
