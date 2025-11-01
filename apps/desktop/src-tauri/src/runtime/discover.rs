use serde_json::Value as Json;
use std::path::{Path, PathBuf};

pub fn backend_runtime_root(backend_id: &str) -> Option<PathBuf> {
    dirs::data_dir().map(|p| p.join("Strata").join("runtimes").join(backend_id))
}

pub fn default_runtime_root() -> Option<PathBuf> {
    backend_runtime_root("llama")
}

fn read_runtime_json(root: &Path) -> Option<Json> {
    let p = root.join("runtime.json");
    let bytes = std::fs::read(&p).ok()?;
    serde_json::from_slice::<Json>(&bytes).ok()
}

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

pub fn runtime_plugin_filename(root: &Path) -> Option<String> {
    let j = read_runtime_json(root)?;
    if let Some(active) = j.get("active_variant").and_then(|v| v.as_str()) {
        if let Some(vmap) = j.get("variants").and_then(|v| v.as_object()) {
            if let Some(entry) = vmap.get(active).and_then(|e| e.as_object()) {
                if let Some(fname) = entry.get("file").and_then(|f| f.as_str()) {
                    return Some(fname.to_string());
                }
            }
        }
    }
    None
}

pub fn runtime_cpu_fallback_path(root: &Path) -> Option<PathBuf> {
    let j = read_runtime_json(root)?;
    if let Some(vmap) = j.get("variants").and_then(|v| v.as_object()) {
        if let Some(entry) = vmap.get("cpu").and_then(|e| e.as_object()) {
            let dir = entry.get("dir").and_then(|d| d.as_str())?;
            let file = entry.get("file").and_then(|f| f.as_str())?;
            return Some(PathBuf::from(dir).join(file));
        }
    }
    None
}

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
