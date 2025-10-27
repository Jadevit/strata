use serde_json::Value as Json;
use std::path::PathBuf;

/// Default per-user runtime root used by the installer and loader.
/// Matches the sidecar installer's layout.
pub fn default_runtime_root() -> Option<PathBuf> {
    dirs::data_dir().map(|p| p.join("Strata").join("runtimes").join("llama"))
}

fn read_runtime_json(root: &std::path::Path) -> Option<Json> {
    let p = root.join("runtime.json");
    let bytes = std::fs::read(&p).ok()?;
    serde_json::from_slice::<Json>(&bytes).ok()
}

/// Returns the current llama lib dir from runtime.json, if present.
pub fn runtime_current_lib_dir(root: &std::path::Path) -> Option<std::path::PathBuf> {
    let j = read_runtime_json(root)?;
    j.get("llama")
        .and_then(|ll| ll.get("current_lib_dir"))
        .and_then(|v| v.as_str())
        .map(std::path::PathBuf::from)
}

/// Whether the runtime bundle is a monolith (single .so/.dylib/.dll) per runtime.json.
pub fn runtime_is_monolith(root: &std::path::Path) -> bool {
    read_runtime_json(root)
        .and_then(|j| j.get("llama").cloned())
        .and_then(|ll| ll.get("monolith").cloned())
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}
