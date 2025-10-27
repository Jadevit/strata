use std::path::Path;

use crate::plugin::load_plugin_once;
use strata_abi::metadata::ModelCoreInfo;

#[inline]
fn make_cstring(s: &str) -> Result<std::ffi::CString, String> {
    std::ffi::CString::new(s).map_err(|_| "string contains interior NUL".to_string())
}

unsafe fn take_plugin_string(
    api_free: strata_abi::ffi::FreeStringFn,
    s: strata_abi::ffi::StrataString,
) -> String {
    if s.ptr.is_null() || s.len == 0 {
        return String::new();
    }
    let out = {
        // rust-2024: unsafe-op inside unsafe fn still requires an unsafe block
        let slice = unsafe { std::slice::from_raw_parts(s.ptr as *const u8, s.len) };
        String::from_utf8_lossy(slice).into_owned()
    };
    unsafe { api_free(s) };
    out
}

/// Read core model metadata from the plugin for a given model path.
/// Returns the ABI-level core info (UI conversion happens at the caller).
pub fn collect_model_metadata_via_plugin(path: &Path) -> Result<ModelCoreInfo, String> {
    let plugin = load_plugin_once()?;
    let cpath = make_cstring(path.to_str().ok_or("invalid UTF-8 in path")?)?;
    unsafe {
        let s = (plugin.api.metadata.collect_json)(cpath.as_ptr());
        let js = take_plugin_string(plugin.api.metadata.free_string, s);
        if js.is_empty() {
            // On error, API returns empty and last_error() has the reason.
            let err = (plugin.api.llm.last_error)();
            let msg = take_plugin_string(plugin.api.llm.free_string, err);
            if msg.is_empty() {
                Err("plugin returned empty metadata".into())
            } else {
                Err(msg)
            }
        } else {
            serde_json::from_str::<ModelCoreInfo>(&js)
                .map_err(|e| format!("bad metadata JSON: {e}"))
        }
    }
}
