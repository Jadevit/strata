// crates/backends/llama/llama-plugin/src/ffi/model.rs
//
// Unsafe, model-centric helpers around llama_sys. Keep *all* llama_model*
// touching code here so higher layers stay safe.

use llama_sys::{
    llama_model, llama_model_chat_template, llama_model_desc, llama_model_get_vocab,
    llama_model_meta_count, llama_model_meta_key_by_index, llama_model_meta_val_str,
    llama_model_meta_val_str_by_index, llama_n_vocab,
};
use std::ffi::{CStr, CString};

/// Return model description (if present).
pub unsafe fn desc(model: *mut llama_model) -> Option<String> {
    let mut buf = vec![0i8; 2048];
    let wrote = llama_model_desc(model, buf.as_mut_ptr(), buf.len());
    if wrote <= 0 {
        return None;
    }
    Some(CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned())
}

/// Return the default chat template as String (None if missing/empty).
pub unsafe fn chat_template(model: *mut llama_model) -> Option<String> {
    let ptr = llama_model_chat_template(model, std::ptr::null());
    if ptr.is_null() {
        return None;
    }
    let s = CStr::from_ptr(ptr).to_string_lossy().into_owned();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// Lookup a single metadata value by key (string), if present.
pub unsafe fn meta_get_str(model: *mut llama_model, key: &CStr) -> Option<String> {
    // Start with 1KB, grow a few times if needed.
    let mut cap = 1024usize;
    for _ in 0..4 {
        let mut buf = vec![0i8; cap];
        let wrote = llama_model_meta_val_str(model, key.as_ptr(), buf.as_mut_ptr(), buf.len());
        if wrote < 0 {
            cap = cap.saturating_mul(2);
            continue;
        }
        if wrote == 0 {
            return None;
        }
        return Some(CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned());
    }
    None
}

/// Iterate all metadata key/value pairs (best-effort).
pub unsafe fn meta_iter(model: *mut llama_model) -> Vec<(String, String)> {
    let count = llama_model_meta_count(model);
    if count <= 0 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(count as usize);
    let mut key_buf = vec![0i8; 1024];
    let mut val_buf = vec![0i8; 4096];

    for i in 0..count {
        let k_ok = llama_model_meta_key_by_index(model, i, key_buf.as_mut_ptr(), key_buf.len());
        if k_ok <= 0 {
            continue;
        }
        let v_ok = llama_model_meta_val_str_by_index(model, i, val_buf.as_mut_ptr(), val_buf.len());
        if v_ok <= 0 {
            continue;
        }
        let k = CStr::from_ptr(key_buf.as_ptr())
            .to_string_lossy()
            .into_owned();
        let v = CStr::from_ptr(val_buf.as_ptr())
            .to_string_lossy()
            .into_owned();
        out.push((k, v));
    }

    out
}

/// Vocab size for this model.
#[inline]
pub unsafe fn n_vocab(model: *mut llama_model) -> usize {
    let vocab = llama_model_get_vocab(model);
    llama_n_vocab(vocab) as usize
}
