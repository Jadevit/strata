// llama-plugin/src/ffi/metadata.rs
//
// Safe-ish wrappers around llama_sys for GGUF metadata scraping.
// Opens models in header-only mode (vocab_only=true) and returns flattened K/Vs.

use std::{collections::HashMap, ffi::CString, path::Path, ptr::NonNull};

use llama_sys::{
    llama_free_model, llama_load_model_from_file, llama_model, llama_model_default_params,
    llama_model_meta_count, llama_model_meta_key_by_index, llama_model_meta_val_str_by_index,
};

#[inline]
fn cstr_from_buf(buf: &[i8]) -> String {
    let len = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    let slice = &buf[..len];
    let u8s: Vec<u8> = slice.iter().map(|&c| c as u8).collect();
    String::from_utf8_lossy(&u8s).to_string()
}

/// Open a model in *header-only* mode (no tensors/mmaps beyond headers).
pub unsafe fn open_header_only(path: &Path) -> Result<NonNull<llama_model>, String> {
    let path_str = path.to_str().ok_or_else(|| "non-UTF8 path".to_string())?;
    let c_path =
        CString::new(path_str).map_err(|_| "invalid model path (interior NUL)".to_string())?;

    let mut params = llama_model_default_params();
    params.vocab_only = true; // header-only reads
    params.use_mmap = true;
    params.n_gpu_layers = 0;

    let ptr = llama_load_model_from_file(c_path.as_ptr(), params);
    NonNull::new(ptr).ok_or_else(|| "llama_load_model_from_file returned null".into())
}

#[inline]
pub unsafe fn close_model(model: NonNull<llama_model>) {
    llama_free_model(model.as_ptr());
}

pub unsafe fn read_all_meta(model: NonNull<llama_model>) -> HashMap<String, String> {
    let n = llama_model_meta_count(model.as_ptr());
    let mut out = HashMap::with_capacity(n as usize);

    // Larger scratch buffers; annotate truncation to aid debugging if needed.
    let mut key_buf = vec![0i8; 1024];
    let mut val_buf = vec![0i8; 8192];

    for i in 0..n {
        let kn =
            llama_model_meta_key_by_index(model.as_ptr(), i, key_buf.as_mut_ptr(), key_buf.len());
        if kn <= 0 {
            continue;
        }
        let truncated_key = (kn as usize) >= key_buf.len().saturating_sub(1);
        let key = cstr_from_buf(&key_buf);

        let vn = llama_model_meta_val_str_by_index(
            model.as_ptr(),
            i,
            val_buf.as_mut_ptr(),
            val_buf.len(),
        );
        if vn <= 0 {
            continue;
        }
        let truncated_val = (vn as usize) >= val_buf.len().saturating_sub(1);
        let mut val = cstr_from_buf(&val_buf);

        if truncated_key || truncated_val {
            val.push_str(" [__truncated__]");
        }

        out.insert(key, val);
    }
    out
}
