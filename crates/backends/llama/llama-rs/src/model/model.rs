// strata-backend-llama/src/model.rs

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr::NonNull;

use crate::context::LlamaContext;
use crate::token::LlamaToken;
use llama_sys::{
    llama_context_params, llama_free_model, llama_load_model_from_file, llama_model,
    llama_model_chat_template, llama_model_desc, llama_model_get_vocab, llama_model_meta_count,
    llama_model_meta_key_by_index, llama_model_meta_val_str, llama_model_meta_val_str_by_index,
    llama_n_vocab, llama_new_context_with_model, llama_token_eos, llama_token_get_text,
    llama_tokenize,
};

/// Safe wrapper around `llama_model*`.
pub struct LlamaModel {
    model: NonNull<llama_model>,
}

impl LlamaModel {
    /// Wrap an existing raw model pointer (non-null).
    pub fn new(model: *mut llama_model) -> Result<Self, String> {
        NonNull::new(model)
            .map(|model| Self { model })
            .ok_or_else(|| "LlamaModel::new received null pointer".to_string())
    }

    /// Convenience loader for callers that don't go through Backend::load.
    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let c_path = CString::new(path).map_err(|e| format!("CString error: {:?}", e))?;
        let params = crate::ffi::default_model_params();
        let model_ptr = unsafe { llama_load_model_from_file(c_path.as_ptr(), params) };
        Self::new(model_ptr)
    }

    #[inline]
    pub fn as_ptr(&self) -> *mut llama_model {
        self.model.as_ptr()
    }

    /// Create a new context for this model.
    pub fn create_context<'a>(
        &'a self,
        params: llama_context_params,
        embeddings_enabled: bool,
    ) -> Result<LlamaContext<'a>, String> {
        let ctx_ptr = unsafe { llama_new_context_with_model(self.as_ptr(), params) };
        let ctx_ptr = NonNull::new(ctx_ptr)
            .ok_or_else(|| "Failed to create LlamaContext (null pointer)".to_string())?;

        Ok(LlamaContext::new(
            self,
            ctx_ptr,
            embeddings_enabled,
            params.n_ctx as u32,
        ))
    }

    /// Robust two-pass tokenize (handles +N / -N probe, retries if needed, parses special tokens).
    pub fn tokenize(&self, text: &str) -> Result<Vec<LlamaToken>, String> {
        let c_text = CString::new(text).map_err(|e| format!("CString error: {:?}", e))?;
        let vocab = unsafe { llama_sys::llama_model_get_vocab(self.as_ptr()) };

        // 1) Probe: some builds return +needed, others -needed
        let probe = unsafe {
            llama_sys::llama_tokenize(
                vocab,
                c_text.as_ptr(),
                c_text.as_bytes().len() as i32,
                std::ptr::null_mut(),
                0,
                /* add_special   */ true,
                /* parse_special */ true,
            )
        };
        let mut needed = if probe < 0 {
            (-probe) as usize
        } else {
            probe as usize
        };
        if needed == 0 {
            return Ok(Vec::new());
        }

        // 2) Allocate exactly and fill
        let mut buf = vec![0i32; needed];
        let filled = unsafe {
            llama_sys::llama_tokenize(
                vocab,
                c_text.as_ptr(),
                c_text.as_bytes().len() as i32,
                buf.as_mut_ptr(),
                buf.len() as i32,
                true,
                true,
            )
        };

        // If runtime says "too small" again (negative), resize and retry once.
        let n = if filled < 0 {
            needed = (-filled) as usize;
            buf.resize(needed, 0);
            let filled2 = unsafe {
                llama_sys::llama_tokenize(
                    vocab,
                    c_text.as_ptr(),
                    c_text.as_bytes().len() as i32,
                    buf.as_mut_ptr(),
                    buf.len() as i32,
                    true,
                    true,
                )
            };
            if filled2 < 0 {
                return Err(format!("llama_tokenize failed after retry: {}", filled2));
            }
            filled2 as usize
        } else {
            filled as usize
        };

        buf.truncate(n);
        Ok(buf.into_iter().map(LlamaToken).collect())
    }

    /// Convert a token id to its string form (UTF-8).
    pub fn token_to_str(&self, token: LlamaToken) -> Result<String, String> {
        let vocab_ptr = unsafe { llama_model_get_vocab(self.as_ptr()) };
        let ptr = unsafe { llama_token_get_text(vocab_ptr, token.0) };
        if ptr.is_null() {
            return Err("llama_token_get_text returned null".to_string());
        }
        let c_str = unsafe { std::ffi::CStr::from_ptr(ptr) };
        Ok(c_str.to_string_lossy().into_owned())
    }

    /// End-of-sequence token.
    pub fn token_eos(&self) -> LlamaToken {
        let vocab_ptr = unsafe { llama_model_get_vocab(self.as_ptr()) };
        let id = unsafe { llama_token_eos(vocab_ptr) };
        LlamaToken(id)
    }

    /// Vocab size (helper for diagnostics or custom sampling).
    pub fn n_vocab(&self) -> usize {
        let vocab_ptr = unsafe { llama_model_get_vocab(self.as_ptr()) };
        unsafe { llama_n_vocab(vocab_ptr) as usize }
    }

    /// Model description string from GGUF, if available.
    pub fn description(&self) -> Option<String> {
        let mut buf = vec![0i8; 2048];
        let wrote = unsafe { llama_model_desc(self.as_ptr(), buf.as_mut_ptr(), buf.len()) };
        if wrote <= 0 {
            return None;
        }
        let s = unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        Some(s)
    }

    /// Return the *default* chat template string if present in metadata.
    pub fn chat_template(&self) -> Option<String> {
        let ptr = unsafe { llama_model_chat_template(self.as_ptr(), std::ptr::null::<c_char>()) };
        if ptr.is_null() {
            return None;
        }
        let s = unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned();
        if s.is_empty() {
            None
        } else {
            Some(s)
        }
    }

    /// Lookup a single metadata value by key (string), if present.
    pub fn meta_get_str(&self, key: &str) -> Option<String> {
        let c_key = CString::new(key).ok()?;
        // Start with 1KB, grow on demand.
        let mut cap = 1024usize;
        for _ in 0..4 {
            let mut buf = vec![0i8; cap];
            let wrote = unsafe {
                llama_model_meta_val_str(self.as_ptr(), c_key.as_ptr(), buf.as_mut_ptr(), buf.len())
            };
            if wrote < 0 {
                cap = cap.saturating_mul(2);
                continue;
            }
            if wrote == 0 {
                return None;
            }
            let s = unsafe { CStr::from_ptr(buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            return Some(s);
        }
        None
    }

    /// Iterate all metadata key/value pairs (best-effort).
    pub fn meta_iter(&self) -> Vec<(String, String)> {
        let count = unsafe { llama_model_meta_count(self.as_ptr()) };
        if count <= 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(count as usize);

        // Generous buffers; llama will fill up to buf_size.
        let mut key_buf = vec![0i8; 1024];
        let mut val_buf = vec![0i8; 4096];

        for i in 0..count {
            let k_ok = unsafe {
                llama_model_meta_key_by_index(self.as_ptr(), i, key_buf.as_mut_ptr(), key_buf.len())
            };
            if k_ok <= 0 {
                continue;
            }
            let v_ok = unsafe {
                llama_model_meta_val_str_by_index(
                    self.as_ptr(),
                    i,
                    val_buf.as_mut_ptr(),
                    val_buf.len(),
                )
            };
            if v_ok <= 0 {
                continue;
            }
            let k = unsafe { CStr::from_ptr(key_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            let v = unsafe { CStr::from_ptr(val_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            out.push((k, v));
        }

        out
    }
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        unsafe { llama_free_model(self.model.as_ptr()) };
    }
}

// SAFETY: llama.cpp models are immutable after load; we only create/destroy contexts
// from them. All mutable state lives in LlamaContext (which remains !Send/!Sync).
unsafe impl Send for LlamaModel {}
unsafe impl Sync for LlamaModel {}
