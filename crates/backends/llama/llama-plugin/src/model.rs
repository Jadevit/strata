// crates/backends/llama/llama-plugin/src/model.rs
//
// Safe wrapper around `llama_model*`. All pointer-level unsafe lives in
// crate::ffi::{model, context, ...}. This file only calls those helpers.

use std::ffi::CString;
use std::ptr::NonNull;

use crate::context::LlamaContext;
use crate::token::LlamaToken;

use crate::ffi; // init/cleanup, default_model_params, etc.
use crate::ffi::context as cctx; // context creation + token/detok helpers
use crate::ffi::model as mffi; // model-centric unsafe helpers

use llama_sys::{llama_context_params, llama_model};

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
    /// Uses crate::ffi::load_model() to stay forward-compatible with llama.cpp.
    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let p = unsafe { ffi::load_model(path)? };
        Ok(Self { model: p })
    }

    #[inline]
    pub fn as_ptr(&self) -> *mut llama_model {
        self.model.as_ptr()
    }

    /// Create a new context for this model (preferred: build params via your params layer).
    pub fn create_context<'a>(
        &'a self,
        params: llama_context_params,
        embeddings_enabled: bool,
    ) -> Result<LlamaContext<'a>, String> {
        let ctx_ptr = unsafe { cctx::create_context_with_params(self.as_ptr(), params)? };
        Ok(LlamaContext::new(
            self,
            ctx_ptr,
            embeddings_enabled,
            params.n_ctx as u32,
        ))
    }

    // --------------------------
    // Tokenization conveniences
    // --------------------------

    /// Robust two-pass tokenize (delegates to ffi::context). Returns model tokens.
    pub fn tokenize(&self, text: &str) -> Result<Vec<LlamaToken>, String> {
        let ids = cctx::tokenize(self.as_ptr(), text)?;
        Ok(ids.into_iter().map(LlamaToken).collect())
    }

    /// Convert a token id to its string form (UTF-8).
    pub fn token_to_str(&self, token: LlamaToken) -> Result<String, String> {
        cctx::token_to_str(self.as_ptr(), token.0)
    }

    /// End-of-sequence token.
    pub fn token_eos(&self) -> LlamaToken {
        LlamaToken(cctx::token_eos(self.as_ptr()))
    }

    /// Vocab size (helper for diagnostics or custom sampling).
    pub fn n_vocab(&self) -> usize {
        unsafe { mffi::n_vocab(self.as_ptr()) }
    }

    // --------------------------
    // Metadata / descriptors
    // --------------------------

    /// Model description string from GGUF, if available.
    pub fn description(&self) -> Option<String> {
        unsafe { mffi::desc(self.as_ptr()) }
    }

    /// Return the *default* chat template string if present in metadata.
    pub fn chat_template(&self) -> Option<String> {
        unsafe { mffi::chat_template(self.as_ptr()) }
    }

    /// Lookup a single metadata value by key (string), if present.
    pub fn meta_get_str(&self, key: &str) -> Option<String> {
        let c_key = CString::new(key).ok()?;
        unsafe { mffi::meta_get_str(self.as_ptr(), &c_key) }
    }

    /// Iterate all metadata key/value pairs (best-effort).
    pub fn meta_iter(&self) -> Vec<(String, String)> {
        unsafe { mffi::meta_iter(self.as_ptr()) }
    }
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        // central cleanup so ownership stays consistent
        unsafe { ffi::cleanup_model(self.model.as_ptr()) };
    }
}

// SAFETY: llama.cpp models are immutable after load; we only create/destroy contexts
// from them. All mutable state lives in LlamaContext (which remains !Send/!Sync).
unsafe impl Send for LlamaModel {}
unsafe impl Sync for LlamaModel {}
