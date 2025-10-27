// crates/backends/llama/llama-plugin/src/context.rs
//
// Borrowed context tied to a model’s lifetime. All mutation lives here;
// all pointer-level `unsafe` is delegated to crate::ffi::{context, batch}.

use std::ptr::NonNull;

use crate::batch::LlamaBatch;
use crate::ffi::context as cffi;
use crate::model::LlamaModel;
use crate::token::LlamaToken;
use llama_sys::llama_context;

/// Borrowed context tied to a model's lifetime.
pub struct LlamaContext<'a> {
    model: &'a LlamaModel,
    ctx: NonNull<llama_context>,
    pub embeddings_enabled: bool,
    /// Active runtime context window (n_ctx) recorded at construction.
    pub n_ctx: u32,
}

impl<'a> LlamaContext<'a> {
    pub fn new(
        model: &'a LlamaModel,
        ctx: NonNull<llama_context>,
        embeddings_enabled: bool,
        n_ctx: u32,
    ) -> Self {
        Self {
            model,
            ctx,
            embeddings_enabled,
            n_ctx,
        }
    }

    #[inline]
    pub fn as_ptr(&self) -> *mut llama_context {
        self.ctx.as_ptr()
    }

    /// Compute the next KV position from llama’s memory bookkeeping.
    pub fn next_position(&self) -> i32 {
        cffi::next_position(self.ctx.as_ptr())
    }

    /// Build a batch and decode it, marking only the final token for logits.
    pub fn evaluate_mut(&mut self, tokens: &[LlamaToken], n_past: i32) -> Result<(), String> {
        let mut batch = LlamaBatch::new(tokens.len());
        for (i, token) in tokens.iter().enumerate() {
            let pos = n_past + i as i32;
            let want_logits = i + 1 == tokens.len();
            batch.add(i, *token, pos, want_logits);
        }
        batch.mark_last_for_logits();
        self.decode(&mut batch)
    }

    /// Decode an already-prepared batch.
    pub fn decode(&mut self, batch: &mut LlamaBatch) -> Result<(), String> {
        cffi::decode_batch(self.ctx.as_ptr(), batch.raw)
    }

    /// Clear the KV cache for this context.
    pub fn clear_kv_cache(&mut self) {
        cffi::clear_kv(self.ctx.as_ptr(), true);
    }

    /// View of the current logits. Length == vocab size.
    pub fn get_logits(&self) -> &[f32] {
        cffi::logits(self.ctx.as_ptr(), self.model.as_ptr())
    }

    /// Optional view of embeddings. Length == hidden size (n_embd).
    pub fn get_embeddings(&self) -> Option<&[f32]> {
        if !self.embeddings_enabled {
            return None;
        }
        cffi::embeddings(self.ctx.as_ptr(), self.model.as_ptr())
    }

    /// Two-pass tokenize (duplicate of model.tokenize for convenience).
    pub fn tokenize(&self, text: &str) -> Result<Vec<LlamaToken>, String> {
        let ids = cffi::tokenize(self.model.as_ptr(), text)?;
        Ok(ids.into_iter().map(LlamaToken).collect())
    }

    /// Per-token text (kept for completeness; streaming prefers byte detok).
    pub fn token_to_str(&self, token: LlamaToken) -> Result<String, String> {
        cffi::token_to_str(self.model.as_ptr(), token.0)
    }

    pub fn token_eos(&self) -> LlamaToken {
        LlamaToken(cffi::token_eos(self.model.as_ptr()))
    }

    /// Detokenize a whole sequence into valid UTF-8 (fixes emoji & quotes).
    pub fn detokenize(
        &self,
        toks: &[LlamaToken],
        remove_special: bool,
        unparse_special: bool,
    ) -> Result<String, String> {
        let ids: Vec<i32> = toks.iter().map(|t| t.0).collect();
        cffi::detokenize_string(self.model.as_ptr(), &ids, remove_special, unparse_special)
    }

    /// Detokenize to raw bytes (preferred for streaming).
    pub fn detokenize_bytes(
        &self,
        toks: &[LlamaToken],
        remove_special: bool,
        unparse_special: bool,
    ) -> Result<Vec<u8>, String> {
        let ids: Vec<i32> = toks.iter().map(|t| t.0).collect();
        cffi::detokenize_bytes(self.model.as_ptr(), &ids, remove_special, unparse_special)
    }
}

impl<'a> Drop for LlamaContext<'a> {
    fn drop(&mut self) {
        // free via the central FFI cleanup so ownership stays consistent
        unsafe { crate::ffi::cleanup_context(self.ctx.as_ptr()) };
    }
}
