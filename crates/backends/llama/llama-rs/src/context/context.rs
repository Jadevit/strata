// Borrowed context tied to a model’s lifetime. All mutation lives here; the
// `LlamaModel` itself is immutable and Sync once loaded.

use std::ffi::{CStr, CString};
use std::ptr::NonNull;
use std::slice;

use crate::batch::LlamaBatch;
use crate::model::LlamaModel;
use crate::token::LlamaToken;
use llama_sys::{
    llama_context, llama_decode, llama_detokenize, llama_get_embeddings, llama_get_logits,
    llama_get_memory, llama_memory_clear, llama_model_get_vocab, llama_model_n_embd, llama_n_vocab,
    llama_token_eos, llama_token_get_text, llama_tokenize,
};

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

    /// Build a batch and decode it, marking only the final token for logits.
    /// Caller is responsible for correct `n_past` bookkeeping.
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
        let rc = unsafe { llama_decode(self.ctx.as_ptr(), batch.raw) };
        if rc != 0 {
            Err(format!("llama_decode failed with code {}", rc))
        } else {
            Ok(())
        }
    }

    /// Clear the KV cache for this context.
    /// Clear the KV cache for this context.
    pub fn clear_kv_cache(&mut self) {
        unsafe {
            let mem = llama_sys::llama_get_memory(self.ctx.as_ptr());
            // Pass `true` to also clear the data buffers;
            // use `false` if you just want to reset sequence bookkeeping.
            llama_sys::llama_memory_clear(mem, true);
        }
    }

    /// View of the current logits. Length == vocab size.
    pub fn get_logits(&self) -> &[f32] {
        let ptr = unsafe { llama_get_logits(self.ctx.as_ptr()) };
        debug_assert!(!ptr.is_null(), "llama_get_logits returned null");
        let vocab_ptr = unsafe { llama_model_get_vocab(self.model.as_ptr()) };
        let vocab_size = unsafe { llama_n_vocab(vocab_ptr) as usize };
        unsafe { slice::from_raw_parts(ptr, vocab_size) }
    }

    /// Optional view of embeddings. Length == hidden size (n_embd).
    pub fn get_embeddings(&self) -> Option<&[f32]> {
        if !self.embeddings_enabled {
            return None;
        }
        let ptr = unsafe { llama_get_embeddings(self.ctx.as_ptr()) };
        if ptr.is_null() {
            return None;
        }
        let n_embd = unsafe { llama_model_n_embd(self.model.as_ptr()) as usize };
        Some(unsafe { slice::from_raw_parts(ptr, n_embd) })
    }

    /// Two-pass tokenize (duplicate of model.tokenize for convenience).
    pub fn tokenize(&self, text: &str) -> Result<Vec<LlamaToken>, String> {
        let c_text = CString::new(text).map_err(|e| format!("CString error: {:?}", e))?;
        let vocab_ptr = unsafe { llama_model_get_vocab(self.model.as_ptr()) };

        // Probe required size (0 buffer)
        let needed = unsafe {
            llama_tokenize(
                vocab_ptr,
                c_text.as_ptr(),
                c_text.as_bytes().len() as i32,
                std::ptr::null_mut(),
                0,
                true,  // add_special
                false, // parse_special
            )
        };
        if needed < 0 {
            return Err(format!("llama_tokenize(size probe) failed: {}", needed));
        }
        let needed = needed as usize;
        if needed == 0 {
            return Ok(Vec::new());
        }

        // Allocate exact size and fill
        let mut buf = vec![0i32; needed];
        let n = unsafe {
            llama_tokenize(
                vocab_ptr,
                c_text.as_ptr(),
                c_text.as_bytes().len() as i32,
                buf.as_mut_ptr(),
                buf.len() as i32,
                true,
                false,
            )
        };
        if n < 0 {
            return Err(format!("llama_tokenize(fill) failed: {}", n));
        }
        Ok(buf[..n as usize].iter().copied().map(LlamaToken).collect())
    }

    /// Per-token text (kept for completeness; streaming prefers byte detok).
    pub fn token_to_str(&self, token: LlamaToken) -> Result<String, String> {
        let vocab_ptr = unsafe { llama_model_get_vocab(self.model.as_ptr()) };
        let ptr = unsafe { llama_token_get_text(vocab_ptr, token.0) };
        if ptr.is_null() {
            return Err("llama_token_get_text returned null".into());
        }
        let c_str = unsafe { CStr::from_ptr(ptr) };
        c_str
            .to_str()
            .map(|s| s.to_string())
            .map_err(|e| format!("Invalid UTF-8 from token: {:?}", e))
    }

    pub fn token_eos(&self) -> LlamaToken {
        let vocab_ptr = unsafe { llama_model_get_vocab(self.model.as_ptr()) };
        let id = unsafe { llama_token_eos(vocab_ptr) };
        LlamaToken(id)
    }

    /// **Detokenize** a whole sequence into valid UTF-8 (fixes emoji & quotes).
    ///
    /// - `remove_special`: true means BOS/EOS can be removed when model is configured that way
    /// - `unparse_special`: true means render special tokens verbatim (we usually keep this false)
    pub fn detokenize(
        &self,
        toks: &[LlamaToken],
        remove_special: bool,
        unparse_special: bool,
    ) -> Result<String, String> {
        let vocab = unsafe { llama_model_get_vocab(self.model.as_ptr()) };
        if vocab.is_null() {
            return Err("llama_model_get_vocab returned null".into());
        }

        let ids: Vec<i32> = toks.iter().map(|t| t.0).collect();

        // Probe: null buffer, 0 size → -needed (or 0 if nothing to render)
        let probe = unsafe {
            llama_detokenize(
                vocab,
                ids.as_ptr(),
                ids.len() as i32,
                std::ptr::null_mut(),
                0,
                remove_special,
                unparse_special,
            )
        };

        // NOTE: 0 is a valid outcome when specials are removed → return ""
        if probe == 0 {
            return Ok(String::new());
        }
        let needed = if probe < 0 { -probe } else { probe };
        if needed < 0 {
            return Err(format!("llama_detokenize(size probe) failed: {}", probe));
        }

        let cap = (needed as usize).saturating_add(8);
        let mut buf = vec![0u8; cap];

        let got = unsafe {
            llama_detokenize(
                vocab,
                ids.as_ptr(),
                ids.len() as i32,
                buf.as_mut_ptr() as *mut _,
                cap as i32,
                remove_special,
                unparse_special,
            )
        };

        if got < 0 {
            return Err(format!("llama_detokenize(fill) failed: {}", got));
        }

        let bytes = &buf[..(got as usize)];
        String::from_utf8(bytes.to_vec())
            .map_err(|e| format!("detokenize produced non-UTF-8: {e:?}"))
    }

    /// Detokenize to raw bytes (valid UTF-8 not guaranteed). Preferred for streaming paths.
    /// Safe wrapper that hides `llama_sys` from upstream crates.
    /// Detokenize to raw bytes (valid UTF-8 not guaranteed). Preferred for streaming paths.
    /// Safe wrapper that hides `llama_sys` from upstream crates.
    pub fn detokenize_bytes(
        &self,
        toks: &[LlamaToken],
        remove_special: bool,
        unparse_special: bool,
    ) -> Result<Vec<u8>, String> {
        let vocab = unsafe { llama_model_get_vocab(self.model.as_ptr()) };
        if vocab.is_null() {
            return Err("llama_model_get_vocab returned null".into());
        }

        let ids: Vec<i32> = toks.iter().map(|t| t.0).collect();

        // Probe: -needed on too small, 0 means "nothing to render" (valid).
        let probe = unsafe {
            llama_detokenize(
                vocab,
                ids.as_ptr(),
                ids.len() as i32,
                std::ptr::null_mut(),
                0,
                remove_special,
                unparse_special,
            )
        };

        // ✅ 0 is fine (e.g., only specials removed) → return empty bytes.
        if probe == 0 {
            return Ok(Vec::new());
        }

        let needed = if probe < 0 { -probe } else { probe };
        if needed < 0 {
            return Err(format!("llama_detokenize(size probe) failed: {}", probe));
        }

        let cap = (needed as usize).saturating_add(8);
        let mut buf = vec![0u8; cap];

        let got = unsafe {
            llama_detokenize(
                vocab,
                ids.as_ptr(),
                ids.len() as i32,
                buf.as_mut_ptr() as *mut _,
                cap as i32,
                remove_special,
                unparse_special,
            )
        };
        if got < 0 {
            return Err(format!("llama_detokenize(fill) failed: {}", got));
        }

        buf.truncate(got as usize);
        Ok(buf)
    }
}

impl<'a> Drop for LlamaContext<'a> {
    fn drop(&mut self) {
        unsafe { crate::ffi::cleanup_context(self.ctx.as_ptr()) };
    }
}
