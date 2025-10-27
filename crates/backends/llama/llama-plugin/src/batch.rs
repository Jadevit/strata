// crates/backends/llama/llama-plugin/src/batch/batch.rs
//
// Thin RAII wrapper over `llama_batch`.
// - Alloc via FFI helpers
// - Own seq_id tiny buffers so llama.cpp never frees them
// - Safe API: add/mark_last/clear/drop

use crate::ffi::batch as ffi_batch;
use crate::ffi::batch::RawBatch;
use crate::token::LlamaToken;
pub struct LlamaBatch {
    pub(crate) raw: RawBatch,
    pub len: usize,                  // capacity requested at init
    seq_buffers: Vec<Box<[i32; 1]>>, // ownership of per-token seq_id slices
}

impl LlamaBatch {
    /// Create a token-mode batch with capacity `n_tokens`.
    pub fn new(n_tokens: usize) -> Self {
        let raw = ffi_batch::init(n_tokens);
        Self {
            raw,
            len: n_tokens,
            seq_buffers: Vec::new(),
        }
    }

    /// Add one token at position `index`.
    /// - `pos` should be `n_past + index`.
    /// - set `logits=true` only for the last token you want logits for (or call mark_last_for_logits()).
    pub fn add(&mut self, index: usize, token: LlamaToken, pos: i32, logits: bool) {
        assert!(index < self.len, "index {} >= capacity {}", index, self.len);

        // Enforce strictly sequential appends.
        let expected = ffi_batch::n_tokens(&self.raw) as usize;
        assert!(
            index == expected,
            "add() must be sequential: expected index {}, got {}",
            expected,
            index
        );

        ffi_batch::set_token(&mut self.raw, index, token.0);
        ffi_batch::set_pos(&mut self.raw, index, pos);
        ffi_batch::set_logits(&mut self.raw, index, logits);

        // Provide a single sequence id [0] by default.
        let boxed = Box::new([0i32]);
        let ptr = boxed.as_ptr() as *mut i32;
        self.seq_buffers.push(boxed); // keep ownership here
        ffi_batch::set_seq_slot(&mut self.raw, index, ptr, 1);

        ffi_batch::set_n_tokens(&mut self.raw, (index + 1) as i32);
    }

    /// Ensure only the last valid token is marked for logits.
    pub fn mark_last_for_logits(&mut self) {
        let n = ffi_batch::n_tokens(&self.raw);
        if n <= 0 {
            return;
        }
        let n = n as usize;
        ffi_batch::reset_all_logits(&mut self.raw, self.len.min(n));
        ffi_batch::set_logits(&mut self.raw, n - 1, true);
    }

    /// Reset the batch to reuse the underlying storage.
    /// - Clears n_tokens
    /// - Zeros logits flags (if present)
    /// - Nulls seq-id pointers and drops owned seq buffers
    pub fn clear(&mut self) {
        ffi_batch::clear_all_seq_slots(&mut self.raw, self.len);
        ffi_batch::reset_all_logits(&mut self.raw, self.len);

        self.seq_buffers.clear();
        ffi_batch::set_n_tokens(&mut self.raw, 0);
    }
}

impl Drop for LlamaBatch {
    fn drop(&mut self) {
        // Ensure llama doesn’t free our seq buffers
        ffi_batch::clear_all_seq_slots(&mut self.raw, self.len);
        // free the batch
        ffi_batch::free(self.raw);
        // seq_buffers drop naturally here
    }
}
