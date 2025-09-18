// strata-backend-llama/src/batch.rs
//
// Thin RAII wrapper over `llama_batch`.
// - Alloc via `llama_batch_init`
// - We own the `seq_id` small buffers and prevent llama.cpp from freeing them.
// - Safe to reuse with `clear()`, or drop to free underlying storage.

use std::ptr;

use crate::token::LlamaToken;
use llama_sys::{llama_batch, llama_batch_free, llama_batch_init};

pub struct LlamaBatch {
    pub raw: llama_batch,
    pub len: usize,                  // capacity requested at init
    seq_buffers: Vec<Box<[i32; 1]>>, // holds ownership for per-token seq_id slices
}

impl LlamaBatch {
    /// Create a token-based batch with capacity `n_tokens`.
    /// `embd = 0` (token mode), `n_seq_max = 1` (single sequence).
    pub fn new(n_tokens: usize) -> Self {
        let raw = unsafe { llama_batch_init(n_tokens as i32, 0, 1) };
        Self {
            raw,
            len: n_tokens,
            seq_buffers: Vec::new(),
        }
    }

    /// Add one token at position `index`.
    /// - `pos` should be `n_past + index` at call site.
    /// - `logits = true` only for the last token you want logits for (use `mark_last_for_logits`).
    pub fn add(&mut self, index: usize, token: LlamaToken, pos: i32, logits: bool) {
        assert!(index < self.len, "index {} >= capacity {}", index, self.len);
        // Enforce strictly sequential appends: avoids gaps / inconsistent n_tokens.
        assert!(
            index == self.raw.n_tokens as usize,
            "add() must be sequential: expected index {}, got {}",
            self.raw.n_tokens,
            index
        );

        unsafe {
            *self.raw.token.add(index) = token.0;
            *self.raw.pos.add(index) = pos;

            if !self.raw.logits.is_null() {
                *self.raw.logits.add(index) = logits as i8;
            }

            // Provide a single sequence id [0] by default.
            if !self.raw.seq_id.is_null() && !self.raw.n_seq_id.is_null() {
                let boxed = Box::new([0i32]);
                let ptr = boxed.as_ptr() as *mut i32;
                self.seq_buffers.push(boxed);

                *self.raw.seq_id.add(index) = ptr;
                *self.raw.n_seq_id.add(index) = 1;
            }
        }

        self.raw.n_tokens = (index + 1) as i32;
    }

    /// Ensure only the last valid token is marked for logits.
    /// Safe to call after a series of `add()` calls.
    pub fn mark_last_for_logits(&mut self) {
        if self.raw.n_tokens <= 0 || self.raw.logits.is_null() {
            return;
        }
        unsafe {
            // zero out all
            for i in 0..(self.raw.n_tokens as usize) {
                *self.raw.logits.add(i) = 0;
            }
            // set last = 1
            *self.raw.logits.add(self.raw.n_tokens as usize - 1) = 1;
        }
    }

    /// Reset the batch so it can be reused.
    /// - Clears `n_tokens`
    /// - Zeros logits flags (if present)
    /// - Nulls out seq-id pointers and drops owned seq buffers
    pub fn clear(&mut self) {
        // Null seq pointers and counts so llama won’t try to free what we own.
        unsafe {
            if !self.raw.seq_id.is_null() && !self.raw.n_seq_id.is_null() {
                for i in 0..self.len {
                    *self.raw.seq_id.add(i) = ptr::null_mut();
                    *self.raw.n_seq_id.add(i) = 0;
                }
            }
            if !self.raw.logits.is_null() {
                for i in 0..self.len {
                    *self.raw.logits.add(i) = 0;
                }
            }
        }

        // Drop our owned seq buffers; next `add()` will recreate as needed.
        self.seq_buffers.clear();
        self.raw.n_tokens = 0;
    }
}

impl Drop for LlamaBatch {
    fn drop(&mut self) {
        unsafe {
            // Prevent llama from freeing our owned seq buffers on free:
            if !self.raw.seq_id.is_null() && !self.raw.n_seq_id.is_null() {
                for i in 0..self.len {
                    *self.raw.seq_id.add(i) = ptr::null_mut();
                    *self.raw.n_seq_id.add(i) = 0;
                }
            }
            llama_batch_free(self.raw);
        }
        // `seq_buffers` drops here (we own that memory)
    }
}
