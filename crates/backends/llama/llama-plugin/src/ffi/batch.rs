// crates/backends/llama/llama-plugin/src/ffi/batch.rs
//
// Safe wrappers around llama_sys batch fiddling.
// All raw pointer writes live here.

use llama_sys::{llama_batch, llama_batch_free, llama_batch_init};
use std::ptr;

pub use llama_sys::llama_batch as RawBatch;

/// Initialize a token-mode batch with capacity `n_tokens`, embd=0, n_seq_max=1.
pub fn init(n_tokens: usize) -> llama_batch {
    unsafe { llama_batch_init(n_tokens as i32, 0, 1) }
}

/// Free a batch previously returned by `init`.
pub fn free(batch: llama_batch) {
    unsafe { llama_batch_free(batch) }
}

#[inline]
pub fn n_tokens(raw: &llama_batch) -> i32 {
    raw.n_tokens
}

#[inline]
pub fn set_n_tokens(raw: &mut llama_batch, v: i32) {
    raw.n_tokens = v;
}

pub fn set_token(raw: &mut llama_batch, index: usize, token: i32) {
    unsafe {
        *raw.token.add(index) = token;
    }
}

pub fn set_pos(raw: &mut llama_batch, index: usize, pos: i32) {
    unsafe {
        *raw.pos.add(index) = pos;
    }
}

/// Set logits flag for a slot (no-op if logits buffer is null).
pub fn set_logits(raw: &mut llama_batch, index: usize, flag: bool) {
    unsafe {
        if !raw.logits.is_null() {
            *raw.logits.add(index) = flag as i8;
        }
    }
}

/// Zero all logits flags up to `len` (no-op if logits buffer is null).
pub fn reset_all_logits(raw: &mut llama_batch, len: usize) {
    unsafe {
        if !raw.logits.is_null() {
            for i in 0..len {
                *raw.logits.add(i) = 0;
            }
        }
    }
}

/// Set seq-id slot to a caller-managed pointer and count (no-op if fields are null).
pub fn set_seq_slot(raw: &mut llama_batch, index: usize, ptr: *mut i32, count: i32) {
    unsafe {
        if !raw.seq_id.is_null() && !raw.n_seq_id.is_null() {
            *raw.seq_id.add(index) = ptr;
            *raw.n_seq_id.add(index) = count;
        }
    }
}

/// Null all seq-id slots (no-op if fields are null).
pub fn clear_all_seq_slots(raw: &mut llama_batch, len: usize) {
    unsafe {
        if !raw.seq_id.is_null() && !raw.n_seq_id.is_null() {
            for i in 0..len {
                *raw.seq_id.add(i) = ptr::null_mut();
                *raw.n_seq_id.add(i) = 0;
            }
        }
    }
}
