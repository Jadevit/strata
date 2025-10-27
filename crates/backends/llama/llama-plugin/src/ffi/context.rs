// crates/backends/llama/llama-plugin/src/ffi/context.rs
//
// Safe-ish wrappers around llama_sys for context- and vocab-adjacent ops.
// All `unsafe` stays in here; higher layers call these helpers.

use std::{ffi::CStr, ffi::CString, ptr::NonNull, slice};

use llama_sys::{
    llama_context, llama_context_default_params, llama_context_params, llama_decode,
    llama_detokenize, llama_get_embeddings, llama_get_logits, llama_get_memory, llama_memory_clear,
    llama_memory_seq_pos_max, llama_model, llama_model_get_vocab, llama_model_n_embd,
    llama_n_vocab, llama_new_context_with_model, llama_token_eos, llama_token_get_text,
    llama_tokenize,
};

/// Default context params (CPU-friendly baseline).
/// Start from upstream defaults to stay forward-compatible.
pub fn default_context_params() -> llama_context_params {
    let mut p = unsafe { llama_context_default_params() };

    let cores = num_cpus::get().max(1) as i32;

    p.n_ctx = 2048;
    p.n_batch = 512;
    p.n_ubatch = 1;
    p.n_seq_max = 1;

    p.n_threads = cores;
    p.n_threads_batch = cores;

    // Leave flash_attn_type as upstream default unless explicitly set.

    // If bindings expose unified KV, you can enable here:
    #[allow(unused_assignments)]
    {
        // p.kv_unified = true;
    }

    p.embeddings = false;
    p.offload_kqv = false;
    p.no_perf = false;
    p.op_offload = false;
    p.swa_full = false;

    p
}

/// Create a context with explicit parameters (preferred).
pub unsafe fn create_context_with_params(
    model: *mut llama_model,
    params: llama_context_params,
) -> Result<NonNull<llama_context>, String> {
    let ptr = llama_new_context_with_model(model, params);
    NonNull::new(ptr).ok_or_else(|| "llama_new_context_with_model returned null".into())
}

/// Convenience: create a context with the local defaults above.
pub unsafe fn create_context_default(
    model: *mut llama_model,
) -> Result<NonNull<llama_context>, String> {
    let params = default_context_params();
    create_context_with_params(model, params)
}

/// Compute the next KV position from llama’s memory bookkeeping.
#[inline]
pub fn next_position(ctx: *mut llama_context) -> i32 {
    unsafe {
        let mem = llama_get_memory(ctx);
        let pos_max = llama_memory_seq_pos_max(mem, 0);
        if pos_max < 0 {
            0
        } else {
            pos_max + 1
        }
    }
}

/// Clear the KV cache. If `clear_data` is true, also clears data buffers.
#[inline]
pub fn clear_kv(ctx: *mut llama_context, clear_data: bool) {
    unsafe {
        let mem = llama_get_memory(ctx);
        llama_memory_clear(mem, clear_data);
    }
}

/// Borrowed view of current logits. Length == vocab size.
/// SAFETY: caller must ensure `ctx`/`model` outlive the returned slice.
pub fn logits<'a>(ctx: *mut llama_context, model: *mut llama_model) -> &'a [f32] {
    unsafe {
        let ptr = llama_get_logits(ctx);
        debug_assert!(!ptr.is_null(), "llama_get_logits returned null");
        let vocab = llama_model_get_vocab(model);
        let vocab_size = llama_n_vocab(vocab) as usize;
        slice::from_raw_parts(ptr, vocab_size)
    }
}

/// Borrowed view of embeddings. Some contexts return null → None.
/// SAFETY: caller must ensure `ctx`/`model` outlive the returned slice.
pub fn embeddings<'a>(ctx: *mut llama_context, model: *mut llama_model) -> Option<&'a [f32]> {
    unsafe {
        let ptr = llama_get_embeddings(ctx);
        if ptr.is_null() {
            return None;
        }
        let n_embd = llama_model_n_embd(model) as usize;
        Some(slice::from_raw_parts(ptr, n_embd))
    }
}

/// Two-pass tokenize with llama's sizing semantics:
/// - Probe may return +needed *or* -needed
/// - Fill may also return -needed (too small) → resize & retry once
pub fn tokenize(model: *mut llama_model, text: &str) -> Result<Vec<i32>, String> {
    let c_text = CString::new(text).map_err(|e| format!("CString error: {e:?}"))?;
    let vocab = unsafe { llama_model_get_vocab(model) };

    // Probe required size (0 buffer). llama may return +needed or -needed.
    let probe = unsafe {
        llama_tokenize(
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

    // Allocate and fill
    let mut buf = vec![0i32; needed];
    let filled = unsafe {
        llama_tokenize(
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
            llama_tokenize(
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
    Ok(buf)
}

pub fn token_to_str(model: *mut llama_model, id: i32) -> Result<String, String> {
    unsafe {
        let vocab = llama_model_get_vocab(model);
        let ptr = llama_token_get_text(vocab, id);
        if ptr.is_null() {
            return Err("llama_token_get_text returned null".into());
        }
        CStr::from_ptr(ptr)
            .to_str()
            .map(|s| s.to_string())
            .map_err(|e| format!("Invalid UTF-8 from token: {e:?}"))
    }
}

#[inline]
pub fn token_eos(model: *mut llama_model) -> i32 {
    unsafe {
        let vocab = llama_model_get_vocab(model);
        llama_token_eos(vocab)
    }
}

/// Detokenize to raw bytes (preferred for streaming).
pub fn detokenize_bytes(
    model: *mut llama_model,
    ids: &[i32],
    remove_special: bool,
    unparse_special: bool,
) -> Result<Vec<u8>, String> {
    unsafe {
        let vocab = llama_model_get_vocab(model);
        if vocab.is_null() {
            return Err("llama_model_get_vocab returned null".into());
        }

        // Probe: -needed on too small, 0 => nothing to render (valid)
        let probe = llama_detokenize(
            vocab,
            ids.as_ptr(),
            ids.len() as i32,
            std::ptr::null_mut(),
            0,
            remove_special,
            unparse_special,
        );

        if probe == 0 {
            return Ok(Vec::new());
        }
        let needed = if probe < 0 { -probe } else { probe };
        if needed < 0 {
            return Err(format!("llama_detokenize(size probe) failed: {probe}"));
        }

        let cap = (needed as usize).saturating_add(8);
        let mut buf = vec![0u8; cap];

        let got = llama_detokenize(
            vocab,
            ids.as_ptr(),
            ids.len() as i32,
            buf.as_mut_ptr() as *mut _,
            cap as i32,
            remove_special,
            unparse_special,
        );
        if got < 0 {
            return Err(format!("llama_detokenize(fill) failed: {got}"));
        }

        buf.truncate(got as usize);
        Ok(buf)
    }
}

/// Detokenize to UTF-8 String (slower; useful for non-streaming).
pub fn detokenize_string(
    model: *mut llama_model,
    ids: &[i32],
    remove_special: bool,
    unparse_special: bool,
) -> Result<String, String> {
    let bytes = detokenize_bytes(model, ids, remove_special, unparse_special)?;
    String::from_utf8(bytes).map_err(|e| format!("detokenize produced non-UTF-8: {e:?}"))
}

/// Thin safe wrapper for llama_decode.
#[inline]
pub fn decode_batch(ctx: *mut llama_context, batch: llama_sys::llama_batch) -> Result<(), String> {
    let rc = unsafe { llama_decode(ctx, batch) };
    if rc != 0 {
        Err(format!("llama_decode failed: {rc}"))
    } else {
        Ok(())
    }
}
