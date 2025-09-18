// strata-backend-llama/src/ffi.rs
//
// Focused FFI helpers around llama.cpp.
// Uses upstream default_*_params() to avoid breakage when headers change.

use std::{ffi::CString, ptr::NonNull, sync::OnceLock};

use llama_sys::{
    llama_backend_free, llama_backend_init, llama_context, llama_context_default_params,
    llama_context_params, llama_decode, llama_free, llama_free_model, llama_load_model_from_file,
    llama_model, llama_model_default_params, llama_model_params, llama_new_context_with_model,
};

/// one-time flags to prevent double init/deinit
static INIT_CALLED: OnceLock<()> = OnceLock::new();
static DEINIT_CALLED: OnceLock<()> = OnceLock::new();

#[inline]
fn trace(msg: &str) {
    #[cfg(feature = "ffi-trace")]
    println!("{msg}");
}

/// Call exactly once near process start.
pub unsafe fn init_backend() {
    if INIT_CALLED.set(()).is_ok() {
        trace("🧠 [FFI] llama_backend_init()");
        llama_backend_init();
    } else {
        trace("↩️ [FFI] init_backend() called again — ignored");
    }
}

/// Optional: call once on clean shutdown.
pub unsafe fn deinit_backend() {
    if DEINIT_CALLED.set(()).is_ok() {
        trace("🧹 [FFI] llama_backend_free()");
        llama_backend_free();
    } else {
        trace("↩️ [FFI] deinit_backend() called again — ignored");
    }
}

/// Default model params (start from upstream defaults to stay future-proof).
pub fn default_model_params() -> llama_model_params {
    let mut p = unsafe { llama_model_default_params() };

    // Conservative, portable defaults (tweak upstream as needed):
    // Keep mmap for fast load; verify tensor shapes.
    p.use_mmap = true;
    p.check_tensors = true;

    p
}

/// Default context params (CPU-friendly baseline).
/// NOTE: Prefer using your higher-level `LlamaParams::to_ffi()` when possible.
/// This exists for simple contexts created directly via FFI.
pub fn default_context_params() -> llama_context_params {
    let mut p = unsafe { llama_context_default_params() };

    let cores = num_cpus::get().max(1) as i32;

    // Reasonable CPU defaults; upstream will fill any new fields for us.
    p.n_ctx = 2048;
    p.n_batch = 512;
    p.n_ubatch = 1;
    p.n_seq_max = 1;

    p.n_threads = cores;
    p.n_threads_batch = cores;

    // We do not touch flash_attn_type here; stick with upstream default.
    // p.flash_attn_type = <leave as provided by llama_context_default_params()>

    // Single-stream decoding tends to benefit from unified KV.
    // If upstream removes/changes this field, leaving it untouched is fine.
    #[allow(unused_assignments)]
    {
        // some bindings expose kv_unified; if not present, this compiles away
        // (leave the block to avoid warnings if the field disappears)
        // p.kv_unified = true;
    }

    p.embeddings = false;
    p.offload_kqv = false;
    p.no_perf = false;
    p.op_offload = false;
    p.swa_full = false;

    p
}

/// Load a model from disk. Caller owns the returned handle.
pub unsafe fn load_model(path: &str) -> Result<NonNull<llama_model>, String> {
    trace(&format!("📦 [FFI] load_model: {path}"));
    let c_path = CString::new(path).map_err(|_| "Invalid model path".to_string())?;
    let ptr = llama_load_model_from_file(c_path.as_ptr(), default_model_params());
    NonNull::new(ptr).ok_or_else(|| "llama_load_model_from_file returned null".into())
}

/// Create a context for an existing model. Caller owns the returned handle.
/// If you need custom params, build them via your params layer and call
/// llama_new_context_with_model(model, custom_params) instead.
pub unsafe fn create_context(model: *mut llama_model) -> Result<NonNull<llama_context>, String> {
    trace("📦 [FFI] create_context()");
    let params = default_context_params();
    let ptr = llama_new_context_with_model(model, params);
    NonNull::new(ptr).ok_or_else(|| "llama_new_context_with_model returned null".into())
}

/// Decode a batch on the given context.
pub unsafe fn decode_batch(
    ctx: *mut llama_context,
    batch: llama_sys::llama_batch,
) -> Result<(), String> {
    let rc = llama_decode(ctx, batch);
    if rc != 0 {
        Err(format!("llama_decode failed: {}", rc))
    } else {
        Ok(())
    }
}

/// Free a model instance.
pub unsafe fn cleanup_model(model: *mut llama_model) {
    trace("🧹 [FFI] llama_free_model()");
    llama_free_model(model);
}

/// Free a context instance.
pub unsafe fn cleanup_context(ctx: *mut llama_context) {
    trace("🧹 [FFI] llama_free(context)");
    llama_free(ctx);
}
