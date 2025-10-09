// strata-backend-llama/src/params.rs
//
// High-level parameter structs + conversion into llama_context_params,
// aligned with latest llama.cpp (flash_attn_type instead of flash_attn).

use llama_sys::{
    ggml_type, llama_attention_type, llama_context_default_params, llama_context_params,
    llama_flash_attn_type, llama_pooling_type, llama_rope_scaling_type,
};

// =========================
// CONTEXT / RUNTIME PARAMS
// =========================

#[derive(Debug, Clone)]
pub struct LlamaParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32, // we force 1 in to_ffi()
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub rope_scaling_type: llama_rope_scaling_type,
    pub pooling_type: llama_pooling_type,
    pub attention_type: llama_attention_type,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: f32,
    pub type_k: ggml_type,
    pub type_v: ggml_type,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub flash_attn_type: llama_flash_attn_type, // << updated
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
}

impl Default for LlamaParams {
    fn default() -> Self {
        Self {
            n_ctx: 4096,
            n_batch: 512,
            n_ubatch: 4,
            n_seq_max: 1, // single sequence by default
            n_threads: 0,
            n_threads_batch: 0,
            rope_scaling_type: 0, // LLAMA_ROPE_SCALING_NONE
            pooling_type: 0,      // LLAMA_POOLING_TYPE_NONE
            attention_type: 0,    // model default (e.g., SCALE_NORM)
            rope_freq_base: 10000.0,
            rope_freq_scale: 1.0,
            yarn_ext_factor: -1.0,
            yarn_attn_factor: 1.0,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,
            yarn_orig_ctx: 0,
            defrag_thold: 0.0,
            type_k: 1, // GGML_TYPE_F16
            type_v: 1, // GGML_TYPE_F16
            embeddings: false,
            offload_kqv: false,
            flash_attn_type: 0, // LLAMA_FLASH_ATTN_DISABLED
            no_perf: false,
            op_offload: false,
            swa_full: false,
        }
    }
}

impl LlamaParams {
    /// Build FFI params from upstream defaults, then override what we care about.
    /// This keeps us forward-compatible when llama.h adds new fields.
    pub fn to_ffi(&self) -> llama_context_params {
        // 1) start from sane upstream defaults
        let mut p = unsafe { llama_context_default_params() };

        // 2) explicit overrides
        p.n_ctx = self.n_ctx;
        p.n_batch = self.n_batch;
        p.n_ubatch = self.n_ubatch;
        p.n_seq_max = 1; // single sequence for now

        p.n_threads = self.n_threads;
        p.n_threads_batch = self.n_threads_batch;

        p.rope_scaling_type = self.rope_scaling_type;
        p.pooling_type = self.pooling_type;
        p.attention_type = self.attention_type;

        p.rope_freq_base = self.rope_freq_base;
        p.rope_freq_scale = self.rope_freq_scale;

        p.yarn_ext_factor = self.yarn_ext_factor;
        p.yarn_attn_factor = self.yarn_attn_factor;
        p.yarn_beta_fast = self.yarn_beta_fast;
        p.yarn_beta_slow = self.yarn_beta_slow;
        p.yarn_orig_ctx = self.yarn_orig_ctx;

        p.defrag_thold = self.defrag_thold;

        p.cb_eval = None;
        p.cb_eval_user_data = std::ptr::null_mut();

        p.type_k = self.type_k;
        p.type_v = self.type_v;

        p.abort_callback = None;
        p.abort_callback_data = std::ptr::null_mut();

        p.embeddings = self.embeddings;
        p.offload_kqv = self.offload_kqv;

        // New API: flash attention is an enum/type, not a bool.
        p.flash_attn_type = self.flash_attn_type;

        p.no_perf = self.no_perf;
        p.op_offload = self.op_offload;
        p.swa_full = self.swa_full;

        // 3) keep unified KV for single-stream decoding if the field exists
        #[allow(unused_assignments)]
        {
            // Some bindings include this field; if removed upstream, this block compiles away.
            // p.kv_unified = true;
        }

        p
    }
}

// =========================
// SAMPLING PARAMS (used by `sampling.rs`)
// =========================

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub greedy: bool,             // if true, argmax; ignore other knobs
    pub temperature: Option<f32>, // > 0.0
    pub top_k: Option<u32>,       // >= 1
    pub top_p: Option<f32>,       // (0,1]
    pub typical: Option<f32>,     // (0,1] — not used by current chain
    pub penalties: Option<PenaltyParams>,
    pub mirostat: Option<MirostatV1>,    // v1
    pub mirostat_v2: Option<MirostatV2>, // v2
}

#[derive(Debug, Clone)]
pub struct PenaltyParams {
    pub last_n: i32,
    pub repeat: f32,
    pub freq: f32,
    pub presence: f32,
}

#[derive(Debug, Clone)]
pub struct MirostatV1 {
    pub seed: u32,
    pub tau: f32,
    pub eta: f32,
    pub m: i32, // typical sequence length
}

#[derive(Debug, Clone)]
pub struct MirostatV2 {
    pub seed: u32,
    pub tau: f32,
    pub eta: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            greedy: false,
            temperature: Some(0.8),
            top_k: Some(40),
            top_p: Some(0.95),
            typical: None,
            penalties: Some(PenaltyParams {
                last_n: 64,
                repeat: 1.1,
                freq: 0.0,
                presence: 0.0,
            }),
            mirostat: None,
            mirostat_v2: None,
        }
    }
}
