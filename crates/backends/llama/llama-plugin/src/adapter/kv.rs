// crates/backends/llama/llama-plugin/src/kv.rs

use crate::{context::LlamaContext, model::LlamaModel, params::LlamaParams, token::LlamaToken};

pub struct KvState {
    ctx: LlamaContext<'static>,
    n_ctx: usize,
}

impl KvState {
    /// Create a new KV/context from a model + params.
    ///
    /// SAFETY: Caller must ensure the backing model outlives `self.ctx`.
    /// In our backend struct, `kv` is declared before `model`, so `kv` drops first,
    /// guaranteeing the context dies before the Arc<LlamaModel>.
    pub fn new(model: &'static LlamaModel, params: &LlamaParams) -> Result<Self, String> {
        let ctx = model
            .create_context(params.to_ffi(), false)
            .map_err(|e| format!("Failed to create context: {e}"))?;
        let n_ctx = ctx.n_ctx as usize;
        Ok(Self { ctx, n_ctx })
    }

    /// Advance KV with a batch of tokens.
    pub fn evaluate(&mut self, tokens: &[LlamaToken]) -> Result<(), String> {
        let n_past = self.ctx.next_position();
        self.ctx
            .evaluate_mut(tokens, n_past)
            .map_err(|e| format!("Evaluate failed: {e}"))
    }

    /// Detokenize to UTF-8 bytes.
    pub fn detokenize_bytes(
        &self,
        tokens: &[LlamaToken],
        remove_special: bool,
        unparse_special: bool,
    ) -> Result<Vec<u8>, String> {
        self.ctx
            .detokenize_bytes(tokens, remove_special, unparse_special)
    }

    /// Sample next token using llama-rs helper.
    pub fn sample(
        &self,
        vocab_size: usize,
        params: &crate::params::SamplingParams,
    ) -> Result<LlamaToken, String> {
        crate::sampling::sample_with_params(&self.ctx, vocab_size, params)
    }

    /// Clear resident KV.
    pub fn clear(&mut self) {
        self.ctx.clear_kv_cache();
    }

    /// Current tokens cached.
    pub fn len(&self) -> usize {
        self.ctx.next_position() as usize
    }

    /// Context window capacity.
    pub fn capacity(&self) -> usize {
        self.n_ctx
    }
}
