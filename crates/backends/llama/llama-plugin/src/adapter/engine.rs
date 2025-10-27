use std::path::Path;
use std::sync::Arc;

use crate::{
    adapter::kv::KvState,
    backends::dispatch::Backend as LlamaCppBackend,
    format::format_with_native_template,
    model::LlamaModel,
    params::{
        LlamaParams, MirostatV1, MirostatV2, PenaltyParams as RsPenaltyParams,
        SamplingParams as RsSamplingParams,
    },
    token::LlamaToken,
};

use strata_abi::backend::{LLMBackend, PromptFlavor};
use strata_abi::sampling::{BackendSamplingCapabilities, SamplingParams as CoreSamplingParams};
use strata_abi::token::Token;

/// Llama backend implementation used by the engine.
/// One instance = one loaded model + one inference context (session).
pub struct LlamaBackendImpl {
    /// Resident model weights (shared across spawned sessions).
    model: Arc<LlamaModel>,
    /// Session KV + sequencing.
    kv: KvState,
    /// Params used to create contexts; kept so we can spawn() cheap fresh sessions.
    params: LlamaParams,
}

impl LlamaBackendImpl {
    fn default_params() -> LlamaParams {
        let mut p = LlamaParams::default();
        p.n_ctx = std::env::var("STRATA_N_CTX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);
        p.n_batch = std::env::var("STRATA_N_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);
        p.n_ubatch = std::env::var("STRATA_N_UBATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16);
        p
    }

    pub fn from_model(model: Arc<LlamaModel>, params: LlamaParams) -> Result<Self, String> {
        // SAFETY: Widen &LlamaModel to 'static for context creation. Drop order is kv, then model.
        let static_ref: &'static LlamaModel =
            unsafe { std::mem::transmute::<&LlamaModel, &'static LlamaModel>(model.as_ref()) };

        let kv = KvState::new(static_ref, &params)?;
        Ok(Self { model, kv, params })
    }

    pub fn spawn(&self) -> Result<Self, String> {
        Self::from_model(Arc::clone(&self.model), self.params.clone())
    }
}

impl LLMBackend for LlamaBackendImpl {
    fn load<P: AsRef<Path>>(model_path: P) -> Result<Self, String> {
        let params = Self::default_params();

        let backend =
            LlamaCppBackend::load(&model_path, params.clone()).map_err(|e| format!("{e}"))?;
        let model = backend.model();

        let static_ref = unsafe { std::mem::transmute::<&LlamaModel, &'static LlamaModel>(&model) };
        let kv = KvState::new(static_ref, &params)?;

        Ok(Self { model, kv, params })
    }

    fn tokenize(&self, text: &str) -> Result<Vec<Token>, String> {
        let toks = self
            .model
            .as_ref()
            .tokenize(text)
            .map_err(|e| format!("Tokenizer failed: {e}"))?;
        Ok(toks.into_iter().map(|t| Token(t.0)).collect())
    }

    fn evaluate(&mut self, tokens: &[Token], _n_past: i32) -> Result<(), String> {
        let llama_tokens: Vec<LlamaToken> = tokens.iter().map(|Token(t)| LlamaToken(*t)).collect();
        self.kv.evaluate(&llama_tokens)
    }

    fn sample(
        &mut self,
        _n_past: i32,
        params: &CoreSamplingParams,
        _token_history: &[Token],
    ) -> Result<Token, String> {
        let mut lp = RsSamplingParams::default();
        lp.greedy = params.greedy;
        lp.temperature = params.temperature;
        lp.top_k = params.top_k;
        lp.top_p = params.top_p;
        lp.typical = params.typical_p;

        if let Some(p) = &params.repetition_penalty {
            lp.penalties = Some(RsPenaltyParams {
                last_n: p.last_n,
                repeat: p.repeat,
                freq: p.frequency,
                presence: p.presence,
            });
        }

        if let Some(m) = &params.mirostat {
            match m.version {
                1 => {
                    lp.mirostat = Some(MirostatV1 {
                        seed: 0,
                        tau: m.tau,
                        eta: m.eta,
                        m: m.m.unwrap_or(100),
                    });
                }
                2 => {
                    lp.mirostat_v2 = Some(MirostatV2 {
                        seed: 0,
                        tau: m.tau,
                        eta: m.eta,
                    });
                }
                _ => {}
            }
        }

        let vocab_size = self.model.as_ref().n_vocab();
        let tok = self.kv.sample(vocab_size, &lp)?;
        Ok(Token(tok.0))
    }

    fn decode_token(&self, token: Token) -> Result<String, String> {
        let llama_tok = LlamaToken(token.0);
        self.model
            .as_ref()
            .token_to_str(llama_tok)
            .map_err(|e| format!("Decode failed: {e}"))
    }

    fn eos_token(&self) -> Token {
        Token(self.model.as_ref().token_eos().0)
    }

    fn prompt_flavor(&self) -> PromptFlavor {
        PromptFlavor::ChatMl
    }

    fn default_stop_strings(&self) -> &'static [&'static str] {
        &["<|im_end|>"]
    }

    fn apply_native_chat_template(
        &self,
        turns: &[strata_abi::backend::ChatTurn],
    ) -> Option<String> {
        format_with_native_template(self.model.as_ref(), turns, None, true)
    }

    fn context_window_hint(&self) -> Option<usize> {
        Some(self.kv.capacity())
    }

    fn detokenize_range(
        &self,
        token_history: &[Token],
        start: usize,
        remove_special: bool,
        unparse_special: bool,
    ) -> Result<Vec<u8>, String> {
        let slice = &token_history[start..];
        if slice.is_empty() {
            return Ok(Vec::new());
        }
        let toks: Vec<LlamaToken> = slice.iter().map(|Token(t)| LlamaToken(*t)).collect();
        self.kv
            .detokenize_bytes(&toks, remove_special, unparse_special)
    }

    // ────────────────────────────────────────────────
    // KV cache plumbing (delegated)
    // ────────────────────────────────────────────────

    fn clear_kv_cache(&mut self) {
        println!("🧹 [llama-plugin] Clearing KV cache");
        self.kv.clear();
    }

    fn kv_len_hint(&self) -> Option<usize> {
        Some(self.kv.len())
    }

    fn sampling_capabilities(&self) -> BackendSamplingCapabilities {
        BackendSamplingCapabilities {
            supports_greedy: true,
            supports_temperature: true,
            supports_top_k: true,
            supports_top_p: true,
            supports_typical_p: true,
            supports_tfs_z: false,
            supports_penalties: true,
            supports_mirostat_v1: true,
            supports_mirostat_v2: true,
        }
    }
}
