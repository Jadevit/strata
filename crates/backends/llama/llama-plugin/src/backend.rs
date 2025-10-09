use std::path::Path;
use std::sync::Arc;

use llama_rs::{
    backends::dispatch::Backend as LlamaCppBackend,
    context::LlamaContext,
    model::LlamaModel,
    params::{
        LlamaParams, MirostatV1, MirostatV2, PenaltyParams as RsPenaltyParams,
        SamplingParams as RsSamplingParams,
    },
    sampling::sample_with_params as rs_sample_with_params,
    token::LlamaToken,
};

use strata_abi::backend::{ChatTurn, LLMBackend, PromptFlavor};
use strata_abi::sampling::SamplingParams as CoreSamplingParams;
use strata_abi::token::Token;

/// Llama backend implementation used by the engine.
/// One instance = one loaded model + one inference context (session).
pub struct LlamaBackendImpl {
    /// Resident model weights (shared across spawned sessions).
    model: Arc<LlamaModel>,
    /// Decode context bound to this session (lifetime widened safely inside constructor).
    context: LlamaContext<'static>,
    /// Params used to create contexts; kept so we can spawn() cheap fresh sessions.
    params: LlamaParams,
}

impl LlamaBackendImpl {
    fn default_params() -> LlamaParams {
        let mut p = LlamaParams::default();

        if let Ok(v) = std::env::var("STRATA_N_CTX") {
            if let Ok(n) = v.parse::<u32>() {
                p.n_ctx = n;
            }
        } else {
            p.n_ctx = 4096;
        }

        if let Ok(v) = std::env::var("STRATA_N_BATCH") {
            if let Ok(n) = v.parse::<u32>() {
                p.n_batch = n;
            }
        } else {
            p.n_batch = 64;
        }

        if let Ok(v) = std::env::var("STRATA_N_UBATCH") {
            if let Ok(n) = v.parse::<u32>() {
                p.n_ubatch = n;
            }
        } else {
            p.n_ubatch = 16;
        }

        p
    }

    pub fn from_model(model: Arc<LlamaModel>, params: LlamaParams) -> Result<Self, String> {
        let static_ref: &'static LlamaModel =
            unsafe { std::mem::transmute::<&LlamaModel, &'static LlamaModel>(model.as_ref()) };

        let ctx_params = params.to_ffi();
        let context = static_ref
            .create_context(ctx_params, false)
            .map_err(|e| format!("Failed to create context: {e}"))?;

        Ok(Self {
            model,
            context,
            params,
        })
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
        let ctx_params = params.to_ffi();
        let context = static_ref
            .create_context(ctx_params, false)
            .map_err(|e| format!("Failed to create context: {e}"))?;

        Ok(Self {
            model,
            context,
            params,
        })
    }

    fn tokenize(&self, text: &str) -> Result<Vec<Token>, String> {
        let toks = self
            .model
            .as_ref()
            .tokenize(text)
            .map_err(|e| format!("Tokenizer failed: {e}"))?;
        Ok(toks.into_iter().map(|t| Token(t.0)).collect())
    }

    fn evaluate(&mut self, tokens: &[Token], n_past: i32) -> Result<(), String> {
        let llama_tokens: Vec<LlamaToken> = tokens.iter().map(|Token(t)| LlamaToken(*t)).collect();
        self.context
            .evaluate_mut(&llama_tokens, n_past)
            .map_err(|e| format!("Evaluate failed: {e}"))
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
        let tok = rs_sample_with_params(&self.context, vocab_size, &lp)?;
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

    fn apply_native_chat_template(&self, _turns: &[ChatTurn]) -> Option<String> {
        None
    }

    fn context_window_hint(&self) -> Option<usize> {
        Some(self.context.n_ctx as usize)
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

        self.context
            .detokenize_bytes(&toks, remove_special, unparse_special)
    }
}
