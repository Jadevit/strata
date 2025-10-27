use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::sampling::{BackendSamplingCapabilities, SamplingParams};
use crate::token::Token;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PromptFlavor {
    ChatMl,
    InstBlock,
    UserAssistant,
    Plain,
    Phi3,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTurn {
    pub role: Role,
    pub content: String,
}
impl ChatTurn {
    #[inline]
    pub fn system<S: Into<String>>(s: S) -> Self {
        Self {
            role: Role::System,
            content: s.into(),
        }
    }
    #[inline]
    pub fn user<S: Into<String>>(s: S) -> Self {
        Self {
            role: Role::User,
            content: s.into(),
        }
    }
    #[inline]
    pub fn assistant<S: Into<String>>(s: S) -> Self {
        Self {
            role: Role::Assistant,
            content: s.into(),
        }
    }
}

/// Backend-agnostic interface for inference engines.
pub trait LLMBackend {
    fn load<P: AsRef<Path>>(model_path: P) -> Result<Self, String>
    where
        Self: Sized;

    fn tokenize(&self, text: &str) -> Result<Vec<Token>, String>;

    fn evaluate(&mut self, tokens: &[Token], n_past: i32) -> Result<(), String>;

    fn sample(
        &mut self,
        n_past: i32,
        params: &SamplingParams,
        token_history: &[Token],
    ) -> Result<Token, String>;

    /// Optional hint so core can choose a reasonable generic prompt wrapper.
    fn prompt_flavor(&self) -> PromptFlavor {
        PromptFlavor::ChatMl
    }

    /// Decode a single token ID into a UTF-8 fragment.
    fn decode_token(&self, token: Token) -> Result<String, String>;

    /// Model’s EOS token.
    fn eos_token(&self) -> Token;

    /// Active context window (n_ctx) if known.
    fn context_window_hint(&self) -> Option<usize> {
        None
    }

    // ========== OPTIONAL HOOKS ==========

    /// If the backend exposes a native chat template (e.g., GGUF chat_template),
    /// return the fully formatted prompt text for `turns`.
    /// Return `None` to let core wrap.
    fn apply_native_chat_template(&self, _turns: &[ChatTurn]) -> Option<String> {
        None
    }

    /// Backend-provided default stop strings (for UI/logging; core doesn’t enforce).
    fn default_stop_strings(&self) -> &'static [&'static str] {
        &[]
    }

    /// Clear any cached sequence/KV state while keeping the model loaded.
    fn clear_kv_cache(&mut self) {}

    /// Current KV length if known (debug/telemetry).
    fn kv_len_hint(&self) -> Option<usize> {
        None
    }

    /// Report what sampler controls are supported.
    fn sampling_capabilities(&self) -> BackendSamplingCapabilities {
        BackendSamplingCapabilities::default()
    }

    /// Detokenize a sub-range to UTF-8 bytes (override with native detokenizer if available).
    fn detokenize_range(
        &self,
        token_history: &[Token],
        start: usize,
        _remove_special: bool,
        _unparse_special: bool,
    ) -> Result<Vec<u8>, String> {
        let mut s = String::new();
        for tok in &token_history[start..] {
            s.push_str(
                &self
                    .decode_token(*tok)
                    .map_err(|e| format!("fallback detok: {e}"))?,
            );
        }
        Ok(s.into_bytes())
    }
}
