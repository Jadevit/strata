use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::sampling::SamplingParams;
use crate::token::Token;

/// Role label for a chat turn (backend-agnostic).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
}

/// High-level prompt wrapper the engine should use when a backend
/// doesn't provide (or we choose not to use) a native chat template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PromptFlavor {
    /// ChatML-style role blocks: `<|im_start|>user ...`
    ChatMl,
    /// Llama-esque instruction blocks: `<s>[INST] user ... [/INST] assistant:`
    InstBlock,
    /// Plain "User:\nAssistant:" roles.
    UserAssistant,
    /// No wrapping at all.
    Plain,
    /// Phi-3 formatting (matches MS Phi-3 Instruct expectations).
    Phi3,
}

/// A single chat turn (role + content).
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

/// Backend-agnostic interface for inference engines (llama.cpp, ggml, transformers, …).
/// A single implementation instance represents a **loaded model** and a **live eval session**.
pub trait LLMBackend {
    /// Load the model artifact at `model_path` and create a session ready for `evaluate()`.
    fn load<P: AsRef<Path>>(model_path: P) -> Result<Self, String>
    where
        Self: Sized;

    /// Tokenize UTF-8 text into model token IDs.
    fn tokenize(&self, text: &str) -> Result<Vec<Token>, String>;

    /// Run one forward pass over `tokens` continuing from `n_past` tokens of history.
    /// Implementations should update internal KV-cache/state.
    fn evaluate(&mut self, tokens: &[Token], n_past: i32) -> Result<(), String>;

    /// Sample the next token based on current logits + `SamplingParams`.
    /// Implementations should read the current logits from the session's state.
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

    /// Model’s EOS (end-of-sequence) token.
    fn eos_token(&self) -> Token;

    /// Active context window (n_ctx) if known. Used to size prompt budgets.
    fn context_window_hint(&self) -> Option<usize> {
        None
    }

    // ========== OPTIONAL HOOKS ==========

    /// If the backend/model exposes a native chat template (e.g. GGUF tokenizer chat_template),
    /// it can format the given turns into a final prompt. Return `None` to let core wrap.
    fn apply_native_chat_template(&self, _turns: &[ChatTurn]) -> Option<String> {
        None
    }

    /// Backend-provided default stop strings (end-of-turn sentinels).
    fn default_stop_strings(&self) -> &'static [&'static str] {
        &[]
    }

    /// UTF-8 detokenizer for a range inside `token_history`. Defaults to a slow but safe
    /// loop over `decode_token()`. Backends should override to use native detokenizers.
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
                    .map_err(|e| format!("fallback detokenize failed: {e}"))?,
            );
        }
        Ok(s.into_bytes())
    }
}
