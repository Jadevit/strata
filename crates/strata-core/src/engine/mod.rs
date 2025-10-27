//! Strata core engine: session orchestration around an LLM backend.

use std::panic;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use crate::format::format::FormattedPrompt;
use crate::memory::SessionMemory;
use strata_abi::backend::{ChatTurn, LLMBackend, Role};
use strata_abi::sampling::SamplingParams;
use strata_abi::token::Token;

// Child modules (private to this crate). They can access private fields here.
mod decode;
mod prefill;
mod utils;

/// Engine = {loaded backend session} + {prompt strategy} + {rolling dialog memory}.
/// One `LLMEngine` is one logical chat session.
pub struct LLMEngine<B: LLMBackend> {
    backend: B,
    sample_params: SamplingParams,
    system_prompt: Option<String>,
    memory: SessionMemory,
    prompt_token_budget: usize,
    stop_flag: Arc<AtomicBool>,
    // ========== KV reuse bookkeeping ==========
    prev_prompt_tokens: Vec<Token>,
    kv_warm: bool,
}

impl<B: LLMBackend> LLMEngine<B> {
    /// Construct with explicit prompt strategy.
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            sample_params: SamplingParams::default(),
            system_prompt: None,
            memory: SessionMemory::new(),
            prompt_token_budget: 3072, // refined in `with_auto`
            stop_flag: Arc::new(AtomicBool::new(false)),
            prev_prompt_tokens: Vec::new(),
            kv_warm: false,
        }
    }

    /// Construct and auto-pick a reasonable strategy from the backend’s flavor hint.
    /// Also derives a prompt token budget from the backend’s `n_ctx` if available.
    pub fn with_auto(backend: B, system: Option<String>) -> Self {
        let mut s = Self::new(backend);
        s.system_prompt = system;

        if let Some(n_ctx) = s.backend.context_window_hint() {
            let budget = ((n_ctx as f32) * 0.75) as usize;
            println!("🧮 context_window_hint = {n_ctx}, prompt_token_budget = {budget}");
            s.set_prompt_token_budget(budget);
        } else {
            println!(
                "🧮 context_window_hint not provided; using default prompt_token_budget = {}",
                s.prompt_token_budget
            );
        }
        s
    }

    /// Set/clear the system prompt used by the formatter (unless the dialog already includes one).
    pub fn set_system_prompt<S: Into<String>>(&mut self, sys: Option<S>) {
        self.system_prompt = sys.map(|s| s.into());
    }

    /// Override the pre-generation prompt token budget.
    pub fn set_prompt_token_budget(&mut self, budget: usize) {
        self.prompt_token_budget = budget.max(1);
    }

    /// Handle you can keep and flip to cancel decoding (`store(true)`).
    pub fn stop_handle(&self) -> Arc<AtomicBool> {
        self.stop_flag.clone()
    }

    /// Explicitly clear any cached KV / sequence state on the backend.
    pub fn clear_kv_cache(&mut self) {
        self.backend.clear_kv_cache();
    }

    #[inline]
    fn clear_stop(&self) {
        self.stop_flag.store(false, Ordering::Relaxed);
    }

    // ─────────────────────────────────────────────
    // Public inference APIs (thin wrappers)
    // ─────────────────────────────────────────────

    /// Stateful single-turn: appends to engine memory, prunes to budget, generates, stores reply.
    pub fn infer(&mut self, user_input: &str) -> Result<String, String> {
        self.memory.push_user(user_input);
        let formatted = self.prune_to_budget_native()?;
        let out = self.infer_with_formatted(formatted)?;
        self.memory.push_assistant(out.clone());
        Ok(out)
    }

    /// Stateless multi-turn (does not mutate engine memory).
    pub fn infer_chat(&mut self, turns: &[ChatTurn]) -> Result<String, String> {
        let formatted = self.format_turns_via_backend(turns)?;
        self.infer_with_formatted(formatted)
    }

    /// Streaming multi-turn. Calls `on_delta` with UTF-8 chunks; also returns the final string.
    pub fn infer_chat_stream<F>(
        &mut self,
        turns: &[ChatTurn],
        on_delta: F,
    ) -> Result<String, String>
    where
        F: FnMut(&str),
    {
        let formatted = self.format_turns_via_backend(turns)?;
        self.stream_with_formatted(formatted, on_delta)
    }

    // ─────────────────────────────────────────────
    // Local helpers kept in the parent (format/budget/limits)
    // ─────────────────────────────────────────────

    /// Decide how many decode steps to allow this turn.
    ///
    /// Env override: `STRATA_MAX_DECODE_TOKENS` (usize) clamps the cap.
    fn compute_step_limit(&self, prompt_len: usize) -> usize {
        let n_ctx = self.backend.context_window_hint().unwrap_or(4096);
        let reserve = ((n_ctx as f32) * 0.02) as usize; // ~2% safety
        let mut step_limit = n_ctx.saturating_sub(prompt_len).saturating_sub(reserve);

        if let Ok(max_decode_str) = std::env::var("STRATA_MAX_DECODE_TOKENS") {
            if let Ok(max_decode) = max_decode_str.parse::<usize>() {
                step_limit = step_limit.min(max_decode.max(1));
            }
        }
        if step_limit == 0 { 32 } else { step_limit }
    }

    fn format_turns_via_backend(&self, turns: &[ChatTurn]) -> Result<FormattedPrompt, String> {
        // Inject system prompt if we have one and caller didn't provide a system turn
        let mut t: Vec<ChatTurn> = Vec::with_capacity(turns.len() + 1);
        let has_sys = turns.iter().any(|tt| matches!(tt.role, Role::System));
        if !has_sys {
            if let Some(sys) = self.system_prompt.as_deref() {
                t.push(ChatTurn::system(sys.to_string()));
            }
        }
        t.extend_from_slice(turns);

        if let Some(text) = self.backend.apply_native_chat_template(&t) {
            let stops = self
                .backend
                .default_stop_strings()
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>();
            Ok(FormattedPrompt {
                text,
                stop_sequences: stops,
                add_space_prefix: true,
            })
        } else {
            Err("No native chat template available for this backend/model; refusing to fall back. Please paste a chat_template or select an explicit formatter in the UI.".into())
        }
    }

    fn prune_to_budget_native(&mut self) -> Result<FormattedPrompt, String> {
        loop {
            let turns = self.memory.turns().to_vec();
            let formatted = self.format_turns_via_backend(&turns)?;
            let toks = self.backend.tokenize(&formatted.text)?;
            if toks.len() <= self.prompt_token_budget {
                return Ok(formatted);
            }
            if !self.memory.drop_oldest_pair() {
                // Can't drop more; proceed anyway with current formatted prompt
                return Ok(formatted);
            }
        }
    }
}

// NOTE: The heavy lifting lives in child modules as `impl LLMEngine<B>`
// with `pub(super)` methods called above:
//
// - prefill.rs:    prefill_incremental(...) + lcp_len(...)
// - decode.rs:     infer_with_formatted(...), stream_with_formatted(...)
// - utils.rs:      utf8_valid_prefix_len(...)
