use std::panic;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use crate::format::prompt_format::{PromptKind, select_prompt};
use crate::format::prompting::{FormattedPrompt, PromptStrategy, normalize_bpe_markers};
use crate::memory::SessionMemory;
use crate::traits::backend::{ChatTurn, LLMBackend, PromptFlavor, Role};
use crate::traits::sampling::SamplingParams;
use crate::traits::token::Token;

/// Engine = {loaded backend session} + {prompt strategy} + {rolling dialog memory}.
/// One `LLMEngine` is one logical chat session.
pub struct LLMEngine<B: LLMBackend> {
    backend: B,
    sample_params: SamplingParams,
    prompt_strategy: Box<dyn PromptStrategy>,

    /// Optional system prompt injected on each call (unless a system is found in the turns).
    system_prompt: Option<String>,

    /// Rolling, short-term dialog memory (used by the stateful `infer()` path).
    memory: SessionMemory,

    /// Soft cap for the *prompt* token count. Keep this below n_ctx to leave room for output.
    prompt_token_budget: usize,

    /// STOP flag (flipped by the host/UI to cancel mid-generation).
    stop_flag: Arc<AtomicBool>,
}

impl<B: LLMBackend> LLMEngine<B> {
    /// Construct with explicit prompt strategy.
    pub fn new(backend: B, prompt_strategy: Box<dyn PromptStrategy>) -> Self {
        Self {
            backend,
            sample_params: SamplingParams::default(),
            prompt_strategy,
            system_prompt: None,
            memory: SessionMemory::new(),
            prompt_token_budget: 3072, // reasonable default; refined in `with_auto`
            stop_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Construct and auto-pick a reasonable strategy from the backend’s flavor hint.
    /// Also derives a prompt token budget from the backend’s `n_ctx` if available.
    pub fn with_auto(backend: B, system: Option<String>) -> Self {
        let kind = match backend.prompt_flavor() {
            PromptFlavor::ChatMl => PromptKind::ChatMl {
                system: system.clone(),
            },
            PromptFlavor::InstBlock => PromptKind::InstBlock,
            PromptFlavor::UserAssistant => PromptKind::UserAssistant,
            PromptFlavor::Plain => PromptKind::Plain,
            PromptFlavor::Phi3 => PromptKind::Phi3 {
                system: system.clone(),
            },
        };

        let strategy = select_prompt(kind);
        let mut s = Self::new(backend, strategy);
        s.system_prompt = system;

        // If backend exposes n_ctx, budget ~75% for the prompt and leave ~25% for output.
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

    /// Swap the prompt strategy at runtime.
    pub fn set_strategy(&mut self, kind: PromptKind) {
        self.prompt_strategy = select_prompt(kind);
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

    #[inline]
    fn clear_stop(&self) {
        self.stop_flag.store(false, Ordering::Relaxed);
    }

    // ────────────────────────────────────────────────────────────────────────────────
    // Public inference APIs
    // ────────────────────────────────────────────────────────────────────────────────

    /// Stateful single-turn: appends to engine memory, prunes to budget, generates, stores reply.
    pub fn infer(&mut self, user_input: &str) -> Result<String, String> {
        self.memory.push_user(user_input);
        let sys = self.system_prompt.as_deref();
        let mut formatted = self.prompt_strategy.format_dialog(self.memory.turns(), sys);
        self.prune_to_budget(&mut formatted)?;
        let out = self.infer_with_formatted(formatted)?;
        self.memory.push_assistant(out.clone());
        Ok(out)
    }

    /// Stateless multi-turn (does not mutate engine memory).
    pub fn infer_chat(&mut self, turns: &[ChatTurn]) -> Result<String, String> {
        if let Some(prompt) = self.backend.apply_native_chat_template(turns) {
            return self.infer_with_formatted(FormattedPrompt {
                text: prompt,
                stop_sequences: vec![],
                add_space_prefix: true,
            });
        }

        let sys = self
            .system_prompt
            .as_deref()
            .or_else(|| system_from_turns(turns));

        let formatted = self.prompt_strategy.format_dialog(turns, sys);
        self.infer_with_formatted(formatted)
    }

    /// Streaming multi-turn. Calls `on_delta` with UTF-8 chunks; also returns the final string.
    pub fn infer_chat_stream<F>(
        &mut self,
        turns: &[ChatTurn],
        mut on_delta: F,
    ) -> Result<String, String>
    where
        F: FnMut(&str),
    {
        if let Some(prompt) = self.backend.apply_native_chat_template(turns) {
            return self.stream_with_formatted(
                FormattedPrompt {
                    text: prompt,
                    stop_sequences: vec![],
                    add_space_prefix: true,
                },
                on_delta,
            );
        }

        let sys = self
            .system_prompt
            .as_deref()
            .or_else(|| system_from_turns(turns));

        let formatted = self.prompt_strategy.format_dialog(turns, sys);
        self.stream_with_formatted(formatted, on_delta)
    }

    // ────────────────────────────────────────────────────────────────────────────────
    // Internal helpers
    // ────────────────────────────────────────────────────────────────────────────────

    /// Trim oldest dialog until the formatted prompt fits the token budget.
    fn prune_to_budget(&mut self, formatted: &mut FormattedPrompt) -> Result<(), String> {
        let mut toks = self.backend.tokenize(&formatted.text)?;
        if toks.len() <= self.prompt_token_budget {
            return Ok(());
        }

        // Drop the oldest (User, Assistant) pair (or single non-system) until we fit.
        loop {
            if !self.memory.drop_oldest_pair() {
                break;
            }
            let sys = self.system_prompt.as_deref();
            *formatted = self.prompt_strategy.format_dialog(self.memory.turns(), sys);
            toks = self.backend.tokenize(&formatted.text)?;
            if toks.len() <= self.prompt_token_budget {
                break;
            }
        }
        Ok(())
    }

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

    /// Prefill in 64-token chunks. Returns `(n_past, token_history, detok_start_idx)`.
    fn prefill(&mut self, prompt_tokens: &[Token]) -> Result<(i32, Vec<Token>, usize), String> {
        const PREFILL_CHUNK: usize = 64; // must be <= backend batch size
        let mut n_past: i32 = 0;
        let mut token_history: Vec<Token> =
            Vec::with_capacity(prompt_tokens.len().saturating_add(1024));

        for (i, chunk) in prompt_tokens.chunks(PREFILL_CHUNK).enumerate() {
            if self.stop_flag.load(Ordering::Relaxed) {
                println!("⏹️ [prefill] STOP requested during prefill.");
                break;
            }
            println!(
                "⚙️ [evaluate] Prefill chunk {i} (len {}), n_past = {n_past}",
                chunk.len()
            );
            self.backend
                .evaluate(chunk, n_past)
                .map_err(|e| format!("❌ [infer] Initial prefill failed: {e}"))?;
            token_history.extend_from_slice(chunk);
            n_past += chunk.len() as i32;
        }
        println!("✅ [prefill] Done ({} tokens)", n_past);

        let detok_start_idx = token_history.len(); // start detokenizing *after* the prompt
        Ok((n_past, token_history, detok_start_idx))
    }

    // ────────────────────────────────────────────────────────────────────────────────
    // Non-streaming core
    // ────────────────────────────────────────────────────────────────────────────────

    fn infer_with_formatted(&mut self, formatted: FormattedPrompt) -> Result<String, String> {
        self.clear_stop();

        panic::catch_unwind(panic::AssertUnwindSafe(|| {
            println!("🧠 [infer] Starting inference");
            println!("🧾 [infer] Formatted prompt: {}", formatted.text);

            // Tokenize full prompt.
            let prompt_tokens = self
                .backend
                .tokenize(&formatted.text)
                .map_err(|e| format!("❌ [infer] Tokenization failed: {e}"))?;
            println!(
                "🔤 [infer] Tokenized input ({} tokens)",
                prompt_tokens.len()
            );

            // Dynamic decode cap.
            let step_limit = self.compute_step_limit(prompt_tokens.len());
            println!("🧮 step_limit={}", step_limit);

            // Prefill (STOP-aware).
            let (mut n_past, mut token_history, mut detok_start_idx) =
                self.prefill(&prompt_tokens)?;

            // UTF-8 streaming state (accumulate valid prefix only).
            let mut out_text = String::new();
            let mut staging_bytes: Vec<u8> = Vec::with_capacity(4096);

            // Decode loop (STOP-aware).
            for step in 0..step_limit {
                if self.stop_flag.load(Ordering::Relaxed) {
                    println!("⏹️ [infer] STOP requested. Ending.");
                    break;
                }
                println!("🔁 [infer] Step {}", step);

                let token = self
                    .backend
                    .sample(n_past, &self.sample_params, &token_history)
                    .map_err(|e| format!("❌ [infer] Sampling failed: {e}"))?;
                println!("🎯 [infer] Sampled token: {:?}", token);

                if token == self.backend.eos_token() {
                    println!("🏁 [infer] Reached EOS token. Ending.");
                    break;
                }

                self.backend
                    .evaluate(&[token], n_past)
                    .map_err(|e| format!("❌ [infer] Re-eval failed at step {step}: {e}"))?;
                token_history.push(token);
                n_past += 1;

                // Detokenize only the new range; emit valid UTF-8 prefix.
                let new_bytes = self.backend.detokenize_range(
                    &token_history,
                    detok_start_idx,
                    /*remove_special*/ true,
                    /*unparse_special*/ false,
                )?;
                if !new_bytes.is_empty() {
                    staging_bytes.extend_from_slice(&new_bytes);
                    let valid_len = utf8_valid_prefix_len(&staging_bytes);
                    if valid_len > 0 {
                        let taken = staging_bytes.drain(..valid_len).collect::<Vec<u8>>();
                        let mut delta = String::from_utf8(taken)
                            .map_err(|e| format!("detokenize produced non-UTF-8: {e}"))?;
                        delta = normalize_bpe_markers(&delta);
                        out_text.push_str(&delta);
                        detok_start_idx = token_history.len();
                    }
                }

                // TODO: honor `formatted.stop_sequences` if you want early stopping on templates.
            }

            let out_text = out_text.trim().to_string();
            println!(
                "✅ [infer] Complete. Output length: {} chars",
                out_text.len()
            );
            Ok(out_text)
        }))
        .map_err(|_| "💥 [infer] PANIC occurred during inference!".to_string())?
    }

    // ────────────────────────────────────────────────────────────────────────────────
    // Streaming core
    // ────────────────────────────────────────────────────────────────────────────────

    fn stream_with_formatted<F>(
        &mut self,
        formatted: FormattedPrompt,
        mut on_delta: F,
    ) -> Result<String, String>
    where
        F: FnMut(&str),
    {
        self.clear_stop();

        panic::catch_unwind(panic::AssertUnwindSafe(|| {
            println!("🧠 [infer-stream] Starting inference");
            println!("🧾 [infer-stream] Formatted prompt: {}", formatted.text);

            // Tokenize full prompt.
            let prompt_tokens = self
                .backend
                .tokenize(&formatted.text)
                .map_err(|e| format!("❌ [infer-stream] Tokenization failed: {e}"))?;
            println!(
                "🔤 [infer-stream] Tokenized input ({} tokens)",
                prompt_tokens.len()
            );

            // Dynamic decode cap.
            let step_limit = self.compute_step_limit(prompt_tokens.len());
            println!("🧮 [stream] step_limit={}", step_limit);

            // Prefill (STOP-aware).
            let (mut n_past, mut token_history, mut detok_start_idx) =
                self.prefill(&prompt_tokens)?;

            // UTF-8 streaming state.
            let mut out_text = String::new();
            let mut staging_bytes: Vec<u8> = Vec::with_capacity(4096);

            // Decode loop (STOP-aware).
            for step in 0..step_limit {
                if self.stop_flag.load(Ordering::Relaxed) {
                    println!("⏹️ [infer-stream] STOP requested. Ending.");
                    break;
                }
                println!("🔁 [infer-stream] Step {}", step);

                let token = self
                    .backend
                    .sample(n_past, &self.sample_params, &token_history)
                    .map_err(|e| format!("❌ [infer-stream] Sampling failed: {e}"))?;
                println!("🎯 [infer-stream] Sampled token: {:?}", token);

                if token == self.backend.eos_token() {
                    println!("🏁 [infer-stream] Reached EOS token. Ending.");
                    break;
                }

                self.backend
                    .evaluate(&[token], n_past)
                    .map_err(|e| format!("❌ [infer-stream] Re-eval failed at step {step}: {e}"))?;
                token_history.push(token);
                n_past += 1;

                // Detokenize only the new range; emit valid UTF-8 prefix.
                let new_bytes = self.backend.detokenize_range(
                    &token_history,
                    detok_start_idx,
                    /*remove_special*/ true,
                    /*unparse_special*/ false,
                )?;
                if !new_bytes.is_empty() {
                    staging_bytes.extend_from_slice(&new_bytes);
                    let valid_len = utf8_valid_prefix_len(&staging_bytes);
                    if valid_len > 0 {
                        let taken = staging_bytes.drain(..valid_len).collect::<Vec<u8>>();
                        let mut delta = String::from_utf8(taken)
                            .map_err(|e| format!("detokenize produced non-UTF-8: {e}"))?;
                        delta = normalize_bpe_markers(&delta);

                        if !delta.is_empty() {
                            on_delta(&delta);
                            out_text.push_str(&delta);
                            detok_start_idx = token_history.len();
                        }
                    }
                }

                // TODO: honor `formatted.stop_sequences` here as well if desired.
            }

            let out_text = out_text.trim().to_string();
            println!(
                "✅ [infer-stream] Complete. Output length: {} chars",
                out_text.len()
            );
            Ok(out_text)
        }))
        .map_err(|_| "💥 [infer-stream] PANIC occurred during inference!".to_string())?
    }
}

// ────────────────────────────────────────────────────────────────────────────────
// Small free helpers
// ────────────────────────────────────────────────────────────────────────────────

/// Length of the longest valid UTF-8 prefix in `bytes`.
fn utf8_valid_prefix_len(bytes: &[u8]) -> usize {
    match std::str::from_utf8(bytes) {
        Ok(_) => bytes.len(),
        Err(e) => e.valid_up_to(),
    }
}

/// Extract a system message from a sequence of turns, if any (first hit wins).
fn system_from_turns(turns: &[ChatTurn]) -> Option<&str> {
    turns.iter().find_map(|t| {
        if matches!(t.role, Role::System) {
            Some(t.content.as_str())
        } else {
            None
        }
    })
}

/// (Kept for future use) Find the earliest stop-string hit in `buf`.
#[allow(dead_code)]
fn first_stop_hit(buf: &str, stops: &[String]) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for s in stops {
        if let Some(i) = buf.find(s) {
            let len = s.len();
            best = Some(match best {
                None => (i, len),
                Some((cur_i, _)) if i < cur_i => (i, len),
                Some(prev) => prev,
            });
        }
    }
    best
}
