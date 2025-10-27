use super::LLMEngine;
use crate::format::format::FormattedPrompt;
use std::panic;
use strata_abi::backend::LLMBackend;

use super::utils::utf8_valid_prefix_len;

impl<B: LLMBackend> LLMEngine<B> {
    pub(super) fn infer_with_formatted(
        &mut self,
        formatted: FormattedPrompt,
    ) -> Result<String, String> {
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

            // Prefill (incremental, STOP-aware).
            let (mut n_past, mut token_history, mut detok_start_idx) =
                self.prefill_incremental(&prompt_tokens)?;

            // UTF-8 streaming state (accumulate valid prefix only).
            let mut out_text = String::new();
            let mut staging_bytes: Vec<u8> = Vec::with_capacity(4096);

            // Decode loop (STOP-aware).
            for step in 0..step_limit {
                if self.stop_flag.load(std::sync::atomic::Ordering::Relaxed) {
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
                        let delta = String::from_utf8(taken)
                            .map_err(|e| format!("detokenize produced non-UTF-8: {e}"))?;

                        // Append, then enforce stops (NOTE: not enforced yet).
                        out_text.push_str(&delta);
                        detok_start_idx = token_history.len();
                    }
                }
            }

            // Mirror generated tokens into prev_prompt_tokens so the next turn LCP sees them.
            self.prev_prompt_tokens = token_history[..detok_start_idx].to_vec();

            let out_text = out_text.trim().to_string();
            println!(
                "✅ [infer] Complete. Output length: {} chars",
                out_text.len()
            );
            Ok(out_text)
        }))
        .map_err(|_| "💥 [infer] PANIC occurred during inference!".to_string())?
    }

    pub(super) fn stream_with_formatted<F>(
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

            // Prefill (incremental, STOP-aware).
            let (mut n_past, mut token_history, mut detok_start_idx) =
                self.prefill_incremental(&prompt_tokens)?;

            // UTF-8 streaming state.
            let mut out_text = String::new();
            let mut staging_bytes: Vec<u8> = Vec::with_capacity(4096);

            // Decode loop (STOP-aware).
            for step in 0..step_limit {
                if self.stop_flag.load(std::sync::atomic::Ordering::Relaxed) {
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
                let new_bytes =
                    self.backend
                        .detokenize_range(&token_history, detok_start_idx, true, false)?;
                if !new_bytes.is_empty() {
                    staging_bytes.extend_from_slice(&new_bytes);
                    let valid_len = utf8_valid_prefix_len(&staging_bytes);
                    if valid_len > 0 {
                        let taken = staging_bytes.drain(..valid_len).collect::<Vec<u8>>();
                        let delta = String::from_utf8(taken)
                            .map_err(|e| format!("detokenize produced non-UTF-8: {e}"))?;

                        if !delta.is_empty() {
                            on_delta(&delta);
                            out_text.push_str(&delta);
                            detok_start_idx = token_history.len();
                        }
                    }
                }
            }

            // Mirror generated tokens into prev_prompt_tokens so the next turn LCP sees them.
            self.prev_prompt_tokens = token_history[..detok_start_idx].to_vec();

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
