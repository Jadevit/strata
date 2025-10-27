use super::LLMEngine;
use std::sync::atomic::Ordering;
use strata_abi::backend::LLMBackend;
use strata_abi::token::Token;

impl<B: LLMBackend> LLMEngine<B> {
    #[inline]
    fn lcp_len(&self, a: &[Token], b: &[Token]) -> usize {
        let n = a.len().min(b.len());
        for i in 0..n {
            if a[i] != b[i] {
                return i;
            }
        }
        n
    }

    /// Incremental prefill with KV reuse; returns (n_past, token_history, detok_start_idx).
    pub(super) fn prefill_incremental(
        &mut self,
        prompt_tokens: &[Token],
    ) -> Result<(i32, Vec<Token>, usize), String> {
        const PREFILL_CHUNK: usize = 64;

        // 1) Compare with previous prompt
        let lcp = self.lcp_len(&self.prev_prompt_tokens, prompt_tokens);
        let append_only = self.kv_warm && lcp == self.prev_prompt_tokens.len();

        if append_only {
            println!(
                "♻️  [prefill] Reusing KV (lcp={}, prev_len={}, new_len={})",
                lcp,
                self.prev_prompt_tokens.len(),
                prompt_tokens.len()
            );
        } else {
            println!(
                "🧹 [prefill] Prompt diverged or cold KV (lcp={}, prev_len={}, new_len={}) → clearing KV",
                lcp,
                self.prev_prompt_tokens.len(),
                prompt_tokens.len()
            );
            self.backend.clear_kv_cache();
        }

        let mut n_past: i32 = if append_only { lcp as i32 } else { 0 };
        let mut token_history: Vec<Token> =
            Vec::with_capacity(prompt_tokens.len().saturating_add(1024));

        // mirror the prompt up to start_idx (either lcp or 0)
        let start_idx = if append_only { lcp } else { 0 };
        if start_idx > 0 {
            token_history.extend_from_slice(&prompt_tokens[..start_idx]);
        }

        // 2) Evaluate only the delta
        for (i, chunk) in prompt_tokens[start_idx..].chunks(PREFILL_CHUNK).enumerate() {
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
                .map_err(|e| format!("❌ [infer] Prefill failed: {e}"))?;
            token_history.extend_from_slice(chunk);
            n_past += chunk.len() as i32;
        }
        println!("✅ [prefill] Done ({} tokens); kv_warm = true", n_past);

        // 3) KV now matches the new prompt
        self.prev_prompt_tokens = token_history.clone();
        self.kv_warm = true;

        let detok_start_idx = token_history.len(); // start detok after the prompt
        Ok((n_past, token_history, detok_start_idx))
    }
}
