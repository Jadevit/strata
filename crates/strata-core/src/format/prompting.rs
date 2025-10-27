//! Model-agnostic prompt formatting strategies (fallback path).

use crate::format::FormattedPrompt;
use strata_abi::backend::{ChatTurn, Role};

/// Format a user input or a full dialog into a complete prompt string for the backend.
pub trait PromptStrategy: Send + Sync {
    /// Back-compat single-turn helper.
    fn format(&self, user_input: &str) -> String {
        self.format_dialog(&[ChatTurn::user(user_input.to_string())], None)
            .text
    }

    /// Proper multi-turn formatting. If `system` is Some(..), it overrides any embedded system.
    fn format_dialog(&self, turns: &[ChatTurn], system: Option<&str>) -> FormattedPrompt;
}

/// UI helper: normalize visible BPE markers (Ġ space, Ċ newline).
#[allow(dead_code)]
pub fn normalize_bpe_markers(piece: &str) -> String {
    piece.replace('Ġ', " ").replace('Ċ', "\n")
}
