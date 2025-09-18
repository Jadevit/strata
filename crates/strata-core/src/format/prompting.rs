//! Model-agnostic prompt formatting strategies.

use crate::traits::backend::{ChatTurn, Role};

/// Final prompt handed to the backend.
#[derive(Debug, Clone)]
pub struct FormattedPrompt {
    pub text: String,
    /// End-of-turn/end-of-response sentinels to watch for in decoded output.
    pub stop_sequences: Vec<String>,
    /// Some llama-based tokenizers prefer a leading space to avoid odd tokenization.
    /// Backends can ignore this if they handle space-prefix internally.
    pub add_space_prefix: bool,
}

/// Format a user input or a full dialog into a complete prompt string for the backend.
/// `format()` remains as a convenience, delegating to `format_dialog()`.
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

/// ChatML-style (<|im_start|>role ... <|im_end|>) with open assistant turn.
pub struct ChatMlFormat {
    system: Option<String>,
}
impl ChatMlFormat {
    pub fn new<S: Into<String>>(system: Option<S>) -> Self {
        Self {
            system: system.map(|s| s.into()),
        }
    }
}
impl PromptStrategy for ChatMlFormat {
    fn format_dialog(&self, turns: &[ChatTurn], system: Option<&str>) -> FormattedPrompt {
        let mut out = String::new();

        if let Some(sys) = system.or(self.system.as_deref()) {
            out.push_str("<|im_start|>system\n");
            out.push_str(sys.trim());
            out.push_str("<|im_end|>\n");
        }

        for t in turns {
            match t.role {
                Role::User => {
                    out.push_str("<|im_start|>user\n");
                    out.push_str(t.content.trim());
                    out.push_str("<|im_end|>\n");
                }
                Role::Assistant => {
                    out.push_str("<|im_start|>assistant\n");
                    out.push_str(t.content.trim());
                    out.push_str("<|im_end|>\n");
                }
                Role::System => {
                    out.push_str("<|im_start|>system\n");
                    out.push_str(t.content.trim());
                    out.push_str("<|im_end|>\n");
                }
            }
        }

        // Open assistant turn for generation.
        out.push_str("<|im_start|>assistant\n");

        FormattedPrompt {
            text: out,
            // include these so we stop if the model tries to start a new turn
            stop_sequences: vec![
                "<|im_end|>".to_string(),
                "<|im_start|>user".to_string(),
                "<|im_start|>system".to_string(),
            ],
            add_space_prefix: true,
        }
    }
}

/// Simple "User/Assistant" with open Assistant turn.
pub struct UserAssistantFormat;
impl PromptStrategy for UserAssistantFormat {
    fn format_dialog(&self, turns: &[ChatTurn], system: Option<&str>) -> FormattedPrompt {
        let mut out = String::new();
        if let Some(sys) = system {
            out.push_str("System: ");
            out.push_str(sys.trim());
            out.push('\n');
        }
        for t in turns {
            match t.role {
                Role::User => {
                    out.push_str("User: ");
                    out.push_str(t.content.trim());
                    out.push('\n');
                }
                Role::Assistant => {
                    out.push_str("Assistant: ");
                    out.push_str(t.content.trim());
                    out.push('\n');
                }
                Role::System => {
                    out.push_str("System: ");
                    out.push_str(t.content.trim());
                    out.push('\n');
                }
            }
        }
        out.push_str("Assistant: ");

        FormattedPrompt {
            text: out,
            stop_sequences: vec!["\nUser:".to_string(), "\nSystem:".to_string()],
            add_space_prefix: true,
        }
    }
}

/// INST-block style (model-agnostic). Collapses dialog into one instruction block, then opens assistant.
pub struct InstBlockFormat;
impl PromptStrategy for InstBlockFormat {
    fn format_dialog(&self, turns: &[ChatTurn], system: Option<&str>) -> FormattedPrompt {
        let mut instruction = String::new();
        if let Some(sys) = system {
            instruction.push_str(sys.trim());
            instruction.push_str("\n\n");
        }
        for t in turns {
            match t.role {
                Role::User => {
                    instruction.push_str("User: ");
                    instruction.push_str(t.content.trim());
                    instruction.push('\n');
                }
                Role::Assistant => {
                    instruction.push_str("Assistant: ");
                    instruction.push_str(t.content.trim());
                    instruction.push('\n');
                }
                Role::System => {
                    instruction.push_str("System: ");
                    instruction.push_str(t.content.trim());
                    instruction.push('\n');
                }
            }
        }
        let mut text = String::new();
        text.push_str("<s>[INST] ");
        text.push_str(instruction.trim_end());
        text.push_str(" [/INST] ");

        FormattedPrompt {
            text,
            stop_sequences: vec!["</s>".to_string()],
            add_space_prefix: true,
        }
    }
}

/// Pass-through (no wrapping).
pub struct PlainFormat;
impl PromptStrategy for PlainFormat {
    fn format_dialog(&self, turns: &[ChatTurn], system: Option<&str>) -> FormattedPrompt {
        let mut out = String::new();
        if let Some(sys) = system {
            out.push_str(sys.trim());
            out.push('\n');
        }
        for t in turns {
            out.push_str(t.content.as_str());
            out.push('\n');
        }
        FormattedPrompt {
            text: out,
            stop_sequences: vec![],
            add_space_prefix: true,
        }
    }
}

/// Microsoft Phi-3 style:
/// <|system|>\n{sys}\n<|end|>\n<|user|>\n{dialog}\n<|end|>\n<|assistant|>\n
pub struct Phi3Format {
    system: Option<String>,
}
impl Phi3Format {
    pub fn new<S: Into<String>>(system: Option<S>) -> Self {
        Self {
            system: system.map(|s| s.into()),
        }
    }
}
impl PromptStrategy for Phi3Format {
    fn format_dialog(&self, turns: &[ChatTurn], system: Option<&str>) -> FormattedPrompt {
        let mut out = String::new();

        // System block (optional)
        if let Some(sys) = system.or(self.system.as_deref()) {
            out.push_str("<|system|>\n");
            out.push_str(sys.trim());
            out.push_str("\n<|end|>\n");
        }

        // Emit dialog as Phi-3 blocks (no inline "User:" / "Assistant:" labels).
        for t in turns {
            match t.role {
                Role::User => {
                    out.push_str("<|user|>\n");
                    out.push_str(t.content.trim());
                    out.push_str("\n<|end|>\n");
                }
                Role::Assistant => {
                    out.push_str("<|assistant|>\n");
                    out.push_str(t.content.trim());
                    out.push_str("\n<|end|>\n");
                }
                Role::System => {
                    // If stray system turns appear in-dialog, treat as a user-visible block.
                    out.push_str("<|user|>\n");
                    out.push_str(t.content.trim());
                    out.push_str("\n<|end|>\n");
                }
            }
        }

        // Open next assistant turn for generation.
        out.push_str("<|assistant|>\n");

        FormattedPrompt {
            text: out,
            // Stop if the model tries to close or start a new block
            stop_sequences: vec![
                "<|end|>".to_string(),
                "<|user|>".to_string(),
                "<|system|>".to_string(),
                "<|assistant|>\n".to_string(), // guard against double-open echoes
            ],
            add_space_prefix: true,
        }
    }
}

/// Custom wrapper with a single "{}" placeholder (uses last user message).
pub struct CustomFormat {
    pattern: String,
}
impl CustomFormat {
    pub fn new(pattern: String) -> Self {
        Self { pattern }
    }
}
impl PromptStrategy for CustomFormat {
    fn format_dialog(&self, turns: &[ChatTurn], system: Option<&str>) -> FormattedPrompt {
        let last_user = turns
            .iter()
            .rev()
            .find(|t| matches!(t.role, Role::User))
            .map(|t| t.content.as_str())
            .unwrap_or_default();

        let mut text = String::new();
        if let Some(sys) = system {
            text.push_str(sys.trim());
            text.push('\n');
        }
        text.push_str(&self.pattern.replace("{}", last_user));

        FormattedPrompt {
            text,
            stop_sequences: vec![],
            add_space_prefix: true,
        }
    }
}
