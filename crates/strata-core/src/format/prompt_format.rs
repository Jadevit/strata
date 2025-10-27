use super::prompting::{PromptStrategy, normalize_bpe_markers}; // if you ever use normalize
use crate::format::FormattedPrompt;
use strata_abi::backend::{ChatTurn, Role};

/// Generic, model-agnostic prompt kinds.
pub enum PromptKind {
    ChatMl {
        system: Option<String>,
    },
    UserAssistant,
    InstBlock,
    Plain,
    Phi3 {
        system: Option<String>,
    },
    /// A simple pattern with a single `{}` placeholder for the last user message.
    Custom {
        pattern: String,
    },
}

/// Factory: select a prompt strategy from a `PromptKind`.
pub fn select_prompt(kind: PromptKind) -> Box<dyn PromptStrategy> {
    match kind {
        PromptKind::ChatMl { system } => Box::new(ChatMlFormat::new(system)),
        PromptKind::UserAssistant => Box::new(UserAssistantFormat),
        PromptKind::InstBlock => Box::new(InstBlockFormat),
        PromptKind::Plain => Box::new(PlainFormat),
        PromptKind::Phi3 { system } => Box::new(Phi3Format::new(system)),
        PromptKind::Custom { pattern } => Box::new(CustomFormat::new(pattern)),
    }
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
            // NOTE: Strata does not enforce these; backends may.
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
            stop_sequences: vec!["\nUser:".into(), "\nSystem:".into()],
            add_space_prefix: true,
        }
    }
}

/// INST-block style.
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
            stop_sequences: vec!["</s>".into()],
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

/// Microsoft Phi-3 style blocks.
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

        if let Some(sys) = system.or(self.system.as_deref()) {
            out.push_str("<|system|>\n");
            out.push_str(sys.trim());
            out.push_str("\n<|end|>\n");
        }

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
                    out.push_str("<|user|>\n");
                    out.push_str(t.content.trim());
                    out.push_str("\n<|end|>\n");
                }
            }
        }

        out.push_str("<|assistant|>\n");

        FormattedPrompt {
            text: out,
            stop_sequences: vec![
                "<|end|>".into(),
                "<|user|>".into(),
                "<|system|>".into(),
                "<|assistant|>\n".into(),
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
