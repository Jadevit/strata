use crate::format::prompting::{
    ChatMlFormat, CustomFormat, InstBlockFormat, Phi3Format, PlainFormat, PromptStrategy,
    UserAssistantFormat,
};

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
