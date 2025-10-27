//! Shared prompt carrier used between any formatter and inference.
//!
//! Plugins are expected to apply native chat templates and
//! enforce stop sequences. Strata itself does not enforce stops; this struct exists
//! so we can pass a finished prompt when a backend doesn’t provide templating.

#[derive(Debug, Clone)]
pub struct FormattedPrompt {
    pub text: String,
    /// Optional textual stop sentinels for backends that want them.
    /// (Strata does not enforce these; backends may.)
    pub stop_sequences: Vec<String>,
    /// Some tokenizers prefer a leading space to avoid odd tokenization;
    /// backends can ignore this if they handle space-prefix internally.
    pub add_space_prefix: bool,
}

impl FormattedPrompt {
    pub fn new<T: Into<String>>(text: T) -> Self {
        Self {
            text: text.into(),
            stop_sequences: Vec::new(),
            add_space_prefix: true,
        }
    }
}
