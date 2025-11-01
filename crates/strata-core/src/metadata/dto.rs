use serde::{Deserialize, Serialize};
use strata_abi::metadata::ModelCoreInfo;

/// Matches the UI’s `ModelMeta` (snake_case -> JSON).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetaOut {
    pub name: Option<String>,
    pub family: Option<String>,
    pub backend: String,
    pub file_type: String,

    pub quantization: Option<String>,
    pub context_length: Option<u32>,
    pub vocab_size: Option<u32>,
    pub eos_token_id: Option<i32>,
    pub bos_token_id: Option<i32>,

    /// "ChatMl" | "InstBlock" | "UserAssistant" | "Plain" | "Phi3"
    pub prompt_flavor_hint: Option<String>,
    pub has_chat_template: bool,

    /// Optional passthrough for advanced/debug views.
    pub raw: Option<std::collections::HashMap<String, String>>,
}

/// Borrow-only mapping to avoid moving from `ModelCoreInfo`.
pub fn to_ui_meta(s: &ModelCoreInfo) -> ModelMetaOut {
    ModelMetaOut {
        name: s.name.clone(),
        family: s.family.clone(),
        backend: s.backend.clone(),
        file_type: s.file_type.clone(),

        quantization: s.quantization.clone(),
        context_length: s.context_length,
        vocab_size: s.vocab_size,
        eos_token_id: s.eos_token_id,
        bos_token_id: s.bos_token_id,

        prompt_flavor_hint: s.prompt_flavor_hint.clone(),
        has_chat_template: s
            .chat_template
            .as_ref()
            .map(|t| !t.is_empty())
            .unwrap_or(false),

        raw: if s.raw.is_empty() {
            None
        } else {
            Some(s.raw.clone())
        },
    }
}
