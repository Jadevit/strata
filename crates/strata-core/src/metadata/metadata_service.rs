//! Global registry + helpers for backend-agnostic metadata collection.

use std::path::Path;
use std::sync::{OnceLock, RwLock};

use crate::traits::metadata::{BackendMetadataProvider, ModelCoreInfo};

struct MetadataService {
    providers: Vec<Box<dyn BackendMetadataProvider>>,
}

impl MetadataService {
    fn new() -> Self {
        Self {
            providers: Vec::new(),
        }
    }
    fn register(&mut self, p: Box<dyn BackendMetadataProvider>) {
        self.providers.push(p);
    }
    fn collect_for(&self, file: &Path) -> Result<ModelCoreInfo, String> {
        for p in &self.providers {
            if p.can_handle(file) {
                return p.collect(file);
            }
        }
        Err(format!(
            "No metadata provider can handle {}",
            file.display()
        ))
    }
}

static REGISTRY: OnceLock<RwLock<MetadataService>> = OnceLock::new();

fn registry() -> &'static RwLock<MetadataService> {
    REGISTRY.get_or_init(|| RwLock::new(MetadataService::new()))
}

/// Register a backend metadata provider at startup.
pub fn register_backend_metadata_provider(p: Box<dyn BackendMetadataProvider>) {
    let mut r = registry().write().expect("metadata registry poisoned");
    r.register(p);
}

/// Collect metadata for the given model file using the first provider that can handle it.
pub fn collect_model_metadata(path: &Path) -> Result<ModelCoreInfo, String> {
    let r = registry().read().expect("metadata registry poisoned");
    r.collect_for(path)
}

// ─────────────────────────────────────────────────────────────────────────────
// UI DTO (co-located for now) + mapper
// ─────────────────────────────────────────────────────────────────────────────

use crate::traits::backend::PromptFlavor;
use serde::Serialize;

/// Matches the UI’s `ModelMeta` (snake_case -> JSON).
#[derive(Debug, Clone, Serialize)]
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

    /// Optional passthrough for advanced/debug views
    pub raw: Option<std::collections::HashMap<String, String>>,
}

/// Borrow-only mapping to avoid moving from `ModelCoreInfo`.
pub fn to_ui_meta(s: &ModelCoreInfo) -> ModelMetaOut {
    let prompt_hint_str = s.prompt_flavor_hint.as_ref().map(|pf| {
        match pf {
            PromptFlavor::ChatMl => "ChatMl",
            PromptFlavor::InstBlock => "InstBlock",
            PromptFlavor::UserAssistant => "UserAssistant",
            PromptFlavor::Plain => "Plain",
            PromptFlavor::Phi3 => "Phi3",
        }
        .to_string()
    });

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

        prompt_flavor_hint: prompt_hint_str,
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
