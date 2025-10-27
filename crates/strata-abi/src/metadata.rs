//! Backend-agnostic model metadata traits & types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::backend::PromptFlavor;

/// Minimal, normalized view Strata expects from any backend.
/// Backends can park extra info under `raw`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCoreInfo {
    /// Display name if known (e.g., "Qwen3 1.7B Instruct").
    pub name: Option<String>,
    /// Family / architecture hint (e.g., "llama", "mistral", "qwen3", "phi3", "onnx").
    pub family: Option<String>,
    /// Which backend produced this (e.g., "llama", "transformers", "onnx").
    pub backend: String,

    /// Absolute path to the model artifact used by this backend (gguf, safetensors, onnx, …).
    pub path: PathBuf,
    /// Lowercase file extension (e.g., "gguf", "safetensors", "onnx").
    pub file_type: String,

    /// Context length (aka n_ctx / max_position_embeddings) if known.
    pub context_length: Option<u32>,
    /// Vocab size if known.
    pub vocab_size: Option<u32>,

    /// Token IDs if known.
    pub eos_token_id: Option<i32>,
    pub bos_token_id: Option<i32>,

    /// Quantization label if known (e.g., "Q8_0", "Q5_K_M", "fp16").
    pub quantization: Option<String>,

    /// Native chat template string if provided by the model.
    pub chat_template: Option<String>,
    /// Hint for a reasonable default prompt wrapper when no native template is used.
    pub prompt_flavor_hint: Option<String>,

    /// Anything else the backend scraped (simple flattened map).
    pub raw: HashMap<String, String>,
}

/// A backend-specific metadata adapter registers one of these with the service.
/// It decides whether it can parse the given file and, if so, returns metadata.
pub trait BackendMetadataProvider: Send + Sync + 'static {
    /// Return true if this provider understands the file (by extension, header, etc.).
    fn can_handle(&self, file: &Path) -> bool;

    /// Scrape and normalize metadata for this file.
    fn collect(&self, file: &Path) -> Result<ModelCoreInfo, String>;
}
