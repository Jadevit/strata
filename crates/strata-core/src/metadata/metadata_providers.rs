//! Central place to register all backend metadata providers.

use std::path::Path;

use crate::metadata::metadata_service::register_backend_metadata_provider;
use crate::traits::backend::PromptFlavor;
use crate::traits::metadata::{BackendMetadataProvider, ModelCoreInfo};

// ---- Llama provider (thin adapter over strata-backend-llama) ------------------

struct LlamaMetadataProvider;

impl BackendMetadataProvider for LlamaMetadataProvider {
    fn can_handle(&self, file: &Path) -> bool {
        // Only GGUF files handled by the llama backend
        llama_rs::metadata::metadata::can_handle(file)
    }

    fn collect(&self, file: &Path) -> Result<ModelCoreInfo, String> {
        use llama_rs::metadata::metadata::scrape_metadata;

        let s = scrape_metadata(file)?; // llama-only scrape (model-only, no context)

        // Derive hints BEFORE moving fields out of `s` to avoid borrow-of-moved errors.
        let fam_lc = s.family.as_deref().map(|f| f.to_ascii_lowercase());
        let prefer_native_template = s.chat_template.is_some();

        Ok(ModelCoreInfo {
            name: s.name,
            family: s.family,
            backend: s.backend,
            path: s.path,
            file_type: s.file_type,
            context_length: s.context_length,
            vocab_size: s.vocab_size,
            eos_token_id: s.eos_token_id,
            bos_token_id: s.bos_token_id,
            quantization: s.quantization,
            chat_template: s.chat_template,

            // If a native chat template exists, hint = None (engine should prefer native).
            // Otherwise pick a reasonable default by family.
            prompt_flavor_hint: if prefer_native_template {
                None
            } else {
                match fam_lc.as_deref() {
                    Some(f) if f.contains("phi") => Some(PromptFlavor::Phi3),
                    _ => Some(PromptFlavor::ChatMl),
                }
            },

            raw: s.raw,
        })
    }
}

/// Call once to register all providers we ship.
pub fn register_all_metadata_providers() {
    register_backend_metadata_provider(Box::new(LlamaMetadataProvider));
    // Add future providers here (transformers, onnx, …)
}
