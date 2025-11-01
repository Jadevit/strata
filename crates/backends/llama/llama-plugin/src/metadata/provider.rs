use std::path::Path;
use strata_abi::metadata::{BackendMetadataProvider, ModelCoreInfo};

use super::{can_handle, scrape_metadata};

pub struct LlamaMetadataProvider;

impl BackendMetadataProvider for LlamaMetadataProvider {
    fn can_handle(&self, file: &Path) -> bool {
        can_handle(file)
    }

    fn collect(&self, file: &Path) -> Result<ModelCoreInfo, String> {
        let s = scrape_metadata(file)?;

        // HARD REQUIREMENT: model must provide a native chat_template.
        if s.chat_template
            .as_deref()
            .map(str::is_empty)
            .unwrap_or(true)
        {
            return Err(format!(
                "model '{}' is missing a native chat template, please refer to the model card!",
                file.display()
            ));
        }

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
            chat_template: s.chat_template, // present & non-empty by here
            prompt_flavor_hint: None,       // absolutely no fallback
            raw: s.raw,
        })
    }
}
