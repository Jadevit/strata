use std::path::Path;
use strata_abi::backend::PromptFlavor;
use strata_abi::metadata::{BackendMetadataProvider, ModelCoreInfo};

pub struct LlamaMetadataProvider;

impl BackendMetadataProvider for LlamaMetadataProvider {
    fn can_handle(&self, file: &Path) -> bool {
        llama_rs::metadata::metadata::can_handle(file)
    }

    fn collect(&self, file: &Path) -> Result<ModelCoreInfo, String> {
        use llama_rs::metadata::metadata::scrape_metadata;

        let s = scrape_metadata(file)?;
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
