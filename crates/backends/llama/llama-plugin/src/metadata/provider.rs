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
                    Some(f) if f.contains("phi") => Some("phi3".to_string()),
                    _ => Some("chatml".to_string()),
                }
            },
            raw: s.raw,
        })
    }
}
