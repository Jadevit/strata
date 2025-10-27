use std::path::Path;
use std::sync::{OnceLock, RwLock};

use super::MetadataService;
use strata_abi::metadata::{BackendMetadataProvider, ModelCoreInfo};

static REGISTRY: OnceLock<RwLock<MetadataService>> = OnceLock::new();

#[inline]
fn registry() -> &'static RwLock<MetadataService> {
    REGISTRY.get_or_init(|| RwLock::new(MetadataService::new()))
}

/// Register a backend metadata provider at startup (static use).
pub fn register_backend_metadata_provider(p: Box<dyn BackendMetadataProvider>) {
    let mut r = registry().write().expect("metadata registry poisoned");
    r.register(p);
}

/// Collect metadata for the given model file using the first provider that can handle it.
pub fn collect_model_metadata(path: &Path) -> Result<ModelCoreInfo, String> {
    let r = registry().read().expect("metadata registry poisoned");
    r.collect_for(path)
}

/// Load dynamic plugins from a directory.
/// Call this once at app startup (after registry init).
pub fn load_metadata_plugins(dir: &Path) -> Result<(), String> {
    let mut r = registry().write().expect("metadata registry poisoned");
    super::dynamic::load_dir_into(&mut *r, dir)
}
