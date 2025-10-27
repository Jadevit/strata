// strata/crates/backends/llama/strata-backend-llama/src/cache.rs
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::backends::dispatch::Backend as LlamaCppBackend;
use crate::model::LlamaModel;
use crate::params::LlamaParams;

/// Global in-process cache of loaded llama models (no context).
static CACHE: Lazy<Mutex<HashMap<PathBuf, Arc<LlamaModel>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn canon<P: AsRef<Path>>(p: P) -> PathBuf {
    std::fs::canonicalize(p.as_ref()).unwrap_or_else(|_| p.as_ref().to_path_buf())
}

/// Load the model (no context) and keep it resident. Idempotent.
pub fn preload_model<P: AsRef<Path>>(model_path: P) -> Result<(), String> {
    let key = canon(&model_path);
    {
        // Fast path: already cached
        if CACHE.lock().unwrap().contains_key(&key) {
            return Ok(());
        }
    }

    // Load via llama backend (loads the model only)
    let params = LlamaParams::default(); // nothing special needed to map the file
    let backend = LlamaCppBackend::load(&key, params.clone())
        .map_err(|e| format!("llama preload failed: {e}"))?;
    let model = backend.model();

    // Insert into cache (keep Arc to keep resident)
    CACHE.lock().unwrap().insert(key, model);
    Ok(())
}

/// Get a cloned Arc to a cached model, if present.
pub fn get_cached_model<P: AsRef<Path>>(model_path: P) -> Option<Arc<LlamaModel>> {
    let key = canon(model_path);
    CACHE.lock().unwrap().get(&key).cloned()
}
