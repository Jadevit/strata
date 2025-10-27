use std::sync::Arc;

use crate::app_state::AppState;
use crate::model::{get_current_model, get_model_path, set_current_model};
use crate::plugin::PluginBackend;

use strata_core::engine::LLMEngine;
use tauri::AppHandle;

use super::loader::load_system_prompt_sync;

/// Ensure an engine exists and matches the requested model id.
/// Creates a new engine if needed (and clears KV on model switch).
pub(crate) fn ensure_engine_for_model(
    app: &AppHandle,
    state: &AppState,
    requested_model: Option<String>,
) -> Result<(), String> {
    // If model ID changed, clear existing engine → explicitly clear KV → drop.
    if let Some(id) = requested_model.as_ref() {
        if get_current_model().as_ref() != Some(id) {
            if let Some(engine) = state.engine.lock().unwrap().as_mut() {
                eprintln!("🧹 [engine] Model switch detected, clearing KV cache before drop");
                engine.clear_kv_cache();
            }
            *state.engine.lock().unwrap() = None;
            set_current_model(id.clone());
        }
    }

    // Initialize engine if missing
    let mut slot = state.engine.lock().unwrap();
    if slot.is_none() {
        let model_path = get_model_path(app)?;
        let backend = PluginBackend::load(&model_path)?;
        let system = load_system_prompt_sync(app);
        let engine = LLMEngine::with_auto(backend, system);
        *slot = Some(engine);
    }
    Ok(())
}
