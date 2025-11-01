// src-tauri/src/engine/service.rs

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use crate::app_state::AppState;
use crate::model::{get_model_path, set_current_model};
use crate::plugin::PluginBackend;

use strata_core::engine::LLMEngine;
use tauri::{AppHandle, Emitter};

use super::loader::load_system_prompt_sync;

/// Ensure an engine exists and matches the requested model id.
/// (kept as-is, used when you pass `model_id` alongside run calls)
pub(crate) fn ensure_engine_for_model(
    app: &AppHandle,
    state: &AppState,
    requested_model: Option<String>,
) -> Result<(), String> {
    if let Some(id) = requested_model.as_ref() {
        if crate::model::get_current_model().as_ref() != Some(id) {
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

/// Hard reinit to the *currently selected* model id.
/// - cancels any in-flight gen
/// - drops the old engine/context/KV
/// - clears session memory
/// - builds a fresh engine for the current model
/// - emits a model-switched event
pub(crate) fn reinit_engine_to_current_model(
    app: &tauri::AppHandle,
    state: &crate::app_state::AppState,
) -> Result<(), String> {
    // 1) stop any in-flight gen
    if let Some(flag) = state.current_stop.lock().unwrap().as_ref() {
        flag.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    // 2) check if we had an engine; drop it if so
    let had_engine = {
        let mut eng_slot = state.engine.lock().unwrap();
        let had = eng_slot.is_some();
        if let Some(engine) = eng_slot.as_mut() {
            eprintln!("🧹 [engine] Clearing KV before engine drop");
            engine.clear_kv_cache();
        }
        let old = eng_slot.take();
        drop(eng_slot);
        drop(old);
        had
    };

    // 3) reset session memory either way
    {
        let mut mem = state.memory.lock().unwrap();
        *mem = strata_core::memory::SessionMemory::new();
    }

    // 4) only build a fresh engine if we previously had one
    if had_engine {
        let model_path = crate::model::get_model_path(app)?;
        let backend = crate::plugin::PluginBackend::load(&model_path)?;
        let system = super::loader::load_system_prompt_sync(app);
        let engine = strata_core::engine::LLMEngine::with_auto(backend, system);

        let mut eng_slot = state.engine.lock().unwrap();
        *eng_slot = Some(engine);
    }

    // 5) notify UI either way
    if let Ok(model_path) = crate::model::get_model_path(app) {
        let _ = app.emit(
            "strata://model-switched",
            model_path.to_string_lossy().to_string(),
        );
    }

    Ok(())
}
