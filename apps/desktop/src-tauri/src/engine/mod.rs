// src-tauri/src/engine/mod.rs
//
// Engine-facing Tauri commands and small shims.
// - We keep service.rs private and expose only the safe entry points here.
// - New: `preload_engine` builds the engine/context once for the currently-selected model.
// - Reinit shim stays crate-visible so other modules can request clean swaps without touching service.rs.

mod loader;
mod service;

use crate::app_state::AppState;
use std::sync::atomic::Ordering;
use strata_abi::backend::ChatTurn;
use tauri::{AppHandle, Emitter, State};

use service::ensure_engine_for_model;

// ---------------------------
// Public Tauri commands
// ---------------------------

#[tauri::command]
pub async fn load_system_prompt(app: AppHandle) -> Result<String, String> {
    loader::load_system_prompt_impl(app).await
}

// Non-streaming inference
#[tauri::command]
pub async fn run_llm(
    prompt: String,
    _tts: bool,
    model_id: Option<String>,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<String, String> {
    {
        let mut mem = state.memory.lock().unwrap();
        mem.push_user(prompt.clone());
    }

    // Spawn blocking for CPU-bound work
    let app2 = app.clone();
    let state2 = AppState {
        memory: std::sync::Arc::clone(&state.memory),
        current_stop: std::sync::Arc::clone(&state.current_stop),
        engine: std::sync::Arc::clone(&state.engine),
    };
    let model_id2 = model_id.clone();

    let reply = tauri::async_runtime::spawn_blocking(move || -> Result<String, String> {
        ensure_engine_for_model(&app2, &state2, model_id2)?;

        let mut guard = state2.engine.lock().unwrap();
        let engine = guard.as_mut().expect("engine initialized");

        {
            let stop = engine.stop_handle();
            *state2.current_stop.lock().unwrap() = Some(stop);
        }

        let turns: Vec<ChatTurn> = {
            let mem = state2.memory.lock().unwrap();
            mem.turns().to_vec()
        };

        let out = engine.infer_chat(&turns)?;
        *state2.current_stop.lock().unwrap() = None;
        Ok(out)
    })
    .await
    .map_err(|e| format!("join error: {e}"))??;

    {
        let mut mem = state.memory.lock().unwrap();
        mem.push_assistant(reply.clone());
    }

    Ok(reply)
}

// Streaming inference
#[tauri::command]
pub async fn run_llm_stream(
    prompt: String,
    _tts: bool,
    model_id: Option<String>,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<(), String> {
    {
        let mut mem = state.memory.lock().unwrap();
        mem.push_user(prompt.clone());
    }

    let app2 = app.clone();
    let state2 = AppState {
        memory: std::sync::Arc::clone(&state.memory),
        current_stop: std::sync::Arc::clone(&state.current_stop),
        engine: std::sync::Arc::clone(&state.engine),
    };
    let model_id2 = model_id.clone();

    tauri::async_runtime::spawn_blocking(move || -> Result<String, String> {
        ensure_engine_for_model(&app2, &state2, model_id2)?;

        let mut guard = state2.engine.lock().unwrap();
        let engine = guard.as_mut().expect("engine initialized");

        {
            let stop = engine.stop_handle();
            *state2.current_stop.lock().unwrap() = Some(stop);
        }

        let turns: Vec<ChatTurn> = {
            let mem = state2.memory.lock().unwrap();
            mem.turns().to_vec()
        };

        let final_text = engine.infer_chat_stream(&turns, |delta| {
            let _ = app2.emit("llm-stream", serde_json::json!({ "delta": delta }));
        })?;

        *state2.current_stop.lock().unwrap() = None;

        {
            let mut mem = state2.memory.lock().unwrap();
            mem.push_assistant(final_text.clone());
        }

        let _ = app2.emit("llm-complete", serde_json::json!({ "text": final_text }));
        Ok(final_text)
    })
    .await
    .map_err(|e| format!("join error: {e}"))??;

    Ok(())
}

// Cancel
#[tauri::command]
pub fn cancel_generation(state: State<'_, AppState>) -> Result<(), String> {
    if let Some(flag) = state.current_stop.lock().unwrap().as_ref() {
        flag.store(true, Ordering::Relaxed);
    }
    Ok(())
}

// ---------------------------
// New: Engine preload
// ---------------------------

/// Build the engine/context once for the *currently selected* model if missing.
/// - No-ops if an engine already exists.
/// - Does not change the selected model.
/// - Runs on a blocking worker so the UI stays snappy.
#[tauri::command]
pub async fn preload_engine(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    let app2 = app.clone();
    let state2 = AppState {
        memory: std::sync::Arc::clone(&state.memory),
        current_stop: std::sync::Arc::clone(&state.current_stop),
        engine: std::sync::Arc::clone(&state.engine),
    };

    tauri::async_runtime::spawn_blocking(move || ensure_engine_for_model(&app2, &state2, None))
        .await
        .map_err(|e| format!("join error: {e}"))??;

    // Optional: signal ready (harmless if nobody listens)
    let _ = app.emit(
        "strata://engine-preloaded",
        crate::model::get_current_model(),
    );

    Ok(())
}

// ---------------------------
// Crate-visible shims
// ---------------------------

/// Re-export a crate-visible shim so other modules (e.g., model) can trigger a clean swap.
/// Keeps service.rs private.
pub(crate) fn reinit_engine_to_current_model(
    app: &tauri::AppHandle,
    state: &crate::app_state::AppState,
) -> Result<(), String> {
    service::reinit_engine_to_current_model(app, state)
}
