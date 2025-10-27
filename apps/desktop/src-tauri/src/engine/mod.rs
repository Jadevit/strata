mod loader;
mod service;

use crate::app_state::AppState;
use std::sync::atomic::Ordering;
use strata_abi::backend::ChatTurn;
use tauri::{AppHandle, Emitter, State};

use service::ensure_engine_for_model;

/// Keep Tauri commands at module root for consistency with model/runtime.

#[tauri::command]
pub async fn load_system_prompt(app: AppHandle) -> Result<String, String> {
    loader::load_system_prompt_impl(app).await
}

// ---------------------------
// Non-streaming inference
// ---------------------------
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

// ---------------------------
// Streaming inference
// ---------------------------
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

// ---------------------------
// Cancel
// ---------------------------
#[tauri::command]
pub fn cancel_generation(state: State<'_, AppState>) -> Result<(), String> {
    if let Some(flag) = state.current_stop.lock().unwrap().as_ref() {
        flag.store(true, Ordering::Relaxed);
    }
    Ok(())
}
