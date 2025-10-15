use std::sync::Arc;
use std::{fs, path::PathBuf, sync::atomic::Ordering};

use strata_abi::backend::ChatTurn;
use strata_core::engine::engine::LLMEngine;
use strata_core::format::prompt_format::PromptKind;
use tauri::{AppHandle, Emitter, Manager, State, path::BaseDirectory};

use crate::app_state::AppState;
use crate::model::{get_current_model, get_model_path, set_current_model};
use crate::plugin::PluginBackend;

// ---------------------------------------------------------------------------
// Prompt strategy
// ---------------------------------------------------------------------------

fn pick_prompt_strategy(model_id: Option<String>, system: Option<String>) -> PromptKind {
    if let Some(id) = model_id {
        let id_lc = id.to_lowercase();
        if id_lc.contains("phi") || id_lc.contains("vira") {
            return PromptKind::Phi3 { system };
        }
    }
    PromptKind::ChatMl { system }
}

// ---------------------------------------------------------------------------
// System prompt loaders
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn load_system_prompt(app: AppHandle) -> Result<String, String> {
    if let Some(path) = app
        .path()
        .resolve("system_prompt.txt", BaseDirectory::Resource)
        .ok()
    {
        if path.exists() {
            return fs::read_to_string(&path)
                .map_err(|e| format!("Failed to read system_prompt at {}: {e}", path.display()));
        }
    }
    let dev = PathBuf::from("resources/system_prompt.txt");
    if dev.exists() {
        return fs::read_to_string(&dev)
            .map_err(|e| format!("Dev fallback read failed at {}: {e}", dev.display()));
    }
    Err("system_prompt.txt not found in resources or ./resources/".into())
}

fn load_system_prompt_sync(app: &AppHandle) -> Option<String> {
    if let Some(path) = app
        .path()
        .resolve("system_prompt.txt", BaseDirectory::Resource)
        .ok()
    {
        if path.exists() {
            if let Ok(text) = fs::read_to_string(&path) {
                return Some(text);
            }
        }
    }
    let dev = PathBuf::from("resources/system_prompt.txt");
    if dev.exists() {
        if let Ok(text) = fs::read_to_string(&dev) {
            return Some(text);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Engine persistence (reuse same llama session across prompts)
// ---------------------------------------------------------------------------

fn ensure_engine_for_model(
    app: &AppHandle,
    state: &AppState,
    requested_model: Option<String>,
) -> Result<(), String> {
    // If model ID changed, clear existing engine → explicitly clear KV → drop backend.
    if let Some(id) = requested_model.as_ref() {
        if get_current_model().as_ref() != Some(id) {
            // If there's an existing engine, clear its KV cache before dropping.
            if let Some(engine) = state.engine.lock().unwrap().as_mut() {
                eprintln!("🧹 [engine] Model switch detected, clearing KV cache before drop");
                engine.clear_kv_cache();
            }

            // Drop the engine (will also drop backend session).
            *state.engine.lock().unwrap() = None;
            set_current_model(id.clone());
        }
    }

    // Initialize a new engine if not already loaded.
    let mut slot = state.engine.lock().unwrap();
    if slot.is_none() {
        let model_path = get_model_path(app)?;
        let backend = PluginBackend::load(&model_path)?;
        let system = load_system_prompt_sync(app);

        let mut engine = LLMEngine::with_auto(backend, system.clone());
        engine.set_strategy(pick_prompt_strategy(
            requested_model.or_else(get_current_model),
            system,
        ));

        *slot = Some(engine);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Non-streaming inference
// ---------------------------------------------------------------------------

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

    let app2 = app.clone();
    let state2 = AppState {
        memory: Arc::clone(&state.memory),
        current_stop: Arc::clone(&state.current_stop),
        engine: Arc::clone(&state.engine),
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

// ---------------------------------------------------------------------------
// Streaming inference
// ---------------------------------------------------------------------------

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
        memory: Arc::clone(&state.memory),
        current_stop: Arc::clone(&state.current_stop),
        engine: Arc::clone(&state.engine),
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

// ---------------------------------------------------------------------------
// Cancel
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn cancel_generation(state: State<'_, AppState>) -> Result<(), String> {
    if let Some(flag) = state.current_stop.lock().unwrap().as_ref() {
        flag.store(true, Ordering::Relaxed);
    }
    Ok(())
}
