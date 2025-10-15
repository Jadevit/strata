use std::sync::Arc;
use strata_abi::backend::{ChatTurn, PromptFlavor};
use strata_core::engine::engine::LLMEngine;
use strata_core::format::prompt_format::PromptKind;

use std::{fs, path::PathBuf, sync::atomic::Ordering};
use tauri::{AppHandle, Emitter, Manager, State, path::BaseDirectory};

use crate::app_state::AppState;
use crate::model::{get_current_model, get_model_path, set_current_model};
use crate::plugin::PluginBackend;

// --- prompt strategy selection ---
fn pick_prompt_strategy(model_id: Option<String>, system: Option<String>) -> PromptKind {
    if let Some(id) = model_id {
        let id_lc = id.to_lowercase();
        if id_lc.contains("phi") || id_lc.contains("vira") {
            return PromptKind::Phi3 { system };
        }
    }
    PromptKind::ChatMl { system }
}

// --- system prompt helpers (kept local to engine for simplicity) ---
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

// --- non-streaming inference ---
#[tauri::command]
pub async fn run_llm(
    prompt: String,
    _tts: bool,
    model_id: Option<String>,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<String, String> {
    if let Some(ref id) = model_id {
        set_current_model(id.clone());
    }
    {
        let mut mem = state.memory.lock().unwrap();
        mem.push_user(prompt.clone());
    }

    let app2 = app.clone();
    let state_mem = Arc::clone(&state.memory);
    let state_stop = Arc::clone(&state.current_stop);
    let model_id2 = model_id.clone();

    let reply = tauri::async_runtime::spawn_blocking(move || -> Result<String, String> {
        let model_path = get_model_path(&app2)?;
        let backend = PluginBackend::load(&model_path)?;
        let system = load_system_prompt_sync(&app2);

        let mut engine = LLMEngine::with_auto(backend, system.clone());
        engine.set_strategy(pick_prompt_strategy(
            model_id2.or_else(get_current_model),
            system,
        ));

        {
            let stop = engine.stop_handle();
            *state_stop.lock().unwrap() = Some(stop);
        }

        let turns: Vec<ChatTurn> = {
            let mem = state_mem.lock().unwrap();
            mem.turns().to_vec()
        };

        let out = engine.infer_chat(&turns)?;
        *state_stop.lock().unwrap() = None;
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

// --- streaming inference ---
#[tauri::command]
pub async fn run_llm_stream(
    prompt: String,
    _tts: bool,
    model_id: Option<String>,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<(), String> {
    if let Some(ref id) = model_id {
        set_current_model(id.clone());
    }
    {
        let mut mem = state.memory.lock().unwrap();
        mem.push_user(prompt.clone());
    }

    let app2 = app.clone();
    let state_mem = Arc::clone(&state.memory);
    let state_stop = Arc::clone(&state.current_stop);
    let model_id2 = model_id.clone();

    tauri::async_runtime::spawn_blocking(move || -> Result<String, String> {
        let model_path = get_model_path(&app2)?;
        let backend = PluginBackend::load(&model_path)?;
        let system = load_system_prompt_sync(&app2);

        let mut engine = LLMEngine::with_auto(backend, system.clone());
        engine.set_strategy(pick_prompt_strategy(
            model_id2.or_else(get_current_model),
            system,
        ));

        {
            let stop = engine.stop_handle();
            *state_stop.lock().unwrap() = Some(stop);
        }

        let turns: Vec<ChatTurn> = {
            let mem = state_mem.lock().unwrap();
            mem.turns().to_vec()
        };

        let final_text = engine.infer_chat_stream(&turns, |delta| {
            let _ = app2.emit("llm-stream", serde_json::json!({ "delta": delta }));
        })?;

        *state_stop.lock().unwrap() = None;
        {
            let mut mem = state_mem.lock().unwrap();
            mem.push_assistant(final_text.clone());
        }
        let _ = app2.emit("llm-complete", serde_json::json!({ "text": final_text }));
        Ok(final_text)
    })
    .await
    .map_err(|e| format!("join error: {e}"))??;

    Ok(())
}

// --- cancel ---
#[tauri::command]
pub fn cancel_generation(state: State<'_, AppState>) -> Result<(), String> {
    if let Some(flag) = state.current_stop.lock().unwrap().as_ref() {
        flag.store(true, Ordering::Relaxed);
    }
    Ok(())
}
