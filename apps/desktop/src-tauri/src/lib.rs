#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    sync::{Arc, Mutex},
};

use tauri::{AppHandle, Emitter, Manager, State, path::BaseDirectory};

mod metadata_indexer;
mod model;

use metadata_indexer::{MetaIndexStatus, MetaIndexer};
use model::{
    ModelEntry, get_current_model, get_model_path, import_into_user_library, list_available_models,
    resolve_models_root, set_current_model, user_models_root,
};

use strata_core::backends::llama_backend::LlamaBackendImpl;
use strata_core::engine::engine::LLMEngine;
use strata_core::format::prompt_format::PromptKind;
use strata_core::memory::SessionMemory;
use strata_core::metadata::metadata_providers::register_all_metadata_providers;
use strata_core::metadata::metadata_service::{ModelMetaOut, collect_model_metadata, to_ui_meta};
use strata_core::traits::backend::{ChatTurn, LLMBackend};

// ---------- Types ----------
type ModelMetaDTO = ModelMetaOut;

/// App state (thread-safe for background tasks).
struct AppState {
    memory: Arc<Mutex<SessionMemory>>,
    current_stop: Arc<Mutex<Option<Arc<AtomicBool>>>>,
}

// ---------- System prompt ----------
#[tauri::command]
async fn load_system_prompt(app: AppHandle) -> Result<String, String> {
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

// ---------- Model discovery / selection ----------
#[tauri::command]
async fn get_model_list(app: AppHandle) -> Result<Vec<ModelEntry>, String> {
    let app2 = app.clone();
    tauri::async_runtime::spawn_blocking(move || list_available_models(app2))
        .await
        .map_err(|e| format!("join error: {e}"))?
}

#[tauri::command]
fn get_active_model() -> Option<String> {
    get_current_model()
}

#[tauri::command]
fn set_active_model_cmd(name: String) {
    set_current_model(name);
}

// Useful for UI: show the library location
#[tauri::command]
fn get_models_root(app: AppHandle) -> Result<String, String> {
    resolve_models_root(&app).map(|p| p.display().to_string())
}

// ---------- Import ----------
#[tauri::command]
async fn import_model(
    app: AppHandle,
    src_path: String,
    family: Option<String>,
) -> Result<ModelEntry, String> {
    let app2 = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let entry = import_into_user_library(&app2, Path::new(&src_path), family.as_deref())?;
        Ok::<_, String>(entry)
    })
    .await
    .map_err(|e| format!("join error: {e}"))?
}

// ---------- Metadata (single file) ----------
#[tauri::command]
async fn get_model_metadata(app: AppHandle) -> Result<ModelMetaDTO, String> {
    let path = get_model_path(&app)?;
    let info = tauri::async_runtime::spawn_blocking(move || collect_model_metadata(&path))
        .await
        .map_err(|e| format!("join error: {e}"))??;
    Ok(to_ui_meta(&info))
}

// ---------- Metadata indexer (background cache) ----------
#[tauri::command]
async fn meta_start_index(
    app: AppHandle,
    index: State<'_, MetaIndexer>,
    force: Option<bool>,
) -> Result<(), String> {
    index.start(app, force.unwrap_or(false))
}

#[tauri::command]
fn meta_status(index: State<'_, MetaIndexer>) -> MetaIndexStatus {
    index.status()
}

#[tauri::command]
fn meta_get_cached(id: String, index: State<'_, MetaIndexer>) -> Option<ModelMetaDTO> {
    index.get(&id)
}

#[tauri::command]
fn meta_clear(index: State<'_, MetaIndexer>) {
    index.clear();
}

// ---------- Inference helpers ----------
fn pick_prompt_strategy(model_id: Option<String>, system: Option<String>) -> PromptKind {
    if let Some(id) = model_id {
        let id_lc = id.to_lowercase();
        if id_lc.contains("phi") || id_lc.contains("vira") {
            return PromptKind::Phi3 { system };
        }
    }
    PromptKind::ChatMl { system }
}

// ---------- Non-streaming ----------
#[tauri::command]
async fn run_llm(
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
        let backend = LlamaBackendImpl::load(&model_path)?;
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

// ---------- Streaming ----------
#[tauri::command]
async fn run_llm_stream(
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
        let backend = LlamaBackendImpl::load(&model_path)?;
        let system = load_system_prompt_sync(&app2);

        let mut engine = LLMEngine::with_auto(backend, system.clone());
        engine.set_strategy(pick_prompt_strategy(
            model_id2.or_else(get_active_model),
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

// ---------- Cancel ----------
#[tauri::command]
fn cancel_generation(state: State<'_, AppState>) -> Result<(), String> {
    if let Some(flag) = state.current_stop.lock().unwrap().as_ref() {
        flag.store(true, Ordering::Relaxed);
    }
    Ok(())
}

// ---------- App bootstrap ----------
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState {
            memory: Arc::new(Mutex::new(SessionMemory::new())),
            current_stop: Arc::new(Mutex::new(None)),
        })
        .manage(MetaIndexer::new())
        // plugins
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|_app| {
            register_all_metadata_providers();
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // system prompt
            load_system_prompt,
            // model list / selection
            get_model_list,
            get_active_model,
            set_active_model_cmd,
            get_models_root,
            // import
            import_model,
            // single-file metadata
            get_model_metadata,
            // metadata indexer controls
            meta_start_index,
            meta_status,
            meta_get_cached,
            meta_clear,
            // inference
            run_llm,
            run_llm_stream,
            cancel_generation,
        ])
        .run(tauri::generate_context!())
        .expect("Failed to launch Tauri app");
}
