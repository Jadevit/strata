// src-tauri/src/model/mod.rs
mod import;
mod list;
mod select;

pub use import::import_into_user_library;
pub use list::{ModelEntry, list_available_models, resolve_models_root, user_models_root};
pub use select::{get_current_model, get_model_path, set_current_model};

use tauri::{AppHandle, Emitter, State};

use crate::app_state::AppState;

// --- Tauri command facades kept at module root to preserve lib.rs handler paths ---

#[tauri::command]
pub async fn get_model_list(app: AppHandle) -> Result<Vec<ModelEntry>, String> {
    let app2 = app.clone();
    tauri::async_runtime::spawn_blocking(move || list_available_models(app2))
        .await
        .map_err(|e| format!("join error: {e}"))?
}

#[tauri::command]
pub fn get_active_model() -> Option<String> {
    get_current_model()
}

#[tauri::command]
pub fn get_models_root(app: AppHandle) -> Result<String, String> {
    resolve_models_root(&app).map(|p| p.display().to_string())
}

#[tauri::command]
pub async fn import_model(
    app: AppHandle,
    src_path: String,
    family: Option<String>,
) -> Result<ModelEntry, String> {
    let app2 = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        import_into_user_library(&app2, std::path::Path::new(&src_path), family.as_deref())
    })
    .await
    .map_err(|e| format!("join error: {e}"))?
}

/// Set the active model *and* hard-reinit the engine so the context/KV swap cleanly.
#[tauri::command]
pub async fn set_active_model_cmd(
    app: AppHandle,
    state: State<'_, AppState>,
    name: String,
) -> Result<(), String> {
    // Persist the selection
    set_current_model(name.clone());

    // Immediately tell the UI we're switching
    let _ = app.emit("strata://model-switch-start", &name);

    let app2 = app.clone();
    let state2 = AppState {
        memory: std::sync::Arc::clone(&state.memory),
        current_stop: std::sync::Arc::clone(&state.current_stop),
        engine: std::sync::Arc::clone(&state.engine),
    };

    tauri::async_runtime::spawn_blocking(move || {
        // Drop old engine + rebuild new one off the main thread
        crate::engine::reinit_engine_to_current_model(&app2, &state2)
    })
    .await
    .map_err(|e| format!("join error: {e}"))??;

    // Tell the UI we’re done
    let _ = app.emit("strata://model-switched", &name);
    Ok(())
}
