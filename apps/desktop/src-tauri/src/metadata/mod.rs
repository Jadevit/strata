//! Metadata module: Tauri commands live here; helpers live in submodules.

mod indexer;
mod provider; // retained for now; no longer used by get_model_metadata

use tauri::{AppHandle, State};

use strata_core::metadata::{ModelMetaOut, collect_model_metadata, to_ui_meta};

pub use indexer::{MetaIndexStatus, MetaIndexer, cached_read_meta_path, cached_write_meta_path};

// ---- Tauri commands ----

#[tauri::command]
pub async fn get_model_metadata(app: AppHandle) -> Result<ModelMetaOut, String> {
    let path = crate::model::get_model_path(&app)?;

    // Fast path: disk cache
    if let Some(cached) = cached_read_meta_path(&path) {
        return Ok(cached);
    }

    // Clone before moving into the blocking worker to avoid use-after-move
    let path_for_worker = path.clone();

    // Slow path: collect from core registry, then cache
    let info =
        tauri::async_runtime::spawn_blocking(move || collect_model_metadata(&path_for_worker))
            .await
            .map_err(|e| format!("join error: {e}"))??;

    let ui = to_ui_meta(&info);
    let _ = cached_write_meta_path(&path, &ui);
    Ok(ui)
}

#[tauri::command]
pub async fn meta_start_index(
    app: AppHandle,
    index: State<'_, MetaIndexer>,
    force: Option<bool>,
) -> Result<(), String> {
    index.start(app, force.unwrap_or(false))
}

#[tauri::command]
pub fn meta_status(index: State<'_, MetaIndexer>) -> MetaIndexStatus {
    index.status()
}

#[tauri::command]
pub fn meta_get_cached(id: String, index: State<'_, MetaIndexer>) -> Option<ModelMetaOut> {
    index.get(&id)
}

#[tauri::command]
pub fn meta_clear(index: State<'_, MetaIndexer>) {
    index.clear();
}
