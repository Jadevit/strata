//! Metadata module: Tauri commands live here; helpers live in submodules.

mod indexer;
mod provider;

use tauri::{AppHandle, State};

use strata_core::metadata::{ModelMetaOut, to_ui_meta};

pub use indexer::{MetaIndexStatus, MetaIndexer};
use provider::collect_model_metadata_via_plugin;

// ---- Tauri commands ----

#[tauri::command]
pub async fn get_model_metadata(app: AppHandle) -> Result<ModelMetaOut, String> {
    let path = crate::model::get_model_path(&app)?;
    let info =
        tauri::async_runtime::spawn_blocking(move || collect_model_metadata_via_plugin(&path))
            .await
            .map_err(|e| format!("join error: {e}"))??;
    Ok(to_ui_meta(&info))
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
