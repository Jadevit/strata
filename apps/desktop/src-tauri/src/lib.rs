#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app_state;
mod engine;
mod metadata;
mod model;
mod plugin;
mod runtime;

use app_state::AppState;
use metadata::MetaIndexer;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState::new())
        .manage(MetaIndexer::new())
        // plugins
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        // setup (no-op for now)
        .setup(|_app| Ok(()))
        .invoke_handler(tauri::generate_handler![
            // system prompt (lives in engine.rs)
            engine::load_system_prompt,
            // model list / selection
            model::get_model_list,
            model::get_active_model,
            model::set_active_model_cmd,
            model::get_models_root,
            // import
            model::import_model,
            // single-file metadata
            metadata::get_model_metadata,
            // metadata indexer controls
            metadata::meta_start_index,
            metadata::meta_status,
            metadata::meta_get_cached,
            metadata::meta_clear,
            // inference
            engine::run_llm,
            engine::run_llm_stream,
            engine::cancel_generation,
            // installer
            runtime::is_llama_runtime_installed,
            runtime::run_runtime_installer,
        ])
        .run(tauri::generate_context!())
        .expect("Failed to launch Tauri app");
}
