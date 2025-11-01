// src-tauri/src/lib.rs
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app_state;
mod engine;
mod metadata;
mod model;
mod plugin;
mod runtime;

use app_state::AppState;
use metadata::MetaIndexer;
use tauri::Emitter; // for app.emit

// ✅ add hwprof (minimal)
use strata_hwprof::{hwprof_profile_path, validate_or_redetect};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState::new())
        .manage(MetaIndexer::new())
        // plugins
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        // setup (intentionally minimal; UI stays snappy)
        .setup(|app| {
            // ✅ kick off hardware detection/cache in the background
            let app_handle = app.handle().clone();
            tauri::async_runtime::spawn_blocking(move || {
                match validate_or_redetect() {
                    Ok(profile) => {
                        eprintln!(
                            "[hwprof] ready: {} | arch={} | threads={} | backends: cpu={} cuda={} rocm={} vulkan={} metal={}",
                            profile.cpu.brand,
                            profile.arch,
                            profile.cpu.threads,
                            profile.backends.cpu,
                            profile.backends.cuda,
                            profile.backends.rocm,
                            profile.backends.vulkan,
                            profile.backends.metal
                        );
                        eprintln!("[hwprof] cache: {}", hwprof_profile_path().display());
                        // keep your existing frontend listener happy
                        let _ = app_handle.emit("strata://hwprofile", &profile);
                    }
                    Err(e) => eprintln!("[hwprof] detection failed: {e:?}"),
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // system prompt
            engine::load_system_prompt,
            // model list / selection
            model::get_model_list,
            model::get_active_model,
            model::set_active_model_cmd,
            model::get_models_root,
            // import
            model::import_model,
            // metadata
            metadata::get_model_metadata,
            metadata::meta_start_index,
            metadata::meta_status,
            metadata::meta_get_cached,
            metadata::meta_clear,
            // inference
            engine::run_llm,
            engine::run_llm_stream,
            engine::cancel_generation,
            // NEW: preload command (safe no-op if engine already exists)
            engine::preload_engine,
            // installer
            runtime::is_llama_runtime_installed,
            runtime::run_runtime_installer,
        ])
        .run(tauri::generate_context!())
        .expect("Failed to launch Tauri app");
}
