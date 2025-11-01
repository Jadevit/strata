// apps/desktop/src-tauri/src/runtime/mod.rs

mod discover;
mod install;

pub use discover::{
    default_runtime_root, runtime_cpu_fallback_path, runtime_current_lib_dir, runtime_is_monolith,
    runtime_plugin_filename,
};

#[tauri::command]
pub fn is_llama_runtime_installed() -> bool {
    default_runtime_root()
        .map(|root| root.join("runtime.json").exists())
        .unwrap_or(false)
}

#[tauri::command]
pub fn run_runtime_installer(
    prefer: Option<String>,
    manifest: Option<String>,
    app: tauri::AppHandle,
) -> Result<(), String> {
    install::run_installer(prefer, manifest, &app)
}
