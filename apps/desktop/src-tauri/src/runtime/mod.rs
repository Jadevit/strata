// apps/desktop/src-tauri/src/runtime/mod.rs

mod discover;
mod install;

// Re-export discover helpers for other modules (plugin locate, etc.)
pub use discover::{default_runtime_root, runtime_current_lib_dir, runtime_is_monolith};

/// Returns true if a runtime.json exists in the default runtime root.
#[tauri::command]
pub fn is_llama_runtime_installed() -> bool {
    default_runtime_root()
        .map(|root| root.join("runtime.json").exists())
        .unwrap_or(false)
}

/// Launch the sidecar installer with optional args.
#[tauri::command]
pub fn run_runtime_installer(
    prefer: Option<String>,
    manifest: Option<String>,
    app: tauri::AppHandle,
) -> Result<(), String> {
    install::run_installer(prefer, manifest, &app)
}
