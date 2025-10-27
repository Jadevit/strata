use std::{fs, path::PathBuf};
use tauri::{AppHandle, Manager, path::BaseDirectory};

// Async helper used by the command in engine::mod
pub async fn load_system_prompt_impl(app: AppHandle) -> Result<String, String> {
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

// Sync helper used during engine init
pub(crate) fn load_system_prompt_sync(app: &AppHandle) -> Option<String> {
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
