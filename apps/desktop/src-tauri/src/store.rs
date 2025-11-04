// src-tauri/src/store.rs
//! Tauri integration layer for the `strata-plugins` crate.
//! Exposes manifest management and runtime installation commands to the frontend.
//! All blocking operations are dispatched to worker threads to keep the UI responsive.

use tauri::{AppHandle, Emitter, State};

use strata_plugins::{
    state::PluginsState,
    tauri_api,
    types::{Manifest, Pref, RuntimeChoice},
};

/// Converts a string argument from the UI into a [`Pref`] variant.
/// Accepts lowercase or mixed-case strings; defaults to `Auto` if unspecified.
fn parse_pref(s: Option<String>) -> Result<Pref, String> {
    match s {
        None => Ok(Pref::Auto),
        Some(v) => {
            let v = v.to_ascii_lowercase();
            match v.as_str() {
                "auto" => Ok(Pref::Auto),
                "cpu" => Ok(Pref::Cpu),
                #[cfg(any(target_os = "linux", target_os = "windows"))]
                "cuda" => Ok(Pref::Cuda),
                #[cfg(any(target_os = "linux", target_os = "windows"))]
                "vulkan" => Ok(Pref::Vulkan),
                #[cfg(target_os = "macos")]
                "metal" => Ok(Pref::Metal),
                _ => Err(format!("unknown preference: {v}")),
            }
        }
    }
}

/// Fetches or refreshes the runtime manifest and caches it in memory.
///
/// Emits:
/// - `strata://store/manifest-refreshed` – contains `{ llama: count }`
#[tauri::command]
pub async fn store_refresh_manifest(
    app: AppHandle,
    state: State<'_, PluginsState>,
    url: Option<String>,
) -> Result<Manifest, String> {
    let st = (*state).clone();

    tauri::async_runtime::spawn_blocking(move || tauri_api::refresh_manifest(&st, url.as_deref()))
        .await
        .map_err(|e| format!("join error: {e}"))?
        .map_err(|e| e.to_string())?;

    let manifest = state
        .manifest()
        .ok_or_else(|| "manifest not loaded (unexpected)".to_string())?;

    let _ = app.emit(
        "strata://store/manifest-refreshed",
        serde_json::json!({ "llama": manifest.llama.len() }),
    );

    Ok(manifest)
}

/// Returns the most recently loaded manifest without performing any network requests.
#[tauri::command]
pub fn store_list_entries(state: State<'_, PluginsState>) -> Result<Manifest, String> {
    state
        .manifest()
        .ok_or_else(|| "no manifest loaded; call store_refresh_manifest first".to_string())
}

/// Generates a runtime installation plan for preview purposes.
/// The result indicates which backends will be installed and which GPU (if any) is active.
#[tauri::command]
pub fn store_plan_install(
    state: State<'_, PluginsState>,
    prefer: Option<String>,
) -> Result<RuntimeChoice, String> {
    let pref = parse_pref(prefer)?;
    tauri_api::plan_install(&*state, pref).map_err(|e| e.to_string())
}

/// Downloads and installs runtime variants based on user preference or hardware detection.
///
/// Emits:
/// - `strata://store/install-start` – installation has begun
/// - `strata://runtime-changed` – new runtime configuration is ready
/// - `strata://store/install-complete` – installation finished successfully
#[tauri::command]
pub async fn store_install_runtime(
    app: AppHandle,
    state: State<'_, PluginsState>,
    prefer: Option<String>,
) -> Result<Vec<String>, String> {
    let pref = parse_pref(prefer)?;
    let st = (*state).clone();

    let _ = app.emit("strata://store/install-start", serde_json::json!({}));

    // Compute which variants will be installed so we can report final state later.
    let choice = tauri_api::plan_install(&st, pref).map_err(|e| e.to_string())?;

    // Perform installation on a background thread.
    let app2 = app.clone();
    let st2 = st.clone();
    let res = tauri::async_runtime::spawn_blocking(move || tauri_api::execute_install(&st2, pref))
        .await
        .map_err(|e| format!("join error: {e}"))?;
    let installed = res.map_err(|e| e.to_string())?;

    let active = choice.active_gpu.unwrap_or_else(|| "cpu".to_string());

    let _ = app.emit(
        "strata://runtime-changed",
        serde_json::json!({ "active": active, "installed": installed }),
    );
    let _ = app2.emit(
        "strata://store/install-complete",
        serde_json::json!({ "installed": installed }),
    );

    Ok(installed)
}

/// Installs an individual plugin. Not yet implemented.
#[tauri::command]
pub async fn store_install_plugin(
    _app: AppHandle,
    _state: State<'_, PluginsState>,
    _plugin_id: String,
    _version: Option<String>,
) -> Result<(), String> {
    Err("plugin install not implemented yet".into())
}

/// Cancels an in-progress installation. Currently a no-op.
#[tauri::command]
pub fn store_cancel(_job_id: Option<String>) -> Result<(), String> {
    Ok(())
}
