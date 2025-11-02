//! Optional helpers you can expose as Tauri commands (blocking work should be spawned).

use crate::errors::{Result, StoreError};
use crate::install::{choose_variants, install_variants, write_runtime_config};
use crate::manifest::fetch_manifest;
use crate::paths::runtimes_llama_dir;
use crate::state::PluginsState;
use crate::types::{Pref, RuntimeChoice};

/// Fetch and cache the manifest in memory (for quick UI reads).
pub fn refresh_manifest(state: &PluginsState, url: Option<&str>) -> Result<()> {
    let m = fetch_manifest(url.unwrap_or(crate::manifest::DEFAULT_MANIFEST_URL.as_str()))?;
    state.set_manifest(m);
    Ok(())
}

/// Plan which variants will be installed, based on `Pref`.
pub fn plan_install(state: &PluginsState, pref: Pref) -> Result<RuntimeChoice> {
    let m = state
        .manifest()
        .ok_or_else(|| StoreError::Msg("manifest not loaded".into()))?;
    let (_entries, choice) = choose_variants(&m, pref);
    state.set_choice(choice.clone());
    Ok(choice)
}

/// Execute install with the current manifest and return installed variants.
/// Caller should `spawn_blocking` this from Tauri.
pub fn execute_install(state: &PluginsState, pref: Pref) -> Result<Vec<String>> {
    let m = state
        .manifest()
        .ok_or_else(|| StoreError::Msg("manifest not loaded".into()))?;

    let (entries, choice) = choose_variants(&m, pref);
    let root = runtimes_llama_dir();

    let installed = install_variants(&entries, &root)?;
    write_runtime_config(&root, &installed, choice.active_gpu.as_deref())?;
    Ok(installed)
}
