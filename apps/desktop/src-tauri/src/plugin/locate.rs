use crate::runtime::{
    default_runtime_root, runtime_cpu_fallback_path, runtime_current_lib_dir,
    runtime_plugin_filename,
};
use std::{
    env,
    path::{Path, PathBuf},
};

const ENV_PLUGIN_PATH: &str = "STRATA_PLUGIN_PATH";
const ENV_RUNTIME_DIR: &str = "STRATA_RUNTIME_DIR";

pub(crate) fn locate_plugin_binary() -> Option<PathBuf> {
    if let Ok(p) = env::var(ENV_PLUGIN_PATH) {
        let p = PathBuf::from(p);
        if p.exists() {
            eprintln!("[plugin] {ENV_PLUGIN_PATH} = {}", p.display());
            return Some(p);
        } else {
            eprintln!(
                "[plugin] {ENV_PLUGIN_PATH} points to missing file: {}",
                p.display()
            );
        }
    }

    let root = env::var(ENV_RUNTIME_DIR)
        .ok()
        .map(PathBuf::from)
        .or_else(default_runtime_root)?;

    if let (Some(dir), Some(file)) = (
        runtime_current_lib_dir(&root),
        runtime_plugin_filename(&root),
    ) {
        let p = dir.join(file);
        if p.exists() {
            eprintln!("[plugin] from runtime.json (active): {}", p.display());
            return Some(p);
        }
    }

    None
}

pub(crate) fn locate_runtime_ll_lib(_plugin_path: &Path) -> Option<PathBuf> {
    None
}
