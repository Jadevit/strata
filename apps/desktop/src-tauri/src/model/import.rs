use std::{
    fs,
    path::{Path, PathBuf},
};
use tauri::AppHandle;

use super::list::{ALLOWED_MODEL_EXTS, ModelEntry, user_models_root};

pub fn import_into_user_library(
    app: &AppHandle,
    src: &Path,
    family: Option<&str>,
) -> Result<ModelEntry, String> {
    if !src.is_file() {
        return Err(format!("Source is not a file: {}", src.display()));
    }

    let ext = src
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())
        .ok_or_else(|| "File has no extension".to_string())?;
    if !ALLOWED_MODEL_EXTS.contains(&ext.as_str()) {
        return Err(format!("Unsupported extension .{}", ext));
    }

    let file_name = src
        .file_name()
        .ok_or_else(|| "Invalid source file name".to_string())?;
    let user_root = user_models_root(app)?;
    let family_dir = family
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .or_else(|| {
            src.parent()
                .and_then(Path::file_name)
                .and_then(|n| n.to_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "Library".to_string());

    let dest_dir = user_root.join(&family_dir);
    if !dest_dir.exists() {
        fs::create_dir_all(&dest_dir).map_err(|e| format!("mkdir {}: {e}", dest_dir.display()))?;
    }

    let dest = dest_dir.join(file_name);
    copy_atomic(src, &dest)
        .map_err(|e| format!("Copy failed {} → {}: {e}", src.display(), dest.display()))?;

    ModelEntry::from_abs_path(&user_root, dest)
        .ok_or_else(|| "Failed to build ModelEntry after import".into())
}

fn copy_atomic(src: &Path, dest: &Path) -> std::io::Result<()> {
    let tmp = dest.with_extension("tmpcopy");
    if tmp.exists() {
        let _ = fs::remove_file(&tmp);
    }
    fs::copy(src, &tmp)?;
    fs::rename(&tmp, dest)?;
    Ok(())
}
