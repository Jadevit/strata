use super::list::{ALLOWED_MODEL_EXTS, ModelEntry, user_models_root};
use std::{
    fs,
    path::{Path, PathBuf},
};
use tauri::AppHandle;

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
    let dest_dir = family
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| user_root.join(s))
        .unwrap_or_else(|| user_root.clone());

    if !dest_dir.exists() {
        fs::create_dir_all(&dest_dir).map_err(|e| format!("mkdir {}: {e}", dest_dir.display()))?;
    }

    let mut dest = dest_dir.join(file_name);

    if dest.exists() {
        let stem = dest.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
        let ext = dest.extension().and_then(|e| e.to_str()).unwrap_or("");
        let salt =
            format!("{:x}", fxhash::hash64(src.to_string_lossy().as_bytes()))[..8].to_string();
        let new_name = if ext.is_empty() {
            format!("{stem}-{salt}")
        } else {
            format!("{stem}-{salt}.{ext}")
        };
        dest = dest_dir.join(new_name);
    }

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
