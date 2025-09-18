use once_cell::sync::Lazy;
use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;
use tauri::{AppHandle, Manager};

#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelEntry {
    /// Stable id: *relative path* under the active root (user library or dev fallback), `/` separators.
    pub id: String,
    /// Human name: file stem (no extension)
    pub name: String,
    /// Absolute path on disk
    pub path: PathBuf,
    /// Backend hint by extension
    pub backend_hint: String,
    /// Lowercased extension
    pub file_type: String,
    /// Parent directory name
    pub family: String,
}

pub const ALLOWED_MODEL_EXTS: &[&str] = &["gguf", "safetensors", "onnx", "bin"];

#[inline]
fn infer_backend_hint(ext: &str) -> &'static str {
    match ext {
        "gguf" | "bin" => "llama",
        "safetensors" => "transformers",
        "onnx" => "onnx",
        _ => "unknown",
    }
}

/// Return the **user models root** (platform-correct), creating it if missing.
pub fn user_models_root(app: &AppHandle) -> Result<PathBuf, String> {
    let mut root = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app_data_dir: {e}"))?;
    root.push("models");
    if !root.exists() {
        fs::create_dir_all(&root)
            .map_err(|e| format!("Failed to create models dir {}: {e}", root.display()))?;
    }
    Ok(root)
}

/// Return a dev fallback `./resources/models` if present (for development only).
fn dev_models_root() -> Option<PathBuf> {
    // 1) packaged resources/models (when available)
    // NOTE: We *don’t* require this in production; just a dev fallback.
    let cwd_res = std::env::current_dir()
        .ok()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("resources")
        .join("models");
    if cwd_res.exists() {
        return Some(cwd_res);
    }
    None
}

/// Preferred models root used for resolving selected model paths.
/// - Always the user library dir.
/// - `list_available_models` will also surface dev models if user library is empty.
pub fn resolve_models_root(app: &AppHandle) -> Result<PathBuf, String> {
    user_models_root(app)
}

/// Build a **relative id** under `models_root` with `/` separators.
fn rel_id(models_root: &Path, abs: &Path) -> Option<String> {
    let rel = abs.strip_prefix(models_root).ok()?;
    let mut parts: Vec<String> = Vec::new();
    for comp in rel.components() {
        if let Component::Normal(os) = comp {
            parts.push(os.to_string_lossy().to_string());
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("/"))
    }
}

impl ModelEntry {
    pub fn from_abs_path(models_root: &Path, abs_path: PathBuf) -> Option<Self> {
        if !abs_path.is_file() {
            return None;
        }
        let ext = abs_path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_lowercase())?;
        if !ALLOWED_MODEL_EXTS.contains(&ext.as_str()) {
            return None;
        }

        let backend_hint = infer_backend_hint(&ext);
        let file_stem = abs_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let family = abs_path
            .parent()
            .and_then(Path::file_name)
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        let id = rel_id(models_root, &abs_path).unwrap_or_else(|| file_stem.clone());

        Some(Self {
            id,
            name: file_stem,
            path: abs_path,
            backend_hint: backend_hint.to_string(),
            file_type: ext,
            family,
        })
    }
}

// ----------------- selection -----------------

static CURRENT_MODEL_ID: Lazy<Mutex<Option<String>>> = Lazy::new(|| Mutex::new(None));

pub fn set_current_model(id: String) {
    let mut slot = CURRENT_MODEL_ID.lock().expect("CURRENT_MODEL_ID poisoned");
    *slot = Some(id);
}

pub fn get_current_model() -> Option<String> {
    CURRENT_MODEL_ID
        .lock()
        .expect("CURRENT_MODEL_ID poisoned")
        .clone()
}

// ----------------- listing + resolving -----------------

/// List available models prioritizing the user library.
/// If the user library is **empty**, include dev fallback `./resources/models`.
pub fn list_available_models(app: AppHandle) -> Result<Vec<ModelEntry>, String> {
    let user_root = user_models_root(&app)?;
    let mut entries = Vec::new();

    walk_dir(&user_root, &user_root, &mut entries, &mut HashSet::new())?;

    // Dev fallback only if user library is empty (avoid id collisions)
    if entries.is_empty() {
        if let Some(dev_root) = dev_models_root() {
            walk_dir(&dev_root, &dev_root, &mut entries, &mut HashSet::new())?;
        }
    }

    // Pleasant sort: family then name
    entries.sort_by(|a, b| {
        (a.family.to_lowercase(), a.name.to_lowercase())
            .cmp(&(b.family.to_lowercase(), b.name.to_lowercase()))
    });

    Ok(entries)
}

fn walk_dir(
    models_root: &Path,
    dir: &Path,
    entries: &mut Vec<ModelEntry>,
    visited: &mut HashSet<PathBuf>,
) -> Result<(), String> {
    if !visited.insert(dir.to_path_buf()) {
        return Ok(());
    }
    let read_dir =
        fs::read_dir(dir).map_err(|e| format!("Failed to read {}: {e}", dir.display()))?;

    for entry in read_dir {
        let entry = entry.map_err(|e| format!("Failed to read entry in {}: {e}", dir.display()))?;
        let path = entry.path();

        if path.is_dir() {
            walk_dir(models_root, &path, entries, visited)?;
        } else if path.is_file() {
            if let Some(model_entry) = ModelEntry::from_abs_path(models_root, path) {
                entries.push(model_entry);
            }
        }
    }
    Ok(())
}

/// Resolve the **absolute path** for the current id:
/// - First try user library
/// - If missing there and library is empty, try dev fallback
pub fn get_model_path(app: &AppHandle) -> Result<PathBuf, String> {
    let rel_id = get_current_model().ok_or("No model selected")?;
    let user_root = user_models_root(app)?;
    let abs_user = user_root.join(Path::new(&rel_id));
    if abs_user.is_file() {
        return Ok(abs_user);
    }

    if let Some(dev_root) = dev_models_root() {
        let abs_dev = dev_root.join(Path::new(&rel_id));
        if abs_dev.is_file() {
            return Ok(abs_dev);
        }
    }

    Err(format!("Selected model not found: {}", rel_id))
}

// ----------------- import -----------------

/// Import a model file into the **user library** (by copy).
/// If `family` is provided, place the file under that subdir; otherwise derive from filename.
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
            // default: parent dir name or "Library"
            src.parent()
                .and_then(Path::file_name)
                .and_then(|n| n.to_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "Library".to_string());

    let dest_dir = user_root.join(&family_dir);
    if !dest_dir.exists() {
        fs::create_dir_all(&dest_dir)
            .map_err(|e| format!("Failed to create {}: {e}", dest_dir.display()))?;
    }

    let dest = dest_dir.join(file_name);
    copy_atomic(src, &dest)
        .map_err(|e| format!("Copy failed {} → {}: {e}", src.display(), dest.display()))?;

    ModelEntry::from_abs_path(&user_root, dest)
        .ok_or_else(|| "Failed to build ModelEntry after import".into())
}

/// Copy with a temporary file then rename to avoid partial file states.
fn copy_atomic(src: &Path, dest: &Path) -> io::Result<()> {
    let tmp = dest.with_extension("tmpcopy");
    if tmp.exists() {
        let _ = fs::remove_file(&tmp);
    }
    fs::copy(src, &tmp)?;
    // Preserve original filename atomically
    fs::rename(&tmp, dest)?;
    Ok(())
}
