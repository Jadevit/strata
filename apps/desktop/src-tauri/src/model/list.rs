use once_cell::sync::Lazy;
use std::{
    fs,
    path::{Component, Path, PathBuf},
    sync::Mutex,
};
use tauri::{AppHandle, Manager};

#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelEntry {
    pub id: String,    // relative under models root with /
    pub name: String,  // file stem
    pub path: PathBuf, // absolute
    pub backend_hint: String,
    pub file_type: String,
    pub family: String, // parent dir
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

/// User models root, created if missing.
pub fn user_models_root(app: &AppHandle) -> Result<PathBuf, String> {
    let mut root = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("resolve app_data_dir: {e}"))?;
    root.push("models");
    if !root.exists() {
        fs::create_dir_all(&root).map_err(|e| format!("mkdir {}: {e}", root.display()))?;
    }
    Ok(root)
}

/// Dev fallback ./resources/models (if present)
fn dev_models_root() -> Option<PathBuf> {
    let p = std::env::current_dir()
        .ok()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("resources")
        .join("models");
    p.exists().then_some(p)
}

pub fn resolve_models_root(app: &AppHandle) -> Result<PathBuf, String> {
    user_models_root(app)
}

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
        let ext = abs_path.extension()?.to_str().map(|s| s.to_lowercase())?;
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

pub fn list_available_models(app: AppHandle) -> Result<Vec<ModelEntry>, String> {
    let user_root = user_models_root(&app)?;
    let mut entries = Vec::new();

    walk_dir(
        &user_root,
        &user_root,
        &mut entries,
        &mut std::collections::HashSet::new(),
    )?;

    if entries.is_empty() {
        if let Some(dev_root) = dev_models_root() {
            walk_dir(
                &dev_root,
                &dev_root,
                &mut entries,
                &mut std::collections::HashSet::new(),
            )?;
        }
    }

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
    visited: &mut std::collections::HashSet<PathBuf>,
) -> Result<(), String> {
    if !visited.insert(dir.to_path_buf()) {
        return Ok(());
    }
    let read_dir = fs::read_dir(dir).map_err(|e| format!("read {}: {e}", dir.display()))?;
    for entry in read_dir {
        let entry = entry.map_err(|e| format!("entry in {}: {e}", dir.display()))?;
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
