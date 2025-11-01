use super::list::user_models_root;
use once_cell::sync::Lazy;
use std::{
    path::{Path, PathBuf},
    sync::Mutex,
};
use tauri::AppHandle;

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

pub fn get_model_path(app: &AppHandle) -> Result<PathBuf, String> {
    let rel_id = get_current_model().ok_or("No model selected")?;
    let user_root = user_models_root(app)?;
    let abs_user = user_root.join(Path::new(&rel_id));
    if abs_user.is_file() {
        return Ok(abs_user);
    }
    Err(format!("Selected model not found: {}", rel_id))
}
