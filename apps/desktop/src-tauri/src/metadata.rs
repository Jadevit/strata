use serde::Serialize;
use std::path::Path;
use tauri::{AppHandle, Emitter, Manager, State};

use crate::model::{ModelEntry, list_available_models};
use crate::plugin::load_plugin_once;
use strata_abi::metadata::ModelCoreInfo;
use strata_core::metadata::metadata_service::{ModelMetaOut, collect_model_metadata, to_ui_meta};

// ---- Single-file metadata via plugin API ----
fn make_cstring(s: &str) -> Result<std::ffi::CString, String> {
    std::ffi::CString::new(s).map_err(|_| "string contains interior NUL".to_string())
}

unsafe fn take_plugin_string(
    api_free: strata_abi::ffi::FreeStringFn,
    s: strata_abi::ffi::StrataString,
) -> String {
    if s.ptr.is_null() || s.len == 0 {
        return String::new();
    }
    let slice = std::slice::from_raw_parts(s.ptr as *const u8, s.len);
    let out = String::from_utf8_lossy(slice).into_owned();
    api_free(s);
    out
}

fn collect_model_metadata_via_plugin(path: &Path) -> Result<ModelCoreInfo, String> {
    let plugin = load_plugin_once()?;
    let cpath = make_cstring(path.to_str().ok_or("invalid UTF-8 in path")?)?;
    unsafe {
        let s = (plugin.api.metadata.collect_json)(cpath.as_ptr());
        let js = take_plugin_string(plugin.api.metadata.free_string, s);
        if js.is_empty() {
            let err = (plugin.api.llm.last_error)();
            let msg = take_plugin_string(plugin.api.llm.free_string, err);
            if msg.is_empty() {
                Err("plugin returned empty metadata".into())
            } else {
                Err(msg)
            }
        } else {
            serde_json::from_str::<ModelCoreInfo>(&js)
                .map_err(|e| format!("bad metadata JSON: {e}"))
        }
    }
}

#[tauri::command]
pub async fn get_model_metadata(app: AppHandle) -> Result<ModelMetaOut, String> {
    let path = crate::model::get_model_path(&app)?;
    let info =
        tauri::async_runtime::spawn_blocking(move || collect_model_metadata_via_plugin(&path))
            .await
            .map_err(|e| format!("join error: {e}"))??;
    Ok(to_ui_meta(&info))
}

// ---- Metadata indexer (merged from metadata_indexer.rs) ----
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndexState {
    Idle,
    Running,
    Ready,
    Error,
}

#[derive(Debug)]
struct StatusInner {
    state: IndexState,
    total: usize,
    done: usize,
    error: Option<String>,
}
impl Default for StatusInner {
    fn default() -> Self {
        Self {
            state: IndexState::Idle,
            total: 0,
            done: 0,
            error: None,
        }
    }
}

#[derive(Debug)]
struct Inner {
    cache: std::collections::HashMap<String, ModelMetaOut>,
    status: StatusInner,
}

/// Thread-safe indexer + cache (managed as Tauri State)
#[derive(Debug, Clone)]
pub struct MetaIndexer {
    inner: std::sync::Arc<std::sync::RwLock<Inner>>,
}

impl MetaIndexer {
    pub fn new() -> Self {
        Self {
            inner: std::sync::Arc::new(std::sync::RwLock::new(Inner {
                cache: std::collections::HashMap::new(),
                status: StatusInner::default(),
            })),
        }
    }

    pub fn start(&self, app: AppHandle, force: bool) -> Result<(), String> {
        {
            let mut g = self.inner.write().unwrap();
            match g.status.state {
                IndexState::Running if !force => {
                    return Ok(());
                }
                _ => {
                    g.status = StatusInner {
                        state: IndexState::Running,
                        total: 0,
                        done: 0,
                        error: None,
                    };
                    if force {
                        g.cache.clear();
                    }
                }
            }
        }

        let me = self.clone();
        let app2 = app.clone();

        tauri::async_runtime::spawn_blocking(move || {
            let list: Vec<ModelEntry> = match list_available_models(app2.clone()) {
                Ok(v) => v,
                Err(e) => {
                    me.fail(&app2, e);
                    return;
                }
            };

            {
                let mut g = me.inner.write().unwrap();
                g.status.total = list.len();
                g.status.done = 0;
            }

            for (i, m) in list.into_iter().enumerate() {
                let meta = match collect_model_metadata(&m.path) {
                    Ok(info) => to_ui_meta(&info),
                    Err(e) => {
                        let _ =
                            app.emit("meta-error", serde_json::json!({ "id": m.id, "error": e }));
                        {
                            let mut g = me.inner.write().unwrap();
                            g.status.done = i + 1;
                        }
                        continue;
                    }
                };

                {
                    let mut g = me.inner.write().unwrap();
                    g.cache.insert(m.id.clone(), meta);
                    g.status.done = i + 1;
                }

                let _ = app.emit(
                    "meta-progress",
                    serde_json::json!({ "done": i + 1, "total": me.total(), "id": m.id, "name": m.name }),
                );
            }

            me.finish(&app);
        });

        Ok(())
    }

    fn finish(&self, app: &AppHandle) {
        {
            let mut g = self.inner.write().unwrap();
            g.status.state = IndexState::Ready;
            g.status.error = None;
        }
        let _ = app.emit(
            "meta-complete",
            serde_json::json!({ "total": self.total() }),
        );
    }

    fn fail(&self, app: &AppHandle, err: String) {
        {
            let mut g = self.inner.write().unwrap();
            g.status.state = IndexState::Error;
            g.status.error = Some(err.clone());
        }
        let _ = app.emit("meta-error", serde_json::json!({ "error": err }));
    }

    pub fn clear(&self) {
        let mut g = self.inner.write().unwrap();
        g.cache.clear();
        g.status = StatusInner::default();
    }

    pub fn get(&self, id: &str) -> Option<ModelMetaOut> {
        self.inner.read().unwrap().cache.get(id).cloned()
    }

    pub fn total(&self) -> usize {
        self.inner.read().unwrap().status.total
    }

    pub fn status(&self) -> MetaIndexStatus {
        let g = self.inner.read().unwrap();
        MetaIndexStatus {
            state: match g.status.state {
                IndexState::Idle => "idle",
                IndexState::Running => "loading",
                IndexState::Ready => "ready",
                IndexState::Error => "error",
            }
            .to_string(),
            total: g.status.total,
            done: g.status.done,
            error: g.status.error.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MetaIndexStatus {
    pub state: String, // "idle" | "loading" | "ready" | "error"
    pub total: usize,
    pub done: usize,
    pub error: Option<String>,
}

// ---- Tauri commands that wrap MetaIndexer ----
#[tauri::command]
pub async fn meta_start_index(
    app: AppHandle,
    index: State<'_, MetaIndexer>,
    force: Option<bool>,
) -> Result<(), String> {
    index.start(app, force.unwrap_or(false))
}

#[tauri::command]
pub fn meta_status(index: State<'_, MetaIndexer>) -> MetaIndexStatus {
    index.status()
}

#[tauri::command]
pub fn meta_get_cached(id: String, index: State<'_, MetaIndexer>) -> Option<ModelMetaOut> {
    index.get(&id)
}

#[tauri::command]
pub fn meta_clear(index: State<'_, MetaIndexer>) {
    index.clear();
}
