use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use serde::Serialize;
use tauri::{AppHandle, Emitter};

use crate::model::{ModelEntry, list_available_models};
use strata_core::metadata::metadata_service::{ModelMetaOut, collect_model_metadata, to_ui_meta};

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
    /// id (ModelEntry.id) -> UI meta
    cache: HashMap<String, ModelMetaOut>,
    status: StatusInner,
}

/// Thread-safe indexer + cache (managed as Tauri State)
#[derive(Debug, Clone)]
pub struct MetaIndexer {
    inner: Arc<RwLock<Inner>>,
}

impl MetaIndexer {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(Inner {
                cache: HashMap::new(),
                status: StatusInner::default(),
            })),
        }
    }

    /// Start or restart the indexer.
    /// If `force=false` and an index is already running, this is a no-op.
    pub fn start(&self, app: AppHandle, force: bool) -> Result<(), String> {
        // atomically decide whether to kick off a run
        {
            let mut g = self.inner.write().unwrap();
            match g.status.state {
                IndexState::Running if !force => {
                    // already running — do nothing
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
            // 1) discover models (blocking)
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

            // 2) scrape each model (blocking), populate cache, emit progress
            for (i, m) in list.into_iter().enumerate() {
                let meta = match collect_model_metadata(&m.path) {
                    Ok(info) => to_ui_meta(&info),
                    Err(e) => {
                        // record error but continue; a single bad file shouldn't kill the run
                        let _ =
                            app.emit("meta-error", serde_json::json!({ "id": m.id, "error": e }));
                        // still advance progress
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
                    serde_json::json!({
                        "done": i + 1,
                        "total": me.total(),
                        "id": m.id,
                        "name": m.name
                    }),
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

    /// Clear cache and reset state to Idle.
    pub fn clear(&self) {
        let mut g = self.inner.write().unwrap();
        g.cache.clear();
        g.status = StatusInner::default();
    }

    /// Fetch a cached entry by id (ModelEntry.id).
    pub fn get(&self, id: &str) -> Option<ModelMetaOut> {
        self.inner.read().unwrap().cache.get(id).cloned()
    }

    /// Number of entries discovered (target count).
    pub fn total(&self) -> usize {
        self.inner.read().unwrap().status.total
    }

    /// Snapshot status for the UI.
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
    /// "idle" | "loading" | "ready" | "error"
    pub state: String,
    pub total: usize,
    pub done: usize,
    pub error: Option<String>,
}
