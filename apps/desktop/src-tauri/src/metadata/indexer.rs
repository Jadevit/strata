use serde::Serialize;
use tauri::{AppHandle, Emitter};

use crate::model::{ModelEntry, list_available_models};
use strata_core::metadata::{ModelMetaOut, collect_model_metadata, to_ui_meta};

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
                // 1) try disk cache
                let meta = if let Some(cached) = cached_read_meta_path(&m.path) {
                    cached
                } else {
                    // 2) collect fresh then persist
                    match collect_model_metadata(&m.path) {
                        Ok(info) => {
                            let ui = to_ui_meta(&info);
                            let _ = cached_write_meta_path(&m.path, &ui);
                            ui
                        }
                        Err(e) => {
                            let _ = app
                                .emit("meta-error", serde_json::json!({ "id": m.id, "error": e }));
                            {
                                me.inner.write().unwrap().status.done = i + 1;
                            }
                            continue;
                        }
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

// ------------------------------
// Tiny on-disk cache (no new deps)
// ~/.local/share/Strata/cache/meta/cache.json
// Keyed by absolute path, invalidated by (size, mtime).
// ------------------------------
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[derive(Debug, Default, Serialize, Deserialize)]
struct CacheFile {
    entries: HashMap<String, CacheEntry>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct CacheEntry {
    size: u64,
    mtime_ns: u128,
    meta: ModelMetaOut,
}

fn meta_cache_root() -> PathBuf {
    strata_hwprof::cache_dir().join("meta")
}
fn meta_cache_file() -> PathBuf {
    meta_cache_root().join("cache.json")
}

fn load_cache() -> CacheFile {
    let path = meta_cache_file();
    if let Ok(bytes) = fs::read(&path) {
        if let Ok(cf) = serde_json::from_slice::<CacheFile>(&bytes) {
            return cf;
        }
    }
    CacheFile::default()
}

fn save_cache(cf: &CacheFile) -> Result<(), String> {
    let root = meta_cache_root();
    fs::create_dir_all(&root).map_err(|e| format!("mkd {}: {e}", root.display()))?;
    let path = meta_cache_file();
    let tmp = root.join("cache.json.tmp");
    let bytes = serde_json::to_vec_pretty(cf).map_err(|e| e.to_string())?;
    fs::write(&tmp, &bytes).map_err(|e| format!("write {}: {e}", tmp.display()))?;
    fs::rename(&tmp, &path).map_err(|e| format!("rename {}: {e}", path.display()))
}

fn fingerprint_for(p: &Path) -> Option<(u64, u128)> {
    let md = fs::metadata(p).ok()?;
    let size = md.len();
    let mtime = md.modified().unwrap_or(SystemTime::UNIX_EPOCH);
    let ns = mtime
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_nanos())?;
    Some((size, ns))
}

/// Public helpers (so mod.rs can reuse cache too)
pub fn cached_read_meta_path(p: &Path) -> Option<ModelMetaOut> {
    let (size, ns) = fingerprint_for(p)?;
    let cf = load_cache();
    let key = p.to_string_lossy().to_string();
    let hit = cf.entries.get(&key)?;
    if hit.size == size && hit.mtime_ns == ns {
        Some(hit.meta.clone())
    } else {
        None
    }
}

pub fn cached_write_meta_path(p: &Path, meta: &ModelMetaOut) -> Result<(), String> {
    let (size, ns) = fingerprint_for(p).ok_or_else(|| "stat failed".to_string())?;
    let mut cf = load_cache();
    let key = p.to_string_lossy().to_string();
    cf.entries.insert(
        key,
        CacheEntry {
            size,
            mtime_ns: ns,
            meta: meta.clone(),
        },
    );
    save_cache(&cf)
}
