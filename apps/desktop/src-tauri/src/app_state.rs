use std::sync::{Arc, Mutex, atomic::AtomicBool};

use strata_core::engine::LLMEngine;
use strata_core::memory::SessionMemory;

use crate::plugin::PluginBackend;

/// Global application state.
pub struct AppState {
    /// Chat transcript for the current session (UI builds turns from this).
    pub memory: Arc<Mutex<SessionMemory>>,

    /// Stop flag holder for in-flight generations.
    pub current_stop: Arc<Mutex<Option<Arc<AtomicBool>>>>,

    /// Persisted engine (owns the PluginBackend + llama session / KV).
    /// We reuse this across prompts to avoid reloading or re-prefilling.
    pub engine: Arc<Mutex<Option<LLMEngine<PluginBackend>>>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            memory: Arc::new(Mutex::new(SessionMemory::new())),
            current_stop: Arc::new(Mutex::new(None)),
            engine: Arc::new(Mutex::new(None)),
        }
    }
}
