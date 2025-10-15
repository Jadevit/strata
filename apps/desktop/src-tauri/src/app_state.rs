use std::sync::{Arc, Mutex, atomic::AtomicBool};
use strata_core::memory::SessionMemory;

pub struct AppState {
    pub memory: Arc<Mutex<SessionMemory>>,
    pub current_stop: Arc<Mutex<Option<Arc<AtomicBool>>>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            memory: Arc::new(Mutex::new(SessionMemory::new())),
            current_stop: Arc::new(Mutex::new(None)),
        }
    }
}
