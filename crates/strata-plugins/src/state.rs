use crate::types::{Manifest, RuntimeChoice};
use std::sync::{Arc, RwLock};

/// Lightweight in-memory state for the store.
#[derive(Clone, Default)]
pub struct PluginsState {
    inner: Arc<RwLock<Inner>>,
}

#[derive(Default)]
struct Inner {
    manifest: Option<Manifest>,
    last_choice: Option<RuntimeChoice>,
}

impl PluginsState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_manifest(&self, m: Manifest) {
        self.inner.write().unwrap().manifest = Some(m);
    }

    pub fn manifest(&self) -> Option<Manifest> {
        self.inner.read().unwrap().manifest.clone()
    }

    pub fn set_choice(&self, c: RuntimeChoice) {
        self.inner.write().unwrap().last_choice = Some(c);
    }

    pub fn last_choice(&self) -> Option<RuntimeChoice> {
        self.inner.read().unwrap().last_choice.clone()
    }
}
