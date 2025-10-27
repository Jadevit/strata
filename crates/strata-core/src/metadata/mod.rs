//! Global registry + helpers for backend-agnostic metadata collection.
//!
//! Design:
//! - `MetadataService` lives in this parent module so child modules can access
//!   its private fields without making them pub(crate).
//! - `service.rs` implements the registry & public API.
//! - `dynamic.rs` contains unsafe dylib utilities (kept small & isolated).
//! - `dto.rs` holds UI-facing DTO + mapping.
//!
//! NOTE: Dynamic plugins require ABI care. See docs in `dynamic.rs`.

use libloading::Library;
use std::path::Path;
use strata_abi::metadata::{BackendMetadataProvider, ModelCoreInfo};

/// In-process registry of metadata providers (static + dynamic).
/// Private fields; only child modules may touch them.
struct MetadataService {
    providers: Vec<Box<dyn BackendMetadataProvider>>,
    /// Keep libraries alive for the duration of the process to ensure any
    /// provider vtables / function pointers remain valid.
    _libs: Vec<Library>,
}

impl MetadataService {
    fn new() -> Self {
        Self {
            providers: Vec::new(),
            _libs: Vec::new(),
        }
    }

    /// Append a provider to the registry (registration order == resolution order).
    fn register(&mut self, p: Box<dyn BackendMetadataProvider>) {
        self.providers.push(p);
    }

    /// Find the first provider that claims to handle this file and collect metadata.
    fn collect_for(&self, file: &Path) -> Result<ModelCoreInfo, String> {
        for p in &self.providers {
            if p.can_handle(file) {
                return p.collect(file);
            }
        }
        Err(format!(
            "No metadata provider can handle {}",
            file.display()
        ))
    }
}

// Public API, registry impl, and dynamic loading entrypoint.
mod service;
pub use service::{
    collect_model_metadata, load_metadata_plugins, register_backend_metadata_provider,
};

// UI DTOs + mapper.
mod dto;
pub use dto::{ModelMetaOut, to_ui_meta};

// Unsafe/dylib helpers are kept private to this module.
mod dynamic;
