//! Global registry + helpers for backend-agnostic metadata collection.
//! Supports both statically-registered providers and dynamic plugins.

use std::ffi::OsStr;
use std::path::Path;
use std::sync::{OnceLock, RwLock};

use libloading::{Library, Symbol};
use once_cell::sync::Lazy;
use strata_abi::metadata::{BackendMetadataProvider, ModelCoreInfo};

struct MetadataService {
    providers: Vec<Box<dyn BackendMetadataProvider>>,
    _libs: Vec<Library>, // keep libs alive so providers don’t get dropped
}

impl MetadataService {
    fn new() -> Self {
        Self {
            providers: Vec::new(),
            _libs: Vec::new(),
        }
    }

    fn register(&mut self, p: Box<dyn BackendMetadataProvider>) {
        self.providers.push(p);
    }

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

    fn load_dynamic_plugins(&mut self, dir: &Path) -> Result<(), String> {
        if !dir.exists() {
            return Ok(()); // no plugins directory is fine
        }

        for entry in std::fs::read_dir(dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();

            let ext = path.extension().and_then(OsStr::to_str).unwrap_or("");
            if !matches!(ext, "so" | "dll" | "dylib") {
                continue;
            }

            unsafe {
                let lib = Library::new(&path).map_err(|e| e.to_string())?;
                // Convention: each plugin exports a `register_plugin` fn
                let func: Symbol<unsafe extern "C" fn(&mut MetadataService)> =
                    lib.get(b"register_plugin").map_err(|e| e.to_string())?;
                func(self);

                // Store lib so it doesn’t drop
                self._libs.push(lib);
            }
        }

        Ok(())
    }
}

static REGISTRY: OnceLock<RwLock<MetadataService>> = OnceLock::new();

fn registry() -> &'static RwLock<MetadataService> {
    REGISTRY.get_or_init(|| RwLock::new(MetadataService::new()))
}

/// Register a backend metadata provider at startup (static use).
pub fn register_backend_metadata_provider(p: Box<dyn BackendMetadataProvider>) {
    let mut r = registry().write().expect("metadata registry poisoned");
    r.register(p);
}

/// Collect metadata for the given model file using the first provider that can handle it.
pub fn collect_model_metadata(path: &Path) -> Result<ModelCoreInfo, String> {
    let r = registry().read().expect("metadata registry poisoned");
    r.collect_for(path)
}

/// Load dynamic plugins from a directory.  
/// Call this once at app startup (after registry init).
pub fn load_metadata_plugins(dir: &Path) -> Result<(), String> {
    let mut r = registry().write().expect("metadata registry poisoned");
    r.load_dynamic_plugins(dir)
}

// ─────────────────────────────────────────────────────────────────────────────
// UI DTO (co-located for now) + mapper
// ─────────────────────────────────────────────────────────────────────────────

use serde::Serialize;
use strata_abi::backend::PromptFlavor;

/// Matches the UI’s `ModelMeta` (snake_case -> JSON).
#[derive(Debug, Clone, Serialize)]
pub struct ModelMetaOut {
    pub name: Option<String>,
    pub family: Option<String>,
    pub backend: String,
    pub file_type: String,

    pub quantization: Option<String>,
    pub context_length: Option<u32>,
    pub vocab_size: Option<u32>,
    pub eos_token_id: Option<i32>,
    pub bos_token_id: Option<i32>,

    /// "ChatMl" | "InstBlock" | "UserAssistant" | "Plain" | "Phi3"
    pub prompt_flavor_hint: Option<String>,
    pub has_chat_template: bool,

    /// Optional passthrough for advanced/debug views
    pub raw: Option<std::collections::HashMap<String, String>>,
}

/// Borrow-only mapping to avoid moving from `ModelCoreInfo`.
pub fn to_ui_meta(s: &ModelCoreInfo) -> ModelMetaOut {
    let prompt_hint_str = s.prompt_flavor_hint.as_ref().map(|pf| {
        match pf {
            PromptFlavor::ChatMl => "ChatMl",
            PromptFlavor::InstBlock => "InstBlock",
            PromptFlavor::UserAssistant => "UserAssistant",
            PromptFlavor::Plain => "Plain",
            PromptFlavor::Phi3 => "Phi3",
        }
        .to_string()
    });

    ModelMetaOut {
        name: s.name.clone(),
        family: s.family.clone(),
        backend: s.backend.clone(),
        file_type: s.file_type.clone(),

        quantization: s.quantization.clone(),
        context_length: s.context_length,
        vocab_size: s.vocab_size,
        eos_token_id: s.eos_token_id,
        bos_token_id: s.bos_token_id,

        prompt_flavor_hint: prompt_hint_str,
        has_chat_template: s
            .chat_template
            .as_ref()
            .map(|t| !t.is_empty())
            .unwrap_or(false),

        raw: if s.raw.is_empty() {
            None
        } else {
            Some(s.raw.clone())
        },
    }
}
