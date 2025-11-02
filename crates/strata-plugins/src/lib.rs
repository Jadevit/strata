//! strata-plugins
//!
//! Blocking I/O helpers for Strata’s plugin store + runtime installer.
//! - Reads a signed/hashed manifest (remote or embedded).
//! - Chooses runtime variants (cpu + best GPU).
//! - Downloads, verifies, unzips, and writes `runtime.json`.
//!
//! Use from Tauri by `spawn_blocking` to keep UI snappy.

pub mod errors;
pub mod install;
pub mod manifest;
pub mod net;
pub mod paths;
pub mod state;
pub mod tauri_api;
pub mod types;

pub use errors::StoreError;
pub use install::{install_variants, write_runtime_config};
pub use manifest::{fetch_manifest, load_embedded_or_remote, verify_entry_sha256};
pub use net::download_to_path;
pub use paths::*;
pub use state::PluginsState;
pub use types::*;
