use std::sync::OnceLock;

use libloading::Library;
use strata_abi::ffi::{PluginApi, PluginEntryFn, PLUGIN_ENTRY_SYMBOL, STRATA_ABI_VERSION};

use super::locate::{locate_plugin_binary, locate_runtime_llama_lib};

pub(crate) struct LoadedPlugin {
    #[allow(dead_code)]
    _preload_llama: Option<Library>,
    #[allow(dead_code)]
    _lib: Library,
    pub(crate) api: &'static PluginApi,
}

// SAFETY: the API is immutable and the library is pinned in memory.
unsafe impl Send for LoadedPlugin {}
unsafe impl Sync for LoadedPlugin {}

static PLUGIN: OnceLock<Result<LoadedPlugin, String>> = OnceLock::new();

pub fn load_plugin_once() -> Result<&'static LoadedPlugin, String> {
    PLUGIN
        .get_or_init(|| {
            let path = locate_plugin_binary().ok_or_else(|| {
                "llama plugin not found in any known location. \
                 Hint: export STRATA_PLUGIN_PATH=<full path to libStrataLlama.*>".to_string()
            })?;

            let preload = locate_runtime_llama_lib(&path)
                .and_then(|ll| unsafe { Library::new(&ll).ok() })
                .map(|lib| {
                    eprintln!("[plugin] preloaded runtime lib");
                    lib
                });

            let lib = unsafe { Library::new(&path) }
                .map_err(|e| format!("failed to load plugin {}: {e}", path.display()))?;

            let entry: libloading::Symbol<PluginEntryFn> = unsafe {
                lib.get(PLUGIN_ENTRY_SYMBOL.as_bytes())
                    .map_err(|e| format!("missing symbol {}: {e}", PLUGIN_ENTRY_SYMBOL))?
            };

            let api_ptr = unsafe { entry() };
            if api_ptr.is_null() {
                return Err("plugin entry returned null".into());
            }

            let api = unsafe { &*api_ptr };
            if api.info.abi_version != STRATA_ABI_VERSION {
                return Err(format!(
                    "ABI mismatch: host={} plugin={}",
                    STRATA_ABI_VERSION, api.info.abi_version
                ));
            }

            Ok(LoadedPlugin { _preload_llama: preload, _lib: lib, api })
        })
        .as_ref()
        .map_err(|e| e.clone())
}
