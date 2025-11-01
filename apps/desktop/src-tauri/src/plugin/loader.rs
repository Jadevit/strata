use super::locate::locate_plugin_binary;
use crate::runtime::{default_runtime_root, runtime_cpu_fallback_path};
use libloading::Library;
use std::sync::OnceLock;
use strata_abi::ffi::{PLUGIN_ENTRY_SYMBOL, PluginApi, PluginEntryFn, STRATA_ABI_VERSION};

pub(crate) struct LoadedPlugin {
    #[allow(dead_code)]
    _lib: Library,
    pub(crate) api: &'static PluginApi,
}

unsafe impl Send for LoadedPlugin {}
unsafe impl Sync for LoadedPlugin {}

static PLUGIN: OnceLock<Result<LoadedPlugin, String>> = OnceLock::new();

pub fn load_plugin_once() -> Result<&'static LoadedPlugin, String> {
    PLUGIN
        .get_or_init(|| {
            if let Some(primary) = locate_plugin_binary() {
                match unsafe { Library::new(&primary) } {
                    Ok(lib) => return init_loaded(lib),
                    Err(e) => eprintln!(
                        "[plugin] failed to load active plugin {}: {e}",
                        primary.display()
                    ),
                }
            }

            if let Some(root) = default_runtime_root() {
                if let Some(cpu_path) = runtime_cpu_fallback_path(&root) {
                    if cpu_path.exists() {
                        eprintln!("[plugin] attempting CPU fallback: {}", cpu_path.display());
                        match unsafe { Library::new(&cpu_path) } {
                            Ok(lib) => return init_loaded(lib),
                            Err(e) => eprintln!(
                                "[plugin] CPU fallback load failed {}: {e}",
                                cpu_path.display()
                            ),
                        }
                    }
                }
            }

            Err("plugin not found or failed to load; try installing/repairing the runtime".into())
        })
        .as_ref()
        .map_err(|e| e.clone())
}

fn init_loaded(lib: Library) -> Result<LoadedPlugin, String> {
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

    Ok(LoadedPlugin { _lib: lib, api })
}
