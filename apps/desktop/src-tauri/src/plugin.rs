use core::ffi::c_void;
use std::{
    env,
    path::{Path, PathBuf},
    ptr, slice,
    sync::OnceLock,
};

use crate::runtime::{default_runtime_root, runtime_current_lib_dir, runtime_is_monolith};
use libloading::Library;
use strata_abi::{
    backend::{ChatTurn, LLMBackend, PromptFlavor},
    ffi::*,
    metadata::ModelCoreInfo,
};

// ---------------------------------------------------------------------------
// Plugin + runtime filename helpers
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
fn plugin_filename() -> &'static str {
    "StrataLlama.dll"
}
#[cfg(target_os = "macos")]
fn plugin_filename() -> &'static str {
    "StrataLlama.dylib"
}
#[cfg(all(unix, not(target_os = "macos")))]
fn plugin_filename() -> &'static str {
    "StrataLlama.so"
}

#[cfg(target_os = "windows")]
fn runtime_llama_filename() -> &'static str {
    "llama.dll"
}
#[cfg(target_os = "macos")]
fn runtime_llama_filename() -> &'static str {
    "libllama.dylib"
}
#[cfg(all(unix, not(target_os = "macos")))]
fn runtime_llama_filename() -> &'static str {
    "libllama.so"
}

// ---------------------------------------------------------------------------
// Locate runtime + plugin binaries
// ---------------------------------------------------------------------------

fn locate_runtime_llama_lib(plugin_path: &Path) -> Option<PathBuf> {
    if let Ok(p) = env::var("STRATA_LLAMA_LIB_PATH") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }

    if let Some(root) = env::var("STRATA_RUNTIME_DIR")
        .map(PathBuf::from)
        .ok()
        .or_else(default_runtime_root)
    {
        if runtime_is_monolith(&root) {
            return None;
        }

        if let Some(dir) = runtime_current_lib_dir(&root) {
            let p = dir.join(runtime_llama_filename());
            if p.exists() {
                return Some(p);
            }
        }

        for candidate in [
            root.join(runtime_llama_filename()),
            root.join("llama_backend").join(runtime_llama_filename()),
        ] {
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    if let Some(dir) = plugin_path.parent() {
        let cand = dir.join("resources/llama").join(runtime_llama_filename());
        if cand.exists() {
            return Some(cand);
        }
    }

    let dev = PathBuf::from("target/debug/resources/llama").join(runtime_llama_filename());
    dev.exists().then_some(dev)
}

fn locate_plugin_binary() -> Option<PathBuf> {
    if let Ok(p) = env::var("STRATA_PLUGIN_PATH") {
        let p = PathBuf::from(p);
        if p.exists() {
            eprintln!("[plugin] STRATA_PLUGIN_PATH = {}", p.display());
            return Some(p);
        } else {
            eprintln!(
                "[plugin] STRATA_PLUGIN_PATH points to missing file: {}",
                p.display()
            );
        }
    }

    if let Some(root) = default_runtime_root() {
        if let Some(cur) = runtime_current_lib_dir(&root) {
            let p = cur.join(plugin_filename());
            if p.exists() {
                eprintln!("[plugin] from runtime.json: {}", p.display());
                return Some(p);
            }
        }

        for variant in ["cuda", "vulkan", "metal", "cpu"] {
            let p = root
                .join(variant)
                .join("llama_backend")
                .join(plugin_filename());
            if p.exists() {
                eprintln!("[plugin] found in {variant} pack: {}", p.display());
                return Some(p);
            }
        }

        for p in [
            root.join("llama_backend").join(plugin_filename()),
            root.join("plugins").join(plugin_filename()),
        ] {
            if p.exists() {
                return Some(p);
            }
        }
    }

    for p in [
        PathBuf::from("target/debug").join(plugin_filename()),
        PathBuf::from("target/release").join(plugin_filename()),
    ] {
        if p.exists() {
            return Some(p);
        }
    }

    None
}

// ---------------------------------------------------------------------------
// LoadedPlugin: global plugin handle
// ---------------------------------------------------------------------------

pub(crate) struct LoadedPlugin {
    _preload_llama: Option<Library>,
    _lib: Library,
    pub(crate) api: &'static PluginApi,
}

// SAFETY: the API is immutable and the library is pinned in memory.
unsafe impl Send for LoadedPlugin {}
unsafe impl Sync for LoadedPlugin {}

static PLUGIN: OnceLock<Result<LoadedPlugin, String>> = OnceLock::new();

fn make_cstring(s: &str) -> Result<std::ffi::CString, String> {
    std::ffi::CString::new(s).map_err(|_| "string contains interior NUL".to_string())
}

unsafe fn take_plugin_string(api_free: FreeStringFn, s: StrataString) -> String {
    if s.ptr.is_null() || s.len == 0 {
        return String::new();
    }
    let slice = slice::from_raw_parts(s.ptr as *const u8, s.len);
    let out = String::from_utf8_lossy(slice).into_owned();
    api_free(s);
    out
}

pub fn load_plugin_once() -> Result<&'static LoadedPlugin, String> {
    PLUGIN
        .get_or_init(|| {
            let path = locate_plugin_binary().ok_or_else(|| {
                "llama plugin not found in any known location. \
                 Hint: export STRATA_PLUGIN_PATH=<full path to libStrataLlama.*>"
                    .to_string()
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

            Ok(LoadedPlugin {
                _preload_llama: preload,
                _lib: lib,
                api,
            })
        })
        .as_ref()
        .map_err(|e| e.clone())
}

// ---------------------------------------------------------------------------
// PluginBackend: host-side adapter implementing LLMBackend
// ---------------------------------------------------------------------------

pub struct PluginBackend {
    plugin: &'static LoadedPlugin,
    session: *mut c_void,
    eos_token_id: i32,
    ctx_len_hint: Option<usize>,
}

impl Drop for PluginBackend {
    fn drop(&mut self) {
        if !self.session.is_null() {
            unsafe { (self.plugin.api.llm.destroy_session)(self.session) };
            self.session = std::ptr::null_mut();
        }
    }
}

impl Clone for PluginBackend {
    fn clone(&self) -> Self {
        // Shallow clone — we only ever use one generation at a time.
        Self {
            plugin: self.plugin,
            session: self.session,
            eos_token_id: self.eos_token_id,
            ctx_len_hint: self.ctx_len_hint,
        }
    }
}

// SAFETY: PluginBackend’s raw session pointer comes from a C-ABI plugin.
// It is never accessed concurrently — every call goes through a Mutex.
unsafe impl Send for PluginBackend {}
unsafe impl Sync for PluginBackend {}

impl PluginBackend {
    /// Public constructor that forwards to the LLMBackend trait.
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self, String> {
        <Self as LLMBackend>::load(model_path)
    }
}

impl LLMBackend for PluginBackend {
    fn load<P: AsRef<Path>>(model_path: P) -> Result<Self, String> {
        let plugin = load_plugin_once()?;
        let cpath = make_cstring(
            model_path
                .as_ref()
                .to_str()
                .ok_or("model path not valid UTF-8")?,
        )?;

        let session = unsafe { (plugin.api.llm.create_session)(cpath.as_ptr()) };
        if session.is_null() {
            let msg = unsafe {
                let s = (plugin.api.llm.last_error)();
                take_plugin_string(plugin.api.llm.free_string, s)
            };
            return Err(if msg.is_empty() {
                "create_session failed".into()
            } else {
                msg
            });
        }

        // Pull metadata to get EOS + context length hint
        let meta_json = unsafe {
            let s = (plugin.api.metadata.collect_json)(cpath.as_ptr());
            take_plugin_string(plugin.api.metadata.free_string, s)
        };

        let (eos, ctx_hint) = if meta_json.is_empty() {
            (-1, None)
        } else {
            match serde_json::from_str::<ModelCoreInfo>(&meta_json) {
                Ok(m) => (
                    m.eos_token_id.unwrap_or(-1),
                    m.context_length.map(|c| c as usize),
                ),
                Err(_) => (-1, None),
            }
        };

        Ok(Self {
            plugin,
            session,
            eos_token_id: eos,
            ctx_len_hint: ctx_hint,
        })
    }

    fn tokenize(&self, text: &str) -> Result<Vec<strata_abi::token::Token>, String> {
        let ctext = make_cstring(text)?;
        let arr = unsafe { (self.plugin.api.llm.tokenize_utf8)(self.session, ctext.as_ptr()) };

        if arr.ptr.is_null() || arr.len == 0 {
            let msg = unsafe {
                let s = (self.plugin.api.llm.last_error)();
                take_plugin_string(self.plugin.api.llm.free_string, s)
            };
            return if msg.is_empty() {
                Ok(Vec::new())
            } else {
                Err(msg)
            };
        }

        let mut v: Vec<i32> = unsafe { Vec::from_raw_parts(arr.ptr, arr.len, arr.len) };
        Ok(v.drain(..).map(strata_abi::token::Token).collect())
    }

    fn evaluate(&mut self, tokens: &[strata_abi::token::Token], n_past: i32) -> Result<(), String> {
        let tmp: Vec<i32> = tokens.iter().map(|t| t.0).collect();
        let rc = unsafe {
            (self.plugin.api.llm.evaluate)(self.session, tmp.as_ptr(), tmp.len(), n_past)
        };
        if rc == ERR_OK {
            Ok(())
        } else {
            let msg = unsafe {
                let s = (self.plugin.api.llm.last_error)();
                take_plugin_string(self.plugin.api.llm.free_string, s)
            };
            Err(if msg.is_empty() {
                "evaluate failed".into()
            } else {
                msg
            })
        }
    }

    fn sample(
        &mut self,
        _n_past: i32,
        params: &strata_abi::sampling::SamplingParams,
        _token_history: &[strata_abi::token::Token],
    ) -> Result<strata_abi::token::Token, String> {
        let params = params.normalized();
        let js = serde_json::to_string(&params).map_err(|e| e.to_string())?;
        let cjs = make_cstring(&js)?;
        let tok = unsafe { (self.plugin.api.llm.sample_json)(self.session, cjs.as_ptr()) };
        if tok >= 0 {
            Ok(strata_abi::token::Token(tok))
        } else {
            let msg = unsafe {
                let s = (self.plugin.api.llm.last_error)();
                take_plugin_string(self.plugin.api.llm.free_string, s)
            };
            Err(if msg.is_empty() {
                "sample failed".into()
            } else {
                msg
            })
        }
    }

    fn decode_token(&self, token: strata_abi::token::Token) -> Result<String, String> {
        let s = unsafe { (self.plugin.api.llm.decode_token)(self.session, token.0) };
        let out = unsafe { take_plugin_string(self.plugin.api.llm.free_string, s) };

        if out.is_empty() {
            let msg = unsafe {
                let se = (self.plugin.api.llm.last_error)();
                take_plugin_string(self.plugin.api.llm.free_string, se)
            };
            if msg.is_empty() { Ok(out) } else { Err(msg) }
        } else {
            Ok(out)
        }
    }

    fn eos_token(&self) -> strata_abi::token::Token {
        strata_abi::token::Token(self.eos_token_id)
    }

    fn context_window_hint(&self) -> Option<usize> {
        self.ctx_len_hint
    }

    fn prompt_flavor(&self) -> PromptFlavor {
        PromptFlavor::ChatMl
    }

    fn default_stop_strings(&self) -> &'static [&'static str] {
        &["<|im_end|>"]
    }

    fn apply_native_chat_template(&self, _turns: &[ChatTurn]) -> Option<String> {
        None
    }

    fn detokenize_range(
        &self,
        token_history: &[strata_abi::token::Token],
        start: usize,
        remove_special: bool,
        unparse_special: bool,
    ) -> Result<Vec<u8>, String> {
        let slice = &token_history[start..];
        if slice.is_empty() {
            return Ok(Vec::new());
        }
        let tmp: Vec<i32> = slice.iter().map(|t| t.0).collect();
        let s = unsafe {
            (self.plugin.api.llm.detokenize_utf8)(
                self.session,
                tmp.as_ptr(),
                tmp.len(),
                remove_special,
                unparse_special,
            )
        };
        let text = unsafe { take_plugin_string(self.plugin.api.llm.free_string, s) };
        Ok(text.into_bytes())
    }
}
