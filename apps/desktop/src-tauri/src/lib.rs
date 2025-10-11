#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    ffi::{CStr, CString},
    fs,
    path::{Path, PathBuf},
    process::Command,
    ptr, slice,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};

use core::ffi::c_void;
use libloading::Library;
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State, path::BaseDirectory};

mod metadata_indexer;
mod model;

use metadata_indexer::{MetaIndexStatus, MetaIndexer};
use model::{
    ModelEntry, get_current_model, get_model_path, import_into_user_library, list_available_models,
    resolve_models_root, set_current_model, user_models_root,
};

// -------- Strata core & ABI --------
use strata_abi::backend::{ChatTurn, LLMBackend, PromptFlavor};
use strata_abi::ffi::*;
use strata_abi::metadata::ModelCoreInfo;
use strata_core::engine::engine::LLMEngine;
use strata_core::format::prompt_format::PromptKind;
use strata_core::memory::SessionMemory;
use strata_core::metadata::metadata_service::{ModelMetaOut, to_ui_meta};

// ---------- Types ----------
type ModelMetaDTO = ModelMetaOut;

/// App state (thread-safe for background tasks).
struct AppState {
    memory: Arc<Mutex<SessionMemory>>,
    current_stop: Arc<Mutex<Option<Arc<AtomicBool>>>>,
}

// ============================================================================
// Runtime / sidecar helpers (unchanged)
// ============================================================================
fn installer_exe_name() -> &'static str {
    #[cfg(windows)]
    {
        "runtime-installer.exe"
    }
    #[cfg(not(windows))]
    {
        "runtime-installer"
    }
}

fn default_runtime_root() -> Option<PathBuf> {
    // match the installer's per-user layout
    dirs::data_dir().map(|p| p.join("Strata").join("runtimes").join("llama"))
}

// -----------------------------
// Runtime config reader helpers
// -----------------------------
use serde_json::Value as Json;

fn read_runtime_json(root: &std::path::Path) -> Option<Json> {
    let p = root.join("runtime.json");
    let bytes = std::fs::read(&p).ok()?;
    serde_json::from_slice::<Json>(&bytes).ok()
}

fn runtime_current_lib_dir(root: &std::path::Path) -> Option<std::path::PathBuf> {
    let j = read_runtime_json(root)?;
    j.get("llama")
        .and_then(|ll| ll.get("current_lib_dir"))
        .and_then(|v| v.as_str())
        .map(std::path::PathBuf::from)
}

fn runtime_is_monolith(root: &std::path::Path) -> bool {
    read_runtime_json(root)
        .and_then(|j| j.get("llama").cloned())
        .and_then(|ll| ll.get("monolith").cloned())
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

fn find_sidecar(app: &AppHandle) -> Option<PathBuf> {
    // 1) packaged app: bundled in Resources
    if let Some(p) = app
        .path()
        .resolve(installer_exe_name(), BaseDirectory::Resource)
        .ok()
    {
        if p.exists() {
            return Some(p);
        }
    }

    // 2) dev fallback: target/ (adjust if your local path differs)
    let dev = PathBuf::from("apps/desktop/src-tauri/sidecar/runtime-installer/target/release")
        .join(installer_exe_name());
    if dev.exists() {
        return Some(dev);
    }

    None
}

#[tauri::command]
fn is_llama_runtime_installed() -> bool {
    if let Some(root) = default_runtime_root() {
        root.join("runtime.json").exists()
    } else {
        false
    }
}

#[tauri::command]
fn run_runtime_installer(
    prefer: Option<String>,
    manifest: Option<String>,
    app: AppHandle,
) -> Result<(), String> {
    let exe = find_sidecar(&app).ok_or_else(|| {
        "runtime-installer not found in Resources or local target; bundle it as a sidecar"
            .to_string()
    })?;

    let install_dir =
        default_runtime_root().ok_or_else(|| "could not resolve user data dir".to_string())?;

    let mut cmd = Command::new(&exe);
    cmd.arg("--install-dir").arg(install_dir);

    if let Some(p) = prefer {
        cmd.arg("--prefer").arg(p);
    }
    if let Some(m) = manifest {
        cmd.arg("--manifest").arg(m);
    }

    let status = cmd
        .status()
        .map_err(|e| format!("failed to spawn installer: {e}"))?;

    if !status.success() {
        return Err(format!(
            "installer exited with code {:?}",
            status.code().unwrap_or(-1)
        ));
    }

    Ok(())
}

// ============================================================================
// 🔌 Plugin loader (C-ABI) → host-side adapter that implements LLMBackend
// ============================================================================

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

fn locate_runtime_llama_lib(plugin_path: &std::path::Path) -> Option<std::path::PathBuf> {
    use std::{env, path::PathBuf};

    // explicit override still wins
    if let Ok(p) = env::var("STRATA_LLAMA_LIB_PATH") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }

    // monolith packs: don't preload libllama.*
    if let Some(root) = env::var("STRATA_RUNTIME_DIR")
        .map(PathBuf::from)
        .ok()
        .or_else(|| default_runtime_root())
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
        // legacy fallbacks
        let flat = root.join(runtime_llama_filename());
        if flat.exists() {
            return Some(flat);
        }
        let nested = root.join("llama_backend").join(runtime_llama_filename());
        if nested.exists() {
            return Some(nested);
        }
    }

    // dev fallback next to plugin build (kept for convenience)
    if let Some(dir) = plugin_path.parent() {
        let cand = dir.join("resources/llama").join(runtime_llama_filename());
        if cand.exists() {
            return Some(cand);
        }
    }
    let dev =
        std::path::PathBuf::from("target/debug/resources/llama").join(runtime_llama_filename());
    if dev.exists() {
        return Some(dev);
    }

    None
}

fn locate_plugin_binary() -> Option<PathBuf> {
    use std::env;

    // 0) explicit override (best for dev)
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

    // 1) installer layout via runtime.json -> current_lib_dir
    if let Some(root) = default_runtime_root() {
        if let Some(cur) = runtime_current_lib_dir(&root) {
            let p = cur.join(plugin_filename());
            if p.exists() {
                eprintln!("[plugin] from runtime.json: {}", p.display());
                return Some(p);
            }
        }

        // 1b) fallback: try known variants if runtime.json missing/old
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

        // 1c) older layouts (just in case)
        let flat = root.join("llama_backend").join(plugin_filename());
        if flat.exists() {
            eprintln!("[plugin] found flat runtime layout: {}", flat.display());
            return Some(flat);
        }
        let alt = root.join("plugins").join(plugin_filename());
        if alt.exists() {
            eprintln!("[plugin] found runtime/plugins layout: {}", alt.display());
            return Some(alt);
        }
    }

    // 2) dev fallbacks
    let dbg = PathBuf::from("target/debug").join(plugin_filename());
    if dbg.exists() {
        eprintln!("[plugin] found debug build: {}", dbg.display());
        return Some(dbg);
    }
    let rel = PathBuf::from("target/release").join(plugin_filename());
    if rel.exists() {
        eprintln!("[plugin] found release build: {}", rel.display());
        return Some(rel);
    }

    None
}

struct LoadedPlugin {
    _preload_llama: Option<Library>, // keep libllama.* alive
    _lib: Library,                   // keep plugin alive
    api: &'static PluginApi,         // pointer into plugin
}

// SAFETY: `PluginApi` points at immutable function pointers & const strings
// exported by the dylib; we don’t mutate it, and we keep the `Library` alive.
unsafe impl Send for LoadedPlugin {}
unsafe impl Sync for LoadedPlugin {}

static PLUGIN: OnceLock<Result<LoadedPlugin, String>> = OnceLock::new();

fn load_plugin_once() -> Result<&'static LoadedPlugin, String> {
    PLUGIN
        .get_or_init(|| {
            let path = locate_plugin_binary().ok_or_else(|| {
                "llama plugin not found in any known location.\n\
                                 Hint: export STRATA_PLUGIN_PATH=<full path to libStrataLlama.*>"
                    .to_string()
            })?;

            // Try to preload libllama.* so the plugin can resolve symbols.
            let preload = match locate_runtime_llama_lib(&path) {
                Some(ll) => match unsafe { Library::new(&ll) } {
                    Ok(lib) => {
                        eprintln!("[plugin] preloaded {}", ll.display());
                        Some(lib)
                    }
                    Err(e) => {
                        eprintln!(
                            "[plugin] WARNING: failed to preload {}: {}",
                            ll.display(),
                            e
                        );
                        None
                    }
                },
                None => {
                    eprintln!("[plugin] WARNING: no runtime llama lib found to preload");
                    None
                }
            };

            // Load the plugin itself
            let lib = unsafe { Library::new(&path) }
                .map_err(|e| format!("failed to load plugin {}: {e}", path.display()))?;

            // Resolve entry symbol
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

// -----------------------------
// Helpers for plugin strings
// -----------------------------

fn make_cstring(s: &str) -> Result<CString, String> {
    CString::new(s).map_err(|_| "string contains interior NUL".to_string())
}

unsafe fn take_plugin_string(api_free: FreeStringFn, s: StrataString) -> String {
    if s.ptr.is_null() || s.len == 0 {
        return String::new();
    }
    let slice = slice::from_raw_parts(s.ptr as *const u8, s.len);
    // Force copy into host-owned string
    let out = String::from_utf8_lossy(slice).into_owned();
    api_free(s);
    out
}

// -----------------------------
// Host-side LLMBackend adapter
// -----------------------------

struct PluginBackend {
    plugin: &'static LoadedPlugin,
    session: *mut c_void,
    eos_token_id: i32,
    ctx_len_hint: Option<usize>,
}

impl Drop for PluginBackend {
    fn drop(&mut self) {
        if !self.session.is_null() {
            unsafe { (self.plugin.api.llm.destroy_session)(self.session) };
            self.session = ptr::null_mut();
        }
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
            // try to fetch last error
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

        // Pull metadata to grab EOS + context length hint
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
            // maybe empty text; try error fetch
            let msg = unsafe {
                let s = (self.plugin.api.llm.last_error)();
                take_plugin_string(self.plugin.api.llm.free_string, s)
            };
            if msg.is_empty() {
                return Ok(Vec::new());
            }
            return Err(msg);
        }
        // take ownership
        let mut v: Vec<i32> = unsafe { Vec::from_raw_parts(arr.ptr, arr.len, arr.len) };
        // plugin gave us ownership; do not call free_ints
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
            // may legitimately be empty; try last_error
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
        // `detokenize_utf8` guarantees UTF-8 text; we send bytes back to core.
        Ok(text.into_bytes())
    }
}

// ============================================================================
// Commands that stayed the same (model discovery / import)
// ============================================================================

#[tauri::command]
async fn get_model_list(app: AppHandle) -> Result<Vec<ModelEntry>, String> {
    let app2 = app.clone();
    tauri::async_runtime::spawn_blocking(move || list_available_models(app2))
        .await
        .map_err(|e| format!("join error: {e}"))?
}

#[tauri::command]
fn get_active_model() -> Option<String> {
    get_current_model()
}

#[tauri::command]
fn set_active_model_cmd(name: String) {
    set_current_model(name);
}

#[tauri::command]
fn get_models_root(app: AppHandle) -> Result<String, String> {
    resolve_models_root(&app).map(|p| p.display().to_string())
}

#[tauri::command]
async fn import_model(
    app: AppHandle,
    src_path: String,
    family: Option<String>,
) -> Result<ModelEntry, String> {
    let app2 = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let entry = import_into_user_library(&app2, Path::new(&src_path), family.as_deref())?;
        Ok::<_, String>(entry)
    })
    .await
    .map_err(|e| format!("join error: {e}"))?
}

// ============================================================================
// Metadata (single file) via plugin API
// ============================================================================

fn collect_model_metadata_via_plugin(path: &Path) -> Result<ModelCoreInfo, String> {
    let plugin = load_plugin_once()?;
    let cpath = make_cstring(path.to_str().ok_or("invalid UTF-8 in path")?)?;
    unsafe {
        let s = (plugin.api.metadata.collect_json)(cpath.as_ptr());
        let js = take_plugin_string(plugin.api.metadata.free_string, s);
        if js.is_empty() {
            let err = (plugin.api.llm.last_error)();
            let msg = take_plugin_string(plugin.api.llm.free_string, err);
            if msg.is_empty() {
                Err("plugin returned empty metadata".into())
            } else {
                Err(msg)
            }
        } else {
            serde_json::from_str::<ModelCoreInfo>(&js)
                .map_err(|e| format!("bad metadata JSON: {e}"))
        }
    }
}

#[tauri::command]
async fn get_model_metadata(app: AppHandle) -> Result<ModelMetaDTO, String> {
    let path = get_model_path(&app)?;
    let info =
        tauri::async_runtime::spawn_blocking(move || collect_model_metadata_via_plugin(&path))
            .await
            .map_err(|e| format!("join error: {e}"))??;
    Ok(to_ui_meta(&info))
}

// ---------- Metadata indexer commands ----------
#[tauri::command]
async fn meta_start_index(
    app: AppHandle,
    index: State<'_, MetaIndexer>,
    force: Option<bool>,
) -> Result<(), String> {
    index.start(app, force.unwrap_or(false))
}

#[tauri::command]
fn meta_status(index: State<'_, MetaIndexer>) -> MetaIndexStatus {
    index.status()
}

#[tauri::command]
fn meta_get_cached(id: String, index: State<'_, MetaIndexer>) -> Option<ModelMetaDTO> {
    index.get(&id)
}

#[tauri::command]
fn meta_clear(index: State<'_, MetaIndexer>) {
    index.clear();
}

// ============================================================================
// Prompt strategy (unchanged)
// ============================================================================

fn pick_prompt_strategy(model_id: Option<String>, system: Option<String>) -> PromptKind {
    if let Some(id) = model_id {
        let id_lc = id.to_lowercase();
        if id_lc.contains("phi") || id_lc.contains("vira") {
            return PromptKind::Phi3 { system };
        }
    }
    PromptKind::ChatMl { system }
}

// ============================================================================
// Inference (non-streaming / streaming) using PluginBackend
// ============================================================================

#[tauri::command]
async fn run_llm(
    prompt: String,
    _tts: bool,
    model_id: Option<String>,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<String, String> {
    if let Some(ref id) = model_id {
        set_current_model(id.clone());
    }
    {
        let mut mem = state.memory.lock().unwrap();
        mem.push_user(prompt.clone());
    }

    let app2 = app.clone();
    let state_mem = Arc::clone(&state.memory);
    let state_stop = Arc::clone(&state.current_stop);
    let model_id2 = model_id.clone();

    let reply = tauri::async_runtime::spawn_blocking(move || -> Result<String, String> {
        let model_path = get_model_path(&app2)?;
        let backend = PluginBackend::load(&model_path)?;
        let system = load_system_prompt_sync(&app2);

        let mut engine = LLMEngine::with_auto(backend, system.clone());
        engine.set_strategy(pick_prompt_strategy(
            model_id2.or_else(get_current_model),
            system,
        ));

        {
            let stop = engine.stop_handle();
            *state_stop.lock().unwrap() = Some(stop);
        }

        let turns: Vec<ChatTurn> = {
            let mem = state_mem.lock().unwrap();
            mem.turns().to_vec()
        };

        let out = engine.infer_chat(&turns)?;
        *state_stop.lock().unwrap() = None;
        Ok(out)
    })
    .await
    .map_err(|e| format!("join error: {e}"))??;

    {
        let mut mem = state.memory.lock().unwrap();
        mem.push_assistant(reply.clone());
    }
    Ok(reply)
}

#[tauri::command]
async fn run_llm_stream(
    prompt: String,
    _tts: bool,
    model_id: Option<String>,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<(), String> {
    if let Some(ref id) = model_id {
        set_current_model(id.clone());
    }
    {
        let mut mem = state.memory.lock().unwrap();
        mem.push_user(prompt.clone());
    }

    let app2 = app.clone();
    let state_mem = Arc::clone(&state.memory);
    let state_stop = Arc::clone(&state.current_stop);
    let model_id2 = model_id.clone();

    tauri::async_runtime::spawn_blocking(move || -> Result<String, String> {
        let model_path = get_model_path(&app2)?;
        let backend = PluginBackend::load(&model_path)?;
        let system = load_system_prompt_sync(&app2);

        let mut engine = LLMEngine::with_auto(backend, system.clone());
        engine.set_strategy(pick_prompt_strategy(
            model_id2.or_else(get_active_model),
            system,
        ));

        {
            let stop = engine.stop_handle();
            *state_stop.lock().unwrap() = Some(stop);
        }

        let turns: Vec<ChatTurn> = {
            let mem = state_mem.lock().unwrap();
            mem.turns().to_vec()
        };

        let final_text = engine.infer_chat_stream(&turns, |delta| {
            let _ = app2.emit("llm-stream", serde_json::json!({ "delta": delta }));
        })?;

        *state_stop.lock().unwrap() = None;
        {
            let mut mem = state_mem.lock().unwrap();
            mem.push_assistant(final_text.clone());
        }
        let _ = app2.emit("llm-complete", serde_json::json!({ "text": final_text }));
        Ok(final_text)
    })
    .await
    .map_err(|e| format!("join error: {e}"))??;

    Ok(())
}

// ============================================================================
// Cancel (unchanged)
// ============================================================================

#[tauri::command]
fn cancel_generation(state: State<'_, AppState>) -> Result<(), String> {
    if let Some(flag) = state.current_stop.lock().unwrap().as_ref() {
        flag.store(true, Ordering::Relaxed);
    }
    Ok(())
}

// ============================================================================
// System prompt (unchanged)
// ============================================================================

#[tauri::command]
async fn load_system_prompt(app: AppHandle) -> Result<String, String> {
    if let Some(path) = app
        .path()
        .resolve("system_prompt.txt", BaseDirectory::Resource)
        .ok()
    {
        if path.exists() {
            return fs::read_to_string(&path)
                .map_err(|e| format!("Failed to read system_prompt at {}: {e}", path.display()));
        }
    }
    let dev = PathBuf::from("resources/system_prompt.txt");
    if dev.exists() {
        return fs::read_to_string(&dev)
            .map_err(|e| format!("Dev fallback read failed at {}: {e}", dev.display()));
    }
    Err("system_prompt.txt not found in resources or ./resources/".into())
}

fn load_system_prompt_sync(app: &AppHandle) -> Option<String> {
    if let Some(path) = app
        .path()
        .resolve("system_prompt.txt", BaseDirectory::Resource)
        .ok()
    {
        if path.exists() {
            if let Ok(text) = fs::read_to_string(&path) {
                return Some(text);
            }
        }
    }
    let dev = PathBuf::from("resources/system_prompt.txt");
    if dev.exists() {
        if let Ok(text) = fs::read_to_string(&dev) {
            return Some(text);
        }
    }
    None
}

// ============================================================================
// App bootstrap
// ============================================================================

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState {
            memory: Arc::new(Mutex::new(SessionMemory::new())),
            current_stop: Arc::new(Mutex::new(None)),
        })
        .manage(MetaIndexer::new())
        // plugins
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        // NOTE: no compile-time registration of metadata providers anymore
        .setup(|_app| Ok(()))
        .invoke_handler(tauri::generate_handler![
            // system prompt
            load_system_prompt,
            // model list / selection
            get_model_list,
            get_active_model,
            set_active_model_cmd,
            get_models_root,
            // import
            import_model,
            // single-file metadata
            get_model_metadata,
            // metadata indexer controls
            meta_start_index,
            meta_status,
            meta_get_cached,
            meta_clear,
            // inference
            run_llm,
            run_llm_stream,
            cancel_generation,
            // installer
            is_llama_runtime_installed,
            run_runtime_installer,
        ])
        .run(tauri::generate_context!())
        .expect("Failed to launch Tauri app");
}
