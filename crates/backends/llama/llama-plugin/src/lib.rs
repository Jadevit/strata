//! Llama plugin: C-ABI shim exposing metadata + LLM APIs at runtime.

// --- internal modules (merged from llama-rs) so `crate::*` paths resolve
pub mod adapter;
pub mod backends;
pub mod batch;
pub mod cache;
pub mod context;
pub mod debug;
pub mod ffi; // contains ffi::{context, metadata, ...}
pub mod format;
pub mod metadata; // safe scraper + provider (replaces old plugin_metadata)
pub mod model;
pub mod params;
pub mod sampling;
pub mod token;

use crate::adapter::LlamaBackendImpl;
use crate::metadata::LlamaMetadataProvider;

use core::ffi::{c_char, c_void};
use std::{
    ffi::{CStr, CString},
    path::Path,
    ptr, slice,
    sync::Once,
};

use serde_json;
use strata_abi::backend::LLMBackend;
use strata_abi::ffi::*;
use strata_abi::metadata::BackendMetadataProvider;
use strata_abi::sampling::SamplingParams;

// -----------------------------
// Error plumbing (thread-local)
// -----------------------------

thread_local! {
    static LAST_ERR: std::cell::RefCell<Option<CString>> = const { std::cell::RefCell::new(None) };
}

fn set_last_error(msg: impl AsRef<str>) -> i32 {
    let s = CString::new(msg.as_ref()).unwrap_or_else(|_| CString::new("invalid utf8").unwrap());
    LAST_ERR.with(|slot| *slot.borrow_mut() = Some(s));
    ERR_FAIL
}

unsafe extern "C" fn last_error() -> StrataString {
    let s = LAST_ERR.with(|slot| slot.borrow().clone());
    match s {
        Some(cs) => make_string(&cs),
        None => StrataString {
            ptr: ptr::null_mut(),
            len: 0,
        },
    }
}

// -----------------------------
// Helpers for FFI allocations
// -----------------------------

fn make_string(s: &CStr) -> StrataString {
    let bytes = s.to_bytes();
    let mut v = Vec::with_capacity(bytes.len() + 1);
    v.extend_from_slice(bytes);
    v.push(0);
    let ptr = v.as_mut_ptr() as *mut c_char;
    let len = bytes.len();
    std::mem::forget(v);
    StrataString { ptr, len }
}

fn make_string_from_utf8(s: &str) -> StrataString {
    let cs = CString::new(s).unwrap_or_else(|_| CString::new("").unwrap());
    make_string(&cs)
}

unsafe extern "C" fn free_string(s: StrataString) {
    if !s.ptr.is_null() {
        let _ = Vec::<u8>::from_raw_parts(s.ptr as *mut u8, s.len + 1, s.len + 1);
    }
}

unsafe extern "C" fn free_ints(arr: Int32Array) {
    if !arr.ptr.is_null() {
        let _ = Vec::<i32>::from_raw_parts(arr.ptr, arr.len, arr.len);
    }
}

// -----------------------------
// Metadata API wrappers
// -----------------------------

unsafe extern "C" fn meta_can_handle(model_path: *const c_char) -> bool {
    if model_path.is_null() {
        return false;
    }
    let c = CStr::from_ptr(model_path);
    let s = match c.to_str() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let prov = LlamaMetadataProvider;
    prov.can_handle(Path::new(s))
}

unsafe extern "C" fn meta_collect_json(model_path: *const c_char) -> StrataString {
    if model_path.is_null() {
        return StrataString {
            ptr: ptr::null_mut(),
            len: 0,
        };
    }
    let c = CStr::from_ptr(model_path);
    let s = match c.to_str() {
        Ok(v) => v,
        Err(e) => {
            set_last_error(format!("invalid UTF-8 in path: {e}"));
            return StrataString {
                ptr: ptr::null_mut(),
                len: 0,
            };
        }
    };
    let prov = LlamaMetadataProvider;
    match prov.collect(Path::new(s)) {
        Ok(info) => match serde_json::to_string(&info) {
            Ok(js) => make_string_from_utf8(&js),
            Err(e) => {
                set_last_error(format!("serde_json failed: {e}"));
                StrataString {
                    ptr: ptr::null_mut(),
                    len: 0,
                }
            }
        },
        Err(e) => {
            set_last_error(e);
            StrataString {
                ptr: ptr::null_mut(),
                len: 0,
            }
        }
    }
}

// -----------------------------
// LLM session handle
// -----------------------------

struct Session {
    inner: LlamaBackendImpl,
}

unsafe extern "C" fn llm_create_session(model_path: *const c_char) -> *mut c_void {
    if model_path.is_null() {
        set_last_error("null model path");
        return ptr::null_mut();
    }
    let c = CStr::from_ptr(model_path);
    let s = match c.to_str() {
        Ok(v) => v,
        Err(e) => {
            set_last_error(format!("invalid UTF-8 in path: {e}"));
            return ptr::null_mut();
        }
    };
    match <LlamaBackendImpl as LLMBackend>::load(Path::new(s)) {
        Ok(inner) => Box::into_raw(Box::new(Session { inner })) as *mut c_void,
        Err(e) => {
            set_last_error(e);
            ptr::null_mut()
        }
    }
}

unsafe extern "C" fn llm_destroy_session(session: *mut c_void) {
    if !session.is_null() {
        let _ = Box::<Session>::from_raw(session as *mut Session);
    }
}

unsafe extern "C" fn llm_tokenize_utf8(session: *mut c_void, text: *const c_char) -> Int32Array {
    if session.is_null() || text.is_null() {
        return Int32Array {
            ptr: ptr::null_mut(),
            len: 0,
        };
    }
    let sref = &mut *(session as *mut Session);
    let c = CStr::from_ptr(text);
    let txt = match c.to_str() {
        Ok(v) => v,
        Err(e) => {
            set_last_error(format!("invalid UTF-8 in text: {e}"));
            return Int32Array {
                ptr: ptr::null_mut(),
                len: 0,
            };
        }
    };
    match sref.inner.tokenize(txt) {
        Ok(tokens) => {
            let mut v: Vec<i32> = tokens.into_iter().map(|t| t.0).collect();
            let ptr = v.as_mut_ptr();
            let len = v.len();
            std::mem::forget(v);
            Int32Array { ptr, len }
        }
        Err(e) => {
            set_last_error(e);
            Int32Array {
                ptr: ptr::null_mut(),
                len: 0,
            }
        }
    }
}

unsafe extern "C" fn llm_format_chat_json(
    session: *mut ::core::ffi::c_void,
    turns_json: *const ::std::os::raw::c_char,
    _add_assistant: bool,
) -> strata_abi::ffi::StrataString {
    if session.is_null() || turns_json.is_null() {
        set_last_error("null session/turns_json");
        return strata_abi::ffi::StrataString {
            ptr: std::ptr::null_mut(),
            len: 0,
        };
    }

    let sref = &mut *(session as *mut Session);

    let js = match ::std::ffi::CStr::from_ptr(turns_json).to_str() {
        Ok(v) => v,
        Err(e) => {
            set_last_error(format!("invalid UTF-8 in turns_json: {e}"));
            return strata_abi::ffi::StrataString {
                ptr: std::ptr::null_mut(),
                len: 0,
            };
        }
    };

    let turns: Vec<strata_abi::backend::ChatTurn> = match ::serde_json::from_str(js) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(format!("bad ChatTurn JSON: {e}"));
            return strata_abi::ffi::StrataString {
                ptr: std::ptr::null_mut(),
                len: 0,
            };
        }
    };

    match sref.inner.apply_native_chat_template(&turns) {
        Some(text) => {
            // Compose the JSON shape expected by the host (FormattedPrompt)
            let stops: Vec<String> = sref
                .inner
                .default_stop_strings()
                .iter()
                .map(|s| s.to_string())
                .collect();

            let payload = match ::serde_json::to_string(&::serde_json::json!({
                "text": text,
                "stop_sequences": stops,
                "add_space_prefix": true
            })) {
                Ok(s) => s,
                Err(e) => {
                    set_last_error(format!("serde_json failed: {e}"));
                    return strata_abi::ffi::StrataString {
                        ptr: std::ptr::null_mut(),
                        len: 0,
                    };
                }
            };

            make_string_from_utf8(&payload)
        }
        None => {
            set_last_error("no native chat template available for this model/backend");
            strata_abi::ffi::StrataString {
                ptr: std::ptr::null_mut(),
                len: 0,
            }
        }
    }
}

unsafe extern "C" fn llm_detokenize_utf8(
    session: *mut c_void,
    tokens: *const i32,
    len: usize,
    remove_special: bool,
    unparse_special: bool,
) -> StrataString {
    if session.is_null() {
        return StrataString {
            ptr: ptr::null_mut(),
            len: 0,
        };
    }
    let s = &mut *(session as *mut Session);
    let ids = slice::from_raw_parts(tokens, len);
    let toks = ids
        .iter()
        .copied()
        .map(strata_abi::token::Token)
        .collect::<Vec<_>>();
    match s
        .inner
        .detokenize_range(&toks, 0, remove_special, unparse_special)
    {
        Ok(bytes) => match String::from_utf8(bytes) {
            Ok(text) => make_string_from_utf8(&text),
            Err(e) => {
                set_last_error(format!("detokenize returned non-UTF8: {e}"));
                StrataString {
                    ptr: ptr::null_mut(),
                    len: 0,
                }
            }
        },
        Err(e) => {
            set_last_error(e);
            StrataString {
                ptr: ptr::null_mut(),
                len: 0,
            }
        }
    }
}

unsafe extern "C" fn llm_evaluate(
    session: *mut c_void,
    tokens: *const i32,
    len: usize,
    n_past: i32,
) -> i32 {
    if session.is_null() || tokens.is_null() {
        return set_last_error("null session/tokens");
    }
    let sref = &mut *(session as *mut Session);
    let ids = slice::from_raw_parts(tokens, len);
    let toks = ids
        .iter()
        .copied()
        .map(strata_abi::token::Token)
        .collect::<Vec<_>>();
    match sref.inner.evaluate(&toks, n_past) {
        Ok(_) => ERR_OK,
        Err(e) => set_last_error(e),
    }
}

unsafe extern "C" fn llm_sample_json(session: *mut c_void, sampling_json: *const c_char) -> i32 {
    if session.is_null() || sampling_json.is_null() {
        return set_last_error("null session/sampling_json");
    }
    let sref = &mut *(session as *mut Session);
    let c = CStr::from_ptr(sampling_json);
    let json = match c.to_str() {
        Ok(v) => v,
        Err(e) => return set_last_error(format!("invalid UTF-8 in sampling_json: {e}")),
    };
    let params: SamplingParams = match serde_json::from_str::<SamplingParams>(json) {
        Ok(p) => p.normalized(),
        Err(e) => return set_last_error(format!("bad SamplingParams JSON: {e}")),
    };
    match sref.inner.sample(0, &params, &[]) {
        Ok(tok) => tok.0,
        Err(e) => set_last_error(e),
    }
}

unsafe extern "C" fn llm_decode_token(session: *mut c_void, token_id: i32) -> StrataString {
    if session.is_null() {
        return StrataString {
            ptr: ptr::null_mut(),
            len: 0,
        };
    }
    let sref = &*(session as *mut Session);
    match sref.inner.decode_token(strata_abi::token::Token(token_id)) {
        Ok(text) => make_string_from_utf8(&text),
        Err(e) => {
            set_last_error(e);
            StrataString {
                ptr: ptr::null_mut(),
                len: 0,
            }
        }
    }
}

unsafe extern "C" fn llm_clear_kv_cache(session: *mut c_void) {
    if session.is_null() {
        return;
    }
    let sref = &mut *(session as *mut Session);
    sref.inner.clear_kv_cache();
}

unsafe extern "C" fn llm_kv_len_hint(session: *mut c_void) -> i32 {
    if session.is_null() {
        return -1;
    }
    let sref = &*(session as *mut Session);
    match sref.inner.kv_len_hint() {
        Some(n) => n as i32,
        None => -1,
    }
}

unsafe extern "C" fn llm_context_window_hint(session: *mut c_void) -> i32 {
    if session.is_null() {
        return 0;
    }
    let sref = &*(session as *mut Session);
    match sref.inner.context_window_hint() {
        Some(n) => n as i32,
        None => 0,
    }
}

// -----------------------------
// Static PluginApi surface
// -----------------------------

static INIT: Once = Once::new();
static mut API: PluginApi = PluginApi {
    info: PluginInfo {
        abi_version: 0,
        id: std::ptr::null(),
        semver: std::ptr::null(),
    },
    metadata: MetadataApi {
        can_handle: meta_can_handle,
        collect_json: meta_collect_json,
        free_string: free_string,
    },
    llm: LlmApi {
        create_session: llm_create_session,
        destroy_session: llm_destroy_session,

        tokenize_utf8: llm_tokenize_utf8,
        free_ints: free_ints,

        evaluate: llm_evaluate,
        sample_json: llm_sample_json,
        decode_token: llm_decode_token,

        detokenize_utf8: llm_detokenize_utf8,
        format_chat_json: llm_format_chat_json,

        last_error: last_error,
        free_string: free_string,

        clear_kv_cache: llm_clear_kv_cache,
        kv_len_hint: llm_kv_len_hint,
        context_window_hint: llm_context_window_hint,
    },
};

#[no_mangle]
pub extern "C" fn strata_plugin_entry_v1() -> *const PluginApi {
    INIT.call_once(|| unsafe {
        let id = CString::new("llama").unwrap();
        let ver = CString::new("0.1.0").unwrap();
        API.info.abi_version = STRATA_ABI_VERSION;
        API.info.id = Box::leak(id.into_boxed_c_str()).as_ptr();
        API.info.semver = Box::leak(ver.into_boxed_c_str()).as_ptr();
    });
    unsafe { &API as *const PluginApi }
}
