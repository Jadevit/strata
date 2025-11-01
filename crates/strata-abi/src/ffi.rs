use core::ffi::{c_char, c_void};

/// Bump this when you break the ABI. Host checks it at load time.
pub const STRATA_ABI_VERSION: u32 = 4; // was 3

pub const PLUGIN_ENTRY_SYMBOL: &str = "strata_plugin_entry_v1";

pub const ERR_OK: i32 = 0;
pub const ERR_FAIL: i32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct StrataString {
    pub ptr: *mut c_char,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Int32Array {
    pub ptr: *mut i32,
    pub len: usize,
}

#[repr(C)]
pub struct PluginInfo {
    pub abi_version: u32,
    pub id: *const c_char,     // "llama"
    pub semver: *const c_char, // "0.1.0"
}

// ---------- Function pointer types (C ABI) ----------

pub type CanHandleFn = unsafe extern "C" fn(model_path: *const c_char) -> bool;
pub type CollectJsonFn = unsafe extern "C" fn(model_path: *const c_char) -> StrataString;
pub type FreeStringFn = unsafe extern "C" fn(s: StrataString);

pub type CreateSessionFn = unsafe extern "C" fn(model_path: *const c_char) -> *mut c_void;
pub type DestroySessionFn = unsafe extern "C" fn(session: *mut c_void);

pub type TokenizeUtf8Fn =
    unsafe extern "C" fn(session: *mut c_void, text: *const c_char) -> Int32Array;
pub type FreeIntsFn = unsafe extern "C" fn(arr: Int32Array);

pub type EvaluateFn =
    unsafe extern "C" fn(session: *mut c_void, tokens: *const i32, len: usize, n_past: i32) -> i32;

/// `sampling_json` is UTF-8 JSON of `strata_abi::sampling::SamplingParams::normalized()`.
/// Returns next token id (>= 0) or a negative error code (ERR_FAIL et al).
pub type SampleJsonFn =
    unsafe extern "C" fn(session: *mut c_void, sampling_json: *const c_char) -> i32;

/// Returns **JSON encoding of `FormattedPrompt`** (fields: text, stop_sequences, add_space_prefix).
pub type FormatChatJsonFn = unsafe extern "C" fn(
    session: *mut c_void,
    turns_json: *const c_char,
    add_assistant: bool,
) -> StrataString;

pub type DecodeTokenFn = unsafe extern "C" fn(session: *mut c_void, token_id: i32) -> StrataString;
pub type DetokenizeUtf8Fn = unsafe extern "C" fn(
    session: *mut c_void,
    tokens: *const i32,
    len: usize,
    remove_special: bool,
    unparse_special: bool,
) -> StrataString;

pub type LastErrorFn = unsafe extern "C" fn() -> StrataString;

// small helpers the host/engine already uses conceptually
pub type ClearKvFn = unsafe extern "C" fn(session: *mut c_void);
pub type KvLenHintFn = unsafe extern "C" fn(session: *mut c_void) -> i32; // -1 if unknown
pub type ContextWindowHintFn = unsafe extern "C" fn(session: *mut c_void) -> i32; // 0 if unknown

// ---------- VTables ----------

#[repr(C)]
pub struct MetadataApi {
    pub can_handle: CanHandleFn,
    /// JSON for `strata_abi::metadata::ModelCoreInfo`.
    pub collect_json: CollectJsonFn,
    pub free_string: FreeStringFn,
}

#[repr(C)]
pub struct LlmApi {
    pub create_session: CreateSessionFn,
    pub destroy_session: DestroySessionFn,

    pub tokenize_utf8: TokenizeUtf8Fn,
    pub free_ints: FreeIntsFn,

    pub evaluate: EvaluateFn,
    pub sample_json: SampleJsonFn,
    pub decode_token: DecodeTokenFn,

    pub detokenize_utf8: DetokenizeUtf8Fn,
    pub format_chat_json: FormatChatJsonFn,

    // Diagnostics & memory management
    pub last_error: LastErrorFn,
    pub free_string: FreeStringFn,

    // KV context hooks
    pub clear_kv_cache: ClearKvFn,
    pub kv_len_hint: KvLenHintFn,
    pub context_window_hint: ContextWindowHintFn,
}

#[repr(C)]
pub struct PluginApi {
    pub info: PluginInfo,
    pub metadata: MetadataApi,
    pub llm: LlmApi,
}

/// Plugin must export `strata_plugin_entry_v1` returning a pointer to a static `PluginApi`.
pub type PluginEntryFn = unsafe extern "C" fn() -> *const PluginApi;
