use core::ffi::{c_char, c_void};

/// Bump this when you break the ABI. Host checks it at load time.
pub const STRATA_ABI_VERSION: u32 = 2;

/// The single entry symbol every plugin must export.
pub const PLUGIN_ENTRY_SYMBOL: &str = "strata_plugin_entry_v1";

/// Simple error codes for calls that need them.
pub const ERR_OK: i32 = 0;
pub const ERR_FAIL: i32 = 1;

/// UTF-8 string buffer allocated by the plugin and freed via the plugin's `free_string`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct StrataString {
    pub ptr: *mut c_char,
    pub len: usize,
}

/// A heap array of i32 allocated by the plugin and freed via the plugin's `free_ints`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Int32Array {
    pub ptr: *mut i32,
    pub len: usize,
}

/// Minimal identity block reported by the plugin at load time.
#[repr(C)]
pub struct PluginInfo {
    /// Must equal STRATA_ABI_VERSION.
    pub abi_version: u32,
    /// Null-terminated id string, e.g. "llama".
    pub id: *const c_char,
    /// Null-terminated semantic version, e.g. "0.1.0".
    pub semver: *const c_char,
}

/// ---------- Function pointer types (C ABI) ----------

pub type CanHandleFn = unsafe extern "C" fn(model_path: *const c_char) -> bool;
pub type CollectJsonFn = unsafe extern "C" fn(model_path: *const c_char) -> StrataString;
pub type FreeStringFn = unsafe extern "C" fn(s: StrataString);

pub type CreateSessionFn = unsafe extern "C" fn(model_path: *const c_char) -> *mut c_void;
pub type DestroySessionFn = unsafe extern "C" fn(session: *mut c_void);

pub type TokenizeUtf8Fn =
    unsafe extern "C" fn(session: *mut c_void, text: *const c_char) -> Int32Array;
pub type FreeIntsFn = unsafe extern "C" fn(arr: Int32Array);

/// tokens: pointer to i32 tokens; len: number of tokens; n_past: past context size
pub type EvaluateFn =
    unsafe extern "C" fn(session: *mut c_void, tokens: *const i32, len: usize, n_past: i32) -> i32;

/// `sampling_json` is a UTF-8 JSON matching `strata_abi::sampling::SamplingParams`.
/// Returns the next token id or a negative error code.
pub type SampleJsonFn =
    unsafe extern "C" fn(session: *mut c_void, sampling_json: *const c_char) -> i32;

/// Returns UTF-8 fragment for token id. Caller must free via `free_string`.
pub type DecodeTokenFn = unsafe extern "C" fn(session: *mut c_void, token_id: i32) -> StrataString;

/// Detokenize a sequence of token ids to valid UTF-8 text.
pub type DetokenizeUtf8Fn = unsafe extern "C" fn(
    session: *mut c_void,
    tokens: *const i32,
    len: usize,
    remove_special: bool,
    unparse_special: bool,
) -> StrataString;

/// Optional: a thread-local last error message for debugging (UTF-8).
pub type LastErrorFn = unsafe extern "C" fn() -> StrataString;

/// ---------- VTables exported by plugin ----------

#[repr(C)]
pub struct MetadataApi {
    pub can_handle: CanHandleFn,
    /// Returns JSON string for `strata_abi::metadata::ModelCoreInfo`.
    pub collect_json: CollectJsonFn,
    /// Free for any `StrataString` returned by metadata calls.
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

    /// Detokenize to valid UTF-8 text (string).
    pub detokenize_utf8: DetokenizeUtf8Fn,

    /// Optional diagnostic message (may be empty).
    pub last_error: LastErrorFn,
    /// Free for any `StrataString` returned by LLM calls.
    pub free_string: FreeStringFn,
}

#[repr(C)]
pub struct PluginApi {
    pub info: PluginInfo,
    pub metadata: MetadataApi,
    pub llm: LlmApi,
}

/// The plugin exposes a single `extern "C"` function named `strata_plugin_entry_v1`
/// returning a pointer to a static `PluginApi`.
///
/// Plugin side (example signature):
///   #[no_mangle]
///   pub extern "C" fn strata_plugin_entry_v1() -> *const PluginApi { ... }
pub type PluginEntryFn = unsafe extern "C" fn() -> *const PluginApi;
