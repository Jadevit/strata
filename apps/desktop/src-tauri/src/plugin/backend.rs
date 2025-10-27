use core::ffi::c_void;
use std::{path::Path, slice};

use crate::plugin::loader::load_plugin_once;
use strata_abi::{
    backend::{ChatTurn, LLMBackend, PromptFlavor},
    ffi::*,
    metadata::ModelCoreInfo,
};

pub struct PluginBackend {
    pub(crate) plugin: &'static super::loader::LoadedPlugin,
    pub(crate) session: *mut c_void,
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

// SAFETY: Raw session pointer is only used behind external locking (LLMEngine).
unsafe impl Send for PluginBackend {}
unsafe impl Sync for PluginBackend {}

fn make_cstring(s: &str) -> Result<std::ffi::CString, String> {
    std::ffi::CString::new(s).map_err(|_| "string contains interior NUL".to_string())
}

unsafe fn take_plugin_string(api_free: FreeStringFn, s: StrataString) -> String {
    if s.ptr.is_null() || s.len == 0 {
        return String::new();
    }
    // Rust 2024 lint: body is safe by default; wrap unsafe ops.
    let out = {
        let slice = unsafe { slice::from_raw_parts(s.ptr as *const u8, s.len) };
        String::from_utf8_lossy(slice).into_owned()
    };
    unsafe { api_free(s) };
    out
}

impl PluginBackend {
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

    fn apply_native_chat_template(&self, turns: &[ChatTurn]) -> Option<String> {
        if turns.is_empty() {
            return Some(String::new());
        }

        // Send ChatTurn[] to the plugin
        let js = match serde_json::to_string(turns) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[plugin] serialize ChatTurn failed: {e}");
                return None;
            }
        };
        let cjs = match std::ffi::CString::new(js) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[plugin] CString::new(turns_json) failed: {e}");
                return None;
            }
        };

        // Ask plugin to apply its native chat template → returns JSON for FormattedPrompt
        let s = unsafe { (self.plugin.api.llm.format_chat_json)(self.session, cjs.as_ptr(), true) };
        if s.ptr.is_null() || s.len == 0 {
            // Pull plugin error (if any) for logging
            let msg = unsafe {
                let se = (self.plugin.api.llm.last_error)();
                take_plugin_string(self.plugin.api.llm.free_string, se)
            };
            if !msg.is_empty() {
                eprintln!("[plugin] format_chat_json failed: {msg}");
            }
            return None;
        }

        let payload = unsafe { take_plugin_string(self.plugin.api.llm.free_string, s) };

        // The plugin returns a JSON object like:
        // { "text": "...", "stop_sequences": ["..."], "add_space_prefix": true }
        #[derive(serde::Deserialize)]
        struct FormattedPrompt {
            text: String,
            #[allow(dead_code)]
            stop_sequences: Option<Vec<String>>,
            #[allow(dead_code)]
            add_space_prefix: Option<bool>,
        }

        match serde_json::from_str::<FormattedPrompt>(&payload) {
            Ok(fp) => Some(fp.text),
            Err(e) => {
                eprintln!("[plugin] malformed FormattedPrompt JSON: {e}; raw={payload}");
                None
            }
        }
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
