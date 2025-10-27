// crates/backends/llama/llama-plugin/src/ffi.rs
//
// Process-wide llama backend init/shutdown + model load/unload,
// and chat-template helpers. Keep context-specific bits in ffi::context.

use std::{ffi::CString, ptr::NonNull, sync::OnceLock};

use llama_sys::{
    llama_backend_free, llama_backend_init, llama_chat_apply_template, llama_chat_message,
    llama_context, llama_free, llama_free_model, llama_load_model_from_file, llama_model,
    llama_model_chat_template, llama_model_default_params, llama_model_params,
};

/// one-time flags to prevent double init/deinit
static INIT_CALLED: OnceLock<()> = OnceLock::new();
static DEINIT_CALLED: OnceLock<()> = OnceLock::new();

#[inline]
fn trace(msg: &str) {
    #[cfg(feature = "ffi-trace")]
    println!("{}", msg);
}

/// Call exactly once near process start.
pub unsafe fn init_backend() {
    if INIT_CALLED.set(()).is_ok() {
        trace("🧠 [FFI] llama_backend_init()");
        llama_backend_init();
    } else {
        trace("↩️ [FFI] init_backend() called again — ignored");
    }
}

/// Optional: call once on clean shutdown.
pub unsafe fn deinit_backend() {
    if DEINIT_CALLED.set(()).is_ok() {
        trace("🧹 [FFI] llama_backend_free()");
        llama_backend_free();
    } else {
        trace("↩️ [FFI] deinit_backend() called again — ignored");
    }
}

/// Default model params (start from upstream defaults to stay future-proof).
pub fn default_model_params() -> llama_model_params {
    let mut p = unsafe { llama_model_default_params() };

    // Conservative, portable defaults (tweak upstream as needed):
    // Keep mmap for fast load; verify tensor shapes.
    p.use_mmap = true;
    p.check_tensors = true;

    p
}

/// Load a model from disk. Caller owns the returned handle.
pub unsafe fn load_model(path: &str) -> Result<NonNull<llama_model>, String> {
    trace(&format!("📦 [FFI] load_model: {path}"));
    let c_path = CString::new(path).map_err(|_| "Invalid model path".to_string())?;
    let ptr = llama_load_model_from_file(c_path.as_ptr(), default_model_params());
    NonNull::new(ptr).ok_or_else(|| "llama_load_model_from_file returned null".into())
}

/// Free a model instance.
pub unsafe fn cleanup_model(model: *mut llama_model) {
    trace("🧹 [FFI] llama_free_model()");
    llama_free_model(model);
}

/// Free a context instance.
pub unsafe fn cleanup_context(ctx: *mut llama_context) {
    trace("🧹 [FFI] llama_free(context)");
    llama_free(ctx);
}

// -------------------------
// Chat template helpers
// -------------------------

/// FFI-friendly holder to keep CStrings alive while passing to C.
pub struct ChatMsgFFI {
    role: CString,
    content: CString,
    c_msg: llama_chat_message,
}

impl ChatMsgFFI {
    pub fn new(role: &str, content: &str) -> Result<Self, String> {
        let role = CString::new(role).map_err(|_| "role has interior NUL".to_string())?;
        let content = CString::new(content).map_err(|_| "content has interior NUL".to_string())?;
        let c_msg = llama_chat_message {
            role: role.as_ptr(),
            content: content.as_ptr(),
        };
        Ok(Self {
            role,
            content,
            c_msg,
        })
    }

    #[inline]
    pub fn as_c(&self) -> llama_chat_message {
        self.c_msg
    }
}

/// Returns the model’s default chat template as a borrowed C string, or None if missing.
pub fn model_default_chat_template(
    model: *mut llama_sys::llama_model,
) -> Option<&'static std::ffi::CStr> {
    let ptr = unsafe { llama_model_chat_template(model, std::ptr::null()) };
    println!("[DEBUG] llama_model_chat_template ptr = {ptr:p}");
    if ptr.is_null() {
        println!("[DEBUG] → Returned NULL (no template pointer)");
        return None;
    }
    unsafe {
        let cstr = std::ffi::CStr::from_ptr(ptr);
        let preview = cstr.to_string_lossy();
        let truncated = if preview.len() > 200 {
            format!("{}...", &preview[..200])
        } else {
            preview.to_string()
        };
        println!(
            "[DEBUG] → Chat template preview (len={}): {}",
            preview.len(),
            truncated
        );
        Some(cstr)
    }
}

/// Apply a concrete template pointer to a slice of ChatMsgFFI and return the formatted prompt.
pub fn apply_chat_template(
    tmpl: &std::ffi::CStr,
    msgs: &[ChatMsgFFI],
    add_assistant: bool,
) -> Result<String, String> {
    // Build a contiguous C array of llama_chat_message
    let mut c_msgs: Vec<llama_chat_message> = Vec::with_capacity(msgs.len());
    for m in msgs {
        let cm = m.as_c();
        c_msgs.push(llama_chat_message {
            role: cm.role,
            content: cm.content,
        });
    }

    // Start with a reasonable buffer, grow if needed just like upstream example.
    let mut cap: i32 = 4096;
    loop {
        let mut buf = vec![0i8; cap as usize];

        let wrote = unsafe {
            llama_chat_apply_template(
                tmpl.as_ptr(),
                c_msgs.as_ptr(),
                c_msgs.len(),
                add_assistant,
                buf.as_mut_ptr(),
                cap,
            )
        };

        if wrote < 0 {
            return Err(format!("llama_chat_apply_template failed ({wrote})"));
        }
        if wrote as usize > buf.len() {
            cap = cap.saturating_mul(2);
            continue;
        }

        let len = wrote as usize;
        if len == 0 {
            return Ok(String::new());
        }

        let s = unsafe {
            let ptr = buf.as_ptr() as *const u8;
            let slice = std::slice::from_raw_parts(ptr, len);
            String::from_utf8(slice.to_vec()).map_err(|e| format!("non-UTF8 from template: {e}"))?
        };
        return Ok(s);
    }
}
