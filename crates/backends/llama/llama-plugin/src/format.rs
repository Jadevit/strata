// llama-plugin/src/format.rs

use std::ffi::CStr;

use strata_abi::backend::{ChatTurn, Role};

use crate::ffi::{apply_chat_template, ChatMsgFFI};
use crate::model::LlamaModel;

/// Convert Strata turns → llama_chat_message[] and apply the model’s (or explicit) template.
/// Returns Some(prompt) if rendered, else None (caller decides what to do).
pub fn format_with_native_template(
    model: &crate::model::LlamaModel,
    turns: &[strata_abi::backend::ChatTurn],
    override_template: Option<&std::ffi::CStr>,
    add_assistant_turn: bool,
) -> Option<String> {
    // Map roles → llama_chat_message[]
    let mut msgs: Vec<crate::ffi::ChatMsgFFI> = Vec::with_capacity(turns.len());
    for t in turns {
        let role = match t.role {
            strata_abi::backend::Role::System => "system",
            strata_abi::backend::Role::User => "user",
            strata_abi::backend::Role::Assistant => "assistant",
        };
        let msg = crate::ffi::ChatMsgFFI::new(role, t.content.as_str()).ok()?;
        msgs.push(msg);
    }

    // Choose the template: explicit override OR the model’s default
    let tmpl = match override_template {
        Some(c) => c,
        None => crate::ffi::model_default_chat_template(model.as_ptr())?,
    };

    // Render via llama.cpp
    match crate::ffi::apply_chat_template(tmpl, &msgs, add_assistant_turn) {
        Ok(s) if !s.is_empty() => Some(s),
        Ok(_) => Some(String::new()),
        Err(_) => None,
    }
}
