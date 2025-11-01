// llama-plugin/src/metadata/scrape.rs
//
// Pure safe code that calls into ffi::metadata to scrape GGUF headers.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::ffi::{metadata as fmeta, model as mffi};

/// Normalized llama metadata result (backend-local).
#[derive(Debug, Clone)]
pub struct LlamaScrape {
    pub name: Option<String>,
    pub family: Option<String>,
    pub backend: String, // always "llama" for this crate
    pub path: PathBuf,
    pub file_type: String, // e.g., "gguf"
    pub context_length: Option<u32>,
    pub vocab_size: Option<u32>,
    pub eos_token_id: Option<i32>,
    pub bos_token_id: Option<i32>,
    pub quantization: Option<String>,
    pub chat_template: Option<String>,
    pub raw: HashMap<String, String>,
}

pub fn can_handle(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false)
}

pub fn scrape_metadata(path: &Path) -> Result<LlamaScrape, String> {
    if !can_handle(path) {
        return Err("unsupported model file (expecting .gguf)".into());
    }

    let model = unsafe { fmeta::open_header_only(path)? };
    let raw = unsafe { fmeta::read_all_meta(model) };

    // `chat_template` now resides in ffi::model
    let chat_template = unsafe { mffi::chat_template(model.as_ptr()) };

    unsafe { fmeta::close_model(model) };

    let name = raw
        .get("general.name")
        .cloned()
        .or_else(|| raw.get("name").cloned());

    let family = raw
        .get("general.architecture")
        .cloned()
        .or_else(|| raw.get("general.basename").cloned());

    let context_length = pick_u32(
        &raw,
        &[
            "llama.context_length",
            "mistral.context_length",
            "qwen.context_length",
            "qwen2.context_length",
            "qwen3.context_length",
            "phi3.context_length",
            "context_length",
        ],
    );

    let vocab_size = pick_u32(
        &raw,
        &[
            "llama.vocab_size",
            "tokenizer.ggml.vocab_size",
            "vocab_size",
        ],
    );

    let eos_token_id = pick_i32(&raw, &["tokenizer.ggml.eos_token_id", "eos_token_id"]);
    let bos_token_id = pick_i32(&raw, &["tokenizer.ggml.bos_token_id", "bos_token_id"]);

    let quantization = raw
        .get("general.quantization")
        .cloned()
        .or_else(|| ft_label_from_code(pick_u32(&raw, &["general.file_type"]).unwrap_or_default()));

    let file_type = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_else(|| "gguf".into());

    Ok(LlamaScrape {
        name,
        family,
        backend: "llama".into(),
        path: path.to_path_buf(),
        file_type,
        context_length,
        vocab_size,
        eos_token_id,
        bos_token_id,
        quantization,
        chat_template,
        raw,
    })
}

// -------- helpers (pure safe) --------

fn parse_u32_loose(s: &str) -> Option<u32> {
    let t = s.trim().trim_matches('"').trim();
    t.parse::<u32>().ok()
}
fn parse_i32_loose(s: &str) -> Option<i32> {
    let t = s.trim().trim_matches('"').trim();
    t.parse::<i32>().ok()
}

fn pick_u32(map: &HashMap<String, String>, keys: &[&str]) -> Option<u32> {
    for k in keys {
        if let Some(v) = map.get(*k) {
            if let Some(n) = parse_u32_loose(v) {
                return Some(n);
            }
        }
    }
    None
}

fn pick_i32(map: &HashMap<String, String>, keys: &[&str]) -> Option<i32> {
    for k in keys {
        if let Some(v) = map.get(*k) {
            if let Some(n) = parse_i32_loose(v) {
                return Some(n);
            }
        }
    }
    None
}

// best-effort mapping for GGUF ftype codes -> labels
fn ft_label_from_code(code: u32) -> Option<String> {
    let label = match code {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_1",
        7 => "Q8_0",
        8 => "Q2_K",
        9 => "Q3_K_S",
        10 => "Q3_K_M",
        11 => "Q3_K_L",
        12 => "Q4_K_S",
        13 => "Q4_K_M",
        14 => "Q5_K_S",
        15 => "Q5_K_M",
        16 => "Q6_K",
        _ => return None,
    };
    Some(label.to_string())
}
