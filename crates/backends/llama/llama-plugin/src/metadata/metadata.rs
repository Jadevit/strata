// strata-backend-llama/src/metadata/metadata.rs
//
// Minimal, self-contained GGUF metadata scraper using the llama C API.
// This does NOT create a decode context, and (critically) opens the model in
// header-only mode (`vocab_only = true`) so we *don’t* mmap or load tensors.
// This keeps the UI snappy while we only need metadata.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::{Path, PathBuf};

use llama_sys::*;

#[derive(Debug, Clone)]
pub struct LlamaScrape {
    pub name: Option<String>,
    pub family: Option<String>,
    pub backend: String, // always "llama" for this crate
    pub path: PathBuf,
    pub file_type: String, // lowercased extension, e.g. "gguf"
    pub context_length: Option<u32>,
    pub vocab_size: Option<u32>,
    pub eos_token_id: Option<i32>,
    pub bos_token_id: Option<i32>,
    pub quantization: Option<String>, // e.g. "Q8_0" if we can infer
    pub chat_template: Option<String>,
    pub raw: HashMap<String, String>, // flattened kvs (stringified)
}

pub fn can_handle(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false)
}

/// Open the GGUF in header-only mode, scrape metadata, and close the model.
pub fn scrape_metadata(path: &Path) -> Result<LlamaScrape, String> {
    let path_str = path.to_str().ok_or_else(|| "non-UTF8 path".to_string())?;
    let c_path = CString::new(path_str).map_err(|_| "invalid model path".to_string())?;

    unsafe {
        // 1) model params: start from default and force header-only read
        let mut params = llama_model_default_params();
        params.vocab_only = true; // header-only; *no tensors* — keeps UI responsive
        params.use_mmap = true; // be OS-friendly
        params.n_gpu_layers = 0; // avoid GPU touches

        // 2) open
        let model = llama_load_model_from_file(c_path.as_ptr(), params);
        if model.is_null() {
            return Err(format!(
                "llama_load_model_from_file failed for {}",
                path.display()
            ));
        }

        // 3) gather metadata KVs as strings
        let mut raw: HashMap<String, String> = HashMap::new();
        let n = llama_model_meta_count(model);
        let mut key_buf = vec![0i8; 512];
        let mut val_buf = vec![0i8; 4096];

        for i in 0..n {
            // key
            let kn = llama_model_meta_key_by_index(model, i, key_buf.as_mut_ptr(), key_buf.len());
            if kn <= 0 {
                continue;
            }
            let key = c_chars_to_string(&key_buf);

            // value (stringified)
            let vn =
                llama_model_meta_val_str_by_index(model, i, val_buf.as_mut_ptr(), val_buf.len());
            if vn <= 0 {
                continue;
            }
            let val = c_chars_to_string(&val_buf);

            raw.insert(key, val);
        }

        // 4) normalized selections
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

        let quantization = raw.get("general.quantization").cloned().or_else(|| {
            ft_label_from_code(pick_u32(&raw, &["general.file_type"]).unwrap_or_default())
        });

        // Native chat template (null → None)
        let chat_template = {
            let tpl_ptr = llama_model_chat_template(model, std::ptr::null::<c_char>());
            if tpl_ptr.is_null() {
                None
            } else {
                Some(CStr::from_ptr(tpl_ptr).to_string_lossy().into_owned())
            }
        };

        // 5) build result
        let out = LlamaScrape {
            name,
            family,
            backend: "llama".to_string(),
            path: path.to_path_buf(),
            file_type: path
                .extension()
                .and_then(|e| e.to_str())
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or_else(|| "gguf".into()),
            context_length,
            vocab_size,
            eos_token_id,
            bos_token_id,
            quantization,
            chat_template,
            raw,
        };

        // 6) close
        llama_free_model(model);
        Ok(out)
    }
}

// ---------- helpers ----------

fn c_chars_to_string(buf: &[i8]) -> String {
    let len = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    let slice = &buf[..len];
    let u8s: Vec<u8> = slice.iter().map(|&c| c as u8).collect();
    String::from_utf8_lossy(&u8s).to_string()
}

fn pick_u32(map: &HashMap<String, String>, keys: &[&str]) -> Option<u32> {
    for k in keys {
        if let Some(v) = map.get(*k) {
            if let Ok(n) = v.trim().parse::<u32>() {
                return Some(n);
            }
        }
    }
    None
}

fn pick_i32(map: &HashMap<String, String>, keys: &[&str]) -> Option<i32> {
    for k in keys {
        if let Some(v) = map.get(*k) {
            if let Ok(n) = v.trim().parse::<i32>() {
                return Some(n);
            }
        }
    }
    None
}

// best-effort mapping for common GGUF ftype codes -> labels
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
