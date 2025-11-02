use crate::errors::{Result, StoreError};
use crate::types::{Manifest, ManifestEntry};
use anyhow::Context;
use once_cell::sync::Lazy;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::Path;

/// Default remote for bundled builds to refresh against (you can override upstream).
pub static DEFAULT_MANIFEST_URL: Lazy<String> = Lazy::new(|| {
    "https://raw.githubusercontent.com/Jadevit/strata-runtimes/main/runtimes/latest/manifest.json"
        .to_string()
});

/// Try embedded JSON first (if you decide to embed it), else fetch remote.
/// For now, we only fetch remote.
pub fn load_embedded_or_remote(url: Option<&str>) -> Result<Manifest> {
    fetch_manifest(url.unwrap_or(&DEFAULT_MANIFEST_URL))
}

/// Fetch manifest JSON (blocking).
pub fn fetch_manifest(url: &str) -> Result<Manifest> {
    let txt = reqwest::blocking::get(url)?.text()?;
    let m: Manifest =
        serde_json::from_str(&txt).with_context(|| format!("invalid manifest JSON from {url}"))?;
    Ok(m)
}

/// Verify sha256 of a downloaded file matches the manifest.
pub fn verify_entry_sha256(entry: &ManifestEntry, zip_path: &Path) -> Result<()> {
    let got = sha256_file(zip_path)?;
    let want = entry.sha256.trim().to_lowercase();

    if got != want {
        return Err(StoreError::Msg(format!(
            "checksum mismatch for {} (got {}, want {})",
            entry.name, got, want
        )));
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut f = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}
