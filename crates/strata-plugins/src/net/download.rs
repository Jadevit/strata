use crate::errors::{Result, StoreError};
use anyhow::Context;
use reqwest::blocking::Client;
use std::fs;
use std::io::{Read, Write};
use std::path::Path;
use std::time::Duration;

/// Blocking HTTPS download with rustls. Caller handles spawn_blocking.
pub fn download_to_path(url: &str, dest: &Path) -> Result<()> {
    let client = Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;

    let mut resp = client
        .get(url)
        .send()
        .with_context(|| format!("GET {url}"))?;
    if !resp.status().is_success() {
        return Err(StoreError::Msg(format!(
            "download failed: {}",
            resp.status()
        )));
    }

    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut out = fs::File::create(dest)?;
    let mut buf = [0u8; 128 * 1024];

    loop {
        let n = resp.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
    }

    Ok(())
}
