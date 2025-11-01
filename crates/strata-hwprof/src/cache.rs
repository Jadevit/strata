use crate::detect::detect_now;
use crate::paths::{cache_hwprof_dir, hwprof_profile_path};
use crate::types::HardwareProfile;
use anyhow::{Context, Result};
use chrono::Utc;
use serde_json as json;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;

pub fn load_cached() -> Option<HardwareProfile> {
    let path = hwprof_profile_path();
    let mut buf = Vec::new();
    let mut f = fs::File::open(&path).ok()?;
    f.read_to_end(&mut buf).ok()?;
    json::from_slice::<HardwareProfile>(&buf).ok()
}

pub fn save_profile(p: &HardwareProfile) -> Result<()> {
    let dir = cache_hwprof_dir();
    fs::create_dir_all(&dir).with_context(|| format!("mkd {}", dir.display()))?;
    let tmp = dir.join("profile.json.tmp");
    fs::write(&tmp, json::to_vec_pretty(p)?)?;
    fs::rename(&tmp, hwprof_profile_path())?;
    Ok(())
}

/// Always returns a profile; if cache exists, returns it without re-detecting.
pub fn load_or_detect() -> Result<HardwareProfile> {
    if let Some(p) = load_cached() {
        return Ok(p);
    }
    detect_and_cache()
}

/// Runs detection now and caches the result, returning the fresh profile.
pub fn detect_and_cache() -> Result<HardwareProfile> {
    let mut p = detect_now()?;
    p.schema = 1;
    p.schema_minor = 1; // NEW: minor bump for additive fields
    p.fingerprint = compute_fingerprint(&p)?;
    let now = Utc::now().to_rfc3339();
    p.created_at = now.clone();
    p.updated_at = now;
    save_profile(&p)?;
    Ok(p)
}

// Utility so detect_now() can compute hash the same way if needed later.
pub fn compute_fingerprint(p: &HardwareProfile) -> Result<String> {
    let mut hasher = Sha256::new();

    // Stable identity bits
    hasher.update(p.os.as_bytes());
    hasher.update(p.arch.as_bytes());
    hasher.update(p.cpu.brand.as_bytes());
    hasher.update(&p.cpu.threads.to_le_bytes());
    hasher.update(&[
        // backends truthiness matters
        p.backends.cpu as u8,
        p.backends.cuda as u8,
        p.backends.rocm as u8,
        p.backends.vulkan as u8,
        p.backends.metal as u8,
    ]);

    // GPU identity (include software flags and drivers/VRAM if present)
    for g in &p.gpus {
        hasher.update(&g.vendor_id.to_le_bytes());
        hasher.update(&g.device_id.to_le_bytes());
        hasher.update(g.vendor.as_bytes());
        hasher.update(g.name.as_bytes());
        hasher.update(&[g.integrated as u8, g.software_renderer as u8]);
        if let Some(vram) = g.vram_bytes {
            hasher.update(&vram.to_le_bytes());
        }
        if let Some(d) = &g.driver {
            if let Some(c) = &d.cuda {
                hasher.update(c.as_bytes());
            }
            if let Some(n) = &d.nvml {
                hasher.update(n.as_bytes());
            }
            if let Some(vk) = &d.vulkan {
                hasher.update(vk.as_bytes());
            }
            if let Some(rc) = &d.rocm {
                hasher.update(rc.as_bytes());
            }
            if let Some(m) = &d.metal {
                hasher.update(m.as_bytes());
            }
        }
    }

    Ok(hex::encode(hasher.finalize()))
}

/// Optional helper: re-detect only if fingerprint changed.
pub fn validate_or_redetect() -> Result<HardwareProfile> {
    let cached = load_cached();
    let fresh = detect_now()?;
    let mut stable = fresh.clone();
    stable.schema = 1;
    stable.schema_minor = 1;
    stable.fingerprint = compute_fingerprint(&stable)?;
    let need_update = match &cached {
        Some(c) => c.fingerprint != stable.fingerprint,
        None => true,
    };
    if need_update {
        let mut p = stable;
        let now = chrono::Utc::now().to_rfc3339();
        p.created_at = cached.map(|c| c.created_at).unwrap_or_else(|| now.clone());
        p.updated_at = now;
        save_profile(&p)?;
        Ok(p)
    } else {
        Ok(cached.unwrap())
    }
}
