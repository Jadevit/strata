use anyhow::{Context, Result, anyhow};
use dirs::data_dir;
use indicatif::{ProgressBar, ProgressStyle};
use libloading::Library;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Debug, Deserialize)]
struct ManifestEntry {
    name: String,    // zip asset name
    sha256: String,  // lowercase hex
    os: String,      // windows-latest | ubuntu-22.04 | macos-14
    arch: String,    // x64 | arm64
    variant: String, // cpu | cuda | vulkan
    url: String,     // direct download URL for asset
}

#[derive(Debug, Deserialize)]
struct Manifest {
    llama: Vec<ManifestEntry>,
}

#[derive(Debug)]
enum GpuPref {
    Cpu,
    Cuda,
    Vulkan,
}

fn current_os() -> &'static str {
    if cfg!(windows) {
        "windows-latest"
    } else if cfg!(target_os = "macos") {
        "macos-14"
    } else {
        "ubuntu-22.04"
    }
}

fn current_arch() -> &'static str {
    if cfg!(target_arch = "x86_64") {
        "x64"
    } else if cfg!(target_arch = "aarch64") {
        "arm64"
    } else {
        "x64"
    } // fallback
}

// super light-weight detection:
// - CUDA: can we load nvcuda (win) / libcuda.so.1 (linux)?
// - Vulkan: can we load vulkan-1 (win) / libvulkan.so.1 (linux)?
fn detect_gpu_pref() -> GpuPref {
    #[cfg(target_os = "windows")]
    {
        if Library::new("nvcuda.dll").is_ok() {
            return GpuPref::Cuda;
        }
        if Library::new("vulkan-1.dll").is_ok() {
            return GpuPref::Vulkan;
        }
        return GpuPref::Cpu;
    }
    #[cfg(target_os = "linux")]
    {
        if Library::new("libcuda.so.1").is_ok() {
            return GpuPref::Cuda;
        }
        if Library::new("libvulkan.so.1").is_ok() {
            return GpuPref::Vulkan;
        }
        return GpuPref::Cpu;
    }
    #[cfg(target_os = "macos")]
    {
        // Metal is system-provided. Ship the default build.
        GpuPref::Cpu
    }
}

fn pick_entry<'a>(manifest: &'a Manifest, pref: &GpuPref) -> Option<&'a ManifestEntry> {
    let os = current_os();
    let arch = current_arch();

    let want = match pref {
        GpuPref::Cuda => "cuda",
        GpuPref::Vulkan => "vulkan",
        GpuPref::Cpu => "cpu",
    };

    // strict match first
    if let Some(e) = manifest
        .llama
        .iter()
        .find(|e| e.os == os && e.arch == arch && e.variant == want)
    {
        return Some(e);
    }
    // fallback order: cuda -> vulkan -> cpu
    for v in ["cuda", "vulkan", "cpu"] {
        if let Some(e) = manifest
            .llama
            .iter()
            .find(|e| e.os == os && e.arch == arch && e.variant == v)
        {
            return Some(e);
        }
    }
    None
}

fn download_with_progress(url: &str, dest: &Path) -> Result<()> {
    let resp = reqwest::blocking::get(url).with_context(|| format!("GET {}", url))?;
    if !resp.status().is_success() {
        return Err(anyhow!("download failed: {}", resp.status()));
    }
    let total = resp.content_length().unwrap_or(0);
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template("[{bar:40}] {bytes}/{total_bytes} ({eta})")?
            .progress_chars("=>-"),
    );

    let mut file = fs::File::create(dest)?;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().transpose().map_err(|e| anyhow!(e))? {
        std::io::copy(&mut chunk.as_ref(), &mut file)?;
        pb.inc(chunk.len() as u64);
    }
    pb.finish_and_clear();
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut f = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn unzip_into(zip_path: &Path, dest: &Path) -> Result<()> {
    let f = fs::File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(f)?;
    fs::create_dir_all(dest)?;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let outpath = dest.join(file.mangled_name());
        if file.name().ends_with('/') {
            fs::create_dir_all(&outpath)?;
        } else {
            if let Some(parent) = outpath.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut outfile = fs::File::create(&outpath)?;
            std::io::copy(&mut file, &mut outfile)?;
        }
    }
    Ok(())
}

fn install_dir() -> Result<PathBuf> {
    // Per-machine shared directory
    #[cfg(target_os = "windows")]
    {
        let base =
            PathBuf::from(env::var("PROGRAMDATA").unwrap_or_else(|_| r"C:\ProgramData".into()));
        return Ok(base.join("Strata").join("runtimes").join("llama"));
    }
    #[cfg(target_os = "macos")]
    {
        Ok(PathBuf::from(
            "/Library/Application Support/Strata/runtimes/llama",
        ))
    }
    #[cfg(target_os = "linux")]
    {
        Ok(PathBuf::from("/opt/strata/runtimes/llama"))
    }
}

fn write_active_config(dir: &Path, variant: &str) -> Result<()> {
    let cfg = serde_json::json!({
        "llama": {
            "active": if variant == "cpu" { "cpu" } else { "gpu" },
            "gpu_backend": if variant == "cpu" { serde_json::Value::Null } else { variant },
            "installed": [variant]
        }
    });
    fs::create_dir_all(dir)?;
    fs::write(dir.join("runtime.json"), serde_json::to_vec_pretty(&cfg)?)?;
    Ok(())
}

fn main() -> Result<()> {
    let manifest_url = env::var("STRATA_RUNTIME_MANIFEST")
        .unwrap_or_else(|_| "https://example.com/manifest.json".into()); // you’ll replace this later

    let want = match env::var("STRATA_FORCE_VARIANT").ok().as_deref() {
        Some("cuda") => GpuPref::Cuda,
        Some("vulkan") => GpuPref::Vulkan,
        _ => detect_gpu_pref(),
    };

    let mtxt = reqwest::blocking::get(&manifest_url)?.text()?;
    let manifest: Manifest = serde_json::from_str(&mtxt)?;

    let entry = pick_entry(&manifest, &want)
        .ok_or_else(|| anyhow!("no runtime pack found for this OS/arch"))?;
    let tmp_dir = env::temp_dir().join("strata-runtime");
    fs::create_dir_all(&tmp_dir)?;

    let zip_path = tmp_dir.join(&entry.name);
    println!("Downloading {}", entry.url);
    download_with_progress(&entry.url, &zip_path)?;

    let sum = sha256_file(&zip_path)?;
    if sum.to_lowercase() != entry.sha256.to_lowercase() {
        return Err(anyhow!("checksum mismatch for {}", entry.name));
    }

    let dest = install_dir()?;
    unzip_into(&zip_path, &dest)?;
    write_active_config(&dest, &entry.variant)?;
    println!("Installed {} → {}", entry.variant, dest.display());
    Ok(())
}
