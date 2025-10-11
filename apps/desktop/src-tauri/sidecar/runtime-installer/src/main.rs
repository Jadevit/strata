use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

#[cfg(any(target_os = "windows", target_os = "linux"))]
use nvml_wrapper::Nvml;

#[cfg(any(target_os = "windows", target_os = "linux"))]
use ash::{vk, Entry};

#[cfg(target_os = "macos")]
use metal::Device as MetalDevice;

/// Where we fetch the manifest by default
const DEFAULT_MANIFEST: &str =
    "https://raw.githubusercontent.com/Jadevit/strata-runtimes/main/runtimes/latest/manifest.json";

/// Core llama library name per OS
#[cfg(target_os = "windows")]
const PLUGIN_NAME: &str = "StrataLlama.dll";
#[cfg(target_os = "linux")]
const PLUGIN_NAME: &str = "libStrataLlama.so";
#[cfg(target_os = "macos")]
const PLUGIN_NAME: &str = "libStrataLlama.dylib";

#[derive(Debug, Clone, ValueEnum)]
enum Pref {
    Auto,
    Cpu,
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    Cuda,
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    Vulkan,
    #[cfg(target_os = "macos")]
    Metal,
}

#[derive(Parser, Debug)]
#[command(version, about = "Strata runtime installer")]
struct Args {
    /// Manifest URL (leave empty to use latest)
    #[arg(long)]
    manifest: Option<String>,

    /// Prefer a specific variant (auto picks best)
    #[arg(long, value_enum, default_value_t = Pref::Auto)]
    prefer: Pref,

    /// Optional explicit install dir (otherwise per-user)
    #[arg(long)]
    install_dir: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct ManifestEntry {
    name: String,    // zip file name
    sha256: String,  // lowercase hex
    os: String,      // windows-latest | ubuntu-22.04 | macos-14
    arch: String,    // x64 | arm64
    variant: String, // cpu | cuda | vulkan | metal
    url: String,     // direct download URL
}

#[derive(Debug, Deserialize)]
struct Manifest {
    llama: Vec<ManifestEntry>,
}

fn current_os_key() -> &'static str {
    if cfg!(target_os = "windows") {
        "windows-latest"
    } else if cfg!(target_os = "macos") {
        "macos-14"
    } else {
        "ubuntu-22.04"
    }
}

fn current_arch_key() -> &'static str {
    if cfg!(target_arch = "x86_64") {
        "x64"
    } else if cfg!(target_arch = "aarch64") {
        "arm64"
    } else {
        "x64"
    }
}

/// Vendor IDs from PCI-SIG
const PCI_VENDOR_NVIDIA: u32 = 0x10DE;
const PCI_VENDOR_AMD: u32 = 0x1002;

/// CUDA check (NVML): true if NVML loads and at least 1 device
#[cfg(any(target_os = "windows", target_os = "linux"))]
fn has_cuda_device() -> bool {
    match Nvml::init() {
        Ok(nvml) => match nvml.device_count() {
            Ok(count) if count > 0 => {
                eprintln!("[detect] NVML: {} CUDA device(s) found", count);
                true
            }
            _ => {
                eprintln!("[detect] NVML: no CUDA devices");
                false
            }
        },
        Err(e) => {
            eprintln!("[detect] NVML load failed: {e:?}");
            false
        }
    }
}

/// Vulkan check (ash): true if AMD GPU present
#[cfg(any(target_os = "windows", target_os = "linux"))]
fn has_amd_vulkan_device() -> bool {
    use std::ffi::CString;

    let entry = match unsafe { Entry::load() } {
        Ok(e) => e,
        Err(e) => {
            eprintln!("[detect] Vulkan loader not available: {e:?}");
            return false;
        }
    };

    let app_name = CString::new("strata-installer").unwrap();
    let engine_name = CString::new("strata-engine").unwrap();

    let application_info = vk::ApplicationInfo {
        s_type: vk::StructureType::APPLICATION_INFO,
        p_next: std::ptr::null(),
        p_application_name: app_name.as_ptr(),
        application_version: 0,
        p_engine_name: engine_name.as_ptr(),
        engine_version: 0,
        api_version: vk::API_VERSION_1_0,
        ..Default::default()
    };

    let create_info = vk::InstanceCreateInfo {
        s_type: vk::StructureType::INSTANCE_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: vk::InstanceCreateFlags::empty(),
        p_application_info: &application_info,
        enabled_layer_count: 0,
        pp_enabled_layer_names: std::ptr::null(),
        enabled_extension_count: 0,
        pp_enabled_extension_names: std::ptr::null(),
        ..Default::default()
    };

    let instance = match unsafe { entry.create_instance(&create_info, None) } {
        Ok(i) => i,
        Err(e) => {
            eprintln!("[detect] vkCreateInstance failed: {e:?}");
            return false;
        }
    };

    let devices = unsafe { instance.enumerate_physical_devices() };
    let result = match devices {
        Ok(list) => {
            let amd_count = list
                .iter()
                .filter(|&&pd| {
                    let props = unsafe { instance.get_physical_device_properties(pd) };
                    props.vendor_id == PCI_VENDOR_AMD
                })
                .count();
            if amd_count > 0 {
                eprintln!("[detect] Vulkan: {} AMD device(s) found", amd_count);
                true
            } else {
                eprintln!("[detect] Vulkan: devices found, but none are AMD");
                false
            }
        }
        Err(e) => {
            eprintln!("[detect] vkEnumeratePhysicalDevices failed: {e:?}");
            false
        }
    };

    unsafe { instance.destroy_instance(None) };
    result
}

/// Metal check: system_default must return Some(device)
#[cfg(target_os = "macos")]
fn has_metal_device() -> bool {
    let ok = metal::Device::system_default().is_some();
    eprintln!(
        "[detect] Metal: {}",
        if ok { "device available" } else { "no device" }
    );
    ok
}

/// Professional detection: NVML > Vulkan(AMD) > CPU
fn detect_gpu_pref() -> Pref {
    #[cfg(target_os = "macos")]
    {
        if has_metal_device() {
            return Pref::Metal;
        }
        return Pref::Cpu;
    }

    #[cfg(any(target_os = "windows", target_os = "linux"))]
    {
        if has_cuda_device() {
            return Pref::Cuda;
        }
        if has_amd_vulkan_device() {
            return Pref::Vulkan;
        }
        Pref::Cpu
    }
}

fn choose_variants<'a>(manifest: &'a Manifest, prefer: &Pref) -> Vec<&'a ManifestEntry> {
    let os = current_os_key();
    let arch = current_arch_key();

    let mut wanted = vec!["cpu"];

    match prefer {
        Pref::Auto => {
            let d = detect_gpu_pref();
            eprintln!("[detect] final choice: {:?}", d);
            match d {
                #[cfg(any(target_os = "linux", target_os = "windows"))]
                Pref::Cuda => wanted.push("cuda"),
                #[cfg(any(target_os = "linux", target_os = "windows"))]
                Pref::Vulkan => wanted.push("vulkan"),
                #[cfg(target_os = "macos")]
                Pref::Metal => wanted.push("metal"),
                _ => {}
            }
        }
        Pref::Cpu => {}
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        Pref::Cuda => wanted.push("cuda"),
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        Pref::Vulkan => wanted.push("vulkan"),
        #[cfg(target_os = "macos")]
        Pref::Metal => wanted.push("metal"),
    }

    wanted
        .iter()
        .filter_map(|v| {
            manifest
                .llama
                .iter()
                .find(|e| e.os == os && e.arch == arch && e.variant == *v)
        })
        .collect()
}

fn default_install_dir() -> Result<PathBuf> {
    let base = dirs::data_dir().ok_or_else(|| anyhow!("no data dir"))?;
    Ok(base.join("Strata").join("runtimes").join("llama"))
}

fn ensure_dir(p: &Path) -> Result<()> {
    fs::create_dir_all(p).with_context(|| format!("mkd {}", p.display()))
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

fn download(url: &str, dest: &Path) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;

    let mut resp = client
        .get(url)
        .send()
        .with_context(|| format!("GET {url}"))?;
    if !resp.status().is_success() {
        return Err(anyhow!("download failed: {}", resp.status()));
    }

    let mut out = fs::File::create(dest)?;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = resp.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
    }
    Ok(())
}

fn unzip_into(zip_path: &Path, dest: &Path) -> Result<()> {
    let f = fs::File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(f)?;
    ensure_dir(dest)?;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let outpath = dest.join(file.mangled_name());
        if file.name().ends_with('/') {
            ensure_dir(&outpath)?;
        } else {
            if let Some(parent) = outpath.parent() {
                ensure_dir(parent)?;
            }
            let mut out = fs::File::create(&outpath)?;
            std::io::copy(&mut file, &mut out)?;
        }
    }
    Ok(())
}

fn write_runtime_config(root: &Path, installed: &[&str], active_gpu: Option<&str>) -> Result<()> {
    // Which variant is active?
    let active_variant = active_gpu.unwrap_or("cpu");
    let current_lib_dir = root.join(active_variant).join("llama_backend");

    let cfg = serde_json::json!({
      "llama": {
        "active": if active_gpu.is_some() { "gpu" } else { "cpu" },
        "gpu_backend": active_gpu,
        "installed": installed,
        "root": root.to_string_lossy(),
        "current_lib_dir": current_lib_dir.to_string_lossy(),   // where StrataLlama.* lives
        "monolith": true
      }
    });

    let path = root.join("runtime.json");
    fs::write(&path, serde_json::to_vec_pretty(&cfg)?)?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    let manifest_url = args
        .manifest
        .or_else(|| env::var("STRATA_RUNTIME_MANIFEST").ok())
        .unwrap_or_else(|| DEFAULT_MANIFEST.to_string());

    let install_dir = if let Ok(p) = env::var("STRATA_RUNTIME_DIR") {
        PathBuf::from(p)
    } else if let Some(p) = args.install_dir {
        p
    } else {
        default_install_dir()?
    };

    println!("Manifest: {manifest_url}");
    println!("Install to: {}", install_dir.display());
    ensure_dir(&install_dir)?;

    let manifest_txt = reqwest::blocking::get(&manifest_url)?.text()?;
    let manifest: Manifest = serde_json::from_str(&manifest_txt)
        .context("invalid manifest JSON (expecting { llama: [...] })")?;

    let entries = choose_variants(&manifest, &args.prefer);
    if entries.is_empty() {
        return Err(anyhow!(
            "no matching runtime packs for OS={} arch={}",
            current_os_key(),
            current_arch_key()
        ));
    }

    let mut installed: Vec<&str> = Vec::new();
    let mut active_gpu: Option<&str> = None;

    let tmp = env::temp_dir().join("strata-runtime");
    ensure_dir(&tmp)?;

    for e in entries {
        let zip_path = tmp.join(&e.name);
        println!("Downloading {} → {}", e.url, zip_path.display());
        download(&e.url, &zip_path)?;

        let want_sha = e.sha256.trim().to_lowercase();
        let got_sha = sha256_file(&zip_path)?.trim().to_string();

        if got_sha != want_sha {
            return Err(anyhow!(
                "checksum mismatch for {} (got {}, want {})",
                e.name,
                got_sha,
                want_sha
            ));
        }

        let dest = install_dir.join(&e.variant);
        println!("Unzipping {} → {}", e.name, dest.display());
        unzip_into(&zip_path, &dest)?;

        installed.push(&e.variant);
        if e.variant != "cpu" {
            active_gpu = Some(e.variant.as_str());
        }

        // Validate plugin presence (monolithic runtime: plugin contains llama.cpp)
        let lib_dir = dest.join("llama_backend");
        let plugin = lib_dir.join(PLUGIN_NAME);

        if !plugin.exists() {
            eprintln!("❌ missing plugin: {}", plugin.display());
            eprintln!(
                "   (Your runtime zip should include {:?} in llama_backend/)",
                PLUGIN_NAME
            );
        } else {
            println!("✅ found {}", plugin.display());
        }
    }

    write_runtime_config(&install_dir, &installed, active_gpu)?;
    println!(
        "✅ Installed variants: {:?} (active: {})",
        installed,
        if active_gpu.is_some() { "gpu" } else { "cpu" }
    );

    Ok(())
}
