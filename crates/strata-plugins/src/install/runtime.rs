use crate::errors::{Result, StoreError};
use crate::manifest::verify_entry_sha256;
use crate::net::download_to_path;
use crate::paths::runtimes_llama_dir;
use crate::types::{Manifest, ManifestEntry, Pref, RuntimeChoice};
use anyhow::Context;
use std::fs;
use std::path::{Path, PathBuf};

#[cfg(any(target_os = "linux", target_os = "windows"))]
use {ash::vk, nvml_wrapper::Nvml};

#[cfg(target_os = "macos")]
use metal::Device as MetalDevice;

use super::unzip::unzip_into;

fn current_os_key() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "windows-latest"
    }
    #[cfg(target_os = "macos")]
    {
        "macos-14"
    }
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        "ubuntu-22.04"
    }
}

fn current_arch_key() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        "x64"
    }
    #[cfg(target_arch = "aarch64")]
    {
        "arm64"
    }
    #[cfg(all(not(target_arch = "x86_64"), not(target_arch = "aarch64")))]
    {
        "x64"
    }
}

/// Pick variants to install based on preference and hardware.
/// Always include "cpu". If a GPU is detected/selected, include it after cpu.
pub fn choose_variants(manifest: &Manifest, prefer: Pref) -> (Vec<&ManifestEntry>, RuntimeChoice) {
    let os = current_os_key();
    let arch = current_arch_key();

    let mut chosen: Vec<&ManifestEntry> = Vec::new();

    // Always select CPU first if available
    if let Some(cpu) = manifest
        .llama
        .iter()
        .find(|e| e.os == os && e.arch == arch && e.variant == "cpu")
    {
        chosen.push(cpu);
    }

    let mut active_gpu: Option<&str> = None;

    match prefer {
        Pref::Auto => {
            #[cfg(target_os = "macos")]
            {
                if has_metal_device() {
                    if let Some(metal) = pick(manifest, os, arch, "metal") {
                        chosen.push(metal);
                        active_gpu = Some("metal");
                    }
                }
            }

            #[cfg(any(target_os = "linux", target_os = "windows"))]
            {
                if has_cuda_device() {
                    if let Some(cuda) = pick(manifest, os, arch, "cuda") {
                        chosen.push(cuda);
                        active_gpu = Some("cuda");
                    }
                } else if has_amd_vulkan_device() {
                    if let Some(vk) = pick(manifest, os, arch, "vulkan") {
                        chosen.push(vk);
                        active_gpu = Some("vulkan");
                    }
                }
            }
        }
        Pref::Cpu => {}
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        Pref::Cuda => {
            if let Some(cuda) = pick(manifest, os, arch, "cuda") {
                chosen.push(cuda);
                active_gpu = Some("cuda");
            }
        }
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        Pref::Vulkan => {
            if let Some(vk) = pick(manifest, os, arch, "vulkan") {
                chosen.push(vk);
                active_gpu = Some("vulkan");
            }
        }
        #[cfg(target_os = "macos")]
        Pref::Metal => {
            if let Some(metal) = pick(manifest, os, arch, "metal") {
                chosen.push(metal);
                active_gpu = Some("metal");
            }
        }
    }

    let choice = RuntimeChoice {
        os: os.to_string(),
        arch: arch.to_string(),
        chosen_variants: chosen.iter().map(|e| e.variant.clone()).collect(),
        active_gpu: active_gpu.map(|s| s.to_string()),
    };

    (chosen, choice)
}

fn pick<'a>(m: &'a Manifest, os: &str, arch: &str, variant: &str) -> Option<&'a ManifestEntry> {
    m.llama
        .iter()
        .find(|e| e.os == os && e.arch == arch && e.variant == variant)
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn has_cuda_device() -> bool {
    match Nvml::init() {
        Ok(nvml) => nvml.device_count().map(|c| c > 0).unwrap_or(false),
        Err(_) => false,
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn has_amd_vulkan_device() -> bool {
    use std::ffi::CString;

    // ✅ Use Entry::load(); this is the ash 0.38 path to the loader
    let entry = match unsafe { ash::Entry::load() } {
        Ok(e) => e,
        Err(e) => {
            eprintln!("[detect] Vulkan loader not available: {e:?}");
            return false;
        }
    };

    let app_name = CString::new("strata-plugins").unwrap();
    let engine_name = CString::new("strata").unwrap();

    let app_info = vk::ApplicationInfo {
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
        p_application_info: &app_info,
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

    let mut found_amd = false;
    if let Ok(devices) = unsafe { instance.enumerate_physical_devices() } {
        for pd in devices {
            let props = unsafe { instance.get_physical_device_properties(pd) };
            if props.vendor_id == 0x1002 {
                found_amd = true;
                break;
            }
        }
    }

    unsafe { instance.destroy_instance(None) };
    found_amd
}

#[cfg(target_os = "macos")]
fn has_metal_device() -> bool {
    MetalDevice::system_default().is_some()
}

/// Install the chosen variants: download, verify, unzip.
pub fn install_variants(entries: &[&ManifestEntry], install_root: &Path) -> Result<Vec<String>> {
    fs::create_dir_all(install_root)?;
    let mut installed = Vec::new();

    for e in entries {
        let zip_path = std::env::temp_dir().join(&e.name);

        download_to_path(&e.url, &zip_path)?;
        verify_entry_sha256(e, &zip_path)?;

        let dest = install_root.join(&e.variant);
        unzip_into(&zip_path, &dest)?;

        installed.push(e.variant.clone());
    }

    Ok(installed)
}

#[cfg(target_os = "windows")]
const CPU_BASENAME: &str = "StrataLlama.dll";
#[cfg(target_os = "linux")]
const CPU_BASENAME: &str = "libStrataLlama.so";
#[cfg(target_os = "macos")]
const CPU_BASENAME: &str = "libStrataLlama.dylib";

#[cfg(target_os = "windows")]
fn basename_for_variant(v: &str) -> &'static str {
    match v {
        "cuda" => "StrataLlama_cuda.dll",
        "vulkan" => "StrataLlama_vulkan.dll",
        "metal" => "StrataLlama_metal.dll",
        _ => CPU_BASENAME, // "cpu"
    }
}
#[cfg(target_os = "linux")]
fn basename_for_variant(v: &str) -> &'static str {
    match v {
        "cuda" => "libStrataLlama_cuda.so",
        "vulkan" => "libStrataLlama_vulkan.so",
        "metal" => "libStrataLlama_metal.so",
        _ => CPU_BASENAME, // "cpu"
    }
}
#[cfg(target_os = "macos")]
fn basename_for_variant(v: &str) -> &'static str {
    match v {
        "metal" => "libStrataLlama_metal.dylib",
        "cuda" => "libStrataLlama_cuda.dylib",
        "vulkan" => "libStrataLlama_vulkan.dylib",
        _ => CPU_BASENAME, // "cpu"
    }
}

/// Write Strata’s runtime.json describing which variant is active.
/// Emits:
/// - top-level `active_variant` (e.g., "cpu" | "cuda" | "vulkan" | "metal")
/// - top-level `current_lib_dir` (absolute dir holding the active plugin)
/// - top-level `variants` map: { variant: { "dir": "...", "file": "..." } }
/// Also keeps the legacy `llama { ... }` block for backwards compatibility.
pub fn write_runtime_config(
    root: &Path,
    installed: &[String],
    active_gpu: Option<&str>,
) -> Result<()> {
    let active_variant = active_gpu.unwrap_or("cpu");
    let current_lib_dir = root.join(active_variant).join("llama_backend");

    // Build `variants` map
    let mut vmap = serde_json::Map::new();
    for variant in installed {
        let dir = root.join(variant).join("llama_backend");
        let file = basename_for_variant(variant);
        vmap.insert(
            variant.clone(),
            serde_json::json!({
                "dir": dir.to_string_lossy(),
                "file": file
            }),
        );
    }

    let json = serde_json::json!({
        "active_variant": active_variant,
        "current_lib_dir": current_lib_dir.to_string_lossy(),
        "variants": vmap,
        "monolith": true,

        // legacy/compat view used elsewhere in the app
        "llama": {
            "active": if active_gpu.is_some() { "gpu" } else { "cpu" },
            "gpu_backend": active_gpu,
            "installed": installed,
            "root": root.to_string_lossy(),
            "current_lib_dir": current_lib_dir.to_string_lossy(),
            "monolith": true
        }
    });

    let path = root.join("runtime.json");
    fs::write(&path, serde_json::to_vec_pretty(&json)?)?;
    Ok(())
}
