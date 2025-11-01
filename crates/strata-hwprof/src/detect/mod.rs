use anyhow::Result;
use raw_cpuid::CpuId;
use sysinfo::System;

use crate::paths::strata_home;
use crate::types::{
    BackendReasons, BackendSupport, CpuInfo, GpuDriverInfo, GpuInfo, HardwareProfile, ProbeTimes,
    StorageInfo,
};

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
use linux::detect_platform;

#[cfg(target_os = "windows")]
mod windows;
#[cfg(target_os = "windows")]
use windows::detect_platform;

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
use macos::detect_platform;

mod util;

pub fn detect_now() -> Result<HardwareProfile> {
    let os = std::env::consts::OS.to_string();
    let arch = std::env::consts::ARCH.to_string();

    let cpu = detect_cpu();

    // RAM
    let sys = System::new_all();
    let bytes = sys.total_memory(); // sysinfo returns bytes
    let ram_gb = (bytes as f64 / (1024.0 * 1024.0 * 1024.0)).round() as u64;

    // OS-specific GPU/backends (+ timings/reasons/diags)
    let plat = detect_platform()?;

    // Attach CUDA driver version to NVIDIA GPUs if provided by platform layer
    let mut gpus = plat.gpus;
    if let Some(cuda_ver) = &plat.cuda_driver_version {
        for g in &mut gpus {
            if g.vendor.eq_ignore_ascii_case("NVIDIA") {
                let mut drv = g.driver.clone().unwrap_or(GpuDriverInfo {
                    cuda: None,
                    nvml: None,
                    vulkan: None,
                    rocm: None,
                    metal: None,
                });
                drv.cuda = Some(cuda_ver.clone());
                g.driver = Some(drv);
            }
        }
    }

    // Optional: storage free space for Strata home
    let data_root = strata_home();
    let storage = match fs2::available_space(&data_root) {
        Ok(free) => Some(StorageInfo {
            data_root: data_root.display().to_string(),
            free_bytes: Some(free),
        }),
        Err(_) => Some(StorageInfo {
            data_root: data_root.display().to_string(),
            free_bytes: None,
        }),
    };

    let backends = BackendSupport {
        cpu: true,
        cuda: plat.cuda,
        rocm: plat.rocm,
        vulkan: plat.vulkan,
        metal: plat.metal,
    };

    Ok(HardwareProfile {
        schema: 0,       // set in cache layer
        schema_minor: 1, // additive schema
        os,
        arch,
        cpu,
        ram_gb,
        gpus,
        backends,
        backend_reasons: Some(plat.backend_reasons),
        storage,
        fingerprint: String::new(),
        created_at: String::new(),
        updated_at: String::new(),
        probe_ms_total: Some(plat.probe_total_ms),
        probe_times: Some(plat.probe_times),
        diagnostics: if util::hwprof_debug() {
            Some(plat.diagnostics)
        } else {
            None
        },
    })
}

fn detect_cpu() -> CpuInfo {
    let cpuid = CpuId::new();

    // Prefer full brand string; fall back to vendor
    let brand = cpuid
        .get_processor_brand_string()
        .map(|b| b.as_str().trim().to_string())
        .or_else(|| cpuid.get_vendor_info().map(|v| v.as_str().to_string()))
        .unwrap_or_else(|| "Unknown CPU".into());

    let (avx2, avx512) = if let Some(f) = cpuid.get_extended_feature_info() {
        let avx2 = f.has_avx2();
        let avx512 = f.has_avx512f()
            || f.has_avx512dq()
            || f.has_avx512cd()
            || f.has_avx512bw()
            || f.has_avx512vl();
        (avx2, avx512)
    } else {
        (false, false)
    };

    let threads = num_cpus::get() as u32;
    let physical = num_cpus::get_physical();

    CpuInfo {
        brand,
        threads,
        physical_cores: Some(physical as u32),
        avx2,
        avx512,
    }
}

/// OS layer returns this and mod.rs assembles the final profile.
pub struct PlatformDetect {
    pub gpus: Vec<GpuInfo>,
    pub cuda: bool,
    pub rocm: bool,
    pub vulkan: bool,
    pub metal: bool,

    pub cuda_driver_version: Option<String>,
    pub backend_reasons: BackendReasons,

    pub probe_times: ProbeTimes,
    pub probe_total_ms: u64,
    pub diagnostics: Vec<String>,
}
