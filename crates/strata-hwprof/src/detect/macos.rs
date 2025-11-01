use super::PlatformDetect;
use crate::types::{BackendReasons, GpuDriverInfo, GpuInfo, ProbeTimes};
use anyhow::Result;

pub fn detect_platform() -> Result<PlatformDetect> {
    let mut reasons = BackendReasons::default();
    let mut times = ProbeTimes::default();
    let mut diags: Vec<String> = Vec::new();

    // Metal presence
    let metal_ok = metal::Device::system_default().is_some();
    if !metal_ok {
        reasons.metal = Some("no_device".into());
    }

    let gpus = enumerate_metal_devices();

    Ok(PlatformDetect {
        gpus,
        cuda: false,
        rocm: false,
        vulkan: false,
        metal: metal_ok,

        cuda_driver_version: None,
        backend_reasons: reasons,
        probe_times: times,
        probe_total_ms: 0,
        diagnostics: diags,
    })
}

fn enumerate_metal_devices() -> Vec<GpuInfo> {
    metal::all_devices()
        .into_iter()
        .map(|d| GpuInfo {
            vendor_id: 0x106B,
            device_id: 0,
            vendor: "Apple".to_string(),
            name: d.name().to_string(),
            driver: Some(GpuDriverInfo {
                cuda: None,
                nvml: None,
                vulkan: None,
                rocm: None,
                metal: None,
            }),
            vram_bytes: None, // Metal API here not queried; can add later
            integrated: true, // Apple Silicon iGPU (eGPU would be false if we detect one later)
            software_renderer: false,
            software_reason: None,
        })
        .collect()
}
