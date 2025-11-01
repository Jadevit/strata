#![allow(non_snake_case)]

use anyhow::{anyhow, Context, Result};
use ash::{vk, Entry};
use libloading::{Library, Symbol};
use std::ffi::CString;
use std::time::Duration;

use super::PlatformDetect;
use crate::detect::util;
use crate::types::{BackendReasons, GpuDriverInfo, GpuInfo, ProbeTimes};

const VENDOR_NVIDIA: u32 = 0x10DE;
const VENDOR_AMD: u32 = 0x1002;
const VENDOR_INTEL: u32 = 0x8086;
const VENDOR_APPLE: u32 = 0x106B;

fn vendor_name(id: u32) -> &'static str {
    match id {
        VENDOR_NVIDIA => "NVIDIA",
        VENDOR_AMD => "AMD",
        VENDOR_INTEL => "Intel",
        VENDOR_APPLE => "Apple",
        _ => "Unknown",
    }
}

pub fn detect_platform() -> Result<PlatformDetect> {
    let mut reasons = BackendReasons::default();
    let mut diags: Vec<String> = Vec::new();
    let mut times = ProbeTimes::default();

    let to = Duration::from_millis(util::env_timeout_ms());

    // Vulkan (timeout wrapped)
    let (vk_out, vk_reason, vk_ms) = util::with_timeout("vulkan", to, || enumerate_vulkan_gpus());
    times.vulkan_ms = Some(vk_ms);
    let (gpus, vulkan_ok) = match vk_out {
        Some(Ok(v)) => v,
        Some(Err(e)) => {
            reasons.vulkan = Some(format!("probe_error:{e}"));
            diags.push(format!("[vulkan] error: {e}"));
            (Vec::new(), false)
        }
        None => {
            reasons.vulkan = Some(vk_reason.unwrap_or_else(|| "timeout".into()));
            diags.push("[vulkan] timeout".into());
            (Vec::new(), false)
        }
    };
    if util::disabled("vulkan") {
        diags.push("[vulkan] disabled by env".into());
        reasons.vulkan = Some("disabled_env".into());
    }

    // CUDA (timeout wrapped)
    let (cu_out, cu_reason, cu_ms) =
        util::with_timeout("cuda", to, || cuda_driver_count_and_version());
    let (cuda_ok, cuda_ver) = match cu_out {
        Some(Ok((cnt, ver))) => {
            let ok = cnt > 0;
            if !ok {
                reasons.cuda = Some("nvml_no_devices".into());
            }
            (ok, ver_string(ver))
        }
        Some(Err(e)) => {
            reasons.cuda = Some(format!("probe_error:{e}"));
            (false, None)
        }
        None => {
            reasons.cuda = Some(cu_reason.unwrap_or_else(|| "timeout".into()));
            (false, None)
        }
    };
    times.nvml_ms = Some(cu_ms);
    if util::disabled("cuda") {
        diags.push("[cuda] disabled by env".into());
        reasons.cuda = Some("disabled_env".into());
    }

    // ROCm (timeout wrapped)
    let (rc_out, rc_reason, rc_ms) = util::with_timeout("rocm", to, || rocm_yes());
    let rocm_ok = match rc_out {
        Some(Ok(b)) => b,
        Some(Err(e)) => {
            reasons.rocm = Some(format!("probe_error:{e}"));
            false
        }
        None => {
            reasons.rocm = Some(rc_reason.unwrap_or_else(|| "timeout".into()));
            false
        }
    };
    times.rocm_ms = Some(rc_ms);
    if util::disabled("rocm") {
        diags.push("[rocm] disabled by env".into());
        reasons.rocm = Some("disabled_env".into());
    }

    // If Vulkan only had software renderers, force false with reason.
    if !vulkan_ok && reasons.vulkan.is_none() && !gpus.is_empty() {
        // If we saw only software adapters, set reason explicitly.
        if gpus.iter().all(|g| g.software_renderer) {
            reasons.vulkan = Some("software_renderer".into());
        } else {
            reasons.vulkan = Some("no_supported_device".into());
        }
    }

    let total_ms =
        times.vulkan_ms.unwrap_or(0) + times.nvml_ms.unwrap_or(0) + times.rocm_ms.unwrap_or(0);

    Ok(PlatformDetect {
        gpus,
        cuda: cuda_ok && !util::disabled("cuda"),
        rocm: rocm_ok && !util::disabled("rocm"),
        vulkan: vulkan_ok && !util::disabled("vulkan"),
        metal: false,

        cuda_driver_version: cuda_ver,
        backend_reasons: reasons,
        probe_times: times,
        probe_total_ms: total_ms,
        diagnostics: diags,
    })
}

fn enumerate_vulkan_gpus() -> Result<(Vec<GpuInfo>, bool)> {
    let entry = unsafe { Entry::load() }?;
    let app = CString::new("strata-hwprof").unwrap();
    let eng = CString::new("strata").unwrap();

    let app_info = vk::ApplicationInfo {
        s_type: vk::StructureType::APPLICATION_INFO,
        p_next: std::ptr::null(),
        p_application_name: app.as_ptr(),
        application_version: 0,
        p_engine_name: eng.as_ptr(),
        engine_version: 0,
        api_version: vk::API_VERSION_1_0,
        ..Default::default()
    };
    let ci = vk::InstanceCreateInfo {
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
    let instance = unsafe { entry.create_instance(&ci, None) }?;
    let devices = unsafe { instance.enumerate_physical_devices() }?;

    let mut out = Vec::new();
    let mut supports_backend = false;

    for pd in devices {
        let props = unsafe { instance.get_physical_device_properties(pd) };
        let dtype = props.device_type;
        let name = cstr_to_string(&props.device_name);
        let vendor_id = props.vendor_id;
        let device_id = props.device_id;

        // Detect software renderer
        let soft = util::is_software_renderer(&name, dtype);
        let software_renderer = soft.is_some();
        let software_reason = soft.map(|s| s.to_string());

        // quick VRAM (device-local heaps)
        let mem_props = unsafe { instance.get_physical_device_memory_properties(pd) };
        let vram_bytes = util::vram_from_heaps(&mem_props);

        // Integrated vs discrete
        let integrated = matches!(dtype, vk::PhysicalDeviceType::INTEGRATED_GPU);

        let mut info = GpuInfo {
            vendor_id,
            device_id,
            vendor: vendor_name(vendor_id).to_string(),
            name: name.clone(),
            driver: Some(GpuDriverInfo {
                cuda: None,
                nvml: None,
                vulkan: Some(util::fmt_vk_driver(props.driver_version)),
                rocm: None,
                metal: None,
            }),
            vram_bytes,
            integrated,
            software_renderer,
            software_reason,
        };
        out.push(info);

        // Only count real AMD hardware for Vulkan backend truthiness
        if !software_renderer && vendor_id == VENDOR_AMD {
            if vram_bytes.unwrap_or(0) > 0 {
                supports_backend = true;
            }
        }
    }

    unsafe { instance.destroy_instance(None) };
    Ok((out, supports_backend))
}

// --- CUDA (driver API, dlopen) ---
#[allow(non_camel_case_types)]
type CuInit = unsafe extern "C" fn(u32) -> i32;
#[allow(non_camel_case_types)]
type CuDevCnt = unsafe extern "C" fn(*mut i32) -> i32;
#[allow(non_camel_case_types)]
type CuDrvVer = unsafe extern "C" fn(*mut i32) -> i32;

fn cuda_driver_count_and_version() -> Result<(i32, i32)> {
    const CANDIDATES: &[&str] = &["libcuda.so.1", "libcuda.so"];
    let mut last_err = None;

    for name in CANDIDATES {
        unsafe {
            match Library::new(name) {
                Ok(lib) => {
                    let cuInit: Symbol<CuInit> = lib.get(b"cuInit").context("get cuInit")?;
                    let cuDeviceGetCount: Symbol<CuDevCnt> = lib
                        .get(b"cuDeviceGetCount")
                        .context("get cuDeviceGetCount")?;
                    let cuDriverGetVersion: Symbol<CuDrvVer> = lib
                        .get(b"cuDriverGetVersion")
                        .context("get cuDriverGetVersion")?;

                    if cuInit(0) != 0 {
                        return Err(anyhow!("cuInit failed"));
                    }
                    let mut cnt = 0i32;
                    if cuDeviceGetCount(&mut cnt as *mut i32) != 0 {
                        return Err(anyhow!("cuDeviceGetCount failed"));
                    }
                    let mut ver = 0i32;
                    let _ = cuDriverGetVersion(&mut ver as *mut i32);
                    return Ok((cnt, ver));
                }
                Err(e) => {
                    last_err = Some(anyhow!(e));
                }
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow!("No CUDA driver library found")))
}

fn ver_string(v: i32) -> Option<String> {
    if v <= 0 {
        return None;
    }
    let major = v / 1000;
    let minor = (v % 1000) / 10;
    Some(format!("{major}.{minor}"))
}

// --- ROCm (HIP) ---
#[allow(non_camel_case_types)]
type HipInit = unsafe extern "C" fn(u32) -> i32;
#[allow(non_camel_case_types)]
type HipGetDevCnt = unsafe extern "C" fn(*mut i32) -> i32;

fn rocm_yes() -> Result<bool> {
    unsafe {
        if let Ok(lib) = Library::new("libamdhip64.so") {
            let hipInit: Symbol<HipInit> = lib.get(b"hipInit").context("get hipInit")?;
            let hipGetDeviceCount: Symbol<HipGetDevCnt> = lib
                .get(b"hipGetDeviceCount")
                .context("get hipGetDeviceCount")?;

            if hipInit(0) == 0 {
                let mut cnt = 0i32;
                if hipGetDeviceCount(&mut cnt as *mut i32) == 0 && cnt > 0 {
                    return Ok(true);
                }
            }
            return Ok(false);
        }
    }
    Err(anyhow!("libamdhip64.so not found"))
}

fn cstr_to_string(arr: &[i8]) -> String {
    let bytes = arr
        .iter()
        .take_while(|b| **b != 0)
        .map(|b| *b as u8)
        .collect::<Vec<_>>();
    String::from_utf8_lossy(&bytes).to_string()
}
