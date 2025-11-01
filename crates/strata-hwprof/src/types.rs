use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub schema: u32, // major, stays 1
    #[serde(default)]
    pub schema_minor: u16, // NEW: additive schema, now 1

    pub os: String,
    pub arch: String,

    pub cpu: CpuInfo,
    pub ram_gb: u64,

    pub gpus: Vec<GpuInfo>,
    pub backends: BackendSupport,

    #[serde(default)]
    pub backend_reasons: Option<BackendReasons>, // NEW: why a backend is false

    #[serde(default)]
    pub storage: Option<StorageInfo>, // NEW: Strata data root free

    pub fingerprint: String,
    pub created_at: String,
    pub updated_at: String,

    // Diagnostics/telemetry (optional)
    #[serde(default)]
    pub probe_ms_total: Option<u64>, // NEW
    #[serde(default)]
    pub probe_times: Option<ProbeTimes>, // NEW
    #[serde(default)]
    pub diagnostics: Option<Vec<String>>, // NEW (debug only)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    pub brand: String,
    pub threads: u32, // logical threads (kept)
    #[serde(default)]
    pub physical_cores: Option<u32>, // NEW: best-effort
    pub avx2: bool,
    pub avx512: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub vendor_id: u32,
    pub device_id: u32,
    pub vendor: String,
    pub name: String,

    #[serde(default)]
    pub driver: Option<GpuDriverInfo>,

    // NEW
    #[serde(default)]
    pub vram_bytes: Option<u64>, // best-effort
    #[serde(default)]
    pub integrated: bool, // inferred
    #[serde(default)]
    pub software_renderer: bool, // llvmpipe/SwiftShader/etc
    #[serde(default)]
    pub software_reason: Option<String>, // "llvmpipe", "SWRast", etc.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDriverInfo {
    #[serde(default)]
    pub cuda: Option<String>,
    #[serde(default)]
    pub nvml: Option<String>,
    // NEW (optional)
    #[serde(default)]
    pub vulkan: Option<String>,
    #[serde(default)]
    pub rocm: Option<String>,
    #[serde(default)]
    pub metal: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendSupport {
    pub cpu: bool,
    pub cuda: bool,
    pub rocm: bool,
    pub vulkan: bool,
    pub metal: bool,
}

// NEW: why a backend isn’t available (timeout/no_device/software_renderer)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BackendReasons {
    #[serde(default)]
    pub cuda: Option<String>,
    #[serde(default)]
    pub rocm: Option<String>,
    #[serde(default)]
    pub vulkan: Option<String>,
    #[serde(default)]
    pub metal: Option<String>,
}

// NEW: where Strata stores data + free space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageInfo {
    pub data_root: String,
    #[serde(default)]
    pub free_bytes: Option<u64>,
}

// NEW: lightweight per-probe timings
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProbeTimes {
    #[serde(default)]
    pub nvml_ms: Option<u64>, // CUDA driver probe time
    #[serde(default)]
    pub vulkan_ms: Option<u64>,
    #[serde(default)]
    pub metal_ms: Option<u64>,
    #[serde(default)]
    pub rocm_ms: Option<u64>,
}
