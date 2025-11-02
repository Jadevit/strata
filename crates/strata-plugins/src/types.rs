use serde::{Deserialize, Serialize};

/// Which pack to prefer. Auto = detect best GPU + include cpu.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Pref {
    Auto,
    Cpu,
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    Cuda,
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    Vulkan,
    #[cfg(target_os = "macos")]
    Metal,
}

/// A single runtime pack entry in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestEntry {
    pub name: String,    // e.g. "llama-cuda-x64.zip"
    pub sha256: String,  // lowercase hex
    pub os: String,      // "windows-latest" | "ubuntu-22.04" | "macos-14"
    pub arch: String,    // "x64" | "arm64"
    pub variant: String, // "cpu" | "cuda" | "vulkan" | "metal"
    pub url: String,     // direct HTTPS URL
}

/// Top-level manifest (can hold more families later).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub llama: Vec<ManifestEntry>,
}

/// Summary of the chosen install plan (for UI).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeChoice {
    pub os: String,
    pub arch: String,
    pub chosen_variants: Vec<String>, // ordered (cpu first, then gpu if any)
    pub active_gpu: Option<String>,
}
