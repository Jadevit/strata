//! Strata hardware profiling crate.
//! Detects CPU/GPU/backends, caches to ~/.local/share/Strata/cache/hwprof/profile.json.

pub mod cache;
pub mod detect;
pub mod paths;
pub mod types;

pub use cache::{detect_and_cache, load_cached, load_or_detect, validate_or_redetect};
pub use paths::{
    cache_dir, cache_hwprof_dir, hwprof_profile_path, logs_dir, models_dir, plugins_dir,
    runtimes_dir, runtimes_llama_dir, runtimes_llama_variant_dir, strata_home,
};
pub use types::{
    BackendReasons, BackendSupport, CpuInfo, GpuDriverInfo, GpuInfo, HardwareProfile, ProbeTimes,
    StorageInfo,
};
