use dirs::data_dir;
use std::path::PathBuf;

pub fn strata_home() -> PathBuf {
    // Linux resolves to ~/.local/share/Strata
    data_dir()
        .unwrap_or_else(|| PathBuf::from("~/.local/share"))
        .join("Strata")
}

pub fn models_dir() -> PathBuf {
    strata_home().join("models")
}
pub fn runtimes_dir() -> PathBuf {
    strata_home().join("runtimes")
}
pub fn runtimes_llama_dir() -> PathBuf {
    runtimes_dir().join("llama")
}
pub fn runtimes_llama_variant_dir(variant: &str) -> PathBuf {
    runtimes_llama_dir().join(variant)
}

pub fn cache_dir() -> PathBuf {
    strata_home().join("cache")
}
pub fn cache_hwprof_dir() -> PathBuf {
    cache_dir().join("hwprof")
}
pub fn hwprof_profile_path() -> PathBuf {
    cache_hwprof_dir().join("profile.json")
}

pub fn logs_dir() -> PathBuf {
    strata_home().join("logs")
}
pub fn plugins_dir() -> PathBuf {
    strata_home().join("plugins")
}
