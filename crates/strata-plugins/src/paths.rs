use dirs::data_dir;
use std::path::PathBuf;

/// ~/.local/share/Strata   (or platform-equivalent)
pub fn strata_home() -> PathBuf {
    data_dir()
        .unwrap_or_else(|| PathBuf::from("~/.local/share"))
        .join("Strata")
}

/// ~/.local/share/Strata/plugins
pub fn plugins_dir() -> PathBuf {
    strata_home().join("plugins")
}

/// ~/.local/share/Strata/runtimes
pub fn runtimes_dir() -> PathBuf {
    strata_home().join("runtimes")
}

/// ~/.local/share/Strata/runtimes/llama
pub fn runtimes_llama_dir() -> PathBuf {
    runtimes_dir().join("llama")
}

/// ~/.local/share/Strata/runtimes/llama/<variant>
pub fn runtimes_llama_variant_dir(variant: &str) -> PathBuf {
    runtimes_llama_dir().join(variant)
}

/// ~/.local/share/Strata/cache
pub fn cache_dir() -> PathBuf {
    strata_home().join("cache")
}

/// ~/.local/share/Strata/logs
pub fn logs_dir() -> PathBuf {
    strata_home().join("logs")
}
