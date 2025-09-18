use crate::params::LlamaParams;
use std::path::Path;

pub struct RocmBackend;

impl RocmBackend {
    pub fn load<P: AsRef<Path>>(_model_path: P, _params: LlamaParams) -> Result<Self, String> {
        Err("ROCm backend not yet implemented".to_string())
    }
}
