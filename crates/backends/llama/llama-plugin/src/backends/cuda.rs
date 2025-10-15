use crate::params::LlamaParams;
use std::path::Path;

pub struct CudaBackend;

impl CudaBackend {
    pub fn load<P: AsRef<Path>>(_model_path: P, _params: LlamaParams) -> Result<Self, String> {
        Err("CUDA backend not yet implemented".to_string())
    }
}
