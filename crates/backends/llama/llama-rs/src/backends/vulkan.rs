use crate::params::LlamaParams;
use std::path::Path;

pub struct VulkanBackend;

impl VulkanBackend {
    pub fn load<P: AsRef<Path>>(_model_path: P, _params: LlamaParams) -> Result<Self, String> {
        Err("Vulkan backend not yet implemented".to_string())
    }
}
