use crate::context::LlamaContext;
use crate::model::LlamaModel;
use crate::params::LlamaParams;
use std::path::Path;
use std::sync::Arc;

use super::cpu::CpuBackend;
// Stubs in place for future expansion; not used yet.
use super::{cuda::CudaBackend, rocm::RocmBackend, vulkan::VulkanBackend};

pub enum BackendKind {
    Cpu(CpuBackend),
    Vulkan(VulkanBackend),
    Cuda(CudaBackend),
    Rocm(RocmBackend),
}

/// Unified runtime backend. For now, always CPU; later, dispatch by hardware.
pub struct Backend {
    inner: BackendKind,
}

impl Backend {
    /// Load model + backend (CPU today; Vulkan/CUDA/ROCm later).
    pub fn load<P: AsRef<Path>>(model_path: P, params: LlamaParams) -> Result<Self, String> {
        // Phase 1: CPU only
        let cpu = CpuBackend::load(model_path, params)?;
        Ok(Self {
            inner: BackendKind::Cpu(cpu),
        })
    }

    /// Create a session context.
    pub fn create_context(&self) -> Result<LlamaContext, String> {
        match &self.inner {
            BackendKind::Cpu(b) => b.create_context(),
            // These arms will be implemented when those backends are real.
            BackendKind::Vulkan(_b) => Err("Vulkan backend not yet implemented".into()),
            BackendKind::Cuda(_b) => Err("CUDA backend not yet implemented".into()),
            BackendKind::Rocm(_b) => Err("ROCm backend not yet implemented".into()),
        }
    }

    /// Access resident model.
    pub fn model(&self) -> Arc<LlamaModel> {
        match &self.inner {
            BackendKind::Cpu(b) => b.model(),
            BackendKind::Vulkan(_b) => unimplemented!("Vulkan backend not yet implemented"),
            BackendKind::Cuda(_b) => unimplemented!("CUDA backend not yet implemented"),
            BackendKind::Rocm(_b) => unimplemented!("ROCm backend not yet implemented"),
        }
    }

    /// Access construction params.
    pub fn params(&self) -> &LlamaParams {
        match &self.inner {
            BackendKind::Cpu(b) => b.params(),
            BackendKind::Vulkan(_b) => unimplemented!("Vulkan backend not yet implemented"),
            BackendKind::Cuda(_b) => unimplemented!("CUDA backend not yet implemented"),
            BackendKind::Rocm(_b) => unimplemented!("ROCm backend not yet implemented"),
        }
    }
}
