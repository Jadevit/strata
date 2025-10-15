use crate::context::LlamaContext;
use crate::model::LlamaModel;
use crate::params::LlamaParams;
use std::path::Path;
use std::sync::Arc;

/// CPU backend: owns loaded model + params to spawn contexts.
pub struct CpuBackend {
    model: Arc<LlamaModel>,
    params: LlamaParams,
}

impl CpuBackend {
    /// Load model weights from disk (no context yet).
    ///
    /// SAFETY NOTE: Caller should have initialized llama runtime once
    /// (e.g., `crate::ffi::init_backend()`) before calling `load`.
    pub fn load<P: AsRef<Path>>(model_path: P, mut params: LlamaParams) -> Result<Self, String> {
        Self::normalize_threads(&mut params);

        #[cfg(feature = "ffi-trace")]
        {
            println!(
                "[cpu] load: path={}, n_ctx={}, n_batch={}, n_ubatch={}, n_threads={}, n_threads_batch={}",
                model_path.as_ref().display(),
                params.n_ctx, params.n_batch, params.n_ubatch, params.n_threads, params.n_threads_batch
            );
        }

        // Use FFI to load the raw model; wrap in safe newtype.
        let path_str = model_path
            .as_ref()
            .to_str()
            .ok_or_else(|| "model path is not valid UTF-8".to_string())?;
        let raw_model = unsafe { crate::ffi::load_model(path_str)? };
        let model = Arc::new(LlamaModel::new(raw_model.as_ptr())?);

        Ok(Self { model, params })
    }

    /// Create a fresh inference context (session).
    pub fn create_context(&self) -> Result<LlamaContext, String> {
        #[cfg(feature = "ffi-trace")]
        {
            println!(
                "[cpu] create_context: n_ctx={}, n_batch={}, n_ubatch={}, embeddings={}",
                self.params.n_ctx,
                self.params.n_batch,
                self.params.n_ubatch,
                self.params.embeddings
            );
        }

        self.model
            .create_context(self.params.to_ffi(), self.params.embeddings)
    }

    /// Resident model (thread-safe; immutable after load).
    pub fn model(&self) -> Arc<LlamaModel> {
        Arc::clone(&self.model)
    }

    /// Params used for contexts.
    pub fn params(&self) -> &LlamaParams {
        &self.params
    }

    /// Fill thread counts if unset using physical cores (fallback to logical).
    fn normalize_threads(p: &mut LlamaParams) {
        let cores_physical = num_cpus::get_physical();
        let cores_logical = num_cpus::get();
        let cores = if cores_physical > 0 {
            cores_physical
        } else {
            cores_logical
        } as i32;

        if p.n_threads <= 0 {
            p.n_threads = cores.max(1);
        }
        if p.n_threads_batch <= 0 {
            p.n_threads_batch = p.n_threads;
        }

        #[cfg(feature = "ffi-trace")]
        eprintln!(
            "[cpu] normalize_threads → n_threads={}, n_threads_batch={} (cores={})",
            p.n_threads, p.n_threads_batch, cores
        );
    }
}
