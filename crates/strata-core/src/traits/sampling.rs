use std::collections::HashMap;

/// User-tunable sampling parameters passed from the engine to backends.
/// Backends should treat these as *desired* knobs; unsupported options
/// must be gracefully ignored or downgraded based on capabilities.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// If true, pick argmax and ignore other stochastic knobs.
    pub greedy: bool,

    // Sampling filters
    pub temperature: Option<f32>, // > 0.0 enables temperature scaling
    pub top_k: Option<u32>,       // >= 1 keeps the K most likely candidates
    pub top_p: Option<f32>,       // (0, 1] nucleus sampling
    pub typical_p: Option<f32>,   // (0, 1] typical sampling
    pub tfs_z: Option<f32>,       // (0, 1] tail-free sampling (not universally supported)

    // Token penalties
    pub repetition_penalty: Option<PenaltyParams>,
    pub penalize_newline: bool,

    // Mirostat options (v1 or v2)
    pub mirostat: Option<MirostatParams>,

    /// Optional per-token logit bias. Keys are raw token IDs as u32 for UI/serialization
    /// friendliness. Backends should convert once at the boundary and ignore unknown IDs.
    pub logit_bias: Option<HashMap<u32, f32>>,
}

#[derive(Debug, Clone)]
pub struct PenaltyParams {
    pub last_n: i32, // number of recent tokens to consider; <=0 disables
    pub repeat: f32, // >= 1.0 reduces repetition; <1.0 increases it (generally undesirable)
    pub frequency: f32,
    pub presence: f32,
}

#[derive(Debug, Clone)]
pub struct MirostatParams {
    pub tau: f32,
    pub eta: f32,
    /// Only used by Mirostat v1; v2 ignores it.
    pub m: Option<i32>,
    /// 1 or 2
    pub version: u8,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            greedy: false,
            temperature: Some(0.8),
            top_k: Some(40),
            top_p: Some(0.95),
            typical_p: None,
            tfs_z: None,
            repetition_penalty: Some(PenaltyParams {
                last_n: 64,
                repeat: 1.1,
                frequency: 0.0,
                presence: 0.0,
            }),
            penalize_newline: false,
            mirostat: None,
            logit_bias: None,
        }
    }
}

impl SamplingParams {
    /// Returns a conflict-free, clamped version of these parameters.
    ///
    /// Precedence:
    /// - `greedy=true` disables temperature/top_k/top_p/typical/tfs/mirostat.
    /// - If Mirostat (v1 or v2) is set, disable top_k/top_p/typical/tfs.
    /// - `typical_p` and `top_p` are mutually exclusive; `typical_p` wins if set.
    ///
    /// Clamps:
    /// - temperature <= 0 → disabled
    /// - top_k < 1 → disabled
    /// - top_p / typical_p / tfs_z ∉ (0, 1] → disabled
    /// - penalties.repeat < 1.0 → clamped to 1.0
    /// - penalties.last_n < 0 → clamped to 0
    pub fn normalized(&self) -> Self {
        let mut p = self.clone();

        // Greedy short-circuit
        if p.greedy {
            p.temperature = None;
            p.top_k = None;
            p.top_p = None;
            p.typical_p = None;
            p.tfs_z = None;
            p.mirostat = None;
            return p;
        }

        // Mirostat overrides classic truncation filters
        if p.mirostat.is_some() {
            p.top_k = None;
            p.top_p = None;
            p.typical_p = None;
            p.tfs_z = None;
        }

        // typical_p vs top_p exclusivity
        if p.typical_p.is_some() {
            p.top_p = None;
        }

        // Clamp/validate simple ranges
        if let Some(t) = p.temperature {
            if t <= 0.0 {
                p.temperature = None;
            }
        }
        if let Some(k) = p.top_k {
            if k < 1 {
                p.top_k = None;
            }
        }
        if let Some(tp) = p.top_p {
            if !(0.0..=1.0).contains(&tp) || tp == 0.0 {
                p.top_p = None;
            }
        }
        if let Some(ty) = p.typical_p {
            if !(0.0..=1.0).contains(&ty) || ty == 0.0 {
                p.typical_p = None;
            }
        }
        if let Some(z) = p.tfs_z {
            if !(0.0..=1.0).contains(&z) || z == 0.0 {
                p.tfs_z = None;
            }
        }

        if let Some(ref mut pen) = p.repetition_penalty {
            if pen.repeat < 1.0 {
                pen.repeat = 1.0;
            }
            if pen.last_n < 0 {
                pen.last_n = 0;
            }
        }

        // Mirostat version sanity – drop invalid config
        if let Some(m) = &p.mirostat {
            if (m.version != 1 && m.version != 2) || m.tau <= 0.0 || m.eta <= 0.0 {
                p.mirostat = None;
            }
        }

        p
    }
}

/// What a backend’s sampler can do. Lets the engine hide unsupported controls
/// and/or downgrade configs at runtime without crashing.
#[derive(Debug, Clone, Copy)]
pub struct BackendSamplingCapabilities {
    pub supports_greedy: bool,
    pub supports_temperature: bool,
    pub supports_top_k: bool,
    pub supports_top_p: bool,
    pub supports_typical_p: bool,
    pub supports_tfs_z: bool,
    pub supports_penalties: bool,
    pub supports_mirostat_v1: bool,
    pub supports_mirostat_v2: bool,
}

impl Default for BackendSamplingCapabilities {
    fn default() -> Self {
        Self {
            supports_greedy: true,
            supports_temperature: true,
            supports_top_k: true,
            supports_top_p: true,
            supports_typical_p: false,
            supports_tfs_z: false,
            supports_penalties: true,
            supports_mirostat_v1: false,
            supports_mirostat_v2: true,
        }
    }
}
