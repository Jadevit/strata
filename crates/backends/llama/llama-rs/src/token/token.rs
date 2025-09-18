// strata-backend-llama/src/token.rs

/// Newtype for llama token IDs (matches `llama_token` = i32).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LlamaToken(pub i32);

impl From<i32> for LlamaToken {
    fn from(val: i32) -> Self {
        LlamaToken(val)
    }
}
impl From<LlamaToken> for i32 {
    fn from(token: LlamaToken) -> Self {
        token.0
    }
}
