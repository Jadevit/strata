// llama-plugin/src/sampling.rs
//
// Safe wrapper over ffi::sampling to produce a LlamaToken.

use crate::{
    context::LlamaContext, ffi::sampling as sffi, params::SamplingParams, token::LlamaToken,
};

pub fn sample_with_params(
    ctx: &LlamaContext,
    vocab_size: usize,
    params: &SamplingParams,
) -> Result<LlamaToken, String> {
    let tok_id = unsafe { sffi::sample_token(ctx.as_ptr(), vocab_size, params)? };
    Ok(LlamaToken(tok_id))
}
