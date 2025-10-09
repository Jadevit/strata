//! Safe sampler wrapper: builds a llama.cpp sampler chain and returns one token.

use crate::context::LlamaContext;
use crate::params::SamplingParams as LlamaSamplingParams;
use crate::token::LlamaToken;

/// Sample one token using the provided params.
/// This is a **safe** wrapper; all unsafe calls stay internal to `llama_rs`.
pub fn sample_with_params(
    ctx: &LlamaContext,
    vocab_size: usize,
    params: &LlamaSamplingParams,
) -> Result<LlamaToken, String> {
    unsafe { sample_with_params_inner(ctx, vocab_size, params) }
}

unsafe fn sample_with_params_inner(
    ctx: &LlamaContext,
    vocab_size: usize,
    params: &LlamaSamplingParams,
) -> Result<LlamaToken, String> {
    use llama_sys::*;
    use std::ptr::NonNull;

    let chain_params = llama_sampler_chain_default_params();
    let smpl = NonNull::new(llama_sampler_chain_init(chain_params))
        .ok_or_else(|| "llama_sampler_chain_init returned null".to_string())?;
    let sp = smpl.as_ptr();

    // Truncation / temperature
    if let Some(k) = params.top_k {
        if k > 0 {
            llama_sampler_chain_add(sp, llama_sampler_init_top_k(k as i32));
        }
    }
    if let Some(p) = params.top_p {
        if p > 0.0 && p <= 1.0 {
            llama_sampler_chain_add(sp, llama_sampler_init_top_p(p as f32, 1));
        }
    }
    if let Some(t) = params.temperature {
        if t > 0.0 {
            llama_sampler_chain_add(sp, llama_sampler_init_temp(t as f32));
        }
    }

    // Penalties
    if let Some(pen) = &params.penalties {
        llama_sampler_chain_add(
            sp,
            llama_sampler_init_penalties(pen.last_n, pen.repeat, pen.freq, pen.presence),
        );
    }

    // Mirostat
    if let Some(m1) = &params.mirostat {
        llama_sampler_chain_add(
            sp,
            llama_sampler_init_mirostat(vocab_size as i32, m1.seed, m1.tau, m1.eta, m1.m),
        );
    }
    if let Some(m2) = &params.mirostat_v2 {
        llama_sampler_chain_add(sp, llama_sampler_init_mirostat_v2(m2.seed, m2.tau, m2.eta));
    }

    // Terminal selector
    if params.greedy {
        llama_sampler_chain_add(sp, llama_sampler_init_top_k(1));
    } else {
        // 0 = use current logits distribution
        llama_sampler_chain_add(sp, llama_sampler_init_dist(0));
    }

    let tok_id = llama_sampler_sample(sp, ctx.as_ptr(), -1);
    llama_sampler_free(sp);

    if tok_id < 0 {
        return Err(format!("sampler returned invalid token id {}", tok_id));
    }
    Ok(LlamaToken(tok_id))
}
