// llama-plugin/src/ffi/sampling.rs
//
// Safe-ish wrappers around llama_sys sampling API (llama_sampler_* chain).
// The public API here is still `unsafe` — high-level wrappers handle safety.

use llama_sys::*;
use std::ptr::NonNull;

pub unsafe fn sample_token(
    ctx: *mut llama_context,
    vocab_size: usize,
    params: &crate::params::SamplingParams,
) -> Result<i32, String> {
    let chain_params = llama_sampler_chain_default_params();
    let chain = NonNull::new(llama_sampler_chain_init(chain_params))
        .ok_or_else(|| "llama_sampler_chain_init returned null".to_string())?;
    let sp = chain.as_ptr();

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
        llama_sampler_chain_add(sp, llama_sampler_init_dist(0));
    }

    let tok_id = llama_sampler_sample(sp, ctx, -1);
    llama_sampler_free(sp);

    if tok_id < 0 {
        Err(format!("sampler returned invalid token id {tok_id}"))
    } else {
        Ok(tok_id)
    }
}
