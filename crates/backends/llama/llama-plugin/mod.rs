// src/mod.rs

pub mod backend; // Plugin backend adapter
pub mod metadata; // Plugin metadata provider

// Internal modules migrated from llama-rs
pub mod backends;
pub mod batch;
pub mod cache;
pub mod context;
pub mod debug;
pub mod ffi;
pub mod metadata_impl {
    pub mod metadata;
}
pub mod model;
pub mod params;
pub mod sampling;
pub mod token;
