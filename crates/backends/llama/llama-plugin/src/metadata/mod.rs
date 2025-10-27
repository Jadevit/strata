// Public surface for llama-plugin metadata (no unsafe here).

mod provider;
mod scrape;

pub use provider::LlamaMetadataProvider;
pub use scrape::{can_handle, scrape_metadata, LlamaScrape};
