//! Placeholder for signed-manifest verification.
//! Keep the API stable so you can drop in Ed25519 later.

use crate::errors::Result;
use crate::types::Manifest;

/// No-op for now; return Ok if JSON parsed.
/// Later: verify detached signature (manifest.json + manifest.sig).
pub fn verify_signed_manifest(_manifest: &Manifest, _maybe_sig: Option<&[u8]>) -> Result<()> {
    Ok(())
}
