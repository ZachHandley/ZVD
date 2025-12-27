//! AAC (Advanced Audio Coding) codec implementation
//!
//! This module provides AAC decoding using Symphonia's AAC codec support.
//! Supported profiles: AAC-LC (Low Complexity), AAC Main, AAC-LTP.
//! HE-AAC v1 (SBR) and HE-AAC v2 (SBR+PS) are NOT supported.
//!
//! ## Patent Notice
//! AAC is covered by patents. Commercial use may require patent licensing from
//! Via Licensing Corporation and other patent holders. See CODEC_LICENSES.md
//! for details.

#[cfg(feature = "aac")]
pub mod decoder;

#[cfg(feature = "aac")]
pub use decoder::{AacDecoder, AacProfile};
