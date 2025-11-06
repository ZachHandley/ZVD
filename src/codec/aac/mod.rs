//! AAC (Advanced Audio Coding) codec implementation
//!
//! This module provides AAC decoding using Symphonia's AAC codec support.
//!
//! ## Patent Notice
//! AAC is covered by patents. Commercial use may require patent licensing from
//! Via Licensing Corporation and other patent holders. See CODEC_LICENSES.md
//! for details.

#[cfg(feature = "aac")]
pub mod decoder;

#[cfg(feature = "aac")]
pub use decoder::AacDecoder;
