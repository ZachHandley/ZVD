//! AV1 codec support
//!
//! This module provides AV1 video encoding and decoding.
//! - Decoding: dav1d (C library bindings)
//! - Encoding: rav1e (pure Rust)

pub mod decoder;
pub mod encoder;

pub use decoder::Av1Decoder;
pub use encoder::Av1Encoder;
