//! AV1 codec support
//!
//! This module provides AV1 video decoding using the dav1d library.

pub mod decoder;

pub use decoder::Av1Decoder;
