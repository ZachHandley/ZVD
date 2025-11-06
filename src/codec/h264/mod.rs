//! H.264/AVC codec implementation using OpenH264
//!
//! This module provides H.264 encoding and decoding using Cisco's OpenH264 library.
//! OpenH264 is licensed under BSD 2-Clause, and Cisco provides a patent license for
//! binary usage.
//!
//! ## Patent Notice
//! H.264/AVC is covered by patents. Cisco's OpenH264 binary plugin is covered by
//! Cisco's patent license. Commercial use may require additional patent licensing.
//! See CODEC_LICENSES.md for details.

#[cfg(feature = "h264")]
pub mod encoder;
#[cfg(feature = "h264")]
pub mod decoder;

#[cfg(feature = "h264")]
pub use encoder::H264Encoder;
#[cfg(feature = "h264")]
pub use decoder::H264Decoder;
