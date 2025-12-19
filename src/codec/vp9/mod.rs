//! VP9 video codec implementation
//!
//! VP9 is a royalty-free video codec developed by Google as the successor to VP8.
//! It offers better compression efficiency and is widely used in WebM containers
//! and modern streaming applications.
//!
//! ## License
//! VP9 is covered by a BSD-style license and is royalty-free.
//! Google has provided a patent license for free use.

#[cfg(feature = "vp9-codec")]
pub mod decoder;
#[cfg(feature = "vp9-codec")]
pub mod encoder;

#[cfg(feature = "vp9-codec")]
pub use decoder::Vp9Decoder;
#[cfg(feature = "vp9-codec")]
pub use encoder::Vp9Encoder;
