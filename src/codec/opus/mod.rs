//! Opus audio codec implementation
//!
//! Opus is a royalty-free, highly versatile audio codec designed for
//! interactive speech and music transmission over the Internet.
//!
//! ## License
//! Opus is covered by the BSD license and is royalty-free.
//! Patent claims have been made available for free use.

#[cfg(feature = "opus-codec")]
pub mod encoder;
#[cfg(feature = "opus-codec")]
pub mod decoder;

#[cfg(feature = "opus-codec")]
pub use encoder::OpusEncoder;
#[cfg(feature = "opus-codec")]
pub use decoder::OpusDecoder;
