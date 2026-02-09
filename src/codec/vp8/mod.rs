//! VP8 video codec implementation
//!
//! VP8 is a royalty-free video codec developed by On2 Technologies and
//! released as open source by Google. It's widely used in WebM containers
//! and WebRTC.
//!
//! ## Features
//!
//! - Enable `vp8-codec` for VP8 support
//! - Enable `vp8-libvpx` for full VP8 encode/decode via libvpx
//!
//! ## License
//! VP8 is covered by a BSD-style license and is royalty-free.
//! Google has released all patents for free use.

#[cfg(feature = "vp8-codec")]
pub mod decoder;
#[cfg(feature = "vp8-codec")]
pub mod encoder;

#[cfg(feature = "vp8-codec")]
pub use decoder::Vp8Decoder;
#[cfg(feature = "vp8-codec")]
pub use encoder::{Vp8Encoder, Vp8EncoderConfig, Vp8RateControl};
