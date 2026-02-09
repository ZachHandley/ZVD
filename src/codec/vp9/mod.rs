//! VP9 video codec implementation
//!
//! VP9 is a royalty-free video codec developed by Google as the successor to VP8.
//! It offers better compression efficiency and is widely used in WebM containers
//! and modern streaming applications.
//!
//! ## Features
//!
//! - Enable `vp9-codec` for pure Rust VP9 support (decoder only currently)
//! - Enable `vp9-libvpx` for full VP9 encode/decode via libvpx
//!
//! ## VP9 Profiles
//!
//! VP9 supports four profiles:
//! - Profile 0: 8-bit, YUV 4:2:0 (most common)
//! - Profile 1: 8-bit, YUV 4:2:2, 4:4:0, 4:4:4
//! - Profile 2: 10/12-bit, YUV 4:2:0
//! - Profile 3: 10/12-bit, YUV 4:2:2, 4:4:0, 4:4:4
//!
//! ## License
//! VP9 is covered by a BSD-style license and is royalty-free.
//! Google has provided a patent license for free use.

#[cfg(feature = "vp9-codec")]
pub mod decoder;
#[cfg(feature = "vp9-codec")]
pub mod encoder;

#[cfg(feature = "vp9-codec")]
pub use decoder::{Vp9Decoder, Vp9Profile as Vp9DecoderProfile};
#[cfg(feature = "vp9-codec")]
pub use encoder::{
    Vp9AqMode, Vp9Encoder, Vp9EncoderConfig, Vp9EncodingPass, Vp9Profile, Vp9RateControl,
    Vp9TuneContent,
};
