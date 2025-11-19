//! VP8 video codec implementation
//!
//! VP8 is a royalty-free video codec developed by On2 Technologies and
//! released as open source by Google. It's widely used in WebM containers
//! and WebRTC.
//!
//! ## System Requirements
//!
//! To use VP8 codec support, enable the `vp8-codec` feature and ensure
//! libvpx is installed on your system:
//! - Debian/Ubuntu: `apt install libvpx-dev`
//! - Arch Linux: `pacman -S libvpx`
//! - macOS: `brew install libvpx`
//! - Fedora: `dnf install libvpx-devel`
//!
//! ## License
//! VP8 is covered by a BSD-style license and is royalty-free.
//! Google has released all patents for free use.
//!
//! ## Example
//!
//! ```no_run
//! use zvd_lib::codec::vp8::{Vp8Encoder, Vp8Decoder};
//! use zvd_lib::codec::{Encoder, Decoder};
//!
//! // Create encoder
//! let encoder = Vp8Encoder::new(1920, 1080)?;
//!
//! // Create decoder
//! let decoder = Vp8Decoder::new()?;
//! # Ok::<(), zvd_lib::error::Error>(())
//! ```

pub mod decoder;
pub mod encoder;

pub use decoder::Vp8Decoder;
pub use encoder::{RateControlMode, Vp8Encoder, Vp8EncoderConfig};
