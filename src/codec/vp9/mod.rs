//! VP9 video codec implementation
//!
//! VP9 is a royalty-free video codec developed by Google as the successor to VP8.
//! It offers better compression efficiency and is widely used in WebM containers
//! and modern streaming applications.
//!
//! ## System Requirements
//!
//! To use VP9 codec support, enable the `vp9-codec` feature and ensure
//! libvpx is installed on your system:
//! - Debian/Ubuntu: `apt install libvpx-dev`
//! - Arch Linux: `pacman -S libvpx`
//! - macOS: `brew install libvpx`
//! - Fedora: `dnf install libvpx-devel`
//!
//! ## License
//! VP9 is covered by a BSD-style license and is royalty-free.
//! Google has provided a patent license for free use.
//!
//! ## Features
//!
//! VP9 improvements over VP8:
//! - Better compression efficiency (~30-50% bitrate savings)
//! - Support for higher bit depths (10-bit, 12-bit)
//! - Tile-based encoding for parallelism
//! - Lossless compression mode
//! - Multiple chroma subsampling formats
//!
//! ## Example
//!
//! ```no_run
//! use zvd_lib::codec::vp9::{Vp9Encoder, Vp9Decoder, Vp9EncoderConfig};
//! use zvd_lib::codec::{Encoder, Decoder};
//! use zvd_lib::util::Rational;
//!
//! // Create encoder with custom configuration
//! let config = Vp9EncoderConfig {
//!     width: 1920,
//!     height: 1080,
//!     bitrate: 3_000_000,
//!     framerate: Rational::new(30, 1),
//!     keyframe_interval: 120,
//!     cpu_used: 4, // Faster encoding
//!     tile_columns: 2, // Parallel encoding
//!     ..Default::default()
//! };
//! let encoder = Vp9Encoder::with_config(config)?;
//!
//! // Create decoder
//! let decoder = Vp9Decoder::new()?;
//! # Ok::<(), zvd_lib::error::Error>(())
//! ```

pub mod decoder;
pub mod encoder;

pub use decoder::Vp9Decoder;
pub use encoder::{RateControlMode, Vp9Encoder, Vp9EncoderConfig};
