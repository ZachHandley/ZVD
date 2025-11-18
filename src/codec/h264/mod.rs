//! H.264/AVC codec implementation using OpenH264
//!
//! This module provides H.264 encoding and decoding using Cisco's OpenH264 library.
//! OpenH264 is licensed under BSD 2-Clause, and Cisco provides a patent license for
//! binary usage.
//!
//! ## System Requirements
//!
//! To use H.264 codec support, enable the `h264` feature and ensure
//! libopenh264 is installed on your system:
//! - Debian/Ubuntu: `apt install libopenh264-dev`
//! - macOS: `brew install openh264`
//! - Fedora: `dnf install openh264-devel`
//!
//! ## Features
//!
//! - Industry-standard H.264/AVC video compression
//! - Widely compatible with all modern devices and players
//! - Configurable bitrate and framerate
//! - Keyframe interval control
//! - Efficient encoding and decoding
//!
//! ## Patent Notice
//!
//! H.264/AVC is covered by patents. Cisco's OpenH264 binary plugin is covered by
//! Cisco's patent license. Commercial use may require additional patent licensing.
//! See CODEC_LICENSES.md for details.
//!
//! ## Security Note
//!
//! Always use the latest version of OpenH264 to ensure security patches are applied.
//! Check https://github.com/cisco/openh264/releases for updates.
//!
//! ## Example
//!
//! ```no_run
//! use zvd_lib::codec::h264::{H264Encoder, H264Decoder, H264EncoderConfig};
//! use zvd_lib::codec::{Encoder, Decoder};
//!
//! // Create encoder with custom configuration
//! let config = H264EncoderConfig {
//!     width: 1920,
//!     height: 1080,
//!     bitrate: 5_000_000,
//!     framerate: 30.0,
//!     keyframe_interval: 60,
//! };
//! let encoder = H264Encoder::with_config(config)?;
//!
//! // Create decoder
//! let decoder = H264Decoder::new()?;
//! # Ok::<(), zvd_lib::error::Error>(())
//! ```

pub mod decoder;
pub mod encoder;

pub use decoder::H264Decoder;
pub use encoder::{H264Encoder, H264EncoderConfig};
