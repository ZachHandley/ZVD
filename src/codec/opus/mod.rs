//! Opus audio codec implementation
//!
//! Opus is a royalty-free, highly versatile audio codec designed for
//! interactive speech and music transmission over the Internet.
//!
//! ## Features
//!
//! - Sample rates: 8kHz to 48kHz (narrowband to fullband)
//! - Bitrates: 6 kbps to 510 kbps
//! - Low latency: 2.5ms to 60ms frame sizes
//! - Optimized for both speech (VoIP) and music (Audio)
//! - Robust packet loss concealment
//! - Bandwidth adaptation
//!
//! ## License
//! Opus is covered by the BSD license and is royalty-free.
//! Patent claims have been made available for free use.
//!
//! ## Example
//!
//! ```no_run
//! use zvd_lib::codec::opus::{OpusEncoder, OpusDecoder, OpusEncoderConfig, OpusApplication};
//! use zvd_lib::codec::{Encoder, Decoder};
//!
//! // Create encoder for music
//! let config = OpusEncoderConfig {
//!     sample_rate: 48000,
//!     channels: 2,
//!     application: OpusApplication::Audio,
//!     bitrate: Some(128000),
//!     complexity: None,
//! };
//! let encoder = OpusEncoder::with_config(config)?;
//!
//! // Create decoder
//! let decoder = OpusDecoder::new(48000, 2)?;
//! # Ok::<(), zvd_lib::error::Error>(())
//! ```

pub mod decoder;
pub mod encoder;

pub use decoder::OpusDecoder;
pub use encoder::{OpusApplication, OpusEncoder, OpusEncoderConfig};
