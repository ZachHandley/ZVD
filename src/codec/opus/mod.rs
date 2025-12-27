//! Opus audio codec implementation
//!
//! Opus is a royalty-free, highly versatile audio codec designed for
//! interactive speech and music transmission over the Internet.
//!
//! ## Features
//!
//! - **Sample rates**: 8000, 12000, 16000, 24000, 48000 Hz
//! - **Channels**: Mono (1) or Stereo (2)
//! - **Bitrates**: 6 kbps to 510 kbps
//! - **Frame sizes**: 2.5, 5, 10, 20, 40, 60 ms
//! - **Packet loss concealment (PLC)**: Automatically generates audio during packet loss
//! - **Forward error correction (FEC)**: Recovers lost packets from subsequent packets
//!
//! ## Example Usage
//!
//! ```no_run
//! use zvd_lib::codec::opus::{OpusEncoder, OpusDecoder, OpusEncoderConfig};
//!
//! // Create encoder for voice over IP
//! let config = OpusEncoderConfig::voice(16000, 1);
//! let mut encoder = OpusEncoder::with_config(config).unwrap();
//!
//! // Create decoder
//! let mut decoder = OpusDecoder::new(16000, 1).unwrap();
//! ```
//!
//! ## License
//!
//! Opus is covered by the BSD license and is royalty-free.
//! Patent claims have been made available for free use.

#[cfg(feature = "opus-codec")]
pub mod decoder;
#[cfg(feature = "opus-codec")]
pub mod encoder;

#[cfg(feature = "opus-codec")]
pub use decoder::{OpusDecoder, OpusDecoderConfig};
#[cfg(feature = "opus-codec")]
pub use encoder::{OpusEncoder, OpusEncoderConfig, OpusFrameDuration, VALID_SAMPLE_RATES};
