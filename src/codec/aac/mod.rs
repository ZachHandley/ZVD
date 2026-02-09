//! AAC (Advanced Audio Coding) codec implementation
//!
//! This module provides AAC decoding using Symphonia's AAC codec support.
//! Supported profiles: AAC-LC (Low Complexity), AAC Main, AAC-LTP.
//! HE-AAC v1 (SBR) and HE-AAC v2 (SBR+PS) are NOT supported.
//!
//! ## Patent Notice
//! AAC is covered by patents. Commercial use may require patent licensing from
//! Via Licensing Corporation and other patent holders. See CODEC_LICENSES.md
//! for details.
//!
//! ## Features
//!
//! - **Profiles**: AAC-LC, AAC Main, AAC-LTP (HE-AAC not supported)
//! - **Sample rates**: 8000-96000 Hz
//! - **Channels**: 1-8 (mono through 7.1)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use zvd_lib::codec::{Decoder, AacDecoder};
//! use zvd_lib::format::Packet;
//!
//! // Create decoder with AudioSpecificConfig extradata
//! let asc = [0x12, 0x10]; // AAC-LC, 44100Hz, stereo
//! let mut decoder = AacDecoder::with_extradata(44100, 2, &asc)?;
//!
//! // Or create without extradata
//! let mut decoder = AacDecoder::new(44100, 2)?;
//!
//! // Decode packets
//! decoder.send_packet(&packet)?;
//! let frame = decoder.receive_frame()?;
//! ```
//!
//! ## Factory Functions
//!
//! - `create_decoder(sample_rate, channels)` - Basic decoder creation
//! - `create_decoder_with_extradata(sample_rate, channels, extradata)` - With AudioSpecificConfig

#[cfg(feature = "aac")]
pub mod decoder;

// Encoder module requires fdk-aac-sys which is not currently in Cargo.toml
// To enable, add fdk-aac dependency and aac-encoder feature
// #[cfg(feature = "aac-encoder")]
// pub mod encoder;

#[cfg(feature = "aac")]
pub use decoder::{create_decoder, create_decoder_with_extradata, AacDecoder, AacProfile};

// #[cfg(feature = "aac-encoder")]
// pub use encoder::AacEncoder;
