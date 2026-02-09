//! Vorbis audio codec implementation
//!
//! Vorbis is a free, open-source audio codec managed by the Xiph.Org Foundation.
//! It's patent-free and royalty-free.
//!
//! ## Features
//!
//! - **Lossy compression**: High quality at low bitrates
//! - **Sample rates**: 8000-192000 Hz
//! - **Channels**: 1-255 channels (mono through 7.1 and beyond)
//! - **Bitrates**: 45-500 kbps
//!
//! ## Decoder
//!
//! The decoder uses Symphonia for container-level decoding.
//!
//! ### Header Requirements
//!
//! Vorbis streams require three header packets before audio decoding:
//! 1. Identification header (packet type 0x01)
//! 2. Comment header (packet type 0x03)
//! 3. Setup header (packet type 0x05)
//!
//! These can be provided via extradata or sent as the first packets.
//!
//! ### Decoder Usage
//!
//! ```rust,ignore
//! use zvd_lib::codec::{Decoder, VorbisDecoder};
//! use zvd_lib::format::Packet;
//!
//! // Create a decoder with extradata containing headers
//! let mut decoder = VorbisDecoder::with_extradata(44100, 2, &headers)?;
//!
//! // Or create without headers and send them first
//! let mut decoder = VorbisDecoder::new(44100, 2)?;
//!
//! // Decode packets
//! decoder.send_packet(&packet)?;
//! let frame = decoder.receive_frame()?;
//! ```
//!
//! ## Encoder
//!
//! The encoder uses vorbis_rs (bindings to aoTuV-patched libvorbis) and requires
//! the `vorbis-encoder` feature.
//!
//! ### Encoder Usage
//!
//! ```rust,ignore
//! use zvd_lib::codec::{Encoder, VorbisEncoder};
//!
//! // Quality-based encoding (-0.1 to 1.0)
//! let mut encoder = VorbisEncoder::new_quality(44100, 2, 0.5)?;
//!
//! // Or bitrate-based encoding
//! let mut encoder = VorbisEncoder::new_bitrate(44100, 2, 128000)?;
//!
//! // Encode frames
//! encoder.send_frame(&frame)?;
//! let packet = encoder.receive_packet()?;
//! ```
//!
//! ## Factory Functions
//!
//! - `create_decoder(sample_rate, channels)` - Basic decoder creation
//! - `create_decoder_with_extradata(sample_rate, channels, extradata)` - With Vorbis headers
//! - `create_encoder(sample_rate, channels, quality)` - Encoder creation (requires `vorbis-encoder` feature)

pub mod decoder;

#[cfg(feature = "vorbis-encoder")]
pub mod encoder;

pub use decoder::{
    create_decoder, create_decoder_with_extradata, VorbisDecoder, VorbisIdHeader,
};

#[cfg(feature = "vorbis-encoder")]
pub use encoder::{VorbisEncoder, VorbisEncoderConfig, VorbisEncodingMode};

/// Create a Vorbis encoder with quality-based encoding
///
/// # Arguments
/// * `sample_rate` - Audio sample rate in Hz
/// * `channels` - Number of audio channels
/// * `quality` - Quality factor (-0.1 to 1.0)
///   - -0.1: ~45 kbps, very small files
///   - 0.0: ~64 kbps, acceptable quality
///   - 0.5: ~128 kbps, good quality (recommended)
///   - 1.0: ~256+ kbps, transparent quality
///
/// # Example
/// ```rust,ignore
/// let encoder = vorbis::create_encoder(44100, 2, 0.5)?;
/// ```
#[cfg(feature = "vorbis-encoder")]
pub fn create_encoder(
    sample_rate: u32,
    channels: u16,
    quality: f32,
) -> crate::error::Result<VorbisEncoder> {
    VorbisEncoder::new_quality(sample_rate, channels, quality)
}

/// Create a Vorbis encoder with bitrate-based encoding (ABR)
///
/// # Arguments
/// * `sample_rate` - Audio sample rate in Hz
/// * `channels` - Number of audio channels
/// * `bitrate` - Target bitrate in bits per second
///
/// # Example
/// ```rust,ignore
/// let encoder = vorbis::create_encoder_bitrate(44100, 2, 128000)?;
/// ```
#[cfg(feature = "vorbis-encoder")]
pub fn create_encoder_bitrate(
    sample_rate: u32,
    channels: u16,
    bitrate: u32,
) -> crate::error::Result<VorbisEncoder> {
    VorbisEncoder::new_bitrate(sample_rate, channels, bitrate)
}
