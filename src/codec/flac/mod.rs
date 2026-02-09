//! FLAC audio codec implementation
//!
//! FLAC (Free Lossless Audio Codec) is an open-source lossless audio codec.
//! It's patent-free and royalty-free.
//!
//! ## Features
//!
//! - **Lossless compression**: Perfect reconstruction of original audio
//! - **Sample rates**: 1-655350 Hz (commonly 44100, 48000, 96000)
//! - **Bit depths**: 4-32 bits per sample (commonly 16, 24)
//! - **Channels**: 1-8 channels (mono through 7.1 surround)
//!
//! ## Decoder Usage
//!
//! The decoder uses Symphonia for container-level decoding.
//!
//! ```rust,ignore
//! use zvd_lib::codec::{Decoder, FlacDecoder};
//! use zvd_lib::format::Packet;
//!
//! // Create a decoder for 44.1kHz stereo audio
//! let mut decoder = FlacDecoder::new(44100, 2)?;
//!
//! // Decode packets
//! decoder.send_packet(&packet)?;
//! let frame = decoder.receive_frame()?;
//! ```
//!
//! ## Encoder Usage
//!
//! The encoder uses flacenc (pure Rust) and requires the `flac-encoder` feature.
//!
//! ```rust,ignore
//! use zvd_lib::codec::{Encoder, FlacEncoder};
//!
//! // Create an encoder for CD quality audio
//! let mut encoder = FlacEncoder::new(44100, 2, 16)?;
//!
//! // Set compression level (0-8, default is 5)
//! encoder.set_compression_level(5)?;
//!
//! // Encode frames
//! encoder.send_frame(&frame)?;
//! let packet = encoder.receive_packet()?;
//! ```
//!
//! ## Factory Functions
//!
//! - `create_decoder(sample_rate, channels)` - Basic decoder creation
//! - `create_decoder_with_extradata(sample_rate, channels, extradata)` - With STREAMINFO
//! - `create_encoder(sample_rate, channels, bits_per_sample)` - Encoder creation (requires `flac-encoder` feature)

pub mod decoder;

#[cfg(feature = "flac-encoder")]
pub mod encoder;

pub use decoder::{create_decoder, create_decoder_with_extradata, FlacDecoder, FlacStreamInfo};

#[cfg(feature = "flac-encoder")]
pub use encoder::{FlacEncoder, FlacEncoderConfig};

/// Create a FLAC encoder with default settings
///
/// # Arguments
/// * `sample_rate` - Audio sample rate in Hz
/// * `channels` - Number of audio channels (1-8)
/// * `bits_per_sample` - Bits per sample (8, 16, 24, or 32)
///
/// # Example
/// ```rust,ignore
/// let encoder = flac::create_encoder(44100, 2, 16)?;
/// ```
#[cfg(feature = "flac-encoder")]
pub fn create_encoder(
    sample_rate: u32,
    channels: u16,
    bits_per_sample: u8,
) -> crate::error::Result<FlacEncoder> {
    FlacEncoder::new(sample_rate, channels, bits_per_sample)
}
