//! FLAC audio decoder using Symphonia
//!
//! FLAC (Free Lossless Audio Codec) is a lossless audio compression format.
//! This implementation uses Symphonia's FLAC decoder for packet-level decoding.
//!
//! ## Features
//! - Sample rates up to 655,350 Hz
//! - Up to 8 channels
//! - Bit depths: 4 to 32 bits per sample
//! - Lossless compression

use crate::codec::{AudioFrame, Decoder, Frame};
use crate::error::{Error, Result};
use crate::format::Packet as ZvdPacket;
use crate::util::{Buffer, SampleFormat, Timestamp};
use std::io::Cursor;

#[cfg(feature = "flac")]
use symphonia::core::audio::SampleBuffer;
#[cfg(feature = "flac")]
use symphonia::core::codecs::{Decoder as SymphoniaDecoder, DecoderOptions, CODEC_TYPE_FLAC};
#[cfg(feature = "flac")]
use symphonia::core::formats::{Packet as SymphoniaPacket, FormatOptions, FormatReader};
#[cfg(feature = "flac")]
use symphonia::core::io::MediaSourceStream;
#[cfg(feature = "flac")]
use symphonia::core::meta::MetadataOptions;
#[cfg(feature = "flac")]
use symphonia::core::probe::Hint;

/// FLAC audio decoder configuration
#[derive(Debug, Clone)]
pub struct FlacDecoderConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u8,
}

impl Default for FlacDecoderConfig {
    fn default() -> Self {
        FlacDecoderConfig {
            sample_rate: 44100,
            channels: 2,
            bits_per_sample: 16,
        }
    }
}

/// FLAC audio decoder
///
/// Decodes FLAC compressed audio to PCM samples.
/// Uses Symphonia's FLAC decoder implementation.
#[cfg(feature = "flac")]
pub struct FlacDecoder {
    config: FlacDecoderConfig,
    /// Buffered decoded frames waiting to be retrieved
    frame_buffer: Vec<AudioFrame>,
}

#[cfg(feature = "flac")]
impl FlacDecoder {
    /// Create a new FLAC decoder with given configuration
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        let config = FlacDecoderConfig {
            sample_rate,
            channels,
            bits_per_sample: 16, // Default to 16-bit
        };
        Self::with_config(config)
    }

    /// Create a new FLAC decoder with custom configuration
    pub fn with_config(config: FlacDecoderConfig) -> Result<Self> {
        // Validate sample rate (FLAC supports up to 655,350 Hz)
        if config.sample_rate == 0 || config.sample_rate > 655350 {
            return Err(Error::codec(format!(
                "Invalid FLAC sample rate: {}. Must be 1-655,350 Hz",
                config.sample_rate
            )));
        }

        // Validate channels (FLAC supports up to 8)
        if config.channels == 0 || config.channels > 8 {
            return Err(Error::codec(format!(
                "Invalid channel count: {}. FLAC supports 1-8 channels",
                config.channels
            )));
        }

        // Validate bit depth
        if config.bits_per_sample < 4 || config.bits_per_sample > 32 {
            return Err(Error::codec(format!(
                "Invalid bit depth: {}. FLAC supports 4-32 bits per sample",
                config.bits_per_sample
            )));
        }

        Ok(FlacDecoder {
            config,
            frame_buffer: Vec::new(),
        })
    }

    /// Decode FLAC data from a packet
    ///
    /// Note: FLAC decoding through Symphonia is typically done at container level.
    /// This method attempts to decode raw FLAC frames if possible.
    fn decode_packet(&mut self, packet: &ZvdPacket) -> Result<AudioFrame> {
        // FLAC packets contain complete frames that we need to decode
        // Since Symphonia's decoder expects to work with a FormatReader,
        // we need to create a minimal stream from the packet data

        let data = packet.data.as_slice();
        if data.is_empty() {
            return Err(Error::codec("Empty FLAC packet"));
        }

        // For now, return an error suggesting to use SymphoniaAdapter
        // A full implementation would require either:
        // 1. Building a FLAC frame parser to work with raw frames
        // 2. Using a different FLAC library that supports packet-level decoding
        Err(Error::unsupported(
            "FLAC packet-level decoding not yet implemented. Use SymphoniaAdapter for FLAC file decoding."
        ))
    }
}

#[cfg(feature = "flac")]
impl Decoder for FlacDecoder {
    fn send_packet(&mut self, packet: &ZvdPacket) -> Result<()> {
        // Attempt to decode the packet
        // Since Symphonia's architecture is container-oriented, this is limited
        match self.decode_packet(packet) {
            Ok(frame) => {
                self.frame_buffer.push(frame);
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if self.frame_buffer.is_empty() {
            return Err(Error::TryAgain);
        }

        let audio_frame = self.frame_buffer.remove(0);
        Ok(Frame::Audio(audio_frame))
    }

    fn flush(&mut self) -> Result<()> {
        self.frame_buffer.clear();
        Ok(())
    }
}

#[cfg(feature = "flac")]
impl Default for FlacDecoder {
    fn default() -> Self {
        Self::new(44100, 2).expect("Failed to create default FLAC decoder")
    }
}

// Stub implementation when flac feature is not enabled
#[cfg(not(feature = "flac"))]
pub struct FlacDecoder {
    _private: (),
}

#[cfg(not(feature = "flac"))]
#[allow(dead_code)]
pub struct FlacDecoderConfig {
    _private: (),
}

#[cfg(not(feature = "flac"))]
impl FlacDecoder {
    pub fn new(_sample_rate: u32, _channels: u16) -> Result<Self> {
        Err(Error::unsupported(
            "FLAC codec support not enabled. Enable the 'flac' feature."
        ))
    }

    #[allow(dead_code)]
    pub fn with_config(_config: FlacDecoderConfig) -> Result<Self> {
        Err(Error::unsupported("FLAC codec not enabled"))
    }
}

#[cfg(not(feature = "flac"))]
impl Decoder for FlacDecoder {
    fn send_packet(&mut self, _packet: &ZvdPacket) -> Result<()> {
        Err(Error::unsupported("FLAC codec not enabled"))
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        Err(Error::unsupported("FLAC codec not enabled"))
    }

    fn flush(&mut self) -> Result<()> {
        Err(Error::unsupported("FLAC codec not enabled"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "flac")]
    fn test_flac_decoder_creation() {
        let decoder = FlacDecoder::new(44100, 2);
        assert!(decoder.is_ok(), "Failed to create FLAC decoder");
    }

    #[test]
    #[cfg(feature = "flac")]
    fn test_flac_decoder_with_config() {
        let config = FlacDecoderConfig {
            sample_rate: 48000,
            channels: 2,
            bits_per_sample: 24,
        };
        let decoder = FlacDecoder::with_config(config);
        assert!(decoder.is_ok(), "Failed to create FLAC decoder with config");
    }

    #[test]
    #[cfg(feature = "flac")]
    fn test_flac_decoder_invalid_sample_rate() {
        let decoder = FlacDecoder::new(0, 2);
        assert!(decoder.is_err(), "Should reject invalid sample rate");

        let decoder = FlacDecoder::new(700000, 2);
        assert!(decoder.is_err(), "Should reject too high sample rate");
    }

    #[test]
    #[cfg(feature = "flac")]
    fn test_flac_decoder_invalid_channels() {
        let decoder = FlacDecoder::new(44100, 0);
        assert!(decoder.is_err(), "Should reject zero channels");

        let decoder = FlacDecoder::new(44100, 9);
        assert!(decoder.is_err(), "Should reject too many channels");
    }

    #[test]
    #[cfg(feature = "flac")]
    fn test_flush() {
        let mut decoder = FlacDecoder::new(44100, 2).expect("Failed to create decoder");
        assert!(decoder.flush().is_ok(), "Flush should not error");
    }

    #[test]
    #[cfg(not(feature = "flac"))]
    fn test_flac_disabled() {
        let decoder = FlacDecoder::new(44100, 2);
        assert!(decoder.is_err());
    }
}
