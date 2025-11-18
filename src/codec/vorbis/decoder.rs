//! Vorbis audio decoder using Symphonia
//!
//! Vorbis is a free, open-source lossy audio codec.
//! This implementation uses Symphonia's Vorbis decoder.
//!
//! ## Features
//! - Sample rates: 8 kHz to 192 kHz
//! - Channels: Up to 255 channels (typically 1-8)
//! - Bitrates: Variable (VBR) or constant (CBR)
//! - Quality-based encoding
//!
//! ## Important: Use SymphoniaAdapter
//! Vorbis decoding in ZVD should use `SymphoniaAdapter` for complete
//! container-level decoding with Ogg container support.

use crate::codec::{AudioFrame, Decoder, Frame};
use crate::error::{Error, Result};
use crate::format::Packet as ZvdPacket;

/// Vorbis decoder configuration
#[derive(Debug, Clone)]
pub struct VorbisDecoderConfig {
    pub sample_rate: u32,
    pub channels: u16,
}

impl Default for VorbisDecoderConfig {
    fn default() -> Self {
        VorbisDecoderConfig {
            sample_rate: 44100,
            channels: 2,
        }
    }
}

/// Vorbis audio decoder
///
/// Note: Vorbis decoding in ZVD is done at container level via SymphoniaAdapter.
/// This struct provides the interface but directs users to the adapter.
#[cfg(feature = "vorbis")]
pub struct VorbisDecoder {
    config: VorbisDecoderConfig,
    frame_buffer: Vec<AudioFrame>,
}

#[cfg(feature = "vorbis")]
impl VorbisDecoder {
    /// Create a new Vorbis decoder
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        let config = VorbisDecoderConfig {
            sample_rate,
            channels,
        };
        Self::with_config(config)
    }

    /// Create decoder with configuration
    pub fn with_config(config: VorbisDecoderConfig) -> Result<Self> {
        // Validate sample rate
        if config.sample_rate == 0 || config.sample_rate > 192000 {
            return Err(Error::codec(format!(
                "Invalid Vorbis sample rate: {}. Must be 1-192,000 Hz",
                config.sample_rate
            )));
        }

        // Validate channels
        if config.channels == 0 || config.channels > 255 {
            return Err(Error::codec(format!(
                "Invalid channel count: {}. Vorbis supports 1-255 channels",
                config.channels
            )));
        }

        Ok(VorbisDecoder {
            config,
            frame_buffer: Vec::new(),
        })
    }
}

#[cfg(feature = "vorbis")]
impl Decoder for VorbisDecoder {
    fn send_packet(&mut self, _packet: &ZvdPacket) -> Result<()> {
        Err(Error::unsupported(
            "Vorbis packet-level decoding not implemented. Use SymphoniaAdapter for Ogg Vorbis file decoding."
        ))
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

#[cfg(feature = "vorbis")]
impl Default for VorbisDecoder {
    fn default() -> Self {
        Self::new(44100, 2).expect("Failed to create default Vorbis decoder")
    }
}

// Stub when feature disabled
#[cfg(not(feature = "vorbis"))]
pub struct VorbisDecoder {
    _private: (),
}

#[cfg(not(feature = "vorbis"))]
#[allow(dead_code)]
pub struct VorbisDecoderConfig {
    _private: (),
}

#[cfg(not(feature = "vorbis"))]
impl VorbisDecoder {
    pub fn new(_sample_rate: u32, _channels: u16) -> Result<Self> {
        Err(Error::unsupported(
            "Vorbis codec support not enabled. Enable the 'vorbis' feature."
        ))
    }

    #[allow(dead_code)]
    pub fn with_config(_config: VorbisDecoderConfig) -> Result<Self> {
        Err(Error::unsupported("Vorbis codec not enabled"))
    }
}

#[cfg(not(feature = "vorbis"))]
impl Decoder for VorbisDecoder {
    fn send_packet(&mut self, _packet: &ZvdPacket) -> Result<()> {
        Err(Error::unsupported("Vorbis codec not enabled"))
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        Err(Error::unsupported("Vorbis codec not enabled"))
    }

    fn flush(&mut self) -> Result<()> {
        Err(Error::unsupported("Vorbis codec not enabled"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "vorbis")]
    fn test_vorbis_decoder_creation() {
        let decoder = VorbisDecoder::new(44100, 2);
        assert!(decoder.is_ok(), "Failed to create Vorbis decoder");
    }

    #[test]
    #[cfg(feature = "vorbis")]
    fn test_vorbis_decoder_with_config() {
        let config = VorbisDecoderConfig {
            sample_rate: 48000,
            channels: 6,
        };
        let decoder = VorbisDecoder::with_config(config);
        assert!(decoder.is_ok());
    }

    #[test]
    #[cfg(feature = "vorbis")]
    fn test_flush() {
        let mut decoder = VorbisDecoder::new(44100, 2).expect("Failed to create decoder");
        assert!(decoder.flush().is_ok());
    }

    #[test]
    #[cfg(not(feature = "vorbis"))]
    fn test_vorbis_disabled() {
        let decoder = VorbisDecoder::new(44100, 2);
        assert!(decoder.is_err());
    }
}
