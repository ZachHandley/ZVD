//! MP3 audio decoder using Symphonia
//!
//! MP3 (MPEG-1/2 Audio Layer III) is a widely-used lossy audio codec.
//! This implementation uses Symphonia's MP3 decoder.
//!
//! ## Features
//! - Sample rates: 8, 11.025, 12, 16, 22.05, 24, 32, 44.1, 48 kHz
//! - Channels: Mono, Stereo, Joint Stereo
//! - Bitrates: 8 to 320 kbps
//! - CBR and VBR support
//!
//! ## Patent Status
//! MP3 patents have expired worldwide as of 2017, making it free to use.
//!
//! ## Important: Use SymphoniaAdapter
//! MP3 decoding in ZVD should use `SymphoniaAdapter` for complete
//! container-level decoding with ID3 tag support.

use crate::codec::{AudioFrame, Decoder, Frame};
use crate::error::{Error, Result};
use crate::format::Packet as ZvdPacket;

/// MP3 decoder configuration
#[derive(Debug, Clone)]
pub struct Mp3DecoderConfig {
    pub sample_rate: u32,
    pub channels: u16,
}

impl Default for Mp3DecoderConfig {
    fn default() -> Self {
        Mp3DecoderConfig {
            sample_rate: 44100,
            channels: 2,
        }
    }
}

/// MP3 audio decoder
///
/// Note: MP3 decoding in ZVD is done at container level via SymphoniaAdapter.
/// This struct provides the interface but directs users to the adapter.
#[cfg(feature = "mp3")]
pub struct Mp3Decoder {
    config: Mp3DecoderConfig,
    frame_buffer: Vec<AudioFrame>,
}

#[cfg(feature = "mp3")]
impl Mp3Decoder {
    /// Create a new MP3 decoder
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        let config = Mp3DecoderConfig {
            sample_rate,
            channels,
        };
        Self::with_config(config)
    }

    /// Create decoder with configuration
    pub fn with_config(config: Mp3DecoderConfig) -> Result<Self> {
        // Validate sample rate (MP3 standard rates)
        const VALID_RATES: &[u32] = &[8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000];
        if !VALID_RATES.contains(&config.sample_rate) {
            return Err(Error::codec(format!(
                "Invalid MP3 sample rate: {}. Must be one of: {:?}",
                config.sample_rate, VALID_RATES
            )));
        }

        // Validate channels
        if config.channels == 0 || config.channels > 2 {
            return Err(Error::codec(format!(
                "Invalid channel count: {}. MP3 supports mono (1) or stereo (2)",
                config.channels
            )));
        }

        Ok(Mp3Decoder {
            config,
            frame_buffer: Vec::new(),
        })
    }
}

#[cfg(feature = "mp3")]
impl Decoder for Mp3Decoder {
    fn send_packet(&mut self, _packet: &ZvdPacket) -> Result<()> {
        Err(Error::unsupported(
            "MP3 packet-level decoding not implemented. Use SymphoniaAdapter for MP3 file decoding with ID3 support."
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

#[cfg(feature = "mp3")]
impl Default for Mp3Decoder {
    fn default() -> Self {
        Self::new(44100, 2).expect("Failed to create default MP3 decoder")
    }
}

// Stub when feature disabled
#[cfg(not(feature = "mp3"))]
pub struct Mp3Decoder {
    _private: (),
}

#[cfg(not(feature = "mp3"))]
#[allow(dead_code)]
pub struct Mp3DecoderConfig {
    _private: (),
}

#[cfg(not(feature = "mp3"))]
impl Mp3Decoder {
    pub fn new(_sample_rate: u32, _channels: u16) -> Result<Self> {
        Err(Error::unsupported(
            "MP3 codec support not enabled. Enable the 'mp3' feature."
        ))
    }

    #[allow(dead_code)]
    pub fn with_config(_config: Mp3DecoderConfig) -> Result<Self> {
        Err(Error::unsupported("MP3 codec not enabled"))
    }
}

#[cfg(not(feature = "mp3"))]
impl Decoder for Mp3Decoder {
    fn send_packet(&mut self, _packet: &ZvdPacket) -> Result<()> {
        Err(Error::unsupported("MP3 codec not enabled"))
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        Err(Error::unsupported("MP3 codec not enabled"))
    }

    fn flush(&mut self) -> Result<()> {
        Err(Error::unsupported("MP3 codec not enabled"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "mp3")]
    fn test_mp3_decoder_creation() {
        let decoder = Mp3Decoder::new(44100, 2);
        assert!(decoder.is_ok(), "Failed to create MP3 decoder");
    }

    #[test]
    #[cfg(feature = "mp3")]
    fn test_mp3_decoder_with_config() {
        let config = Mp3DecoderConfig {
            sample_rate: 48000,
            channels: 2,
        };
        let decoder = Mp3Decoder::with_config(config);
        assert!(decoder.is_ok());
    }

    #[test]
    #[cfg(feature = "mp3")]
    fn test_mp3_invalid_sample_rate() {
        let decoder = Mp3Decoder::new(96000, 2);
        assert!(decoder.is_err(), "Should reject invalid sample rate");
    }

    #[test]
    #[cfg(feature = "mp3")]
    fn test_flush() {
        let mut decoder = Mp3Decoder::new(44100, 2).expect("Failed to create decoder");
        assert!(decoder.flush().is_ok());
    }

    #[test]
    #[cfg(not(feature = "mp3"))]
    fn test_mp3_disabled() {
        let decoder = Mp3Decoder::new(44100, 2);
        assert!(decoder.is_err());
    }
}
