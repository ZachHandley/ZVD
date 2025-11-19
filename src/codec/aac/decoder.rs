//! AAC decoder using Symphonia
//!
//! AAC (Advanced Audio Coding) is a lossy audio codec successor to MP3.
//! This implementation uses Symphonia's AAC decoder (LC-AAC profile only).
//!
//! ## Features
//! - Sample rates: 8 to 96 kHz
//! - Channels: Up to 48 channels (typically 1-7.1)
//! - Profiles: LC-AAC (Low Complexity) only
//! - Container formats: ADTS, M4A/MP4
//!
//! ## Patent Status
//! AAC is patent-encumbered. See CODEC_LICENSES.md for details.
//! Enable with `aac` feature flag.
//!
//! ## Limitations
//! - HE-AAC (High Efficiency) not supported
//! - HE-AACv2 not supported
//! - Only LC-AAC profile is decoded
//!
//! ## Important: Use SymphoniaAdapter
//! AAC decoding in ZVD should use `SymphoniaAdapter` for complete
//! container-level decoding with AudioSpecificConfig parsing.

use crate::codec::{AudioFrame, Decoder, Frame};
use crate::error::{Error, Result};
use crate::format::Packet as ZvdPacket;

/// AAC decoder configuration
#[derive(Debug, Clone)]
pub struct AacDecoderConfig {
    pub sample_rate: u32,
    pub channels: u16,
    /// AudioSpecificConfig data (codec extradata)
    pub extradata: Option<Vec<u8>>,
}

impl Default for AacDecoderConfig {
    fn default() -> Self {
        AacDecoderConfig {
            sample_rate: 44100,
            channels: 2,
            extradata: None,
        }
    }
}

/// AAC audio decoder (LC-AAC only)
///
/// Note: AAC decoding in ZVD is done at container level via SymphoniaAdapter.
/// This struct provides the interface but directs users to the adapter.
#[cfg(feature = "aac")]
pub struct AacDecoder {
    config: AacDecoderConfig,
    frame_buffer: Vec<AudioFrame>,
}

#[cfg(feature = "aac")]
impl AacDecoder {
    /// Create a new AAC decoder
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        let config = AacDecoderConfig {
            sample_rate,
            channels,
            extradata: None,
        };
        Self::with_config(config)
    }

    /// Create decoder with AudioSpecificConfig extradata
    pub fn with_extradata(sample_rate: u32, channels: u16, extradata: &[u8]) -> Result<Self> {
        let config = AacDecoderConfig {
            sample_rate,
            channels,
            extradata: Some(extradata.to_vec()),
        };
        Self::with_config(config)
    }

    /// Create decoder with configuration
    pub fn with_config(config: AacDecoderConfig) -> Result<Self> {
        // Validate sample rate (AAC supports 8-96 kHz)
        if config.sample_rate < 8000 || config.sample_rate > 96000 {
            return Err(Error::codec(format!(
                "Invalid AAC sample rate: {}. Must be 8,000-96,000 Hz",
                config.sample_rate
            )));
        }

        // Validate channels (AAC supports up to 48, but typically 1-8)
        if config.channels == 0 || config.channels > 48 {
            return Err(Error::codec(format!(
                "Invalid channel count: {}. AAC supports 1-48 channels",
                config.channels
            )));
        }

        Ok(AacDecoder {
            config,
            frame_buffer: Vec::new(),
        })
    }
}

#[cfg(feature = "aac")]
impl Decoder for AacDecoder {
    fn send_packet(&mut self, _packet: &ZvdPacket) -> Result<()> {
        Err(Error::unsupported(
            "AAC packet-level decoding not implemented. Use SymphoniaAdapter for AAC/M4A file decoding. Only LC-AAC profile is supported."
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

#[cfg(feature = "aac")]
impl Default for AacDecoder {
    fn default() -> Self {
        Self::new(44100, 2).expect("Failed to create default AAC decoder")
    }
}

// Stub when feature disabled
#[cfg(not(feature = "aac"))]
pub struct AacDecoder {
    _private: (),
}

#[cfg(not(feature = "aac"))]
#[allow(dead_code)]
pub struct AacDecoderConfig {
    _private: (),
}

#[cfg(not(feature = "aac"))]
impl AacDecoder {
    pub fn new(_sample_rate: u32, _channels: u16) -> Result<Self> {
        Err(Error::unsupported(
            "AAC codec support not enabled. Enable the 'aac' feature and review CODEC_LICENSES.md for patent requirements."
        ))
    }

    #[allow(dead_code)]
    pub fn with_extradata(_sample_rate: u32, _channels: u16, _extradata: &[u8]) -> Result<Self> {
        Err(Error::unsupported("AAC codec not enabled"))
    }

    #[allow(dead_code)]
    pub fn with_config(_config: AacDecoderConfig) -> Result<Self> {
        Err(Error::unsupported("AAC codec not enabled"))
    }
}

#[cfg(not(feature = "aac"))]
impl Decoder for AacDecoder {
    fn send_packet(&mut self, _packet: &ZvdPacket) -> Result<()> {
        Err(Error::unsupported("AAC codec not enabled"))
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        Err(Error::unsupported("AAC codec not enabled"))
    }

    fn flush(&mut self) -> Result<()> {
        Err(Error::unsupported("AAC codec not enabled"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "aac")]
    fn test_aac_decoder_creation() {
        let decoder = AacDecoder::new(44100, 2);
        assert!(decoder.is_ok(), "Failed to create AAC decoder");
    }

    #[test]
    #[cfg(feature = "aac")]
    fn test_aac_decoder_with_extradata() {
        let extradata = vec![0x11, 0x90]; // Minimal AAC config
        let decoder = AacDecoder::with_extradata(44100, 2, &extradata);
        assert!(decoder.is_ok());
    }

    #[test]
    #[cfg(feature = "aac")]
    fn test_aac_invalid_sample_rate() {
        let decoder = AacDecoder::new(7000, 2);
        assert!(decoder.is_err(), "Should reject too low sample rate");

        let decoder = AacDecoder::new(100000, 2);
        assert!(decoder.is_err(), "Should reject too high sample rate");
    }

    #[test]
    #[cfg(feature = "aac")]
    fn test_flush() {
        let mut decoder = AacDecoder::new(44100, 2).expect("Failed to create decoder");
        assert!(decoder.flush().is_ok());
    }

    #[test]
    #[cfg(not(feature = "aac"))]
    fn test_aac_disabled() {
        let decoder = AacDecoder::new(44100, 2);
        assert!(decoder.is_err());
    }
}
