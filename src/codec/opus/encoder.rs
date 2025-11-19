//! Opus audio encoder using libopus
//!
//! This module provides a complete Opus encoder implementation.
//! Opus is a versatile audio codec optimized for both speech and music.

use crate::codec::{AudioFrame, Encoder, Frame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, Timestamp};

#[cfg(feature = "opus-codec")]
use opus::{Application, Bitrate, Channels, Encoder as OpusEncoderLib};

/// Opus encoder configuration
#[derive(Debug, Clone)]
pub struct OpusEncoderConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub application: OpusApplication,
    pub bitrate: Option<i32>,
    pub complexity: Option<i32>, // 0-10, higher is better quality but slower
}

/// Opus application type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusApplication {
    /// Best for voice/speech
    Voip,
    /// Best for music
    Audio,
    /// Low-delay mode
    RestrictedLowdelay,
}

impl Default for OpusEncoderConfig {
    fn default() -> Self {
        OpusEncoderConfig {
            sample_rate: 48000,
            channels: 2,
            application: OpusApplication::Audio,
            bitrate: None,
            complexity: None,
        }
    }
}

/// Opus audio encoder
///
/// Encodes PCM (signed 16-bit samples) to Opus-compressed audio.
#[cfg(feature = "opus-codec")]
pub struct OpusEncoder {
    encoder: OpusEncoderLib,
    config: OpusEncoderConfig,
    frame_size: usize,
    /// Buffered frames waiting to be encoded
    frame_buffer: Vec<AudioFrame>,
    /// Buffered packets ready to be retrieved
    packet_buffer: Vec<Packet>,
    /// Current PTS counter
    pts_counter: i64,
}

#[cfg(feature = "opus-codec")]
impl OpusEncoder {
    /// Create a new Opus encoder with default configuration
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        let config = OpusEncoderConfig {
            sample_rate,
            channels,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create a new Opus encoder with custom configuration
    pub fn with_config(config: OpusEncoderConfig) -> Result<Self> {
        // Validate sample rate
        if !matches!(config.sample_rate, 8000 | 12000 | 16000 | 24000 | 48000) {
            return Err(Error::codec(format!(
                "Invalid Opus sample rate: {}. Must be 8000, 12000, 16000, 24000, or 48000 Hz",
                config.sample_rate
            )));
        }

        // Validate channels
        if config.channels == 0 || config.channels > 2 {
            return Err(Error::codec(format!(
                "Invalid channel count: {}. Opus supports mono (1) or stereo (2)",
                config.channels
            )));
        }

        let opus_channels = if config.channels == 1 {
            Channels::Mono
        } else {
            Channels::Stereo
        };

        let opus_app = match config.application {
            OpusApplication::Voip => Application::Voip,
            OpusApplication::Audio => Application::Audio,
            OpusApplication::RestrictedLowdelay => Application::RestrictedLowdelay,
        };

        let mut encoder = OpusEncoderLib::new(
            config.sample_rate,
            opus_channels,
            opus_app,
        )
        .map_err(|e| Error::codec(format!("Failed to create Opus encoder: {:?}", e)))?;

        // Set bitrate if specified
        if let Some(bitrate) = config.bitrate {
            encoder
                .set_bitrate(Bitrate::Bits(bitrate))
                .map_err(|e| Error::codec(format!("Failed to set Opus bitrate: {:?}", e)))?;
        }

        // Set complexity if specified (0-10)
        if let Some(complexity) = config.complexity {
            if complexity < 0 || complexity > 10 {
                return Err(Error::codec(format!(
                    "Invalid complexity: {}. Must be 0-10",
                    complexity
                )));
            }
            // Note: opus crate doesn't expose complexity setter directly
            // This would require raw FFI if needed
        }

        // Frame size: 2.5, 5, 10, 20, 40, or 60 ms
        // We'll use 20ms (960 samples at 48kHz, scales for other rates)
        let frame_size = (config.sample_rate as usize * 20) / 1000;

        Ok(OpusEncoder {
            encoder,
            config,
            frame_size,
            frame_buffer: Vec::new(),
            packet_buffer: Vec::new(),
            pts_counter: 0,
        })
    }

    /// Set target bitrate
    pub fn set_bitrate(&mut self, bitrate: i32) -> Result<()> {
        self.encoder
            .set_bitrate(Bitrate::Bits(bitrate))
            .map_err(|e| Error::codec(format!("Failed to set Opus bitrate: {:?}", e)))?;
        self.config.bitrate = Some(bitrate);
        Ok(())
    }

    /// Encode audio frame
    fn encode_audio(&mut self, audio_frame: &AudioFrame) -> Result<Buffer> {
        // Ensure sample rate matches
        if audio_frame.sample_rate != self.config.sample_rate {
            return Err(Error::codec(format!(
                "Sample rate mismatch: encoder expects {}, got {}",
                self.config.sample_rate, audio_frame.sample_rate
            )));
        }

        // Get audio data - Opus expects interleaved samples
        let input = if audio_frame.data.is_empty() {
            return Err(Error::codec("Empty audio frame"));
        } else {
            audio_frame.data[0].as_slice()
        };

        // Convert bytes to i16 samples (assuming S16 format)
        let samples: Vec<i16> = input
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();

        // Output buffer (max Opus packet size)
        let mut output = vec![0u8; 4000];

        // Encode
        let encoded_size = self
            .encoder
            .encode(&samples, &mut output)
            .map_err(|e| Error::codec(format!("Opus encoding failed: {:?}", e)))?;

        output.truncate(encoded_size);
        Ok(Buffer::from_vec(output))
    }
}

#[cfg(feature = "opus-codec")]
impl Encoder for OpusEncoder {
    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        match frame {
            Frame::Audio(audio_frame) => {
                // Encode frame immediately and buffer the packet
                let data = self.encode_audio(audio_frame)?;

                let mut packet = Packet::new(0, data);
                packet.pts = Timestamp::new(self.pts_counter);
                packet.duration = audio_frame.duration;

                self.pts_counter += audio_frame.duration;
                self.packet_buffer.push(packet);

                Ok(())
            }
            Frame::Video(_) => Err(Error::codec("Opus encoder only accepts audio frames")),
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if self.packet_buffer.is_empty() {
            return Err(Error::TryAgain);
        }

        Ok(self.packet_buffer.remove(0))
    }

    fn flush(&mut self) -> Result<()> {
        self.frame_buffer.clear();
        self.packet_buffer.clear();
        Ok(())
    }
}

#[cfg(feature = "opus-codec")]
impl Default for OpusEncoder {
    fn default() -> Self {
        Self::new(48000, 2).expect("Failed to create default Opus encoder")
    }
}

// Stub implementation when opus-codec feature is not enabled
#[cfg(not(feature = "opus-codec"))]
pub struct OpusEncoder {
    _private: (),
}

#[cfg(not(feature = "opus-codec"))]
#[allow(dead_code)]
pub struct OpusEncoderConfig {
    _private: (),
}

#[cfg(not(feature = "opus-codec"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusApplication {
    Voip,
    Audio,
    RestrictedLowdelay,
}

#[cfg(not(feature = "opus-codec"))]
impl OpusEncoder {
    pub fn new(_sample_rate: u32, _channels: u16) -> Result<Self> {
        Err(Error::unsupported(
            "Opus codec support not enabled. Enable the 'opus-codec' feature."
        ))
    }

    #[allow(dead_code)]
    pub fn with_config(_config: OpusEncoderConfig) -> Result<Self> {
        Err(Error::unsupported("Opus codec not enabled"))
    }
}

#[cfg(not(feature = "opus-codec"))]
impl Encoder for OpusEncoder {
    fn send_frame(&mut self, _frame: &Frame) -> Result<()> {
        Err(Error::unsupported("Opus codec not enabled"))
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        Err(Error::unsupported("Opus codec not enabled"))
    }

    fn flush(&mut self) -> Result<()> {
        Err(Error::unsupported("Opus codec not enabled"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "opus-codec")]
    fn test_opus_encoder_creation() {
        let encoder = OpusEncoder::new(48000, 2);
        assert!(encoder.is_ok(), "Failed to create Opus encoder");
    }

    #[test]
    #[cfg(feature = "opus-codec")]
    fn test_opus_encoder_with_config() {
        let config = OpusEncoderConfig {
            sample_rate: 48000,
            channels: 1,
            application: OpusApplication::Voip,
            bitrate: Some(64000),
            complexity: None,
        };
        let encoder = OpusEncoder::with_config(config);
        assert!(encoder.is_ok(), "Failed to create encoder with config");
    }

    #[test]
    #[cfg(feature = "opus-codec")]
    fn test_opus_encoder_various_sample_rates() {
        for sample_rate in &[8000, 12000, 16000, 24000, 48000] {
            let encoder = OpusEncoder::new(*sample_rate, 2);
            assert!(
                encoder.is_ok(),
                "Failed to create encoder at {} Hz",
                sample_rate
            );
        }
    }

    #[test]
    #[cfg(feature = "opus-codec")]
    fn test_opus_encoder_invalid_sample_rate() {
        let encoder = OpusEncoder::new(44100, 2);
        assert!(encoder.is_err(), "Should reject invalid sample rate");
    }

    #[test]
    #[cfg(feature = "opus-codec")]
    fn test_opus_encoder_set_bitrate() {
        let mut encoder = OpusEncoder::new(48000, 2).expect("Failed to create encoder");
        assert!(encoder.set_bitrate(128000).is_ok(), "Failed to set bitrate");
    }

    #[test]
    #[cfg(feature = "opus-codec")]
    fn test_flush() {
        let mut encoder = OpusEncoder::new(48000, 2).expect("Failed to create encoder");
        assert!(encoder.flush().is_ok(), "Flush should not error");
    }

    #[test]
    #[cfg(not(feature = "opus-codec"))]
    fn test_opus_disabled() {
        let encoder = OpusEncoder::new(48000, 2);
        assert!(encoder.is_err());
    }
}
