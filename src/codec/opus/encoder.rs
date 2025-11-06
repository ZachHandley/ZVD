//! Opus audio encoder

use crate::codec::{AudioFrame, Encoder, Frame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, Timestamp};
use opus::{Application, Channels, Encoder as OpusEncoderLib};

/// Opus encoder configuration
pub struct OpusEncoderConfig {
    pub sample_rate: u32,
    pub channels: Channels,
    pub application: Application,
    pub bitrate: Option<i32>,
}

impl Default for OpusEncoderConfig {
    fn default() -> Self {
        OpusEncoderConfig {
            sample_rate: 48000,
            channels: Channels::Stereo,
            application: Application::Audio,
            bitrate: None,
        }
    }
}

/// Opus audio encoder
pub struct OpusEncoder {
    encoder: OpusEncoderLib,
    config: OpusEncoderConfig,
    frame_size: usize,
}

impl OpusEncoder {
    /// Create a new Opus encoder with default configuration
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        let config = OpusEncoderConfig {
            sample_rate,
            channels: if channels == 1 {
                Channels::Mono
            } else {
                Channels::Stereo
            },
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create a new Opus encoder with custom configuration
    pub fn with_config(config: OpusEncoderConfig) -> Result<Self> {
        let encoder = OpusEncoderLib::new(
            config.sample_rate,
            config.channels,
            config.application,
        )
        .map_err(|e| Error::codec(format!("Failed to create Opus encoder: {:?}", e)))?;

        // Frame size: 2.5, 5, 10, 20, 40, or 60 ms
        // We'll use 20ms (960 samples at 48kHz)
        let frame_size = (config.sample_rate as usize * 20) / 1000;

        Ok(OpusEncoder {
            encoder,
            config,
            frame_size,
        })
    }

    /// Set target bitrate
    pub fn set_bitrate(&mut self, bitrate: i32) -> Result<()> {
        self.encoder
            .set_bitrate(opus::Bitrate::Bits(bitrate))
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

        // Output buffer
        let mut output = vec![0u8; 4000]; // Max Opus packet size

        // Encode
        let encoded_size = self
            .encoder
            .encode(&samples, &mut output)
            .map_err(|e| Error::codec(format!("Opus encoding failed: {:?}", e)))?;

        output.truncate(encoded_size);
        Ok(Buffer::from_vec(output))
    }
}

impl Encoder for OpusEncoder {
    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        match frame {
            Frame::Audio(_) => {
                // In a real implementation, we'd buffer the frame for encoding
                Ok(())
            }
            Frame::Video(_) => Err(Error::codec("Opus encoder only accepts audio frames")),
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        // Simplified implementation - in reality, we'd need to buffer frames
        // and encode when we have enough samples
        Err(Error::TryAgain)
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

impl OpusEncoder {
    /// Convenience method to encode a frame directly to a packet
    pub fn encode_frame(&mut self, frame: &AudioFrame) -> Result<Packet> {
        let data = self.encode_audio(frame)?;
        let mut packet = Packet::new(0, data);
        packet.pts = frame.pts;
        packet.duration = frame.duration;
        Ok(packet)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opus_encoder_creation() {
        let encoder = OpusEncoder::new(48000, 2);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_opus_encoder_with_config() {
        let config = OpusEncoderConfig {
            sample_rate: 48000,
            channels: Channels::Mono,
            application: Application::Voip,
            bitrate: Some(64000),
        };
        let encoder = OpusEncoder::with_config(config);
        assert!(encoder.is_ok());
    }
}
