//! Opus audio decoder using libopus
//!
//! This module provides a complete Opus decoder implementation.
//! Opus is a versatile audio codec optimized for both speech and music.
//!
//! # Features
//!
//! - Sample rates: 8kHz, 12kHz, 16kHz, 24kHz, 48kHz
//! - Channels: Mono, Stereo
//! - Bitrates: 6 kbps to 510 kbps
//! - Low latency: 2.5ms to 60ms frame sizes
//! - Packet loss concealment

use crate::codec::{AudioFrame, Decoder, Frame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, SampleFormat, Timestamp};

#[cfg(feature = "opus-codec")]
use opus::{Channels, Decoder as OpusDecoderLib};

/// Opus audio decoder
///
/// Decodes Opus-compressed audio to PCM (signed 16-bit samples).
#[cfg(feature = "opus-codec")]
pub struct OpusDecoder {
    decoder: OpusDecoderLib,
    sample_rate: u32,
    channels: u16,
    /// Buffered decoded frames
    frame_buffer: Vec<AudioFrame>,
}

#[cfg(feature = "opus-codec")]
impl OpusDecoder {
    /// Create a new Opus decoder
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate (8000, 12000, 16000, 24000, or 48000 Hz)
    /// * `channels` - Number of channels (1 = mono, 2 = stereo)
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        // Validate sample rate
        if !matches!(sample_rate, 8000 | 12000 | 16000 | 24000 | 48000) {
            return Err(Error::codec(format!(
                "Invalid Opus sample rate: {}. Must be 8000, 12000, 16000, 24000, or 48000 Hz",
                sample_rate
            )));
        }

        // Validate channels
        if channels == 0 || channels > 2 {
            return Err(Error::codec(format!(
                "Invalid channel count: {}. Opus supports mono (1) or stereo (2)",
                channels
            )));
        }

        let opus_channels = if channels == 1 {
            Channels::Mono
        } else {
            Channels::Stereo
        };

        let decoder = OpusDecoderLib::new(sample_rate, opus_channels)
            .map_err(|e| Error::codec(format!("Failed to create Opus decoder: {:?}", e)))?;

        Ok(OpusDecoder {
            decoder,
            sample_rate,
            channels,
            frame_buffer: Vec::new(),
        })
    }

    /// Decode Opus packet to audio frame
    fn decode_opus(&mut self, data: &[u8], pts: Timestamp) -> Result<AudioFrame> {
        // Maximum frame size for Opus is 120ms at 48kHz
        let max_frame_size = (self.sample_rate as usize * 120) / 1000;
        let mut output = vec![0i16; max_frame_size * self.channels as usize];

        // Decode (false = don't use FEC for packet loss concealment)
        let decoded_samples = self
            .decoder
            .decode(data, &mut output, false)
            .map_err(|e| Error::codec(format!("Opus decoding failed: {:?}", e)))?;

        // Convert i16 samples to bytes (little-endian PCM)
        let mut pcm_data = Vec::with_capacity(decoded_samples * self.channels as usize * 2);
        for sample in output.iter().take(decoded_samples * self.channels as usize) {
            pcm_data.extend_from_slice(&sample.to_le_bytes());
        }

        // Create audio frame
        let mut frame = AudioFrame::new(
            decoded_samples,
            self.sample_rate,
            self.channels,
            SampleFormat::I16,
        );
        frame.data.push(Buffer::from_vec(pcm_data));
        frame.pts = pts;
        frame.duration = decoded_samples as i64;

        Ok(frame)
    }
}

#[cfg(feature = "opus-codec")]
impl Decoder for OpusDecoder {
    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if packet.data.is_empty() {
            return Err(Error::codec("Empty Opus packet"));
        }

        // Decode packet immediately and buffer the frame
        let frame = self.decode_opus(packet.data.as_slice(), packet.pts)?;
        self.frame_buffer.push(frame);

        Ok(())
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

#[cfg(feature = "opus-codec")]
impl Default for OpusDecoder {
    fn default() -> Self {
        Self::new(48000, 2).expect("Failed to create default Opus decoder")
    }
}

// Stub implementation when opus-codec feature is not enabled
#[cfg(not(feature = "opus-codec"))]
pub struct OpusDecoder {
    _private: (),
}

#[cfg(not(feature = "opus-codec"))]
impl OpusDecoder {
    pub fn new(_sample_rate: u32, _channels: u16) -> Result<Self> {
        Err(Error::unsupported(
            "Opus codec support not enabled. Enable the 'opus-codec' feature."
        ))
    }
}

#[cfg(not(feature = "opus-codec"))]
impl Decoder for OpusDecoder {
    fn send_packet(&mut self, _packet: &Packet) -> Result<()> {
        Err(Error::unsupported("Opus codec not enabled"))
    }

    fn receive_frame(&mut self) -> Result<Frame> {
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
    fn test_opus_decoder_creation() {
        let decoder = OpusDecoder::new(48000, 2);
        assert!(decoder.is_ok(), "Failed to create Opus decoder");
    }

    #[test]
    #[cfg(feature = "opus-codec")]
    fn test_opus_decoder_mono() {
        let decoder = OpusDecoder::new(48000, 1);
        assert!(decoder.is_ok(), "Failed to create mono Opus decoder");
    }

    #[test]
    #[cfg(feature = "opus-codec")]
    fn test_opus_decoder_various_sample_rates() {
        for sample_rate in &[8000, 12000, 16000, 24000, 48000] {
            let decoder = OpusDecoder::new(*sample_rate, 2);
            assert!(
                decoder.is_ok(),
                "Failed to create decoder at {} Hz",
                sample_rate
            );
        }
    }

    #[test]
    #[cfg(feature = "opus-codec")]
    fn test_opus_decoder_invalid_sample_rate() {
        let decoder = OpusDecoder::new(44100, 2);
        assert!(decoder.is_err(), "Should reject invalid sample rate");
    }

    #[test]
    #[cfg(feature = "opus-codec")]
    fn test_opus_decoder_invalid_channels() {
        let decoder = OpusDecoder::new(48000, 3);
        assert!(decoder.is_err(), "Should reject > 2 channels");
    }

    #[test]
    #[cfg(feature = "opus-codec")]
    fn test_flush() {
        let mut decoder = OpusDecoder::new(48000, 2).expect("Failed to create decoder");
        assert!(decoder.flush().is_ok(), "Flush should not error");
    }

    #[test]
    #[cfg(not(feature = "opus-codec"))]
    fn test_opus_disabled() {
        let decoder = OpusDecoder::new(48000, 2);
        assert!(decoder.is_err());
    }
}
