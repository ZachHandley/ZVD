//! FLAC audio encoder
//!
//! This module provides a FLAC encoder implementation for lossless audio compression.

use crate::codec::{AudioFrame, Encoder, Frame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, SampleFormat, Timestamp};

/// FLAC encoder configuration
#[derive(Debug, Clone)]
pub struct FlacEncoderConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u32,
    pub compression_level: u32, // 0-8, higher is better compression but slower
    pub block_size: u32,        // Typically 4096 samples
}

impl Default for FlacEncoderConfig {
    fn default() -> Self {
        FlacEncoderConfig {
            sample_rate: 44100,
            channels: 2,
            bits_per_sample: 16,
            compression_level: 5,
            block_size: 4096,
        }
    }
}

impl FlacEncoderConfig {
    /// Validate the encoder configuration
    pub fn validate(&self) -> Result<()> {
        // Validate sample rate
        if self.sample_rate == 0 || self.sample_rate > 655_350 {
            return Err(Error::codec(format!(
                "Invalid FLAC sample rate: {}. Must be 1-655,350 Hz",
                self.sample_rate
            )));
        }

        // Validate channels
        if self.channels == 0 || self.channels > 8 {
            return Err(Error::codec(format!(
                "Invalid channel count: {}. FLAC supports 1-8 channels",
                self.channels
            )));
        }

        // Validate bit depth
        if self.bits_per_sample < 4 || self.bits_per_sample > 32 {
            return Err(Error::codec(format!(
                "Invalid bits per sample: {}. FLAC supports 4-32 bits",
                self.bits_per_sample
            )));
        }

        // Validate compression level
        if self.compression_level > 8 {
            return Err(Error::codec(format!(
                "Invalid compression level: {}. Must be 0-8",
                self.compression_level
            )));
        }

        // Validate block size
        if self.block_size < 16 || self.block_size > 65535 {
            return Err(Error::codec(format!(
                "Invalid block size: {}. Must be 16-65535",
                self.block_size
            )));
        }

        Ok(())
    }
}

/// FLAC audio encoder
///
/// Encodes PCM audio to FLAC lossless compressed format.
///
/// # Example
///
/// ```no_run
/// use zvd_lib::codec::flac::FlacEncoder;
/// use zvd_lib::codec::{Encoder, AudioFrame, Frame};
/// use zvd_lib::util::SampleFormat;
///
/// let mut encoder = FlacEncoder::new(44100, 2)?;
///
/// // Create audio frame
/// let frame = AudioFrame::new(4096, 2, SampleFormat::I16);
/// // ... fill frame with data ...
///
/// encoder.send_frame(&Frame::Audio(frame))?;
/// encoder.flush()?;
///
/// // Retrieve encoded packet
/// let packet = encoder.receive_packet()?;
/// # Ok::<(), zvd_lib::error::Error>(())
/// ```
pub struct FlacEncoder {
    config: FlacEncoderConfig,
    /// Buffered frames waiting to be encoded
    frame_buffer: Vec<AudioFrame>,
    /// Buffered packets ready to be retrieved
    packet_buffer: Vec<Packet>,
    /// Current PTS counter
    pts_counter: i64,
    /// Total samples encoded (for stream info)
    total_samples: u64,
    /// Internal encoder state
    encoder_state: FlacEncoderState,
}

/// Internal FLAC encoder state
struct FlacEncoderState {
    sample_buffer: Vec<i32>, // Internal buffer for samples
    header_written: bool,
}

impl FlacEncoder {
    /// Create a new FLAC encoder with default configuration
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        let config = FlacEncoderConfig {
            sample_rate,
            channels,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create a new FLAC encoder with custom configuration
    pub fn with_config(config: FlacEncoderConfig) -> Result<Self> {
        config.validate()?;

        Ok(FlacEncoder {
            config,
            frame_buffer: Vec::new(),
            packet_buffer: Vec::new(),
            pts_counter: 0,
            total_samples: 0,
            encoder_state: FlacEncoderState {
                sample_buffer: Vec::new(),
                header_written: false,
            },
        })
    }

    /// Set compression level (0-8)
    pub fn set_compression_level(&mut self, level: u32) -> Result<()> {
        if level > 8 {
            return Err(Error::codec(format!(
                "Invalid compression level: {}. Must be 0-8",
                level
            )));
        }
        self.config.compression_level = level;
        Ok(())
    }

    /// Set block size
    pub fn set_block_size(&mut self, block_size: u32) -> Result<()> {
        if block_size < 16 || block_size > 65535 {
            return Err(Error::codec(format!(
                "Invalid block size: {}. Must be 16-65535",
                block_size
            )));
        }
        self.config.block_size = block_size;
        Ok(())
    }

    /// Get encoder configuration
    pub fn config(&self) -> &FlacEncoderConfig {
        &self.config
    }

    /// Encode audio samples to FLAC
    fn encode_samples(&mut self, samples: &[i32], pts: Timestamp) -> Result<()> {
        // Write stream header if not done yet
        if !self.encoder_state.header_written {
            let header_packet = self.create_stream_header()?;
            self.packet_buffer.push(header_packet);
            self.encoder_state.header_written = true;
        }

        // Add samples to buffer
        self.encoder_state.sample_buffer.extend_from_slice(samples);

        // Encode when we have enough samples for a block
        let samples_per_block = (self.config.block_size * self.config.channels as u32) as usize;

        while self.encoder_state.sample_buffer.len() >= samples_per_block {
            let block_samples: Vec<i32> = self
                .encoder_state
                .sample_buffer
                .drain(..samples_per_block)
                .collect();

            let encoded_data = self.encode_block(&block_samples)?;

            let packet = Packet {
                stream_index: 0,
                data: Buffer::from_vec(encoded_data),
                pts,
                dts: pts,
                duration: Timestamp::new((self.config.block_size / self.config.sample_rate) as i64),
                keyframe: false,
            };

            self.packet_buffer.push(packet);
            self.total_samples += self.config.block_size as u64;
        }

        Ok(())
    }

    /// Create FLAC stream header
    fn create_stream_header(&self) -> Result<Packet> {
        let mut header = Vec::new();

        // FLAC signature
        header.extend_from_slice(b"fLaC");

        // STREAMINFO block (metadata block type 0, last metadata block)
        header.push(0x80); // Last metadata block flag + block type 0

        // Block length (34 bytes for STREAMINFO)
        header.extend_from_slice(&[0, 0, 34]);

        // Min block size (16 bits)
        header.extend_from_slice(&(self.config.block_size as u16).to_be_bytes());
        // Max block size (16 bits)
        header.extend_from_slice(&(self.config.block_size as u16).to_be_bytes());

        // Min frame size (24 bits) - 0 for unknown
        header.extend_from_slice(&[0, 0, 0]);
        // Max frame size (24 bits) - 0 for unknown
        header.extend_from_slice(&[0, 0, 0]);

        // Sample rate (20 bits) + channels (3 bits) + bits per sample (5 bits)
        let sample_rate_20 = (self.config.sample_rate & 0xFFFFF) as u32;
        let channels_3 = ((self.config.channels - 1) & 0x07) as u32;
        let bits_per_sample_5 = ((self.config.bits_per_sample - 1) & 0x1F) as u32;

        let combined = (sample_rate_20 << 12) | (channels_3 << 9) | (bits_per_sample_5 << 4);
        header.push((combined >> 16) as u8);
        header.push((combined >> 8) as u8);
        header.push((combined >> 0) as u8);

        // Total samples (36 bits) - 0 for unknown, pad 4 bits
        header.push(0);
        header.extend_from_slice(&[0, 0, 0, 0]);

        // MD5 signature (128 bits) - all zeros for now
        header.extend_from_slice(&[0u8; 16]);

        Ok(Packet {
            stream_index: 0,
            data: Buffer::from_vec(header),
            pts: Timestamp::new(0),
            dts: Timestamp::new(0),
            duration: Timestamp::new(0),
            keyframe: true,
        })
    }

    /// Encode a block of samples
    fn encode_block(&self, samples: &[i32]) -> Result<Vec<u8>> {
        // This is a simplified FLAC encoder
        // In a full implementation, this would:
        // 1. Apply fixed/LPC prediction
        // 2. Compute residuals
        // 3. Encode residuals with Rice codes
        // 4. Write frame header and encoded data

        let mut encoded = Vec::new();

        // Frame header sync code
        encoded.extend_from_slice(&[0xFF, 0xF8]);

        // Block size and sample rate codes
        let block_size_code = match self.config.block_size {
            4096 => 0x08,
            _ => 0x00, // Use end-of-header value
        };
        encoded.push((block_size_code << 4) | 0x04); // 44.1kHz

        // Channel assignment (left-right stereo)
        let channel_code = match self.config.channels {
            1 => 0x00,
            2 => 0x10,
            _ => 0x00,
        };
        encoded.push(channel_code | ((self.config.bits_per_sample as u8 - 1) << 1));

        // Frame number (8-bit for simplicity)
        encoded.push(0x00);

        // CRC-8 placeholder
        encoded.push(0x00);

        // Simplified: Store samples as verbatim (no compression)
        // In production, use proper FLAC encoding with prediction and Rice coding
        for &sample in samples {
            match self.config.bits_per_sample {
                16 => {
                    let s = sample as i16;
                    encoded.extend_from_slice(&s.to_be_bytes());
                }
                24 => {
                    let bytes = sample.to_be_bytes();
                    encoded.extend_from_slice(&bytes[1..4]);
                }
                32 => {
                    encoded.extend_from_slice(&sample.to_be_bytes());
                }
                _ => {
                    let s = sample as i16;
                    encoded.extend_from_slice(&s.to_be_bytes());
                }
            }
        }

        // Frame footer CRC-16 placeholder
        encoded.extend_from_slice(&[0x00, 0x00]);

        Ok(encoded)
    }

    /// Convert audio frame samples to i32
    fn convert_samples_to_i32(&self, frame: &AudioFrame) -> Result<Vec<i32>> {
        if frame.data.is_empty() {
            return Err(Error::codec("Audio frame has no data"));
        }

        let mut samples = Vec::new();
        let data = frame.data[0].as_slice();

        match frame.format {
            SampleFormat::I16 => {
                for chunk in data.chunks_exact(2) {
                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                    samples.push(sample as i32);
                }
            }
            SampleFormat::I32 => {
                for chunk in data.chunks_exact(4) {
                    let sample = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    samples.push(sample);
                }
            }
            SampleFormat::F32 => {
                for chunk in data.chunks_exact(4) {
                    let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    // Convert float (-1.0 to 1.0) to i32
                    let i32_sample = (sample * 2147483647.0) as i32;
                    samples.push(i32_sample);
                }
            }
            _ => {
                return Err(Error::codec(format!(
                    "Unsupported sample format for FLAC encoding: {:?}",
                    frame.format
                )));
            }
        }

        Ok(samples)
    }
}

impl Encoder for FlacEncoder {
    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let audio_frame = match frame {
            Frame::Audio(af) => af,
            _ => return Err(Error::codec("FLAC encoder only accepts audio frames")),
        };

        // Validate frame parameters
        if audio_frame.channels != self.config.channels {
            return Err(Error::codec(format!(
                "Frame channel count {} doesn't match encoder config {}",
                audio_frame.channels, self.config.channels
            )));
        }

        // Convert samples to i32
        let samples = self.convert_samples_to_i32(audio_frame)?;

        // Encode samples
        self.encode_samples(&samples, audio_frame.pts)?;

        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if self.packet_buffer.is_empty() {
            return Err(Error::TryAgain);
        }

        Ok(self.packet_buffer.remove(0))
    }

    fn flush(&mut self) -> Result<()> {
        // Encode any remaining samples
        if !self.encoder_state.sample_buffer.is_empty() {
            // Pad with zeros if necessary
            let samples_per_block =
                (self.config.block_size * self.config.channels as u32) as usize;
            while self.encoder_state.sample_buffer.len() < samples_per_block {
                self.encoder_state.sample_buffer.push(0);
            }

            let block_samples: Vec<i32> = self
                .encoder_state
                .sample_buffer
                .drain(..)
                .collect();

            let encoded_data = self.encode_block(&block_samples)?;

            let packet = Packet {
                stream_index: 0,
                data: Buffer::from_vec(encoded_data),
                pts: Timestamp::new(self.pts_counter),
                dts: Timestamp::new(self.pts_counter),
                duration: Timestamp::new(
                    (self.config.block_size / self.config.sample_rate) as i64
                ),
                keyframe: false,
            };

            self.packet_buffer.push(packet);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flac_encoder_creation() {
        let encoder = FlacEncoder::new(44100, 2);
        assert!(encoder.is_ok());

        let encoder = encoder.unwrap();
        assert_eq!(encoder.config.sample_rate, 44100);
        assert_eq!(encoder.config.channels, 2);
    }

    #[test]
    fn test_flac_encoder_config_validation() {
        // Valid config
        let config = FlacEncoderConfig::default();
        assert!(config.validate().is_ok());

        // Invalid sample rate (too high)
        let mut config = FlacEncoderConfig::default();
        config.sample_rate = 1_000_000;
        assert!(config.validate().is_err());

        // Invalid channels
        let mut config = FlacEncoderConfig::default();
        config.channels = 0;
        assert!(config.validate().is_err());

        let mut config = FlacEncoderConfig::default();
        config.channels = 9;
        assert!(config.validate().is_err());

        // Invalid bit depth
        let mut config = FlacEncoderConfig::default();
        config.bits_per_sample = 3;
        assert!(config.validate().is_err());

        let mut config = FlacEncoderConfig::default();
        config.bits_per_sample = 33;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_flac_encoder_compression_levels() {
        let mut encoder = FlacEncoder::new(44100, 2).unwrap();

        // Valid compression levels
        assert!(encoder.set_compression_level(0).is_ok());
        assert!(encoder.set_compression_level(5).is_ok());
        assert!(encoder.set_compression_level(8).is_ok());

        // Invalid compression level
        assert!(encoder.set_compression_level(9).is_err());
    }

    #[test]
    fn test_flac_encoder_block_sizes() {
        let mut encoder = FlacEncoder::new(44100, 2).unwrap();

        // Valid block sizes
        assert!(encoder.set_block_size(4096).is_ok());
        assert!(encoder.set_block_size(1024).is_ok());

        // Invalid block sizes
        assert!(encoder.set_block_size(15).is_err());
        assert!(encoder.set_block_size(65536).is_err());
    }

    #[test]
    fn test_flac_encoder_sample_rates() {
        // Common sample rates
        assert!(FlacEncoder::new(8000, 2).is_ok());
        assert!(FlacEncoder::new(16000, 2).is_ok());
        assert!(FlacEncoder::new(44100, 2).is_ok());
        assert!(FlacEncoder::new(48000, 2).is_ok());
        assert!(FlacEncoder::new(96000, 2).is_ok());
        assert!(FlacEncoder::new(192000, 2).is_ok());
    }

    #[test]
    fn test_flac_encoder_channels() {
        // Valid channel counts
        assert!(FlacEncoder::new(44100, 1).is_ok()); // Mono
        assert!(FlacEncoder::new(44100, 2).is_ok()); // Stereo
        assert!(FlacEncoder::new(44100, 6).is_ok()); // 5.1
        assert!(FlacEncoder::new(44100, 8).is_ok()); // 7.1
    }

    #[test]
    fn test_flac_encoder_send_wrong_frame_type() {
        use crate::codec::VideoFrame;
        use crate::util::PixelFormat;

        let mut encoder = FlacEncoder::new(44100, 2).unwrap();
        let video_frame = VideoFrame::new(640, 480, PixelFormat::YUV420P);

        let result = encoder.send_frame(&Frame::Video(video_frame));
        assert!(result.is_err());
    }

    #[test]
    fn test_flac_encoder_wrong_channels() {
        let mut encoder = FlacEncoder::new(44100, 2).unwrap();

        // Create mono frame for stereo encoder
        let frame = AudioFrame::new(1024, 1, SampleFormat::I16);

        let result = encoder.send_frame(&Frame::Audio(frame));
        assert!(result.is_err());
    }
}
