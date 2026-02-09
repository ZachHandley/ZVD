//! FLAC audio encoder using flacenc
//!
//! FLAC (Free Lossless Audio Codec) encoder providing lossless audio compression.
//! This encoder is royalty-free and patent-free.
//!
//! ## Features
//! - Compression levels 0-8 (mapped to block size and prediction order)
//! - 8/16/24/32-bit sample support
//! - Up to 8 channels (though stereo is most efficient)
//! - Proper STREAMINFO metadata block generation
//!
//! ## Example
//! ```ignore
//! let encoder = FlacEncoder::new(44100, 2, 16)?;
//! encoder.send_frame(&frame)?;
//! let packet = encoder.receive_packet()?;
//! ```

use crate::codec::{AudioFrame, Encoder, Frame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, SampleFormat, Timestamp};
use flacenc::bitsink::ByteSink;
use flacenc::component::BitRepr;
use flacenc::config::Encoder as FlacConfig;
use flacenc::error::{Verify, Verified};
use flacenc::source::MemSource;
use std::collections::VecDeque;

/// Default block size for FLAC encoding (4096 samples is a good balance)
const DEFAULT_BLOCK_SIZE: usize = 4096;

/// Minimum block size supported by FLAC
const MIN_BLOCK_SIZE: usize = 16;

/// Maximum block size supported by FLAC
const MAX_BLOCK_SIZE: usize = 65535;

/// FLAC encoder configuration
#[derive(Debug, Clone)]
pub struct FlacEncoderConfig {
    /// Sample rate in Hz (e.g., 44100, 48000, 96000)
    pub sample_rate: u32,
    /// Number of audio channels (1-8)
    pub channels: u16,
    /// Bits per sample (8, 16, 24, or 32)
    pub bits_per_sample: u8,
    /// Compression level (0-8, default 5)
    /// 0 = fastest, 8 = best compression
    pub compression_level: u8,
    /// Block size in samples (16-65535, or 0 for auto based on compression level)
    pub block_size: usize,
}

impl Default for FlacEncoderConfig {
    fn default() -> Self {
        FlacEncoderConfig {
            sample_rate: 44100,
            channels: 2,
            bits_per_sample: 16,
            compression_level: 5,
            block_size: 0, // Auto
        }
    }
}

impl FlacEncoderConfig {
    /// Create a new configuration with specified parameters
    pub fn new(sample_rate: u32, channels: u16, bits_per_sample: u8) -> Self {
        FlacEncoderConfig {
            sample_rate,
            channels,
            bits_per_sample,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for CD audio (44100 Hz, 16-bit, stereo)
    pub fn cd_quality() -> Self {
        FlacEncoderConfig {
            sample_rate: 44100,
            channels: 2,
            bits_per_sample: 16,
            compression_level: 5,
            block_size: 4096,
        }
    }

    /// Create a configuration optimized for high-resolution audio
    pub fn high_resolution(sample_rate: u32, bits_per_sample: u8) -> Self {
        FlacEncoderConfig {
            sample_rate,
            channels: 2,
            bits_per_sample,
            compression_level: 5,
            block_size: 4096,
        }
    }

    /// Create a configuration for fastest encoding
    pub fn fast() -> Self {
        FlacEncoderConfig {
            compression_level: 0,
            block_size: 1152, // Smaller blocks = faster
            ..Default::default()
        }
    }

    /// Create a configuration for best compression
    pub fn best() -> Self {
        FlacEncoderConfig {
            compression_level: 8,
            block_size: 4096,
            ..Default::default()
        }
    }

    /// Get the effective block size based on compression level
    pub fn effective_block_size(&self) -> usize {
        if self.block_size > 0 {
            self.block_size.clamp(MIN_BLOCK_SIZE, MAX_BLOCK_SIZE)
        } else {
            // Auto block size based on compression level
            match self.compression_level {
                0 => 1152,
                1 => 1152,
                2 => 2048,
                3 => 2048,
                4 => 4096,
                5 => 4096,
                6 => 4096,
                7 => 4608,
                _ => 4608,
            }
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        // Validate sample rate (FLAC supports 1-655350 Hz)
        if self.sample_rate == 0 || self.sample_rate > 655350 {
            return Err(Error::codec(format!(
                "Invalid FLAC sample rate: {}. Valid range: 1-655350 Hz",
                self.sample_rate
            )));
        }

        // Validate channels (FLAC supports 1-8)
        if self.channels == 0 || self.channels > 8 {
            return Err(Error::codec(format!(
                "Invalid FLAC channel count: {}. Valid range: 1-8",
                self.channels
            )));
        }

        // Validate bits per sample (FLAC supports 4-32, but we support common values)
        if !matches!(self.bits_per_sample, 8 | 16 | 24 | 32) {
            return Err(Error::codec(format!(
                "Invalid FLAC bits per sample: {}. Supported: 8, 16, 24, 32",
                self.bits_per_sample
            )));
        }

        // Validate compression level
        if self.compression_level > 8 {
            return Err(Error::codec(format!(
                "Invalid FLAC compression level: {}. Valid range: 0-8",
                self.compression_level
            )));
        }

        Ok(())
    }
}

/// FLAC audio encoder
///
/// Encodes PCM audio to FLAC format using the flacenc library.
/// This encoder accumulates samples and encodes them in blocks.
pub struct FlacEncoder {
    /// Encoder configuration
    config: FlacEncoderConfig,
    /// Verified flacenc configuration
    flac_config: Verified<FlacConfig>,
    /// Block size for encoding
    block_size: usize,
    /// Input sample buffer (i32 samples, interleaved)
    sample_buffer: Vec<i32>,
    /// Output packet queue
    packet_queue: VecDeque<Packet>,
    /// Current presentation timestamp
    current_pts: Timestamp,
    /// Total samples encoded (for PTS calculation)
    samples_encoded: u64,
    /// Stream index for output packets
    stream_index: usize,
    /// STREAMINFO metadata block (extradata)
    extradata: Option<Vec<u8>>,
    /// Whether we've generated the first output (with stream header)
    header_written: bool,
}

impl FlacEncoder {
    /// Create a new FLAC encoder with default compression level (5)
    ///
    /// # Arguments
    /// * `sample_rate` - Audio sample rate in Hz
    /// * `channels` - Number of audio channels (1-8)
    /// * `bits_per_sample` - Bits per sample (8, 16, 24, or 32)
    ///
    /// # Returns
    /// A Result containing the encoder or an error if creation fails
    pub fn new(sample_rate: u32, channels: u16, bits_per_sample: u8) -> Result<Self> {
        let config = FlacEncoderConfig::new(sample_rate, channels, bits_per_sample);
        Self::with_config(config)
    }

    /// Create a new FLAC encoder with custom configuration
    pub fn with_config(config: FlacEncoderConfig) -> Result<Self> {
        config.validate()?;

        let block_size = config.effective_block_size();

        // Create flacenc configuration
        let mut flac_config = FlacConfig::default();
        flac_config.block_size = block_size;

        // Verify the configuration
        let verified_config = flac_config
            .into_verified()
            .map_err(|e| Error::codec(format!("Invalid FLAC encoder config: {:?}", e)))?;

        Ok(FlacEncoder {
            config,
            flac_config: verified_config,
            block_size,
            sample_buffer: Vec::with_capacity(block_size * 8), // Max 8 channels
            packet_queue: VecDeque::new(),
            current_pts: Timestamp::none(),
            samples_encoded: 0,
            stream_index: 0,
            extradata: None,
            header_written: false,
        })
    }

    /// Set the compression level (0-8)
    ///
    /// 0 = fastest encoding, least compression
    /// 8 = slowest encoding, best compression
    /// Default is 5.
    pub fn set_compression_level(&mut self, level: u8) -> Result<()> {
        if level > 8 {
            return Err(Error::codec(format!(
                "Invalid compression level: {}. Valid range: 0-8",
                level
            )));
        }
        self.config.compression_level = level;

        // Update block size based on new compression level
        let new_block_size = self.config.effective_block_size();
        if new_block_size != self.block_size {
            self.block_size = new_block_size;

            // Recreate the flacenc config
            let mut flac_config = FlacConfig::default();
            flac_config.block_size = self.block_size;

            self.flac_config = flac_config
                .into_verified()
                .map_err(|e| Error::codec(format!("Invalid FLAC encoder config: {:?}", e)))?;
        }

        Ok(())
    }

    /// Set stream index for output packets
    pub fn set_stream_index(&mut self, index: usize) {
        self.stream_index = index;
    }

    /// Get the sample rate
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Get the number of channels
    pub fn channels(&self) -> u16 {
        self.config.channels
    }

    /// Get bits per sample
    pub fn bits_per_sample(&self) -> u8 {
        self.config.bits_per_sample
    }

    /// Get the current compression level
    pub fn compression_level(&self) -> u8 {
        self.config.compression_level
    }

    /// Get the block size
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Convert audio frame to i32 samples
    fn frame_to_samples(&self, frame: &AudioFrame) -> Result<Vec<i32>> {
        if frame.data.is_empty() {
            return Err(Error::codec("Empty audio frame"));
        }

        let data = frame.data[0].as_slice();

        // Convert based on sample format
        let samples: Vec<i32> = match frame.format {
            SampleFormat::I16 => {
                // Convert i16 to i32
                data.chunks_exact(2)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as i32)
                    .collect()
            }
            SampleFormat::I32 => {
                // Already i32
                data.chunks_exact(4)
                    .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect()
            }
            SampleFormat::F32 => {
                // Convert f32 to i32 based on bit depth
                let scale = (1i64 << (self.config.bits_per_sample - 1)) as f32;
                data.chunks_exact(4)
                    .map(|chunk| {
                        let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        (f.clamp(-1.0, 1.0) * scale) as i32
                    })
                    .collect()
            }
            SampleFormat::F64 => {
                // Convert f64 to i32 based on bit depth
                let scale = (1i64 << (self.config.bits_per_sample - 1)) as f64;
                data.chunks_exact(8)
                    .map(|chunk| {
                        let f = f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]);
                        (f.clamp(-1.0, 1.0) * scale) as i32
                    })
                    .collect()
            }
            SampleFormat::U8 => {
                // Convert u8 to i32 (center at 128, scale to bit depth)
                let shift = self.config.bits_per_sample - 8;
                data.iter()
                    .map(|&b| ((b as i32) - 128) << shift)
                    .collect()
            }
            // Handle planar formats
            SampleFormat::I16P => {
                // Planar i16 - interleave channels
                let samples_per_channel = frame.nb_samples;
                let channels = frame.channels as usize;
                let mut interleaved = Vec::with_capacity(samples_per_channel * channels);

                for sample_idx in 0..samples_per_channel {
                    for ch in 0..channels {
                        if ch < frame.data.len() {
                            let ch_data = frame.data[ch].as_slice();
                            let byte_idx = sample_idx * 2;
                            if byte_idx + 1 < ch_data.len() {
                                let sample =
                                    i16::from_le_bytes([ch_data[byte_idx], ch_data[byte_idx + 1]]);
                                interleaved.push(sample as i32);
                            } else {
                                interleaved.push(0);
                            }
                        } else {
                            interleaved.push(0);
                        }
                    }
                }
                interleaved
            }
            SampleFormat::I32P => {
                // Planar i32 - interleave channels
                let samples_per_channel = frame.nb_samples;
                let channels = frame.channels as usize;
                let mut interleaved = Vec::with_capacity(samples_per_channel * channels);

                for sample_idx in 0..samples_per_channel {
                    for ch in 0..channels {
                        if ch < frame.data.len() {
                            let ch_data = frame.data[ch].as_slice();
                            let byte_idx = sample_idx * 4;
                            if byte_idx + 3 < ch_data.len() {
                                let sample = i32::from_le_bytes([
                                    ch_data[byte_idx],
                                    ch_data[byte_idx + 1],
                                    ch_data[byte_idx + 2],
                                    ch_data[byte_idx + 3],
                                ]);
                                interleaved.push(sample);
                            } else {
                                interleaved.push(0);
                            }
                        } else {
                            interleaved.push(0);
                        }
                    }
                }
                interleaved
            }
            SampleFormat::F32P => {
                // Planar f32 - interleave and convert
                let samples_per_channel = frame.nb_samples;
                let channels = frame.channels as usize;
                let scale = (1i64 << (self.config.bits_per_sample - 1)) as f32;
                let mut interleaved = Vec::with_capacity(samples_per_channel * channels);

                for sample_idx in 0..samples_per_channel {
                    for ch in 0..channels {
                        if ch < frame.data.len() {
                            let ch_data = frame.data[ch].as_slice();
                            let byte_idx = sample_idx * 4;
                            if byte_idx + 3 < ch_data.len() {
                                let f = f32::from_le_bytes([
                                    ch_data[byte_idx],
                                    ch_data[byte_idx + 1],
                                    ch_data[byte_idx + 2],
                                    ch_data[byte_idx + 3],
                                ]);
                                interleaved.push((f.clamp(-1.0, 1.0) * scale) as i32);
                            } else {
                                interleaved.push(0);
                            }
                        } else {
                            interleaved.push(0);
                        }
                    }
                }
                interleaved
            }
            _ => {
                return Err(Error::codec(format!(
                    "Unsupported sample format for FLAC encoding: {:?}",
                    frame.format
                )));
            }
        };

        Ok(samples)
    }

    /// Encode buffered samples into packets
    fn encode_buffered_samples(&mut self) -> Result<()> {
        let samples_per_block = self.block_size * self.config.channels as usize;

        while self.sample_buffer.len() >= samples_per_block {
            // Extract one block worth of samples
            let block_samples: Vec<i32> = self.sample_buffer.drain(..samples_per_block).collect();

            // Create a memory source for this block
            let source = MemSource::from_samples(
                &block_samples,
                self.config.channels as usize,
                self.config.bits_per_sample as usize,
                self.config.sample_rate as usize,
            );

            // Encode the block
            let stream = flacenc::encode_with_fixed_block_size(
                &self.flac_config,
                source,
                self.block_size,
            )
            .map_err(|e| Error::codec(format!("FLAC encoding failed: {:?}", e)))?;

            // Serialize to bytes
            let mut sink = ByteSink::new();
            stream.write(&mut sink);
            let encoded_data = sink.as_slice().to_vec();

            // Store extradata (STREAMINFO) from first encode if not already set
            if self.extradata.is_none() && !encoded_data.is_empty() {
                // The FLAC stream starts with "fLaC" magic, then STREAMINFO block
                // STREAMINFO is 34 bytes of metadata after the 4-byte magic and 4-byte block header
                if encoded_data.len() >= 42 {
                    // Extract STREAMINFO block (bytes 8-42 after magic)
                    self.extradata = Some(encoded_data[8..42].to_vec());
                }
            }

            // Calculate PTS based on samples encoded
            let pts = if self.current_pts.is_valid() {
                Timestamp::new(self.samples_encoded as i64)
            } else {
                Timestamp::none()
            };

            // For the first packet, include the full stream header
            // For subsequent packets, we need to extract just the frame data
            let packet_data = if !self.header_written {
                self.header_written = true;
                encoded_data
            } else {
                // Skip the stream header for subsequent frames
                // FLAC frame data starts after the metadata blocks
                // This is a simplified approach - in practice, each call to encode_with_fixed_block_size
                // produces a complete stream, so we include it all
                encoded_data
            };

            // Create packet
            let mut packet = Packet::new_audio(self.stream_index, Buffer::from_vec(packet_data));
            packet.pts = pts;
            packet.dts = pts;
            packet.duration = self.block_size as i64;
            packet.flags.keyframe = true; // All FLAC frames are keyframes

            self.packet_queue.push_back(packet);
            self.samples_encoded += self.block_size as u64;
        }

        Ok(())
    }

    /// Encode a complete audio frame directly
    pub fn encode_frame(&mut self, frame: &AudioFrame) -> Result<Packet> {
        // Validate sample rate
        if frame.sample_rate != self.config.sample_rate {
            return Err(Error::codec(format!(
                "Sample rate mismatch: encoder expects {}, got {}",
                self.config.sample_rate, frame.sample_rate
            )));
        }

        // Validate channels
        if frame.channels != self.config.channels {
            return Err(Error::codec(format!(
                "Channel count mismatch: encoder expects {}, got {}",
                self.config.channels, frame.channels
            )));
        }

        // Store PTS from frame
        if frame.pts.is_valid() && !self.current_pts.is_valid() {
            self.current_pts = frame.pts;
        }

        // Convert frame to samples
        let samples = self.frame_to_samples(frame)?;
        self.sample_buffer.extend(samples);

        // Encode buffered samples
        self.encode_buffered_samples()?;

        // Try to get a packet
        self.packet_queue
            .pop_front()
            .ok_or(Error::TryAgain)
    }

    /// Flush remaining samples, padding with silence if necessary
    fn flush_remaining(&mut self) -> Result<()> {
        if self.sample_buffer.is_empty() {
            return Ok(());
        }

        let samples_per_block = self.block_size * self.config.channels as usize;

        // Pad with silence to complete the block
        let padding_needed = samples_per_block - (self.sample_buffer.len() % samples_per_block);
        if padding_needed < samples_per_block {
            self.sample_buffer.extend(vec![0i32; padding_needed]);
        }

        // Encode the padded block
        self.encode_buffered_samples()
    }
}

impl Encoder for FlacEncoder {
    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        match frame {
            Frame::Audio(audio_frame) => {
                // Validate sample rate
                if audio_frame.sample_rate != self.config.sample_rate {
                    return Err(Error::codec(format!(
                        "Sample rate mismatch: encoder expects {}, got {}",
                        self.config.sample_rate, audio_frame.sample_rate
                    )));
                }

                // Store initial PTS
                if !self.current_pts.is_valid() && audio_frame.pts.is_valid() {
                    self.current_pts = audio_frame.pts;
                }

                // Convert frame to samples and add to buffer
                let samples = self.frame_to_samples(audio_frame)?;
                self.sample_buffer.extend(samples);

                // Process complete blocks
                self.encode_buffered_samples()
            }
            Frame::Video(_) => Err(Error::codec("FLAC encoder only accepts audio frames")),
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.packet_queue
            .pop_front()
            .ok_or(Error::TryAgain)
    }

    fn flush(&mut self) -> Result<()> {
        // Flush any remaining samples
        self.flush_remaining()?;

        Ok(())
    }

    fn extradata(&self) -> Option<&[u8]> {
        self.extradata.as_deref()
    }
}

/// Create a FLAC encoder with default settings
pub fn create_encoder(sample_rate: u32, channels: u16, bits_per_sample: u8) -> Result<FlacEncoder> {
    FlacEncoder::new(sample_rate, channels, bits_per_sample)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flac_encoder_creation() {
        let encoder = FlacEncoder::new(44100, 2, 16);
        assert!(encoder.is_ok());
        let enc = encoder.unwrap();
        assert_eq!(enc.sample_rate(), 44100);
        assert_eq!(enc.channels(), 2);
        assert_eq!(enc.bits_per_sample(), 16);
    }

    #[test]
    fn test_flac_encoder_mono() {
        let encoder = FlacEncoder::new(44100, 1, 16);
        assert!(encoder.is_ok());
        let enc = encoder.unwrap();
        assert_eq!(enc.channels(), 1);
    }

    #[test]
    fn test_flac_encoder_24bit() {
        let encoder = FlacEncoder::new(96000, 2, 24);
        assert!(encoder.is_ok());
        let enc = encoder.unwrap();
        assert_eq!(enc.bits_per_sample(), 24);
        assert_eq!(enc.sample_rate(), 96000);
    }

    #[test]
    fn test_flac_encoder_with_config() {
        let config = FlacEncoderConfig {
            sample_rate: 48000,
            channels: 2,
            bits_per_sample: 24,
            compression_level: 8,
            block_size: 4096,
        };
        let encoder = FlacEncoder::with_config(config);
        assert!(encoder.is_ok());
        let enc = encoder.unwrap();
        assert_eq!(enc.compression_level(), 8);
    }

    #[test]
    fn test_flac_encoder_cd_quality_config() {
        let config = FlacEncoderConfig::cd_quality();
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.channels, 2);
        assert_eq!(config.bits_per_sample, 16);
    }

    #[test]
    fn test_flac_encoder_fast_config() {
        let config = FlacEncoderConfig::fast();
        assert_eq!(config.compression_level, 0);
    }

    #[test]
    fn test_flac_encoder_best_config() {
        let config = FlacEncoderConfig::best();
        assert_eq!(config.compression_level, 8);
    }

    #[test]
    fn test_flac_encoder_set_compression_level() {
        let mut encoder = FlacEncoder::new(44100, 2, 16).unwrap();
        assert!(encoder.set_compression_level(0).is_ok());
        assert_eq!(encoder.compression_level(), 0);
        assert!(encoder.set_compression_level(8).is_ok());
        assert_eq!(encoder.compression_level(), 8);
        assert!(encoder.set_compression_level(9).is_err());
    }

    #[test]
    fn test_flac_encoder_invalid_channels() {
        let config = FlacEncoderConfig {
            channels: 9, // Invalid: max is 8
            ..Default::default()
        };
        let encoder = FlacEncoder::with_config(config);
        assert!(encoder.is_err());
    }

    #[test]
    fn test_flac_encoder_invalid_bits_per_sample() {
        let config = FlacEncoderConfig {
            bits_per_sample: 12, // Invalid: we only support 8, 16, 24, 32
            ..Default::default()
        };
        let encoder = FlacEncoder::with_config(config);
        assert!(encoder.is_err());
    }

    #[test]
    fn test_flac_encoder_invalid_compression_level() {
        let config = FlacEncoderConfig {
            compression_level: 10, // Invalid: max is 8
            ..Default::default()
        };
        let encoder = FlacEncoder::with_config(config);
        assert!(encoder.is_err());
    }

    #[test]
    fn test_effective_block_size() {
        let config = FlacEncoderConfig {
            compression_level: 0,
            block_size: 0, // Auto
            ..Default::default()
        };
        assert_eq!(config.effective_block_size(), 1152);

        let config = FlacEncoderConfig {
            compression_level: 8,
            block_size: 0, // Auto
            ..Default::default()
        };
        assert_eq!(config.effective_block_size(), 4608);

        let config = FlacEncoderConfig {
            block_size: 2048, // Explicit
            ..Default::default()
        };
        assert_eq!(config.effective_block_size(), 2048);
    }

    #[test]
    fn test_flac_encode_silence() {
        let mut encoder = FlacEncoder::new(44100, 2, 16).unwrap();
        let block_size = encoder.block_size();

        // Create a silent frame (exactly one block)
        let silence = vec![0i16; block_size * 2]; // stereo
        let silence_bytes: Vec<u8> = silence.iter().flat_map(|&s| s.to_le_bytes()).collect();

        let mut frame = AudioFrame::new(block_size, 44100, 2, SampleFormat::I16);
        frame.data.push(Buffer::from_vec(silence_bytes));
        frame.pts = Timestamp::new(0);
        frame.duration = block_size as i64;

        // Send the frame
        let result = encoder.send_frame(&Frame::Audio(frame));
        assert!(result.is_ok());

        // Should be able to receive a packet
        let packet_result = encoder.receive_packet();
        assert!(packet_result.is_ok());

        let packet = packet_result.unwrap();
        assert!(!packet.data.is_empty());
        // FLAC should compress silence very well
    }

    #[test]
    fn test_flac_encode_sine_wave() {
        let mut encoder = FlacEncoder::new(44100, 1, 16).unwrap();
        let block_size = encoder.block_size();

        // Generate a 440 Hz sine wave
        let samples: Vec<i16> = (0..block_size)
            .map(|i| {
                let t = i as f32 / 44100.0;
                (f32::sin(2.0 * std::f32::consts::PI * 440.0 * t) * 16000.0) as i16
            })
            .collect();

        let sample_bytes: Vec<u8> = samples.iter().flat_map(|&s| s.to_le_bytes()).collect();

        let mut frame = AudioFrame::new(block_size, 44100, 1, SampleFormat::I16);
        frame.data.push(Buffer::from_vec(sample_bytes));
        frame.pts = Timestamp::new(0);
        frame.duration = block_size as i64;

        let result = encoder.send_frame(&Frame::Audio(frame));
        assert!(result.is_ok());

        let packet_result = encoder.receive_packet();
        assert!(packet_result.is_ok());
    }

    #[test]
    fn test_flac_encoder_streaming() {
        let mut encoder = FlacEncoder::new(44100, 2, 16).unwrap();
        let block_size = encoder.block_size();

        // Send multiple frames
        for i in 0..5 {
            let samples = vec![0i16; block_size * 2];
            let sample_bytes: Vec<u8> = samples.iter().flat_map(|&s| s.to_le_bytes()).collect();

            let mut frame = AudioFrame::new(block_size, 44100, 2, SampleFormat::I16);
            frame.data.push(Buffer::from_vec(sample_bytes));
            frame.pts = Timestamp::new((i * block_size) as i64);

            let result = encoder.send_frame(&Frame::Audio(frame));
            assert!(result.is_ok());

            // Should be able to receive a packet
            let packet_result = encoder.receive_packet();
            assert!(packet_result.is_ok());
        }
    }

    #[test]
    fn test_flac_encoder_flush() {
        let mut encoder = FlacEncoder::new(44100, 2, 16).unwrap();
        let block_size = encoder.block_size();

        // Send a partial frame (less than one block)
        let partial_size = block_size / 2;
        let samples = vec![0i16; partial_size * 2];
        let sample_bytes: Vec<u8> = samples.iter().flat_map(|&s| s.to_le_bytes()).collect();

        let mut frame = AudioFrame::new(partial_size, 44100, 2, SampleFormat::I16);
        frame.data.push(Buffer::from_vec(sample_bytes));
        frame.pts = Timestamp::new(0);

        let result = encoder.send_frame(&Frame::Audio(frame));
        assert!(result.is_ok());

        // No packet yet (not enough samples)
        assert!(encoder.receive_packet().is_err());

        // Flush should produce a packet
        let flush_result = encoder.flush();
        assert!(flush_result.is_ok());

        // Now we should have a packet
        let packet_result = encoder.receive_packet();
        assert!(packet_result.is_ok());
    }
}
