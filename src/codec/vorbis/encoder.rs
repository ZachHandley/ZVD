//! Vorbis audio encoder using vorbis_rs
//!
//! Vorbis is an open-source, patent-free lossy audio codec.
//! This encoder uses the vorbis_rs crate which provides bindings to libvorbis
//! with the aoTuV and Lancer patchsets for improved encoding quality.
//!
//! ## Features
//! - Quality-based VBR encoding (quality -0.1 to 1.0)
//! - Average bitrate (ABR) encoding
//! - Up to 255 channels (though stereo is typical)
//! - Vorbis comment (metadata) support
//!
//! ## Example
//! ```ignore
//! // Quality-based encoding
//! let encoder = VorbisEncoder::new_quality(44100, 2, 0.5)?;
//!
//! // Or bitrate-based encoding
//! let encoder = VorbisEncoder::new_bitrate(44100, 2, 128000)?;
//!
//! encoder.send_frame(&frame)?;
//! let packet = encoder.receive_packet()?;
//! ```

use crate::codec::{AudioFrame, Encoder, Frame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, SampleFormat, Timestamp};
use std::collections::VecDeque;
use std::num::{NonZeroU8, NonZeroU32};
use vorbis_rs::{VorbisBitrateManagementStrategy, VorbisEncoderBuilder};

/// Default frame size for Vorbis encoding (samples per channel)
const DEFAULT_FRAME_SIZE: usize = 1024;

/// Vorbis encoder configuration
#[derive(Debug, Clone)]
pub struct VorbisEncoderConfig {
    /// Sample rate in Hz (e.g., 44100, 48000)
    pub sample_rate: u32,
    /// Number of audio channels (1 = mono, 2 = stereo)
    pub channels: u16,
    /// Encoding mode
    pub mode: VorbisEncodingMode,
    /// Vorbis comments (metadata)
    pub comments: Vec<(String, String)>,
}

/// Vorbis encoding mode
#[derive(Debug, Clone)]
pub enum VorbisEncodingMode {
    /// Quality-based VBR (-0.1 to 1.0)
    /// -0.1 = ~45 kbps, 0.0 = ~64 kbps, 0.5 = ~128 kbps, 1.0 = ~256+ kbps
    Quality(f32),
    /// Average bitrate (ABR) in bits per second
    Bitrate(u32),
    /// Constrained bitrate (min, nominal, max) in bits per second
    Constrained {
        min_bitrate: Option<u32>,
        nominal_bitrate: u32,
        max_bitrate: Option<u32>,
    },
}

impl Default for VorbisEncoderConfig {
    fn default() -> Self {
        VorbisEncoderConfig {
            sample_rate: 44100,
            channels: 2,
            mode: VorbisEncodingMode::Quality(0.5),
            comments: Vec::new(),
        }
    }
}

impl VorbisEncoderConfig {
    /// Create a configuration for quality-based encoding
    pub fn quality(sample_rate: u32, channels: u16, quality: f32) -> Self {
        VorbisEncoderConfig {
            sample_rate,
            channels,
            mode: VorbisEncodingMode::Quality(quality.clamp(-0.1, 1.0)),
            ..Default::default()
        }
    }

    /// Create a configuration for bitrate-based encoding
    pub fn bitrate(sample_rate: u32, channels: u16, bitrate: u32) -> Self {
        VorbisEncoderConfig {
            sample_rate,
            channels,
            mode: VorbisEncodingMode::Bitrate(bitrate),
            ..Default::default()
        }
    }

    /// Create a low quality configuration (smaller files)
    pub fn low_quality(sample_rate: u32, channels: u16) -> Self {
        VorbisEncoderConfig {
            sample_rate,
            channels,
            mode: VorbisEncodingMode::Quality(0.0), // ~64 kbps
            ..Default::default()
        }
    }

    /// Create a medium quality configuration (good balance)
    pub fn medium_quality(sample_rate: u32, channels: u16) -> Self {
        VorbisEncoderConfig {
            sample_rate,
            channels,
            mode: VorbisEncodingMode::Quality(0.5), // ~128 kbps
            ..Default::default()
        }
    }

    /// Create a high quality configuration (best quality)
    pub fn high_quality(sample_rate: u32, channels: u16) -> Self {
        VorbisEncoderConfig {
            sample_rate,
            channels,
            mode: VorbisEncodingMode::Quality(0.9), // ~256 kbps
            ..Default::default()
        }
    }

    /// Add a Vorbis comment (metadata tag)
    pub fn add_comment<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        self.comments.push((key.into(), value.into()));
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        // Validate sample rate (Vorbis supports a wide range)
        if self.sample_rate == 0 || self.sample_rate > 200000 {
            return Err(Error::codec(format!(
                "Invalid Vorbis sample rate: {}. Valid range: 1-200000 Hz",
                self.sample_rate
            )));
        }

        // Validate channels
        if self.channels == 0 || self.channels > 255 {
            return Err(Error::codec(format!(
                "Invalid Vorbis channel count: {}. Valid range: 1-255",
                self.channels
            )));
        }

        // Validate quality/bitrate
        match &self.mode {
            VorbisEncodingMode::Quality(q) => {
                if *q < -0.1 || *q > 1.0 {
                    return Err(Error::codec(format!(
                        "Invalid Vorbis quality: {}. Valid range: -0.1 to 1.0",
                        q
                    )));
                }
            }
            VorbisEncodingMode::Bitrate(b) => {
                if *b < 16000 || *b > 500000 {
                    return Err(Error::codec(format!(
                        "Invalid Vorbis bitrate: {}. Valid range: 16000-500000 bps",
                        b
                    )));
                }
            }
            VorbisEncodingMode::Constrained {
                nominal_bitrate, ..
            } => {
                if *nominal_bitrate < 16000 || *nominal_bitrate > 500000 {
                    return Err(Error::codec(format!(
                        "Invalid Vorbis nominal bitrate: {}. Valid range: 16000-500000 bps",
                        nominal_bitrate
                    )));
                }
            }
        }

        Ok(())
    }
}

/// Vorbis audio encoder
///
/// Encodes PCM audio to Ogg Vorbis format using the vorbis_rs library.
/// This encoder accumulates samples and encodes them in blocks.
pub struct VorbisEncoder {
    /// Encoder configuration
    config: VorbisEncoderConfig,
    /// Input sample buffer (planar f32 samples per channel)
    sample_buffer: Vec<Vec<f32>>,
    /// Output packet queue
    packet_queue: VecDeque<Packet>,
    /// Current presentation timestamp
    current_pts: Timestamp,
    /// Total samples encoded (for PTS calculation)
    samples_encoded: u64,
    /// Stream index for output packets
    stream_index: usize,
    /// Ogg Vorbis header data (for extradata)
    extradata: Option<Vec<u8>>,
    /// Frame size for encoding
    frame_size: usize,
    /// Whether the encoder has been initialized
    initialized: bool,
    /// Accumulated output data from encoding
    output_buffer: Vec<u8>,
}

impl VorbisEncoder {
    /// Create a new Vorbis encoder with quality-based encoding
    ///
    /// # Arguments
    /// * `sample_rate` - Audio sample rate in Hz
    /// * `channels` - Number of audio channels
    /// * `quality` - Quality factor (-0.1 to 1.0)
    ///   - -0.1: ~45 kbps, very small files
    ///   - 0.0: ~64 kbps, acceptable quality
    ///   - 0.5: ~128 kbps, good quality (default)
    ///   - 1.0: ~256+ kbps, transparent quality
    pub fn new_quality(sample_rate: u32, channels: u16, quality: f32) -> Result<Self> {
        let config = VorbisEncoderConfig::quality(sample_rate, channels, quality);
        Self::with_config(config)
    }

    /// Create a new Vorbis encoder with bitrate-based encoding (ABR)
    ///
    /// # Arguments
    /// * `sample_rate` - Audio sample rate in Hz
    /// * `channels` - Number of audio channels
    /// * `bitrate` - Target bitrate in bits per second
    pub fn new_bitrate(sample_rate: u32, channels: u16, bitrate: u32) -> Result<Self> {
        let config = VorbisEncoderConfig::bitrate(sample_rate, channels, bitrate);
        Self::with_config(config)
    }

    /// Create a new Vorbis encoder with custom configuration
    pub fn with_config(config: VorbisEncoderConfig) -> Result<Self> {
        config.validate()?;

        // Initialize sample buffers for each channel
        let sample_buffer: Vec<Vec<f32>> = (0..config.channels)
            .map(|_| Vec::with_capacity(DEFAULT_FRAME_SIZE * 2))
            .collect();

        Ok(VorbisEncoder {
            config,
            sample_buffer,
            packet_queue: VecDeque::new(),
            current_pts: Timestamp::none(),
            samples_encoded: 0,
            stream_index: 0,
            extradata: None,
            frame_size: DEFAULT_FRAME_SIZE,
            initialized: false,
            output_buffer: Vec::new(),
        })
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

    /// Get the encoding quality (if quality mode)
    pub fn quality(&self) -> Option<f32> {
        match &self.config.mode {
            VorbisEncodingMode::Quality(q) => Some(*q),
            _ => None,
        }
    }

    /// Get the target bitrate (if bitrate mode)
    pub fn bitrate(&self) -> Option<u32> {
        match &self.config.mode {
            VorbisEncodingMode::Bitrate(b) => Some(*b),
            VorbisEncodingMode::Constrained {
                nominal_bitrate, ..
            } => Some(*nominal_bitrate),
            _ => None,
        }
    }

    /// Add a Vorbis comment (metadata tag)
    pub fn add_comment<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        self.config.comments.push((key.into(), value.into()));
    }

    /// Convert audio frame to planar f32 samples
    fn frame_to_planar_samples(&self, frame: &AudioFrame) -> Result<Vec<Vec<f32>>> {
        if frame.data.is_empty() {
            return Err(Error::codec("Empty audio frame"));
        }

        let channels = self.config.channels as usize;
        let mut planar_samples: Vec<Vec<f32>> = vec![Vec::new(); channels];

        match frame.format {
            // Interleaved formats - need to de-interleave
            SampleFormat::I16 => {
                let data = frame.data[0].as_slice();
                let samples: Vec<i16> = data
                    .chunks_exact(2)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();

                for (i, sample) in samples.iter().enumerate() {
                    let ch = i % channels;
                    planar_samples[ch].push(*sample as f32 / 32768.0);
                }
            }
            SampleFormat::I32 => {
                let data = frame.data[0].as_slice();
                let samples: Vec<i32> = data
                    .chunks_exact(4)
                    .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();

                for (i, sample) in samples.iter().enumerate() {
                    let ch = i % channels;
                    planar_samples[ch].push(*sample as f32 / 2147483648.0);
                }
            }
            SampleFormat::F32 => {
                let data = frame.data[0].as_slice();
                let samples: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();

                for (i, sample) in samples.iter().enumerate() {
                    let ch = i % channels;
                    planar_samples[ch].push(*sample);
                }
            }
            SampleFormat::F64 => {
                let data = frame.data[0].as_slice();
                let samples: Vec<f64> = data
                    .chunks_exact(8)
                    .map(|chunk| {
                        f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ])
                    })
                    .collect();

                for (i, sample) in samples.iter().enumerate() {
                    let ch = i % channels;
                    planar_samples[ch].push(*sample as f32);
                }
            }
            SampleFormat::U8 => {
                let data = frame.data[0].as_slice();
                for (i, &sample) in data.iter().enumerate() {
                    let ch = i % channels;
                    planar_samples[ch].push((sample as f32 - 128.0) / 128.0);
                }
            }
            // Planar formats - already separated
            SampleFormat::I16P => {
                for (ch, ch_buf) in planar_samples.iter_mut().enumerate().take(channels) {
                    if ch < frame.data.len() {
                        let ch_data = frame.data[ch].as_slice();
                        for chunk in ch_data.chunks_exact(2) {
                            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                            ch_buf.push(sample as f32 / 32768.0);
                        }
                    }
                }
            }
            SampleFormat::I32P => {
                for (ch, ch_buf) in planar_samples.iter_mut().enumerate().take(channels) {
                    if ch < frame.data.len() {
                        let ch_data = frame.data[ch].as_slice();
                        for chunk in ch_data.chunks_exact(4) {
                            let sample =
                                i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                            ch_buf.push(sample as f32 / 2147483648.0);
                        }
                    }
                }
            }
            SampleFormat::F32P => {
                for (ch, ch_buf) in planar_samples.iter_mut().enumerate().take(channels) {
                    if ch < frame.data.len() {
                        let ch_data = frame.data[ch].as_slice();
                        for chunk in ch_data.chunks_exact(4) {
                            let sample =
                                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                            ch_buf.push(sample);
                        }
                    }
                }
            }
            SampleFormat::F64P => {
                for (ch, ch_buf) in planar_samples.iter_mut().enumerate().take(channels) {
                    if ch < frame.data.len() {
                        let ch_data = frame.data[ch].as_slice();
                        for chunk in ch_data.chunks_exact(8) {
                            let sample = f64::from_le_bytes([
                                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5],
                                chunk[6], chunk[7],
                            ]);
                            ch_buf.push(sample as f32);
                        }
                    }
                }
            }
            _ => {
                return Err(Error::codec(format!(
                    "Unsupported sample format for Vorbis encoding: {:?}",
                    frame.format
                )));
            }
        }

        Ok(planar_samples)
    }

    /// Encode accumulated samples
    fn encode_accumulated(&mut self) -> Result<()> {
        // Check if we have enough samples
        let min_samples = self.frame_size;
        if self.sample_buffer.is_empty() || self.sample_buffer[0].len() < min_samples {
            return Ok(());
        }

        // Determine how many complete frames we can encode
        let samples_available = self.sample_buffer[0].len();
        let frames_to_encode = samples_available / self.frame_size;

        if frames_to_encode == 0 {
            return Ok(());
        }

        let samples_to_encode = frames_to_encode * self.frame_size;

        // Extract samples for encoding
        let mut encode_samples: Vec<Vec<f32>> = Vec::with_capacity(self.config.channels as usize);
        for ch_buffer in &mut self.sample_buffer {
            let samples: Vec<f32> = ch_buffer.drain(..samples_to_encode).collect();
            encode_samples.push(samples);
        }

        // Create output buffer
        let mut output = Vec::new();

        // Build the encoder
        let bitrate_strategy = match &self.config.mode {
            VorbisEncodingMode::Quality(q) => VorbisBitrateManagementStrategy::QualityVbr {
                target_quality: *q,
            },
            VorbisEncodingMode::Bitrate(b) => {
                let target = NonZeroU32::new(*b).unwrap_or(NonZeroU32::new(128000).unwrap());
                VorbisBitrateManagementStrategy::Vbr {
                    target_bitrate: target,
                }
            },
            VorbisEncodingMode::Constrained {
                nominal_bitrate,
                max_bitrate,
                ..
            } => {
                // Use ConstrainedAbr if max_bitrate is provided, otherwise use Abr
                if let Some(max_br) = max_bitrate {
                    let max = NonZeroU32::new(*max_br).unwrap_or(NonZeroU32::new(320000).unwrap());
                    VorbisBitrateManagementStrategy::ConstrainedAbr {
                        maximum_bitrate: max,
                    }
                } else {
                    let avg = NonZeroU32::new(*nominal_bitrate).unwrap_or(NonZeroU32::new(128000).unwrap());
                    VorbisBitrateManagementStrategy::Abr {
                        average_bitrate: avg,
                    }
                }
            },
        };

        // Build comments
        let comments: Vec<(&str, &str)> = self
            .config
            .comments
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        // Convert sample rate and channels to NonZero types
        let sample_rate = NonZeroU32::new(self.config.sample_rate)
            .ok_or_else(|| Error::codec("Sample rate cannot be zero"))?;
        let channels = NonZeroU8::new(self.config.channels as u8)
            .ok_or_else(|| Error::codec("Channel count cannot be zero"))?;

        // Create encoder using the builder
        let mut encoder_builder = VorbisEncoderBuilder::new(
            sample_rate,
            channels,
            &mut output,
        )
        .map_err(|e| Error::codec(format!("Failed to create Vorbis encoder builder: {:?}", e)))?;

        encoder_builder.bitrate_management_strategy(bitrate_strategy);

        for (key, value) in &comments {
            encoder_builder.comment_tag(*key, *value);
        }

        let mut encoder = encoder_builder
            .build()
            .map_err(|e| Error::codec(format!("Failed to build Vorbis encoder: {:?}", e)))?;

        // Encode the audio block
        // Convert to the format expected by vorbis_rs: &[impl AsRef<[f32]>]
        let samples_refs: Vec<&[f32]> = encode_samples.iter().map(|v| v.as_slice()).collect();

        encoder
            .encode_audio_block(samples_refs)
            .map_err(|e| Error::codec(format!("Vorbis encoding failed: {:?}", e)))?;

        // Finish encoding to get all data
        encoder
            .finish()
            .map_err(|e| Error::codec(format!("Failed to finish Vorbis encoding: {:?}", e)))?;

        // Store extradata from first successful encode (Vorbis headers are embedded in Ogg)
        if self.extradata.is_none() && !output.is_empty() {
            // For Ogg Vorbis, the extradata is typically the first few Ogg pages containing
            // the identification, comment, and setup headers
            // We'll store the entire initial output as extradata for container muxing
            self.extradata = Some(output.clone());
        }

        // Calculate PTS
        let pts = if self.current_pts.is_valid() {
            Timestamp::new(self.samples_encoded as i64)
        } else {
            Timestamp::none()
        };

        // Create packet
        let mut packet = Packet::new_audio(self.stream_index, Buffer::from_vec(output));
        packet.pts = pts;
        packet.dts = pts;
        packet.duration = samples_to_encode as i64;
        packet.flags.keyframe = true; // Vorbis frames are effectively independent

        self.packet_queue.push_back(packet);
        self.samples_encoded += samples_to_encode as u64;

        Ok(())
    }

    /// Flush remaining samples
    fn flush_remaining(&mut self) -> Result<()> {
        // If we have any remaining samples, encode them
        if !self.sample_buffer.is_empty() && !self.sample_buffer[0].is_empty() {
            let remaining = self.sample_buffer[0].len();

            // Pad to frame size if needed
            let padding_needed = if remaining % self.frame_size != 0 {
                self.frame_size - (remaining % self.frame_size)
            } else {
                0
            };

            if padding_needed > 0 {
                for ch_buffer in &mut self.sample_buffer {
                    ch_buffer.extend(vec![0.0f32; padding_needed]);
                }
            }

            self.encode_accumulated()?;
        }

        Ok(())
    }
}

impl Encoder for VorbisEncoder {
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

                // Validate channels
                if audio_frame.channels != self.config.channels {
                    return Err(Error::codec(format!(
                        "Channel count mismatch: encoder expects {}, got {}",
                        self.config.channels, audio_frame.channels
                    )));
                }

                // Store initial PTS
                if !self.current_pts.is_valid() && audio_frame.pts.is_valid() {
                    self.current_pts = audio_frame.pts;
                }

                // Convert frame to planar samples
                let planar_samples = self.frame_to_planar_samples(audio_frame)?;

                // Add samples to buffer
                for (ch, samples) in planar_samples.into_iter().enumerate() {
                    if ch < self.sample_buffer.len() {
                        self.sample_buffer[ch].extend(samples);
                    }
                }

                // Try to encode complete blocks
                self.encode_accumulated()
            }
            Frame::Video(_) => Err(Error::codec("Vorbis encoder only accepts audio frames")),
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.packet_queue.pop_front().ok_or(Error::TryAgain)
    }

    fn flush(&mut self) -> Result<()> {
        self.flush_remaining()
    }

    fn extradata(&self) -> Option<&[u8]> {
        self.extradata.as_deref()
    }
}

/// Create a Vorbis encoder with quality-based encoding
pub fn create_encoder_quality(sample_rate: u32, channels: u16, quality: f32) -> Result<VorbisEncoder> {
    VorbisEncoder::new_quality(sample_rate, channels, quality)
}

/// Create a Vorbis encoder with bitrate-based encoding
pub fn create_encoder_bitrate(sample_rate: u32, channels: u16, bitrate: u32) -> Result<VorbisEncoder> {
    VorbisEncoder::new_bitrate(sample_rate, channels, bitrate)
}

/// Create a Vorbis encoder with default quality (0.5)
pub fn create_encoder(sample_rate: u32, channels: u16, quality: f32) -> Result<VorbisEncoder> {
    VorbisEncoder::new_quality(sample_rate, channels, quality)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vorbis_encoder_creation_quality() {
        let encoder = VorbisEncoder::new_quality(44100, 2, 0.5);
        assert!(encoder.is_ok());
        let enc = encoder.unwrap();
        assert_eq!(enc.sample_rate(), 44100);
        assert_eq!(enc.channels(), 2);
        assert_eq!(enc.quality(), Some(0.5));
    }

    #[test]
    fn test_vorbis_encoder_creation_bitrate() {
        let encoder = VorbisEncoder::new_bitrate(44100, 2, 128000);
        assert!(encoder.is_ok());
        let enc = encoder.unwrap();
        assert_eq!(enc.bitrate(), Some(128000));
    }

    #[test]
    fn test_vorbis_encoder_mono() {
        let encoder = VorbisEncoder::new_quality(44100, 1, 0.5);
        assert!(encoder.is_ok());
        let enc = encoder.unwrap();
        assert_eq!(enc.channels(), 1);
    }

    #[test]
    fn test_vorbis_encoder_with_config() {
        let config = VorbisEncoderConfig {
            sample_rate: 48000,
            channels: 2,
            mode: VorbisEncodingMode::Quality(0.7),
            comments: vec![
                ("TITLE".to_string(), "Test Audio".to_string()),
                ("ARTIST".to_string(), "Test Artist".to_string()),
            ],
        };
        let encoder = VorbisEncoder::with_config(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_vorbis_encoder_low_quality_config() {
        let config = VorbisEncoderConfig::low_quality(44100, 2);
        let encoder = VorbisEncoder::with_config(config);
        assert!(encoder.is_ok());
        assert_eq!(encoder.unwrap().quality(), Some(0.0));
    }

    #[test]
    fn test_vorbis_encoder_medium_quality_config() {
        let config = VorbisEncoderConfig::medium_quality(44100, 2);
        let encoder = VorbisEncoder::with_config(config);
        assert!(encoder.is_ok());
        assert_eq!(encoder.unwrap().quality(), Some(0.5));
    }

    #[test]
    fn test_vorbis_encoder_high_quality_config() {
        let config = VorbisEncoderConfig::high_quality(44100, 2);
        let encoder = VorbisEncoder::with_config(config);
        assert!(encoder.is_ok());
        assert_eq!(encoder.unwrap().quality(), Some(0.9));
    }

    #[test]
    fn test_vorbis_encoder_invalid_quality() {
        let config = VorbisEncoderConfig {
            mode: VorbisEncodingMode::Quality(1.5), // Invalid: max is 1.0
            ..Default::default()
        };
        let encoder = VorbisEncoder::with_config(config);
        assert!(encoder.is_err());
    }

    #[test]
    fn test_vorbis_encoder_invalid_channels() {
        let encoder = VorbisEncoder::new_quality(44100, 0, 0.5);
        assert!(encoder.is_err());
    }

    #[test]
    fn test_vorbis_encoder_invalid_sample_rate() {
        let encoder = VorbisEncoder::new_quality(0, 2, 0.5);
        assert!(encoder.is_err());
    }

    #[test]
    fn test_vorbis_encoder_add_comment() {
        let mut encoder = VorbisEncoder::new_quality(44100, 2, 0.5).unwrap();
        encoder.add_comment("TITLE", "Test Song");
        encoder.add_comment("ARTIST", "Test Artist");
        // Comments are stored in config
        assert_eq!(encoder.config.comments.len(), 2);
    }

    #[test]
    fn test_vorbis_quality_clamping() {
        // Quality should be clamped to valid range
        let config = VorbisEncoderConfig::quality(44100, 2, -0.5); // Below min
        assert!(
            matches!(config.mode, VorbisEncodingMode::Quality(q) if (-0.11..=-0.09).contains(&q))
        );

        let config = VorbisEncoderConfig::quality(44100, 2, 2.0); // Above max
        assert!(matches!(config.mode, VorbisEncodingMode::Quality(q) if (0.99..=1.01).contains(&q)));
    }

    #[test]
    fn test_vorbis_encode_silence() {
        let mut encoder = VorbisEncoder::new_quality(44100, 2, 0.5).unwrap();
        let frame_size = 1024;

        // Create a silent frame
        let silence = vec![0.0f32; frame_size * 2]; // stereo
        let silence_bytes: Vec<u8> = silence.iter().flat_map(|&s| s.to_le_bytes()).collect();

        let mut frame = AudioFrame::new(frame_size, 44100, 2, SampleFormat::F32);
        frame.data.push(Buffer::from_vec(silence_bytes));
        frame.pts = Timestamp::new(0);
        frame.duration = frame_size as i64;

        // Send the frame
        let result = encoder.send_frame(&Frame::Audio(frame));
        assert!(result.is_ok());

        // Should be able to receive a packet
        let packet_result = encoder.receive_packet();
        assert!(packet_result.is_ok());

        let packet = packet_result.unwrap();
        assert!(!packet.data.is_empty());
    }

    #[test]
    fn test_vorbis_encode_sine_wave() {
        let mut encoder = VorbisEncoder::new_quality(44100, 1, 0.5).unwrap();
        let frame_size = 1024;

        // Generate a 440 Hz sine wave
        let samples: Vec<f32> = (0..frame_size)
            .map(|i| {
                let t = i as f32 / 44100.0;
                f32::sin(2.0 * std::f32::consts::PI * 440.0 * t) * 0.5
            })
            .collect();

        let sample_bytes: Vec<u8> = samples.iter().flat_map(|&s| s.to_le_bytes()).collect();

        let mut frame = AudioFrame::new(frame_size, 44100, 1, SampleFormat::F32);
        frame.data.push(Buffer::from_vec(sample_bytes));
        frame.pts = Timestamp::new(0);
        frame.duration = frame_size as i64;

        let result = encoder.send_frame(&Frame::Audio(frame));
        assert!(result.is_ok());

        let packet_result = encoder.receive_packet();
        assert!(packet_result.is_ok());
    }

    #[test]
    fn test_vorbis_encoder_streaming() {
        let mut encoder = VorbisEncoder::new_quality(44100, 2, 0.5).unwrap();
        let frame_size = 1024;

        // Send multiple frames
        for i in 0..5 {
            let samples = vec![0.0f32; frame_size * 2];
            let sample_bytes: Vec<u8> = samples.iter().flat_map(|&s| s.to_le_bytes()).collect();

            let mut frame = AudioFrame::new(frame_size, 44100, 2, SampleFormat::F32);
            frame.data.push(Buffer::from_vec(sample_bytes));
            frame.pts = Timestamp::new((i * frame_size) as i64);

            let result = encoder.send_frame(&Frame::Audio(frame));
            assert!(result.is_ok());

            // Should be able to receive a packet
            let packet_result = encoder.receive_packet();
            assert!(packet_result.is_ok());
        }
    }

    #[test]
    fn test_vorbis_encoder_flush() {
        let mut encoder = VorbisEncoder::new_quality(44100, 2, 0.5).unwrap();
        let frame_size = 512; // Half the block size

        // Send a partial frame
        let samples = vec![0.0f32; frame_size * 2];
        let sample_bytes: Vec<u8> = samples.iter().flat_map(|&s| s.to_le_bytes()).collect();

        let mut frame = AudioFrame::new(frame_size, 44100, 2, SampleFormat::F32);
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
