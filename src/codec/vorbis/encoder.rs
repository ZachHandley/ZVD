//! Vorbis audio encoder
//!
//! This module provides a Vorbis encoder interface for lossy audio compression.
//!
//! ## Important Note
//!
//! **For new projects, use Opus instead of Vorbis.** Opus provides:
//! - Better quality at all bitrates
//! - Lower latency
//! - More flexible frame sizes
//! - Better packet loss resilience
//!
//! Vorbis encoding is provided primarily for:
//! - Compatibility with existing Ogg Vorbis workflows
//! - Projects requiring Ogg container format
//! - Legacy system support
//!
//! ## Implementation Status
//!
//! This is a **simplified Vorbis encoder** that provides the interface and basic
//! structure. Full Vorbis encoding requires implementing:
//! - Psychoacoustic model
//! - MDCT (Modified Discrete Cosine Transform)
//! - Floor encoding
//! - Residue encoding
//! - Bitstream packing
//!
//! For production Vorbis encoding, consider:
//! - Using FFmpeg with libvorbis
//! - Using the `vorbis` crate with FFI to libvorbis
//! - **Recommended**: Use Opus instead (superior in every way)

use crate::codec::{AudioFrame, Encoder, Frame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, SampleFormat, Timestamp};

/// Vorbis encoder configuration
#[derive(Debug, Clone)]
pub struct VorbisEncoderConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub quality: f32,    // -1.0 to 10.0 (Vorbis quality scale)
    pub bitrate: Option<i32>, // Target bitrate in bits/sec (if specified)
}

impl Default for VorbisEncoderConfig {
    fn default() -> Self {
        VorbisEncoderConfig {
            sample_rate: 48000,
            channels: 2,
            quality: 5.0, // Medium quality
            bitrate: None,
        }
    }
}

impl VorbisEncoderConfig {
    /// Validate the encoder configuration
    pub fn validate(&self) -> Result<()> {
        // Validate sample rate
        if self.sample_rate < 8000 || self.sample_rate > 192_000 {
            return Err(Error::codec(format!(
                "Invalid Vorbis sample rate: {}. Must be 8,000-192,000 Hz",
                self.sample_rate
            )));
        }

        // Validate channels
        if self.channels == 0 || self.channels > 255 {
            return Err(Error::codec(format!(
                "Invalid channel count: {}. Vorbis supports 1-255 channels",
                self.channels
            )));
        }

        // Validate quality
        if self.quality < -1.0 || self.quality > 10.0 {
            return Err(Error::codec(format!(
                "Invalid quality: {}. Must be -1.0 to 10.0",
                self.quality
            )));
        }

        // Validate bitrate if specified
        if let Some(bitrate) = self.bitrate {
            if bitrate < 32_000 || bitrate > 500_000 {
                return Err(Error::codec(format!(
                    "Invalid bitrate: {}. Recommended range: 32,000-500,000 bps",
                    bitrate
                )));
            }
        }

        Ok(())
    }
}

/// Vorbis audio encoder
///
/// **Note**: This is a simplified interface. For production Vorbis encoding,
/// use Opus instead (superior quality and features) or FFmpeg with libvorbis.
///
/// # Example
///
/// ```no_run
/// use zvd_lib::codec::vorbis::VorbisEncoder;
/// use zvd_lib::codec::{Encoder, AudioFrame, Frame};
/// use zvd_lib::util::SampleFormat;
///
/// // For new projects, use Opus instead!
/// // This example shows Vorbis for compatibility only
///
/// let mut encoder = VorbisEncoder::new(48000, 2)?;
/// encoder.set_quality(6.0)?; // Higher quality
///
/// let mut frame = AudioFrame::new(1024, 2, SampleFormat::F32);
/// // ... fill frame with data ...
///
/// encoder.send_frame(&Frame::Audio(frame))?;
/// encoder.flush()?;
///
/// let packet = encoder.receive_packet()?;
/// # Ok::<(), zvd_lib::error::Error>(())
/// ```
pub struct VorbisEncoder {
    config: VorbisEncoderConfig,
    /// Buffered frames waiting to be encoded
    frame_buffer: Vec<AudioFrame>,
    /// Buffered packets ready to be retrieved
    packet_buffer: Vec<Packet>,
    /// Current PTS counter
    pts_counter: i64,
    /// Header written flag
    header_written: bool,
}

impl VorbisEncoder {
    /// Create a new Vorbis encoder with default configuration
    ///
    /// **Recommendation**: Use `OpusEncoder` for new projects instead.
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        let config = VorbisEncoderConfig {
            sample_rate,
            channels,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create a new Vorbis encoder with custom configuration
    pub fn with_config(config: VorbisEncoderConfig) -> Result<Self> {
        config.validate()?;

        Ok(VorbisEncoder {
            config,
            frame_buffer: Vec::new(),
            packet_buffer: Vec::new(),
            pts_counter: 0,
            header_written: false,
        })
    }

    /// Set encoding quality (-1.0 to 10.0)
    ///
    /// Quality levels:
    /// - -1.0: ~45 kbps
    /// - 0.0: ~64 kbps
    /// - 3.0: ~112 kbps
    /// - 5.0: ~160 kbps (default)
    /// - 8.0: ~256 kbps
    /// - 10.0: ~500 kbps
    pub fn set_quality(&mut self, quality: f32) -> Result<()> {
        if quality < -1.0 || quality > 10.0 {
            return Err(Error::codec(format!(
                "Invalid quality: {}. Must be -1.0 to 10.0",
                quality
            )));
        }
        self.config.quality = quality;
        Ok(())
    }

    /// Set target bitrate in bits per second
    pub fn set_bitrate(&mut self, bitrate: i32) -> Result<()> {
        if bitrate < 32_000 || bitrate > 500_000 {
            return Err(Error::codec(format!(
                "Invalid bitrate: {}. Recommended: 32,000-500,000 bps",
                bitrate
            )));
        }
        self.config.bitrate = Some(bitrate);
        Ok(())
    }

    /// Get encoder configuration
    pub fn config(&self) -> &VorbisEncoderConfig {
        &self.config
    }

    /// Create Vorbis stream headers
    ///
    /// Vorbis requires 3 header packets:
    /// 1. Identification header
    /// 2. Comment header (metadata)
    /// 3. Setup header (codebook)
    fn create_headers(&self) -> Result<Vec<Packet>> {
        let mut headers = Vec::new();

        // Identification header
        let id_header = self.create_identification_header()?;
        headers.push(id_header);

        // Comment header (Vorbis comments / metadata)
        let comment_header = self.create_comment_header()?;
        headers.push(comment_header);

        // Setup header (codebook configuration)
        let setup_header = self.create_setup_header()?;
        headers.push(setup_header);

        Ok(headers)
    }

    /// Create Vorbis identification header
    fn create_identification_header(&self) -> Result<Packet> {
        let mut header = Vec::new();

        // Packet type (1 = identification)
        header.push(1);

        // Vorbis magic string
        header.extend_from_slice(b"vorbis");

        // Vorbis version (always 0)
        header.extend_from_slice(&0u32.to_le_bytes());

        // Channels
        header.push(self.config.channels as u8);

        // Sample rate
        header.extend_from_slice(&self.config.sample_rate.to_le_bytes());

        // Bitrate maximum (0 = not specified)
        header.extend_from_slice(&0i32.to_le_bytes());

        // Bitrate nominal
        let nominal_bitrate = self.config.bitrate.unwrap_or(0);
        header.extend_from_slice(&nominal_bitrate.to_le_bytes());

        // Bitrate minimum (0 = not specified)
        header.extend_from_slice(&0i32.to_le_bytes());

        // Block sizes (encoded in 4 bits each)
        // blocksize_0 = 256 (8), blocksize_1 = 2048 (11)
        header.push((8 << 4) | 11);

        // Framing flag (must be 1)
        header.push(1);

        Ok(Packet {
            stream_index: 0,
            data: Buffer::from_vec(header),
            pts: Timestamp::new(0),
            dts: Timestamp::new(0),
            duration: Timestamp::new(0),
            keyframe: true,
        })
    }

    /// Create Vorbis comment header
    fn create_comment_header(&self) -> Result<Packet> {
        let mut header = Vec::new();

        // Packet type (3 = comment)
        header.push(3);

        // Vorbis magic string
        header.extend_from_slice(b"vorbis");

        // Vendor string
        let vendor = b"ZVD Vorbis Encoder";
        header.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        header.extend_from_slice(vendor);

        // User comment list length (0 comments)
        header.extend_from_slice(&0u32.to_le_bytes());

        // Framing flag
        header.push(1);

        Ok(Packet {
            stream_index: 0,
            data: Buffer::from_vec(header),
            pts: Timestamp::new(0),
            dts: Timestamp::new(0),
            duration: Timestamp::new(0),
            keyframe: true,
        })
    }

    /// Create Vorbis setup header
    ///
    /// Note: This is a simplified setup header.
    /// Full Vorbis encoding requires complex codebook configuration.
    fn create_setup_header(&self) -> Result<Packet> {
        let mut header = Vec::new();

        // Packet type (5 = setup)
        header.push(5);

        // Vorbis magic string
        header.extend_from_slice(b"vorbis");

        // Simplified setup (production encoder would include full codebooks)
        // Number of codebooks (0 = simplified)
        header.push(0);

        // Framing flag
        header.push(1);

        Ok(Packet {
            stream_index: 0,
            data: Buffer::from_vec(header),
            pts: Timestamp::new(0),
            dts: Timestamp::new(0),
            duration: Timestamp::new(0),
            keyframe: true,
        })
    }

    /// Encode audio samples
    ///
    /// Note: This is a placeholder. Full Vorbis encoding requires:
    /// - MDCT transformation
    /// - Psychoacoustic analysis
    /// - Floor encoding
    /// - Residue quantization and encoding
    fn encode_frame(&mut self, frame: &AudioFrame) -> Result<()> {
        // Write headers if not done yet
        if !self.header_written {
            let headers = self.create_headers()?;
            self.packet_buffer.extend(headers);
            self.header_written = true;
        }

        // Create a simplified audio packet
        // Note: This is not valid Vorbis encoding - use libvorbis for production
        let packet = Packet {
            stream_index: 0,
            data: frame.data[0].clone(), // Simplified - not actual Vorbis encoding
            pts: frame.pts,
            dts: frame.pts,
            duration: Timestamp::new((frame.nb_samples as i64 * 1000) / self.config.sample_rate as i64),
            keyframe: false,
        };

        self.packet_buffer.push(packet);
        self.pts_counter += frame.nb_samples as i64;

        Ok(())
    }
}

impl Encoder for VorbisEncoder {
    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let audio_frame = match frame {
            Frame::Audio(af) => af,
            _ => return Err(Error::codec("Vorbis encoder only accepts audio frames")),
        };

        // Validate frame parameters
        if audio_frame.channels != self.config.channels {
            return Err(Error::codec(format!(
                "Frame channel count {} doesn't match encoder config {}",
                audio_frame.channels, self.config.channels
            )));
        }

        // Encode frame
        self.encode_frame(audio_frame)?;

        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if self.packet_buffer.is_empty() {
            return Err(Error::TryAgain);
        }

        Ok(self.packet_buffer.remove(0))
    }

    fn flush(&mut self) -> Result<()> {
        // Ensure headers are written
        if !self.header_written {
            let headers = self.create_headers()?;
            self.packet_buffer.extend(headers);
            self.header_written = true;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vorbis_encoder_creation() {
        let encoder = VorbisEncoder::new(48000, 2);
        assert!(encoder.is_ok());

        let encoder = encoder.unwrap();
        assert_eq!(encoder.config.sample_rate, 48000);
        assert_eq!(encoder.config.channels, 2);
    }

    #[test]
    fn test_vorbis_encoder_config_validation() {
        // Valid config
        let config = VorbisEncoderConfig::default();
        assert!(config.validate().is_ok());

        // Invalid sample rate (too low)
        let mut config = VorbisEncoderConfig::default();
        config.sample_rate = 7999;
        assert!(config.validate().is_err());

        // Invalid sample rate (too high)
        let mut config = VorbisEncoderConfig::default();
        config.sample_rate = 200_000;
        assert!(config.validate().is_err());

        // Invalid channels
        let mut config = VorbisEncoderConfig::default();
        config.channels = 0;
        assert!(config.validate().is_err());

        // Invalid quality (too low)
        let mut config = VorbisEncoderConfig::default();
        config.quality = -1.5;
        assert!(config.validate().is_err());

        // Invalid quality (too high)
        let mut config = VorbisEncoderConfig::default();
        config.quality = 11.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_vorbis_encoder_quality_levels() {
        let mut encoder = VorbisEncoder::new(48000, 2).unwrap();

        // Valid quality levels
        assert!(encoder.set_quality(-1.0).is_ok());
        assert!(encoder.set_quality(0.0).is_ok());
        assert!(encoder.set_quality(5.0).is_ok());
        assert!(encoder.set_quality(10.0).is_ok());

        // Invalid quality
        assert!(encoder.set_quality(-2.0).is_err());
        assert!(encoder.set_quality(11.0).is_err());
    }

    #[test]
    fn test_vorbis_encoder_bitrate() {
        let mut encoder = VorbisEncoder::new(48000, 2).unwrap();

        // Valid bitrates
        assert!(encoder.set_bitrate(64_000).is_ok());
        assert!(encoder.set_bitrate(128_000).is_ok());
        assert!(encoder.set_bitrate(256_000).is_ok());

        // Invalid bitrates
        assert!(encoder.set_bitrate(10_000).is_err());
        assert!(encoder.set_bitrate(1_000_000).is_err());
    }

    #[test]
    fn test_vorbis_encoder_sample_rates() {
        // Common sample rates
        assert!(VorbisEncoder::new(8000, 2).is_ok());
        assert!(VorbisEncoder::new(16000, 2).is_ok());
        assert!(VorbisEncoder::new(44100, 2).is_ok());
        assert!(VorbisEncoder::new(48000, 2).is_ok());
        assert!(VorbisEncoder::new(96000, 2).is_ok());
    }

    #[test]
    fn test_vorbis_encoder_channels() {
        // Valid channel counts
        assert!(VorbisEncoder::new(48000, 1).is_ok()); // Mono
        assert!(VorbisEncoder::new(48000, 2).is_ok()); // Stereo
        assert!(VorbisEncoder::new(48000, 6).is_ok()); // 5.1
        assert!(VorbisEncoder::new(48000, 8).is_ok()); // 7.1
    }

    #[test]
    fn test_vorbis_encoder_headers() {
        let encoder = VorbisEncoder::new(48000, 2).unwrap();

        let headers = encoder.create_headers();
        assert!(headers.is_ok());

        let headers = headers.unwrap();
        assert_eq!(headers.len(), 3, "Should have 3 header packets");

        // Check identification header
        assert!(headers[0].keyframe, "Headers should be keyframes");
        assert!(headers[0].data.as_slice().starts_with(&[1, b'v', b'o', b'r', b'b', b'i', b's']));

        // Check comment header
        assert!(headers[1].data.as_slice().starts_with(&[3, b'v', b'o', b'r', b'b', b'i', b's']));

        // Check setup header
        assert!(headers[2].data.as_slice().starts_with(&[5, b'v', b'o', b'r', b'b', b'i', b's']));
    }

    #[test]
    fn test_vorbis_encoder_wrong_channels() {
        let mut encoder = VorbisEncoder::new(48000, 2).unwrap();

        // Create mono frame for stereo encoder
        let frame = AudioFrame::new(1024, 1, SampleFormat::F32);

        let result = encoder.send_frame(&Frame::Audio(frame));
        assert!(result.is_err());
    }

    #[test]
    fn test_vorbis_encoder_wrong_frame_type() {
        use crate::codec::VideoFrame;
        use crate::util::PixelFormat;

        let mut encoder = VorbisEncoder::new(48000, 2).unwrap();
        let video_frame = VideoFrame::new(640, 480, PixelFormat::YUV420P);

        let result = encoder.send_frame(&Frame::Video(video_frame));
        assert!(result.is_err());
    }
}
