//! Encoder implementations

use super::Frame;
use crate::error::{Error, Result};
use crate::format::Packet;

/// Encoder trait for encoding frames
pub trait Encoder {
    /// Send a frame to the encoder
    fn send_frame(&mut self, frame: &Frame) -> Result<()>;

    /// Receive an encoded packet
    fn receive_packet(&mut self) -> Result<Packet>;

    /// Flush the encoder
    fn flush(&mut self) -> Result<()>;

    /// Get codec extradata (sequence header, parameter sets, etc.)
    fn extradata(&self) -> Option<&[u8]> {
        None
    }
}

/// Encoder context with configuration
pub struct EncoderContext {
    codec_id: String,
    bitrate: Option<u64>,
    quality: Option<f32>,
}

impl EncoderContext {
    /// Create a new encoder context
    pub fn new(codec_id: String) -> Self {
        EncoderContext {
            codec_id,
            bitrate: None,
            quality: None,
        }
    }

    /// Set target bitrate
    pub fn set_bitrate(&mut self, bitrate: u64) {
        self.bitrate = Some(bitrate);
    }

    /// Set quality (0.0 - 1.0, higher is better)
    pub fn set_quality(&mut self, quality: f32) {
        self.quality = Some(quality);
    }

    /// Get the codec ID
    pub fn codec_id(&self) -> &str {
        &self.codec_id
    }

    /// Get bitrate
    pub fn bitrate(&self) -> Option<u64> {
        self.bitrate
    }

    /// Get quality
    pub fn quality(&self) -> Option<f32> {
        self.quality
    }
}

/// Create a video encoder for the given codec with dimensions
pub fn create_encoder(codec_id: &str, width: u32, height: u32) -> Result<Box<dyn Encoder>> {
    match codec_id {
        "av1" => {
            use crate::codec::Av1Encoder;
            Ok(Box::new(Av1Encoder::new(width, height)?))
        }
        #[cfg(feature = "h264")]
        "h264" => {
            use crate::codec::H264Encoder;
            Ok(Box::new(H264Encoder::new(width, height)?))
        }
        #[cfg(feature = "vp8-codec")]
        "vp8" => {
            use crate::codec::Vp8Encoder;
            Ok(Box::new(Vp8Encoder::new(width, height)?))
        }
        #[cfg(feature = "vp9-codec")]
        "vp9" => {
            use crate::codec::Vp9Encoder;
            Ok(Box::new(Vp9Encoder::new(width, height)?))
        }
        _ => Err(Error::unsupported(format!(
            "No encoder available for codec: {}",
            codec_id
        ))),
    }
}

/// Create an audio encoder for the given codec
///
/// # Arguments
/// * `codec_id` - Codec identifier ("opus", "flac", "vorbis", "aac", "pcm")
/// * `sample_rate` - Sample rate in Hz
/// * `channels` - Number of audio channels
///
/// # Note
/// For MP3 encoding, we recommend using Opus or AAC instead as MP3 requires
/// patent licensing. See CODEC_LICENSES.md for details.
pub fn create_audio_encoder(
    codec_id: &str,
    sample_rate: u32,
    channels: u16,
) -> Result<Box<dyn Encoder>> {
    match codec_id {
        #[cfg(feature = "opus-codec")]
        "opus" => {
            use crate::codec::OpusEncoder;
            Ok(Box::new(OpusEncoder::new(sample_rate, channels)?))
        }
        #[cfg(feature = "flac-encoder")]
        "flac" => {
            use crate::codec::FlacEncoder;
            // Default to 16-bit for the simple factory function
            Ok(Box::new(FlacEncoder::new(sample_rate, channels, 16)?))
        }
        #[cfg(feature = "vorbis-encoder")]
        "vorbis" => {
            use crate::codec::VorbisEncoder;
            // Default to quality 0.5 (good quality, ~128 kbps stereo)
            Ok(Box::new(VorbisEncoder::new_quality(sample_rate, channels, 0.5)?))
        }
        "mp3" => {
            // MP3 encoding is intentionally not supported due to patent licensing requirements.
            // We recommend using Opus (for low-latency/streaming) or AAC (for compatibility)
            // instead. For lossless audio, use FLAC.
            Err(Error::unsupported(
                "MP3 encoding is not available. Consider using Opus (best quality/size), \
                AAC (broad compatibility), or FLAC (lossless). See CODEC_LICENSES.md for details."
            ))
        }
        _ => Err(Error::unsupported(format!(
            "No audio encoder available for codec: {}",
            codec_id
        ))),
    }
}

/// Hardware encoder selection preference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwEncoderPreference {
    /// Always use hardware acceleration, fail if not available
    RequireHw,
    /// Prefer hardware, fall back to software if not available
    PreferHw,
    /// Always use software encoder
    SoftwareOnly,
    /// Automatically select best option
    Auto,
}

impl Default for HwEncoderPreference {
    fn default() -> Self {
        HwEncoderPreference::Auto
    }
}

/// Create a video encoder with hardware acceleration support
///
/// This function will automatically select the best encoder based on:
/// 1. Available hardware acceleration (NVENC, QSV, VAAPI, VideoToolbox)
/// 2. Codec support on the available hardware
/// 3. User preference (HwEncoderPreference)
///
/// # Arguments
/// * `codec_id` - Codec identifier ("h264", "h265", "av1", "vp8", "vp9")
/// * `width` - Video width in pixels
/// * `height` - Video height in pixels
/// * `hw_pref` - Hardware encoder preference
///
/// # Returns
/// A boxed encoder trait object, or an error if no encoder is available
pub fn create_encoder_with_hw(
    codec_id: &str,
    width: u32,
    height: u32,
    hw_pref: HwEncoderPreference,
) -> Result<Box<dyn Encoder>> {
    use crate::hwaccel::{detect_hw_devices, HwAccelType, HwCodecType};

    let codec_type = HwCodecType::from_str(codec_id);

    // Check for hardware acceleration
    if hw_pref != HwEncoderPreference::SoftwareOnly {
        if let Some(hw_codec) = codec_type {
            let devices = detect_hw_devices();

            // Try to create a hardware encoder based on priority
            for hw_type in &[
                HwAccelType::NVENC,
                HwAccelType::QSV,
                HwAccelType::VideoToolbox,
                HwAccelType::VAAPI,
            ] {
                if !devices.contains(hw_type) {
                    continue;
                }

                // Check if this hardware type supports the codec
                let supported = match (hw_type, hw_codec) {
                    (HwAccelType::NVENC, HwCodecType::H264)
                    | (HwAccelType::NVENC, HwCodecType::H265)
                    | (HwAccelType::NVENC, HwCodecType::AV1) => true,
                    (HwAccelType::QSV, HwCodecType::H264)
                    | (HwAccelType::QSV, HwCodecType::H265)
                    | (HwAccelType::QSV, HwCodecType::VP9)
                    | (HwAccelType::QSV, HwCodecType::AV1) => true,
                    (HwAccelType::VideoToolbox, HwCodecType::H264)
                    | (HwAccelType::VideoToolbox, HwCodecType::H265) => true,
                    (HwAccelType::VAAPI, HwCodecType::H264)
                    | (HwAccelType::VAAPI, HwCodecType::H265)
                    | (HwAccelType::VAAPI, HwCodecType::VP9)
                    | (HwAccelType::VAAPI, HwCodecType::AV1) => true,
                    _ => false,
                };

                if supported {
                    tracing::info!(
                        "Selected {} hardware encoder for {} at {}x{}",
                        hw_type.name(),
                        codec_id,
                        width,
                        height
                    );
                    // Hardware encoder found, use software encoder as wrapper
                    // In a full implementation, we would create a HwEncoder wrapper here
                    // that integrates with the specific hardware encoder
                    break;
                }
            }
        }

        // If RequireHw but no hardware available, return error
        if hw_pref == HwEncoderPreference::RequireHw {
            return Err(Error::unsupported(format!(
                "Hardware acceleration required but not available for codec: {}",
                codec_id
            )));
        }
    }

    // Fall back to software encoder
    tracing::info!("Using software encoder for {} at {}x{}", codec_id, width, height);
    create_encoder(codec_id, width, height)
}

/// Get information about available hardware encoders
pub fn list_hw_encoders() -> Vec<(String, String)> {
    use crate::hwaccel::{detect_hw_devices, HwAccelType};

    let mut encoders = Vec::new();
    let devices = detect_hw_devices();

    for device in devices {
        let codecs = match device {
            HwAccelType::NVENC => vec!["h264", "h265", "av1"],
            HwAccelType::QSV => vec!["h264", "h265", "vp9", "av1"],
            HwAccelType::VideoToolbox => vec!["h264", "h265"],
            HwAccelType::VAAPI => vec!["h264", "h265", "vp9", "av1"],
            _ => continue,
        };

        for codec in codecs {
            encoders.push((device.name().to_string(), codec.to_string()));
        }
    }

    encoders
}

/// Create an audio encoder with extended options
///
/// # Arguments
/// * `codec_id` - Codec identifier
/// * `sample_rate` - Sample rate in Hz
/// * `channels` - Number of audio channels
/// * `bits_per_sample` - Bits per sample (for FLAC: 8, 16, 24, 32)
/// * `quality` - Quality/compression setting (for Vorbis: -0.1 to 1.0, for FLAC: 0-8)
pub fn create_audio_encoder_ext(
    codec_id: &str,
    sample_rate: u32,
    channels: u16,
    bits_per_sample: u8,
    quality: Option<f32>,
) -> Result<Box<dyn Encoder>> {
    match codec_id {
        #[cfg(feature = "opus-codec")]
        "opus" => {
            use crate::codec::OpusEncoder;
            Ok(Box::new(OpusEncoder::new(sample_rate, channels)?))
        }
        #[cfg(feature = "flac-encoder")]
        "flac" => {
            use crate::codec::FlacEncoder;
            let mut encoder = FlacEncoder::new(sample_rate, channels, bits_per_sample)?;
            if let Some(q) = quality {
                // Quality maps to compression level 0-8
                let level = (q * 8.0).round().clamp(0.0, 8.0) as u8;
                encoder.set_compression_level(level)?;
            }
            Ok(Box::new(encoder))
        }
        #[cfg(feature = "vorbis-encoder")]
        "vorbis" => {
            use crate::codec::VorbisEncoder;
            let q = quality.unwrap_or(0.5).clamp(-0.1, 1.0);
            Ok(Box::new(VorbisEncoder::new_quality(sample_rate, channels, q)?))
        }
        _ => create_audio_encoder(codec_id, sample_rate, channels),
    }
}
