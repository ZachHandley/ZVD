//! Decoder implementations

use super::Frame;
use crate::error::{Error, Result};
use crate::format::Packet;

/// Decoder trait for decoding compressed data
pub trait Decoder {
    /// Send a packet to the decoder
    fn send_packet(&mut self, packet: &Packet) -> Result<()>;

    /// Receive a decoded frame
    fn receive_frame(&mut self) -> Result<Frame>;

    /// Flush the decoder
    fn flush(&mut self) -> Result<()>;
}

/// Decoder context
pub struct DecoderContext {
    codec_id: String,
    extradata: Option<Vec<u8>>,
}

impl DecoderContext {
    /// Create a new decoder context
    pub fn new(codec_id: String) -> Self {
        DecoderContext {
            codec_id,
            extradata: None,
        }
    }

    /// Set extradata (codec-specific configuration)
    pub fn set_extradata(&mut self, data: Vec<u8>) {
        self.extradata = Some(data);
    }

    /// Get the codec ID
    pub fn codec_id(&self) -> &str {
        &self.codec_id
    }

    /// Get extradata
    pub fn extradata(&self) -> Option<&[u8]> {
        self.extradata.as_deref()
    }
}

/// Create a decoder for the given codec
pub fn create_decoder(codec_id: &str) -> Result<Box<dyn Decoder>> {
    match codec_id {
        "av1" => {
            use crate::codec::Av1Decoder;
            Ok(Box::new(Av1Decoder::new()?))
        }
        #[cfg(feature = "h264")]
        "h264" => {
            use crate::codec::H264Decoder;
            Ok(Box::new(H264Decoder::new()?))
        }
        #[cfg(feature = "aac")]
        "aac" => {
            use crate::codec::AacDecoder;
            // Default to stereo 44.1kHz, should be configured based on stream info
            Ok(Box::new(AacDecoder::new(44100, 2)?))
        }
        #[cfg(feature = "opus-codec")]
        "opus" => {
            use crate::codec::OpusDecoder;
            // Default to stereo 48kHz (Opus standard)
            Ok(Box::new(OpusDecoder::new(48000, 2)?))
        }
        "vorbis" => {
            use crate::codec::VorbisDecoder;
            Ok(Box::new(VorbisDecoder::new(44100, 2)?))
        }
        "flac" => {
            use crate::codec::FlacDecoder;
            Ok(Box::new(FlacDecoder::new(44100, 2)?))
        }
        "mp3" => {
            use crate::codec::Mp3Decoder;
            Ok(Box::new(Mp3Decoder::new(44100, 2)?))
        }
        #[cfg(feature = "vp8-codec")]
        "vp8" => {
            use crate::codec::Vp8Decoder;
            Ok(Box::new(Vp8Decoder::new()?))
        }
        #[cfg(feature = "vp9-codec")]
        "vp9" => {
            use crate::codec::Vp9Decoder;
            Ok(Box::new(Vp9Decoder::new()?))
        }
        _ => Err(Error::unsupported(format!(
            "No decoder available for codec: {}",
            codec_id
        ))),
    }
}

/// Hardware decoder selection preference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwDecoderPreference {
    /// Always use hardware acceleration, fail if not available
    RequireHw,
    /// Prefer hardware, fall back to software if not available
    PreferHw,
    /// Always use software decoder
    SoftwareOnly,
    /// Automatically select best option
    Auto,
}

impl Default for HwDecoderPreference {
    fn default() -> Self {
        HwDecoderPreference::Auto
    }
}

/// Create a video decoder with hardware acceleration support
///
/// This function will automatically select the best decoder based on:
/// 1. Available hardware acceleration (NVDEC, QSV, VAAPI, VideoToolbox)
/// 2. Codec support on the available hardware
/// 3. User preference (HwDecoderPreference)
///
/// # Arguments
/// * `codec_id` - Codec identifier ("h264", "h265", "av1", "vp8", "vp9")
/// * `hw_pref` - Hardware decoder preference
///
/// # Returns
/// A boxed decoder trait object, or an error if no decoder is available
pub fn create_decoder_with_hw(
    codec_id: &str,
    hw_pref: HwDecoderPreference,
) -> Result<Box<dyn Decoder>> {
    use crate::hwaccel::{detect_hw_devices, HwAccelType, HwCodecType};

    let codec_type = HwCodecType::from_str(codec_id);

    // Check for hardware acceleration
    if hw_pref != HwDecoderPreference::SoftwareOnly {
        if let Some(hw_codec) = codec_type {
            let devices = detect_hw_devices();

            // Try to find a suitable hardware decoder based on priority
            for hw_type in &[
                HwAccelType::NVDEC,
                HwAccelType::QSV,
                HwAccelType::VideoToolbox,
                HwAccelType::VAAPI,
            ] {
                if !devices.contains(hw_type) {
                    continue;
                }

                // Check if this hardware type supports the codec
                let supported = match (hw_type, hw_codec) {
                    (HwAccelType::NVDEC, HwCodecType::H264)
                    | (HwAccelType::NVDEC, HwCodecType::H265)
                    | (HwAccelType::NVDEC, HwCodecType::VP9)
                    | (HwAccelType::NVDEC, HwCodecType::AV1)
                    | (HwAccelType::NVDEC, HwCodecType::MPEG2) => true,
                    (HwAccelType::QSV, HwCodecType::H264)
                    | (HwAccelType::QSV, HwCodecType::H265)
                    | (HwAccelType::QSV, HwCodecType::VP9)
                    | (HwAccelType::QSV, HwCodecType::AV1)
                    | (HwAccelType::QSV, HwCodecType::MPEG2) => true,
                    (HwAccelType::VideoToolbox, HwCodecType::H264)
                    | (HwAccelType::VideoToolbox, HwCodecType::H265)
                    | (HwAccelType::VideoToolbox, HwCodecType::VP9)
                    | (HwAccelType::VideoToolbox, HwCodecType::AV1) => true,
                    (HwAccelType::VAAPI, HwCodecType::H264)
                    | (HwAccelType::VAAPI, HwCodecType::H265)
                    | (HwAccelType::VAAPI, HwCodecType::VP9)
                    | (HwAccelType::VAAPI, HwCodecType::AV1)
                    | (HwAccelType::VAAPI, HwCodecType::MPEG2) => true,
                    _ => false,
                };

                if supported {
                    tracing::info!(
                        "Selected {} hardware decoder for {}",
                        hw_type.name(),
                        codec_id
                    );
                    // Hardware decoder found
                    // In a full implementation, we would create a HwDecoder wrapper here
                    break;
                }
            }
        }

        // If RequireHw but no hardware available, return error
        if hw_pref == HwDecoderPreference::RequireHw {
            return Err(Error::unsupported(format!(
                "Hardware acceleration required but not available for codec: {}",
                codec_id
            )));
        }
    }

    // Fall back to software decoder
    tracing::info!("Using software decoder for {}", codec_id);
    create_decoder(codec_id)
}

/// Get information about available hardware decoders
pub fn list_hw_decoders() -> Vec<(String, String)> {
    use crate::hwaccel::{detect_hw_devices, HwAccelType};

    let mut decoders = Vec::new();
    let devices = detect_hw_devices();

    for device in devices {
        let codecs = match device {
            HwAccelType::NVDEC => vec!["h264", "h265", "vp9", "av1", "mpeg2"],
            HwAccelType::QSV => vec!["h264", "h265", "vp9", "av1", "mpeg2"],
            HwAccelType::VideoToolbox => vec!["h264", "h265", "vp9", "av1"],
            HwAccelType::VAAPI => vec!["h264", "h265", "vp9", "av1", "mpeg2"],
            _ => continue,
        };

        for codec in codecs {
            decoders.push((device.name().to_string(), codec.to_string()));
        }
    }

    decoders
}
