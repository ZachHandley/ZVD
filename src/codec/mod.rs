//! Codec implementations (encoders and decoders)

pub mod av1;
pub mod decoder;
pub mod encoder;
pub mod frame;
pub mod pcm;

pub use av1::Av1Decoder;
pub use decoder::{Decoder, DecoderContext};
pub use encoder::{Encoder, EncoderContext};
pub use frame::{AudioFrame, Frame, PictureType, VideoFrame};
pub use pcm::{PcmConfig, PcmDecoder, PcmEncoder};

use crate::error::Result;
use crate::util::MediaType;

/// Codec capability flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodecCapabilities {
    /// Codec supports lossy compression
    pub lossy: bool,
    /// Codec supports lossless compression
    pub lossless: bool,
    /// Codec supports intra-only coding
    pub intra_only: bool,
    /// Codec supports inter-frame prediction
    pub inter: bool,
}

/// Codec information
#[derive(Debug, Clone)]
pub struct CodecInfo {
    /// Codec identifier
    pub id: String,
    /// Codec name
    pub name: String,
    /// Long descriptive name
    pub long_name: String,
    /// Media type
    pub media_type: MediaType,
    /// Codec capabilities
    pub capabilities: CodecCapabilities,
}

/// Get codec information by ID
pub fn get_codec_info(id: &str) -> Option<CodecInfo> {
    match id {
        "h264" => Some(CodecInfo {
            id: "h264".to_string(),
            name: "H.264".to_string(),
            long_name: "H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10".to_string(),
            media_type: MediaType::Video,
            capabilities: CodecCapabilities {
                lossy: true,
                lossless: false,
                intra_only: false,
                inter: true,
            },
        }),
        "h265" | "hevc" => Some(CodecInfo {
            id: "hevc".to_string(),
            name: "H.265".to_string(),
            long_name: "H.265 / HEVC / High Efficiency Video Coding".to_string(),
            media_type: MediaType::Video,
            capabilities: CodecCapabilities {
                lossy: true,
                lossless: false,
                intra_only: false,
                inter: true,
            },
        }),
        "vp8" => Some(CodecInfo {
            id: "vp8".to_string(),
            name: "VP8".to_string(),
            long_name: "On2 VP8".to_string(),
            media_type: MediaType::Video,
            capabilities: CodecCapabilities {
                lossy: true,
                lossless: false,
                intra_only: false,
                inter: true,
            },
        }),
        "vp9" => Some(CodecInfo {
            id: "vp9".to_string(),
            name: "VP9".to_string(),
            long_name: "Google VP9".to_string(),
            media_type: MediaType::Video,
            capabilities: CodecCapabilities {
                lossy: true,
                lossless: false,
                intra_only: false,
                inter: true,
            },
        }),
        "av1" => Some(CodecInfo {
            id: "av1".to_string(),
            name: "AV1".to_string(),
            long_name: "AOMedia Video 1".to_string(),
            media_type: MediaType::Video,
            capabilities: CodecCapabilities {
                lossy: true,
                lossless: true,
                intra_only: false,
                inter: true,
            },
        }),
        "aac" => Some(CodecInfo {
            id: "aac".to_string(),
            name: "AAC".to_string(),
            long_name: "Advanced Audio Coding".to_string(),
            media_type: MediaType::Audio,
            capabilities: CodecCapabilities {
                lossy: true,
                lossless: false,
                intra_only: true,
                inter: false,
            },
        }),
        "mp3" => Some(CodecInfo {
            id: "mp3".to_string(),
            name: "MP3".to_string(),
            long_name: "MPEG Audio Layer 3".to_string(),
            media_type: MediaType::Audio,
            capabilities: CodecCapabilities {
                lossy: true,
                lossless: false,
                intra_only: true,
                inter: false,
            },
        }),
        "opus" => Some(CodecInfo {
            id: "opus".to_string(),
            name: "Opus".to_string(),
            long_name: "Opus Interactive Audio Codec".to_string(),
            media_type: MediaType::Audio,
            capabilities: CodecCapabilities {
                lossy: true,
                lossless: false,
                intra_only: true,
                inter: false,
            },
        }),
        "pcm" | "pcm_u8" | "pcm_s16le" | "pcm_s32le" | "pcm_f32le" | "pcm_f64le" => {
            Some(CodecInfo {
                id: "pcm".to_string(),
                name: "PCM".to_string(),
                long_name: "Pulse Code Modulation (uncompressed)".to_string(),
                media_type: MediaType::Audio,
                capabilities: CodecCapabilities {
                    lossy: false,
                    lossless: true,
                    intra_only: true,
                    inter: false,
                },
            })
        }
        _ => None,
    }
}
