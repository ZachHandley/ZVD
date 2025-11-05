//! Container format handling (demuxing and muxing)
//!
//! This module provides functionality for reading and writing various
//! multimedia container formats.

pub mod demuxer;
pub mod muxer;
pub mod packet;
pub mod stream;
pub mod wav;

pub use demuxer::{Demuxer, DemuxerContext};
pub use muxer::{Muxer, MuxerContext};
pub use packet::Packet;
pub use stream::{AudioInfo, Stream, StreamInfo, VideoInfo};

use crate::error::{Error, Result};
use crate::util::MediaType;

/// Format capability flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatCapabilities {
    /// Format supports seeking
    pub seekable: bool,
    /// Format supports multiple streams
    pub multi_stream: bool,
    /// Format supports timestamps
    pub timestamps: bool,
    /// Format supports metadata
    pub metadata: bool,
}

/// Container format information
#[derive(Debug, Clone)]
pub struct FormatInfo {
    /// Format name (e.g., "mp4", "mkv", "avi")
    pub name: String,
    /// Long/descriptive name
    pub long_name: String,
    /// File extensions (e.g., ["mp4", "m4v"])
    pub extensions: Vec<String>,
    /// MIME types
    pub mime_types: Vec<String>,
    /// Format capabilities
    pub capabilities: FormatCapabilities,
}

/// Detect format from file extension
pub fn detect_format_from_extension(path: &str) -> Option<&'static str> {
    let ext = path.rsplit('.').next()?.to_lowercase();
    match ext.as_str() {
        "mp4" | "m4v" | "m4a" => Some("mp4"),
        "mkv" | "mka" | "mks" => Some("matroska"),
        "webm" => Some("webm"),
        "avi" => Some("avi"),
        "mov" => Some("mov"),
        "flv" => Some("flv"),
        "ts" | "m2ts" => Some("mpegts"),
        "mp3" => Some("mp3"),
        "ogg" | "oga" => Some("ogg"),
        "wav" => Some("wav"),
        "flac" => Some("flac"),
        _ => None,
    }
}

/// Get format information by name
pub fn get_format_info(name: &str) -> Option<FormatInfo> {
    match name {
        "mp4" => Some(FormatInfo {
            name: "mp4".to_string(),
            long_name: "MPEG-4 Part 14".to_string(),
            extensions: vec!["mp4".to_string(), "m4v".to_string(), "m4a".to_string()],
            mime_types: vec!["video/mp4".to_string()],
            capabilities: FormatCapabilities {
                seekable: true,
                multi_stream: true,
                timestamps: true,
                metadata: true,
            },
        }),
        "matroska" => Some(FormatInfo {
            name: "matroska".to_string(),
            long_name: "Matroska".to_string(),
            extensions: vec!["mkv".to_string(), "mka".to_string(), "mks".to_string()],
            mime_types: vec!["video/x-matroska".to_string()],
            capabilities: FormatCapabilities {
                seekable: true,
                multi_stream: true,
                timestamps: true,
                metadata: true,
            },
        }),
        "wav" => Some(FormatInfo {
            name: "wav".to_string(),
            long_name: "WAV / WAVE (Waveform Audio)".to_string(),
            extensions: vec!["wav".to_string()],
            mime_types: vec!["audio/wav".to_string(), "audio/wave".to_string()],
            capabilities: FormatCapabilities {
                seekable: true,
                multi_stream: false,
                timestamps: true,
                metadata: true,
            },
        }),
        _ => None,
    }
}
