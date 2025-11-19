//! Media Metadata Structures
//!
//! This module defines data structures for storing media file metadata,
//! similar to FFprobe's output format.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Complete media file metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaMetadata {
    /// File path
    pub file_path: String,
    /// File size in bytes
    pub file_size: u64,
    /// Container format information
    pub format: ContainerFormat,
    /// List of streams (video, audio, subtitle)
    pub streams: Vec<StreamInfo>,
    /// Total duration in seconds
    pub duration: f64,
    /// Overall bitrate in bits/second
    pub bitrate: Option<u64>,
}

impl MediaMetadata {
    /// Convert to JSON string
    pub fn to_json(&self) -> crate::error::Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| crate::error::Error::InvalidData(format!("JSON serialization failed: {}", e)))
    }

    /// Convert to compact JSON string
    pub fn to_json_compact(&self) -> crate::error::Result<String> {
        serde_json::to_string(self)
            .map_err(|e| crate::error::Error::InvalidData(format!("JSON serialization failed: {}", e)))
    }
}

impl fmt::Display for MediaMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Input #0, {} from '{}':", self.format.name, self.file_path)?;
        writeln!(f, "  File Size: {} bytes ({:.2} MB)", self.file_size, self.file_size as f64 / 1_048_576.0)?;

        if self.duration > 0.0 {
            let hours = (self.duration / 3600.0) as u32;
            let minutes = ((self.duration % 3600.0) / 60.0) as u32;
            let seconds = self.duration % 60.0;
            writeln!(f, "  Duration: {:02}:{:02}:{:05.2}", hours, minutes, seconds)?;
        }

        if let Some(bitrate) = self.bitrate {
            writeln!(f, "  Bitrate: {:.2} kbps", bitrate as f64 / 1000.0)?;
        }

        writeln!(f, "")?;

        // Print stream information
        for stream in &self.streams {
            writeln!(f, "  Stream #{}: {}", stream.index, stream)?;
        }

        Ok(())
    }
}

/// Container format information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerFormat {
    /// Format name (e.g., "mp4", "webm", "matroska")
    pub name: String,
    /// Long format name (e.g., "MP4 (MPEG-4 Part 14)")
    pub long_name: String,
    /// MIME type if applicable
    pub mime_type: Option<String>,
    /// File extensions
    pub extensions: Vec<String>,
}

/// Stream type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamType {
    Video,
    Audio,
    Subtitle,
    Data,
    Attachment,
    Unknown,
}

impl fmt::Display for StreamType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StreamType::Video => write!(f, "Video"),
            StreamType::Audio => write!(f, "Audio"),
            StreamType::Subtitle => write!(f, "Subtitle"),
            StreamType::Data => write!(f, "Data"),
            StreamType::Attachment => write!(f, "Attachment"),
            StreamType::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Stream information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamInfo {
    /// Stream index
    pub index: u32,
    /// Stream type
    pub stream_type: StreamType,
    /// Codec information
    pub codec: CodecInfo,
    /// Duration in seconds
    pub duration: Option<f64>,
    /// Bitrate in bits/second
    pub bitrate: Option<u64>,
    /// Video-specific information
    pub video_info: Option<VideoInfo>,
    /// Audio-specific information
    pub audio_info: Option<AudioInfo>,
}

impl fmt::Display for StreamInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} ({})", self.stream_type, self.codec.long_name, self.codec.name)?;

        if let Some(ref video) = self.video_info {
            write!(f, ", {}x{}", video.width, video.height)?;
            if let Some(fps) = video.frame_rate {
                write!(f, ", {:.2} fps", fps)?;
            }
            if let Some(ref pix_fmt) = video.pixel_format {
                write!(f, ", {}", pix_fmt)?;
            }
        }

        if let Some(ref audio) = self.audio_info {
            write!(f, ", {} Hz", audio.sample_rate)?;
            write!(f, ", {} channels", audio.channels)?;
            if let Some(ref fmt) = audio.sample_format {
                write!(f, ", {}", fmt)?;
            }
        }

        if let Some(bitrate) = self.bitrate {
            write!(f, ", {:.2} kbps", bitrate as f64 / 1000.0)?;
        }

        Ok(())
    }
}

/// Codec information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecInfo {
    /// Short codec name (e.g., "h264", "opus")
    pub name: String,
    /// Long codec name (e.g., "H.264 / AVC / MPEG-4 AVC")
    pub long_name: String,
    /// Codec type
    pub codec_type: StreamType,
}

/// Video stream information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoInfo {
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Pixel format (e.g., "yuv420p", "rgb24")
    pub pixel_format: Option<String>,
    /// Frame rate (fps)
    pub frame_rate: Option<f64>,
    /// Aspect ratio (e.g., "16:9")
    pub aspect_ratio: Option<String>,
    /// Bit depth
    pub bit_depth: Option<u8>,
    /// Color space
    pub color_space: Option<String>,
}

/// Audio stream information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioInfo {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u8,
    /// Channel layout (e.g., "stereo", "5.1")
    pub channel_layout: Option<String>,
    /// Sample format (e.g., "s16", "fltp")
    pub sample_format: Option<String>,
    /// Bits per sample
    pub bits_per_sample: Option<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_type_display() {
        assert_eq!(StreamType::Video.to_string(), "Video");
        assert_eq!(StreamType::Audio.to_string(), "Audio");
    }

    #[test]
    fn test_codec_info_creation() {
        let codec = CodecInfo {
            name: "h264".to_string(),
            long_name: "H.264 / AVC".to_string(),
            codec_type: StreamType::Video,
        };

        assert_eq!(codec.name, "h264");
    }

    #[test]
    fn test_video_info_creation() {
        let video = VideoInfo {
            width: 1920,
            height: 1080,
            pixel_format: Some("yuv420p".to_string()),
            frame_rate: Some(30.0),
            aspect_ratio: Some("16:9".to_string()),
            bit_depth: Some(8),
            color_space: Some("bt709".to_string()),
        };

        assert_eq!(video.width, 1920);
        assert_eq!(video.height, 1080);
    }

    #[test]
    fn test_audio_info_creation() {
        let audio = AudioInfo {
            sample_rate: 48000,
            channels: 2,
            channel_layout: Some("stereo".to_string()),
            sample_format: Some("fltp".to_string()),
            bits_per_sample: Some(16),
        };

        assert_eq!(audio.sample_rate, 48000);
        assert_eq!(audio.channels, 2);
    }

    #[test]
    fn test_media_metadata_to_json() {
        let metadata = MediaMetadata {
            file_path: "test.mp4".to_string(),
            file_size: 1024,
            format: ContainerFormat {
                name: "mp4".to_string(),
                long_name: "MP4".to_string(),
                mime_type: Some("video/mp4".to_string()),
                extensions: vec!["mp4".to_string()],
            },
            streams: vec![],
            duration: 10.0,
            bitrate: Some(1000000),
        };

        let json = metadata.to_json();
        assert!(json.is_ok());
    }

    #[test]
    fn test_stream_info_display() {
        let stream = StreamInfo {
            index: 0,
            stream_type: StreamType::Video,
            codec: CodecInfo {
                name: "h264".to_string(),
                long_name: "H.264 / AVC".to_string(),
                codec_type: StreamType::Video,
            },
            duration: Some(10.0),
            bitrate: Some(5000000),
            video_info: Some(VideoInfo {
                width: 1920,
                height: 1080,
                pixel_format: Some("yuv420p".to_string()),
                frame_rate: Some(30.0),
                aspect_ratio: Some("16:9".to_string()),
                bit_depth: Some(8),
                color_space: Some("bt709".to_string()),
            }),
            audio_info: None,
        };

        let display = format!("{}", stream);
        assert!(display.contains("Video"));
        assert!(display.contains("H.264"));
        assert!(display.contains("1920x1080"));
    }
}
