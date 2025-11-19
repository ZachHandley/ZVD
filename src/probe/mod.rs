//! Media File Probing and Metadata Extraction
//!
//! This module provides FFprobe-equivalent functionality for inspecting media files
//! and extracting comprehensive metadata about codecs, streams, and containers.
//!
//! # Overview
//!
//! The probe module can extract:
//! - **Container format** (MP4, WebM, MKV, AVI, etc.)
//! - **Stream information** (video, audio, subtitle streams)
//! - **Codec details** (H.264, H.265, VP9, Opus, etc.)
//! - **Technical metadata** (resolution, bitrate, duration, frame rate)
//! - **File information** (size, creation time, etc.)
//!
//! # Usage
//!
//! ```rust,no_run
//! use zvd_lib::probe::MediaProbe;
//!
//! // Probe a media file
//! let probe = MediaProbe::new("video.mp4")?;
//! let metadata = probe.analyze()?;
//!
//! // Print human-readable summary
//! println!("{}", metadata.to_string());
//!
//! // Or get JSON output
//! let json = metadata.to_json()?;
//! # Ok::<(), zvd_lib::error::Error>(())
//! ```

pub mod metadata;
pub mod format_detector;
pub mod stream_analyzer;

use crate::error::{Error, Result};
use std::path::Path;
use std::fs;

pub use metadata::*;
pub use format_detector::FormatDetector;
pub use stream_analyzer::StreamAnalyzer;

/// Media file probe - FFprobe equivalent for ZVD
pub struct MediaProbe {
    /// Path to media file
    file_path: String,
    /// File size in bytes
    file_size: u64,
}

impl MediaProbe {
    /// Create new media probe for a file
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file_path = path.as_ref().to_string_lossy().to_string();

        let metadata = fs::metadata(&path).map_err(|e| {
            Error::InvalidData(format!("Failed to access file: {}", e))
        })?;

        let file_size = metadata.len();

        Ok(Self {
            file_path,
            file_size,
        })
    }

    /// Analyze the media file and extract all metadata
    pub fn analyze(&self) -> Result<MediaMetadata> {
        // Read file header to detect format
        let mut file = fs::File::open(&self.file_path).map_err(|e| {
            Error::InvalidData(format!("Failed to open file: {}", e))
        })?;

        // Detect container format
        let format = FormatDetector::detect(&mut file)?;

        // Analyze streams based on container type
        let streams = StreamAnalyzer::analyze_streams(&mut file, &format)?;

        // Calculate duration from streams if available
        let duration = Self::calculate_duration(&streams);

        // Calculate bitrate
        let bitrate = if duration > 0.0 {
            Some(((self.file_size * 8) as f64 / duration) as u64)
        } else {
            None
        };

        Ok(MediaMetadata {
            file_path: self.file_path.clone(),
            file_size: self.file_size,
            format,
            streams,
            duration,
            bitrate,
        })
    }

    /// Calculate total duration from streams
    fn calculate_duration(streams: &[StreamInfo]) -> f64 {
        streams
            .iter()
            .filter_map(|s| s.duration)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_media_probe_creation() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"test data").unwrap();

        let probe = MediaProbe::new(file.path());
        assert!(probe.is_ok());

        let probe = probe.unwrap();
        assert_eq!(probe.file_size, 9);
    }

    #[test]
    fn test_media_probe_nonexistent_file() {
        let probe = MediaProbe::new("/nonexistent/file.mp4");
        assert!(probe.is_err());
    }

    #[test]
    fn test_calculate_duration() {
        let streams = vec![
            StreamInfo {
                index: 0,
                stream_type: StreamType::Video,
                codec: CodecInfo {
                    name: "h264".to_string(),
                    long_name: "H.264 / AVC".to_string(),
                    codec_type: StreamType::Video,
                },
                duration: Some(10.0),
                bitrate: None,
                video_info: None,
                audio_info: None,
            },
            StreamInfo {
                index: 1,
                stream_type: StreamType::Audio,
                codec: CodecInfo {
                    name: "opus".to_string(),
                    long_name: "Opus".to_string(),
                    codec_type: StreamType::Audio,
                },
                duration: Some(10.5),
                bitrate: None,
                video_info: None,
                audio_info: None,
            },
        ];

        let duration = MediaProbe::calculate_duration(&streams);
        assert_eq!(duration, 10.5);
    }
}
