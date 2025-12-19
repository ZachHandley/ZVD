//! Container Format Detection
//!
//! This module detects container formats by examining file headers and magic numbers.

use crate::error::{Error, Result};
use crate::probe::metadata::ContainerFormat;
use std::io::{Read, Seek, SeekFrom};

/// Format detector for container formats
pub struct FormatDetector;

impl FormatDetector {
    /// Detect container format from file
    pub fn detect<R: Read + Seek>(reader: &mut R) -> Result<ContainerFormat> {
        // Read first 16 bytes for magic number detection
        let mut header = [0u8; 16];
        reader.seek(SeekFrom::Start(0))?;
        reader.read_exact(&mut header)?;

        // Reset to start
        reader.seek(SeekFrom::Start(0))?;

        // Check various container formats
        if Self::is_mp4(&header) {
            Ok(Self::mp4_format())
        } else if Self::is_webm(&header) {
            Ok(Self::webm_format())
        } else if Self::is_matroska(&header) {
            Ok(Self::matroska_format())
        } else if Self::is_avi(&header) {
            Ok(Self::avi_format())
        } else if Self::is_wav(&header) {
            Ok(Self::wav_format())
        } else if Self::is_ogg(&header) {
            Ok(Self::ogg_format())
        } else if Self::is_flac(&header) {
            Ok(Self::flac_format())
        } else if Self::is_mp3(&header) {
            Ok(Self::mp3_format())
        } else if Self::is_y4m(&header) {
            Ok(Self::y4m_format())
        } else {
            Ok(Self::unknown_format())
        }
    }

    /// Check if file is MP4
    fn is_mp4(header: &[u8]) -> bool {
        if header.len() < 8 {
            return false;
        }
        // MP4 files have 'ftyp' atom at offset 4
        &header[4..8] == b"ftyp"
    }

    /// Check if file is WebM
    fn is_webm(header: &[u8]) -> bool {
        if header.len() < 4 {
            return false;
        }
        // WebM/Matroska EBML header starts with 0x1A 0x45 0xDF 0xA3
        header[0] == 0x1A && header[1] == 0x45 && header[2] == 0xDF && header[3] == 0xA3
    }

    /// Check if file is Matroska (same as WebM, differentiated by DocType)
    fn is_matroska(header: &[u8]) -> bool {
        Self::is_webm(header)
    }

    /// Check if file is AVI
    fn is_avi(header: &[u8]) -> bool {
        if header.len() < 12 {
            return false;
        }
        // AVI: RIFF....AVI
        &header[0..4] == b"RIFF" && &header[8..12] == b"AVI "
    }

    /// Check if file is WAV
    fn is_wav(header: &[u8]) -> bool {
        if header.len() < 12 {
            return false;
        }
        // WAV: RIFF....WAVE
        &header[0..4] == b"RIFF" && &header[8..12] == b"WAVE"
    }

    /// Check if file is Ogg
    fn is_ogg(header: &[u8]) -> bool {
        if header.len() < 4 {
            return false;
        }
        &header[0..4] == b"OggS"
    }

    /// Check if file is FLAC
    fn is_flac(header: &[u8]) -> bool {
        if header.len() < 4 {
            return false;
        }
        &header[0..4] == b"fLaC"
    }

    /// Check if file is MP3
    fn is_mp3(header: &[u8]) -> bool {
        if header.len() < 3 {
            return false;
        }
        // MP3 can start with ID3v2 tag or frame sync
        (&header[0..3] == b"ID3") || (header[0] == 0xFF && (header[1] & 0xE0) == 0xE0)
    }

    /// Check if file is Y4M
    fn is_y4m(header: &[u8]) -> bool {
        if header.len() < 9 {
            return false;
        }
        &header[0..9] == b"YUV4MPEG2"
    }

    // Format constructors

    fn mp4_format() -> ContainerFormat {
        ContainerFormat {
            name: "mp4".to_string(),
            long_name: "MP4 (MPEG-4 Part 14)".to_string(),
            mime_type: Some("video/mp4".to_string()),
            extensions: vec!["mp4".to_string(), "m4v".to_string(), "m4a".to_string()],
        }
    }

    fn webm_format() -> ContainerFormat {
        ContainerFormat {
            name: "webm".to_string(),
            long_name: "WebM".to_string(),
            mime_type: Some("video/webm".to_string()),
            extensions: vec!["webm".to_string()],
        }
    }

    fn matroska_format() -> ContainerFormat {
        ContainerFormat {
            name: "matroska".to_string(),
            long_name: "Matroska / MKV".to_string(),
            mime_type: Some("video/x-matroska".to_string()),
            extensions: vec!["mkv".to_string(), "mka".to_string(), "mks".to_string()],
        }
    }

    fn avi_format() -> ContainerFormat {
        ContainerFormat {
            name: "avi".to_string(),
            long_name: "AVI (Audio Video Interleaved)".to_string(),
            mime_type: Some("video/x-msvideo".to_string()),
            extensions: vec!["avi".to_string()],
        }
    }

    fn wav_format() -> ContainerFormat {
        ContainerFormat {
            name: "wav".to_string(),
            long_name: "WAV / WAVE (Waveform Audio)".to_string(),
            mime_type: Some("audio/wav".to_string()),
            extensions: vec!["wav".to_string()],
        }
    }

    fn ogg_format() -> ContainerFormat {
        ContainerFormat {
            name: "ogg".to_string(),
            long_name: "Ogg".to_string(),
            mime_type: Some("audio/ogg".to_string()),
            extensions: vec!["ogg".to_string(), "oga".to_string(), "ogv".to_string()],
        }
    }

    fn flac_format() -> ContainerFormat {
        ContainerFormat {
            name: "flac".to_string(),
            long_name: "FLAC (Free Lossless Audio Codec)".to_string(),
            mime_type: Some("audio/flac".to_string()),
            extensions: vec!["flac".to_string()],
        }
    }

    fn mp3_format() -> ContainerFormat {
        ContainerFormat {
            name: "mp3".to_string(),
            long_name: "MP3 (MPEG audio layer 3)".to_string(),
            mime_type: Some("audio/mpeg".to_string()),
            extensions: vec!["mp3".to_string()],
        }
    }

    fn y4m_format() -> ContainerFormat {
        ContainerFormat {
            name: "y4m".to_string(),
            long_name: "YUV4MPEG2".to_string(),
            mime_type: None,
            extensions: vec!["y4m".to_string()],
        }
    }

    fn unknown_format() -> ContainerFormat {
        ContainerFormat {
            name: "unknown".to_string(),
            long_name: "Unknown Format".to_string(),
            mime_type: None,
            extensions: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_mp4_detection() {
        let header = b"\x00\x00\x00\x20ftypmp42";
        assert!(FormatDetector::is_mp4(header));
    }

    #[test]
    fn test_webm_detection() {
        let header = b"\x1A\x45\xDF\xA3\x01\x00\x00\x00";
        assert!(FormatDetector::is_webm(header));
    }

    #[test]
    fn test_avi_detection() {
        let header = b"RIFF\x00\x00\x00\x00AVI \x00\x00";
        assert!(FormatDetector::is_avi(header));
    }

    #[test]
    fn test_wav_detection() {
        let header = b"RIFF\x00\x00\x00\x00WAVE\x00\x00";
        assert!(FormatDetector::is_wav(header));
    }

    #[test]
    fn test_ogg_detection() {
        let header = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00";
        assert!(FormatDetector::is_ogg(header));
    }

    #[test]
    fn test_flac_detection() {
        let header = b"fLaC\x00\x00\x00\x22\x00\x00\x00\x00";
        assert!(FormatDetector::is_flac(header));
    }

    #[test]
    fn test_mp3_detection() {
        let header_id3 = b"ID3\x03\x00\x00\x00\x00\x00\x00";
        assert!(FormatDetector::is_mp3(header_id3));

        let header_sync = b"\xFF\xFB\x90\x00\x00\x00\x00\x00";
        assert!(FormatDetector::is_mp3(header_sync));
    }

    #[test]
    fn test_y4m_detection() {
        let header = b"YUV4MPEG2 W1920 H1080";
        assert!(FormatDetector::is_y4m(header));
    }

    #[test]
    fn test_format_detection_mp4() {
        let data = b"\x00\x00\x00\x20ftypmp42\x00\x00\x00\x00";
        let mut cursor = Cursor::new(data);

        let format = FormatDetector::detect(&mut cursor).unwrap();
        assert_eq!(format.name, "mp4");
        assert_eq!(format.long_name, "MP4 (MPEG-4 Part 14)");
    }

    #[test]
    fn test_format_detection_flac() {
        let data = b"fLaC\x00\x00\x00\x22\x00\x00\x00\x00";
        let mut cursor = Cursor::new(data);

        let format = FormatDetector::detect(&mut cursor).unwrap();
        assert_eq!(format.name, "flac");
    }

    #[test]
    fn test_format_detection_unknown() {
        let data = b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        let mut cursor = Cursor::new(data);

        let format = FormatDetector::detect(&mut cursor).unwrap();
        assert_eq!(format.name, "unknown");
    }
}
