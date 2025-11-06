//! HLS (HTTP Live Streaming) support
//!
//! HLS is Apple's adaptive bitrate streaming protocol.
//! Segments video into small HTTP-downloadable files with a manifest playlist.

use super::QualityProfile;
use crate::error::{Error, Result};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// HLS segment
#[derive(Debug, Clone)]
pub struct HlsSegment {
    pub index: u32,
    pub duration: Duration,
    pub filename: String,
    pub size: u64,
}

/// HLS playlist generator
pub struct HlsPlaylist {
    segments: Vec<HlsSegment>,
    target_duration: Duration,
    sequence_number: u32,
    output_dir: PathBuf,
}

impl HlsPlaylist {
    /// Create a new HLS playlist
    pub fn new(output_dir: &Path, target_duration: Duration) -> Self {
        HlsPlaylist {
            segments: Vec::new(),
            target_duration,
            sequence_number: 0,
            output_dir: output_dir.to_path_buf(),
        }
    }

    /// Add a segment to the playlist
    pub fn add_segment(&mut self, filename: &str, duration: Duration, size: u64) {
        let segment = HlsSegment {
            index: self.sequence_number,
            duration,
            filename: filename.to_string(),
            size,
        };

        self.segments.push(segment);
        self.sequence_number += 1;
    }

    /// Generate master playlist (for adaptive bitrate)
    pub fn generate_master_playlist(&self, profiles: &[QualityProfile]) -> String {
        let mut playlist = String::from("#EXTM3U\n");
        playlist.push_str("#EXT-X-VERSION:3\n\n");

        for profile in profiles {
            playlist.push_str(&format!(
                "#EXT-X-STREAM-INF:BANDWIDTH={},RESOLUTION={}x{},FRAME-RATE={}\n",
                profile.bitrate, profile.width, profile.height, profile.framerate
            ));
            playlist.push_str(&format!("{}.m3u8\n", profile.name));
        }

        playlist
    }

    /// Generate media playlist
    pub fn generate_media_playlist(&self, is_live: bool) -> String {
        let mut playlist = String::from("#EXTM3U\n");
        playlist.push_str("#EXT-X-VERSION:3\n");
        playlist.push_str(&format!(
            "#EXT-X-TARGETDURATION:{}\n",
            self.target_duration.as_secs()
        ));
        playlist.push_str(&format!("#EXT-X-MEDIA-SEQUENCE:{}\n", self.sequence_number));

        if !is_live {
            playlist.push_str("#EXT-X-PLAYLIST-TYPE:VOD\n");
        }

        for segment in &self.segments {
            playlist.push_str(&format!("#EXTINF:{:.3},\n", segment.duration.as_secs_f64()));
            playlist.push_str(&format!("{}\n", segment.filename));
        }

        if !is_live {
            playlist.push_str("#EXT-X-ENDLIST\n");
        }

        playlist
    }

    /// Save playlist to file
    pub fn save(&self, filename: &str, is_live: bool) -> Result<()> {
        let path = self.output_dir.join(filename);
        let content = self.generate_media_playlist(is_live);

        std::fs::write(&path, content)
            .map_err(|e| Error::Io(e))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_hls_playlist_creation() {
        let temp_dir = std::env::temp_dir();
        let playlist = HlsPlaylist::new(&temp_dir, Duration::from_secs(6));
        assert_eq!(playlist.segments.len(), 0);
    }

    #[test]
    fn test_hls_add_segment() {
        let temp_dir = std::env::temp_dir();
        let mut playlist = HlsPlaylist::new(&temp_dir, Duration::from_secs(6));

        playlist.add_segment("segment0.ts", Duration::from_secs(6), 1024);
        assert_eq!(playlist.segments.len(), 1);
        assert_eq!(playlist.segments[0].filename, "segment0.ts");
    }

    #[test]
    fn test_hls_generate_playlist() {
        let temp_dir = std::env::temp_dir();
        let mut playlist = HlsPlaylist::new(&temp_dir, Duration::from_secs(6));

        playlist.add_segment("segment0.ts", Duration::from_secs(6), 1024);
        playlist.add_segment("segment1.ts", Duration::from_secs(6), 1024);

        let content = playlist.generate_media_playlist(false);
        assert!(content.contains("#EXTM3U"));
        assert!(content.contains("segment0.ts"));
        assert!(content.contains("#EXT-X-ENDLIST"));
    }
}
