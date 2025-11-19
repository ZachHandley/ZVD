//! Low-Latency HLS (LL-HLS) support
//!
//! LL-HLS is an extension to HLS that enables sub-second latencies (2-3 seconds)
//! compared to traditional HLS latencies of 20-30 seconds.
//!
//! ## Key Features
//!
//! - **Partial Segments**: Segments split into smaller parts delivered incrementally
//! - **Preload Hints**: Directives telling clients about upcoming content
//! - **Blocking Playlist Reload**: Clients can wait for updates instead of polling
//! - **Rendition Reports**: Information about other quality variants
//! - **Lower Target Duration**: 2-6 seconds vs traditional 6-10 seconds
//!
//! ## Protocol Enhancements (RFC 8216 Delta Updates)
//!
//! - `#EXT-X-SERVER-CONTROL`: Server capabilities
//! - `#EXT-X-PART-INF`: Partial segment target duration
//! - `#EXT-X-PART`: Partial segment entry
//! - `#EXT-X-PRELOAD-HINT`: Upcoming segment hints
//! - `#EXT-X-RENDITION-REPORT`: Cross-variant information

use super::{HlsSegment, QualityProfile};
use crate::error::{Error, Result};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// LL-HLS partial segment (sub-segment)
#[derive(Debug, Clone)]
pub struct PartialSegment {
    /// Part index within parent segment
    pub part_index: u32,
    /// Duration of this part
    pub duration: Duration,
    /// Filename
    pub filename: String,
    /// Size in bytes
    pub size: u64,
    /// Is this an independent part (has keyframe)?
    pub independent: bool,
    /// Byte range within file (optional, for byte-range serving)
    pub byte_range: Option<(u64, u64)>, // (start, length)
}

/// LL-HLS enhanced segment with partial segments
#[derive(Debug, Clone)]
pub struct LlHlsSegment {
    /// Base segment info
    pub segment: HlsSegment,
    /// Partial segments (parts) within this segment
    pub parts: Vec<PartialSegment>,
    /// Is this segment complete?
    pub complete: bool,
}

impl LlHlsSegment {
    /// Create a new LL-HLS segment
    pub fn new(segment: HlsSegment) -> Self {
        LlHlsSegment {
            segment,
            parts: Vec::new(),
            complete: false,
        }
    }

    /// Add a partial segment
    pub fn add_part(&mut self, part: PartialSegment) {
        self.parts.push(part);
    }

    /// Mark segment as complete
    pub fn complete(&mut self) {
        self.complete = true;
    }

    /// Get total duration from all parts
    pub fn total_duration(&self) -> Duration {
        self.parts.iter().map(|p| p.duration).sum()
    }
}

/// Preload hint types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreloadHintType {
    /// Next partial segment
    Part,
    /// Next full segment (if no parts)
    Map,
}

/// Preload hint for upcoming content
#[derive(Debug, Clone)]
pub struct PreloadHint {
    /// Type of content
    pub hint_type: PreloadHintType,
    /// URI to preload
    pub uri: String,
    /// Byte range offset (for partial content)
    pub byte_range_start: Option<u64>,
    /// Byte range length
    pub byte_range_length: Option<u64>,
}

/// Server control parameters
#[derive(Debug, Clone)]
pub struct ServerControl {
    /// Can clients use blocking playlist reload?
    pub can_block_reload: bool,
    /// Maximum blocking duration (if can_block_reload)
    pub hold_back: Option<Duration>,
    /// Part hold back (for partial segments)
    pub part_hold_back: Option<Duration>,
    /// Can skip older segments?
    pub can_skip_until: Option<Duration>,
}

impl Default for ServerControl {
    fn default() -> Self {
        ServerControl {
            can_block_reload: true,
            hold_back: Some(Duration::from_secs(3)),
            part_hold_back: Some(Duration::from_millis(500)),
            can_skip_until: None,
        }
    }
}

/// Rendition report for cross-variant information
#[derive(Debug, Clone)]
pub struct RenditionReport {
    /// URI of the variant playlist
    pub uri: String,
    /// Last MSN (Media Sequence Number)
    pub last_msn: u32,
    /// Last part index
    pub last_part: Option<u32>,
}

/// Low-Latency HLS playlist generator
pub struct LlHlsPlaylist {
    segments: Vec<LlHlsSegment>,
    target_duration: Duration,
    part_target_duration: Duration,
    sequence_number: u32,
    output_dir: PathBuf,
    server_control: ServerControl,
    preload_hints: Vec<PreloadHint>,
    rendition_reports: Vec<RenditionReport>,
}

impl LlHlsPlaylist {
    /// Create a new LL-HLS playlist
    ///
    /// - target_duration: Target segment duration (2-6 seconds recommended)
    /// - part_target_duration: Target partial segment duration (200-500ms recommended)
    pub fn new(
        output_dir: &Path,
        target_duration: Duration,
        part_target_duration: Duration,
    ) -> Self {
        LlHlsPlaylist {
            segments: Vec::new(),
            target_duration,
            part_target_duration,
            sequence_number: 0,
            output_dir: output_dir.to_path_buf(),
            server_control: ServerControl::default(),
            preload_hints: Vec::new(),
            rendition_reports: Vec::new(),
        }
    }

    /// Set server control parameters
    pub fn set_server_control(&mut self, control: ServerControl) {
        self.server_control = control;
    }

    /// Add a segment with parts
    pub fn add_segment(&mut self, segment: LlHlsSegment) {
        self.segments.push(segment);
        self.sequence_number += 1;
    }

    /// Add a preload hint
    pub fn add_preload_hint(&mut self, hint: PreloadHint) {
        self.preload_hints.push(hint);
    }

    /// Add a rendition report
    pub fn add_rendition_report(&mut self, report: RenditionReport) {
        self.rendition_reports.push(report);
    }

    /// Generate LL-HLS master playlist
    pub fn generate_master_playlist(&self, profiles: &[QualityProfile]) -> String {
        let mut playlist = String::from("#EXTM3U\n");
        playlist.push_str("#EXT-X-VERSION:9\n"); // Version 9 for LL-HLS
        playlist.push_str("#EXT-X-INDEPENDENT-SEGMENTS\n\n");

        for profile in profiles {
            playlist.push_str(&format!(
                "#EXT-X-STREAM-INF:BANDWIDTH={},AVERAGE-BANDWIDTH={},RESOLUTION={}x{},FRAME-RATE={}\n",
                profile.bitrate,
                (profile.bitrate as f64 * 0.9) as u32, // Average slightly lower
                profile.width,
                profile.height,
                profile.framerate
            ));
            playlist.push_str(&format!("{}.m3u8\n", profile.name));
        }

        playlist
    }

    /// Generate LL-HLS media playlist
    pub fn generate_media_playlist(&self, is_live: bool) -> String {
        let mut playlist = String::from("#EXTM3U\n");
        playlist.push_str("#EXT-X-VERSION:9\n"); // Version 9 required for LL-HLS

        // Target durations
        playlist.push_str(&format!(
            "#EXT-X-TARGETDURATION:{}\n",
            self.target_duration.as_secs()
        ));

        // Server control for LL-HLS
        if is_live {
            let mut control_line = String::from("#EXT-X-SERVER-CONTROL");

            if self.server_control.can_block_reload {
                control_line.push_str(":CAN-BLOCK-RELOAD=YES");
            }

            if let Some(hold_back) = self.server_control.hold_back {
                control_line.push_str(&format!(
                    ",HOLD-BACK={:.3}",
                    hold_back.as_secs_f64()
                ));
            }

            if let Some(part_hold_back) = self.server_control.part_hold_back {
                control_line.push_str(&format!(
                    ",PART-HOLD-BACK={:.3}",
                    part_hold_back.as_secs_f64()
                ));
            }

            if let Some(can_skip) = self.server_control.can_skip_until {
                control_line.push_str(&format!(
                    ",CAN-SKIP-UNTIL={:.3}",
                    can_skip.as_secs_f64()
                ));
            }

            playlist.push_str(&control_line);
            playlist.push('\n');
        }

        // Part info (target duration for partial segments)
        playlist.push_str(&format!(
            "#EXT-X-PART-INF:PART-TARGET={:.3}\n",
            self.part_target_duration.as_secs_f64()
        ));

        // Media sequence
        playlist.push_str(&format!("#EXT-X-MEDIA-SEQUENCE:{}\n", self.sequence_number));

        // Playlist type
        if !is_live {
            playlist.push_str("#EXT-X-PLAYLIST-TYPE:VOD\n");
        }

        // Segments with partial segments
        for ll_segment in &self.segments {
            // If segment has parts, write them first
            for part in &ll_segment.parts {
                playlist.push_str(&format!(
                    "#EXT-X-PART:DURATION={:.5},URI=\"{}\"",
                    part.duration.as_secs_f64(),
                    part.filename
                ));

                if part.independent {
                    playlist.push_str(",INDEPENDENT=YES");
                }

                if let Some((start, length)) = part.byte_range {
                    playlist.push_str(&format!(",BYTERANGE={}@{}", length, start));
                }

                playlist.push('\n');
            }

            // Write full segment entry
            playlist.push_str(&format!(
                "#EXTINF:{:.3},\n",
                ll_segment.segment.duration.as_secs_f64()
            ));

            playlist.push_str(&format!("{}\n", ll_segment.segment.filename));
        }

        // Preload hints (for upcoming content)
        for hint in &self.preload_hints {
            let type_str = match hint.hint_type {
                PreloadHintType::Part => "PART",
                PreloadHintType::Map => "MAP",
            };

            playlist.push_str(&format!(
                "#EXT-X-PRELOAD-HINT:TYPE={},URI=\"{}\"",
                type_str, hint.uri
            ));

            if let Some(start) = hint.byte_range_start {
                if let Some(length) = hint.byte_range_length {
                    playlist.push_str(&format!(",BYTERANGE-START={},BYTERANGE-LENGTH={}", start, length));
                }
            }

            playlist.push('\n');
        }

        // Rendition reports (info about other variants)
        for report in &self.rendition_reports {
            playlist.push_str(&format!(
                "#EXT-X-RENDITION-REPORT:URI=\"{}\",LAST-MSN={}",
                report.uri, report.last_msn
            ));

            if let Some(last_part) = report.last_part {
                playlist.push_str(&format!(",LAST-PART={}", last_part));
            }

            playlist.push('\n');
        }

        // End list for VOD
        if !is_live {
            playlist.push_str("#EXT-X-ENDLIST\n");
        }

        playlist
    }

    /// Save playlist to file
    pub fn save(&self, filename: &str, is_live: bool) -> Result<()> {
        let path = self.output_dir.join(filename);
        let content = self.generate_media_playlist(is_live);

        std::fs::write(&path, content).map_err(|e| Error::Io(e))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ll_hls_playlist_creation() {
        let temp_dir = std::env::temp_dir();
        let playlist = LlHlsPlaylist::new(
            &temp_dir,
            Duration::from_secs(4),
            Duration::from_millis(300),
        );

        assert_eq!(playlist.target_duration, Duration::from_secs(4));
        assert_eq!(playlist.part_target_duration, Duration::from_millis(300));
    }

    #[test]
    fn test_partial_segment() {
        let part = PartialSegment {
            part_index: 0,
            duration: Duration::from_millis(300),
            filename: "seg0_part0.m4s".to_string(),
            size: 12345,
            independent: true,
            byte_range: None,
        };

        assert_eq!(part.duration, Duration::from_millis(300));
        assert!(part.independent);
    }

    #[test]
    fn test_ll_hls_segment_with_parts() {
        let base_segment = HlsSegment {
            index: 0,
            duration: Duration::from_secs(4),
            filename: "seg0.ts".to_string(),
            size: 100000,
        };

        let mut ll_segment = LlHlsSegment::new(base_segment);

        // Add 4 parts of 1 second each
        for i in 0..4 {
            ll_segment.add_part(PartialSegment {
                part_index: i,
                duration: Duration::from_secs(1),
                filename: format!("seg0_part{}.m4s", i),
                size: 25000,
                independent: i == 0,
                byte_range: None,
            });
        }

        ll_segment.complete();

        assert_eq!(ll_segment.parts.len(), 4);
        assert_eq!(ll_segment.total_duration(), Duration::from_secs(4));
        assert!(ll_segment.complete);
    }

    #[test]
    fn test_server_control() {
        let control = ServerControl {
            can_block_reload: true,
            hold_back: Some(Duration::from_secs(3)),
            part_hold_back: Some(Duration::from_millis(500)),
            can_skip_until: Some(Duration::from_secs(12)),
        };

        assert!(control.can_block_reload);
        assert_eq!(control.hold_back, Some(Duration::from_secs(3)));
    }

    #[test]
    fn test_preload_hint() {
        let hint = PreloadHint {
            hint_type: PreloadHintType::Part,
            uri: "seg5_part0.m4s".to_string(),
            byte_range_start: Some(0),
            byte_range_length: Some(25000),
        };

        assert_eq!(hint.hint_type, PreloadHintType::Part);
        assert_eq!(hint.uri, "seg5_part0.m4s");
    }

    #[test]
    fn test_ll_hls_master_playlist() {
        let temp_dir = std::env::temp_dir();
        let playlist = LlHlsPlaylist::new(
            &temp_dir,
            Duration::from_secs(3),
            Duration::from_millis(300),
        );

        let profiles = vec![
            QualityProfile::new("720p", 2_500_000, 1280, 720, 30),
            QualityProfile::new("1080p", 5_000_000, 1920, 1080, 30),
        ];

        let master = playlist.generate_master_playlist(&profiles);

        assert!(master.contains("#EXT-X-VERSION:9"));
        assert!(master.contains("#EXT-X-INDEPENDENT-SEGMENTS"));
        assert!(master.contains("720p.m3u8"));
        assert!(master.contains("1080p.m3u8"));
    }

    #[test]
    fn test_ll_hls_media_playlist() {
        let temp_dir = std::env::temp_dir();
        let mut playlist = LlHlsPlaylist::new(
            &temp_dir,
            Duration::from_secs(4),
            Duration::from_millis(300),
        );

        // Add segment with parts
        let base_segment = HlsSegment {
            index: 0,
            duration: Duration::from_secs(4),
            filename: "seg0.ts".to_string(),
            size: 100000,
        };

        let mut ll_segment = LlHlsSegment::new(base_segment);
        ll_segment.add_part(PartialSegment {
            part_index: 0,
            duration: Duration::from_secs(1),
            filename: "seg0_part0.m4s".to_string(),
            size: 25000,
            independent: true,
            byte_range: None,
        });
        ll_segment.complete();

        playlist.add_segment(ll_segment);

        let media_playlist = playlist.generate_media_playlist(true);

        assert!(media_playlist.contains("#EXT-X-VERSION:9"));
        assert!(media_playlist.contains("#EXT-X-SERVER-CONTROL"));
        assert!(media_playlist.contains("#EXT-X-PART-INF"));
        assert!(media_playlist.contains("#EXT-X-PART:"));
    }
}
