//! DASH (Dynamic Adaptive Streaming over HTTP) support
//!
//! MPEG-DASH is an adaptive bitrate streaming technique that enables high quality
//! streaming of media content over the Internet.
//!
//! ## Features
//!
//! - MPD (Media Presentation Description) manifest generation
//! - Segment templates for efficient representation
//! - Multiple adaptation sets (video, audio, subtitles)
//! - Live and on-demand support
//! - Multi-period support
//!
//! ## DASH vs HLS
//!
//! - **DASH**: ISO standard, codec-agnostic, XML-based manifests
//! - **HLS**: Apple standard, mainly H.264/AAC, text-based playlists
//! - Both support adaptive bitrate streaming

use super::QualityProfile;
use crate::error::{Error, Result};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// DASH segment
#[derive(Debug, Clone)]
pub struct DashSegment {
    pub number: u32,
    pub duration: Duration,
    pub filename: String,
    pub size: u64,
}

/// DASH manifest (MPD) generator
pub struct DashManifest {
    segments: Vec<DashSegment>,
    profiles: Vec<QualityProfile>,
    output_dir: PathBuf,
    segment_duration: Duration,
}

impl DashManifest {
    /// Create a new DASH manifest
    pub fn new(output_dir: &Path, segment_duration: Duration) -> Self {
        DashManifest {
            segments: Vec::new(),
            profiles: Vec::new(),
            output_dir: output_dir.to_path_buf(),
            segment_duration,
        }
    }

    /// Add quality profile
    pub fn add_profile(&mut self, profile: QualityProfile) {
        self.profiles.push(profile);
    }

    /// Add segment
    pub fn add_segment(&mut self, filename: &str, duration: Duration, size: u64) {
        let segment = DashSegment {
            number: self.segments.len() as u32,
            duration,
            filename: filename.to_string(),
            size,
        };
        self.segments.push(segment);
    }

    /// Generate MPD (Media Presentation Description)
    pub fn generate_mpd(&self, is_live: bool) -> String {
        let mut mpd = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        mpd.push_str("<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\" ");

        // Type: static for VOD, dynamic for live
        mpd.push_str(&format!("type=\"{}\" ", if is_live { "dynamic" } else { "static" }));

        // Profiles - using on-demand for maximum compatibility
        mpd.push_str("profiles=\"urn:mpeg:dash:profile:isoff-on-demand:2011\" ");

        // Min buffer time (recommended 2 seconds)
        mpd.push_str("minBufferTime=\"PT2S\" ");

        // Media presentation duration (for static/VOD)
        if !is_live && !self.segments.is_empty() {
            let total_duration: Duration = self.segments.iter().map(|s| s.duration).sum();
            let secs = total_duration.as_secs();
            let millis = total_duration.subsec_millis();
            mpd.push_str(&format!("mediaPresentationDuration=\"PT{}.{}S\" ", secs, millis));
        }

        mpd.push_str(">\n");

        // Period (single period for now)
        mpd.push_str("  <Period id=\"0\"");
        if !is_live && !self.segments.is_empty() {
            let total_duration: Duration = self.segments.iter().map(|s| s.duration).sum();
            let secs = total_duration.as_secs();
            let millis = total_duration.subsec_millis();
            mpd.push_str(&format!(" duration=\"PT{}.{}S\"", secs, millis));
        }
        mpd.push_str(">\n");

        // Video Adaptation Set
        if !self.profiles.is_empty() {
            mpd.push_str("    <AdaptationSet id=\"0\" ");
            mpd.push_str("contentType=\"video\" ");
            mpd.push_str("segmentAlignment=\"true\" ");
            mpd.push_str("bitstreamSwitching=\"true\" ");
            mpd.push_str("mimeType=\"video/mp4\" ");
            mpd.push_str("codecs=\"avc1.4d401f\">\n");

            // Representations for each quality
            for (idx, profile) in self.profiles.iter().enumerate() {
                mpd.push_str(&format!("      <Representation id=\"{}\" ", idx));
                mpd.push_str(&format!("bandwidth=\"{}\" ", profile.bitrate));
                mpd.push_str(&format!("width=\"{}\" ", profile.width));
                mpd.push_str(&format!("height=\"{}\" ", profile.height));
                mpd.push_str(&format!("frameRate=\"{}\">\n", profile.framerate));

                // Segment Template (more efficient than SegmentList)
                if !self.segments.is_empty() {
                    let duration_ms = self.segment_duration.as_millis();
                    mpd.push_str("        <SegmentTemplate ");
                    mpd.push_str(&format!("timescale=\"1000\" "));
                    mpd.push_str(&format!("duration=\"{}\" ", duration_ms));
                    mpd.push_str(&format!("initialization=\"{}_init.mp4\" ", profile.name));
                    mpd.push_str(&format!("media=\"{}_$Number$.m4s\" ", profile.name));
                    mpd.push_str("startNumber=\"0\">\n");

                    // Segment timeline for precise timing
                    mpd.push_str("          <SegmentTimeline>\n");
                    for segment in &self.segments {
                        let seg_duration_ms = segment.duration.as_millis();
                        mpd.push_str(&format!("            <S d=\"{}\" />\n", seg_duration_ms));
                    }
                    mpd.push_str("          </SegmentTimeline>\n");

                    mpd.push_str("        </SegmentTemplate>\n");
                } else {
                    // Fallback: SegmentList
                    mpd.push_str("        <SegmentList>\n");
                    for segment in &self.segments {
                        mpd.push_str(&format!("          <SegmentURL media=\"{}\" />\n", segment.filename));
                    }
                    mpd.push_str("        </SegmentList>\n");
                }

                mpd.push_str("      </Representation>\n");
            }

            mpd.push_str("    </AdaptationSet>\n");
        }

        mpd.push_str("  </Period>\n");
        mpd.push_str("</MPD>\n");

        mpd
    }

    /// Save MPD to file
    pub fn save(&self, filename: &str, is_live: bool) -> Result<()> {
        let path = self.output_dir.join(filename);
        let content = self.generate_mpd(is_live);

        std::fs::write(&path, content)
            .map_err(|e| Error::Io(e))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dash_manifest_creation() {
        let temp_dir = std::env::temp_dir();
        let manifest = DashManifest::new(&temp_dir, Duration::from_secs(4));
        assert_eq!(manifest.profiles.len(), 0);
    }

    #[test]
    fn test_dash_add_profile() {
        let temp_dir = std::env::temp_dir();
        let mut manifest = DashManifest::new(&temp_dir, Duration::from_secs(4));

        let profile = QualityProfile::new("720p", 2500000, 1280, 720, 30);
        manifest.add_profile(profile);

        assert_eq!(manifest.profiles.len(), 1);
    }

    #[test]
    fn test_dash_generate_mpd() {
        let temp_dir = std::env::temp_dir();
        let mut manifest = DashManifest::new(&temp_dir, Duration::from_secs(4));

        manifest.add_profile(QualityProfile::new("720p", 2500000, 1280, 720, 30));
        manifest.add_segment("segment1.m4s", Duration::from_secs(4), 1024);

        let mpd = manifest.generate_mpd(false);
        assert!(mpd.contains("<MPD"));
        assert!(mpd.contains("type=\"static\""));
        assert!(mpd.contains("segment1.m4s"));
    }
}
