//! Streaming pipeline for HLS and DASH
//!
//! Provides high-level APIs for creating adaptive streaming content

use super::hls::{HlsPlaylist, HlsSegment};
use super::dash::DashManifest;
use super::segmenter::{Segmenter, SegmentConfig, SegmentInfo};
use super::QualityProfile;
use crate::error::{Error, Result};
use crate::format::{Packet, Stream};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// HLS streaming pipeline configuration
#[derive(Debug, Clone)]
pub struct HlsConfig {
    /// Output directory for segments and playlists
    pub output_dir: PathBuf,
    /// Target segment duration
    pub target_duration: Duration,
    /// Quality profiles for adaptive bitrate
    pub profiles: Vec<QualityProfile>,
    /// Is this a live stream?
    pub is_live: bool,
    /// Playlist name
    pub playlist_name: String,
}

impl HlsConfig {
    /// Create a new HLS configuration
    pub fn new<P: AsRef<Path>>(output_dir: P, target_duration: Duration) -> Self {
        HlsConfig {
            output_dir: output_dir.as_ref().to_path_buf(),
            target_duration,
            profiles: vec![QualityProfile::new("source", 5_000_000, 1920, 1080, 30)],
            is_live: false,
            playlist_name: "playlist".to_string(),
        }
    }

    /// Add a quality profile
    pub fn with_profile(mut self, profile: QualityProfile) -> Self {
        self.profiles.push(profile);
        self
    }

    /// Set as live stream
    pub fn as_live(mut self) -> Self {
        self.is_live = true;
        self
    }

    /// Set playlist name
    pub fn with_name(mut self, name: &str) -> Self {
        self.playlist_name = name.to_string();
        self
    }
}

/// HLS streaming pipeline
pub struct HlsPipeline {
    config: HlsConfig,
    segmenter: Segmenter,
    playlist: HlsPlaylist,
}

impl HlsPipeline {
    /// Create a new HLS pipeline
    pub fn new(config: HlsConfig) -> Result<Self> {
        // Create segment config
        let segment_config = SegmentConfig::new(&config.output_dir, config.target_duration);

        // Create segmenter
        let segmenter = Segmenter::new(segment_config)?;

        // Create playlist
        let playlist = HlsPlaylist::new(&config.output_dir, config.target_duration);

        Ok(HlsPipeline {
            config,
            segmenter,
            playlist,
        })
    }

    /// Add a stream to the pipeline
    pub fn add_stream(&mut self, stream: Stream) -> Result<()> {
        self.segmenter.add_stream(stream)
    }

    /// Write a packet to the pipeline
    pub fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        self.segmenter.write_packet(packet)
    }

    /// Finalize the pipeline and generate playlists
    pub fn finalize(&mut self) -> Result<()> {
        // Finalize segmentation
        self.segmenter.finalize()?;

        // Add all segments to playlist
        for segment in self.segmenter.segments() {
            self.playlist.add_segment(
                &segment.filename,
                segment.duration,
                segment.size,
            );
        }

        // Save media playlist
        let media_playlist_name = format!("{}.m3u8", self.config.playlist_name);
        self.playlist.save(&media_playlist_name, self.config.is_live)?;

        // Save master playlist if we have multiple profiles
        if self.config.profiles.len() > 1 {
            let master_content = self.playlist.generate_master_playlist(&self.config.profiles);
            let master_path = self.config.output_dir.join("master.m3u8");
            std::fs::write(master_path, master_content)
                .map_err(|e| Error::Io(e))?;
        }

        Ok(())
    }

    /// Get the number of segments created
    pub fn segment_count(&self) -> usize {
        self.segmenter.segment_count()
    }

    /// Get segment information
    pub fn segments(&self) -> &[SegmentInfo] {
        self.segmenter.segments()
    }
}

/// DASH streaming pipeline configuration
#[derive(Debug, Clone)]
pub struct DashConfig {
    /// Output directory
    pub output_dir: PathBuf,
    /// Segment duration
    pub segment_duration: Duration,
    /// Quality profiles
    pub profiles: Vec<QualityProfile>,
    /// Is live?
    pub is_live: bool,
}

impl DashConfig {
    /// Create new DASH configuration
    pub fn new<P: AsRef<Path>>(output_dir: P, segment_duration: Duration) -> Self {
        DashConfig {
            output_dir: output_dir.as_ref().to_path_buf(),
            segment_duration,
            profiles: vec![],
            is_live: false,
        }
    }

    /// Add quality profile
    pub fn with_profile(mut self, profile: QualityProfile) -> Self {
        self.profiles.push(profile);
        self
    }

    /// Set as live
    pub fn as_live(mut self) -> Self {
        self.is_live = true;
        self
    }
}

/// DASH streaming pipeline
pub struct DashPipeline {
    config: DashConfig,
    segmenter: Segmenter,
    manifest: DashManifest,
}

impl DashPipeline {
    /// Create new DASH pipeline
    pub fn new(config: DashConfig) -> Result<Self> {
        let segment_config = SegmentConfig::new(&config.output_dir, config.segment_duration);
        let segmenter = Segmenter::new(segment_config)?;

        // Create DASH manifest
        let manifest = DashManifest::new(&config.output_dir, config.segment_duration);

        Ok(DashPipeline {
            config,
            segmenter,
            manifest,
        })
    }

    /// Add stream
    pub fn add_stream(&mut self, stream: Stream) -> Result<()> {
        self.segmenter.add_stream(stream)
    }

    /// Write packet
    pub fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        self.segmenter.write_packet(packet)
    }

    /// Finalize and generate MPD manifest
    pub fn finalize(&mut self) -> Result<()> {
        // Finalize segmentation
        self.segmenter.finalize()?;

        // Add quality profiles to manifest
        for profile in &self.config.profiles {
            self.manifest.add_profile(profile.clone());
        }

        // Add all segments to manifest
        for segment in self.segmenter.segments() {
            self.manifest.add_segment(
                &segment.filename,
                segment.duration,
                segment.size,
            );
        }

        // Save MPD manifest
        self.manifest.save("manifest.mpd", self.config.is_live)?;

        Ok(())
    }

    /// Get the number of segments created
    pub fn segment_count(&self) -> usize {
        self.segmenter.segment_count()
    }

    /// Get segment information
    pub fn segments(&self) -> &[SegmentInfo] {
        self.segmenter.segments()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hls_config() {
        let config = HlsConfig::new("/tmp/hls", Duration::from_secs(6))
            .with_profile(QualityProfile::new("720p", 2_500_000, 1280, 720, 30))
            .as_live()
            .with_name("stream");

        assert_eq!(config.target_duration, Duration::from_secs(6));
        assert_eq!(config.profiles.len(), 2); // source + 720p
        assert!(config.is_live);
        assert_eq!(config.playlist_name, "stream");
    }

    #[test]
    fn test_hls_pipeline_creation() {
        let temp_dir = std::env::temp_dir().join("zvd_hls_test");
        let config = HlsConfig::new(&temp_dir, Duration::from_secs(6));
        let pipeline = HlsPipeline::new(config);
        assert!(pipeline.is_ok());
    }

    #[test]
    fn test_dash_config() {
        let config = DashConfig::new("/tmp/dash", Duration::from_secs(4))
            .with_profile(QualityProfile::new("1080p", 5_000_000, 1920, 1080, 30))
            .as_live();

        assert_eq!(config.segment_duration, Duration::from_secs(4));
        assert_eq!(config.profiles.len(), 1);
        assert!(config.is_live);
    }
}
