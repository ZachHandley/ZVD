//! Video segmentation for streaming protocols
//!
//! This module provides segmentation capabilities for creating streaming-ready
//! video segments (HLS .ts files, DASH .m4s files, etc).
//!
//! ## Features
//!
//! - Keyframe-aligned segmentation
//! - MPEG-TS output format
//! - Configurable target duration
//! - Automatic segment boundary detection
//! - Metadata tracking (duration, size, timestamps)

use crate::error::{Error, Result};
use crate::format::{Packet, Stream, StreamInfo};
use crate::format::mpegts::MpegtsMuxer;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Segment configuration
#[derive(Debug, Clone)]
pub struct SegmentConfig {
    /// Target segment duration
    pub target_duration: Duration,
    /// Output directory for segments
    pub output_dir: PathBuf,
    /// Segment filename pattern (e.g., "segment_%05d.ts")
    pub filename_pattern: String,
    /// Start segment index
    pub start_index: u32,
    /// Maximum segment duration (enforced even without keyframe)
    pub max_duration: Duration,
}

impl SegmentConfig {
    /// Create a new segment configuration
    pub fn new<P: AsRef<Path>>(output_dir: P, target_duration: Duration) -> Self {
        SegmentConfig {
            target_duration,
            output_dir: output_dir.as_ref().to_path_buf(),
            filename_pattern: "segment_%05d.ts".to_string(),
            start_index: 0,
            max_duration: target_duration * 2, // Allow up to 2x target before forcing split
        }
    }

    /// Get the filename for a segment index
    pub fn segment_filename(&self, index: u32) -> String {
        // Simple pattern replacement (for production, use a proper template library)
        self.filename_pattern.replace("%05d", &format!("{:05}", index))
    }

    /// Get the full path for a segment
    pub fn segment_path(&self, index: u32) -> PathBuf {
        self.output_dir.join(self.segment_filename(index))
    }
}

/// Information about a completed segment
#[derive(Debug, Clone)]
pub struct SegmentInfo {
    /// Segment index
    pub index: u32,
    /// Segment filename
    pub filename: String,
    /// Actual duration
    pub duration: Duration,
    /// File size in bytes
    pub size: u64,
    /// Starting timestamp
    pub start_time: Duration,
}

/// Video segmenter for streaming protocols
pub struct Segmenter {
    config: SegmentConfig,
    current_segment: Option<CurrentSegment>,
    segments: Vec<SegmentInfo>,
    current_index: u32,
    streams: Vec<Stream>,
}

struct CurrentSegment {
    index: u32,
    muxer: MpegtsMuxer<BufWriter<File>>,
    start_time: Duration,
    last_time: Duration,
    packet_count: usize,
    path: PathBuf,
}

impl Segmenter {
    /// Create a new segmenter
    pub fn new(config: SegmentConfig) -> Result<Self> {
        // Create output directory if it doesn't exist
        std::fs::create_dir_all(&config.output_dir)
            .map_err(|e| Error::Io(e))?;

        Ok(Segmenter {
            current_index: config.start_index,
            config,
            current_segment: None,
            segments: Vec::new(),
            streams: Vec::new(),
        })
    }

    /// Add a stream
    pub fn add_stream(&mut self, stream: Stream) -> Result<()> {
        self.streams.push(stream);
        Ok(())
    }

    /// Start a new segment
    fn start_segment(&mut self, start_time: Duration) -> Result<()> {
        // Close current segment if any
        if self.current_segment.is_some() {
            self.close_segment()?;
        }

        let path = self.config.segment_path(self.current_index);
        let file = File::create(&path)
            .map_err(|e| Error::Io(e))?;
        let writer = BufWriter::new(file);

        // Create MPEG-TS muxer for this segment
        let mut muxer = MpegtsMuxer::new(writer);

        // Add all streams to the muxer
        for stream in &self.streams {
            muxer.add_stream(stream.clone())?;
        }

        // Write MPEG-TS header (PAT and PMT)
        muxer.write_header()?;

        self.current_segment = Some(CurrentSegment {
            index: self.current_index,
            muxer,
            start_time,
            last_time: start_time,
            packet_count: 0,
            path,
        });

        Ok(())
    }

    /// Close the current segment
    fn close_segment(&mut self) -> Result<()> {
        if let Some(mut segment) = self.current_segment.take() {
            // Write trailer and flush the muxer
            segment.muxer.write_trailer()?;
            drop(segment.muxer);

            // Get file size
            let metadata = std::fs::metadata(&segment.path)
                .map_err(|e| Error::Io(e))?;

            let duration = segment.last_time - segment.start_time;

            let info = SegmentInfo {
                index: segment.index,
                filename: self.config.segment_filename(segment.index),
                duration,
                size: metadata.len(),
                start_time: segment.start_time,
            };

            self.segments.push(info);
            self.current_index += 1;
        }

        Ok(())
    }

    /// Write a packet to the segmenter
    pub fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        let packet_time = Duration::from_millis(packet.pts.value as u64);

        // Start first segment if needed
        if self.current_segment.is_none() {
            self.start_segment(packet_time)?;
        }

        let segment = self.current_segment.as_mut()
            .ok_or_else(|| Error::invalid_state("No current segment"))?;

        let segment_duration = packet_time.saturating_sub(segment.start_time);

        // Check if we should start a new segment
        let should_split = (packet.flags.keyframe && segment_duration >= self.config.target_duration)
            || segment_duration >= self.config.max_duration;

        if should_split && segment.packet_count > 0 {
            // Start new segment at this keyframe
            self.start_segment(packet_time)?;
            let segment = self.current_segment.as_mut().unwrap();
            segment.last_time = packet_time;
            segment.packet_count += 1;
            // Write packet to the new segment's muxer
            segment.muxer.write_packet(packet)?;
        } else {
            // Write to current segment
            segment.last_time = packet_time;
            segment.packet_count += 1;
            // Write packet to muxer
            segment.muxer.write_packet(packet)?;
        }

        Ok(())
    }

    /// Finalize segmentation and close all segments
    pub fn finalize(&mut self) -> Result<()> {
        self.close_segment()?;
        Ok(())
    }

    /// Get all completed segments
    pub fn segments(&self) -> &[SegmentInfo] {
        &self.segments
    }

    /// Get segment count
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::Rational;
    use crate::format::PacketFlags;

    #[test]
    fn test_segment_config() {
        let config = SegmentConfig::new("/tmp/segments", Duration::from_secs(6));
        assert_eq!(config.target_duration, Duration::from_secs(6));
        assert_eq!(config.segment_filename(0), "segment_00000.ts");
        assert_eq!(config.segment_filename(123), "segment_00123.ts");
    }

    #[test]
    fn test_segmenter_creation() {
        let temp_dir = std::env::temp_dir().join("zvd_test_segmenter");
        let config = SegmentConfig::new(&temp_dir, Duration::from_secs(6));
        let segmenter = Segmenter::new(config);
        assert!(segmenter.is_ok());
    }

    #[test]
    fn test_segment_splitting() {
        let temp_dir = std::env::temp_dir().join("zvd_test_split");
        let config = SegmentConfig::new(&temp_dir, Duration::from_secs(6));
        let mut segmenter = Segmenter::new(config).unwrap();

        // Simulate packets over time
        for i in 0..20 {
            let is_keyframe = i % 5 == 0; // Keyframe every 5 packets
            let pts_ms = i * 1000; // 1 second per packet

            let packet = Packet {
                stream_index: 0,
                data: vec![0; 100],
                pts: Rational::new(pts_ms, 1),
                dts: None,
                duration: 1000,
                flags: PacketFlags {
                    keyframe: is_keyframe,
                    ..Default::default()
                },
            };

            segmenter.write_packet(&packet).unwrap();
        }

        segmenter.finalize().unwrap();

        // Should have created multiple segments (20 seconds of video with 6s segments = ~3 segments)
        assert!(segmenter.segment_count() >= 2);
    }
}
