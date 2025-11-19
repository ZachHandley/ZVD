//! WebM/Matroska muxer implementation
//!
//! This module provides WebM/Matroska container writing using the mkv-element crate.
//!
//! ## Supported Codecs
//!
//! **Video:**
//! - VP8 (V_VP8 - royalty-free)
//! - VP9 (V_VP9 - royalty-free)
//! - AV1 (V_AV1 - royalty-free)
//!
//! **Audio:**
//! - Vorbis (A_VORBIS - royalty-free)
//! - Opus (A_OPUS - royalty-free)
//!
//! ## Features
//!
//! - ✅ Multiple video tracks
//! - ✅ Multiple audio tracks
//! - ✅ Simple clusters for frame storage
//! - All codecs are royalty-free and patent-free
//!
//! ## Notes
//!
//! - WebM is a subset of Matroska optimized for web use
//! - Only VP8, VP9, AV1, Vorbis, and Opus codecs are allowed in WebM
//! - For H.264/H.265/AAC, use MP4 containers instead

#[cfg(feature = "webm-support")]
use crate::error::{Error, Result};
#[cfg(feature = "webm-support")]
use crate::format::{Muxer, MuxerContext, Packet, Stream, StreamInfo};
#[cfg(feature = "webm-support")]
use crate::util::MediaType;
#[cfg(feature = "webm-support")]
use mkv_element::prelude::*;
#[cfg(feature = "webm-support")]
use mkv_element::io::blocking_impl::*;
#[cfg(feature = "webm-support")]
use std::collections::HashMap;
#[cfg(feature = "webm-support")]
use std::fs::File;
#[cfg(feature = "webm-support")]
use std::io::{BufWriter, Write};
#[cfg(feature = "webm-support")]
use std::path::Path;

#[cfg(feature = "webm-support")]
/// WebM/Matroska muxer
pub struct WebmMuxer {
    writer: Option<BufWriter<File>>,
    context: MuxerContext,
    track_numbers: HashMap<usize, u64>,
    path: Option<std::path::PathBuf>,
    segment_position: u64,
    cluster_timecode: u64,
    max_cluster_duration: u64, // In milliseconds
}

#[cfg(feature = "webm-support")]
impl WebmMuxer {
    /// Create a new WebM muxer
    pub fn new() -> Self {
        WebmMuxer {
            writer: None,
            context: MuxerContext::new("webm".to_string()),
            track_numbers: HashMap::new(),
            path: None,
            segment_position: 0,
            cluster_timecode: 0,
            max_cluster_duration: 2000, // 2 seconds per cluster
        }
    }

    /// Map codec name to WebM codec ID
    fn codec_to_codec_id(codec: &str) -> Result<String> {
        let codec_lower = codec.to_lowercase();
        match codec_lower.as_str() {
            "vp8" => Ok("V_VP8".to_string()),
            "vp9" | "vp09" => Ok("V_VP9".to_string()),
            "av1" | "av01" => Ok("V_AV1".to_string()),
            "vorbis" => Ok("A_VORBIS".to_string()),
            "opus" => Ok("A_OPUS".to_string()),
            _ => Err(Error::format(format!(
                "Unsupported codec for WebM: {}. Supported: VP8, VP9, AV1, Vorbis, Opus",
                codec
            ))),
        }
    }

    /// Create track entry from stream info
    fn create_track_entry(stream_info: &StreamInfo, track_number: u64) -> Result<TrackEntry> {
        let codec_id = Self::codec_to_codec_id(&stream_info.codec)?;

        let track_type = match stream_info.media_type {
            MediaType::Video => TrackType(1), // 1 = video
            MediaType::Audio => TrackType(2), // 2 = audio
            MediaType::Subtitle => TrackType(17), // 17 = subtitle
            _ => {
                return Err(Error::format(format!(
                    "Unsupported media type for WebM: {:?}",
                    stream_info.media_type
                )))
            }
        };

        let mut track_entry = TrackEntry {
            track_number: TrackNumber(track_number),
            track_uid: Some(TrackUid(track_number)), // Use track number as UID
            track_type: Some(track_type),
            codec_id: Some(CodecId(codec_id)),
            language: Some(Language("eng".to_string())),
            ..Default::default()
        };

        // Add video-specific info
        if let Some(ref video_info) = stream_info.video_info {
            track_entry.video = Some(Video {
                pixel_width: Some(PixelWidth(video_info.width as u64)),
                pixel_height: Some(PixelHeight(video_info.height as u64)),
                display_width: Some(DisplayWidth(video_info.width as u64)),
                display_height: Some(DisplayHeight(video_info.height as u64)),
                ..Default::default()
            });
        }

        // Add audio-specific info
        if let Some(ref audio_info) = stream_info.audio_info {
            track_entry.audio = Some(Audio {
                sampling_frequency: Some(SamplingFrequency(audio_info.sample_rate as f64)),
                channels: Some(Channels(audio_info.channels as u64)),
                bit_depth: Some(BitDepth(16)), // Default to 16-bit
                ..Default::default()
            });
        }

        Ok(track_entry)
    }
}

#[cfg(feature = "webm-support")]
impl Default for WebmMuxer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "webm-support")]
impl Muxer for WebmMuxer {
    fn create(&mut self, path: &Path) -> Result<()> {
        self.path = Some(path.to_path_buf());
        Ok(())
    }

    fn add_stream(&mut self, stream: Stream) -> Result<usize> {
        let index = self.context.add_stream(stream);
        Ok(index)
    }

    fn write_header(&mut self) -> Result<()> {
        let path = self
            .path
            .as_ref()
            .ok_or_else(|| Error::format("Path not set"))?;

        let file = File::create(path)
            .map_err(|e| Error::format(format!("Failed to create WebM file: {}", e)))?;

        let mut writer = BufWriter::new(file);

        // 1. Write EBML Header
        let ebml = Ebml {
            ebml_version: Some(EbmlVersion(1)),
            ebml_read_version: Some(EbmlReadVersion(1)),
            ebml_max_id_length: Some(EbmlMaxIdLength(4)),
            ebml_max_size_length: Some(EbmlMaxSizeLength(8)),
            doc_type: Some(DocType("webm".to_string())),
            doc_type_version: Some(DocTypeVersion(2)),
            doc_type_read_version: Some(DocTypeReadVersion(2)),
            ..Default::default()
        };

        ebml.write_to(&mut writer)
            .map_err(|e| Error::format(format!("Failed to write EBML header: {}", e)))?;

        // 2. Write Segment header (we'll come back to fill in size later)
        let segment_id = Header::new(0x18538067); // Segment ID
        segment_id.write_to(&mut writer)
            .map_err(|e| Error::format(format!("Failed to write Segment header: {}", e)))?;

        // Write unknown size (-1) for streaming
        let unknown_size: u64 = 0x00FFFFFFFFFFFFFF;
        writer.write_all(&unknown_size.to_be_bytes())
            .map_err(|e| Error::format(format!("Failed to write segment size: {}", e)))?;

        self.segment_position = 13; // Position after EBML header (typically ~30 bytes)

        // 3. Write Segment Info
        let info = Info {
            timestamp_scale: Some(TimestampScale(1_000_000)), // 1ms per tick
            muxing_app: Some(MuxingApp("ZVD - Rust Video Library".to_string())),
            writing_app: Some(WritingApp("ZVD WebM Muxer".to_string())),
            ..Default::default()
        };

        info.write_to(&mut writer)
            .map_err(|e| Error::format(format!("Failed to write Segment Info: {}", e)))?;

        // 4. Write Tracks
        let mut track_entries = Vec::new();
        let mut track_number: u64 = 1;

        for stream in self.context.streams() {
            let track_entry = Self::create_track_entry(&stream.info, track_number)?;
            self.track_numbers.insert(stream.info.index, track_number);
            track_entries.push(track_entry);
            track_number += 1;
        }

        let tracks = Tracks {
            track_entries,
            ..Default::default()
        };

        tracks
            .write_to(&mut writer)
            .map_err(|e| Error::format(format!("Failed to write Tracks: {}", e)))?;

        self.writer = Some(writer);
        self.context.set_header_written();
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| Error::format("WebM writer not initialized"))?;

        let track_number = self
            .track_numbers
            .get(&packet.stream_index)
            .ok_or_else(|| Error::format(format!("Track not found for stream {}", packet.stream_index)))?;

        // Convert PTS to timecode (assuming 1ms timescale)
        let timecode = packet.pts.value as u64;

        // Check if we need to start a new cluster
        if self.cluster_timecode == 0 || timecode - self.cluster_timecode > self.max_cluster_duration {
            // Write new cluster
            let cluster = Cluster {
                timestamp: Some(Timestamp(timecode)),
                ..Default::default()
            };

            // Write cluster header
            let cluster_id = Header::new(0x1F43B675); // Cluster ID
            cluster_id.write_to(writer)
                .map_err(|e| Error::format(format!("Failed to write Cluster header: {}", e)))?;

            // Write unknown size for streaming
            let unknown_size: u64 = 0x00FFFFFFFFFFFFFF;
            writer.write_all(&unknown_size.to_be_bytes())
                .map_err(|e| Error::format(format!("Failed to write cluster size: {}", e)))?;

            // Write timestamp
            let timestamp_id = Header::new(0xE7); // Timestamp ID
            timestamp_id.write_to(writer)
                .map_err(|e| Error::format(format!("Failed to write Timestamp header: {}", e)))?;

            // Encode timestamp value
            let ts_bytes = timecode.to_be_bytes();
            let ts_len = 8 - ts_bytes.iter().take_while(|&&b| b == 0).count();
            writer.write_all(&[ts_len as u8])
                .map_err(|e| Error::format(format!("Failed to write timestamp length: {}", e)))?;
            writer.write_all(&ts_bytes[8 - ts_len..])
                .map_err(|e| Error::format(format!("Failed to write timestamp value: {}", e)))?;

            self.cluster_timecode = timecode;
        }

        // Write SimpleBlock
        let simple_block_id = Header::new(0xA3); // SimpleBlock ID
        simple_block_id.write_to(writer)
            .map_err(|e| Error::format(format!("Failed to write SimpleBlock header: {}", e)))?;

        // Calculate SimpleBlock size: track number (1-4 bytes) + timecode (2 bytes) + flags (1 byte) + data
        let track_num_size = if *track_number < 128 { 1 } else { 2 };
        let block_size = track_num_size + 2 + 1 + packet.data.len();

        // Write size
        Self::write_element_size(writer, block_size as u64)?;

        // Write track number (EBML variable-size integer)
        Self::write_vint(writer, *track_number)?;

        // Write relative timecode (relative to cluster timecode)
        let relative_timecode = (timecode - self.cluster_timecode) as i16;
        writer.write_all(&relative_timecode.to_be_bytes())
            .map_err(|e| Error::format(format!("Failed to write timecode: {}", e)))?;

        // Write flags
        let mut flags = 0u8;
        if packet.flags.keyframe {
            flags |= 0x80; // Keyframe flag
        }
        writer.write_all(&[flags])
            .map_err(|e| Error::format(format!("Failed to write flags: {}", e)))?;

        // Write frame data
        writer.write_all(&packet.data)
            .map_err(|e| Error::format(format!("Failed to write frame data: {}", e)))?;

        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        if let Some(mut writer) = self.writer.take() {
            writer.flush()
                .map_err(|e| Error::format(format!("Failed to flush WebM file: {}", e)))?;
        }
        Ok(())
    }
}

#[cfg(feature = "webm-support")]
impl WebmMuxer {
    /// Write EBML variable-size integer
    fn write_vint<W: Write>(writer: &mut W, value: u64) -> Result<()> {
        if value < 127 {
            writer.write_all(&[0x80 | value as u8])
                .map_err(|e| Error::format(format!("Failed to write vint: {}", e)))?;
        } else if value < 16383 {
            writer.write_all(&[0x40 | (value >> 8) as u8, value as u8])
                .map_err(|e| Error::format(format!("Failed to write vint: {}", e)))?;
        } else {
            // For larger values, use 3-4 byte encoding
            let bytes = value.to_be_bytes();
            let len = 8 - bytes.iter().take_while(|&&b| b == 0).count();
            if len <= 3 {
                writer.write_all(&[0x20 | bytes[8 - len]])
                    .map_err(|e| Error::format(format!("Failed to write vint: {}", e)))?;
                writer.write_all(&bytes[8 - len + 1..])
                    .map_err(|e| Error::format(format!("Failed to write vint: {}", e)))?;
            } else {
                writer.write_all(&[0x10 | bytes[8 - len]])
                    .map_err(|e| Error::format(format!("Failed to write vint: {}", e)))?;
                writer.write_all(&bytes[8 - len + 1..])
                    .map_err(|e| Error::format(format!("Failed to write vint: {}", e)))?;
            }
        }
        Ok(())
    }

    /// Write element size
    fn write_element_size<W: Write>(writer: &mut W, size: u64) -> Result<()> {
        // Use variable-size encoding for element sizes
        if size < 127 {
            writer.write_all(&[0x80 | size as u8])
                .map_err(|e| Error::format(format!("Failed to write size: {}", e)))?;
        } else if size < 16383 {
            writer.write_all(&[0x40 | (size >> 8) as u8, size as u8])
                .map_err(|e| Error::format(format!("Failed to write size: {}", e)))?;
        } else if size < 2097151 {
            writer.write_all(&[0x20 | (size >> 16) as u8, (size >> 8) as u8, size as u8])
                .map_err(|e| Error::format(format!("Failed to write size: {}", e)))?;
        } else {
            writer.write_all(&[0x10 | (size >> 24) as u8, (size >> 16) as u8, (size >> 8) as u8, size as u8])
                .map_err(|e| Error::format(format!("Failed to write size: {}", e)))?;
        }
        Ok(())
    }
}

#[cfg(not(feature = "webm-support"))]
/// WebM muxer (disabled - enable with 'webm-support' feature)
pub struct WebmMuxer;

#[cfg(not(feature = "webm-support"))]
impl WebmMuxer {
    pub fn new() -> Self {
        panic!("WebM support not enabled. Build with --features webm-support");
    }
}

#[cfg(not(feature = "webm-support"))]
impl Default for WebmMuxer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[cfg(feature = "webm-support")]
mod tests {
    use super::*;

    #[test]
    fn test_webm_muxer_creation() {
        let muxer = WebmMuxer::new();
        assert_eq!(muxer.context.format_name(), "webm");
    }

    #[test]
    fn test_codec_mapping() {
        assert_eq!(WebmMuxer::codec_to_codec_id("vp8").unwrap(), "V_VP8");
        assert_eq!(WebmMuxer::codec_to_codec_id("vp9").unwrap(), "V_VP9");
        assert_eq!(WebmMuxer::codec_to_codec_id("av1").unwrap(), "V_AV1");
        assert_eq!(WebmMuxer::codec_to_codec_id("vorbis").unwrap(), "A_VORBIS");
        assert_eq!(WebmMuxer::codec_to_codec_id("opus").unwrap(), "A_OPUS");
    }

    #[test]
    fn test_unsupported_codec() {
        let result = WebmMuxer::codec_to_codec_id("h264");
        assert!(result.is_err());
    }
}
