//! WebM/Matroska muxer implementation
//!
//! This module provides WebM/Matroska container writing using webm-iterable.
//! Supports VP8, VP9, AV1 video and Vorbis, Opus audio (all royalty-free).

#[cfg(feature = "webm-support")]
use crate::error::{Error, Result};
#[cfg(feature = "webm-support")]
use crate::format::{Muxer, MuxerContext, Packet, Stream, StreamInfo};
#[cfg(feature = "webm-support")]
use crate::util::MediaType;
#[cfg(feature = "webm-support")]
use ebml_iterable::WriteOptions;
#[cfg(feature = "webm-support")]
use std::collections::HashMap;
#[cfg(feature = "webm-support")]
use std::fs::File;
#[cfg(feature = "webm-support")]
use std::io::BufWriter;
#[cfg(feature = "webm-support")]
use std::path::Path;
#[cfg(feature = "webm-support")]
use webm_iterable::matroska_spec::{MatroskaSpec, Master, Block, SimpleBlock, EbmlSpecification};
#[cfg(feature = "webm-support")]
use webm_iterable::{WebmWriter, WebmElement};

#[cfg(feature = "webm-support")]
/// WebM/Matroska muxer
pub struct WebmMuxer {
    writer: Option<WebmWriter<BufWriter<File>>>,
    context: MuxerContext,
    track_numbers: HashMap<usize, u64>,
    path: Option<std::path::PathBuf>,
    next_track_number: u64,
    /// Cluster timestamp in milliseconds
    cluster_timestamp: u64,
    /// Whether we've written the segment info
    segment_info_written: bool,
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
            next_track_number: 1,
            cluster_timestamp: 0,
            segment_info_written: false,
        }
    }

    /// Map codec to WebM codec ID
    fn map_codec_id(codec: &str) -> Result<String> {
        match codec.to_lowercase().as_str() {
            "vp8" => Ok("V_VP8".to_string()),
            "vp9" => Ok("V_VP9".to_string()),
            "av1" => Ok("V_AV1".to_string()),
            "vorbis" => Ok("A_VORBIS".to_string()),
            "opus" => Ok("A_OPUS".to_string()),
            _ => Err(Error::format(format!(
                "Unsupported codec for WebM: {}",
                codec
            ))),
        }
    }

    /// Get timescale for stream (usually 1000 for WebM - milliseconds)
    fn get_timescale(_stream_info: &StreamInfo) -> u64 {
        1000000 // WebM uses nanoseconds for timecode scale
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

        let buf_writer = BufWriter::new(file);

        // Create WebM writer with default options
        let write_opts = WriteOptions::default();
        let mut writer = WebmWriter::new(buf_writer, write_opts);

        // Write EBML header
        let ebml_header = WebmElement::EbmlHead {
            ebml_version: 1,
            ebml_read_version: 1,
            ebml_max_id_length: 4,
            ebml_max_size_length: 8,
            doc_type: "webm".to_string(),
            doc_type_version: 2,
            doc_type_read_version: 2,
        };

        writer
            .write(&ebml_header)
            .map_err(|e| Error::format(format!("Failed to write EBML header: {}", e)))?;

        // Start Segment
        writer
            .write(&WebmElement::StartSegment)
            .map_err(|e| Error::format(format!("Failed to start segment: {}", e)))?;

        // Write Segment Info
        let info = WebmElement::Info {
            timecode_scale: 1000000, // 1ms
            muxing_app: "ZVD - Rust Multimedia Library".to_string(),
            writing_app: "ZVD WebM Muxer".to_string(),
            duration: None, // Will be updated in trailer if possible
        };

        writer
            .write(&info)
            .map_err(|e| Error::format(format!("Failed to write segment info: {}", e)))?;

        self.segment_info_written = true;

        // Write Tracks
        writer
            .write(&WebmElement::StartTracks)
            .map_err(|e| Error::format(format!("Failed to start tracks: {}", e)))?;

        for stream in self.context.streams() {
            let track_number = self.next_track_number;
            self.track_numbers.insert(stream.info.index, track_number);
            self.next_track_number += 1;

            let codec_id = Self::map_codec_id(&stream.info.codec)?;

            let track_type = match stream.info.media_type {
                MediaType::Video => 1u64, // Video track type
                MediaType::Audio => 2u64, // Audio track type
                _ => {
                    return Err(Error::format(format!(
                        "Unsupported media type for WebM: {:?}",
                        stream.info.media_type
                    )))
                }
            };

            let track = if stream.info.media_type == MediaType::Video {
                let video_info = stream
                    .info
                    .video_info
                    .as_ref()
                    .ok_or_else(|| Error::format("Video track missing video info"))?;

                WebmElement::Track {
                    track_number,
                    track_uid: track_number,
                    track_type,
                    codec_id,
                    codec_private: stream.info.codec_private.clone(),
                    video: Some(WebmElement::Video {
                        pixel_width: video_info.width as u64,
                        pixel_height: video_info.height as u64,
                        display_width: Some(video_info.width as u64),
                        display_height: Some(video_info.height as u64),
                    }),
                    audio: None,
                }
            } else {
                let audio_info = stream
                    .info
                    .audio_info
                    .as_ref()
                    .ok_or_else(|| Error::format("Audio track missing audio info"))?;

                WebmElement::Track {
                    track_number,
                    track_uid: track_number,
                    track_type,
                    codec_id,
                    codec_private: stream.info.codec_private.clone(),
                    video: None,
                    audio: Some(WebmElement::Audio {
                        sampling_frequency: audio_info.sample_rate as f64,
                        channels: audio_info.channels as u64,
                        bit_depth: audio_info.bits_per_sample.map(|b| b as u64),
                    }),
                }
            };

            writer
                .write(&track)
                .map_err(|e| Error::format(format!("Failed to write track: {}", e)))?;
        }

        writer
            .write(&WebmElement::EndTracks)
            .map_err(|e| Error::format(format!("Failed to end tracks: {}", e)))?;

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

        // Start cluster if needed (at beginning or on keyframe after some data)
        if !self.segment_info_written || packet.flags.keyframe {
            // Start a new cluster
            writer
                .write(&WebmElement::StartCluster {
                    timecode: packet.pts.value as u64,
                })
                .map_err(|e| Error::format(format!("Failed to start cluster: {}", e)))?;

            self.cluster_timestamp = packet.pts.value as u64;
            self.segment_info_written = true;
        }

        // Calculate relative timecode within cluster
        let relative_timecode = (packet.pts.value as u64)
            .saturating_sub(self.cluster_timestamp) as i16;

        // Write SimpleBlock
        let simple_block = WebmElement::SimpleBlock {
            track_number: *track_number,
            timecode: relative_timecode,
            keyframe: packet.flags.keyframe,
            invisible: false,
            lacing: None,
            data: packet.data.as_slice().to_vec(),
        };

        writer
            .write(&simple_block)
            .map_err(|e| Error::format(format!("Failed to write block: {}", e)))?;

        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        if let Some(mut writer) = self.writer.take() {
            // End the current cluster
            writer
                .write(&WebmElement::EndCluster)
                .map_err(|e| Error::format(format!("Failed to end cluster: {}", e)))?;

            // End the segment
            writer
                .write(&WebmElement::EndSegment)
                .map_err(|e| Error::format(format!("Failed to end segment: {}", e)))?;

            // Finalize
            writer
                .finalize()
                .map_err(|e| Error::format(format!("Failed to finalize WebM file: {}", e)))?;
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
