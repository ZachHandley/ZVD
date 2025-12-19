//! WebM muxer (AV1/Vpx video + Opus audio)
//!
//! This is a minimal WebM writer implemented in pure Rust using the `mkv-element`
//! crate to emit EBML elements. It currently supports one video track (AV1/VP9/VP8)
//! and an optional Opus audio track.

use crate::error::{Error, Result};
use crate::format::{Muxer, MuxerContext, Packet, Stream};
use crate::util::{MediaType, Rational};
use mkv_element::io::blocking_impl::WriteTo;
use mkv_element::{prelude::*, ClusterBlock};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Seek, Write};
use std::path::Path;

/// WebM muxer
pub struct WebmMuxer {
    writer: Option<File>,
    context: MuxerContext,
    track_map: HashMap<usize, u64>, // stream_index -> track_number
    timecode_scale_ns: u64,
    cluster_timecode: u64,
    segment_start: u64,
    // pending blocks until we have a cluster header
    pending_blocks: Vec<ClusterBlock>,
    cues: Vec<CuePoint>,
    pending_keyframe: Option<(u64, u64)>, // (timecode, track_number)
    max_blocks_per_cluster: usize,
    max_cluster_span_tc: u64,
}

impl WebmMuxer {
    pub fn new() -> Self {
        WebmMuxer {
            writer: None,
            context: MuxerContext::new("webm".to_string()),
            track_map: HashMap::new(),
            timecode_scale_ns: 1_000_000, // 1ms
            cluster_timecode: 0,
            segment_start: 0,
            pending_blocks: Vec::new(),
            cues: Vec::new(),
            pending_keyframe: None,
            max_blocks_per_cluster: 32,
            max_cluster_span_tc: 1_000, // ~1 second with 1ms scale
        }
    }

    fn encode_simple_block(
        track_number: u64,
        rel_timecode: i16,
        keyframe: bool,
        data: &[u8],
    ) -> SimpleBlock {
        // Build SimpleBlock body manually:
        // TrackNumber (vint), Timecode (i16 BE), Flags, Payload
        let mut body = Vec::new();
        Self::write_vint(track_number, &mut body);
        body.extend_from_slice(&rel_timecode.to_be_bytes());
        let mut flags = 0u8;
        if keyframe {
            flags |= 0x80;
        }
        body.push(flags);
        body.extend_from_slice(data);
        SimpleBlock(body)
    }

    fn write_vint(value: u64, buf: &mut Vec<u8>) {
        // Minimal-length vint for TrackNumber
        for width in 1..=8 {
            let max = (1u64 << (7 * width)) - 1;
            if value <= max {
                let marker = 1u8 << (8 - width);
                let mut bytes = vec![0u8; width];
                let mut v = value;
                for b in bytes.iter_mut().rev() {
                    *b = (v & 0xFF) as u8;
                    v >>= 8;
                }
                bytes[0] |= marker;
                buf.extend_from_slice(&bytes);
                return;
            }
        }
        // Fallback to 8 bytes
        buf.extend_from_slice(&[0x01, 0, 0, 0, 0, 0, 0, 0]);
    }

    fn build_info(&self) -> Info {
        Info {
            timestamp_scale: TimestampScale(self.timecode_scale_ns),
            muxing_app: MuxingApp("zvd".to_string()),
            writing_app: WritingApp("zvd".to_string()),
            ..Default::default()
        }
    }

    fn build_tracks(&self) -> Tracks {
        let mut entries = Vec::new();
        for stream in self.context.streams() {
            let track_number = *self.track_map.get(&stream.info.index).unwrap_or(&1u64);
            let track_uid = track_number as u64;

            match stream.info.media_type {
                MediaType::Video => {
                    let vinfo = stream.info.video_info.as_ref().expect("Video info missing");
                    let codec_id = match stream.info.codec_id.as_str() {
                        "av1" => "V_AV1",
                        "vp9" => "V_VP9",
                        "vp8" => "V_VP8",
                        other => other,
                    };
                    let mut entry = TrackEntry::default();
                    entry.track_number = TrackNumber(track_number);
                    entry.track_uid = TrackUid(track_uid);
                    entry.track_type = TrackType(1);
                    entry.flag_enabled = FlagEnabled(1);
                    entry.flag_default = FlagDefault(1);
                    entry.flag_forced = FlagForced(0);
                    entry.flag_lacing = FlagLacing(0);
                    entry.max_block_addition_id = MaxBlockAdditionId(0);
                    entry.language = Language("und".to_string());
                    entry.codec_id = CodecId(codec_id.to_string());
                    entry.codec_delay = CodecDelay(0);
                    entry.seek_pre_roll = SeekPreRoll(0);
                    entry.codec_private =
                        stream.extradata.as_ref().map(|d| CodecPrivate(d.clone()));
                    entry.video = Some(Video {
                        pixel_width: PixelWidth(vinfo.width as u64),
                        pixel_height: PixelHeight(vinfo.height as u64),
                        ..Default::default()
                    });
                    entries.push(entry);
                }
                MediaType::Audio => {
                    if let Some(ainfo) = stream.info.audio_info.as_ref() {
                        let codec_id = match stream.info.codec_id.as_str() {
                            "opus" => "A_OPUS",
                            other => other,
                        };
                        let mut entry = TrackEntry::default();
                        entry.track_number = TrackNumber(track_number);
                        entry.track_uid = TrackUid(track_uid);
                        entry.track_type = TrackType(2);
                        entry.flag_enabled = FlagEnabled(1);
                        entry.flag_default = FlagDefault(1);
                        entry.flag_forced = FlagForced(0);
                        entry.flag_lacing = FlagLacing(0);
                        entry.max_block_addition_id = MaxBlockAdditionId(0);
                        entry.language = Language("und".to_string());
                        entry.codec_id = CodecId(codec_id.to_string());
                        entry.codec_private =
                            stream.extradata.as_ref().map(|d| CodecPrivate(d.clone()));
                        entry.codec_delay = CodecDelay(6_500_000); // 6.5ms
                        entry.seek_pre_roll = SeekPreRoll(80_000_000); // 80ms per spec
                        entry.audio = Some(Audio {
                            sampling_frequency: SamplingFrequency(ainfo.sample_rate as f64),
                            output_sampling_frequency: None,
                            channels: Channels(ainfo.channels as u64),
                            bit_depth: Some(BitDepth(ainfo.bits_per_sample as u64)),
                            emphasis: Emphasis(0),
                            crc32: None,
                            void: None,
                        });
                        entries.push(entry);
                    }
                }
                _ => {}
            }
        }

        Tracks {
            track_entry: entries,
            ..Default::default()
        }
    }

    fn write_ebml_header(&mut self) -> Result<()> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| Error::format("Writer not initialized"))?;

        let ebml = Ebml {
            ebml_max_id_length: EbmlMaxIdLength(4),
            ebml_max_size_length: EbmlMaxSizeLength(8),
            doc_type: Some(DocType("webm".to_string())),
            doc_type_version: Some(DocTypeVersion(4)),
            doc_type_read_version: Some(DocTypeReadVersion(2)),
            ..Default::default()
        };
        ebml.write_to(writer)
            .map_err(|e| Error::format(format!("Failed to write EBML header: {}", e)))
    }

    fn start_segment(&mut self) -> Result<()> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| Error::format("Writer not initialized"))?;

        // Write Segment header with unknown size (all 1s)
        writer
            .write_all(&[0x18, 0x53, 0x80, 0x67]) // Segment ID
            .map_err(|e| Error::format(format!("Failed to write segment ID: {}", e)))?;
        // Size all ones for unknown
        writer
            .write_all(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF])
            .map_err(|e| Error::format(format!("Failed to write segment size: {}", e)))?;

        self.segment_start = writer
            .stream_position()
            .map_err(|e| Error::format(format!("Failed to get stream position: {}", e)))?;
        Ok(())
    }

    fn write_cluster(&mut self, blocks: Vec<ClusterBlock>, timecode: u64) -> Result<()> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| Error::format("Writer not initialized"))?;

        let cluster_pos = writer
            .stream_position()
            .map_err(|e| Error::format(format!("Failed to get stream position: {}", e)))?;

        let cluster = Cluster {
            timestamp: Timestamp(timecode),
            blocks,
            ..Default::default()
        };

        cluster
            .write_to(writer)
            .map_err(|e| Error::format(format!("Failed to write cluster: {}", e)))?;

        if let Some((timecode_abs, track_number)) = self.pending_keyframe.take() {
            let cue = CuePoint {
                cue_time: CueTime(timecode_abs),
                cue_track_positions: vec![CueTrackPositions {
                    cue_track: CueTrack(track_number),
                    cue_cluster_position: CueClusterPosition(
                        cluster_pos.saturating_sub(self.segment_start),
                    ),
                    cue_codec_state: CueCodecState(0),
                    ..Default::default()
                }],
                ..Default::default()
            };
            self.cues.push(cue);
        }

        Ok(())
    }

    fn timecode_units(&self, pts: i64, time_base: Rational) -> u64 {
        let num = time_base.num as i128;
        let den = time_base.den as i128;
        if den == 0 {
            return 0;
        }
        let pts_i = pts as i128;
        let nanos = pts_i * num * 1_000_000_000 / den;
        (nanos / self.timecode_scale_ns as i128) as u64
    }
}

impl Default for WebmMuxer {
    fn default() -> Self {
        Self::new()
    }
}

impl Muxer for WebmMuxer {
    fn create(&mut self, path: &Path) -> Result<()> {
        let file = File::create(path)
            .map_err(|e| Error::format(format!("Failed to create WebM file: {}", e)))?;
        self.writer = Some(file);
        Ok(())
    }

    fn add_stream(&mut self, stream: Stream) -> Result<usize> {
        let idx = self.context.add_stream(stream.clone());
        let track_number = (self.track_map.len() as u64) + 1;
        self.track_map.insert(stream.info.index, track_number);
        Ok(idx)
    }

    fn write_header(&mut self) -> Result<()> {
        self.write_ebml_header()?;
        self.start_segment()?;

        let info = self.build_info();
        let tracks = self.build_tracks();

        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| Error::format("Writer not initialized"))?;

        info.write_to(writer)
            .map_err(|e| Error::format(format!("Failed to write Info: {}", e)))?;
        tracks
            .write_to(writer)
            .map_err(|e| Error::format(format!("Failed to write Tracks: {}", e)))?;

        self.context.set_header_written();
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.context.streams().is_empty() {
            return Err(Error::format("No streams added to WebM muxer"));
        }

        let track_number = *self
            .track_map
            .get(&packet.stream_index)
            .ok_or_else(|| Error::format(format!("Unknown stream {}", packet.stream_index)))?;

        let (time_base, media_type) = {
            let stream = self
                .context
                .streams()
                .iter()
                .find(|s| s.info.index == packet.stream_index)
                .ok_or_else(|| Error::format("Stream not found"))?;
            (stream.info.time_base, stream.info.media_type)
        };
        let pkt_timecode = self.timecode_units(packet.pts.value, time_base);

        let should_rotate = if self.pending_blocks.is_empty() {
            self.cluster_timecode = pkt_timecode;
            false
        } else {
            let rel = pkt_timecode.saturating_sub(self.cluster_timecode);
            rel > self.max_cluster_span_tc
                || rel > i16::MAX as u64
                || self.pending_blocks.len() >= self.max_blocks_per_cluster
        };

        if should_rotate {
            let blocks = std::mem::take(&mut self.pending_blocks);
            self.write_cluster(blocks, self.cluster_timecode)?;
            self.cluster_timecode = pkt_timecode;
        }

        let rel = pkt_timecode.saturating_sub(self.cluster_timecode);
        let rel_i16 = rel as i16;

        let sb = Self::encode_simple_block(
            track_number,
            rel_i16,
            packet.flags.keyframe,
            packet.data.as_slice(),
        );
        self.pending_blocks.push(sb.into());

        if packet.flags.keyframe && matches!(media_type, MediaType::Video) {
            if self.pending_keyframe.is_none() {
                self.pending_keyframe = Some((pkt_timecode, track_number));
            }
        }

        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        if !self.pending_blocks.is_empty() {
            let blocks = std::mem::take(&mut self.pending_blocks);
            self.write_cluster(blocks, self.cluster_timecode)?;
        }

        // Write cues for seekability (after clusters)
        if !self.cues.is_empty() {
            let writer = self
                .writer
                .as_mut()
                .ok_or_else(|| Error::format("Writer not initialized"))?;

            let cues = Cues {
                cue_point: std::mem::take(&mut self.cues),
                ..Default::default()
            };

            cues.write_to(writer)
                .map_err(|e| Error::format(format!("Failed to write cues: {}", e)))?;
        }
        Ok(())
    }
}
