//! MP4 muxer implementation
//!
//! This module provides MP4 container writing using the mp4 crate.
//! Note: H.264 codec is patent-encumbered. See CODEC_LICENSES.md for details.

#[cfg(feature = "mp4-support")]
use crate::error::{Error, Result};
#[cfg(feature = "mp4-support")]
use crate::format::{Muxer, MuxerContext, Packet, Stream, StreamInfo};
#[cfg(feature = "mp4-support")]
use crate::util::MediaType;
#[cfg(feature = "mp4-support")]
use mp4::{
    AacConfig, AudioObjectType, AvcConfig, ChannelConfig, MediaConfig, Mp4Config, Mp4Sample,
    Mp4Writer, SampleFreqIndex, TrackConfig, TrackType,
};
#[cfg(feature = "mp4-support")]
use std::collections::HashMap;
#[cfg(feature = "mp4-support")]
use std::fs::File;
#[cfg(feature = "mp4-support")]
use std::io::BufWriter;
#[cfg(feature = "mp4-support")]
use std::path::Path;

#[cfg(feature = "mp4-support")]
/// MP4 muxer
pub struct Mp4Muxer {
    writer: Option<Mp4Writer<BufWriter<File>>>,
    context: MuxerContext,
    track_ids: HashMap<usize, u32>,
    path: Option<std::path::PathBuf>,
}

#[cfg(feature = "mp4-support")]
impl Mp4Muxer {
    /// Create a new MP4 muxer
    pub fn new() -> Self {
        Mp4Muxer {
            writer: None,
            context: MuxerContext::new("mp4".to_string()),
            track_ids: HashMap::new(),
            path: None,
        }
    }

    /// Map sample rate to MP4 sample frequency index
    fn sample_rate_to_freq_index(sample_rate: u32) -> SampleFreqIndex {
        match sample_rate {
            96000 => SampleFreqIndex::Freq96000,
            88200 => SampleFreqIndex::Freq88200,
            64000 => SampleFreqIndex::Freq64000,
            48000 => SampleFreqIndex::Freq48000,
            44100 => SampleFreqIndex::Freq44100,
            32000 => SampleFreqIndex::Freq32000,
            24000 => SampleFreqIndex::Freq24000,
            22050 => SampleFreqIndex::Freq22050,
            16000 => SampleFreqIndex::Freq16000,
            12000 => SampleFreqIndex::Freq12000,
            11025 => SampleFreqIndex::Freq11025,
            8000 => SampleFreqIndex::Freq8000,
            7350 => SampleFreqIndex::Freq7350,
            _ => SampleFreqIndex::Freq48000, // Default to 48kHz
        }
    }

    /// Map channel count to MP4 channel config
    fn channels_to_config(channels: u16) -> ChannelConfig {
        match channels {
            1 => ChannelConfig::Mono,
            2 => ChannelConfig::Stereo,
            3 => ChannelConfig::Three,
            4 => ChannelConfig::Four,
            5 => ChannelConfig::Five,
            6 => ChannelConfig::FiveOne,
            7 => ChannelConfig::SevenOne,
            8 => ChannelConfig::SevenOne,
            _ => ChannelConfig::Stereo, // Default to stereo
        }
    }

    /// Create a track config from stream info
    fn create_track_config(stream_info: &StreamInfo) -> Result<TrackConfig> {
        let timescale = stream_info.time_base.den.abs() as u32;

        match stream_info.media_type {
            MediaType::Video => {
                let video_info = stream_info
                    .video_info
                    .as_ref()
                    .ok_or_else(|| Error::format("Video track missing video info"))?;

                // Create AVC config for H.264
                // Note: For proper H.264 MP4 muxing, SPS/PPS should be extracted from:
                // 1. The extradata/codec private data if available in stream_info
                // 2. The first IDR frame's NAL units (parse for NAL type 7=SPS, 8=PPS)
                // 3. H.264 Annex B format uses 0x00000001 start codes before NAL units
                let avc_config = AvcConfig {
                    width: video_info.width as u16,
                    height: video_info.height as u16,
                    seq_param_set: vec![], // Extract from stream extradata or first packet NAL units
                    pic_param_set: vec![],  // Extract from stream extradata or first packet NAL units
                };

                Ok(TrackConfig {
                    track_type: TrackType::Video,
                    timescale,
                    language: "und".to_string(),
                    media_conf: MediaConfig::AvcConfig(avc_config),
                })
            }
            MediaType::Audio => {
                let audio_info = stream_info
                    .audio_info
                    .as_ref()
                    .ok_or_else(|| Error::format("Audio track missing audio info"))?;

                // Create AAC config
                let aac_config = AacConfig {
                    bitrate: audio_info.bit_rate.unwrap_or(128000) as u32,
                    profile: AudioObjectType::AacLowComplexity,
                    freq_index: Self::sample_rate_to_freq_index(audio_info.sample_rate),
                    chan_conf: Self::channels_to_config(audio_info.channels),
                };

                Ok(TrackConfig {
                    track_type: TrackType::Audio,
                    timescale,
                    language: "und".to_string(),
                    media_conf: MediaConfig::AacConfig(aac_config),
                })
            }
            _ => Err(Error::format(format!(
                "Unsupported media type for MP4: {:?}",
                stream_info.media_type
            ))),
        }
    }
}

#[cfg(feature = "mp4-support")]
impl Default for Mp4Muxer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "mp4-support")]
impl Muxer for Mp4Muxer {
    fn create(&mut self, path: &Path) -> Result<()> {
        // Store path for later when we write header
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
            .map_err(|e| Error::format(format!("Failed to create MP4 file: {}", e)))?;

        let buf_writer = BufWriter::new(file);

        // Create MP4 config
        let config = Mp4Config {
            major_brand: str::parse("isom").unwrap(),
            minor_version: 512,
            compatible_brands: vec![
                str::parse("isom").unwrap(),
                str::parse("iso2").unwrap(),
                str::parse("avc1").unwrap(),
                str::parse("mp41").unwrap(),
            ],
            timescale: 1000,
        };

        // Initialize writer
        let mut writer = Mp4Writer::write_start(buf_writer, &config)
            .map_err(|e| Error::format(format!("Failed to initialize MP4 writer: {}", e)))?;

        // Add tracks for each stream in context
        // MP4 track IDs start from 1 and are sequential
        let mut track_id: u32 = 1;
        for stream in self.context.streams() {
            let track_config = Self::create_track_config(&stream.info)?;

            // mp4 crate's add_track returns Result<()>
            writer
                .add_track(&track_config)
                .map_err(|e| Error::format(format!("Failed to add track: {}", e)))?;

            self.track_ids.insert(stream.info.index, track_id);
            track_id += 1;
        }

        self.writer = Some(writer);
        self.context.set_header_written();
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| Error::format("MP4 writer not initialized"))?;

        let track_id = self
            .track_ids
            .get(&packet.stream_index)
            .ok_or_else(|| Error::format(format!("Track not found for stream {}", packet.stream_index)))?;

        // Convert our packet to MP4 sample
        let sample = Mp4Sample {
            start_time: packet.pts.value as u64,
            duration: packet.duration as u32,
            rendering_offset: 0,
            is_sync: packet.flags.keyframe,
            bytes: bytes::Bytes::copy_from_slice(packet.data.as_slice()),
        };

        writer
            .write_sample(*track_id, &sample)
            .map_err(|e| Error::format(format!("Failed to write sample: {}", e)))?;

        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        if let Some(mut writer) = self.writer.take() {
            writer
                .write_end()
                .map_err(|e| Error::format(format!("Failed to finalize MP4 file: {}", e)))?;
        }
        Ok(())
    }
}

#[cfg(not(feature = "mp4-support"))]
/// MP4 muxer (disabled - enable with 'mp4-support' feature)
pub struct Mp4Muxer;

#[cfg(not(feature = "mp4-support"))]
impl Mp4Muxer {
    pub fn new() -> Self {
        panic!("MP4 support not enabled. Build with --features mp4-support");
    }
}

#[cfg(not(feature = "mp4-support"))]
impl Default for Mp4Muxer {
    fn default() -> Self {
        Self::new()
    }
}
