//! MP4 demuxer implementation
//!
//! This module provides MP4 container support using the mp4 crate.
//! Note: H.264 codec is patent-encumbered. See CODEC_LICENSES.md for details.

#[cfg(feature = "mp4-support")]
use crate::error::{Error, Result};
#[cfg(feature = "mp4-support")]
use crate::format::packet::PacketFlags;
#[cfg(feature = "mp4-support")]
use crate::format::{AudioInfo, Demuxer, DemuxerContext, Packet, Stream, StreamInfo, VideoInfo};
#[cfg(feature = "mp4-support")]
use crate::util::{Buffer, MediaType, Rational, Timestamp};
#[cfg(feature = "mp4-support")]
use mp4::{ChannelConfig, Mp4Reader, Mp4Track, SampleFreqIndex, TrackType};
#[cfg(feature = "mp4-support")]
use std::collections::HashMap;
#[cfg(feature = "mp4-support")]
use std::fs::File;
#[cfg(feature = "mp4-support")]
use std::path::Path;

#[cfg(feature = "mp4-support")]
/// MP4 demuxer
pub struct Mp4Demuxer {
    reader: Option<Mp4Reader<File>>,
    context: DemuxerContext,
    /// Track current sample index for each track
    track_samples: HashMap<u32, u32>,
}

#[cfg(feature = "mp4-support")]
impl Mp4Demuxer {
    /// Create a new MP4 demuxer
    pub fn new() -> Self {
        Mp4Demuxer {
            reader: None,
            context: DemuxerContext::new("mp4".to_string()),
            track_samples: HashMap::new(),
        }
    }

    /// Map MP4 codec string to our codec identifier
    fn map_codec(mp4_codec: &str) -> String {
        match mp4_codec {
            "avc1" | "avc3" => "h264".to_string(),
            "hev1" | "hvc1" => "h265".to_string(),
            "av01" => "av1".to_string(),
            "vp09" => "vp9".to_string(),
            "mp4a" => "aac".to_string(),
            "Opus" | "opus" => "opus".to_string(),
            _ => mp4_codec.to_lowercase(),
        }
    }

    /// Map sample frequency index to Hz
    fn freq_index_to_hz(freq_index: SampleFreqIndex) -> u32 {
        match freq_index {
            SampleFreqIndex::Freq96000 => 96000,
            SampleFreqIndex::Freq88200 => 88200,
            SampleFreqIndex::Freq64000 => 64000,
            SampleFreqIndex::Freq48000 => 48000,
            SampleFreqIndex::Freq44100 => 44100,
            SampleFreqIndex::Freq32000 => 32000,
            SampleFreqIndex::Freq24000 => 24000,
            SampleFreqIndex::Freq22050 => 22050,
            SampleFreqIndex::Freq16000 => 16000,
            SampleFreqIndex::Freq12000 => 12000,
            SampleFreqIndex::Freq11025 => 11025,
            SampleFreqIndex::Freq8000 => 8000,
            SampleFreqIndex::Freq7350 => 7350,
            _ => 48000, // Default
        }
    }

    /// Map channel config to channel count
    fn channel_config_to_count(chan_conf: ChannelConfig) -> u16 {
        match chan_conf {
            ChannelConfig::Mono => 1,
            ChannelConfig::Stereo => 2,
            ChannelConfig::Three => 3,
            ChannelConfig::Four => 4,
            ChannelConfig::Five => 5,
            ChannelConfig::FiveOne => 6,
            ChannelConfig::SevenOne => 8,
            _ => 2, // Default to stereo
        }
    }

    /// Extract stream information from MP4 track
    fn extract_stream_info(track: &Mp4Track, track_id: u32) -> Result<StreamInfo> {
        let codec_str = track.media_type()?.to_string();
        let codec_id = Self::map_codec(&codec_str);

        let media_type = match track.track_type()? {
            TrackType::Video => MediaType::Video,
            TrackType::Audio => MediaType::Audio,
            _ => MediaType::Unknown,
        };

        let mut stream_info = StreamInfo::new(track_id as usize, media_type, codec_id);

        // Set time base from track timescale
        let timescale = track.timescale();
        stream_info.time_base = Rational::new(1, timescale as i64);

        // Extract video-specific information
        if track.track_type()? == TrackType::Video {
            stream_info.video_info = Some(VideoInfo {
                width: track.width() as u32,
                height: track.height() as u32,
                frame_rate: {
                    // Use frame_rate() method
                    let fps = track.frame_rate();
                    Rational::new((fps * 1000.0) as i64, 1000)
                },
                aspect_ratio: Rational::new(1, 1),
                pix_fmt: "yuv420p".to_string(), // Most common for H.264
                bits_per_sample: 8,
            });
        }

        // Extract audio-specific information
        if track.track_type()? == TrackType::Audio {
            let sample_rate = if let Ok(freq_index) = track.sample_freq_index() {
                Self::freq_index_to_hz(freq_index)
            } else {
                48000 // Default
            };

            let channels = if let Ok(chan_conf) = track.channel_config() {
                Self::channel_config_to_count(chan_conf)
            } else {
                2 // Default to stereo
            };

            stream_info.audio_info = Some(AudioInfo {
                sample_rate,
                channels,
                sample_fmt: "s16".to_string(), // Most common for AAC
                bits_per_sample: 16,
                bit_rate: Some(track.bitrate() as u64),
            });
        }

        Ok(stream_info)
    }
}

#[cfg(feature = "mp4-support")]
impl Default for Mp4Demuxer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "mp4-support")]
impl Demuxer for Mp4Demuxer {
    fn open(&mut self, path: &Path) -> Result<()> {
        // Open the file
        let file = File::open(path)
            .map_err(|e| Error::format(format!("Failed to open MP4 file: {}", e)))?;

        let size = file
            .metadata()
            .map_err(|e| Error::format(format!("Failed to get file metadata: {}", e)))?
            .len();

        // Create MP4 reader
        let reader = Mp4Reader::read_header(file, size)
            .map_err(|e| Error::format(format!("Failed to read MP4 header: {}", e)))?;

        // Extract stream information for each track
        for (track_id, track) in reader.tracks() {
            match Self::extract_stream_info(track, *track_id) {
                Ok(stream_info) => {
                    let stream = Stream {
                        info: stream_info,
                        extradata: None,
                    };
                    self.context.add_stream(stream);
                    self.track_samples.insert(*track_id, 1); // MP4 samples are 1-indexed
                }
                Err(e) => {
                    eprintln!("Warning: Failed to extract track {} info: {}", track_id, e);
                }
            }
        }

        self.reader = Some(reader);
        Ok(())
    }

    fn streams(&self) -> &[Stream] {
        self.context.streams()
    }

    fn read_packet(&mut self) -> Result<Packet> {
        let reader = self
            .reader
            .as_mut()
            .ok_or_else(|| Error::format("MP4 reader not initialized"))?;

        // Find the next track with available samples
        let mut next_track_id = None;
        let mut min_sample_id = u32::MAX;

        for (track_id, current_sample) in &self.track_samples {
            let track = reader
                .tracks()
                .get(track_id)
                .ok_or_else(|| Error::format(format!("Track {} not found", track_id)))?;

            if *current_sample <= track.sample_count() && *current_sample < min_sample_id {
                min_sample_id = *current_sample;
                next_track_id = Some(*track_id);
            }
        }

        let track_id = next_track_id.ok_or_else(|| Error::EndOfStream)?;

        let sample_id = self.track_samples[&track_id];

        // Read the sample
        let sample = reader
            .read_sample(track_id, sample_id)
            .map_err(|e| Error::format(format!("Failed to read sample: {}", e)))?
            .ok_or_else(|| Error::format("Sample not found"))?;

        // Update sample index for this track
        if let Some(count) = self.track_samples.get_mut(&track_id) {
            *count += 1;
        }

        // Find stream info for this track
        let stream = self
            .context
            .streams()
            .iter()
            .find(|s| s.info.index == track_id as usize)
            .ok_or_else(|| Error::format(format!("Stream {} not found", track_id)))?;

        let pts = Timestamp::new(sample.start_time as i64);
        let dts = pts; // MP4 doesn't separate DTS/PTS in samples

        let mut flags = PacketFlags::default();
        flags.keyframe = sample.is_sync;

        let packet = Packet {
            stream_index: track_id as usize,
            codec_type: stream.info.media_type,
            data: Buffer::from_vec(sample.bytes.to_vec()),
            pts,
            dts,
            duration: sample.duration as i64,
            flags,
            position: -1,
        };

        Ok(packet)
    }

    fn seek(&mut self, _stream_index: usize, _timestamp: i64) -> Result<()> {
        // Seeking in MP4 requires:
        // 1. Converting timestamp to timescale units for the specific track
        // 2. Binary search through the sample table to find the closest keyframe
        // 3. Updating track_samples iterators to start from that position
        // 4. Handling stss (sync sample table) for keyframe locations
        // This would leverage the mp4 crate's sample iteration capabilities
        Err(Error::format("Seeking not yet implemented for MP4"))
    }

    fn close(&mut self) -> Result<()> {
        self.reader = None;
        self.track_samples.clear();
        Ok(())
    }
}

#[cfg(not(feature = "mp4-support"))]
/// MP4 demuxer (disabled - enable with 'mp4-support' feature)
pub struct Mp4Demuxer;

#[cfg(not(feature = "mp4-support"))]
impl Mp4Demuxer {
    pub fn new() -> Self {
        panic!("MP4 support not enabled. Build with --features mp4-support");
    }
}

#[cfg(not(feature = "mp4-support"))]
impl Default for Mp4Demuxer {
    fn default() -> Self {
        Self::new()
    }
}
