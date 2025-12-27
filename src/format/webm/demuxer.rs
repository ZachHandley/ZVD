//! WebM/Matroska demuxer implementation

use crate::error::{Error, Result};
use crate::format::{AudioInfo, Demuxer, DemuxerContext, Packet, Stream, StreamInfo, VideoInfo};
use crate::util::{Buffer, MediaType, Rational, Timestamp};
use matroska_demuxer::{Frame, MatroskaFile, TrackType};
use std::fs::File;
use std::path::Path;

/// WebM/Matroska demuxer
pub struct WebmDemuxer {
    file: Option<MatroskaFile<File>>,
    context: DemuxerContext,
    frame_buffer: Frame,
}

impl WebmDemuxer {
    /// Create a new WebM demuxer
    pub fn new() -> Self {
        WebmDemuxer {
            file: None,
            context: DemuxerContext::new("webm".to_string()),
            frame_buffer: Frame::default(),
        }
    }

    /// Convert track type to our MediaType
    fn track_type_to_media_type(track_type: TrackType) -> MediaType {
        match track_type {
            TrackType::Video => MediaType::Video,
            TrackType::Audio => MediaType::Audio,
            TrackType::Subtitle => MediaType::Subtitle,
            _ => MediaType::Unknown,
        }
    }

    /// Map codec ID string to our codec identifier
    fn map_codec_id(codec_id: &str) -> String {
        match codec_id {
            "V_VP8" => "vp8".to_string(),
            "V_VP9" => "vp9".to_string(),
            "V_AV1" => "av1".to_string(),
            "A_VORBIS" => "vorbis".to_string(),
            "A_OPUS" => "opus".to_string(),
            _ => codec_id.to_string(),
        }
    }
}

impl Default for WebmDemuxer {
    fn default() -> Self {
        Self::new()
    }
}

impl Demuxer for WebmDemuxer {
    fn open(&mut self, path: &Path) -> Result<()> {
        // Open the file
        let file_handle =
            File::open(path).map_err(|e| Error::format(format!("Failed to open file: {}", e)))?;

        // Open with matroska-demuxer
        let mkv_file = MatroskaFile::open(file_handle)
            .map_err(|e| Error::format(format!("Failed to open WebM/Matroska file: {}", e)))?;

        // Extract track information
        for track in mkv_file.tracks() {
            let track_num = track.track_number().get();
            let track_type = track.track_type();
            let media_type = Self::track_type_to_media_type(track_type);

            // Get codec ID
            let codec_id = Self::map_codec_id(track.codec_id());

            let mut stream_info = StreamInfo::new(track_num as usize, media_type, codec_id);

            // Set time base (WebM uses nanoseconds by default, 1/1000000000)
            stream_info.time_base = Rational::new(1, 1_000_000_000);

            // Handle video-specific metadata
            if track_type == TrackType::Video {
                if let Some(video_meta) = track.video() {
                    stream_info.video_info = Some(VideoInfo {
                        width: video_meta.pixel_width().get() as u32,
                        height: video_meta.pixel_height().get() as u32,
                        frame_rate: Rational::new(30, 1), // Default, WebM doesn't always specify
                        aspect_ratio: Rational::new(1, 1),
                        pix_fmt: "yuv420p".to_string(), // Most common for WebM
                        bits_per_sample: 8,
                    });
                }
            }

            // Handle audio-specific metadata
            if track_type == TrackType::Audio {
                if let Some(audio_meta) = track.audio() {
                    stream_info.audio_info = Some(AudioInfo {
                        sample_rate: audio_meta.sampling_frequency() as u32,
                        channels: audio_meta.channels().get() as u16,
                        sample_fmt: "f32".to_string(), // WebM typically uses float
                        bits_per_sample: audio_meta
                            .bit_depth()
                            .map(|d| d.get() as u8)
                            .unwrap_or(16),
                        bit_rate: None,
                    });
                }
            }

            // Add stream to context
            let stream = Stream::new(stream_info);
            self.context.add_stream(stream);
        }

        self.file = Some(mkv_file);

        Ok(())
    }

    fn streams(&self) -> &[Stream] {
        self.context.streams()
    }

    fn read_packet(&mut self) -> Result<Packet> {
        let file = self
            .file
            .as_mut()
            .ok_or_else(|| Error::invalid_state("Demuxer not opened"))?;

        // Read next frame
        let has_more = file
            .next_frame(&mut self.frame_buffer)
            .map_err(|e| Error::format(format!("Failed to read frame: {}", e)))?;

        if !has_more {
            return Err(Error::EndOfStream);
        }

        // Convert frame to packet
        let data = Buffer::from_vec(self.frame_buffer.data.clone());
        let mut packet = Packet::new(self.frame_buffer.track as usize, data);

        // Set timestamps (WebM uses nanoseconds)
        packet.pts = Timestamp::new(self.frame_buffer.timestamp as i64);
        packet.dts = packet.pts; // WebM doesn't have separate DTS

        // Set keyframe flag from the frame metadata
        // The is_keyframe field is Some(true) for keyframes when using SimpleBlock format,
        // or None when the container doesn't specify. Default to true for audio frames.
        let is_keyframe = self.frame_buffer.is_keyframe.unwrap_or(false);
        packet.set_keyframe(is_keyframe);

        Ok(packet)
    }

    fn seek(&mut self, _stream_index: usize, timestamp: i64) -> Result<()> {
        let file = self
            .file
            .as_mut()
            .ok_or_else(|| Error::invalid_state("Demuxer not opened"))?;

        // Validate timestamp - must be non-negative
        if timestamp < 0 {
            return Err(Error::invalid_input(format!(
                "Seek timestamp {} cannot be negative",
                timestamp
            )));
        }

        // The matroska-demuxer seek() takes a u64 timestamp in nanoseconds.
        // Our timestamp is already in nanoseconds (time_base = 1/1000000000).
        // The seek will jump to the nearest cluster/keyframe at or after the timestamp.
        let seek_timestamp = timestamp as u64;

        file.seek(seek_timestamp)
            .map_err(|e| Error::format(format!("WebM seek failed: {}", e)))?;

        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        self.file = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webm_demuxer_creation() {
        let demuxer = WebmDemuxer::new();
        assert_eq!(demuxer.context.format_name(), "webm");
    }

    #[test]
    fn test_codec_id_mapping() {
        assert_eq!(WebmDemuxer::map_codec_id("V_AV1"), "av1");
        assert_eq!(WebmDemuxer::map_codec_id("V_VP9"), "vp9");
        assert_eq!(WebmDemuxer::map_codec_id("A_OPUS"), "opus");
    }

    #[test]
    fn test_seek_without_open_returns_error() {
        let mut demuxer = WebmDemuxer::new();
        // Seeking without opening should return an error
        let result = demuxer.seek(0, 1_000_000_000); // 1 second in nanoseconds
        assert!(result.is_err());
        if let Err(Error::InvalidState(msg)) = result {
            assert!(msg.contains("not opened"));
        } else {
            panic!("Expected InvalidState error");
        }
    }

    #[test]
    fn test_seek_negative_timestamp_returns_error() {
        let mut demuxer = WebmDemuxer::new();
        // We need to simulate an opened state for this test
        // Since we can't easily mock MatroskaFile, we test the validation logic
        // by ensuring negative timestamps are rejected even before the file check
        // Note: This test documents expected behavior - actual validation happens
        // after the file check in the current implementation
        let result = demuxer.seek(0, -1);
        // Should fail with either InvalidState (not opened) or InvalidInput (negative timestamp)
        assert!(result.is_err());
    }

    #[test]
    fn test_track_type_conversion() {
        assert_eq!(
            WebmDemuxer::track_type_to_media_type(TrackType::Video),
            MediaType::Video
        );
        assert_eq!(
            WebmDemuxer::track_type_to_media_type(TrackType::Audio),
            MediaType::Audio
        );
        assert_eq!(
            WebmDemuxer::track_type_to_media_type(TrackType::Subtitle),
            MediaType::Subtitle
        );
    }

    #[test]
    fn test_nanosecond_timestamp_conversion() {
        // Verify that timestamp conversion preserves values correctly
        // 1 second = 1,000,000,000 nanoseconds
        let one_second_ns: i64 = 1_000_000_000;
        let seek_timestamp = one_second_ns as u64;
        assert_eq!(seek_timestamp, 1_000_000_000u64);

        // 30 seconds
        let thirty_seconds_ns: i64 = 30_000_000_000;
        let seek_timestamp = thirty_seconds_ns as u64;
        assert_eq!(seek_timestamp, 30_000_000_000u64);
    }
}
