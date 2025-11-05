//! Stream information and metadata

use crate::util::{MediaType, Rational};
use std::collections::HashMap;

/// Information about a media stream
#[derive(Debug, Clone)]
pub struct StreamInfo {
    /// Stream index
    pub index: usize,

    /// Media type
    pub media_type: MediaType,

    /// Codec identifier
    pub codec_id: String,

    /// Time base for this stream
    pub time_base: Rational,

    /// Start time
    pub start_time: i64,

    /// Duration in time_base units
    pub duration: i64,

    /// Number of frames (if known)
    pub nb_frames: Option<u64>,

    /// Stream metadata
    pub metadata: HashMap<String, String>,

    /// Video-specific info
    pub video_info: Option<VideoInfo>,

    /// Audio-specific info
    pub audio_info: Option<AudioInfo>,
}

impl StreamInfo {
    /// Create a new stream info
    pub fn new(index: usize, media_type: MediaType, codec_id: String) -> Self {
        StreamInfo {
            index,
            media_type,
            codec_id,
            time_base: Rational::new(1, 90000), // Default to 90kHz
            start_time: 0,
            duration: 0,
            nb_frames: None,
            metadata: HashMap::new(),
            video_info: None,
            audio_info: None,
        }
    }

    /// Get duration in seconds
    pub fn duration_seconds(&self) -> f64 {
        self.duration as f64 * self.time_base.to_f64()
    }
}

/// Video stream information
#[derive(Debug, Clone)]
pub struct VideoInfo {
    /// Width in pixels
    pub width: u32,

    /// Height in pixels
    pub height: u32,

    /// Frame rate
    pub frame_rate: Rational,

    /// Pixel aspect ratio
    pub aspect_ratio: Rational,

    /// Pixel format
    pub pix_fmt: String,

    /// Bits per raw sample
    pub bits_per_sample: u8,
}

impl VideoInfo {
    /// Create new video info
    pub fn new(width: u32, height: u32) -> Self {
        VideoInfo {
            width,
            height,
            frame_rate: Rational::new(25, 1), // Default 25fps
            aspect_ratio: Rational::new(1, 1),
            pix_fmt: String::from("yuv420p"),
            bits_per_sample: 8,
        }
    }

    /// Get display aspect ratio
    pub fn display_aspect_ratio(&self) -> Rational {
        let width = self.width as i64;
        let height = self.height as i64;
        Rational::new(width, height) * self.aspect_ratio
    }
}

/// Audio stream information
#[derive(Debug, Clone)]
pub struct AudioInfo {
    /// Sample rate in Hz
    pub sample_rate: u32,

    /// Number of channels
    pub channels: u16,

    /// Sample format
    pub sample_fmt: String,

    /// Bits per sample
    pub bits_per_sample: u8,

    /// Bitrate in bits per second (if known)
    pub bit_rate: Option<u64>,
}

impl AudioInfo {
    /// Create new audio info
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        AudioInfo {
            sample_rate,
            channels,
            sample_fmt: String::from("s16"),
            bits_per_sample: 16,
            bit_rate: None,
        }
    }
}

/// A media stream
#[derive(Debug, Clone)]
pub struct Stream {
    /// Stream information
    pub info: StreamInfo,

    /// Extra codec data (e.g., SPS/PPS for H.264)
    pub extradata: Option<Vec<u8>>,
}

impl Stream {
    /// Create a new stream
    pub fn new(info: StreamInfo) -> Self {
        Stream {
            info,
            extradata: None,
        }
    }
}
