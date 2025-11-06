//! Frame representation for uncompressed media data

use crate::util::{Buffer, PixelFormat, SampleFormat, Timestamp};
use std::fmt;

/// A frame of uncompressed media data
#[derive(Debug, Clone)]
pub enum Frame {
    Video(VideoFrame),
    Audio(AudioFrame),
}

impl Frame {
    /// Get the presentation timestamp
    pub fn pts(&self) -> Timestamp {
        match self {
            Frame::Video(f) => f.pts,
            Frame::Audio(f) => f.pts,
        }
    }

    /// Set the presentation timestamp
    pub fn set_pts(&mut self, pts: Timestamp) {
        match self {
            Frame::Video(f) => f.pts = pts,
            Frame::Audio(f) => f.pts = pts,
        }
    }
}

/// A video frame
#[derive(Debug, Clone)]
pub struct VideoFrame {
    /// Frame data (may be multiple planes)
    pub data: Vec<Buffer>,

    /// Line sizes for each plane
    pub linesize: Vec<usize>,

    /// Width in pixels
    pub width: u32,

    /// Height in pixels
    pub height: u32,

    /// Pixel format
    pub format: PixelFormat,

    /// Presentation timestamp
    pub pts: Timestamp,

    /// Duration
    pub duration: i64,

    /// Is keyframe
    pub keyframe: bool,

    /// Picture type (I, P, B)
    pub pict_type: PictureType,
}

impl VideoFrame {
    /// Create a new video frame
    pub fn new(width: u32, height: u32, format: PixelFormat) -> Self {
        VideoFrame {
            data: Vec::new(),
            linesize: Vec::new(),
            width,
            height,
            format,
            pts: Timestamp::none(),
            duration: 0,
            keyframe: false,
            pict_type: PictureType::None,
        }
    }

    /// Get the number of planes
    pub fn num_planes(&self) -> usize {
        self.data.len()
    }

    /// Get a plane by index
    pub fn plane(&self, index: usize) -> Option<&Buffer> {
        self.data.get(index)
    }
}

/// Picture type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PictureType {
    /// Intra frame
    I,
    /// Predicted frame
    P,
    /// Bidirectional frame
    B,
    /// None/Unknown
    None,
}

impl fmt::Display for PictureType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PictureType::I => write!(f, "I"),
            PictureType::P => write!(f, "P"),
            PictureType::B => write!(f, "B"),
            PictureType::None => write!(f, "?"),
        }
    }
}

/// An audio frame
#[derive(Debug, Clone)]
pub struct AudioFrame {
    /// Audio data (may be multiple channels for planar format)
    pub data: Vec<Buffer>,

    /// Number of samples
    pub nb_samples: usize,

    /// Sample rate
    pub sample_rate: u32,

    /// Number of channels
    pub channels: u16,

    /// Sample format
    pub format: SampleFormat,

    /// Presentation timestamp
    pub pts: Timestamp,

    /// Duration
    pub duration: i64,
}

impl AudioFrame {
    /// Create a new audio frame
    pub fn new(nb_samples: usize, sample_rate: u32, channels: u16, format: SampleFormat) -> Self {
        AudioFrame {
            data: Vec::new(),
            nb_samples,
            sample_rate,
            channels,
            format,
            pts: Timestamp::none(),
            duration: 0,
        }
    }

    /// Get the number of planes
    pub fn num_planes(&self) -> usize {
        if self.format.is_planar() {
            self.channels as usize
        } else {
            1
        }
    }

    /// Get total number of samples across all channels
    pub fn total_samples(&self) -> usize {
        self.nb_samples * self.channels as usize
    }
}
