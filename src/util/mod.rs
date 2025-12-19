//! Common utilities and data structures

pub mod buffer;
pub mod pixfmt;
pub mod rational;
pub mod samplefmt;
pub mod timestamp;

// Additional utility modules
pub mod alpha;
pub mod aspect;
pub mod burnin;
pub mod captions;
pub mod colorspace;
pub mod detection;
pub mod edl;
pub mod embedding;
pub mod interpolation;
pub mod lut;
pub mod proxy;
pub mod quality;
pub mod scopes;
pub mod stabilization;
pub mod sync;
pub mod telecine;
pub mod tenbit;
pub mod thumbnail;
pub mod timecode;
pub mod watermark;

pub use buffer::{Buffer, BufferRef};
pub use pixfmt::PixelFormat;
pub use rational::Rational;
pub use samplefmt::SampleFormat;
pub use timestamp::Timestamp;

/// Common media types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaType {
    /// Video stream
    Video,
    /// Audio stream
    Audio,
    /// Subtitle stream
    Subtitle,
    /// Data stream
    Data,
    /// Unknown stream type
    Unknown,
}

impl fmt::Display for MediaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MediaType::Video => write!(f, "video"),
            MediaType::Audio => write!(f, "audio"),
            MediaType::Subtitle => write!(f, "subtitle"),
            MediaType::Data => write!(f, "data"),
            MediaType::Unknown => write!(f, "unknown"),
        }
    }
}

use std::fmt;
