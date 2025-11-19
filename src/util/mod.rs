//! Common utilities and data structures

pub mod rational;
pub mod timestamp;
pub mod timecode;
pub mod colorspace;
pub mod buffer;
pub mod pixfmt;
pub mod samplefmt;
pub mod thumbnail;
pub mod quality;
pub mod lut;
pub mod scopes;
pub mod proxy;

pub use rational::Rational;
pub use timestamp::Timestamp;
pub use timecode::{FrameRate, Timecode};
pub use colorspace::{ColorConverter, ColorRange, ColorStandard, Hsv, Rgb, Yuv};
pub use buffer::{Buffer, BufferRef};
pub use pixfmt::PixelFormat;
pub use samplefmt::SampleFormat;
pub use thumbnail::{
    FrameData, FrameQuality, SceneDetector, ThumbnailGenerator, ThumbnailMethod,
};
pub use quality::{calculate_psnr, calculate_ssim, QualityComparison, QualityMetrics};
pub use scopes::{Histogram, VectorscopeYUV, WaveformMode, WaveformScope};
pub use proxy::{
    ProxyCodec, ProxyConfig, ProxyGenerator, ProxyMetadata, ProxyPreset, ProxyQuality,
    ProxyResolution,
};

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
