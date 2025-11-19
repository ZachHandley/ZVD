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
pub mod interpolation;
pub mod tenbit;
pub mod edl;
pub mod stabilization;
pub mod burnin;
pub mod sync;

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
pub use interpolation::{
    FrameData as InterpolationFrameData, FrameFormat, FrameInterpolator, FrameRateConverter,
    InterpolationMethod, MotionVector, SlowMotionGenerator,
};
pub use tenbit::{DitherMethod, TenBitAnalyzer, TenBitConverter, TenBitFormat, TenBitFrame};
pub use edl::{
    EditType, EdlExporter, Timecode as EdlTimecode, Timeline, TimelineEvent, TrackType,
};
pub use stabilization::{
    RollingShutterCorrector, SmoothingFilter, Stabilizer, StabilizationMode,
    StabilizationStats, Trajectory, Transform2D,
};
pub use burnin::{
    BurninConfig, BurninGenerator, BurninPosition, Color as BurninColor, TimecodeFormat,
    TimecodeStyle,
};
pub use sync::{AudioSpike, DriftCorrector, SyncDetector, SyncMethod, SyncResult};

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
