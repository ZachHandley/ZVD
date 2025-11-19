//! Audio and video filtering and processing

pub mod audio;
pub mod chain;
pub mod denoise;
pub mod dsp;
pub mod graph;
pub mod loudness;
pub mod metering;
pub mod resampling;
pub mod video;

pub use audio::{NormalizeFilter, ResampleFilter, VolumeFilter};
pub use denoise::{AdaptiveSharpen, BilateralFilter, SpatialDenoiser, TemporalDenoiser, UnsharpMask};
pub use dsp::{Compressor, DeEsser, EqBandType, Expander, Limiter, ParametricEq};
pub use loudness::{LoudnessMeter, LoudnessNormalizer, targets};
pub use metering::{PeakMeter, PhaseCorrelationMeter, RmsLevelMeter, SpectrumAnalyzer, VuMeter};
pub use resampling::{AudioResampler, PitchShifter, PolyphaseResampler, ResamplingMethod};
pub use chain::FilterChain;
pub use graph::{FilterGraph, FilterNode};
pub use video::{
    BlurFilter, BrightnessContrastFilter, ChromaKeyFilter, CropFilter, DeinterlaceFilter,
    DeinterlaceMethod, FlipFilter, LutFilter, RotateFilter, ScaleFilter, SharpenFilter,
};

use crate::error::Result;
use crate::codec::Frame;

/// Filter trait for processing frames
pub trait Filter {
    /// Process an input frame and produce output frame(s)
    fn filter(&mut self, input: Frame) -> Result<Vec<Frame>>;

    /// Flush any buffered frames
    fn flush(&mut self) -> Result<Vec<Frame>>;
}

/// Filter descriptor
#[derive(Debug, Clone)]
pub struct FilterDescriptor {
    /// Filter name
    pub name: String,

    /// Filter description
    pub description: String,

    /// Input types
    pub input_types: Vec<String>,

    /// Output types
    pub output_types: Vec<String>,
}
