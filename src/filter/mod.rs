//! Audio and video filtering and processing

pub mod audio;
pub mod chain;
pub mod graph;
pub mod video;

// Additional filter modules
pub mod denoise;
pub mod dsp;
pub mod loudness;
pub mod metering;
pub mod resampling;

pub use audio::{NormalizeFilter, ResampleFilter, VolumeFilter};
pub use chain::FilterChain;
pub use graph::{FilterGraph, FilterNode};
pub use video::{CropFilter, RotateFilter, ScaleFilter};

use crate::codec::Frame;
use crate::error::Result;

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
