//! Audio and video filtering and processing

pub mod graph;
pub mod video;
pub mod audio;

pub use graph::{FilterGraph, FilterNode};

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
