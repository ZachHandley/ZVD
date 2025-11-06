//! Filter chain for sequential filter application
//!
//! This module provides a simpler alternative to FilterGraph for the common
//! case of applying filters sequentially in a pipeline.

use super::Filter;
use crate::codec::Frame;
use crate::error::Result;

/// A chain of filters applied sequentially
pub struct FilterChain {
    filters: Vec<Box<dyn Filter>>,
}

impl FilterChain {
    /// Create a new empty filter chain
    pub fn new() -> Self {
        FilterChain {
            filters: Vec::new(),
        }
    }

    /// Add a filter to the end of the chain
    pub fn add(mut self, filter: Box<dyn Filter>) -> Self {
        self.filters.push(filter);
        self
    }

    /// Add a filter to the end of the chain (mutable version)
    pub fn push(&mut self, filter: Box<dyn Filter>) {
        self.filters.push(filter);
    }

    /// Get the number of filters in the chain
    pub fn len(&self) -> usize {
        self.filters.len()
    }

    /// Check if the chain is empty
    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    /// Process a frame through all filters in the chain
    pub fn process(&mut self, mut frame: Frame) -> Result<Frame> {
        for filter in &mut self.filters {
            let mut frames = filter.filter(frame)?;
            if frames.is_empty() {
                return Err(crate::error::Error::filter(
                    "Filter returned no frames (buffering?)",
                ));
            }
            // For now, we only support filters that produce exactly one frame
            // TODO: Handle filters that produce multiple frames (e.g., frame interpolation)
            frame = frames.remove(0);
        }
        Ok(frame)
    }

    /// Process multiple frames through the chain
    pub fn process_many(&mut self, frames: Vec<Frame>) -> Result<Vec<Frame>> {
        let mut output = Vec::new();
        for frame in frames {
            output.push(self.process(frame)?);
        }
        Ok(output)
    }

    /// Flush all filters in the chain
    pub fn flush(&mut self) -> Result<Vec<Frame>> {
        let mut output = Vec::new();
        for filter in &mut self.filters {
            output.extend(filter.flush()?);
        }
        Ok(output)
    }
}

impl Default for FilterChain {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{AudioFrame, Frame};
    use crate::filter::audio::VolumeFilter;

    #[test]
    fn test_empty_chain() {
        let chain = FilterChain::new();
        assert_eq!(chain.len(), 0);
        assert!(chain.is_empty());
    }

    #[test]
    fn test_chain_builder() {
        let chain = FilterChain::new()
            .add(Box::new(VolumeFilter::new(0.5)))
            .add(Box::new(VolumeFilter::new(2.0)));
        assert_eq!(chain.len(), 2);
    }
}
