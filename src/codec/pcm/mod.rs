//! PCM (Pulse Code Modulation) codec
//!
//! PCM is uncompressed audio, the simplest and most straightforward codec.
//! This module handles conversion between raw PCM data and audio frames.

pub mod decoder;
pub mod encoder;

pub use decoder::PcmDecoder;
pub use encoder::PcmEncoder;

use crate::util::SampleFormat;

/// PCM codec configuration
#[derive(Debug, Clone)]
pub struct PcmConfig {
    /// Sample format
    pub sample_format: SampleFormat,
    /// Number of channels
    pub channels: u16,
    /// Sample rate in Hz
    pub sample_rate: u32,
}

impl PcmConfig {
    /// Create a new PCM configuration
    pub fn new(sample_format: SampleFormat, channels: u16, sample_rate: u32) -> Self {
        PcmConfig {
            sample_format,
            channels,
            sample_rate,
        }
    }

    /// Get bytes per sample for a single channel
    pub fn bytes_per_sample(&self) -> usize {
        self.sample_format.sample_size()
    }

    /// Get bytes per frame (all channels)
    pub fn bytes_per_frame(&self) -> usize {
        self.bytes_per_sample() * self.channels as usize
    }

    /// Calculate number of samples from byte count
    pub fn samples_from_bytes(&self, bytes: usize) -> usize {
        bytes / self.bytes_per_frame()
    }
}
