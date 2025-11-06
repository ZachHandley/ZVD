//! Muxer for writing container formats

use super::{Packet, Stream};
use crate::error::{Error, Result};
use std::path::Path;

/// Muxer trait for writing container formats
pub trait Muxer {
    /// Create a new file for muxing
    fn create(&mut self, path: &Path) -> Result<()>;

    /// Add a stream to the output
    fn add_stream(&mut self, stream: Stream) -> Result<usize>;

    /// Write the header
    fn write_header(&mut self) -> Result<()>;

    /// Write a packet
    fn write_packet(&mut self, packet: &Packet) -> Result<()>;

    /// Write the trailer and close the file
    fn write_trailer(&mut self) -> Result<()>;
}

/// Context for muxing operations
pub struct MuxerContext {
    streams: Vec<Stream>,
    format_name: String,
    header_written: bool,
}

impl MuxerContext {
    /// Create a new muxer context
    pub fn new(format_name: String) -> Self {
        MuxerContext {
            streams: Vec::new(),
            format_name,
            header_written: false,
        }
    }

    /// Get the format name
    pub fn format_name(&self) -> &str {
        &self.format_name
    }

    /// Get all streams
    pub fn streams(&self) -> &[Stream] {
        &self.streams
    }

    /// Add a stream
    pub fn add_stream(&mut self, stream: Stream) -> usize {
        let index = self.streams.len();
        self.streams.push(stream);
        index
    }

    /// Check if header has been written
    pub fn is_header_written(&self) -> bool {
        self.header_written
    }

    /// Mark header as written
    pub fn set_header_written(&mut self) {
        self.header_written = true;
    }
}

/// Create a muxer for the given format
pub fn create_muxer(format: &str) -> Result<Box<dyn Muxer>> {
    use super::wav::WavMuxer;
    use super::y4m::Y4mMuxer;

    match format {
        "wav" => Ok(Box::new(WavMuxer::new())),
        "y4m" => Ok(Box::new(Y4mMuxer::new())),
        _ => Err(Error::unsupported(format!(
            "No muxer available for format: {}",
            format
        ))),
    }
}
