//! Demuxer for reading container formats

use super::{Packet, Stream};
use crate::error::{Error, Result};
use std::path::Path;

/// Demuxer trait for reading container formats
pub trait Demuxer {
    /// Open a file for demuxing
    fn open(&mut self, path: &Path) -> Result<()>;

    /// Get the list of streams in this container
    fn streams(&self) -> &[Stream];

    /// Read the next packet
    fn read_packet(&mut self) -> Result<Packet>;

    /// Seek to a specific timestamp (in stream time_base units)
    fn seek(&mut self, stream_index: usize, timestamp: i64) -> Result<()>;

    /// Close the demuxer
    fn close(&mut self) -> Result<()>;
}

/// Context for demuxing operations
pub struct DemuxerContext {
    streams: Vec<Stream>,
    format_name: String,
    duration: i64,
}

impl DemuxerContext {
    /// Create a new demuxer context
    pub fn new(format_name: String) -> Self {
        DemuxerContext {
            streams: Vec::new(),
            format_name,
            duration: 0,
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

    /// Get a specific stream
    pub fn stream(&self, index: usize) -> Option<&Stream> {
        self.streams.get(index)
    }

    /// Add a stream
    pub fn add_stream(&mut self, stream: Stream) {
        self.streams.push(stream);
    }

    /// Get the duration
    pub fn duration(&self) -> i64 {
        self.duration
    }

    /// Set the duration
    pub fn set_duration(&mut self, duration: i64) {
        self.duration = duration;
    }
}

/// Create a demuxer for the given file
pub fn create_demuxer(path: &Path) -> Result<Box<dyn Demuxer>> {
    // For now, return an error - we'll implement specific demuxers later
    Err(Error::unsupported(format!(
        "No demuxer available for file: {}",
        path.display()
    )))
}
