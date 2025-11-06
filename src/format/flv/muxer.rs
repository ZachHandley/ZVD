//! FLV muxer implementation

use super::{FlvHeader, FlvTagHeader, FlvTagType, FlvVideoCodec, FlvAudioCodec};
use crate::error::{Error, Result};
use crate::format::{Muxer, Packet, Stream};
use std::io::Write;

/// FLV muxer
pub struct FlvMuxer<W: Write> {
    writer: W,
    header: FlvHeader,
    streams: Vec<Stream>,
    started: bool,
    last_timestamp: u32,
}

impl<W: Write> FlvMuxer<W> {
    /// Create a new FLV muxer
    pub fn new(writer: W, has_video: bool, has_audio: bool) -> Self {
        FlvMuxer {
            writer,
            header: FlvHeader::new(has_video, has_audio),
            streams: Vec::new(),
            started: false,
            last_timestamp: 0,
        }
    }

    /// Write previous tag size
    fn write_prev_tag_size(&mut self, size: u32) -> Result<()> {
        self.writer.write_all(&size.to_be_bytes())
            .map_err(|e| Error::Io(e))?;
        Ok(())
    }

    /// Write FLV tag
    fn write_tag(&mut self, tag_type: FlvTagType, data: &[u8], timestamp: u32) -> Result<()> {
        let header = FlvTagHeader::new(tag_type, data.len() as u32, timestamp);

        // Write tag header
        self.writer.write_all(&header.to_bytes())
            .map_err(|e| Error::Io(e))?;

        // Write tag data
        self.writer.write_all(data)
            .map_err(|e| Error::Io(e))?;

        // Write previous tag size
        let prev_size = 11 + data.len() as u32;
        self.write_prev_tag_size(prev_size)?;

        self.last_timestamp = timestamp;
        Ok(())
    }

    /// Write metadata tag
    fn write_metadata(&mut self) -> Result<()> {
        // Placeholder - would write onMetaData script data tag
        // with duration, width, height, videocodecid, audiocodecid, etc.
        Ok(())
    }
}

impl<W: Write> Muxer for FlvMuxer<W> {
    fn create(&mut self, _path: &std::path::Path) -> Result<()> { Ok(()) }
    fn add_stream(&mut self, stream: Stream) -> Result<usize> {
        let index = self.streams.len();
        self.streams.push(stream);
        Ok(index)
    }

    fn write_header(&mut self) -> Result<()> {
        if self.started {
            return Err(Error::invalid_state("Header already written"));
        }

        // Write FLV header
        self.writer.write_all(&self.header.to_bytes())
            .map_err(|e| Error::Io(e))?;

        // Write PreviousTagSize0 (always 0)
        self.write_prev_tag_size(0)?;

        // Write metadata
        self.write_metadata()?;

        self.started = true;
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if !self.started {
            return Err(Error::invalid_state("Header not written"));
        }

        // Placeholder - would determine tag type from packet stream index
        // and write appropriate FLV tag with encoded data
        let timestamp_value = if packet.pts.is_valid() { packet.pts.value } else { 0 };
        let timestamp = (timestamp_value / 1000) as u32; // Convert to milliseconds

        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        // FLV has no trailer - all done in tags
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flv_muxer_creation() {
        let buffer = Vec::new();
        let muxer = FlvMuxer::new(buffer, true, true);
        assert!(!muxer.started);
        assert_eq!(muxer.last_timestamp, 0);
    }

    #[test]
    fn test_flv_header_write() {
        let buffer = Vec::new();
        let mut muxer = FlvMuxer::new(buffer, true, false);
        let result = muxer.write_header();
        assert!(result.is_ok());
        assert!(muxer.started);
    }
}
