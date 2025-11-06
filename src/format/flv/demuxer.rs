//! FLV demuxer implementation

use super::{FlvHeader, FlvTagHeader, FlvTagType};
use crate::error::{Error, Result};
use crate::format::{Demuxer, Packet, Stream};
use std::io::{Read, Seek, SeekFrom};

/// FLV demuxer
pub struct FlvDemuxer<R: Read + Seek> {
    reader: R,
    header: Option<FlvHeader>,
    streams: Vec<Stream>,
    duration: Option<f64>,
}

impl<R: Read + Seek> FlvDemuxer<R> {
    /// Create a new FLV demuxer
    pub fn new(reader: R) -> Self {
        FlvDemuxer {
            reader,
            header: None,
            streams: Vec::new(),
            duration: None,
        }
    }

    /// Parse FLV header
    fn parse_header(&mut self) -> Result<FlvHeader> {
        let mut signature = [0u8; 3];
        self.reader.read_exact(&mut signature)
            .map_err(|e| Error::Io(e))?;

        if &signature != b"FLV" {
            return Err(Error::invalid_input("Not a valid FLV file"));
        }

        let mut version = [0u8; 1];
        self.reader.read_exact(&mut version)
            .map_err(|e| Error::Io(e))?;

        let mut flags = [0u8; 1];
        self.reader.read_exact(&mut flags)
            .map_err(|e| Error::Io(e))?;

        let has_video = (flags[0] & 0x01) != 0;
        let has_audio = (flags[0] & 0x04) != 0;

        let mut offset_bytes = [0u8; 4];
        self.reader.read_exact(&mut offset_bytes)
            .map_err(|e| Error::Io(e))?;
        let data_offset = u32::from_be_bytes(offset_bytes);

        Ok(FlvHeader {
            signature,
            version: version[0],
            has_video,
            has_audio,
            data_offset,
        })
    }

    /// Read FLV tag header
    fn read_tag_header(&mut self) -> Result<FlvTagHeader> {
        let mut tag_type_byte = [0u8; 1];
        self.reader.read_exact(&mut tag_type_byte)
            .map_err(|e| Error::Io(e))?;

        let tag_type = FlvTagType::from_u8(tag_type_byte[0])
            .ok_or_else(|| Error::invalid_input("Invalid FLV tag type"))?;

        // Data size (24 bits)
        let mut size_bytes = [0u8; 3];
        self.reader.read_exact(&mut size_bytes)
            .map_err(|e| Error::Io(e))?;
        let data_size = ((size_bytes[0] as u32) << 16)
            | ((size_bytes[1] as u32) << 8)
            | (size_bytes[2] as u32);

        // Timestamp (24 bits + 8 bit extension)
        let mut ts_bytes = [0u8; 3];
        self.reader.read_exact(&mut ts_bytes)
            .map_err(|e| Error::Io(e))?;
        let mut ts_ext = [0u8; 1];
        self.reader.read_exact(&mut ts_ext)
            .map_err(|e| Error::Io(e))?;

        let timestamp = ((ts_ext[0] as u32) << 24)
            | ((ts_bytes[0] as u32) << 16)
            | ((ts_bytes[1] as u32) << 8)
            | (ts_bytes[2] as u32);

        // Stream ID (always 0)
        let mut stream_id_bytes = [0u8; 3];
        self.reader.read_exact(&mut stream_id_bytes)
            .map_err(|e| Error::Io(e))?;

        Ok(FlvTagHeader {
            tag_type,
            data_size,
            timestamp,
            stream_id: 0,
        })
    }
}

impl<R: Read + Seek> Demuxer for FlvDemuxer<R> {
    fn close(&mut self) -> Result<()> { Ok(()) }
    fn open(&mut self, _path: &std::path::Path) -> Result<()> {
        let header = self.parse_header()?;

        // Skip to first tag (skip PreviousTagSize0)
        self.reader.seek(SeekFrom::Start(header.data_offset as u64))
            .map_err(|e| Error::Io(e))?;
        let mut prev_tag = [0u8; 4];
        self.reader.read_exact(&mut prev_tag)
            .map_err(|e| Error::Io(e))?;

        self.header = Some(header);
        Ok(())
    }

    fn read_packet(&mut self) -> Result<Packet> {
        // Placeholder - would read next FLV tag
        // Parse tag header, read tag data, convert to Packet
        Err(Error::TryAgain)
    }

    fn seek(&mut self, _stream_index: usize, timestamp: i64) -> Result<()> {
        // Placeholder - would seek to keyframe near timestamp
        Ok(())
    }

    fn streams(&self) -> &[crate::format::Stream] {
        &self.streams
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_flv_demuxer_creation() {
        let data = Vec::new();
        let cursor = Cursor::new(data);
        let demuxer = FlvDemuxer::new(cursor);
        assert!(demuxer.header.is_none());
    }
}
