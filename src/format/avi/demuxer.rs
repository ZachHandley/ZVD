//! AVI demuxer implementation

use super::{AviMainHeader, AviStreamHeader, RiffChunk};
use crate::error::{Error, Result};
use crate::format::{Demuxer, Packet, StreamInfo};
use std::io::{Read, Seek, SeekFrom};

/// AVI demuxer
pub struct AviDemuxer<R: Read + Seek> {
    reader: R,
    main_header: Option<AviMainHeader>,
    stream_headers: Vec<AviStreamHeader>,
    streams: Vec<StreamInfo>,
    current_frame: u64,
}

impl<R: Read + Seek> AviDemuxer<R> {
    /// Create a new AVI demuxer
    pub fn new(reader: R) -> Self {
        AviDemuxer {
            reader,
            main_header: None,
            stream_headers: Vec::new(),
            streams: Vec::new(),
            current_frame: 0,
        }
    }

    /// Read RIFF chunk header
    fn read_chunk_header(&mut self) -> Result<RiffChunk> {
        let mut fourcc = [0u8; 4];
        let mut size_buf = [0u8; 4];

        self.reader
            .read_exact(&mut fourcc)
            .map_err(|e| Error::Io(e))?;
        self.reader
            .read_exact(&mut size_buf)
            .map_err(|e| Error::Io(e))?;

        let size = u32::from_le_bytes(size_buf);

        Ok(RiffChunk { fourcc, size })
    }

    /// Parse AVI headers
    fn parse_headers(&mut self) -> Result<()> {
        // Read RIFF header
        let riff = self.read_chunk_header()?;
        if &riff.fourcc != b"RIFF" {
            return Err(Error::invalid_input("Not a valid AVI file"));
        }

        // Read AVI type
        let mut avi_type = [0u8; 4];
        self.reader
            .read_exact(&mut avi_type)
            .map_err(|e| Error::Io(e))?;
        if &avi_type != b"AVI " {
            return Err(Error::invalid_input("Not a valid AVI file"));
        }

        // Placeholder - would parse hdrl LIST chunk with avih, strl, etc.
        // For now, create a default header
        self.main_header = Some(AviMainHeader::new(1920, 1080, 30));

        Ok(())
    }
}

impl<R: Read + Seek> Demuxer for AviDemuxer<R> {
    fn open(&mut self, _path: &std::path::Path) -> Result<()> {
        self.parse_headers()
    }

    fn read_packet(&mut self) -> Result<Packet> {
        // Placeholder - would read movi LIST chunks
        // For now, return TryAgain to indicate no more packets
        Err(Error::TryAgain)
    }

    fn seek(&mut self, _stream_index: usize, timestamp: i64) -> Result<()> {
        // Placeholder - would seek to keyframe near timestamp
        self.current_frame = timestamp as u64;
        Ok(())
    }

    fn streams(&self) -> &[crate::format::Stream] {
        // Return empty slice for now - we're not using the streams field
        &[]
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_avi_demuxer_creation() {
        let data = Vec::new();
        let cursor = Cursor::new(data);
        let demuxer = AviDemuxer::new(cursor);
        assert_eq!(demuxer.current_frame, 0);
    }
}
