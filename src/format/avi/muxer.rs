//! AVI muxer implementation

use super::{AviMainHeader, AviStreamHeader, RiffChunk};
use crate::error::{Error, Result};
use crate::format::{Muxer, Packet, Stream};
use std::io::Write;

/// AVI muxer
pub struct AviMuxer<W: Write> {
    writer: W,
    main_header: AviMainHeader,
    stream_headers: Vec<AviStreamHeader>,
    streams: Vec<Stream>,
    frame_count: u32,
    started: bool,
}

impl<W: Write> AviMuxer<W> {
    /// Create a new AVI muxer
    pub fn new(writer: W, width: u32, height: u32, fps: u32) -> Self {
        AviMuxer {
            writer,
            main_header: AviMainHeader::new(width, height, fps),
            stream_headers: Vec::new(),
            streams: Vec::new(),
            frame_count: 0,
            started: false,
        }
    }

    /// Add video stream
    pub fn add_video_stream(&mut self, codec_fourcc: [u8; 4]) {
        let header = AviStreamHeader::video(
            codec_fourcc,
            self.main_header.width,
            self.main_header.height,
            1_000_000 / self.main_header.microsec_per_frame,
        );
        self.stream_headers.push(header);
    }

    /// Add audio stream
    pub fn add_audio_stream(&mut self, sample_rate: u32, channels: u16) {
        let header = AviStreamHeader::audio(1, sample_rate, channels);
        self.stream_headers.push(header);
    }

    /// Write RIFF chunk header
    fn write_chunk_header(&mut self, fourcc: &[u8; 4], size: u32) -> Result<()> {
        self.writer.write_all(fourcc)
            .map_err(|e| Error::Io(e))?;
        self.writer.write_all(&size.to_le_bytes())
            .map_err(|e| Error::Io(e))?;
        Ok(())
    }

    /// Write AVI headers
    fn write_headers(&mut self) -> Result<()> {
        // Write RIFF header
        self.write_chunk_header(b"RIFF", 0)?; // Size will be updated at end
        self.writer.write_all(b"AVI ")
            .map_err(|e| Error::Io(e))?;

        // Placeholder - would write hdrl LIST with avih, strl chunks

        // Write movi LIST header
        self.write_chunk_header(b"LIST", 0)?;
        self.writer.write_all(b"movi")
            .map_err(|e| Error::Io(e))?;

        Ok(())
    }
}

impl<W: Write> Muxer for AviMuxer<W> {
    fn create(&mut self, _path: &std::path::Path) -> Result<()> {
        // For writer-based muxer, this is a no-op
        Ok(())
    }

    fn add_stream(&mut self, stream: Stream) -> Result<usize> {
        let index = self.streams.len();
        self.streams.push(stream);
        Ok(index)
    }

    fn write_header(&mut self) -> Result<()> {
        if self.started {
            return Err(Error::invalid_state("Header already written"));
        }

        self.write_headers()?;
        self.started = true;
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if !self.started {
            return Err(Error::invalid_state("Header not written"));
        }

        // Placeholder - would write chunk with proper FourCC
        // Format: "00dc" for video, "01wb" for audio
        self.frame_count += 1;

        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        // Placeholder - would write idx1 index chunk
        // and update RIFF/movi sizes
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avi_muxer_creation() {
        let buffer = Vec::new();
        let muxer = AviMuxer::new(buffer, 1920, 1080, 30);
        assert!(!muxer.started);
        assert_eq!(muxer.frame_count, 0);
    }

    #[test]
    fn test_avi_add_streams() {
        let buffer = Vec::new();
        let mut muxer = AviMuxer::new(buffer, 1920, 1080, 30);
        muxer.add_video_stream(*b"H264");
        muxer.add_audio_stream(48000, 2);
        assert_eq!(muxer.stream_headers.len(), 2);
    }
}
