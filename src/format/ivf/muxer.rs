//! IVF muxer (video-only)
//!
//! This provides a minimal IVF writer suitable for AV1/VP9 elementary streams.
//! It is deliberately small and dependency-free.

use crate::error::{Error, Result};
use crate::format::{Muxer, MuxerContext, Packet, Stream};
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

/// IVF muxer for video-only streams
pub struct IvfMuxer {
    writer: Option<File>,
    context: MuxerContext,
    frame_count: u32,
    width: u16,
    height: u16,
    // IVF stores a simple timebase as fps_num/fps_den (integers)
    fps_num: u32,
    fps_den: u32,
    // FourCC defining the codec (e.g., "AV01" or "VP90")
    fourcc: [u8; 4],
}

impl IvfMuxer {
    pub fn new() -> Self {
        IvfMuxer {
            writer: None,
            context: MuxerContext::new("ivf".to_string()),
            frame_count: 0,
            width: 0,
            height: 0,
            fps_num: 30,
            fps_den: 1,
            fourcc: *b"AV01",
        }
    }

    fn write_header(&mut self) -> Result<()> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| Error::format("IVF writer not initialized"))?;

        let mut header = [0u8; 32];
        header[0..4].copy_from_slice(b"DKIF"); // signature
        header[4..6].copy_from_slice(&0u16.to_le_bytes()); // version
        header[6..8].copy_from_slice(&32u16.to_le_bytes()); // header size
        header[8..12].copy_from_slice(&self.fourcc); // codec fourcc
        header[12..14].copy_from_slice(&self.width.to_le_bytes());
        header[14..16].copy_from_slice(&self.height.to_le_bytes());
        header[16..20].copy_from_slice(&self.fps_num.to_le_bytes());
        header[20..24].copy_from_slice(&self.fps_den.to_le_bytes());
        header[24..28].copy_from_slice(&self.frame_count.to_le_bytes()); // placeholder
        header[28..32].copy_from_slice(&0u32.to_le_bytes()); // unused

        writer
            .write_all(&header)
            .map_err(|e| Error::format(format!("Failed to write IVF header: {}", e)))?;

        Ok(())
    }

    fn update_frame_count(&mut self) -> Result<()> {
        if let Some(writer) = self.writer.as_mut() {
            writer
                .seek(SeekFrom::Start(24))
                .map_err(|e| Error::format(format!("Failed to seek IVF header: {}", e)))?;
            writer
                .write_all(&self.frame_count.to_le_bytes())
                .map_err(|e| Error::format(format!("Failed to write frame count: {}", e)))?;
            writer
                .seek(SeekFrom::End(0))
                .map_err(|e| Error::format(format!("Failed to seek to end of file: {}", e)))?;
        }
        Ok(())
    }
}

impl Default for IvfMuxer {
    fn default() -> Self {
        Self::new()
    }
}

impl Muxer for IvfMuxer {
    fn create(&mut self, path: &Path) -> Result<()> {
        let file = File::create(path)
            .map_err(|e| Error::format(format!("Failed to create IVF file: {}", e)))?;
        self.writer = Some(file);
        Ok(())
    }

    fn add_stream(&mut self, stream: Stream) -> Result<usize> {
        // IVF supports a single video stream. Validate codec and stash geometry/timebase.
        if !self.context.streams().is_empty() {
            return Err(Error::unsupported(
                "IVF muxer supports only a single video stream".to_string(),
            ));
        }

        if stream.info.media_type != crate::util::MediaType::Video {
            return Err(Error::unsupported(
                "IVF muxer only supports video streams".to_string(),
            ));
        }

        // Choose fourcc based on codec id
        self.fourcc = match stream.info.codec_id.as_str() {
            "av1" => *b"AV01",
            "vp9" => *b"VP90",
            "vp8" => *b"VP80",
            other => {
                return Err(Error::unsupported(format!(
                    "IVF codec not supported: {} (expected av1/vp9/vp8)",
                    other
                )))
            }
        };

        // Derive width/height/fps
        let video = stream
            .info
            .video_info
            .as_ref()
            .ok_or_else(|| Error::format("Video info missing for IVF stream"))?;

        self.width = video.width as u16;
        self.height = video.height as u16;
        self.fps_num = video.frame_rate.num as u32;
        self.fps_den = video.frame_rate.den as u32;
        if self.fps_den == 0 {
            self.fps_den = 1;
        }

        // Write header now that we know basic stream metadata
        self.write_header()?;

        Ok(self.context.add_stream(stream))
    }

    fn write_header(&mut self) -> Result<()> {
        // Header already written in add_stream once we had stream metadata.
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| Error::format("IVF writer not initialized"))?;

        let frame_len = packet.data.len() as u32;
        writer
            .write_all(&frame_len.to_le_bytes())
            .map_err(|e| Error::format(format!("Failed to write IVF frame size: {}", e)))?;

        // IVF expects timestamp as little-endian u64 in stream timebase units
        writer
            .write_all(&(packet.pts.value as u64).to_le_bytes())
            .map_err(|e| Error::format(format!("Failed to write IVF timestamp: {}", e)))?;

        writer
            .write_all(packet.data.as_slice())
            .map_err(|e| Error::format(format!("Failed to write IVF frame: {}", e)))?;

        self.frame_count = self.frame_count.saturating_add(1);
        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        self.update_frame_count()
    }
}
