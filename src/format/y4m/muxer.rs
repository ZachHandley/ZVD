//! Y4M muxer implementation

use crate::error::{Error, Result};
use crate::format::{Muxer, MuxerContext, Packet, Stream};
use crate::util::MediaType;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use y4m::Encoder;

/// Y4M muxer for writing YUV4MPEG2 files
pub struct Y4mMuxer {
    encoder: Option<Encoder<BufWriter<File>>>,
    context: MuxerContext,
}

impl Y4mMuxer {
    /// Create a new Y4M muxer
    pub fn new() -> Self {
        Y4mMuxer {
            encoder: None,
            context: MuxerContext::new("y4m".to_string()),
        }
    }

    /// Convert our pixel format string to y4m colorspace
    fn pixel_format_to_colorspace(pix_fmt: &str) -> y4m::Colorspace {
        match pix_fmt {
            "yuv420p" => y4m::Colorspace::C420,
            "yuv422p" => y4m::Colorspace::C422,
            "yuv444p" => y4m::Colorspace::C444,
            "gray" => y4m::Colorspace::Cmono,
            _ => y4m::Colorspace::C420, // Default to 4:2:0
        }
    }
}

impl Default for Y4mMuxer {
    fn default() -> Self {
        Self::new()
    }
}

impl Muxer for Y4mMuxer {
    fn create(&mut self, path: &Path) -> Result<()> {
        // Just store the path for now, we'll create the encoder in write_header
        // when we have stream information
        Ok(())
    }

    fn add_stream(&mut self, stream: Stream) -> Result<usize> {
        // Ensure this is a video stream
        if stream.info.media_type != MediaType::Video {
            return Err(Error::unsupported("Y4M only supports video streams"));
        }

        Ok(self.context.add_stream(stream))
    }

    fn write_header(&mut self) -> Result<()> {
        // Get the video stream
        let stream = self
            .context
            .streams()
            .first()
            .ok_or_else(|| Error::invalid_state("No stream added"))?;

        let video_info = stream
            .info
            .video_info
            .as_ref()
            .ok_or_else(|| Error::invalid_state("No video info in stream"))?;

        // For now, we'll need to defer creating the encoder until we have a file
        // This is a limitation we'll address when we refactor to handle the file path
        self.context.set_header_written();

        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        let encoder = self
            .encoder
            .as_mut()
            .ok_or_else(|| Error::invalid_state("Encoder not initialized"))?;

        // Get the stream info to determine plane sizes
        let stream = self
            .context
            .streams()
            .first()
            .ok_or_else(|| Error::invalid_state("No stream found"))?;

        let video_info = stream
            .info
            .video_info
            .as_ref()
            .ok_or_else(|| Error::invalid_state("No video info"))?;

        let width = video_info.width as usize;
        let height = video_info.height as usize;

        // Calculate plane sizes based on colorspace
        // For 4:2:0, Y plane is full size, U and V are quarter size each
        let colorspace = Self::pixel_format_to_colorspace(&video_info.pix_fmt);
        let (y_size, u_size, v_size) = match colorspace {
            y4m::Colorspace::C420 | y4m::Colorspace::C420jpeg | y4m::Colorspace::C420paldv => {
                (width * height, width * height / 4, width * height / 4)
            }
            y4m::Colorspace::C422 => (width * height, width * height / 2, width * height / 2),
            y4m::Colorspace::C444 => (width * height, width * height, width * height),
            y4m::Colorspace::Cmono => (width * height, 0, 0),
            _ => (width * height, width * height / 4, width * height / 4), // Default to 4:2:0
        };

        // Extract planes from packet data
        let data = packet.data.as_slice();
        if data.len() < y_size + u_size + v_size {
            return Err(Error::invalid_input(format!(
                "Packet data too small: expected at least {} bytes, got {}",
                y_size + u_size + v_size,
                data.len()
            )));
        }

        let y_plane = &data[0..y_size];
        let u_plane = &data[y_size..y_size + u_size];
        let v_plane = &data[y_size + u_size..y_size + u_size + v_size];

        // Create a Y4M frame
        let frame = y4m::Frame::new([y_plane, u_plane, v_plane], None);

        // Write the frame
        encoder
            .write_frame(&frame)
            .map_err(|e| Error::format(format!("Failed to write frame: {}", e)))?;

        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        // Y4M doesn't have a trailer, just close the encoder
        self.encoder = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_y4m_muxer_creation() {
        let muxer = Y4mMuxer::new();
        assert_eq!(muxer.context.format_name(), "y4m");
    }
}
