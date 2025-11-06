//! Y4M demuxer implementation

use crate::error::{Error, Result};
use crate::format::{Demuxer, DemuxerContext, Packet, Stream, StreamInfo, VideoInfo};
use crate::util::{Buffer, MediaType, Rational};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use y4m::Decoder;

/// Y4M demuxer for reading YUV4MPEG2 files
pub struct Y4mDemuxer {
    decoder: Option<Decoder<BufReader<File>>>,
    context: DemuxerContext,
    frame_number: u64,
}

impl Y4mDemuxer {
    /// Create a new Y4M demuxer
    pub fn new() -> Self {
        Y4mDemuxer {
            decoder: None,
            context: DemuxerContext::new("y4m".to_string()),
            frame_number: 0,
        }
    }

    /// Convert y4m colorspace to our pixel format string
    fn colorspace_to_pixel_format(colorspace: y4m::Colorspace) -> String {
        match colorspace {
            y4m::Colorspace::C420 | y4m::Colorspace::C420jpeg | y4m::Colorspace::C420paldv => {
                "yuv420p".to_string()
            }
            y4m::Colorspace::C422 => "yuv422p".to_string(),
            y4m::Colorspace::C444 => "yuv444p".to_string(),
            y4m::Colorspace::Cmono => "gray".to_string(),
            _ => format!("{:?}", colorspace).to_lowercase(),
        }
    }
}

impl Default for Y4mDemuxer {
    fn default() -> Self {
        Self::new()
    }
}

impl Demuxer for Y4mDemuxer {
    fn open(&mut self, path: &Path) -> Result<()> {
        // Open the file
        let file = File::open(path)
            .map_err(|e| Error::format(format!("Failed to open file: {}", e)))?;

        let reader = BufReader::new(file);

        // Create Y4M decoder
        let decoder = y4m::decode(reader)
            .map_err(|e| Error::format(format!("Failed to decode Y4M header: {}", e)))?;

        // Get video parameters
        let width = decoder.get_width();
        let height = decoder.get_height();
        let framerate = decoder.get_framerate();
        let colorspace = decoder.get_colorspace();

        // Create stream info for the video stream
        let mut stream_info = StreamInfo::new(0, MediaType::Video, "rawvideo".to_string());

        // Set time base from framerate
        stream_info.time_base = Rational::new(framerate.den as i64, framerate.num as i64);

        // Set video info
        stream_info.video_info = Some(VideoInfo {
            width: width as u32,
            height: height as u32,
            pix_fmt: Self::colorspace_to_pixel_format(colorspace),
            frame_rate: Rational::new(framerate.num as i64, framerate.den as i64),
            aspect_ratio: Rational::new(1, 1), // Default to square pixels
            bits_per_sample: 8, // Y4M typically uses 8-bit samples
        });

        // Add stream to context
        let stream = Stream::new(stream_info);
        self.context.add_stream(stream);

        self.decoder = Some(decoder);

        Ok(())
    }

    fn streams(&self) -> &[Stream] {
        self.context.streams()
    }

    fn read_packet(&mut self) -> Result<Packet> {
        let decoder = self
            .decoder
            .as_mut()
            .ok_or_else(|| Error::invalid_state("Demuxer not opened"))?;

        // Read next frame
        let frame = decoder.read_frame().map_err(|e| match e {
            y4m::Error::EOF => Error::EndOfStream,
            _ => Error::format(format!("Failed to read frame: {}", e)),
        })?;

        // Convert frame planes to a single buffer
        // Y4M frames are stored as separate planes (Y, U, V)
        let y_plane = frame.get_y_plane();
        let u_plane = frame.get_u_plane();
        let v_plane = frame.get_v_plane();

        // Concatenate planes into a single buffer
        let mut data = Vec::with_capacity(y_plane.len() + u_plane.len() + v_plane.len());
        data.extend_from_slice(y_plane);
        data.extend_from_slice(u_plane);
        data.extend_from_slice(v_plane);

        // Create packet
        let mut packet = Packet::new(0, Buffer::from_vec(data));

        // Set PTS based on frame number and time base
        packet.pts = crate::util::Timestamp::new(self.frame_number as i64);
        packet.dts = crate::util::Timestamp::new(self.frame_number as i64);
        packet.duration = 1; // Each frame has duration of 1 in time_base units

        self.frame_number += 1;

        Ok(packet)
    }

    fn seek(&mut self, _stream_index: usize, _timestamp: i64) -> Result<()> {
        // Y4M doesn't support seeking in a simple way since it's a sequential format
        // We would need to re-open the file and read frames sequentially to seek
        Err(Error::unsupported("Y4M seeking not yet implemented"))
    }

    fn close(&mut self) -> Result<()> {
        self.decoder = None;
        self.frame_number = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_y4m_demuxer_creation() {
        let demuxer = Y4mDemuxer::new();
        assert_eq!(demuxer.context.format_name(), "y4m");
        assert_eq!(demuxer.frame_number, 0);
    }
}
