//! H.264 decoder using OpenH264
//!
//! This module provides a complete H.264 decoder implementation using the OpenH264 library.
//! OpenH264 is Cisco's open-source H.264 codec implementation.
//!
//! # System Requirements
//!
//! libopenh264 must be installed on the system:
//! - Debian/Ubuntu: `apt install libopenh264-dev`
//! - macOS: `brew install openh264`
//! - Fedora: `dnf install openh264-devel`
//!
//! # Security Note
//!
//! Always use the latest version of OpenH264 to ensure security patches are applied.

use crate::codec::{Decoder, Frame, VideoFrame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, PixelFormat, Timestamp};

#[cfg(feature = "h264")]
use openh264::decoder::Decoder as OpenH264Decoder;

/// H.264 decoder wrapping OpenH264
///
/// Decodes H.264/AVC video streams using Cisco's OpenH264 library.
#[cfg(feature = "h264")]
pub struct H264Decoder {
    decoder: OpenH264Decoder,
    frame_count: u64,
    /// Buffered decoded frames waiting to be retrieved
    frame_buffer: Vec<VideoFrame>,
}

#[cfg(feature = "h264")]
impl H264Decoder {
    /// Create a new H.264 decoder
    pub fn new() -> Result<Self> {
        let decoder = OpenH264Decoder::new()
            .map_err(|e| Error::codec(format!("Failed to create H.264 decoder: {:?}", e)))?;

        Ok(H264Decoder {
            decoder,
            frame_count: 0,
            frame_buffer: Vec::new(),
        })
    }

    /// Convert OpenH264 YUVBuffer to VideoFrame
    fn yuv_to_video_frame(&self, yuv: &openh264::formats::YUVBuffer, pts: Timestamp) -> Result<VideoFrame> {
        let width = yuv.width();
        let height = yuv.height();

        // OpenH264 outputs YUV420P
        let pixel_format = PixelFormat::YUV420P;

        // Get strides
        let y_stride = yuv.y_stride();
        let u_stride = yuv.u_stride();
        let v_stride = yuv.v_stride();

        // Get plane data
        let y_plane = yuv.y();
        let u_plane = yuv.u();
        let v_plane = yuv.v();

        // Create buffers - copy data to ensure ownership
        let y_buffer = Buffer::from_vec(y_plane.to_vec());
        let u_buffer = Buffer::from_vec(u_plane.to_vec());
        let v_buffer = Buffer::from_vec(v_plane.to_vec());

        // Create video frame
        let mut video_frame = VideoFrame::new(width as u32, height as u32, pixel_format);
        video_frame.data = vec![y_buffer, u_buffer, v_buffer];
        video_frame.linesize = vec![y_stride, u_stride, v_stride];
        video_frame.pts = pts;

        Ok(video_frame)
    }
}

#[cfg(feature = "h264")]
impl Decoder for H264Decoder {
    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if packet.data.is_empty() {
            return Err(Error::codec("Empty H.264 packet"));
        }

        // Decode the packet
        let data = packet.data.as_slice();

        // OpenH264 expects NAL units
        match self.decoder.decode(data) {
            Ok(Some(yuv)) => {
                // Frame decoded successfully - convert and buffer it
                let video_frame = self.yuv_to_video_frame(&yuv, packet.pts)?;
                self.frame_buffer.push(video_frame);
                self.frame_count += 1;
                Ok(())
            }
            Ok(None) => {
                // No frame yet, need more data
                Ok(())
            }
            Err(e) => Err(Error::codec(format!("H.264 decoding failed: {:?}", e))),
        }
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if self.frame_buffer.is_empty() {
            return Err(Error::TryAgain);
        }

        let video_frame = self.frame_buffer.remove(0);
        Ok(Frame::Video(video_frame))
    }

    fn flush(&mut self) -> Result<()> {
        self.frame_buffer.clear();
        self.frame_count = 0;
        Ok(())
    }
}

#[cfg(feature = "h264")]
impl Default for H264Decoder {
    fn default() -> Self {
        Self::new().expect("Failed to create default H.264 decoder")
    }
}

// Stub implementation when h264 feature is not enabled
#[cfg(not(feature = "h264"))]
pub struct H264Decoder {
    _private: (),
}

#[cfg(not(feature = "h264"))]
impl H264Decoder {
    pub fn new() -> Result<Self> {
        Err(Error::unsupported(
            "H.264 codec support not enabled. Enable the 'h264' feature and ensure libopenh264 is installed."
        ))
    }
}

#[cfg(not(feature = "h264"))]
impl Decoder for H264Decoder {
    fn send_packet(&mut self, _packet: &Packet) -> Result<()> {
        Err(Error::unsupported("H.264 codec not enabled"))
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        Err(Error::unsupported("H.264 codec not enabled"))
    }

    fn flush(&mut self) -> Result<()> {
        Err(Error::unsupported("H.264 codec not enabled"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "h264")]
    fn test_h264_decoder_creation() {
        let decoder = H264Decoder::new();
        assert!(
            decoder.is_ok(),
            "Decoder creation failed. Make sure libopenh264 is installed."
        );
    }

    #[test]
    #[cfg(feature = "h264")]
    fn test_flush() {
        let mut decoder = H264Decoder::new().expect("Failed to create decoder");
        assert!(decoder.flush().is_ok(), "Flush should not error");
    }

    #[test]
    #[cfg(not(feature = "h264"))]
    fn test_h264_disabled() {
        let decoder = H264Decoder::new();
        assert!(decoder.is_err());
    }
}
