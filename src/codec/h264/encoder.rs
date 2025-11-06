//! H.264 encoder using OpenH264

use crate::codec::{Encoder, Frame, VideoFrame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{PixelFormat, Timestamp};
use openh264::encoder::Encoder as OpenH264Encoder;
use openh264::formats::YUVBuffer;

/// H.264 encoder wrapping OpenH264
pub struct H264Encoder {
    encoder: OpenH264Encoder,
    width: usize,
    height: usize,
    frame_count: u64,
    pts_queue: Vec<Timestamp>, // Queue to track PTS for encoded packets
}

impl H264Encoder {
    /// Create a new H.264 encoder with the given dimensions
    pub fn new(width: u32, height: u32) -> Result<Self> {
        Self::with_bitrate(width, height, 1_000_000) // Default 1 Mbps
    }

    /// Create a new H.264 encoder with specified bitrate
    pub fn with_bitrate(width: u32, height: u32, _bitrate: u32) -> Result<Self> {
        // Create encoder with default config
        let encoder = OpenH264Encoder::new()
            .map_err(|e| Error::codec(format!("Failed to create H.264 encoder: {:?}", e)))?;

        Ok(H264Encoder {
            encoder,
            width: width as usize,
            height: height as usize,
            frame_count: 0,
            pts_queue: Vec::new(),
        })
    }

    /// Convert VideoFrame to YUVBuffer for OpenH264
    fn video_frame_to_yuv(&self, video_frame: &VideoFrame) -> Result<YUVBuffer> {
        // Ensure we have YUV420P format
        if video_frame.format != PixelFormat::YUV420P {
            return Err(Error::codec(format!(
                "Unsupported pixel format for H.264 encoding: {:?}. Only YUV420P is supported.",
                video_frame.format
            )));
        }

        if video_frame.data.len() < 3 {
            return Err(Error::codec("VideoFrame must have 3 planes (Y, U, V)"));
        }

        // Get plane data
        let y_plane = video_frame.data[0].as_slice();
        let u_plane = video_frame.data[1].as_slice();
        let v_plane = video_frame.data[2].as_slice();

        let y_stride = video_frame.linesize[0];
        let u_stride = video_frame.linesize[1];
        let v_stride = video_frame.linesize[2];

        // Create contiguous YUV420P data
        // For YUV420P: Y plane is full size, U and V are 1/4 size each
        let y_size = self.width * self.height;
        let uv_size = (self.width / 2) * (self.height / 2);
        let total_size = y_size + uv_size * 2;

        let mut yuv_data = vec![0u8; total_size];

        // Copy Y plane
        for y in 0..self.height {
            let src_start = y * y_stride;
            let src_end = src_start + self.width;
            let dst_start = y * self.width;
            let dst_end = dst_start + self.width;

            if src_end <= y_plane.len() && dst_end <= yuv_data.len() {
                yuv_data[dst_start..dst_end].copy_from_slice(&y_plane[src_start..src_end]);
            }
        }

        // Copy U plane
        let u_offset = y_size;
        let uv_width = self.width / 2;
        let uv_height = self.height / 2;
        for y in 0..uv_height {
            let src_start = y * u_stride;
            let src_end = src_start + uv_width;
            let dst_start = u_offset + y * uv_width;
            let dst_end = dst_start + uv_width;

            if src_end <= u_plane.len() && dst_end <= yuv_data.len() {
                yuv_data[dst_start..dst_end].copy_from_slice(&u_plane[src_start..src_end]);
            }
        }

        // Copy V plane
        let v_offset = y_size + uv_size;
        for y in 0..uv_height {
            let src_start = y * v_stride;
            let src_end = src_start + uv_width;
            let dst_start = v_offset + y * uv_width;
            let dst_end = dst_start + uv_width;

            if src_end <= v_plane.len() && dst_end <= yuv_data.len() {
                yuv_data[dst_start..dst_end].copy_from_slice(&v_plane[src_start..src_end]);
            }
        }

        // Create YUVBuffer from contiguous data
        Ok(YUVBuffer::from_vec(yuv_data, self.width, self.height))
    }
}

impl Encoder for H264Encoder {
    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        match frame {
            Frame::Video(video_frame) => {
                // Convert to YUV buffer
                let yuv = self.video_frame_to_yuv(video_frame)?;

                // Encode frame
                let bitstream = self
                    .encoder
                    .encode(&yuv)
                    .map_err(|e| Error::codec(format!("H.264 encoding failed: {:?}", e)))?;

                // Store PTS for later retrieval
                self.pts_queue.push(video_frame.pts);
                self.frame_count += 1;

                Ok(())
            }
            Frame::Audio(_) => Err(Error::codec("H.264 encoder only accepts video frames")),
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        // OpenH264 returns encoded data immediately in encode() call
        // This is a simplified implementation - in a real scenario, we'd need
        // to buffer the bitstream from encode() and return it here

        // For now, return TryAgain to indicate no packet is immediately available
        // A more complete implementation would buffer packets from encode()
        Err(Error::TryAgain)
    }

    fn flush(&mut self) -> Result<()> {
        // OpenH264 encoder doesn't require explicit flushing
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h264_encoder_creation() {
        let encoder = H264Encoder::new(640, 480);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_h264_encoder_with_bitrate() {
        let encoder = H264Encoder::with_bitrate(1920, 1080, 5_000_000);
        assert!(encoder.is_ok());
    }
}
