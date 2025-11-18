//! H.264 encoder using OpenH264
//!
//! This module provides a complete H.264 encoder implementation using the OpenH264 library.
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

use crate::codec::{Encoder, Frame, VideoFrame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, PixelFormat, Timestamp};

#[cfg(feature = "h264")]
use openh264::encoder::{Encoder as OpenH264Encoder, EncoderConfig};
#[cfg(feature = "h264")]
use openh264::formats::YUVBuffer;

/// H.264 encoder configuration
#[derive(Debug, Clone)]
pub struct H264EncoderConfig {
    pub width: u32,
    pub height: u32,
    pub bitrate: u32,
    pub framerate: f32,
    pub keyframe_interval: u32,
}

impl Default for H264EncoderConfig {
    fn default() -> Self {
        H264EncoderConfig {
            width: 640,
            height: 480,
            bitrate: 1_000_000,
            framerate: 30.0,
            keyframe_interval: 60,
        }
    }
}

/// H.264 encoder wrapping OpenH264
///
/// Encodes video frames to H.264/AVC using Cisco's OpenH264 library.
#[cfg(feature = "h264")]
pub struct H264Encoder {
    encoder: OpenH264Encoder,
    config: H264EncoderConfig,
    frame_count: u64,
    /// Buffered packets waiting to be retrieved
    packet_buffer: Vec<Packet>,
}

#[cfg(feature = "h264")]
impl H264Encoder {
    /// Create a new H.264 encoder with the given dimensions
    pub fn new(width: u32, height: u32) -> Result<Self> {
        let config = H264EncoderConfig {
            width,
            height,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create a new H.264 encoder with custom configuration
    pub fn with_config(config: H264EncoderConfig) -> Result<Self> {
        // Create OpenH264 encoder config
        let mut encoder_config = EncoderConfig::new(config.width, config.height);
        encoder_config.set_bitrate_bps(config.bitrate);
        encoder_config.set_max_frame_rate(config.framerate);

        // Create encoder
        let encoder = OpenH264Encoder::with_config(encoder_config)
            .map_err(|e| Error::codec(format!("Failed to create H.264 encoder: {:?}", e)))?;

        Ok(H264Encoder {
            encoder,
            config,
            frame_count: 0,
            packet_buffer: Vec::new(),
        })
    }

    /// Set target bitrate
    pub fn set_bitrate(&mut self, bitrate: u32) {
        self.config.bitrate = bitrate;
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

        // Validate dimensions
        if video_frame.width != self.config.width || video_frame.height != self.config.height {
            return Err(Error::codec(format!(
                "Frame dimensions {}x{} don't match encoder {}x{}",
                video_frame.width, video_frame.height, self.config.width, self.config.height
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
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let y_size = width * height;
        let uv_size = (width / 2) * (height / 2);
        let total_size = y_size + uv_size * 2;

        let mut yuv_data = vec![0u8; total_size];

        // Copy Y plane
        for y in 0..height {
            let src_start = y * y_stride;
            let src_end = src_start + width;
            let dst_start = y * width;
            let dst_end = dst_start + width;

            if src_end <= y_plane.len() && dst_end <= yuv_data.len() {
                yuv_data[dst_start..dst_end].copy_from_slice(&y_plane[src_start..src_end]);
            }
        }

        // Copy U plane
        let u_offset = y_size;
        let uv_width = width / 2;
        let uv_height = height / 2;
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
        Ok(YUVBuffer::from_vec(yuv_data, width, height))
    }
}

#[cfg(feature = "h264")]
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

                // Create packet from bitstream
                let data = Buffer::from_vec(bitstream.to_vec());
                let mut packet = Packet::new(0, data);
                packet.pts = video_frame.pts;
                packet.duration = video_frame.duration;
                packet.is_keyframe = self.frame_count % self.config.keyframe_interval as u64 == 0;

                self.packet_buffer.push(packet);
                self.frame_count += 1;

                Ok(())
            }
            Frame::Audio(_) => Err(Error::codec("H.264 encoder only accepts video frames")),
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if self.packet_buffer.is_empty() {
            return Err(Error::TryAgain);
        }

        Ok(self.packet_buffer.remove(0))
    }

    fn flush(&mut self) -> Result<()> {
        self.packet_buffer.clear();
        Ok(())
    }
}

#[cfg(feature = "h264")]
impl Default for H264Encoder {
    fn default() -> Self {
        Self::new(640, 480).expect("Failed to create default H.264 encoder")
    }
}

// Stub implementation when h264 feature is not enabled
#[cfg(not(feature = "h264"))]
pub struct H264Encoder {
    _private: (),
}

#[cfg(not(feature = "h264"))]
#[allow(dead_code)]
pub struct H264EncoderConfig {
    _private: (),
}

#[cfg(not(feature = "h264"))]
impl H264Encoder {
    pub fn new(_width: u32, _height: u32) -> Result<Self> {
        Err(Error::unsupported(
            "H.264 codec support not enabled. Enable the 'h264' feature and ensure libopenh264 is installed."
        ))
    }

    #[allow(dead_code)]
    pub fn with_config(_config: H264EncoderConfig) -> Result<Self> {
        Err(Error::unsupported("H.264 codec not enabled"))
    }
}

#[cfg(not(feature = "h264"))]
impl Encoder for H264Encoder {
    fn send_frame(&mut self, _frame: &Frame) -> Result<()> {
        Err(Error::unsupported("H.264 codec not enabled"))
    }

    fn receive_packet(&mut self) -> Result<Packet> {
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
    fn test_h264_encoder_creation() {
        let encoder = H264Encoder::new(640, 480);
        assert!(
            encoder.is_ok(),
            "Encoder creation failed. Make sure libopenh264 is installed."
        );
    }

    #[test]
    #[cfg(feature = "h264")]
    fn test_h264_encoder_with_config() {
        let config = H264EncoderConfig {
            width: 1920,
            height: 1080,
            bitrate: 5_000_000,
            framerate: 60.0,
            keyframe_interval: 120,
        };
        let encoder = H264Encoder::with_config(config);
        assert!(
            encoder.is_ok(),
            "Encoder creation failed. Make sure libopenh264 is installed."
        );
    }

    #[test]
    #[cfg(feature = "h264")]
    fn test_flush() {
        let mut encoder = H264Encoder::new(640, 480).expect("Failed to create encoder");
        assert!(encoder.flush().is_ok(), "Flush should not error");
    }

    #[test]
    #[cfg(not(feature = "h264"))]
    fn test_h264_disabled() {
        let encoder = H264Encoder::new(640, 480);
        assert!(encoder.is_err());
    }
}
