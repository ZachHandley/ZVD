//! DNxHD encoder implementation

use super::{DnxhdFrameHeader, DnxhdProfile};
use crate::codec::{Encoder, Frame, VideoFrame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, PixelFormat, Timestamp};

/// DNxHD encoder configuration
#[derive(Debug, Clone, Default)]
pub struct DnxhdEncoderConfig {
    /// Quantization scale (1-31 for 8-bit, 1-255 for 10-bit)
    pub qscale: u8,
}

/// DNxHD video encoder
pub struct DnxhdEncoder {
    profile: DnxhdProfile,
    width: u32,
    height: u32,
    frame_count: u64,
    pending_packet: Option<Packet>,
    #[allow(dead_code)]
    config: DnxhdEncoderConfig,
}

impl DnxhdEncoder {
    /// Create a new DNxHD encoder with default config
    pub fn new(width: u32, height: u32, profile: DnxhdProfile) -> Result<Self> {
        Self::with_config(width, height, profile, DnxhdEncoderConfig::default())
    }

    /// Create a new DNxHD encoder with custom config
    pub fn with_config(
        width: u32,
        height: u32,
        profile: DnxhdProfile,
        config: DnxhdEncoderConfig,
    ) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::invalid_input("Width and height must be non-zero"));
        }

        // Validate resolution for DNxHD (not DNxHR)
        if !profile.is_dnxhr() {
            if width != 1920 || (height != 1080 && height != 1088) {
                return Err(Error::invalid_input(
                    "DNxHD requires 1920x1080 or 1920x1088. Use DNxHR for other resolutions.",
                ));
            }
        }

        // Validate qscale range
        let max_qscale = if profile.is_10bit() { 255 } else { 31 };
        if config.qscale > max_qscale {
            return Err(Error::invalid_input(format!(
                "qscale {} exceeds maximum {} for this profile",
                config.qscale, max_qscale
            )));
        }

        Ok(DnxhdEncoder {
            profile,
            width,
            height,
            frame_count: 0,
            pending_packet: None,
            config,
        })
    }

    /// Get the profile
    pub fn profile(&self) -> DnxhdProfile {
        self.profile
    }

    /// Encode frame header
    fn encode_frame_header(&self) -> Vec<u8> {
        let header = DnxhdFrameHeader::new(self.width as u16, self.height as u16, self.profile);

        let mut data = Vec::new();

        // Header prefix (0x000002800001 as 6 bytes)
        data.extend_from_slice(&[0x00, 0x00, 0x02, 0x80, 0x00, 0x01]);

        // Compression ID
        data.extend_from_slice(&header.compression_id.to_be_bytes());

        // Width and height
        data.extend_from_slice(&header.width.to_be_bytes());
        data.extend_from_slice(&header.height.to_be_bytes());

        // Flags
        let mut flags = 0u8;
        if header.is_progressive {
            flags |= 0x01;
        }
        if header.is_422 {
            flags |= 0x02;
        }
        data.push(flags);

        // Bit depth
        data.push(header.bit_depth);

        // Additional header fields (placeholder)
        // Real implementation would include:
        // - Macroblock count
        // - Quantization matrices
        // - Huffman tables

        data
    }

    /// Encode frame data (placeholder)
    fn encode_frame_data(&self, _frame: &VideoFrame) -> Result<Vec<u8>> {
        // Placeholder - actual DNxHD encoding involves:
        // - Block-based DCT transformation
        // - Quantization based on profile
        // - Variable-length coding
        // - Rate control to match target bitrate
        //
        // Would typically use a library like libavcodec

        // For now, return minimal valid structure
        let mut data = Vec::new();

        // Macroblock data placeholder
        let mb_width = (self.width + 15) / 16;
        let mb_height = (self.height + 15) / 16;
        let mb_count = mb_width * mb_height;

        for _ in 0..mb_count {
            // Minimal macroblock data
            data.extend_from_slice(&[0x00; 64]); // Placeholder coefficients
        }

        Ok(data)
    }
}

impl Encoder for DnxhdEncoder {
    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let video_frame = match frame {
            Frame::Video(vf) => vf,
            Frame::Audio(_) => return Err(Error::codec("DNxHD encoder only accepts video frames")),
        };

        if video_frame.width != self.width || video_frame.height != self.height {
            return Err(Error::invalid_input("Frame dimensions don't match encoder"));
        }

        // Encode frame header
        let mut encoded_data = self.encode_frame_header();

        // Encode frame data
        let frame_data = self.encode_frame_data(video_frame)?;
        encoded_data.extend_from_slice(&frame_data);

        // Create packet
        let data = Buffer::from_vec(encoded_data);
        let mut packet = Packet::new(0, data);
        packet.pts = video_frame.pts;
        packet.dts = video_frame.pts;
        packet.set_keyframe(true); // DNxHD frames are typically all intra-coded

        self.pending_packet = Some(packet);
        self.frame_count += 1;

        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.pending_packet.take().ok_or_else(|| Error::TryAgain)
    }

    fn flush(&mut self) -> Result<()> {
        // DNxHD has no delayed frames
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dnxhd_encoder_creation() {
        let encoder = DnxhdEncoder::new(1920, 1080, DnxhdProfile::Dnxhd115);
        assert!(encoder.is_ok());

        let encoder = encoder.unwrap();
        assert_eq!(encoder.width, 1920);
        assert_eq!(encoder.height, 1080);
    }

    #[test]
    fn test_dnxhd_encoder_invalid_resolution() {
        // DNxHD (not HR) requires 1920x1080
        let result = DnxhdEncoder::new(1280, 720, DnxhdProfile::Dnxhd115);
        assert!(result.is_err());
    }

    #[test]
    fn test_dnxhr_encoder_flexible_resolution() {
        // DNxHR supports any resolution
        let result = DnxhdEncoder::new(1280, 720, DnxhdProfile::DnxhrHq);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dnxhd_encode_frame() {
        let mut encoder = DnxhdEncoder::new(1920, 1080, DnxhdProfile::DnxhrHq).unwrap();
        let mut frame = VideoFrame::new(1920, 1080, PixelFormat::YUV422P);
        frame.pts = Timestamp::new(0);

        let send_result = encoder.send_frame(&Frame::Video(frame));
        assert!(send_result.is_ok());

        let packet_result = encoder.receive_packet();
        assert!(packet_result.is_ok());

        let packet = packet_result.unwrap();
        assert!(packet.data.len() > 0);
        assert!(packet.is_keyframe());
    }
}
