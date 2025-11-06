//! DNxHD encoder implementation

use super::{DnxhdProfile, DnxhdFrameHeader};
use crate::codec::{Encoder, Frame, VideoFrame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, PixelFormat, Timestamp};

/// DNxHD video encoder
pub struct DnxhdEncoder {
    profile: DnxhdProfile,
    width: u32,
    height: u32,
    frame_count: u64,
}

impl DnxhdEncoder {
    /// Create a new DNxHD encoder
    pub fn new(width: u32, height: u32, profile: DnxhdProfile) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::invalid_input("Width and height must be non-zero"));
        }

        // Validate resolution for DNxHD (not DNxHR)
        if !profile.is_dnxhr() {
            if width != 1920 || (height != 1080 && height != 1088) {
                return Err(Error::invalid_input(
                    "DNxHD requires 1920x1080 or 1920x1088. Use DNxHR for other resolutions."
                ));
            }
        }

        Ok(DnxhdEncoder {
            profile,
            width,
            height,
            frame_count: 0,
        })
    }

    /// Get the profile
    pub fn profile(&self) -> DnxhdProfile {
        self.profile
    }

    /// Encode frame header
    fn encode_frame_header(&self) -> Vec<u8> {
        let header = DnxhdFrameHeader::new(
            self.width as u16,
            self.height as u16,
            self.profile,
        );

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
    fn encode_frame_data(&self, frame: &VideoFrame) -> Result<Vec<u8>> {
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
    type Frame = VideoFrame;
    type Config = ();

    fn encode(&mut self, frame: &Self::Frame) -> Result<EncodedPacket> {
        if frame.width() != self.width || frame.height() != self.height {
            return Err(Error::invalid_input("Frame dimensions don't match encoder"));
        }

        // Encode frame header
        let mut encoded_data = self.encode_frame_header();

        // Encode frame data
        let frame_data = self.encode_frame_data(frame)?;
        encoded_data.extend_from_slice(&frame_data);

        let pts = frame.timestamp();
        self.frame_count += 1;

        Ok(EncodedPacket {
            data: Buffer::from_vec(encoded_data),
            pts,
            dts: pts,
            is_keyframe: true, // DNxHD frames are typically all intra-coded
        })
    }

    fn flush(&mut self) -> Result<Vec<EncodedPacket>> {
        // DNxHD has no delayed frames
        Ok(Vec::new())
    }

    fn configure(&mut self, _config: Self::Config) -> Result<()> {
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
        let frame = VideoFrame::new(1920, 1080, PixelFormat::Yuv422p, 0);

        let result = encoder.encode(&frame);
        assert!(result.is_ok());

        let packet = result.unwrap();
        assert!(packet.data.len() > 0);
        assert!(packet.is_keyframe);
    }
}
