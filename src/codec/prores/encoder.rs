//! ProRes encoder implementation

use super::{ProResProfile, ProResFrameHeader};
use crate::codec::{Encoder, Frame, VideoFrame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, PixelFormat, Timestamp};

/// ProRes video encoder
pub struct ProResEncoder {
    profile: ProResProfile,
    width: u32,
    height: u32,
    frame_count: u64,
    pending_packet: Option<Packet>,
}

impl ProResEncoder {
    /// Create a new ProRes encoder
    pub fn new(width: u32, height: u32, profile: ProResProfile) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::invalid_input("Width and height must be non-zero"));
        }

        Ok(ProResEncoder {
            profile,
            width,
            height,
            frame_count: 0,
            pending_packet: None,
        })
    }

    /// Get the profile
    pub fn profile(&self) -> ProResProfile {
        self.profile
    }

    /// Encode frame header
    fn encode_frame_header(&self) -> Vec<u8> {
        let header = ProResFrameHeader::new(
            self.width as u16,
            self.height as u16,
            self.profile,
        );

        let mut data = Vec::new();

        // Frame size (will be updated)
        data.extend_from_slice(&0u32.to_be_bytes());

        // Frame identifier "icpf"
        data.extend_from_slice(&header.frame_identifier);

        // Header size
        data.extend_from_slice(&header.header_size.to_be_bytes());

        // Version and flags
        data.push(header.version);

        // Encoder ID
        data.extend_from_slice(&header.encoder_id);

        // Dimensions
        data.extend_from_slice(&header.width.to_be_bytes());
        data.extend_from_slice(&header.height.to_be_bytes());

        // Chroma format and other flags
        data.push(header.chroma_format);
        data.push(header.interlace_mode);
        data.push(header.aspect_ratio);
        data.push(header.framerate_code);

        // Color information
        data.push(header.color_primaries);
        data.push(header.transfer_characteristics);
        data.push(header.matrix_coefficients);

        // Alpha info
        data.push(header.alpha_info);

        // Pad to header size
        while data.len() < header.header_size as usize {
            data.push(0);
        }

        data
    }

    /// Encode frame data (placeholder)
    fn encode_frame_data(&self, frame: &VideoFrame) -> Result<Vec<u8>> {
        // Placeholder - actual ProRes encoding is complex, involving:
        // - DCT transformation
        // - Quantization
        // - Slice encoding
        // - Frame data organization
        //
        // Would typically use a library like libavcodec or implement
        // the full ProRes specification

        // For now, return a minimal valid structure
        let mut data = Vec::new();

        // Picture header
        data.push(0x00); // Picture header size

        // Slice data placeholder
        let slice_count = (self.height + 15) / 16; // Slices are typically 16 lines
        for _ in 0..slice_count {
            // Slice header and data placeholder
            data.extend_from_slice(&[0x00; 128]); // Minimal slice
        }

        Ok(data)
    }
}

impl Encoder for ProResEncoder {
    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let video_frame = match frame {
            Frame::Video(vf) => vf,
            Frame::Audio(_) => return Err(Error::codec("ProRes encoder only accepts video frames")),
        };

        if video_frame.width != self.width || video_frame.height != self.height {
            return Err(Error::invalid_input("Frame dimensions don't match encoder"));
        }

        // Encode frame header
        let mut encoded_data = self.encode_frame_header();

        // Encode frame data
        let frame_data = self.encode_frame_data(video_frame)?;
        encoded_data.extend_from_slice(&frame_data);

        // Update frame size in header
        let total_size = encoded_data.len() as u32;
        encoded_data[0..4].copy_from_slice(&total_size.to_be_bytes());

        // Create packet
        let data = Buffer::from_vec(encoded_data);
        let mut packet = Packet::new(0, data);
        packet.pts = video_frame.pts;
        packet.dts = video_frame.pts;
        packet.set_keyframe(true); // ProRes frames are typically all keyframes

        self.pending_packet = Some(packet);
        self.frame_count += 1;

        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.pending_packet.take().ok_or_else(|| Error::TryAgain)
    }

    fn flush(&mut self) -> Result<()> {
        // ProRes has no delayed frames
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prores_encoder_creation() {
        let encoder = ProResEncoder::new(1920, 1080, ProResProfile::Standard);
        assert!(encoder.is_ok());

        let encoder = encoder.unwrap();
        assert_eq!(encoder.width, 1920);
        assert_eq!(encoder.height, 1080);
        assert_eq!(encoder.profile(), ProResProfile::Standard);
    }

    #[test]
    fn test_prores_encoder_invalid_dimensions() {
        let result = ProResEncoder::new(0, 1080, ProResProfile::Standard);
        assert!(result.is_err());
    }

    #[test]
    fn test_prores_encode_frame() {
        let mut encoder = ProResEncoder::new(1920, 1080, ProResProfile::Standard).unwrap();
        let mut frame = VideoFrame::new(1920, 1080, PixelFormat::YUV420P);
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
