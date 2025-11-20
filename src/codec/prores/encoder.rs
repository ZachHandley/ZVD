//! ProRes encoder implementation - Pure Rust!

use super::{ProResProfile, ProResFrameHeader};
use super::bitstream::ProResBitstreamWriter;
use super::slice::{Slice, SliceEncoder};
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
    qp: u8,
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
            qp: 16, // Default QP
            pending_packet: None,
        })
    }

    /// Set quantization parameter (0-255)
    pub fn set_qp(&mut self, qp: u8) {
        self.qp = qp;
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

    /// Encode frame data using our pure Rust implementation!
    fn encode_frame_data(&self, frame: &VideoFrame) -> Result<Vec<u8>> {
        let width = self.width as usize;
        let height = self.height as usize;

        // Determine chroma subsampling based on profile
        let is_444 = self.profile.has_alpha(); // 4444 profiles use 4:4:4
        let chroma_width = if is_444 { width } else { width / 2 };

        // Convert Y plane to i16 (subtract 128 for signed values)
        let mut y_plane = vec![0i16; width * height];
        for i in 0..y_plane.len().min(frame.y_plane.len()) {
            y_plane[i] = frame.y_plane[i] as i16 - 128;
        }

        // Convert U plane to i16 with subsampling
        let mut u_plane = vec![0i16; chroma_width * height];
        self.subsample_chroma(&frame.u_plane, &mut u_plane, width, height, is_444)?;

        // Convert V plane to i16 with subsampling
        let mut v_plane = vec![0i16; chroma_width * height];
        self.subsample_chroma(&frame.v_plane, &mut v_plane, width, height, is_444)?;

        // Encode slices (16 lines per slice)
        let slice_height = 16;
        let mut writer = ProResBitstreamWriter::new();

        for y in (0..height).step_by(slice_height) {
            let actual_height = slice_height.min(height - y);

            // Encode Y plane slice
            let y_encoder = SliceEncoder::new(self.profile, width, height, self.qp);
            let mut y_slice = y_encoder.encode_slice(&y_plane, y, actual_height)?;
            y_slice.encode(&mut writer);

            // Encode U plane slice
            let u_encoder = SliceEncoder::new(self.profile, chroma_width, height, self.qp);
            let mut u_slice = u_encoder.encode_slice(&u_plane, y, actual_height)?;
            u_slice.encode(&mut writer);

            // Encode V plane slice
            let v_encoder = SliceEncoder::new(self.profile, chroma_width, height, self.qp);
            let mut v_slice = v_encoder.encode_slice(&v_plane, y, actual_height)?;
            v_slice.encode(&mut writer);
        }

        Ok(writer.finish())
    }

    /// Subsample chroma plane from input frame
    fn subsample_chroma(
        &self,
        input: &[u8],
        output: &mut [i16],
        width: usize,
        height: usize,
        is_444: bool,
    ) -> Result<()> {
        if is_444 {
            // 4:4:4 - no subsampling, same resolution as Y
            for i in 0..output.len().min(input.len()) {
                output[i] = input[i] as i16 - 128;
            }
        } else {
            // 4:2:2 - horizontal subsampling (half width)
            let chroma_width = width / 2;
            for y in 0..height {
                for x in 0..chroma_width {
                    // Average two horizontal pixels
                    let x_src = x * 2;
                    let idx_src1 = y * width + x_src;
                    let idx_src2 = idx_src1 + 1;
                    let idx_dst = y * chroma_width + x;

                    if idx_src2 < input.len() && idx_dst < output.len() {
                        let avg = (input[idx_src1] as i16 + input[idx_src2] as i16) / 2;
                        output[idx_dst] = avg - 128;
                    } else if idx_src1 < input.len() && idx_dst < output.len() {
                        output[idx_dst] = input[idx_src1] as i16 - 128;
                    }
                }
            }
        }
        Ok(())
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
    fn test_prores_encoder_set_qp() {
        let mut encoder = ProResEncoder::new(1920, 1080, ProResProfile::Standard).unwrap();
        encoder.set_qp(20);
        assert_eq!(encoder.qp, 20);
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

    #[test]
    fn test_prores_encoder_all_profiles() {
        let profiles = vec![
            ProResProfile::Proxy,
            ProResProfile::Lt,
            ProResProfile::Standard,
            ProResProfile::Hq,
            ProResProfile::FourFourFourFour,
            ProResProfile::FourFourFourFourXq,
        ];

        for profile in profiles {
            let encoder = ProResEncoder::new(320, 240, profile);
            assert!(encoder.is_ok(), "Failed to create encoder for {:?}", profile);
        }
    }

    #[test]
    fn test_prores_encode_small_frame() {
        let mut encoder = ProResEncoder::new(16, 16, ProResProfile::Standard).unwrap();
        let mut frame = VideoFrame::new(16, 16, PixelFormat::YUV420P);

        // Fill with test pattern
        for i in 0..frame.y_plane.len() {
            frame.y_plane[i] = (i % 256) as u8;
        }

        frame.pts = Timestamp::new(0);

        let send_result = encoder.send_frame(&Frame::Video(frame));
        assert!(send_result.is_ok());

        let packet = encoder.receive_packet().unwrap();
        assert!(packet.data.len() > 148); // Header + some data
    }
}
