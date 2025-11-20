//! DNxHD encoder implementation - Pure Rust!

use super::{DnxhdProfile, DnxhdFrameHeader};
use super::bitstream::DnxhdBitstreamWriter;
use super::data::CidData;
use super::dct::DnxhdDct;
use super::macroblock::{Macroblock, MacroblockProcessor};
use super::quant::DnxhdQuantizer;
use super::vlc::DnxhdVlcEncoder;
use crate::codec::{Encoder, Frame, VideoFrame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, Timestamp};

/// DNxHD video encoder
pub struct DnxhdEncoder {
    profile: DnxhdProfile,
    width: u32,
    height: u32,
    frame_count: u64,
    pending_packet: Option<Packet>,
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
            pending_packet: None,
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

    /// Encode frame data using our pure Rust implementation!
    fn encode_frame_data(&self, frame: &VideoFrame) -> Result<Vec<u8>> {
        let width = self.width as usize;
        let height = self.height as usize;

        // Get CID data for this profile
        let cid_data = CidData::for_profile(self.profile);
        let is_444 = cid_data.is_444;

        // Default quantization scale (can be adjusted for rate control)
        let qscale = 1024u16;

        // Create macroblock processor
        let mb_processor = MacroblockProcessor::new(width, height, is_444);

        // Create bitstream writer
        let mut writer = DnxhdBitstreamWriter::new();

        // Calculate macroblock dimensions
        let mb_width = (width + 15) / 16;
        let mb_height = (height + 15) / 16;

        // DC predictors (per component)
        let mut last_dc_y = 0i32;
        let mut last_dc_cb = 0i32;
        let mut last_dc_cr = 0i32;

        // Encode macroblocks
        for mb_y in 0..mb_height {
            for mb_x in 0..mb_width {
                // Extract macroblock from frame
                let mb = mb_processor.extract_macroblock(
                    &frame.y_plane,
                    &frame.u_plane,
                    &frame.v_plane,
                    mb_x,
                    mb_y,
                )?;

                // Write quantization scale (11 bits)
                writer.write_bits(qscale as u32, 11);

                // Create quantizer and VLC encoder
                let quantizer = DnxhdQuantizer::new(cid_data, qscale);
                let vlc_encoder = DnxhdVlcEncoder::new(cid_data);

                // Encode Y blocks
                for block_idx in 0..mb.y_blocks.len() {
                    // DCT
                    let mut dct_coeffs = [0i16; 64];
                    DnxhdDct::forward_dct(&mb.y_blocks[block_idx], &mut dct_coeffs)?;

                    // Quantize
                    let mut quant_coeffs = [0i16; 64];
                    quantizer.quantize_luma(&dct_coeffs, &mut quant_coeffs)?;

                    // DC prediction
                    let dc_value = quant_coeffs[0] as i32;
                    let dc_diff = dc_value - last_dc_y;
                    last_dc_y = dc_value;

                    // Encode DC
                    vlc_encoder.encode_dc(&mut writer, dc_diff as i16)?;

                    // Encode AC
                    vlc_encoder.encode_ac(&mut writer, &quant_coeffs)?;
                }

                // Encode Cb blocks
                for block_idx in 0..mb.cb_blocks.len() {
                    // DCT
                    let mut dct_coeffs = [0i16; 64];
                    DnxhdDct::forward_dct(&mb.cb_blocks[block_idx], &mut dct_coeffs)?;

                    // Quantize
                    let mut quant_coeffs = [0i16; 64];
                    quantizer.quantize_chroma(&dct_coeffs, &mut quant_coeffs)?;

                    // DC prediction
                    let dc_value = quant_coeffs[0] as i32;
                    let dc_diff = dc_value - last_dc_cb;
                    last_dc_cb = dc_value;

                    // Encode DC
                    vlc_encoder.encode_dc(&mut writer, dc_diff as i16)?;

                    // Encode AC
                    vlc_encoder.encode_ac(&mut writer, &quant_coeffs)?;
                }

                // Encode Cr blocks
                for block_idx in 0..mb.cr_blocks.len() {
                    // DCT
                    let mut dct_coeffs = [0i16; 64];
                    DnxhdDct::forward_dct(&mb.cr_blocks[block_idx], &mut dct_coeffs)?;

                    // Quantize
                    let mut quant_coeffs = [0i16; 64];
                    quantizer.quantize_chroma(&dct_coeffs, &mut quant_coeffs)?;

                    // DC prediction
                    let dc_value = quant_coeffs[0] as i32;
                    let dc_diff = dc_value - last_dc_cr;
                    last_dc_cr = dc_value;

                    // Encode DC
                    vlc_encoder.encode_dc(&mut writer, dc_diff as i16)?;

                    // Encode AC
                    vlc_encoder.encode_ac(&mut writer, &quant_coeffs)?;
                }
            }
        }

        Ok(writer.finish())
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
