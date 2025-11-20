//! DNxHD decoder implementation - Pure Rust!

use super::{DnxhdProfile, DnxhdFrameHeader};
use super::bitstream::DnxhdBitstreamReader;
use super::data::CidData;
use super::dct::DnxhdDct;
use super::macroblock::{Macroblock, MacroblockProcessor};
use super::quant::DnxhdQuantizer;
use super::vlc::DnxhdVlcDecoder;
use crate::codec::{Decoder, Frame, VideoFrame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{PixelFormat, Timestamp};

/// DNxHD video decoder
pub struct DnxhdDecoder {
    width: u32,
    height: u32,
    profile: Option<DnxhdProfile>,
    pending_frame: Option<Frame>,
}

impl DnxhdDecoder {
    /// Create a new DNxHD decoder
    pub fn new() -> Self {
        DnxhdDecoder {
            width: 0,
            height: 0,
            profile: None,
            pending_frame: None,
        }
    }

    /// Parse frame header
    fn parse_frame_header(&mut self, data: &[u8]) -> Result<DnxhdFrameHeader> {
        if data.len() < 16 {
            return Err(Error::invalid_input("Insufficient data for DNxHD header"));
        }

        // Check header prefix (0x000002800001)
        if &data[0..6] != &[0x00, 0x00, 0x02, 0x80, 0x00, 0x01] {
            return Err(Error::invalid_input("Invalid DNxHD header prefix"));
        }

        // Compression ID
        let compression_id = u32::from_be_bytes([data[6], data[7], data[8], data[9]]);

        // Width and height
        let width = u16::from_be_bytes([data[10], data[11]]);
        let height = u16::from_be_bytes([data[12], data[13]]);

        self.width = width as u32;
        self.height = height as u32;

        // Flags
        let flags = if data.len() > 14 { data[14] } else { 0x03 };
        let is_progressive = (flags & 0x01) != 0;
        let is_422 = (flags & 0x02) != 0;

        // Bit depth
        let bit_depth = if data.len() > 15 { data[15] } else { 8 };

        Ok(DnxhdFrameHeader {
            header_prefix: 0x000002800001,
            compression_id,
            width,
            height,
            is_progressive,
            is_422,
            bit_depth,
        })
    }

    /// Decode frame data using our pure Rust implementation!
    fn decode_frame_data(&self, data: &[u8], header: &DnxhdFrameHeader) -> Result<VideoFrame> {
        let width = self.width as usize;
        let height = self.height as usize;

        // Determine profile from CID
        let profile = self.profile.unwrap_or(DnxhdProfile::DnxhrHq);
        let cid_data = CidData::for_profile(profile);

        // Determine pixel format
        let is_444 = !header.is_422;
        let pixel_format = if is_444 {
            if header.bit_depth == 10 {
                PixelFormat::YUV444P10LE
            } else {
                PixelFormat::YUV444P
            }
        } else {
            if header.bit_depth == 10 {
                PixelFormat::YUV422P10LE
            } else {
                PixelFormat::YUV422P
            }
        };

        // Create output frame
        let mut frame = VideoFrame::new(self.width, self.height, pixel_format);

        // Create macroblock processor
        let mb_processor = MacroblockProcessor::new(width, height, is_444);

        // Create bitstream reader
        let mut reader = DnxhdBitstreamReader::new(data);

        // Calculate macroblock dimensions
        let mb_width = (width + 15) / 16;
        let mb_height = (height + 15) / 16;

        // DC predictors (per component)
        let mut last_dc_y = 1 << (header.bit_depth + 2);
        let mut last_dc_cb = 1 << (header.bit_depth + 2);
        let mut last_dc_cr = 1 << (header.bit_depth + 2);

        // Decode macroblocks
        for mb_y in 0..mb_height {
            for mb_x in 0..mb_width {
                // Read quantization scale (11 bits)
                let qscale = reader.read_bits(11)? as u16;

                // Create quantizer and VLC decoder
                let quantizer = DnxhdQuantizer::new(cid_data, qscale);
                let vlc_decoder = DnxhdVlcDecoder::new(cid_data);

                // Create macroblock
                let mut mb = if is_444 {
                    Macroblock::new_444()
                } else {
                    Macroblock::new_422()
                };
                mb.qscale = qscale;

                // Decode Y blocks
                for block_idx in 0..mb.y_blocks.len() {
                    let mut quant_coeffs = [0i16; 64];

                    // Decode DC with prediction
                    let dc_diff = vlc_decoder.decode_dc(&mut reader)?;
                    last_dc_y += dc_diff as i32;
                    quant_coeffs[0] = last_dc_y as i16;

                    // Decode AC coefficients
                    vlc_decoder.decode_ac(&mut reader, &mut quant_coeffs)?;

                    // Dequantize
                    let mut dct_coeffs = [0i16; 64];
                    quantizer.dequantize_luma(&quant_coeffs, &mut dct_coeffs)?;

                    // IDCT
                    DnxhdDct::inverse_dct(&dct_coeffs, &mut mb.y_blocks[block_idx])?;
                }

                // Decode Cb blocks
                for block_idx in 0..mb.cb_blocks.len() {
                    let mut quant_coeffs = [0i16; 64];

                    // Decode DC with prediction
                    let dc_diff = vlc_decoder.decode_dc(&mut reader)?;
                    last_dc_cb += dc_diff as i32;
                    quant_coeffs[0] = last_dc_cb as i16;

                    // Decode AC coefficients
                    vlc_decoder.decode_ac(&mut reader, &mut quant_coeffs)?;

                    // Dequantize
                    let mut dct_coeffs = [0i16; 64];
                    quantizer.dequantize_chroma(&quant_coeffs, &mut dct_coeffs)?;

                    // IDCT
                    DnxhdDct::inverse_dct(&dct_coeffs, &mut mb.cb_blocks[block_idx])?;
                }

                // Decode Cr blocks
                for block_idx in 0..mb.cr_blocks.len() {
                    let mut quant_coeffs = [0i16; 64];

                    // Decode DC with prediction
                    let dc_diff = vlc_decoder.decode_dc(&mut reader)?;
                    last_dc_cr += dc_diff as i32;
                    quant_coeffs[0] = last_dc_cr as i16;

                    // Decode AC coefficients
                    vlc_decoder.decode_ac(&mut reader, &mut quant_coeffs)?;

                    // Dequantize
                    let mut dct_coeffs = [0i16; 64];
                    quantizer.dequantize_chroma(&quant_coeffs, &mut dct_coeffs)?;

                    // IDCT
                    DnxhdDct::inverse_dct(&dct_coeffs, &mut mb.cr_blocks[block_idx])?;
                }

                // Insert macroblock into frame
                mb_processor.insert_macroblock(
                    &mb,
                    &mut frame.y_plane,
                    &mut frame.u_plane,
                    &mut frame.v_plane,
                    mb_x,
                    mb_y,
                )?;
            }
        }

        frame.pts = Timestamp::new(0);
        Ok(frame)
    }
}

impl Default for DnxhdDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder for DnxhdDecoder {
    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        // Parse frame header
        let header = self.parse_frame_header(packet.data.as_slice())?;

        // Decode frame data (skip header)
        let data_slice = packet.data.as_slice();
        let mut video_frame = self.decode_frame_data(&data_slice[16..], &header)?;

        video_frame.pts = packet.pts;

        self.pending_frame = Some(Frame::Video(video_frame));

        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        self.pending_frame.take().ok_or_else(|| Error::TryAgain)
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
    fn test_dnxhd_decoder_creation() {
        let decoder = DnxhdDecoder::new();
        assert_eq!(decoder.width, 0);
        assert_eq!(decoder.height, 0);
    }
}
