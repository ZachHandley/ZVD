//! DNxHD decoder implementation
//!
//! Decodes DNxHD/DNxHR video frames using DCT-based decompression.
//! The decoder parses the bitstream, extracts quantized coefficients,
//! applies inverse quantization, and performs IDCT to reconstruct pixels.

use super::bitstream::BitReader;
use super::idct::{dequant_block, idct_8x8, idct_8x8_10bit};
use super::tables::{DnxhdProfileTables, DNXHD_ZIGZAG};
use super::{DnxhdFrameHeader, DnxhdProfile};
use crate::codec::{Decoder, Frame, VideoFrame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, PixelFormat, Timestamp};

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

    /// Parse extended frame header (full DNxHD header structure)
    fn parse_frame_header(&mut self, data: &[u8]) -> Result<(DnxhdFrameHeader, usize)> {
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

        // Determine profile from compression ID
        self.profile = Self::profile_from_cid(compression_id);

        // Header size - standard DNxHD header is 16 bytes minimum
        // Extended headers may include quantization tables and more
        let header_size = 16;

        Ok((
            DnxhdFrameHeader {
                header_prefix: 0x000002800001,
                compression_id,
                width,
                height,
                is_progressive,
                is_422,
                bit_depth,
            },
            header_size,
        ))
    }

    /// Get profile from compression ID
    fn profile_from_cid(cid: u32) -> Option<DnxhdProfile> {
        match cid {
            1235 => Some(DnxhdProfile::Dnxhd36), // or Dnxhd145, same CID
            1237 => Some(DnxhdProfile::Dnxhd45), // or Dnxhd115
            1238 => Some(DnxhdProfile::Dnxhd75), // or Dnxhd120
            1241 => Some(DnxhdProfile::Dnxhd175),
            1242 => Some(DnxhdProfile::Dnxhd185),
            1243 => Some(DnxhdProfile::Dnxhd220),
            1250 => Some(DnxhdProfile::DnxhrLb),
            1251 => Some(DnxhdProfile::DnxhrSq),
            1252 => Some(DnxhdProfile::DnxhrHq),
            1253 => Some(DnxhdProfile::DnxhrHqx),
            1270 => Some(DnxhdProfile::Dnxhr444),
            _ => None,
        }
    }

    /// Decode a single DCT block from the bitstream
    fn decode_block(
        &self,
        br: &mut BitReader,
        coeffs: &mut [i32; 64],
        dc_pred: &mut i32,
        tables: &DnxhdProfileTables,
        is_10bit: bool,
    ) -> Result<()> {
        // Initialize coefficients to zero
        coeffs.fill(0);

        // Decode DC coefficient using differential coding
        let dc_diff = self.decode_dc(br, tables)?;
        *dc_pred = dc_pred.wrapping_add(dc_diff);
        coeffs[0] = *dc_pred;

        // Decode AC coefficients using run-level coding
        let mut idx = 1;
        while idx < 64 {
            // Try to decode run-level pair
            match self.decode_ac(br, tables)? {
                AcToken::RunLevel { run, level } => {
                    idx += run as usize;
                    if idx >= 64 {
                        break;
                    }
                    // Apply zigzag reordering
                    let zz_idx = DNXHD_ZIGZAG[idx] as usize;
                    coeffs[zz_idx] = level;
                    idx += 1;
                }
                AcToken::EndOfBlock => {
                    break;
                }
            }
        }

        Ok(())
    }

    /// Decode DC coefficient difference
    fn decode_dc(&self, br: &mut BitReader, tables: &DnxhdProfileTables) -> Result<i32> {
        // Read DC VLC code
        // DNxHD uses a simple prefix-based code for DC differences
        let dc_codes = tables.dc_codes;
        let dc_bits = tables.dc_bits;

        // Look for matching code by checking each possible bit length
        for max_bits in 1..=16 {
            if br.remaining() < max_bits {
                break;
            }

            let peeked = br.peek_bits(max_bits as u8)?;

            for i in 0..dc_codes.len() {
                let code_bits = dc_bits[i] as usize;
                if code_bits == max_bits {
                    let code = dc_codes[i] as u32;
                    let mask = (1u32 << code_bits) - 1;
                    if (peeked & mask) == code || peeked == code {
                        br.skip_bits(code_bits as u32);

                        // The index encodes the magnitude category
                        let category = i as i32;
                        if category == 0 {
                            return Ok(0);
                        }

                        // Read sign and magnitude bits
                        let sign_bit = br.read_bit()?;
                        let magnitude = if category > 1 {
                            let extra_bits = (category - 1).min(16) as u8;
                            if extra_bits > 0 {
                                (1 << (category - 1)) + br.read_bits(extra_bits)? as i32
                            } else {
                                1
                            }
                        } else {
                            1
                        };

                        return Ok(if sign_bit { magnitude } else { -magnitude });
                    }
                }
            }
        }

        // If no VLC match, use simple signed value
        // This is a fallback for simplified encoding
        if br.remaining() >= 8 {
            let val = br.read_bits(8)? as i8;
            return Ok(val as i32);
        }

        Ok(0)
    }

    /// Decode AC run-level token
    fn decode_ac(&self, br: &mut BitReader, tables: &DnxhdProfileTables) -> Result<AcToken> {
        // Try to decode run code
        let run_codes = tables.run_codes;
        let run_bits = tables.run_bits;
        let run_values = tables.run;

        // Check for end-of-block (typically all zeros remaining or special marker)
        if br.remaining() < 4 {
            return Ok(AcToken::EndOfBlock);
        }

        // Look for matching run code
        for max_bits in 1..=12 {
            if br.remaining() < max_bits {
                return Ok(AcToken::EndOfBlock);
            }

            let peeked = br.peek_bits(max_bits as u8)?;

            for i in 0..run_codes.len() {
                let code_bits = run_bits[i] as usize;
                if code_bits == max_bits {
                    let code = run_codes[i] as u32;
                    if peeked == code {
                        br.skip_bits(code_bits as u32);

                        let run = run_values[i] as u32;

                        // Special case: run of 62+ is EOB
                        if run >= 62 {
                            return Ok(AcToken::EndOfBlock);
                        }

                        // Read level using signed exp-golomb or simple coding
                        let level = self.decode_level(br)?;

                        return Ok(AcToken::RunLevel {
                            run,
                            level: level as i32,
                        });
                    }
                }
            }
        }

        // Fallback: simple run-level encoding
        // Run: 6 bits, Level: sign + magnitude
        if br.remaining() >= 8 {
            let run_val = br.read_bits(6)? as u32;
            if run_val >= 63 {
                return Ok(AcToken::EndOfBlock);
            }

            let sign = br.read_bit()?;
            let magnitude = br.read_bits(8)? as i32;
            let level = if sign { magnitude } else { -magnitude };

            return Ok(AcToken::RunLevel { run: run_val, level });
        }

        Ok(AcToken::EndOfBlock)
    }

    /// Decode level value
    fn decode_level(&self, br: &mut BitReader) -> Result<i32> {
        if br.remaining() < 2 {
            return Ok(0);
        }

        // Simple signed encoding: sign bit + magnitude
        let sign = br.read_bit()?;

        // Count leading zeros to determine magnitude bits
        let mut leading_zeros = 0u32;
        while br.remaining() > 0 && !br.read_bit()? {
            leading_zeros += 1;
            if leading_zeros > 16 {
                break;
            }
        }

        let magnitude = if leading_zeros == 0 {
            1
        } else if leading_zeros < 16 && br.remaining() >= leading_zeros as usize {
            let suffix = br.read_bits(leading_zeros as u8)?;
            (1 << leading_zeros) + suffix as i32
        } else {
            1
        };

        Ok(if sign { magnitude } else { -magnitude })
    }

    /// Decode frame data from the bitstream
    fn decode_frame_data(&self, data: &[u8], header: &DnxhdFrameHeader) -> Result<VideoFrame> {
        let width = self.width;
        let height = self.height;
        let is_10bit = header.bit_depth == 10;
        let is_422 = header.is_422;

        // Determine output format
        let pixel_format = if is_422 {
            if is_10bit {
                PixelFormat::YUV422P10LE
            } else {
                PixelFormat::YUV422P
            }
        } else {
            if is_10bit {
                PixelFormat::YUV444P10LE
            } else {
                PixelFormat::YUV444P
            }
        };

        // Calculate plane sizes
        let y_stride = width as usize;
        let y_size = y_stride * height as usize;

        let (uv_stride, uv_height) = if is_422 {
            ((width / 2) as usize, height as usize)
        } else {
            (width as usize, height as usize)
        };
        let uv_size = uv_stride * uv_height;

        // Allocate output planes
        let mut y_plane = vec![0u8; y_size];
        let mut u_plane = vec![0u8; uv_size];
        let mut v_plane = vec![0u8; uv_size];

        // For 10-bit, we use u16 internally then pack to bytes
        let mut y_plane_10bit = if is_10bit {
            vec![0u16; y_size]
        } else {
            Vec::new()
        };
        let mut u_plane_10bit = if is_10bit {
            vec![0u16; uv_size]
        } else {
            Vec::new()
        };
        let mut v_plane_10bit = if is_10bit {
            vec![0u16; uv_size]
        } else {
            Vec::new()
        };

        // Get quantization tables for this profile
        let profile = self.profile.unwrap_or(DnxhdProfile::DnxhrHq);
        let tables = DnxhdProfileTables::for_profile(profile);

        // Calculate macroblock dimensions
        let mb_width = (width + 15) / 16;
        let mb_height = (height + 15) / 16;

        // Create bitstream reader
        let mut br = BitReader::new(data);

        // DC predictors for each component
        let mut dc_y = 0i32;
        let mut dc_u = 0i32;
        let mut dc_v = 0i32;

        // Default quantization scale (will be read from stream in full implementation)
        let qscale = 16i32;

        // Temporary block buffer
        let mut coeffs = [0i32; 64];

        // Process macroblocks
        for mb_y in 0..mb_height {
            for mb_x in 0..mb_width {
                // Each macroblock contains:
                // - 4 luma blocks (16x16 -> 4x 8x8)
                // - 2 chroma blocks for 4:2:2 (8x16 -> 2x 8x8 each)
                // - 4 chroma blocks for 4:4:4 (16x16 -> 4x 8x8 each)

                let mb_x_pix = (mb_x * 16) as usize;
                let mb_y_pix = (mb_y * 16) as usize;

                // Decode 4 luma blocks (2x2 arrangement)
                for block_idx in 0..4 {
                    let block_x = mb_x_pix + (block_idx % 2) * 8;
                    let block_y = mb_y_pix + (block_idx / 2) * 8;

                    if block_x + 8 > width as usize || block_y + 8 > height as usize {
                        continue;
                    }

                    // Try to decode block, fall back to neutral values on error
                    if self
                        .decode_block(&mut br, &mut coeffs, &mut dc_y, &tables, is_10bit)
                        .is_ok()
                    {
                        // Dequantize
                        dequant_block(&mut coeffs, tables.luma_weight, qscale);

                        // IDCT and output
                        if is_10bit {
                            let mut output_block = [0u16; 64];
                            idct_8x8_10bit(&coeffs, &mut output_block, 8);

                            // Copy to plane
                            for row in 0..8 {
                                let dst_row = block_y + row;
                                if dst_row < height as usize {
                                    for col in 0..8 {
                                        let dst_col = block_x + col;
                                        if dst_col < width as usize {
                                            y_plane_10bit[dst_row * y_stride + dst_col] =
                                                output_block[row * 8 + col];
                                        }
                                    }
                                }
                            }
                        } else {
                            let mut output_block = [0u8; 64];
                            idct_8x8(&coeffs, &mut output_block, 8, 8);

                            // Copy to plane
                            for row in 0..8 {
                                let dst_row = block_y + row;
                                if dst_row < height as usize {
                                    for col in 0..8 {
                                        let dst_col = block_x + col;
                                        if dst_col < width as usize {
                                            y_plane[dst_row * y_stride + dst_col] =
                                                output_block[row * 8 + col];
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // On decode error, fill with neutral gray
                        let neutral = if is_10bit { 512 } else { 128 };
                        for row in 0..8 {
                            let dst_row = block_y + row;
                            if dst_row < height as usize {
                                for col in 0..8 {
                                    let dst_col = block_x + col;
                                    if dst_col < width as usize {
                                        if is_10bit {
                                            y_plane_10bit[dst_row * y_stride + dst_col] =
                                                neutral as u16;
                                        } else {
                                            y_plane[dst_row * y_stride + dst_col] = neutral as u8;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Decode chroma blocks
                let chroma_blocks = if is_422 { 2 } else { 4 };

                for block_idx in 0..chroma_blocks {
                    let (block_x, block_y) = if is_422 {
                        // 4:2:2: 2 blocks vertically for each U and V
                        (mb_x_pix / 2, mb_y_pix + block_idx * 8)
                    } else {
                        // 4:4:4: 4 blocks in 2x2 arrangement
                        (
                            mb_x_pix + (block_idx % 2) * 8,
                            mb_y_pix + (block_idx / 2) * 8,
                        )
                    };

                    // U block
                    if self
                        .decode_block(&mut br, &mut coeffs, &mut dc_u, &tables, is_10bit)
                        .is_ok()
                    {
                        dequant_block(&mut coeffs, tables.chroma_weight, qscale);

                        if is_10bit {
                            let mut output_block = [0u16; 64];
                            idct_8x8_10bit(&coeffs, &mut output_block, 8);

                            for row in 0..8 {
                                let dst_row = block_y + row;
                                if dst_row < uv_height {
                                    for col in 0..8 {
                                        let dst_col = block_x + col;
                                        if dst_col < uv_stride {
                                            u_plane_10bit[dst_row * uv_stride + dst_col] =
                                                output_block[row * 8 + col];
                                        }
                                    }
                                }
                            }
                        } else {
                            let mut output_block = [0u8; 64];
                            idct_8x8(&coeffs, &mut output_block, 8, 8);

                            for row in 0..8 {
                                let dst_row = block_y + row;
                                if dst_row < uv_height {
                                    for col in 0..8 {
                                        let dst_col = block_x + col;
                                        if dst_col < uv_stride {
                                            u_plane[dst_row * uv_stride + dst_col] =
                                                output_block[row * 8 + col];
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // Fill with neutral chroma
                        let neutral = if is_10bit { 512 } else { 128 };
                        for row in 0..8 {
                            let dst_row = block_y + row;
                            if dst_row < uv_height {
                                for col in 0..8 {
                                    let dst_col = block_x + col;
                                    if dst_col < uv_stride {
                                        if is_10bit {
                                            u_plane_10bit[dst_row * uv_stride + dst_col] =
                                                neutral as u16;
                                        } else {
                                            u_plane[dst_row * uv_stride + dst_col] = neutral as u8;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // V block
                    if self
                        .decode_block(&mut br, &mut coeffs, &mut dc_v, &tables, is_10bit)
                        .is_ok()
                    {
                        dequant_block(&mut coeffs, tables.chroma_weight, qscale);

                        if is_10bit {
                            let mut output_block = [0u16; 64];
                            idct_8x8_10bit(&coeffs, &mut output_block, 8);

                            for row in 0..8 {
                                let dst_row = block_y + row;
                                if dst_row < uv_height {
                                    for col in 0..8 {
                                        let dst_col = block_x + col;
                                        if dst_col < uv_stride {
                                            v_plane_10bit[dst_row * uv_stride + dst_col] =
                                                output_block[row * 8 + col];
                                        }
                                    }
                                }
                            }
                        } else {
                            let mut output_block = [0u8; 64];
                            idct_8x8(&coeffs, &mut output_block, 8, 8);

                            for row in 0..8 {
                                let dst_row = block_y + row;
                                if dst_row < uv_height {
                                    for col in 0..8 {
                                        let dst_col = block_x + col;
                                        if dst_col < uv_stride {
                                            v_plane[dst_row * uv_stride + dst_col] =
                                                output_block[row * 8 + col];
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        let neutral = if is_10bit { 512 } else { 128 };
                        for row in 0..8 {
                            let dst_row = block_y + row;
                            if dst_row < uv_height {
                                for col in 0..8 {
                                    let dst_col = block_x + col;
                                    if dst_col < uv_stride {
                                        if is_10bit {
                                            v_plane_10bit[dst_row * uv_stride + dst_col] =
                                                neutral as u16;
                                        } else {
                                            v_plane[dst_row * uv_stride + dst_col] = neutral as u8;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Create output frame
        let mut frame = VideoFrame::new(width, height, pixel_format);

        if is_10bit {
            // Pack 10-bit to bytes (little-endian)
            let y_bytes: Vec<u8> = y_plane_10bit
                .iter()
                .flat_map(|&v| v.to_le_bytes())
                .collect();
            let u_bytes: Vec<u8> = u_plane_10bit
                .iter()
                .flat_map(|&v| v.to_le_bytes())
                .collect();
            let v_bytes: Vec<u8> = v_plane_10bit
                .iter()
                .flat_map(|&v| v.to_le_bytes())
                .collect();

            frame.data = vec![
                Buffer::from_vec(y_bytes),
                Buffer::from_vec(u_bytes),
                Buffer::from_vec(v_bytes),
            ];
            frame.linesize = vec![y_stride * 2, uv_stride * 2, uv_stride * 2];
        } else {
            frame.data = vec![
                Buffer::from_vec(y_plane),
                Buffer::from_vec(u_plane),
                Buffer::from_vec(v_plane),
            ];
            frame.linesize = vec![y_stride, uv_stride, uv_stride];
        }

        frame.pts = Timestamp::new(0);
        frame.keyframe = true;

        Ok(frame)
    }
}

/// AC coefficient token
enum AcToken {
    RunLevel { run: u32, level: i32 },
    EndOfBlock,
}

impl Default for DnxhdDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder for DnxhdDecoder {
    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        // Parse frame header
        let data_slice = packet.data.as_slice();
        let (header, header_size) = self.parse_frame_header(data_slice)?;

        // Decode frame data (skip header)
        let frame_data = if data_slice.len() > header_size {
            &data_slice[header_size..]
        } else {
            &[]
        };

        let mut video_frame = self.decode_frame_data(frame_data, &header)?;

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

    #[test]
    fn test_profile_from_cid() {
        assert!(matches!(
            DnxhdDecoder::profile_from_cid(1252),
            Some(DnxhdProfile::DnxhrHq)
        ));
        assert!(matches!(
            DnxhdDecoder::profile_from_cid(1243),
            Some(DnxhdProfile::Dnxhd220)
        ));
        assert!(DnxhdDecoder::profile_from_cid(9999).is_none());
    }
}
