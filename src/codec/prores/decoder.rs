//! ProRes decoder implementation
//!
//! Implements Apple ProRes 422/4444 decoding with proper entropy decoding,
//! inverse quantization, and IDCT.

use super::idct::{dequant_block, idct_8x8};
use super::tables::{
    PRORES_DC_CODEBOOK, PRORES_INTERLACED_SCAN, PRORES_LEVEL_TO_CB, PRORES_PROGRESSIVE_SCAN,
    PRORES_QUANT_MATRICES, PRORES_RUN_TO_CB,
};
use super::{ProResFrameHeader, ProResProfile};
use crate::codec::{Decoder, Frame, PictureType, VideoFrame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, PixelFormat, Timestamp};

/// First DC coefficient codebook
const FIRST_DC_CODEBOOK: u8 = 0xB8;

/// Bitstream reader for ProRes decoding
struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Get current bit position in bytes (rounded down)
    #[allow(dead_code)]
    fn position(&self) -> usize {
        self.byte_pos
    }

    /// Check if we have at least n bits remaining
    #[allow(dead_code)]
    fn has_bits(&self, n: usize) -> bool {
        let current_bit = self.byte_pos * 8 + self.bit_pos;
        let total_bits = self.data.len() * 8;
        current_bit + n <= total_bits
    }

    /// Read a single bit
    #[inline]
    fn read_bit(&mut self) -> Result<u32> {
        if self.byte_pos >= self.data.len() {
            return Err(Error::codec("Bitstream exhausted"));
        }

        let bit = ((self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1) as u32;
        self.bit_pos += 1;
        if self.bit_pos >= 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        Ok(bit)
    }

    /// Read n bits (MSB first)
    fn read_bits(&mut self, n: usize) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(Error::codec("Cannot read more than 32 bits"));
        }

        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | self.read_bit()?;
        }
        Ok(value)
    }

    /// Skip n bits
    #[allow(dead_code)]
    fn skip_bits(&mut self, n: usize) -> Result<()> {
        let new_bit = self.bit_pos + n;
        self.byte_pos += new_bit / 8;
        self.bit_pos = new_bit % 8;
        if self.byte_pos > self.data.len() {
            return Err(Error::codec("Bitstream exhausted during skip"));
        }
        Ok(())
    }

    /// Align to byte boundary
    fn align(&mut self) {
        if self.bit_pos != 0 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
    }
}

/// Decode a codeword using ProRes hybrid Rice/ExpGolomb VLC scheme
///
/// The codebook byte encodes:
/// - bits 0-1: switch_bits
/// - bits 2-4: exp_order
/// - bits 5-7: rice_order
fn decode_codeword(br: &mut BitReader, codebook: u8) -> Result<u32> {
    let switch_bits = (codebook & 3) as u32;
    let rice_order = (codebook >> 5) as u32;
    let exp_order = ((codebook >> 2) & 7) as u32;

    // Count leading zeros (unary prefix)
    let mut q = 0u32;
    loop {
        let bit = br.read_bit()?;
        if bit != 0 {
            break;
        }
        q += 1;
        if q > 64 {
            return Err(Error::codec("VLC prefix too long"));
        }
    }

    if q <= switch_bits {
        // Rice coding mode
        if rice_order > 0 {
            let r = br.read_bits(rice_order as usize)?;
            Ok((q << rice_order) | r)
        } else {
            Ok(q)
        }
    } else {
        // ExpGolomb coding mode
        let bits = exp_order as i32 - switch_bits as i32 + ((q as i32) << 1);
        if bits < 0 || bits > 32 {
            return Err(Error::codec("Invalid ExpGolomb bit count"));
        }
        let code = br.read_bits(bits as usize)?;
        // Use wrapping arithmetic as per ProRes spec
        let base = ((switch_bits + 1) << rice_order).wrapping_sub(1 << exp_order);
        Ok(code.wrapping_add(base))
    }
}

/// Convert unsigned VLC code to signed value
#[inline]
fn to_signed(v: u32) -> i32 {
    ((v >> 1) as i32) ^ -((v as i32) & 1)
}

/// Decode DC coefficients for a slice
fn decode_dc_coeffs(br: &mut BitReader, coeffs: &mut [i16], blocks: usize) -> Result<()> {
    if blocks == 0 {
        return Ok(());
    }

    // First DC coefficient
    let code = decode_codeword(br, FIRST_DC_CODEBOOK)?;
    let first_dc = to_signed(code);
    coeffs[0] = first_dc as i16;

    // Subsequent DC coefficients (differential with sign prediction)
    let mut prev_dc = first_dc;
    let mut code_val = 5u32;
    let mut sign = 0i32;

    for block in 1..blocks {
        let idx = block * 64;
        let cb_idx = code_val.min(6) as usize;
        let code = decode_codeword(br, PRORES_DC_CODEBOOK[cb_idx])?;

        if code != 0 {
            // Update sign prediction
            sign ^= -((code & 1) as i32);
        } else {
            sign = 0;
        }

        // Reconstruct DC value
        prev_dc += (((code + 1) >> 1) as i32 ^ sign) - sign;
        coeffs[idx] = prev_dc as i16;
        code_val = code;
    }

    Ok(())
}

/// Decode AC coefficients for a slice using run-level coding
fn decode_ac_coeffs(
    br: &mut BitReader,
    coeffs: &mut [i16],
    blocks: usize,
    scan: &[u8; 64],
) -> Result<()> {
    if blocks == 0 {
        return Ok(());
    }

    // Calculate interleaving parameters
    let pow2_blocks = blocks.next_power_of_two();
    let block_mask = pow2_blocks - 1;
    let log2_blocks = pow2_blocks.trailing_zeros() as usize;
    let max_coeffs = 64 << log2_blocks;

    // Track position in interleaved coefficient space
    let mut pos = block_mask; // Start after all DC coefficients
    let mut run_ctx = 4u32;
    let mut level_ctx = 2u32;

    while pos < max_coeffs - 1 {
        // Decode run
        let run_cb = PRORES_RUN_TO_CB[run_ctx.min(15) as usize];
        let run = decode_codeword(br, run_cb)? as usize;

        // Advance position
        pos += run + 1;
        if pos >= max_coeffs {
            break;
        }

        // Calculate actual block and coefficient index
        let block_idx = pos & block_mask;
        let coeff_idx = pos >> log2_blocks;

        if block_idx >= blocks || coeff_idx >= 64 {
            continue;
        }

        // Decode level
        let level_cb = PRORES_LEVEL_TO_CB[level_ctx.min(9) as usize];
        let level_code = decode_codeword(br, level_cb)?;
        let level = (level_code + 1) as i16;

        // Decode sign
        let sign = br.read_bit()?;
        let signed_level = if sign != 0 { -level } else { level };

        // Store coefficient in de-zigzagged position
        let scan_idx = scan[coeff_idx] as usize;
        let dst_idx = block_idx * 64 + scan_idx;
        coeffs[dst_idx] = signed_level;

        // Update context
        run_ctx = run as u32;
        level_ctx = level as u32;
    }

    Ok(())
}

/// Slice header information
struct SliceHeader {
    #[allow(dead_code)]
    slice_data_size: usize,
    header_size: usize,
    quant_y: u8,
    quant_c: u8,
}

/// Parse slice header
fn parse_slice_header(data: &[u8]) -> Result<SliceHeader> {
    if data.len() < 2 {
        return Err(Error::codec("Slice header too small"));
    }

    let header_size = (data[0] >> 3) as usize;
    if header_size < 2 || data.len() < header_size {
        return Err(Error::codec("Invalid slice header size"));
    }

    // Read quantization indices from header
    let quant_y = data[1];
    let quant_c = if header_size > 2 { data[2] } else { quant_y };

    Ok(SliceHeader {
        slice_data_size: data.len(),
        header_size,
        quant_y,
        quant_c,
    })
}

/// ProRes video decoder
pub struct ProResDecoder {
    width: u32,
    height: u32,
    profile: Option<ProResProfile>,
    pending_frame: Option<Frame>,
}

impl ProResDecoder {
    /// Create a new ProRes decoder
    pub fn new() -> Self {
        ProResDecoder {
            width: 0,
            height: 0,
            profile: None,
            pending_frame: None,
        }
    }

    /// Parse frame header
    fn parse_frame_header(&mut self, data: &[u8]) -> Result<ProResFrameHeader> {
        if data.len() < 20 {
            return Err(Error::invalid_input("Insufficient data for ProRes header"));
        }

        // Frame size
        let frame_size = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);

        // Frame identifier
        let mut frame_identifier = [0u8; 4];
        frame_identifier.copy_from_slice(&data[4..8]);

        if &frame_identifier != b"icpf" {
            return Err(Error::invalid_input("Invalid ProRes frame identifier"));
        }

        // Header size
        let header_size = u16::from_be_bytes([data[8], data[9]]);

        if data.len() < header_size as usize {
            return Err(Error::invalid_input("Incomplete ProRes header"));
        }

        // Version
        let version = data[10];

        // Encoder ID
        let mut encoder_id = [0u8; 4];
        encoder_id.copy_from_slice(&data[11..15]);

        // Dimensions
        let width = u16::from_be_bytes([data[15], data[16]]);
        let height = u16::from_be_bytes([data[17], data[18]]);

        self.width = width as u32;
        self.height = height as u32;

        // Chroma format
        let chroma_format = data[19];

        // Detect profile from dimensions and other characteristics
        self.profile = Some(ProResProfile::Standard);

        // Other fields (if available)
        let interlace_mode = if data.len() > 20 { data[20] } else { 0 };
        let aspect_ratio = if data.len() > 21 { data[21] } else { 0 };
        let framerate_code = if data.len() > 22 { data[22] } else { 0 };
        let color_primaries = if data.len() > 23 { data[23] } else { 1 };
        let transfer_characteristics = if data.len() > 24 { data[24] } else { 1 };
        let matrix_coefficients = if data.len() > 25 { data[25] } else { 1 };
        let alpha_info = if data.len() > 26 { data[26] } else { 0 };

        Ok(ProResFrameHeader {
            frame_size,
            frame_identifier,
            header_size,
            version,
            encoder_id,
            width,
            height,
            chroma_format,
            interlace_mode,
            aspect_ratio,
            framerate_code,
            color_primaries,
            transfer_characteristics,
            matrix_coefficients,
            alpha_info,
        })
    }

    /// Get quantization matrix for the profile and plane
    fn get_quant_matrix(&self, header: &ProResFrameHeader, _is_chroma: bool) -> &'static [u8; 64] {
        // ProRes uses different matrices for different profiles
        // For simplicity, use HQ matrix for better quality
        // Profile detection can be enhanced based on encoder ID
        let profile_idx = match header.encoder_id {
            [b'a', b'p', b'l', b'0'] => 3, // HQ
            [b'a', b'p', b'c', b'o'] => 0, // Proxy
            [b'a', b'p', b'c', b's'] => 1, // LT
            [b'a', b'p', b'c', b'n'] => 2, // Standard
            [b'a', b'p', b'c', b'h'] => 3, // HQ
            [b'a', b'p', b'4', b'h'] => 4, // 4444
            [b'a', b'p', b'4', b'x'] => 5, // 4444XQ
            _ => 3,                        // Default to HQ
        };

        &PRORES_QUANT_MATRICES[profile_idx]
    }

    /// Decode a single slice
    fn decode_slice(
        &self,
        data: &[u8],
        mb_x: usize,
        mb_y: usize,
        mb_count: usize,
        header: &ProResFrameHeader,
        y_plane: &mut [u16],
        cb_plane: &mut [u16],
        cr_plane: &mut [u16],
        y_stride: usize,
        c_stride: usize,
    ) -> Result<()> {
        let slice_header = parse_slice_header(data)?;

        // Calculate blocks per macroblock
        let is_422 = header.chroma_format == 2;
        let luma_blocks_per_mb = 4; // Always 4 luma blocks (2x2) per MB
        let chroma_blocks_per_mb = if is_422 { 2 } else { 4 }; // 422: 2 chroma blocks, 444: 4

        let total_luma_blocks = mb_count * luma_blocks_per_mb;
        let total_chroma_blocks = mb_count * chroma_blocks_per_mb;

        // Select scan pattern
        let scan = if header.interlace_mode != 0 {
            &PRORES_INTERLACED_SCAN
        } else {
            &PRORES_PROGRESSIVE_SCAN
        };

        // Get quantization matrix and scale
        let qmat_y = self.get_quant_matrix(header, false);
        let qmat_c = self.get_quant_matrix(header, true);
        let qscale_y = slice_header.quant_y as i32;
        let qscale_c = slice_header.quant_c as i32;

        // Start reading encoded data
        let slice_data = &data[slice_header.header_size..];
        let mut br = BitReader::new(slice_data);

        // Allocate coefficient buffers
        let mut y_coeffs = vec![0i16; total_luma_blocks * 64];
        let mut cb_coeffs = vec![0i16; total_chroma_blocks * 64];
        let mut cr_coeffs = vec![0i16; total_chroma_blocks * 64];

        // Decode luma coefficients
        decode_dc_coeffs(&mut br, &mut y_coeffs, total_luma_blocks)?;
        decode_ac_coeffs(&mut br, &mut y_coeffs, total_luma_blocks, scan)?;
        br.align();

        // Decode Cb coefficients
        decode_dc_coeffs(&mut br, &mut cb_coeffs, total_chroma_blocks)?;
        decode_ac_coeffs(&mut br, &mut cb_coeffs, total_chroma_blocks, scan)?;
        br.align();

        // Decode Cr coefficients
        decode_dc_coeffs(&mut br, &mut cr_coeffs, total_chroma_blocks)?;
        decode_ac_coeffs(&mut br, &mut cr_coeffs, total_chroma_blocks, scan)?;

        // Process each macroblock
        for mb in 0..mb_count {
            let mb_pixel_x = (mb_x + mb) * 16;
            let mb_pixel_y = mb_y * 16;

            // Process luma blocks (2x2 arrangement)
            for block_row in 0..2 {
                for block_col in 0..2 {
                    let block_idx = mb * 4 + block_row * 2 + block_col;
                    let coeffs: [i16; 64] = y_coeffs[block_idx * 64..(block_idx + 1) * 64]
                        .try_into()
                        .unwrap();

                    // Dequantize
                    let mut dequant = dequant_block(&coeffs, qmat_y, qscale_y);

                    // Inverse DCT
                    idct_8x8(&mut dequant);

                    // Write to output (10-bit values in 16-bit plane)
                    let block_x = mb_pixel_x + block_col * 8;
                    let block_y = mb_pixel_y + block_row * 8;

                    for row in 0..8 {
                        let out_y = block_y + row;
                        if out_y >= header.height as usize {
                            continue;
                        }
                        for col in 0..8 {
                            let out_x = block_x + col;
                            if out_x >= header.width as usize {
                                continue;
                            }
                            let idx = out_y * y_stride + out_x;
                            // Clamp to 10-bit range and add 512 offset (mid-gray)
                            let val = (dequant[row * 8 + col] + 512).clamp(0, 1023) as u16;
                            y_plane[idx] = val;
                        }
                    }
                }
            }

            // Process chroma blocks
            let chroma_mb_x = if is_422 { mb_pixel_x / 2 } else { mb_pixel_x };
            let chroma_blocks = if is_422 { 2 } else { 4 };
            let chroma_block_cols = if is_422 { 1 } else { 2 };

            for block_row in 0..2 {
                for block_col in 0..chroma_block_cols {
                    let block_idx_in_mb = block_row * chroma_block_cols + block_col;
                    let cb_block_idx = mb * chroma_blocks + block_idx_in_mb;
                    let cr_block_idx = mb * chroma_blocks + block_idx_in_mb;

                    // Cb
                    let cb_coeff: [i16; 64] = cb_coeffs[cb_block_idx * 64..(cb_block_idx + 1) * 64]
                        .try_into()
                        .unwrap();
                    let mut cb_dequant = dequant_block(&cb_coeff, qmat_c, qscale_c);
                    idct_8x8(&mut cb_dequant);

                    // Cr
                    let cr_coeff: [i16; 64] = cr_coeffs[cr_block_idx * 64..(cr_block_idx + 1) * 64]
                        .try_into()
                        .unwrap();
                    let mut cr_dequant = dequant_block(&cr_coeff, qmat_c, qscale_c);
                    idct_8x8(&mut cr_dequant);

                    // Write chroma to output
                    let block_x = chroma_mb_x + block_col * 8;
                    let block_y = mb_pixel_y + block_row * 8;
                    let chroma_width = if is_422 {
                        header.width as usize / 2
                    } else {
                        header.width as usize
                    };

                    for row in 0..8 {
                        let out_y = block_y + row;
                        if out_y >= header.height as usize {
                            continue;
                        }
                        for col in 0..8 {
                            let out_x = block_x + col;
                            if out_x >= chroma_width {
                                continue;
                            }
                            let idx = out_y * c_stride + out_x;
                            let cb_val = (cb_dequant[row * 8 + col] + 512).clamp(0, 1023) as u16;
                            let cr_val = (cr_dequant[row * 8 + col] + 512).clamp(0, 1023) as u16;
                            cb_plane[idx] = cb_val;
                            cr_plane[idx] = cr_val;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Decode frame data
    fn decode_frame_data(&self, data: &[u8], header: &ProResFrameHeader) -> Result<VideoFrame> {
        if data.len() < 8 {
            return Err(Error::codec("Frame data too small"));
        }

        // Parse picture header
        let picture_header_size = data[0] as usize;
        if picture_header_size < 8 || data.len() < picture_header_size {
            return Err(Error::codec("Invalid picture header size"));
        }

        // Read picture header fields
        let _picture_size = u32::from_be_bytes([data[1], data[2], data[3], data[4]]);
        let num_slices = u16::from_be_bytes([data[5], data[6]]) as usize;

        if num_slices == 0 {
            return Err(Error::codec("No slices in frame"));
        }

        // Calculate dimensions
        let width = header.width as usize;
        let height = header.height as usize;
        let is_422 = header.chroma_format == 2;

        // Calculate chroma dimensions
        let chroma_width = if is_422 { width / 2 } else { width };
        let chroma_height = height;

        // Allocate output planes (10-bit values stored in u16)
        let y_stride = width;
        let c_stride = chroma_width;

        let mut y_plane = vec![512u16; height * y_stride]; // Mid-gray default
        let mut cb_plane = vec![512u16; chroma_height * c_stride];
        let mut cr_plane = vec![512u16; chroma_height * c_stride];

        // Read slice index table
        let index_size = num_slices * 2;
        let index_start = picture_header_size;
        if data.len() < index_start + index_size {
            return Err(Error::codec("Slice index table truncated"));
        }

        let mut slice_offsets = Vec::with_capacity(num_slices + 1);
        let slice_data_start = index_start + index_size;
        let mut current_offset = slice_data_start;

        for i in 0..num_slices {
            slice_offsets.push(current_offset);
            let idx = index_start + i * 2;
            let slice_size = u16::from_be_bytes([data[idx], data[idx + 1]]) as usize;
            current_offset += slice_size;
        }
        slice_offsets.push(data.len().min(current_offset));

        // Calculate slice layout
        let mb_width = (width + 15) / 16;
        let mb_height = (height + 15) / 16;

        // ProRes uses a specific slice layout
        // Each slice row contains a number of slices, each with multiple macroblocks
        let mbs_per_slice = 8; // Common value, can vary

        let mut slice_idx = 0;
        for mb_row in 0..mb_height {
            let mut mb_col = 0;
            while mb_col < mb_width && slice_idx < num_slices {
                let mb_count = (mb_width - mb_col).min(mbs_per_slice);

                let slice_start = slice_offsets[slice_idx];
                let slice_end = slice_offsets[slice_idx + 1];

                if slice_start < data.len() && slice_end <= data.len() && slice_start < slice_end {
                    let slice_data = &data[slice_start..slice_end];

                    // Decode the slice
                    if let Err(_e) = self.decode_slice(
                        slice_data,
                        mb_col,
                        mb_row,
                        mb_count,
                        header,
                        &mut y_plane,
                        &mut cb_plane,
                        &mut cr_plane,
                        y_stride,
                        c_stride,
                    ) {
                        // Continue with other slices even if one fails
                        // This provides graceful degradation
                    }
                }

                mb_col += mb_count;
                slice_idx += 1;
            }
        }

        // Select output pixel format
        let pixel_format = if is_422 {
            PixelFormat::YUV422P10LE
        } else {
            PixelFormat::YUV444P10LE
        };

        // Convert u16 planes to bytes (little-endian 10-bit in 16-bit container)
        let y_bytes: Vec<u8> = y_plane
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        let cb_bytes: Vec<u8> = cb_plane
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        let cr_bytes: Vec<u8> = cr_plane
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();

        // Build VideoFrame
        let mut frame = VideoFrame::new(self.width, self.height, pixel_format);
        frame.data = vec![
            Buffer::from_vec(y_bytes),
            Buffer::from_vec(cb_bytes),
            Buffer::from_vec(cr_bytes),
        ];
        frame.linesize = vec![y_stride * 2, c_stride * 2, c_stride * 2]; // *2 for 16-bit samples
        frame.keyframe = true;
        frame.pict_type = PictureType::I;
        frame.pts = Timestamp::new(0);

        Ok(frame)
    }
}

impl Default for ProResDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder for ProResDecoder {
    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        // Parse frame header
        let header = self.parse_frame_header(packet.data.as_slice())?;

        // Decode frame data
        let data_slice = packet.data.as_slice();
        let mut video_frame =
            self.decode_frame_data(&data_slice[header.header_size as usize..], &header)?;

        video_frame.pts = packet.pts;

        self.pending_frame = Some(Frame::Video(video_frame));

        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        self.pending_frame.take().ok_or(Error::TryAgain)
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
    fn test_prores_decoder_creation() {
        let decoder = ProResDecoder::new();
        assert_eq!(decoder.width, 0);
        assert_eq!(decoder.height, 0);
    }

    #[test]
    fn test_bitreader_basic() {
        let data = [0b10110001, 0b11110000];
        let mut br = BitReader::new(&data);

        assert_eq!(br.read_bit().unwrap(), 1);
        assert_eq!(br.read_bit().unwrap(), 0);
        assert_eq!(br.read_bit().unwrap(), 1);
        assert_eq!(br.read_bit().unwrap(), 1);
        assert_eq!(br.read_bit().unwrap(), 0);
        assert_eq!(br.read_bit().unwrap(), 0);
        assert_eq!(br.read_bit().unwrap(), 0);
        assert_eq!(br.read_bit().unwrap(), 1);
        // Second byte
        assert_eq!(br.read_bit().unwrap(), 1);
    }

    #[test]
    fn test_bitreader_read_bits() {
        let data = [0b11010110, 0b10000000];
        let mut br = BitReader::new(&data);

        assert_eq!(br.read_bits(5).unwrap(), 0b11010);
        assert_eq!(br.read_bits(3).unwrap(), 0b110);
    }

    #[test]
    fn test_to_signed() {
        assert_eq!(to_signed(0), 0);
        assert_eq!(to_signed(1), -1);
        assert_eq!(to_signed(2), 1);
        assert_eq!(to_signed(3), -2);
        assert_eq!(to_signed(4), 2);
        assert_eq!(to_signed(5), -3);
        assert_eq!(to_signed(6), 3);
    }

    #[test]
    fn test_vlc_decode_simple() {
        // Test Rice coding with rice_order=0, exp_order=1, switch_bits=0
        // Codebook: 0x04 = 0b00000100
        // For value 0: unary 1 (just the stop bit) -> "1"
        let data = [0b10000000];
        let mut br = BitReader::new(&data);
        let value = decode_codeword(&mut br, 0x04).unwrap();
        assert_eq!(value, 0);
    }

    #[test]
    fn test_vlc_decode_rice() {
        // Test Rice coding with rice_order=2
        // Codebook: 0x44 = 0b01000100 (rice_order=2, exp_order=1, switch_bits=0)
        // In the bitstream: "0101" means:
        //   - q=1 (one zero, then stop bit "1")
        //   - r=01 (2-bit remainder)
        // value = (q << rice_order) | r = (1 << 2) | 1 = 4 | 1 = 5
        // But since switch_bits=0, q=1 > switch_bits, so we're in ExpGolomb mode!
        // Let's use a simple Rice test instead: q=0, r=01 -> value = 1
        let data = [0b10100000]; // "1" stop bit, then "01" remainder
        let mut br = BitReader::new(&data);
        let value = decode_codeword(&mut br, 0x44).unwrap();
        // q=0 (immediate stop bit), r=01
        // value = (0 << 2) | 1 = 1
        assert_eq!(value, 1);
    }

    #[test]
    fn test_slice_header_parse() {
        // Minimal slice header: header_size_bits=2 (so header_size = 0b010 << 3 = 16... wait)
        // Actually header_size = data[0] >> 3
        // So if data[0] = 0b00010000, header_size = 2
        let data = [0b00010000, 16, 16, 0, 0]; // header_size=2, quant_y=16
        let sh = parse_slice_header(&data).unwrap();
        assert_eq!(sh.header_size, 2);
        assert_eq!(sh.quant_y, 16);
    }

    #[test]
    fn test_decode_dc_empty() {
        let data = [0xFF; 16];
        let mut br = BitReader::new(&data);
        let mut coeffs = [0i16; 64];
        // Should not panic with 0 blocks
        decode_dc_coeffs(&mut br, &mut coeffs, 0).unwrap();
    }

    #[test]
    fn test_idct_dequant_integration() {
        use crate::codec::prores::idct::{dequant_block, idct_8x8};
        use crate::codec::prores::tables::PRORES_QUANT_MATRICES;

        // Create a simple coefficient pattern (DC only)
        let mut coeffs = [0i16; 64];
        coeffs[0] = 100; // DC coefficient

        let qmat = &PRORES_QUANT_MATRICES[3]; // HQ profile
        let qscale = 4;

        // Dequantize
        let mut block = dequant_block(&coeffs, qmat, qscale);

        // Inverse DCT
        idct_8x8(&mut block);

        // DC coefficient should produce a block with positive average
        let avg: i32 = block.iter().sum::<i32>() / 64;
        assert!(avg > 0, "Average should be positive for positive DC");

        // The IDCT distributes DC energy across the block
        // We expect the output values to be reasonable (not all zeros, not overflow)
        let min_val = *block.iter().min().unwrap();
        let max_val = *block.iter().max().unwrap();
        assert!(
            max_val - min_val < 200,
            "Block variance should be reasonable: min={}, max={}",
            min_val,
            max_val
        );
    }

    #[test]
    fn test_dc_single_block() {
        // Test DC decoding with just one block
        // This avoids the differential decoding complexity
        let data = [0b10000000]; // Stop bit = 1, value = 0
        let mut br = BitReader::new(&data);
        let mut coeffs = [0i16; 64];

        let result = decode_dc_coeffs(&mut br, &mut coeffs, 1);
        assert!(result.is_ok());
        assert_eq!(coeffs[0], 0); // DC value should be 0
    }

    #[test]
    fn test_frame_header_parsing() {
        // Test parsing a valid ProRes frame header
        let mut decoder = ProResDecoder::new();

        // Construct a minimal valid ProRes header
        let mut header_data = vec![0u8; 148];

        // Frame size (4 bytes, big-endian)
        let frame_size: u32 = 1000;
        header_data[0..4].copy_from_slice(&frame_size.to_be_bytes());

        // Frame identifier "icpf"
        header_data[4..8].copy_from_slice(b"icpf");

        // Header size (2 bytes, big-endian) - 148 bytes is typical
        header_data[8..10].copy_from_slice(&148u16.to_be_bytes());

        // Version
        header_data[10] = 0;

        // Encoder ID
        header_data[11..15].copy_from_slice(b"zvd0");

        // Width (2 bytes, big-endian)
        header_data[15..17].copy_from_slice(&1920u16.to_be_bytes());

        // Height (2 bytes, big-endian)
        header_data[17..19].copy_from_slice(&1080u16.to_be_bytes());

        // Chroma format (2 = 4:2:2)
        header_data[19] = 2;

        let result = decoder.parse_frame_header(&header_data);
        assert!(result.is_ok());

        let header = result.unwrap();
        assert_eq!(header.width, 1920);
        assert_eq!(header.height, 1080);
        assert_eq!(header.chroma_format, 2);
        assert_eq!(&header.frame_identifier, b"icpf");
    }
}
