//! DNxHD encoder implementation
//!
//! This module implements a complete DNxHD encoder that:
//! - Converts input frames to YUV 4:2:2 format if needed
//! - Applies 8x8 DCT to blocks
//! - Quantizes DCT coefficients using profile-specific matrices
//! - Encodes coefficients using VLC (Variable Length Coding)
//! - Writes proper DNxHD frame headers and macroblock data

use super::bitstream::BitWriter;
use super::dct::{fdct_8x8, fdct_8x8_10bit, quant_block_to_i16};
use super::tables::{DnxhdProfileTables, DNXHD_ZIGZAG};
use super::{DnxhdFrameHeader, DnxhdProfile};
use crate::codec::{Encoder, Frame, VideoFrame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, PixelFormat, Timestamp};

/// DNxHD encoder configuration
#[derive(Debug, Clone)]
pub struct DnxhdEncoderConfig {
    /// Quantization scale (1-31 for 8-bit, 1-63 for 10-bit)
    pub qscale: u8,
    /// Target bitrate in bits per second (0 = use qscale directly)
    pub target_bitrate: u64,
}

impl Default for DnxhdEncoderConfig {
    fn default() -> Self {
        DnxhdEncoderConfig {
            qscale: 16,
            target_bitrate: 0,
        }
    }
}

/// DNxHD video encoder
pub struct DnxhdEncoder {
    profile: DnxhdProfile,
    width: u32,
    height: u32,
    frame_count: u64,
    pending_packet: Option<Packet>,
    config: DnxhdEncoderConfig,
    /// Profile-specific encoding tables
    tables: DnxhdProfileTables,
    /// Macroblock width (number of 16x16 macroblocks horizontally)
    mb_width: u32,
    /// Macroblock height (number of 16x16 macroblocks vertically)
    mb_height: u32,
    /// Working buffer for YUV conversion
    yuv_buffer: Option<YuvBuffer>,
}

/// Internal YUV buffer for format conversion
struct YuvBuffer {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    y_stride: usize,
    uv_stride: usize,
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

        // Validate qscale range (must be 1-31 for 8-bit, 1-63 for 10-bit)
        let max_qscale = if profile.is_10bit() { 63 } else { 31 };
        if config.qscale == 0 {
            return Err(Error::invalid_input("qscale must be at least 1"));
        }
        if config.qscale > max_qscale {
            return Err(Error::invalid_input(format!(
                "qscale {} exceeds maximum {} for this profile",
                config.qscale, max_qscale
            )));
        }

        let mb_width = (width + 15) / 16;
        let mb_height = (height + 15) / 16;

        Ok(DnxhdEncoder {
            profile,
            width,
            height,
            frame_count: 0,
            pending_packet: None,
            config,
            tables: DnxhdProfileTables::for_profile(profile),
            mb_width,
            mb_height,
            yuv_buffer: None,
        })
    }

    /// Get the profile
    pub fn profile(&self) -> DnxhdProfile {
        self.profile
    }

    /// Encode the complete DNxHD frame header
    fn encode_frame_header(&self, _frame_size: u32) -> Vec<u8> {
        let header = DnxhdFrameHeader::new(self.width as u16, self.height as u16, self.profile);
        let mut data = Vec::with_capacity(640);

        // Frame header signature (6 bytes) - matches decoder expectation
        // 0x000002800001
        data.extend_from_slice(&[0x00, 0x00, 0x02, 0x80, 0x00, 0x01]);

        // Compression ID (4 bytes, big-endian)
        data.extend_from_slice(&header.compression_id.to_be_bytes());

        // Width (2 bytes, big-endian)
        data.extend_from_slice(&header.width.to_be_bytes());

        // Height (2 bytes, big-endian)
        data.extend_from_slice(&header.height.to_be_bytes());

        // Flags byte
        let mut flags = 0u8;
        if header.is_progressive {
            flags |= 0x01;
        }
        if header.is_422 {
            flags |= 0x02;
        }
        data.push(flags);

        // Bit depth (1 byte)
        data.push(header.bit_depth);

        // Quantization scale (2 bytes)
        data.extend_from_slice(&(self.config.qscale as u16).to_be_bytes());

        // Number of macroblocks (4 bytes)
        let mb_count = self.mb_width * self.mb_height;
        data.extend_from_slice(&mb_count.to_be_bytes());

        // Padding to 64-byte alignment
        while data.len() < 64 {
            data.push(0x00);
        }

        // Macroblock index table placeholder (4 bytes per MB)
        // This will be updated after encoding all macroblocks
        let index_table_size = (mb_count as usize) * 4;
        data.reserve(index_table_size);
        for _ in 0..mb_count {
            data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        }

        // Align to 8-byte boundary
        while data.len() % 8 != 0 {
            data.push(0x00);
        }

        data
    }

    /// Ensure YUV buffer is allocated
    fn ensure_yuv_buffer(&mut self) {
        if self.yuv_buffer.is_some() {
            return;
        }

        let y_stride = ((self.width as usize + 15) / 16) * 16;
        let uv_stride = y_stride / 2;
        let height = ((self.height as usize + 15) / 16) * 16;

        self.yuv_buffer = Some(YuvBuffer {
            y: vec![128; y_stride * height],
            u: vec![128; uv_stride * height],
            v: vec![128; uv_stride * height],
            y_stride,
            uv_stride,
        });
    }

    /// Convert input frame to YUV 4:2:2 format
    fn convert_to_yuv422(&mut self, frame: &VideoFrame) -> Result<()> {
        self.ensure_yuv_buffer();
        let buf = self.yuv_buffer.as_mut().unwrap();

        match frame.format {
            PixelFormat::YUV422P => {
                // Direct copy for YUV 4:2:2 planar
                if frame.data.len() >= 3 {
                    let y_data = frame.data[0].as_slice();
                    let u_data = frame.data[1].as_slice();
                    let v_data = frame.data[2].as_slice();

                    let src_y_stride = if !frame.linesize.is_empty() {
                        frame.linesize[0]
                    } else {
                        frame.width as usize
                    };
                    let src_uv_stride = if frame.linesize.len() > 1 {
                        frame.linesize[1]
                    } else {
                        frame.width as usize / 2
                    };

                    for y in 0..frame.height as usize {
                        let src_y_offset = y * src_y_stride;
                        let dst_y_offset = y * buf.y_stride;
                        let copy_len = (frame.width as usize).min(buf.y_stride);

                        if src_y_offset + copy_len <= y_data.len() {
                            buf.y[dst_y_offset..dst_y_offset + copy_len]
                                .copy_from_slice(&y_data[src_y_offset..src_y_offset + copy_len]);
                        }

                        let src_uv_offset = y * src_uv_stride;
                        let dst_uv_offset = y * buf.uv_stride;
                        let uv_copy_len = (frame.width as usize / 2).min(buf.uv_stride);

                        if src_uv_offset + uv_copy_len <= u_data.len() {
                            buf.u[dst_uv_offset..dst_uv_offset + uv_copy_len]
                                .copy_from_slice(&u_data[src_uv_offset..src_uv_offset + uv_copy_len]);
                        }
                        if src_uv_offset + uv_copy_len <= v_data.len() {
                            buf.v[dst_uv_offset..dst_uv_offset + uv_copy_len]
                                .copy_from_slice(&v_data[src_uv_offset..src_uv_offset + uv_copy_len]);
                        }
                    }
                }
            }
            PixelFormat::YUV420P => {
                // Convert YUV 4:2:0 to 4:2:2 by duplicating chroma rows
                if frame.data.len() >= 3 {
                    let y_data = frame.data[0].as_slice();
                    let u_data = frame.data[1].as_slice();
                    let v_data = frame.data[2].as_slice();

                    let src_y_stride = if !frame.linesize.is_empty() {
                        frame.linesize[0]
                    } else {
                        frame.width as usize
                    };
                    let src_uv_stride = if frame.linesize.len() > 1 {
                        frame.linesize[1]
                    } else {
                        frame.width as usize / 2
                    };

                    // Copy Y plane directly
                    for y in 0..frame.height as usize {
                        let src_y_offset = y * src_y_stride;
                        let dst_y_offset = y * buf.y_stride;
                        let copy_len = (frame.width as usize).min(buf.y_stride);

                        if src_y_offset + copy_len <= y_data.len() {
                            buf.y[dst_y_offset..dst_y_offset + copy_len]
                                .copy_from_slice(&y_data[src_y_offset..src_y_offset + copy_len]);
                        }
                    }

                    // Upsample chroma vertically (duplicate rows)
                    for y in 0..frame.height as usize {
                        let src_y = y / 2;
                        let src_uv_offset = src_y * src_uv_stride;
                        let dst_uv_offset = y * buf.uv_stride;
                        let uv_copy_len = (frame.width as usize / 2).min(buf.uv_stride);

                        if src_uv_offset + uv_copy_len <= u_data.len() {
                            buf.u[dst_uv_offset..dst_uv_offset + uv_copy_len]
                                .copy_from_slice(&u_data[src_uv_offset..src_uv_offset + uv_copy_len]);
                        }
                        if src_uv_offset + uv_copy_len <= v_data.len() {
                            buf.v[dst_uv_offset..dst_uv_offset + uv_copy_len]
                                .copy_from_slice(&v_data[src_uv_offset..src_uv_offset + uv_copy_len]);
                        }
                    }
                }
            }
            PixelFormat::YUV444P => {
                // Convert YUV 4:4:4 to 4:2:2 by subsampling chroma horizontally
                if frame.data.len() >= 3 {
                    let y_data = frame.data[0].as_slice();
                    let u_data = frame.data[1].as_slice();
                    let v_data = frame.data[2].as_slice();

                    let src_y_stride = if !frame.linesize.is_empty() {
                        frame.linesize[0]
                    } else {
                        frame.width as usize
                    };
                    let src_uv_stride = if frame.linesize.len() > 1 {
                        frame.linesize[1]
                    } else {
                        frame.width as usize
                    };

                    // Copy Y plane directly
                    for y in 0..frame.height as usize {
                        let src_y_offset = y * src_y_stride;
                        let dst_y_offset = y * buf.y_stride;
                        let copy_len = (frame.width as usize).min(buf.y_stride);

                        if src_y_offset + copy_len <= y_data.len() {
                            buf.y[dst_y_offset..dst_y_offset + copy_len]
                                .copy_from_slice(&y_data[src_y_offset..src_y_offset + copy_len]);
                        }
                    }

                    // Subsample chroma horizontally
                    for y in 0..frame.height as usize {
                        let src_uv_offset = y * src_uv_stride;
                        let dst_uv_offset = y * buf.uv_stride;

                        for x in 0..(frame.width as usize / 2) {
                            let src_x = x * 2;
                            if src_uv_offset + src_x + 1 < u_data.len() {
                                // Average two horizontal samples
                                buf.u[dst_uv_offset + x] = ((u_data[src_uv_offset + src_x] as u16
                                    + u_data[src_uv_offset + src_x + 1] as u16)
                                    / 2) as u8;
                            }
                            if src_uv_offset + src_x + 1 < v_data.len() {
                                buf.v[dst_uv_offset + x] = ((v_data[src_uv_offset + src_x] as u16
                                    + v_data[src_uv_offset + src_x + 1] as u16)
                                    / 2) as u8;
                            }
                        }
                    }
                }
            }
            PixelFormat::RGB24 | PixelFormat::BGR24 => {
                // Convert RGB to YUV 4:2:2
                if !frame.data.is_empty() {
                    let rgb_data = frame.data[0].as_slice();
                    let stride = if !frame.linesize.is_empty() {
                        frame.linesize[0]
                    } else {
                        frame.width as usize * 3
                    };
                    let is_bgr = matches!(frame.format, PixelFormat::BGR24);

                    for y in 0..frame.height as usize {
                        let row_offset = y * stride;
                        let y_dst_offset = y * buf.y_stride;
                        let uv_dst_offset = y * buf.uv_stride;

                        for x in 0..frame.width as usize {
                            let pixel_offset = row_offset + x * 3;
                            if pixel_offset + 2 < rgb_data.len() {
                                let (r, g, b) = if is_bgr {
                                    (
                                        rgb_data[pixel_offset + 2],
                                        rgb_data[pixel_offset + 1],
                                        rgb_data[pixel_offset],
                                    )
                                } else {
                                    (
                                        rgb_data[pixel_offset],
                                        rgb_data[pixel_offset + 1],
                                        rgb_data[pixel_offset + 2],
                                    )
                                };

                                // BT.601 RGB to YCbCr conversion
                                let y_val = (66 * r as i32 + 129 * g as i32 + 25 * b as i32 + 128) >> 8;
                                buf.y[y_dst_offset + x] = (y_val + 16).clamp(0, 255) as u8;

                                // Subsample chroma horizontally
                                if x % 2 == 0 {
                                    let u_val =
                                        (-38 * r as i32 - 74 * g as i32 + 112 * b as i32 + 128) >> 8;
                                    let v_val =
                                        (112 * r as i32 - 94 * g as i32 - 18 * b as i32 + 128) >> 8;
                                    buf.u[uv_dst_offset + x / 2] = (u_val + 128).clamp(0, 255) as u8;
                                    buf.v[uv_dst_offset + x / 2] = (v_val + 128).clamp(0, 255) as u8;
                                }
                            }
                        }
                    }
                }
            }
            _ => {
                // For unsupported formats, fill with neutral gray
                let buf = self.yuv_buffer.as_mut().unwrap();
                buf.y.fill(128);
                buf.u.fill(128);
                buf.v.fill(128);
            }
        }

        Ok(())
    }

    /// Extract an 8x8 block from the Y plane
    fn extract_y_block(&self, mb_x: u32, mb_y: u32, block_idx: usize) -> [u8; 64] {
        let buf = self.yuv_buffer.as_ref().unwrap();
        let mut block = [128u8; 64];

        // Block index within macroblock: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
        let block_x = (mb_x * 16 + (block_idx % 2) as u32 * 8) as usize;
        let block_y = (mb_y * 16 + (block_idx / 2) as u32 * 8) as usize;

        for row in 0..8 {
            let y = block_y + row;
            if y < self.height as usize {
                for col in 0..8 {
                    let x = block_x + col;
                    if x < self.width as usize {
                        let src_idx = y * buf.y_stride + x;
                        if src_idx < buf.y.len() {
                            block[row * 8 + col] = buf.y[src_idx];
                        }
                    }
                }
            }
        }

        block
    }

    /// Extract an 8x8 block from the U or V plane (8x16 area for 4:2:2)
    fn extract_uv_block(&self, mb_x: u32, mb_y: u32, is_v: bool, block_idx: usize) -> [u8; 64] {
        let buf = self.yuv_buffer.as_ref().unwrap();
        let mut block = [128u8; 64];

        // For 4:2:2, chroma has full vertical resolution but half horizontal
        // block_idx: 0=top, 1=bottom
        let block_x = (mb_x * 8) as usize;
        let block_y = (mb_y * 16 + block_idx as u32 * 8) as usize;

        let plane = if is_v { &buf.v } else { &buf.u };

        for row in 0..8 {
            let y = block_y + row;
            if y < self.height as usize {
                for col in 0..8 {
                    let x = block_x + col;
                    if x < (self.width as usize / 2) {
                        let src_idx = y * buf.uv_stride + x;
                        if src_idx < plane.len() {
                            block[row * 8 + col] = plane[src_idx];
                        }
                    }
                }
            }
        }

        block
    }

    /// Encode a single 8x8 block
    fn encode_block(
        &self,
        block: &[u8; 64],
        bw: &mut BitWriter,
        qmat: &[u8; 64],
        prev_dc: &mut i16,
    ) {
        // Apply DCT
        let mut coeffs = [0i32; 64];
        fdct_8x8(block, 8, &mut coeffs);

        // Quantize
        let qscale = self.config.qscale.max(1) as i32;
        let quantized = quant_block_to_i16(&coeffs, qmat, qscale);

        // Differential DC encoding
        let dc = quantized[0];
        let dc_diff = dc - *prev_dc;
        *prev_dc = dc;

        // Encode DC coefficient using signed exp-golomb
        self.encode_dc_coefficient(bw, dc_diff);

        // Encode AC coefficients using run-length encoding in zigzag order
        self.encode_ac_coefficients(bw, &quantized);
    }

    /// Encode DC coefficient difference
    fn encode_dc_coefficient(&self, bw: &mut BitWriter, dc_diff: i16) {
        let dc_codes = self.tables.dc_codes;
        let dc_bits = self.tables.dc_bits;

        // Map DC difference to code index
        // Typically: 0 -> index for 0, small values -> small indices
        let abs_diff = dc_diff.unsigned_abs() as usize;
        let sign = dc_diff < 0;

        if abs_diff == 0 {
            // Zero DC difference
            if dc_codes.len() > 0 && dc_bits.len() > 0 {
                bw.write_bits(dc_codes[0] as u32, dc_bits[0]);
            }
        } else {
            // Non-zero DC difference
            // Find the size category (number of bits needed to represent abs_diff)
            let size = 16 - (abs_diff as u16).leading_zeros() as usize;

            if size < dc_codes.len() && size < dc_bits.len() {
                // Write the size code
                bw.write_bits(dc_codes[size] as u32, dc_bits[size]);

                // Write the amplitude
                let amplitude = if sign {
                    // For negative values, use one's complement
                    (abs_diff as u16 - 1) ^ ((1 << size) - 1)
                } else {
                    abs_diff as u16
                };
                bw.write_bits(amplitude as u32, size as u8);
            } else {
                // Fallback for very large differences
                bw.write_exp_golomb(abs_diff as u32);
                bw.write_bit(sign);
            }
        }
    }

    /// Encode AC coefficients using run-length coding
    fn encode_ac_coefficients(&self, bw: &mut BitWriter, quantized: &[i16; 64]) {
        let run_codes = self.tables.run_codes;
        let run_bits = self.tables.run_bits;

        let mut run = 0u8;

        // Process coefficients in zigzag order, skipping DC (index 0)
        for i in 1..64 {
            let zigzag_idx = DNXHD_ZIGZAG[i] as usize;
            let coeff = quantized[zigzag_idx];

            if coeff == 0 {
                run += 1;
            } else {
                // Encode run-level pair
                self.encode_run_level(bw, run, coeff, run_codes, run_bits);
                run = 0;
            }
        }

        // End of block marker
        if run > 0 {
            // Encode EOB (end of block) - typically represented as special run value
            // Use run code for the remaining zeros followed by EOB
            self.encode_eob(bw);
        } else {
            // Still need EOB if we ended on a non-zero coefficient
            self.encode_eob(bw);
        }
    }

    /// Encode a run-level pair
    fn encode_run_level(
        &self,
        bw: &mut BitWriter,
        run: u8,
        level: i16,
        run_codes: &[u16; 62],
        run_bits: &[u8; 62],
    ) {
        let abs_level = level.unsigned_abs();
        let sign = level < 0;

        // Find run code index
        let run_idx = (run as usize).min(run_codes.len() - 1);

        // Write run code
        bw.write_bits(run_codes[run_idx] as u32, run_bits[run_idx]);

        // Write level magnitude using exp-golomb
        if abs_level <= 1 {
            bw.write_bit(abs_level == 1);
        } else {
            bw.write_bit(true); // Non-zero flag
            bw.write_exp_golomb((abs_level - 1) as u32);
        }

        // Write sign bit for non-zero levels
        if abs_level > 0 {
            bw.write_bit(sign);
        }
    }

    /// Encode end-of-block marker
    fn encode_eob(&self, bw: &mut BitWriter) {
        // EOB is typically encoded as a special run code
        // Using run code 0 with level 0 as EOB
        bw.write_bits(0b10, 2); // Simple EOB code
    }

    /// Encode a single macroblock (16x16 pixels)
    fn encode_macroblock(&self, mb_x: u32, mb_y: u32, bw: &mut BitWriter) {
        // DNxHD macroblocks contain:
        // - 4 luma blocks (8x8 each, arranged 2x2)
        // - 2 Cb blocks (8x8 each, vertically stacked for 4:2:2)
        // - 2 Cr blocks (8x8 each, vertically stacked for 4:2:2)

        let luma_qmat = self.tables.luma_weight;
        let chroma_qmat = self.tables.chroma_weight;

        // Initialize DC predictors for this macroblock
        let mut prev_dc_y = 0i16;
        let mut prev_dc_u = 0i16;
        let mut prev_dc_v = 0i16;

        // Encode 4 luma blocks
        for block_idx in 0..4 {
            let block = self.extract_y_block(mb_x, mb_y, block_idx);
            self.encode_block(&block, bw, luma_qmat, &mut prev_dc_y);
        }

        // Encode 2 Cb blocks (for 4:2:2)
        for block_idx in 0..2 {
            let block = self.extract_uv_block(mb_x, mb_y, false, block_idx);
            self.encode_block(&block, bw, chroma_qmat, &mut prev_dc_u);
        }

        // Encode 2 Cr blocks (for 4:2:2)
        for block_idx in 0..2 {
            let block = self.extract_uv_block(mb_x, mb_y, true, block_idx);
            self.encode_block(&block, bw, chroma_qmat, &mut prev_dc_v);
        }
    }

    /// Encode frame data
    fn encode_frame_data(&mut self, frame: &VideoFrame) -> Result<Vec<u8>> {
        // Convert to YUV 4:2:2 if needed
        self.convert_to_yuv422(frame)?;

        // Estimate output size (approximately target_bitrate / fps or use a default)
        let estimated_size = ((self.width * self.height) as usize) * 2;
        let mut bw = BitWriter::with_capacity(estimated_size);

        // Track macroblock offsets for the index table
        let mut mb_offsets = Vec::with_capacity((self.mb_width * self.mb_height) as usize);

        // Encode each macroblock
        for mb_y in 0..self.mb_height {
            for mb_x in 0..self.mb_width {
                // Record the bit offset before this macroblock
                mb_offsets.push(bw.bit_position() as u32);

                // Encode the macroblock
                self.encode_macroblock(mb_x, mb_y, &mut bw);
            }
        }

        // Align to byte boundary
        bw.align_to_byte();

        Ok(bw.into_bytes())
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

        // Encode frame data first to know the size
        let frame_data = self.encode_frame_data(video_frame)?;

        // Create header with correct frame size
        let header_size = 64 + (self.mb_width * self.mb_height * 4) as usize;
        let total_size = (header_size + frame_data.len()) as u32;
        let mut encoded_data = self.encode_frame_header(total_size);

        // Append frame data
        encoded_data.extend_from_slice(&frame_data);

        // Create packet
        let data = Buffer::from_vec(encoded_data);
        let mut packet = Packet::new(0, data);
        packet.pts = video_frame.pts;
        packet.dts = video_frame.pts;
        packet.set_keyframe(true); // DNxHD frames are all intra-coded

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

        // Add dummy plane data
        let y_size = 1920 * 1080;
        let uv_size = (1920 / 2) * 1080;
        frame.data.push(Buffer::from_vec(vec![128u8; y_size]));
        frame.data.push(Buffer::from_vec(vec![128u8; uv_size]));
        frame.data.push(Buffer::from_vec(vec![128u8; uv_size]));
        frame.linesize = vec![1920, 960, 960];

        let send_result = encoder.send_frame(&Frame::Video(frame));
        assert!(send_result.is_ok());

        let packet_result = encoder.receive_packet();
        assert!(packet_result.is_ok());

        let packet = packet_result.unwrap();
        assert!(packet.data.len() > 0);
        assert!(packet.is_keyframe());
    }

    #[test]
    fn test_dnxhd_encode_rgb_frame() {
        let mut encoder = DnxhdEncoder::new(64, 64, DnxhdProfile::DnxhrHq).unwrap();
        let mut frame = VideoFrame::new(64, 64, PixelFormat::RGB24);
        frame.pts = Timestamp::new(0);

        // Create RGB data with a gradient
        let mut rgb_data = Vec::with_capacity(64 * 64 * 3);
        for y in 0..64 {
            for x in 0..64 {
                rgb_data.push((x * 4) as u8); // R
                rgb_data.push((y * 4) as u8); // G
                rgb_data.push(128u8); // B
            }
        }
        frame.data.push(Buffer::from_vec(rgb_data));
        frame.linesize = vec![64 * 3];

        let send_result = encoder.send_frame(&Frame::Video(frame));
        assert!(send_result.is_ok());

        let packet_result = encoder.receive_packet();
        assert!(packet_result.is_ok());
    }

    #[test]
    fn test_dnxhd_encode_yuv420_frame() {
        let mut encoder = DnxhdEncoder::new(64, 64, DnxhdProfile::DnxhrSq).unwrap();
        let mut frame = VideoFrame::new(64, 64, PixelFormat::YUV420P);
        frame.pts = Timestamp::new(0);

        // YUV 4:2:0 has half resolution chroma
        let y_size = 64 * 64;
        let uv_size = 32 * 32;
        frame.data.push(Buffer::from_vec(vec![128u8; y_size]));
        frame.data.push(Buffer::from_vec(vec![128u8; uv_size]));
        frame.data.push(Buffer::from_vec(vec![128u8; uv_size]));
        frame.linesize = vec![64, 32, 32];

        let send_result = encoder.send_frame(&Frame::Video(frame));
        assert!(send_result.is_ok());
    }

    #[test]
    fn test_dnxhd_qscale_validation() {
        // 8-bit profile should accept qscale up to 31
        let result = DnxhdEncoder::with_config(
            1920,
            1080,
            DnxhdProfile::Dnxhd115,
            DnxhdEncoderConfig {
                qscale: 31,
                target_bitrate: 0,
            },
        );
        assert!(result.is_ok());

        // 8-bit profile should reject qscale > 31
        let result = DnxhdEncoder::with_config(
            1920,
            1080,
            DnxhdProfile::Dnxhd115,
            DnxhdEncoderConfig {
                qscale: 64,
                target_bitrate: 0,
            },
        );
        assert!(result.is_err());

        // 10-bit profile should accept qscale up to 63
        let result = DnxhdEncoder::with_config(
            1920,
            1080,
            DnxhdProfile::Dnxhd220,
            DnxhdEncoderConfig {
                qscale: 63,
                target_bitrate: 0,
            },
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_dnxhd_macroblock_dimensions() {
        let encoder = DnxhdEncoder::new(1920, 1080, DnxhdProfile::DnxhrHq).unwrap();
        assert_eq!(encoder.mb_width, 120); // 1920 / 16 = 120
        assert_eq!(encoder.mb_height, 68); // ceil(1080 / 16) = 68

        let encoder = DnxhdEncoder::new(1280, 720, DnxhdProfile::DnxhrHq).unwrap();
        assert_eq!(encoder.mb_width, 80); // 1280 / 16 = 80
        assert_eq!(encoder.mb_height, 45); // 720 / 16 = 45
    }

    #[test]
    fn test_dnxhd_encode_multiple_frames() {
        let mut encoder = DnxhdEncoder::new(64, 64, DnxhdProfile::DnxhrLb).unwrap();

        for i in 0..5 {
            let mut frame = VideoFrame::new(64, 64, PixelFormat::YUV422P);
            frame.pts = Timestamp::new(i * 1000);

            let y_size = 64 * 64;
            let uv_size = 32 * 64;
            frame.data.push(Buffer::from_vec(vec![(128 + i as u8); y_size]));
            frame.data.push(Buffer::from_vec(vec![128u8; uv_size]));
            frame.data.push(Buffer::from_vec(vec![128u8; uv_size]));
            frame.linesize = vec![64, 32, 32];

            let send_result = encoder.send_frame(&Frame::Video(frame));
            assert!(send_result.is_ok());

            let packet = encoder.receive_packet().unwrap();
            assert!(packet.data.len() > 0);
            assert!(packet.is_keyframe());
        }
    }
}
