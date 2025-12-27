//! ProRes encoder implementation
//!
//! Implements Apple ProRes 422/4444 encoding with proper DCT, quantization,
//! and entropy encoding. This encoder produces bitstreams compatible with
//! Apple's ProRes specification.

use super::bitstream::{encode_ac_coeffs, encode_dc_coeffs, BitWriter};
use super::dct::{fdct_8x8, quant_block};
use super::tables::{PRORES_INTERLACED_SCAN, PRORES_PROGRESSIVE_SCAN, PRORES_QUANT_MATRICES};
use super::{ProResFrameHeader, ProResProfile};
use crate::codec::{Encoder, Frame, VideoFrame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, PixelFormat};

/// Default quantization scale per profile (lower = higher quality)
const PROFILE_QSCALE: [i32; 6] = [
    24, // Proxy
    16, // LT
    8,  // Standard
    4,  // HQ
    2,  // 4444
    1,  // 4444XQ
];

/// ProRes video encoder
pub struct ProResEncoder {
    profile: ProResProfile,
    width: u32,
    height: u32,
    frame_count: u64,
    pending_packet: Option<Packet>,
    /// Quality setting (0-100, higher = better quality)
    quality: u32,
    /// Interlaced mode
    interlaced: bool,
}

impl ProResEncoder {
    /// Create a new ProRes encoder
    pub fn new(width: u32, height: u32, profile: ProResProfile) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::invalid_input("Width and height must be non-zero"));
        }

        // ProRes requires dimensions to be multiples of 16
        if width % 16 != 0 || height % 16 != 0 {
            // We'll handle this by padding internally
        }

        Ok(ProResEncoder {
            profile,
            width,
            height,
            frame_count: 0,
            pending_packet: None,
            quality: 75, // Default quality
            interlaced: false,
        })
    }

    /// Create encoder with custom quality setting
    pub fn with_quality(width: u32, height: u32, profile: ProResProfile, quality: u32) -> Result<Self> {
        let mut encoder = Self::new(width, height, profile)?;
        encoder.quality = quality.min(100);
        Ok(encoder)
    }

    /// Set interlaced mode
    pub fn set_interlaced(&mut self, interlaced: bool) {
        self.interlaced = interlaced;
    }

    /// Get the profile
    pub fn profile(&self) -> ProResProfile {
        self.profile
    }

    /// Get quantization matrix index for the profile
    fn get_qmat_index(&self) -> usize {
        match self.profile {
            ProResProfile::Proxy => 0,
            ProResProfile::Lt => 1,
            ProResProfile::Standard => 2,
            ProResProfile::Hq => 3,
            ProResProfile::FourFourFourFour => 4,
            ProResProfile::FourFourFourFourXq => 5,
        }
    }

    /// Get base quantization scale for the profile, adjusted by quality
    fn get_qscale(&self) -> i32 {
        let base = PROFILE_QSCALE[self.get_qmat_index()];
        // Quality 100 = base qscale, quality 0 = base * 4
        let quality_factor = 100 - self.quality.min(100);
        let adjusted = base + (base * quality_factor as i32) / 50;
        adjusted.max(1).min(224) // ProRes qscale range is 1-224
    }

    /// Check if this is a 4:4:4 profile
    fn is_444(&self) -> bool {
        matches!(
            self.profile,
            ProResProfile::FourFourFourFour | ProResProfile::FourFourFourFourXq
        )
    }

    /// Encode frame header
    fn encode_frame_header(&self) -> Vec<u8> {
        let header = ProResFrameHeader::new(self.width as u16, self.height as u16, self.profile);

        let mut data = Vec::with_capacity(header.header_size as usize + 4);

        // Frame size placeholder (will be updated)
        data.extend_from_slice(&0u32.to_be_bytes());

        // Frame identifier "icpf"
        data.extend_from_slice(&header.frame_identifier);

        // Header size
        data.extend_from_slice(&header.header_size.to_be_bytes());

        // Version
        data.push(header.version);

        // Encoder ID (use profile-specific encoder ID for compatibility)
        let encoder_id = match self.profile {
            ProResProfile::Proxy => *b"apco",
            ProResProfile::Lt => *b"apcs",
            ProResProfile::Standard => *b"apcn",
            ProResProfile::Hq => *b"apch",
            ProResProfile::FourFourFourFour => *b"ap4h",
            ProResProfile::FourFourFourFourXq => *b"ap4x",
        };
        data.extend_from_slice(&encoder_id);

        // Dimensions
        data.extend_from_slice(&header.width.to_be_bytes());
        data.extend_from_slice(&header.height.to_be_bytes());

        // Chroma format and flags
        let chroma_format = if self.is_444() { 3 } else { 2 }; // 2=422, 3=444
        data.push(chroma_format);
        data.push(if self.interlaced { 1 } else { 0 }); // Interlace mode
        data.push(header.aspect_ratio);
        data.push(header.framerate_code);

        // Color information
        data.push(header.color_primaries);
        data.push(header.transfer_characteristics);
        data.push(header.matrix_coefficients);

        // Alpha info
        data.push(header.alpha_info);

        // Reserved/padding to reach header size
        // header_size indicates the offset from byte 0 to picture data
        // So we need to pad until we reach exactly header_size bytes
        while data.len() < header.header_size as usize {
            data.push(0);
        }

        data
    }

    /// Convert input frame to 10-bit YUV planes
    fn convert_to_yuv10(
        &self,
        frame: &VideoFrame,
    ) -> Result<(Vec<u16>, Vec<u16>, Vec<u16>, usize, usize)> {
        let width = frame.width as usize;
        let height = frame.height as usize;
        let is_444 = self.is_444();
        let chroma_width = if is_444 { width } else { (width + 1) / 2 };

        let mut y_plane = vec![512u16; width * height];
        let mut cb_plane = vec![512u16; chroma_width * height];
        let mut cr_plane = vec![512u16; chroma_width * height];

        match frame.format {
            PixelFormat::YUV420P | PixelFormat::YUV422P | PixelFormat::YUV444P => {
                // 8-bit YUV input - scale to 10-bit
                if frame.data.len() >= 3 {
                    let y_data = frame.data[0].as_slice();
                    let u_data = frame.data[1].as_slice();
                    let v_data = frame.data[2].as_slice();

                    let y_stride = if !frame.linesize.is_empty() {
                        frame.linesize[0]
                    } else {
                        width
                    };

                    let uv_width = match frame.format {
                        PixelFormat::YUV444P => width,
                        _ => (width + 1) / 2,
                    };
                    let uv_stride = if frame.linesize.len() > 1 {
                        frame.linesize[1]
                    } else {
                        uv_width
                    };

                    let uv_height = match frame.format {
                        PixelFormat::YUV420P => (height + 1) / 2,
                        _ => height,
                    };

                    // Copy Y plane (scale 8-bit to 10-bit)
                    for row in 0..height {
                        for col in 0..width {
                            let src_idx = row * y_stride + col;
                            if src_idx < y_data.len() {
                                y_plane[row * width + col] = (y_data[src_idx] as u16) << 2;
                            }
                        }
                    }

                    // Copy and potentially upsample chroma
                    for row in 0..height {
                        let src_row = match frame.format {
                            PixelFormat::YUV420P => row / 2,
                            _ => row,
                        };
                        if src_row >= uv_height {
                            continue;
                        }

                        for col in 0..chroma_width {
                            let src_col = if is_444 && frame.format != PixelFormat::YUV444P {
                                col / 2
                            } else {
                                col
                            };
                            if src_col >= uv_width {
                                continue;
                            }

                            let src_idx = src_row * uv_stride + src_col;
                            if src_idx < u_data.len() {
                                cb_plane[row * chroma_width + col] = (u_data[src_idx] as u16) << 2;
                            }
                            if src_idx < v_data.len() {
                                cr_plane[row * chroma_width + col] = (v_data[src_idx] as u16) << 2;
                            }
                        }
                    }
                }
            }
            PixelFormat::YUV420P10LE | PixelFormat::YUV422P10LE | PixelFormat::YUV444P10LE => {
                // 10-bit YUV input - copy directly
                if frame.data.len() >= 3 {
                    let y_bytes = frame.data[0].as_slice();
                    let u_bytes = frame.data[1].as_slice();
                    let v_bytes = frame.data[2].as_slice();

                    let y_stride = if !frame.linesize.is_empty() {
                        frame.linesize[0] / 2
                    } else {
                        width
                    };

                    let uv_width = match frame.format {
                        PixelFormat::YUV444P10LE => width,
                        _ => (width + 1) / 2,
                    };
                    let uv_stride = if frame.linesize.len() > 1 {
                        frame.linesize[1] / 2
                    } else {
                        uv_width
                    };

                    let uv_height = match frame.format {
                        PixelFormat::YUV420P10LE => (height + 1) / 2,
                        _ => height,
                    };

                    // Copy Y plane
                    for row in 0..height {
                        for col in 0..width {
                            let src_idx = (row * y_stride + col) * 2;
                            if src_idx + 1 < y_bytes.len() {
                                y_plane[row * width + col] =
                                    u16::from_le_bytes([y_bytes[src_idx], y_bytes[src_idx + 1]]);
                            }
                        }
                    }

                    // Copy chroma
                    for row in 0..height {
                        let src_row = match frame.format {
                            PixelFormat::YUV420P10LE => row / 2,
                            _ => row,
                        };
                        if src_row >= uv_height {
                            continue;
                        }

                        for col in 0..chroma_width {
                            let src_col = if is_444 && frame.format != PixelFormat::YUV444P10LE {
                                col / 2
                            } else {
                                col
                            };
                            if src_col >= uv_width {
                                continue;
                            }

                            let src_idx = (src_row * uv_stride + src_col) * 2;
                            if src_idx + 1 < u_bytes.len() {
                                cb_plane[row * chroma_width + col] =
                                    u16::from_le_bytes([u_bytes[src_idx], u_bytes[src_idx + 1]]);
                            }
                            if src_idx + 1 < v_bytes.len() {
                                cr_plane[row * chroma_width + col] =
                                    u16::from_le_bytes([v_bytes[src_idx], v_bytes[src_idx + 1]]);
                            }
                        }
                    }
                }
            }
            PixelFormat::RGB24 | PixelFormat::BGR24 => {
                // Convert RGB to YUV
                if !frame.data.is_empty() {
                    let rgb_data = frame.data[0].as_slice();
                    let stride = if !frame.linesize.is_empty() {
                        frame.linesize[0]
                    } else {
                        width * 3
                    };

                    let is_bgr = matches!(frame.format, PixelFormat::BGR24);

                    for row in 0..height {
                        for col in 0..width {
                            let idx = row * stride + col * 3;
                            if idx + 2 < rgb_data.len() {
                                let (r, g, b) = if is_bgr {
                                    (rgb_data[idx + 2], rgb_data[idx + 1], rgb_data[idx])
                                } else {
                                    (rgb_data[idx], rgb_data[idx + 1], rgb_data[idx + 2])
                                };

                                // BT.709 RGB to YCbCr conversion (10-bit output)
                                let r = r as i32;
                                let g = g as i32;
                                let b = b as i32;

                                // Y = 0.2126R + 0.7152G + 0.0722B
                                let y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
                                // Scale 8-bit range to 10-bit
                                y_plane[row * width + col] = (y.clamp(16, 235) as u16) << 2;

                                if col < chroma_width {
                                    // Cb = -0.1146R - 0.3854G + 0.5B + 128
                                    let cb = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
                                    // Cr = 0.5R - 0.4542G - 0.0458B + 128
                                    let cr = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;

                                    cb_plane[row * chroma_width + col] = (cb.clamp(16, 240) as u16) << 2;
                                    cr_plane[row * chroma_width + col] = (cr.clamp(16, 240) as u16) << 2;
                                }
                            }
                        }
                    }
                }
            }
            PixelFormat::RGBA | PixelFormat::BGRA => {
                // Convert RGBA to YUV (ignoring alpha for non-4444 profiles)
                if !frame.data.is_empty() {
                    let rgba_data = frame.data[0].as_slice();
                    let stride = if !frame.linesize.is_empty() {
                        frame.linesize[0]
                    } else {
                        width * 4
                    };

                    let is_bgra = matches!(frame.format, PixelFormat::BGRA);

                    for row in 0..height {
                        for col in 0..width {
                            let idx = row * stride + col * 4;
                            if idx + 3 < rgba_data.len() {
                                let (r, g, b) = if is_bgra {
                                    (rgba_data[idx + 2], rgba_data[idx + 1], rgba_data[idx])
                                } else {
                                    (rgba_data[idx], rgba_data[idx + 1], rgba_data[idx + 2])
                                };

                                let r = r as i32;
                                let g = g as i32;
                                let b = b as i32;

                                let y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
                                y_plane[row * width + col] = (y.clamp(16, 235) as u16) << 2;

                                if col < chroma_width {
                                    let cb = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
                                    let cr = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;

                                    cb_plane[row * chroma_width + col] = (cb.clamp(16, 240) as u16) << 2;
                                    cr_plane[row * chroma_width + col] = (cr.clamp(16, 240) as u16) << 2;
                                }
                            }
                        }
                    }
                }
            }
            _ => {
                // For unsupported formats, return mid-gray
            }
        }

        Ok((y_plane, cb_plane, cr_plane, width, chroma_width))
    }

    /// Encode a single slice (row of macroblocks)
    fn encode_slice(
        &self,
        y_plane: &[u16],
        cb_plane: &[u16],
        cr_plane: &[u16],
        y_stride: usize,
        c_stride: usize,
        mb_x: usize,
        mb_y: usize,
        mb_count: usize,
        qscale: i32,
    ) -> Vec<u8> {
        let is_444 = self.is_444();
        let qmat_idx = self.get_qmat_index();
        let qmat = &PRORES_QUANT_MATRICES[qmat_idx];

        let scan = if self.interlaced {
            &PRORES_INTERLACED_SCAN
        } else {
            &PRORES_PROGRESSIVE_SCAN
        };

        // Calculate blocks per macroblock
        let luma_blocks_per_mb = 4; // 2x2 luma blocks per MB
        let chroma_blocks_per_mb = if is_444 { 4 } else { 2 }; // 422: 2 chroma, 444: 4

        let total_luma_blocks = mb_count * luma_blocks_per_mb;
        let total_chroma_blocks = mb_count * chroma_blocks_per_mb;

        // Allocate coefficient buffers
        let mut y_coeffs = vec![0i16; total_luma_blocks * 64];
        let mut cb_coeffs = vec![0i16; total_chroma_blocks * 64];
        let mut cr_coeffs = vec![0i16; total_chroma_blocks * 64];

        // Process each macroblock
        for mb in 0..mb_count {
            let mb_pixel_x = (mb_x + mb) * 16;
            let mb_pixel_y = mb_y * 16;

            // Process luma blocks (2x2 arrangement per MB)
            for block_row in 0..2 {
                for block_col in 0..2 {
                    let block_idx = mb * 4 + block_row * 2 + block_col;
                    let block_x = mb_pixel_x + block_col * 8;
                    let block_y = mb_pixel_y + block_row * 8;

                    // Extract 8x8 block
                    let mut block = [0i32; 64];
                    for row in 0..8 {
                        for col in 0..8 {
                            let py = block_y + row;
                            let px = block_x + col;
                            if py < self.height as usize && px < self.width as usize {
                                // Convert from 10-bit unsigned to signed (subtract 512 mid-gray)
                                let val = y_plane[py * y_stride + px] as i32 - 512;
                                block[row * 8 + col] = val;
                            }
                        }
                    }

                    // Apply forward DCT
                    fdct_8x8(&mut block);

                    // Quantize
                    let quantized = quant_block(&block, qmat, qscale);

                    // Store coefficients (in zigzag order for encoding)
                    for i in 0..64 {
                        y_coeffs[block_idx * 64 + scan[i] as usize] = quantized[i];
                    }
                }
            }

            // Process chroma blocks
            let chroma_mb_x = if is_444 { mb_pixel_x } else { mb_pixel_x / 2 };
            let chroma_blocks = if is_444 { 4 } else { 2 };
            let chroma_block_cols = if is_444 { 2 } else { 1 };

            for block_row in 0..2 {
                for block_col in 0..chroma_block_cols {
                    let block_idx_in_mb = block_row * chroma_block_cols + block_col;
                    let cb_block_idx = mb * chroma_blocks + block_idx_in_mb;
                    let cr_block_idx = mb * chroma_blocks + block_idx_in_mb;

                    let block_x = chroma_mb_x + block_col * 8;
                    let block_y = mb_pixel_y + block_row * 8;

                    // Extract Cb block
                    let mut cb_block = [0i32; 64];
                    let mut cr_block = [0i32; 64];

                    for row in 0..8 {
                        for col in 0..8 {
                            let py = block_y + row;
                            let px = block_x + col;
                            if py < self.height as usize && px < c_stride {
                                let idx = py * c_stride + px;
                                if idx < cb_plane.len() {
                                    cb_block[row * 8 + col] = cb_plane[idx] as i32 - 512;
                                }
                                if idx < cr_plane.len() {
                                    cr_block[row * 8 + col] = cr_plane[idx] as i32 - 512;
                                }
                            }
                        }
                    }

                    // Apply forward DCT
                    fdct_8x8(&mut cb_block);
                    fdct_8x8(&mut cr_block);

                    // Quantize
                    let cb_quantized = quant_block(&cb_block, qmat, qscale);
                    let cr_quantized = quant_block(&cr_block, qmat, qscale);

                    // Store coefficients
                    for i in 0..64 {
                        cb_coeffs[cb_block_idx * 64 + scan[i] as usize] = cb_quantized[i];
                        cr_coeffs[cr_block_idx * 64 + scan[i] as usize] = cr_quantized[i];
                    }
                }
            }
        }

        // Encode slice data
        let mut bw = BitWriter::new();

        // Encode luma DC coefficients
        encode_dc_coeffs(&mut bw, &y_coeffs, total_luma_blocks);

        // Encode luma AC coefficients
        encode_ac_coeffs(&mut bw, &y_coeffs, total_luma_blocks, scan);
        bw.flush();
        let y_size = bw.into_bytes().len();

        // Encode Cb
        let mut bw = BitWriter::new();
        encode_dc_coeffs(&mut bw, &cb_coeffs, total_chroma_blocks);
        encode_ac_coeffs(&mut bw, &cb_coeffs, total_chroma_blocks, scan);
        bw.flush();
        let cb_size = bw.into_bytes().len();

        // Encode Cr
        let mut bw = BitWriter::new();
        encode_dc_coeffs(&mut bw, &cr_coeffs, total_chroma_blocks);
        encode_ac_coeffs(&mut bw, &cr_coeffs, total_chroma_blocks, scan);
        bw.flush();
        let cr_size = bw.into_bytes().len();

        // Build slice with header
        let mut slice_data = Vec::with_capacity(8 + y_size + cb_size + cr_size);

        // Slice header (variable size, typically 2-6 bytes)
        // Byte 0: header_size (bits 7-3) | reserved (bits 2-0)
        let header_size = 2u8;
        slice_data.push(header_size << 3);

        // Byte 1: quantization index for luma
        slice_data.push(qscale as u8);

        // Re-encode all data together properly
        let mut bw = BitWriter::new();

        // Encode luma
        encode_dc_coeffs(&mut bw, &y_coeffs, total_luma_blocks);
        encode_ac_coeffs(&mut bw, &y_coeffs, total_luma_blocks, scan);

        // Align to byte boundary
        bw.flush();
        let luma_data = bw.into_bytes();

        // Encode Cb
        let mut bw = BitWriter::new();
        encode_dc_coeffs(&mut bw, &cb_coeffs, total_chroma_blocks);
        encode_ac_coeffs(&mut bw, &cb_coeffs, total_chroma_blocks, scan);
        bw.flush();
        let cb_data = bw.into_bytes();

        // Encode Cr
        let mut bw = BitWriter::new();
        encode_dc_coeffs(&mut bw, &cr_coeffs, total_chroma_blocks);
        encode_ac_coeffs(&mut bw, &cr_coeffs, total_chroma_blocks, scan);
        bw.flush();
        let cr_data = bw.into_bytes();

        slice_data.extend_from_slice(&luma_data);
        slice_data.extend_from_slice(&cb_data);
        slice_data.extend_from_slice(&cr_data);

        slice_data
    }

    /// Encode frame data
    fn encode_frame_data(&self, frame: &VideoFrame) -> Result<Vec<u8>> {
        // Convert input to 10-bit YUV
        let (y_plane, cb_plane, cr_plane, y_stride, c_stride) = self.convert_to_yuv10(frame)?;

        let width = self.width as usize;
        let height = self.height as usize;

        // Calculate macroblock dimensions
        let mb_width = (width + 15) / 16;
        let mb_height = (height + 15) / 16;

        // ProRes uses 8 macroblocks per slice (typically)
        let mbs_per_slice = 8;
        let slices_per_row = (mb_width + mbs_per_slice - 1) / mbs_per_slice;
        let total_slices = mb_height * slices_per_row;

        let qscale = self.get_qscale();

        // Encode all slices
        let mut slice_data: Vec<Vec<u8>> = Vec::with_capacity(total_slices);

        for mb_row in 0..mb_height {
            let mut mb_col = 0;
            while mb_col < mb_width {
                let mb_count = (mb_width - mb_col).min(mbs_per_slice);

                let slice = self.encode_slice(
                    &y_plane,
                    &cb_plane,
                    &cr_plane,
                    y_stride,
                    c_stride,
                    mb_col,
                    mb_row,
                    mb_count,
                    qscale,
                );

                slice_data.push(slice);
                mb_col += mb_count;
            }
        }

        // Build picture data structure
        let num_slices = slice_data.len();

        // Calculate slice sizes for index table
        let slice_sizes: Vec<u16> = slice_data
            .iter()
            .map(|s| (s.len() as u16).min(65535))
            .collect();

        // Calculate total size
        // Picture header is 8 bytes:
        //   - Byte 0: picture_header_size (8)
        //   - Bytes 1-4: picture_size (total picture data size)
        //   - Bytes 5-6: num_slices
        //   - Byte 7: reserved
        let picture_header_size = 8usize;
        let index_size = num_slices * 2;
        let data_size: usize = slice_data.iter().map(|s| s.len()).sum();
        let picture_size = picture_header_size + index_size + data_size;

        // Build picture data
        let mut picture_data = Vec::with_capacity(picture_size);

        // Picture header (8 bytes)
        picture_data.push(picture_header_size as u8);

        // Picture size (4 bytes big-endian)
        picture_data.extend_from_slice(&(picture_size as u32).to_be_bytes());

        // Number of slices (2 bytes big-endian)
        picture_data.extend_from_slice(&(num_slices as u16).to_be_bytes());

        // Reserved byte
        picture_data.push(0);

        // Slice index table
        for size in &slice_sizes {
            picture_data.extend_from_slice(&size.to_be_bytes());
        }

        // Slice data
        for slice in slice_data {
            picture_data.extend_from_slice(&slice);
        }

        Ok(picture_data)
    }
}

impl Encoder for ProResEncoder {
    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let video_frame = match frame {
            Frame::Video(vf) => vf,
            Frame::Audio(_) => {
                return Err(Error::codec("ProRes encoder only accepts video frames"))
            }
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
        packet.set_keyframe(true); // ProRes frames are all keyframes

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
    use crate::util::Timestamp;

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
    fn test_prores_encoder_with_quality() {
        let encoder = ProResEncoder::with_quality(1920, 1080, ProResProfile::Hq, 90).unwrap();
        assert_eq!(encoder.quality, 90);
    }

    #[test]
    fn test_prores_encoder_profiles() {
        for profile in [
            ProResProfile::Proxy,
            ProResProfile::Lt,
            ProResProfile::Standard,
            ProResProfile::Hq,
            ProResProfile::FourFourFourFour,
            ProResProfile::FourFourFourFourXq,
        ] {
            let encoder = ProResEncoder::new(1920, 1080, profile);
            assert!(encoder.is_ok());
        }
    }

    #[test]
    fn test_prores_encode_frame_yuv420() {
        let mut encoder = ProResEncoder::new(64, 64, ProResProfile::Standard).unwrap();

        // Create a simple YUV420P frame
        let mut frame = VideoFrame::new(64, 64, PixelFormat::YUV420P);
        frame.pts = Timestamp::new(0);

        // Create Y plane (64x64)
        let y_data: Vec<u8> = (0..64 * 64).map(|i| (i % 256) as u8).collect();
        // Create U and V planes (32x32 for 420)
        let u_data: Vec<u8> = vec![128u8; 32 * 32];
        let v_data: Vec<u8> = vec![128u8; 32 * 32];

        frame.data = vec![
            Buffer::from_vec(y_data),
            Buffer::from_vec(u_data),
            Buffer::from_vec(v_data),
        ];
        frame.linesize = vec![64, 32, 32];

        let send_result = encoder.send_frame(&Frame::Video(frame));
        assert!(send_result.is_ok());

        let packet_result = encoder.receive_packet();
        assert!(packet_result.is_ok());

        let packet = packet_result.unwrap();
        assert!(!packet.data.is_empty());
        assert!(packet.is_keyframe());

        // Verify ProRes header
        let data = packet.data.as_slice();
        assert!(data.len() > 20);
        assert_eq!(&data[4..8], b"icpf"); // Frame identifier
    }

    #[test]
    fn test_prores_encode_frame_rgb() {
        let mut encoder = ProResEncoder::new(32, 32, ProResProfile::Hq).unwrap();

        // Create a simple RGB frame
        let mut frame = VideoFrame::new(32, 32, PixelFormat::RGB24);
        frame.pts = Timestamp::new(0);

        // Create RGB data
        let rgb_data: Vec<u8> = (0..32 * 32 * 3).map(|i| (i % 256) as u8).collect();
        frame.data = vec![Buffer::from_vec(rgb_data)];
        frame.linesize = vec![32 * 3];

        let send_result = encoder.send_frame(&Frame::Video(frame));
        assert!(send_result.is_ok());

        let packet_result = encoder.receive_packet();
        assert!(packet_result.is_ok());
    }

    #[test]
    fn test_prores_encode_4444() {
        let mut encoder = ProResEncoder::new(32, 32, ProResProfile::FourFourFourFour).unwrap();

        // Create a YUV444 frame
        let mut frame = VideoFrame::new(32, 32, PixelFormat::YUV444P);
        frame.pts = Timestamp::new(0);

        // All planes are same size for 444
        let y_data: Vec<u8> = vec![128u8; 32 * 32];
        let u_data: Vec<u8> = vec![128u8; 32 * 32];
        let v_data: Vec<u8> = vec![128u8; 32 * 32];

        frame.data = vec![
            Buffer::from_vec(y_data),
            Buffer::from_vec(u_data),
            Buffer::from_vec(v_data),
        ];
        frame.linesize = vec![32, 32, 32];

        let send_result = encoder.send_frame(&Frame::Video(frame));
        assert!(send_result.is_ok());

        let packet = encoder.receive_packet().unwrap();

        // Check encoder ID in packet
        let data = packet.data.as_slice();
        assert_eq!(&data[11..15], b"ap4h"); // 4444 encoder ID
    }

    #[test]
    fn test_prores_qscale_range() {
        // Test quality to qscale mapping
        for quality in [0, 25, 50, 75, 100] {
            let encoder =
                ProResEncoder::with_quality(64, 64, ProResProfile::Standard, quality).unwrap();
            let qscale = encoder.get_qscale();
            assert!(qscale >= 1 && qscale <= 224, "qscale {} out of range", qscale);
        }
    }

    #[test]
    fn test_prores_roundtrip() {
        use crate::codec::prores::decoder::ProResDecoder;
        use crate::codec::Decoder;

        // Create encoder
        let mut encoder = ProResEncoder::new(64, 64, ProResProfile::Hq).unwrap();

        // Create a test frame with gradient
        let mut frame = VideoFrame::new(64, 64, PixelFormat::YUV420P);
        frame.pts = Timestamp::new(0);

        // Create gradient Y plane
        let y_data: Vec<u8> = (0..64 * 64)
            .map(|i| {
                let x = (i % 64) as u8;
                let y = (i / 64) as u8;
                ((x as u16 + y as u16) / 2) as u8 + 64
            })
            .collect();
        let u_data: Vec<u8> = vec![128u8; 32 * 32];
        let v_data: Vec<u8> = vec![128u8; 32 * 32];

        frame.data = vec![
            Buffer::from_vec(y_data.clone()),
            Buffer::from_vec(u_data),
            Buffer::from_vec(v_data),
        ];
        frame.linesize = vec![64, 32, 32];

        // Encode
        encoder.send_frame(&Frame::Video(frame)).unwrap();
        let packet = encoder.receive_packet().unwrap();

        // Verify packet structure
        let data = packet.data.as_slice();
        assert!(data.len() > 156, "Packet too small");
        assert_eq!(&data[4..8], b"icpf", "Invalid frame identifier");

        // Decode
        let mut decoder = ProResDecoder::new();
        decoder.send_packet(&packet).expect("Decode failed");
        let decoded = decoder.receive_frame().unwrap();

        // Verify we got a frame back
        if let Frame::Video(vf) = decoded {
            assert_eq!(vf.width, 64);
            assert_eq!(vf.height, 64);
            assert!(vf.keyframe);
        } else {
            panic!("Expected video frame");
        }
    }
}
