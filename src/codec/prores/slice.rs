//! ProRes Slice Structure and Encoding
//!
//! ProRes frames are divided into horizontal slices for parallel processing.
//! Each slice contains macroblocks (8×8 blocks) that are encoded independently.

use super::{ProResProfile, ProResFrameHeader};
use super::bitstream::{ProResBitstreamReader, ProResBitstreamWriter};
use super::dct::ProResDct;
use super::quant::ProResQuantizer;
use super::vlc::{encode_dct_coefficients, decode_dct_coefficients};
use crate::error::{Error, Result};

/// ProRes slice header
#[derive(Debug, Clone)]
pub struct SliceHeader {
    /// Slice size in bytes
    pub slice_size: u16,
    /// Y position of slice
    pub y_pos: u16,
    /// Quantization parameter
    pub qp: u8,
}

impl SliceHeader {
    /// Parse slice header from bitstream
    pub fn parse(reader: &mut ProResBitstreamReader) -> Result<Self> {
        let slice_size = reader.read_u16()?;
        let y_pos = reader.read_u16()?;
        let qp = reader.read_u8()?;

        Ok(Self {
            slice_size,
            y_pos,
            qp,
        })
    }

    /// Encode slice header to bitstream
    pub fn encode(&self, writer: &mut ProResBitstreamWriter) {
        writer.write_u16(self.slice_size);
        writer.write_u16(self.y_pos);
        writer.write_u8(self.qp);
    }
}

/// ProRes slice containing encoded macroblocks
pub struct Slice {
    /// Slice header
    pub header: SliceHeader,
    /// Encoded macroblock data
    pub data: Vec<u8>,
}

impl Slice {
    /// Create new slice
    pub fn new(y_pos: u16, qp: u8) -> Self {
        Self {
            header: SliceHeader {
                slice_size: 0,
                y_pos,
                qp,
            },
            data: Vec::new(),
        }
    }

    /// Parse slice from bitstream
    pub fn parse(reader: &mut ProResBitstreamReader) -> Result<Self> {
        let header = SliceHeader::parse(reader)?;

        // Read slice data
        let mut data = vec![0u8; header.slice_size as usize - 5]; // 5 bytes for header
        for byte in &mut data {
            *byte = reader.read_u8()?;
        }

        Ok(Self {
            header,
            data,
        })
    }

    /// Encode slice to bitstream
    pub fn encode(&mut self, writer: &mut ProResBitstreamWriter) {
        // Update slice size
        self.header.slice_size = (5 + self.data.len()) as u16;

        // Write header
        self.header.encode(writer);

        // Write data
        for &byte in &self.data {
            writer.write_u8(byte);
        }
    }
}

/// Slice encoder for ProRes
pub struct SliceEncoder {
    profile: ProResProfile,
    width: usize,
    height: usize,
    quantizer: ProResQuantizer,
}

impl SliceEncoder {
    /// Create new slice encoder
    pub fn new(profile: ProResProfile, width: usize, height: usize, qp: u8) -> Self {
        Self {
            profile,
            width,
            height,
            quantizer: ProResQuantizer::new(profile, qp),
        }
    }

    /// Encode a slice of pixels
    pub fn encode_slice(
        &self,
        pixels: &[i16],
        y_start: usize,
        slice_height: usize,
    ) -> Result<Slice> {
        let mut writer = ProResBitstreamWriter::new();
        let qp = 16; // Default QP

        // Process 8×8 blocks
        for y in (y_start..y_start + slice_height).step_by(8) {
            for x in (0..self.width).step_by(8) {
                // Extract 8×8 block
                let mut block = [0i16; 64];
                self.extract_block(pixels, x, y, &mut block)?;

                // DCT
                let mut dct_coeffs = [0i16; 64];
                ProResDct::forward_dct(&block, &mut dct_coeffs)?;

                // Quantize
                let mut quant_coeffs = [0i16; 64];
                self.quantizer.quantize(&dct_coeffs, &mut quant_coeffs)?;

                // VLC encode
                encode_dct_coefficients(&mut writer, &quant_coeffs)?;
            }
        }

        let data = writer.finish();
        let mut slice = Slice::new(y_start as u16, qp);
        slice.data = data;

        Ok(slice)
    }

    /// Extract 8×8 block from image
    fn extract_block(&self, pixels: &[i16], x: usize, y: usize, block: &mut [i16; 64]) -> Result<()> {
        for by in 0..8 {
            for bx in 0..8 {
                let px = x + bx;
                let py = y + by;

                if px < self.width && py < self.height {
                    let idx = py * self.width + px;
                    if idx < pixels.len() {
                        block[by * 8 + bx] = pixels[idx];
                    }
                }
            }
        }
        Ok(())
    }
}

/// Slice decoder for ProRes
pub struct SliceDecoder {
    profile: ProResProfile,
    width: usize,
    height: usize,
    quantizer: ProResQuantizer,
}

impl SliceDecoder {
    /// Create new slice decoder
    pub fn new(profile: ProResProfile, width: usize, height: usize, qp: u8) -> Self {
        Self {
            profile,
            width,
            height,
            quantizer: ProResQuantizer::new(profile, qp),
        }
    }

    /// Decode a slice to pixels
    pub fn decode_slice(
        &self,
        slice: &Slice,
        pixels: &mut [i16],
    ) -> Result<()> {
        let mut reader = ProResBitstreamReader::new(&slice.data);
        let y_start = slice.header.y_pos as usize;
        let slice_height = 16; // Standard slice height

        // Process 8×8 blocks
        for y in (y_start..y_start + slice_height).step_by(8) {
            if y >= self.height {
                break;
            }

            for x in (0..self.width).step_by(8) {
                if !reader.has_more() {
                    break;
                }

                // VLC decode
                let mut quant_coeffs = [0i16; 64];
                decode_dct_coefficients(&mut reader, &mut quant_coeffs)?;

                // Dequantize
                let mut dct_coeffs = [0i16; 64];
                self.quantizer.dequantize(&quant_coeffs, &mut dct_coeffs)?;

                // IDCT
                let mut block = [0i16; 64];
                ProResDct::inverse_dct(&dct_coeffs, &mut block)?;

                // Insert block into image
                self.insert_block(pixels, x, y, &block)?;
            }
        }

        Ok(())
    }

    /// Insert 8×8 block into image
    fn insert_block(&self, pixels: &mut [i16], x: usize, y: usize, block: &[i16; 64]) -> Result<()> {
        for by in 0..8 {
            for bx in 0..8 {
                let px = x + bx;
                let py = y + by;

                if px < self.width && py < self.height {
                    let idx = py * self.width + px;
                    if idx < pixels.len() {
                        pixels[idx] = block[by * 8 + bx].clamp(-128, 127);
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_header_encode_decode() {
        let header = SliceHeader {
            slice_size: 1024,
            y_pos: 16,
            qp: 12,
        };

        let mut writer = ProResBitstreamWriter::new();
        header.encode(&mut writer);

        let data = writer.finish();
        let mut reader = ProResBitstreamReader::new(&data);

        let decoded = SliceHeader::parse(&mut reader).unwrap();
        assert_eq!(decoded.slice_size, 1024);
        assert_eq!(decoded.y_pos, 16);
        assert_eq!(decoded.qp, 12);
    }

    #[test]
    fn test_slice_creation() {
        let slice = Slice::new(0, 16);
        assert_eq!(slice.header.y_pos, 0);
        assert_eq!(slice.header.qp, 16);
    }

    #[test]
    fn test_slice_encoder_extract_block() {
        let encoder = SliceEncoder::new(ProResProfile::Standard, 16, 16, 16);
        let pixels = vec![100i16; 16 * 16];

        let mut block = [0i16; 64];
        encoder.extract_block(&pixels, 0, 0, &mut block).unwrap();

        // All values should be 100
        for &val in &block {
            assert_eq!(val, 100);
        }
    }

    #[test]
    fn test_slice_decoder_insert_block() {
        let decoder = SliceDecoder::new(ProResProfile::Standard, 16, 16, 16);
        let mut pixels = vec![0i16; 16 * 16];
        let block = [50i16; 64];

        decoder.insert_block(&mut pixels, 0, 0, &block).unwrap();

        // First 8×8 should be 50
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(pixels[y * 16 + x], 50);
            }
        }
    }

    #[test]
    fn test_slice_encode_decode_roundtrip() {
        let width = 16;
        let height = 16;
        let qp = 16;

        // Create test pixels
        let mut pixels = vec![0i16; width * height];
        for i in 0..pixels.len() {
            pixels[i] = (i % 128) as i16;
        }

        // Encode slice
        let encoder = SliceEncoder::new(ProResProfile::Standard, width, height, qp);
        let slice = encoder.encode_slice(&pixels, 0, 16).unwrap();

        // Decode slice
        let decoder = SliceDecoder::new(ProResProfile::Standard, width, height, qp);
        let mut decoded_pixels = vec![0i16; width * height];
        decoder.decode_slice(&slice, &mut decoded_pixels).unwrap();

        // Should be approximately equal (quantization is lossy)
        for i in 0..pixels.len() {
            let diff = (pixels[i] - decoded_pixels[i]).abs();
            assert!(diff < 30, "Pixel {} diff too large: {}", i, diff);
        }
    }
}
