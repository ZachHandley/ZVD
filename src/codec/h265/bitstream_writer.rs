//! Bitstream Writer for H.265/HEVC Encoder
//!
//! This module implements bitstream generation for H.265 encoding, including
//! CABAC entropy encoding and NAL unit construction.
//!
//! # Overview
//!
//! The bitstream writer is the final stage of the encoder pipeline:
//! - Encodes quantized coefficients using CABAC
//! - Writes VPS/SPS/PPS/Slice headers
//! - Constructs NAL units with proper start codes
//! - Handles motion vector coding
//! - Signals prediction modes and split decisions
//!
//! # CABAC Encoding
//!
//! Context-Adaptive Binary Arithmetic Coding:
//! - Context models for different syntax elements
//! - Adaptive probability estimation
//! - Arithmetic coding with range subdivision
//!
//! # NAL Unit Structure
//!
//! ```text
//! Start Code (0x000001)
//! NAL Header (2 bytes)
//! Payload (RBSP)
//! ```

use crate::codec::h265::{NalUnitType, Vps, Sps, Pps, SliceHeader, IntraMode};
use crate::codec::h265::mv::MotionVector;
use crate::error::{Error, Result};

/// Bitstream writer with bit-level precision
pub struct BitstreamWriter {
    /// Output buffer
    buffer: Vec<u8>,
    /// Current byte being written
    current_byte: u8,
    /// Number of bits written in current byte (0-7)
    bit_position: u8,
}

impl BitstreamWriter {
    /// Create new bitstream writer
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            current_byte: 0,
            bit_position: 0,
        }
    }

    /// Write a single bit
    pub fn write_bit(&mut self, bit: bool) {
        if bit {
            self.current_byte |= 1 << (7 - self.bit_position);
        }

        self.bit_position += 1;

        if self.bit_position == 8 {
            self.buffer.push(self.current_byte);
            self.current_byte = 0;
            self.bit_position = 0;
        }
    }

    /// Write multiple bits from a value
    pub fn write_bits(&mut self, value: u32, num_bits: u8) {
        for i in (0..num_bits).rev() {
            let bit = (value >> i) & 1 == 1;
            self.write_bit(bit);
        }
    }

    /// Write unsigned Exp-Golomb code
    pub fn write_ue(&mut self, value: u32) {
        let value_plus1 = value + 1;
        let leading_zeros = 31 - value_plus1.leading_zeros();

        // Write leading zeros
        for _ in 0..leading_zeros {
            self.write_bit(false);
        }

        // Write the value
        self.write_bits(value_plus1, leading_zeros as u8 + 1);
    }

    /// Write signed Exp-Golomb code
    pub fn write_se(&mut self, value: i32) {
        let mapped = if value <= 0 {
            (-value * 2) as u32
        } else {
            (value * 2 - 1) as u32
        };
        self.write_ue(mapped);
    }

    /// Flush remaining bits (byte align with 1-bits)
    pub fn byte_align(&mut self) {
        if self.bit_position > 0 {
            while self.bit_position < 8 {
                self.write_bit(true);
            }
        }
    }

    /// Get the written data
    pub fn finish(mut self) -> Vec<u8> {
        self.byte_align();
        self.buffer
    }

    /// Get number of bits written
    pub fn num_bits(&self) -> usize {
        self.buffer.len() * 8 + self.bit_position as usize
    }
}

/// CABAC (Context-Adaptive Binary Arithmetic Coding) encoder
pub struct CabacEncoder {
    /// Arithmetic coder low value
    low: u32,
    /// Arithmetic coder range
    range: u32,
    /// Outstanding bits counter
    outstanding_bits: u32,
    /// Output bitstream
    writer: BitstreamWriter,
}

impl CabacEncoder {
    /// Create new CABAC encoder
    pub fn new() -> Self {
        Self {
            low: 0,
            range: 510, // Initial range (256..=510)
            outstanding_bits: 0,
            writer: BitstreamWriter::new(),
        }
    }

    /// Encode a binary decision with context
    ///
    /// # Arguments
    /// - `bin`: Binary value (0 or 1)
    /// - `ctx_prob`: Context probability state (0-63)
    pub fn encode_bin(&mut self, bin: bool, ctx_prob: u8) {
        // LPS (Least Probable Symbol) probability
        let lps_prob = Self::get_lps_probability(ctx_prob);

        let range_lps = (self.range * lps_prob as u32) >> 8;

        if bin {
            // MPS (Most Probable Symbol)
            self.range -= range_lps;
        } else {
            // LPS
            self.low += self.range - range_lps;
            self.range = range_lps;
        }

        // Renormalization
        while self.range < 256 {
            if self.low >= 512 {
                self.write_out_bit(true);
                self.low -= 512;
            } else if self.low < 256 {
                self.write_out_bit(false);
            } else {
                self.outstanding_bits += 1;
                self.low -= 256;
            }

            self.range <<= 1;
            self.low <<= 1;
        }
    }

    /// Encode a bypass bin (equiprobable, no context)
    pub fn encode_bin_bypass(&mut self, bin: bool) {
        self.low <<= 1;
        if bin {
            self.low += self.range;
        }

        if self.low >= 1024 {
            self.write_out_bit(true);
            self.low -= 1024;
        } else if self.low < 512 {
            self.write_out_bit(false);
        } else {
            self.outstanding_bits += 1;
            self.low -= 512;
        }
    }

    /// Write out a bit with outstanding bits
    fn write_out_bit(&mut self, bit: bool) {
        self.writer.write_bit(bit);

        // Write outstanding bits (inverted)
        for _ in 0..self.outstanding_bits {
            self.writer.write_bit(!bit);
        }
        self.outstanding_bits = 0;
    }

    /// Get LPS probability from state
    fn get_lps_probability(state: u8) -> u8 {
        // Simplified LPS probability table
        match state {
            0..=15 => 128 - (state * 4),
            16..=31 => 64 - ((state - 16) * 2),
            32..=47 => 32 - (state - 32),
            _ => 16,
        }
    }

    /// Terminate CABAC encoding
    pub fn terminate(&mut self) {
        self.range -= 2;
        self.low += self.range;
        self.range = 2;

        // Renormalize and flush
        while self.range < 256 {
            if self.low >= 512 {
                self.write_out_bit(true);
                self.low -= 512;
            } else if self.low < 256 {
                self.write_out_bit(false);
            } else {
                self.outstanding_bits += 1;
                self.low -= 256;
            }
            self.range <<= 1;
            self.low <<= 1;
        }

        // Flush remaining bits
        self.write_out_bit(true);
    }

    /// Finish encoding and get bitstream
    pub fn finish(self) -> Vec<u8> {
        self.writer.finish()
    }
}

/// NAL unit writer
pub struct NalWriter;

impl NalWriter {
    /// Write NAL unit with start code
    pub fn write_nal_unit(nal_type: NalUnitType, payload: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();

        // Start code: 0x000001
        output.push(0x00);
        output.push(0x00);
        output.push(0x01);

        // NAL header (2 bytes)
        let nal_header = Self::create_nal_header(nal_type);
        output.extend_from_slice(&nal_header);

        // Payload with emulation prevention
        output.extend_from_slice(&Self::add_emulation_prevention(payload));

        output
    }

    /// Create NAL header (2 bytes)
    fn create_nal_header(nal_type: NalUnitType) -> [u8; 2] {
        let type_val = match nal_type {
            NalUnitType::VPS => 32,
            NalUnitType::SPS => 33,
            NalUnitType::PPS => 34,
            NalUnitType::IDR_W_RADL => 19,
            NalUnitType::IDR_N_LP => 20,
            NalUnitType::TRAIL_R => 1,
            _ => 0,
        };

        // Forbidden zero bit (1) + Type (6) + Layer ID (6) + Temporal ID + 1 (3)
        let byte0 = (type_val << 1) as u8;
        let byte1 = 0x01; // Layer ID = 0, Temporal ID = 0

        [byte0, byte1]
    }

    /// Add emulation prevention bytes (0x03 after 0x000000, 0x000001, 0x000002)
    fn add_emulation_prevention(data: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();
        let mut zero_count = 0;

        for &byte in data {
            if zero_count == 2 && byte <= 0x03 {
                // Insert emulation prevention byte
                output.push(0x03);
                zero_count = 0;
            }

            output.push(byte);

            if byte == 0x00 {
                zero_count += 1;
            } else {
                zero_count = 0;
            }
        }

        output
    }
}

/// Header writer for VPS/SPS/PPS
pub struct HeaderWriter;

impl HeaderWriter {
    /// Write VPS (Video Parameter Set)
    pub fn write_vps(vps: &Vps) -> Result<Vec<u8>> {
        let mut writer = BitstreamWriter::new();

        // vps_video_parameter_set_id
        writer.write_bits(vps.id as u32, 4);

        // vps_reserved_three_2bits
        writer.write_bits(3, 2);

        // vps_max_layers_minus1
        writer.write_bits(0, 6);

        // vps_max_sub_layers_minus1
        writer.write_bits(vps.max_sub_layers - 1, 3);

        // vps_temporal_id_nesting_flag
        writer.write_bit(true);

        // vps_reserved_0xffff_16bits
        writer.write_bits(0xFFFF, 16);

        // Simplified: skip profile_tier_level, etc.
        writer.byte_align();

        Ok(writer.finish())
    }

    /// Write SPS (Sequence Parameter Set)
    pub fn write_sps(sps: &Sps) -> Result<Vec<u8>> {
        let mut writer = BitstreamWriter::new();

        // sps_video_parameter_set_id
        writer.write_bits(0, 4);

        // sps_max_sub_layers_minus1
        writer.write_bits(0, 3);

        // sps_temporal_id_nesting_flag
        writer.write_bit(true);

        // Simplified profile_tier_level
        writer.write_bits(1, 2); // general_profile_space
        writer.write_bit(false); // general_tier_flag
        writer.write_bits(1, 5); // general_profile_idc (Main)

        // sps_seq_parameter_set_id
        writer.write_ue(sps.id as u32);

        // chroma_format_idc
        writer.write_ue(1); // 4:2:0

        // pic_width_in_luma_samples
        writer.write_ue(sps.width as u32);

        // pic_height_in_luma_samples
        writer.write_ue(sps.height as u32);

        // conformance_window_flag
        writer.write_bit(false);

        // bit_depth_luma_minus8
        writer.write_ue(sps.bit_depth_luma - 8);

        // bit_depth_chroma_minus8
        writer.write_ue(sps.bit_depth_chroma - 8);

        // Simplified: skip remaining fields
        writer.byte_align();

        Ok(writer.finish())
    }

    /// Write PPS (Picture Parameter Set)
    pub fn write_pps(pps: &Pps) -> Result<Vec<u8>> {
        let mut writer = BitstreamWriter::new();

        // pps_pic_parameter_set_id
        writer.write_ue(pps.id as u32);

        // pps_seq_parameter_set_id
        writer.write_ue(pps.sps_id as u32);

        // dependent_slice_segments_enabled_flag
        writer.write_bit(false);

        // output_flag_present_flag
        writer.write_bit(false);

        // num_extra_slice_header_bits
        writer.write_bits(0, 3);

        // Simplified: skip remaining fields
        writer.byte_align();

        Ok(writer.finish())
    }
}

/// Coefficient coder using CABAC
pub struct CoefficientCoder;

impl CoefficientCoder {
    /// Encode quantized coefficients
    pub fn encode_coefficients(
        cabac: &mut CabacEncoder,
        coeffs: &[i16],
        width: usize,
        height: usize,
    ) -> Result<()> {
        // Encode in reverse scan order (high freq to low freq)
        let last_nonzero = coeffs.iter().rposition(|&c| c != 0);

        if let Some(last_idx) = last_nonzero {
            // Encode last significant coefficient position
            cabac.encode_bin((last_idx & 1) != 0, 20);
            cabac.encode_bin((last_idx & 2) != 0, 20);

            // Encode coefficient levels
            for i in (0..=last_idx).rev() {
                let coeff = coeffs[i];

                if coeff != 0 {
                    // Significant flag
                    cabac.encode_bin(true, 25);

                    // Sign
                    cabac.encode_bin_bypass(coeff < 0);

                    // Absolute level - 1
                    let abs_level = (coeff.abs() - 1) as u32;
                    for bit_idx in 0..4 {
                        if abs_level > bit_idx {
                            cabac.encode_bin(true, 30 + bit_idx as u8);
                        } else {
                            cabac.encode_bin(false, 30 + bit_idx as u8);
                            break;
                        }
                    }
                } else {
                    // Not significant
                    cabac.encode_bin(false, 25);
                }
            }
        }

        Ok(())
    }
}

/// Motion vector coder
pub struct MvCoder;

impl MvCoder {
    /// Encode motion vector difference (MVD)
    pub fn encode_mvd(cabac: &mut CabacEncoder, mvd_x: i16, mvd_y: i16) -> Result<()> {
        // Encode X component
        Self::encode_mvd_component(cabac, mvd_x)?;

        // Encode Y component
        Self::encode_mvd_component(cabac, mvd_y)?;

        Ok(())
    }

    /// Encode single MVD component
    fn encode_mvd_component(cabac: &mut CabacEncoder, mvd: i16) -> Result<()> {
        let abs_mvd = mvd.abs() as u32;

        if abs_mvd == 0 {
            // Zero flag
            cabac.encode_bin(false, 40);
            return Ok(());
        }

        // Non-zero flag
        cabac.encode_bin(true, 40);

        // Greater than 1 flag
        if abs_mvd > 1 {
            cabac.encode_bin(true, 41);

            // Remaining value (abs_mvd - 2)
            let remaining = abs_mvd - 2;
            for bit_idx in 0..8 {
                if remaining > bit_idx {
                    cabac.encode_bin_bypass(true);
                } else {
                    cabac.encode_bin_bypass(false);
                    break;
                }
            }
        } else {
            cabac.encode_bin(false, 41);
        }

        // Sign
        cabac.encode_bin_bypass(mvd < 0);

        Ok(())
    }
}

/// Intra mode coder
pub struct IntraModeCoder;

impl IntraModeCoder {
    /// Encode intra prediction mode
    pub fn encode_mode(cabac: &mut CabacEncoder, mode: IntraMode, mpm_list: &[IntraMode; 3]) -> Result<()> {
        // Check if mode is in MPM list
        let mpm_idx = mpm_list.iter().position(|&m| m == mode);

        if let Some(idx) = mpm_idx {
            // Mode is in MPM list
            cabac.encode_bin(true, 50);

            // Encode index (0, 1, or 2)
            if idx == 0 {
                cabac.encode_bin_bypass(false);
            } else {
                cabac.encode_bin_bypass(true);
                cabac.encode_bin_bypass(idx == 2);
            }
        } else {
            // Mode is not in MPM list
            cabac.encode_bin(false, 50);

            // Encode mode index (0-34, excluding MPM modes)
            let mode_idx = Self::mode_to_index(mode);
            cabac.encode_bin_bypass((mode_idx & 16) != 0);
            cabac.encode_bin_bypass((mode_idx & 8) != 0);
            cabac.encode_bin_bypass((mode_idx & 4) != 0);
            cabac.encode_bin_bypass((mode_idx & 2) != 0);
            cabac.encode_bin_bypass((mode_idx & 1) != 0);
        }

        Ok(())
    }

    /// Convert intra mode to index
    fn mode_to_index(mode: IntraMode) -> u32 {
        match mode {
            IntraMode::Planar => 0,
            IntraMode::DC => 1,
            IntraMode::Angular(idx) => idx as u32 + 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitstream_writer_creation() {
        let writer = BitstreamWriter::new();
        assert_eq!(writer.num_bits(), 0);
    }

    #[test]
    fn test_write_single_bit() {
        let mut writer = BitstreamWriter::new();
        writer.write_bit(true);
        writer.write_bit(false);
        writer.write_bit(true);
        writer.write_bit(false);
        writer.write_bit(true);
        writer.write_bit(false);
        writer.write_bit(true);
        writer.write_bit(false);

        let output = writer.finish();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], 0b10101010);
    }

    #[test]
    fn test_write_bits() {
        let mut writer = BitstreamWriter::new();
        writer.write_bits(0b1011, 4);
        writer.write_bits(0b0110, 4);

        let output = writer.finish();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], 0b10110110);
    }

    #[test]
    fn test_write_ue() {
        let mut writer = BitstreamWriter::new();
        writer.write_ue(0); // 1
        writer.write_ue(1); // 010
        writer.write_ue(2); // 011

        let bits = writer.num_bits();
        assert!(bits > 0);
    }

    #[test]
    fn test_write_se() {
        let mut writer = BitstreamWriter::new();
        writer.write_se(0);  // 1
        writer.write_se(1);  // 010
        writer.write_se(-1); // 011

        let bits = writer.num_bits();
        assert!(bits > 0);
    }

    #[test]
    fn test_byte_align() {
        let mut writer = BitstreamWriter::new();
        writer.write_bits(0b101, 3);
        writer.byte_align();

        let output = writer.finish();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], 0b10111111);
    }

    #[test]
    fn test_cabac_encoder_creation() {
        let cabac = CabacEncoder::new();
        assert_eq!(cabac.range, 510);
        assert_eq!(cabac.low, 0);
    }

    #[test]
    fn test_cabac_encode_bin() {
        let mut cabac = CabacEncoder::new();
        cabac.encode_bin(true, 32);
        cabac.encode_bin(false, 32);
        cabac.terminate();

        let output = cabac.finish();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_cabac_encode_bin_bypass() {
        let mut cabac = CabacEncoder::new();
        cabac.encode_bin_bypass(true);
        cabac.encode_bin_bypass(false);
        cabac.encode_bin_bypass(true);
        cabac.terminate();

        let output = cabac.finish();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_nal_writer_header() {
        let header = NalWriter::create_nal_header(NalUnitType::VPS);
        assert_eq!(header[0] >> 1, 32);
    }

    #[test]
    fn test_nal_writer_write_unit() {
        let payload = vec![0x01, 0x02, 0x03];
        let nal = NalWriter::write_nal_unit(NalUnitType::SPS, &payload);

        assert!(nal.len() >= 6); // Start code + header + payload
        assert_eq!(&nal[0..3], &[0x00, 0x00, 0x01]);
    }

    #[test]
    fn test_emulation_prevention() {
        let data = vec![0x00, 0x00, 0x00];
        let output = NalWriter::add_emulation_prevention(&data);

        assert_eq!(output.len(), 4);
        assert_eq!(output[2], 0x03); // Emulation prevention byte
    }

    #[test]
    fn test_coefficient_coder() {
        let mut cabac = CabacEncoder::new();
        let coeffs = vec![10i16, 5, 0, 0, -3, 0, 0, 0];

        let result = CoefficientCoder::encode_coefficients(&mut cabac, &coeffs, 4, 2);
        assert!(result.is_ok());
    }

    #[test]
    fn test_coefficient_coder_all_zero() {
        let mut cabac = CabacEncoder::new();
        let coeffs = vec![0i16; 16];

        let result = CoefficientCoder::encode_coefficients(&mut cabac, &coeffs, 4, 4);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mv_coder_zero() {
        let mut cabac = CabacEncoder::new();
        let result = MvCoder::encode_mvd(&mut cabac, 0, 0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mv_coder_nonzero() {
        let mut cabac = CabacEncoder::new();
        let result = MvCoder::encode_mvd(&mut cabac, 8, -4);
        assert!(result.is_ok());
    }

    #[test]
    fn test_intra_mode_coder() {
        let mut cabac = CabacEncoder::new();
        let mpm_list = [IntraMode::Planar, IntraMode::DC, IntraMode::Angular(26)];

        let result = IntraModeCoder::encode_mode(&mut cabac, IntraMode::Planar, &mpm_list);
        assert!(result.is_ok());
    }

    #[test]
    fn test_intra_mode_coder_non_mpm() {
        let mut cabac = CabacEncoder::new();
        let mpm_list = [IntraMode::Planar, IntraMode::DC, IntraMode::Angular(26)];

        let result = IntraModeCoder::encode_mode(&mut cabac, IntraMode::Angular(10), &mpm_list);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mode_to_index() {
        assert_eq!(IntraModeCoder::mode_to_index(IntraMode::Planar), 0);
        assert_eq!(IntraModeCoder::mode_to_index(IntraMode::DC), 1);
        assert_eq!(IntraModeCoder::mode_to_index(IntraMode::Angular(0)), 2);
    }
}
