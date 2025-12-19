//! Bitstream writer for ProRes encoding.
//!
//! This module provides utilities for writing variable-length codes (VLC)
//! used in ProRes entropy encoding. It is the inverse of the BitReader
//! and decode functions in decoder.rs.

use super::tables::{PRORES_DC_CODEBOOK, PRORES_LEVEL_TO_CB, PRORES_RUN_TO_CB};

const FIRST_DC_CODEBOOK: u8 = 0xB8;

/// Bitstream writer for encoding ProRes data.
pub struct BitWriter {
    data: Vec<u8>,
    current_byte: u8,
    bit_count: usize,
}

impl BitWriter {
    /// Create a new BitWriter with the specified initial capacity.
    pub fn new() -> Self {
        BitWriter {
            data: Vec::with_capacity(4096),
            current_byte: 0,
            bit_count: 0,
        }
    }

    /// Create a new BitWriter with preallocated capacity.
    #[allow(dead_code)]
    pub fn with_capacity(capacity: usize) -> Self {
        BitWriter {
            data: Vec::with_capacity(capacity),
            current_byte: 0,
            bit_count: 0,
        }
    }

    /// Write a single bit to the stream.
    #[inline]
    pub fn write_bit(&mut self, bit: u32) {
        self.current_byte = (self.current_byte << 1) | ((bit & 1) as u8);
        self.bit_count += 1;
        if self.bit_count == 8 {
            self.data.push(self.current_byte);
            self.current_byte = 0;
            self.bit_count = 0;
        }
    }

    /// Write multiple bits to the stream (MSB first).
    #[inline]
    pub fn write_bits(&mut self, value: u32, num_bits: usize) {
        debug_assert!(num_bits <= 32);
        for i in (0..num_bits).rev() {
            self.write_bit((value >> i) & 1);
        }
    }

    /// Flush any remaining bits, padding with zeros.
    pub fn flush(&mut self) {
        if self.bit_count > 0 {
            self.current_byte <<= 8 - self.bit_count;
            self.data.push(self.current_byte);
            self.current_byte = 0;
            self.bit_count = 0;
        }
    }

    /// Get the written data, consuming the writer.
    pub fn into_bytes(mut self) -> Vec<u8> {
        self.flush();
        self.data
    }

    /// Get a reference to the current data (without final partial byte).
    #[allow(dead_code)]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get current bit position.
    #[allow(dead_code)]
    pub fn bit_position(&self) -> usize {
        self.data.len() * 8 + self.bit_count
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert signed value to unsigned for VLC encoding.
///
/// This is the inverse of `to_signed` in decoder.rs.
/// to_signed maps: 0->0, 1->-1, 2->1, 3->-2, 4->2, 5->-3, 6->3, ...
/// So from_signed maps: 0->0, -1->1, 1->2, -2->3, 2->4, -3->5, 3->6, ...
#[inline]
pub fn from_signed(v: i32) -> u32 {
    if v == 0 {
        0
    } else if v < 0 {
        // Negative: -1->1, -2->3, -3->5 => (-v)*2 - 1
        ((-v) * 2 - 1) as u32
    } else {
        // Positive: 1->2, 2->4, 3->6 => v*2
        (v * 2) as u32
    }
}

/// Encode a codeword using the ProRes hybrid Rice/ExpGolomb VLC scheme.
///
/// This is the inverse of `decode_codeword` in decoder.rs.
/// The codebook byte encodes:
/// - bits 0-1: switch_bits
/// - bits 2-4: exp_order
/// - bits 5-7: rice_order
pub fn encode_vlc_codeword(bw: &mut BitWriter, value: u32, codebook: u8) {
    let switch_bits = (codebook & 3) as u32;
    let rice_order = (codebook >> 5) as u32;
    let exp_order = ((codebook >> 2) & 7) as u32;

    // Determine which mode to use based on the value
    // Rice coding threshold: values <= (switch_bits + 1) << rice_order can use Rice
    let rice_threshold = ((switch_bits + 1) << rice_order) as u32;

    if value < rice_threshold {
        // Rice coding
        let q = value >> rice_order;
        let r = value & ((1 << rice_order) - 1);

        // Write q zeros followed by 1
        for _ in 0..q {
            bw.write_bit(0);
        }
        bw.write_bit(1);

        // Write r bits of remainder
        if rice_order > 0 {
            bw.write_bits(r, rice_order as usize);
        }
    } else {
        // ExpGolomb coding
        // Need to find q such that: bits = exp_order - switch_bits + 2*q
        // And: value = code + (1 << exp_order) - ((switch_bits + 1) << rice_order)
        // Where code is in range [0, (1 << bits) - 1]

        let adj = value as i64 - ((switch_bits as i64 + 1) << rice_order) + (1i64 << exp_order);
        if adj < 0 {
            // Fallback to writing as Rice if calculation doesn't work
            // This shouldn't happen with valid input
            let q = value >> rice_order;
            let r = value & ((1 << rice_order) - 1);
            for _ in 0..q {
                bw.write_bit(0);
            }
            bw.write_bit(1);
            if rice_order > 0 {
                bw.write_bits(r, rice_order as usize);
            }
            return;
        }

        // Find the minimum number of bits needed
        let mut bits = exp_order as usize;
        while (1u64 << bits) <= adj as u64 {
            bits += 1;
        }

        // Calculate q from bits
        // bits = exp_order - switch_bits + 2*q
        // q = (bits - exp_order + switch_bits) / 2
        let q = if bits as u32 >= exp_order.saturating_sub(switch_bits) {
            (bits as u32 + switch_bits - exp_order + 1) / 2
        } else {
            0
        };

        // Ensure q > switch_bits (required for ExpGolomb mode)
        let q = q.max(switch_bits + 1);

        // Recalculate bits based on q
        let actual_bits = (exp_order as i32 - switch_bits as i32 + (q << 1) as i32) as usize;

        // Write q zeros followed by 1
        for _ in 0..q {
            bw.write_bit(0);
        }
        bw.write_bit(1);

        // Write the code
        bw.write_bits(adj as u32, actual_bits);
    }
}

/// Encode DC coefficients for a slice.
///
/// DC coefficients are encoded differentially with sign prediction.
/// First DC uses FIRST_DC_CODEBOOK, subsequent use adaptive codebook.
pub fn encode_dc_coeffs(bw: &mut BitWriter, coeffs: &[i16], blocks_per_slice: usize) {
    if blocks_per_slice == 0 {
        return;
    }

    // Encode first DC coefficient
    let first_dc = coeffs[0] as i32;
    encode_vlc_codeword(bw, from_signed(first_dc), FIRST_DC_CODEBOOK);

    // Encode subsequent DC coefficients differentially
    let mut prev_dc = first_dc;
    let mut code = 5u32;
    let mut sign = 0i32;

    for block in 1..blocks_per_slice {
        let idx = block * 64;
        let dc = coeffs[idx] as i32;
        let diff = dc - prev_dc;

        // Calculate the code value using sign prediction
        // This reverses the decoder logic:
        // prev_dc += (((code + 1) >> 1) as i32 ^ sign) - sign
        // So: diff = (((code + 1) >> 1) as i32 ^ sign) - sign

        // Determine the absolute difference and new sign state
        let abs_diff = diff.abs();
        let new_sign = if diff < 0 { -1i32 } else { 0i32 };

        // The code encodes: abs_diff * 2 - (sign_change ? 0 : 1)
        // But we need to match the decoder's sign prediction logic
        let sign_change = if diff != 0 { new_sign != sign } else { false };

        let code_value = if diff == 0 {
            0u32
        } else {
            let base = (abs_diff as u32) * 2;
            if sign_change {
                base - 1 // Odd code = sign flip
            } else {
                base // Even code = same sign
            }
        };

        encode_vlc_codeword(bw, code_value, PRORES_DC_CODEBOOK[code.min(6) as usize]);

        // Update state for next iteration
        code = code_value;
        if code_value != 0 {
            sign ^= -((code_value & 1) as i32);
        } else {
            sign = 0;
        }
        prev_dc = dc;
    }
}

/// Encode AC coefficients for a slice using run-level coding.
///
/// AC coefficients are encoded with interleaved block ordering,
/// similar to how the decoder reads them.
pub fn encode_ac_coeffs(
    bw: &mut BitWriter,
    coeffs: &[i16],
    blocks_per_slice: usize,
    scan: &[u8; 64],
) {
    if blocks_per_slice == 0 {
        return;
    }

    let pow2_blocks = (blocks_per_slice as u32).next_power_of_two();
    let block_mask = (pow2_blocks as usize) - 1;
    let log2_block_count = pow2_blocks.trailing_zeros() as usize;
    let max_coeffs = 64usize << log2_block_count;

    // Build list of (position, level) pairs
    // Position uses interleaved ordering
    let mut run_levels: Vec<(usize, i16)> = Vec::new();

    for pos in (block_mask + 1)..max_coeffs {
        let block_idx = pos & block_mask;
        let coeff_idx = pos >> log2_block_count;

        if block_idx >= blocks_per_slice || coeff_idx >= 64 {
            continue;
        }

        let scan_idx = scan[coeff_idx] as usize;
        let src_idx = block_idx * 64 + scan_idx;
        let level = coeffs[src_idx];

        if level != 0 {
            run_levels.push((pos, level));
        }
    }

    // Encode run-level pairs
    let mut prev_pos = block_mask;
    let mut run = 4u32;
    let mut level_ctx = 2u32;

    for &(pos, level) in &run_levels {
        let current_run = pos - prev_pos - 1;

        // Encode run
        encode_vlc_codeword(
            bw,
            current_run as u32,
            PRORES_RUN_TO_CB[run.min(15) as usize],
        );

        // Encode level (minus 1, since level is always >= 1)
        let abs_level = level.unsigned_abs() as u32;
        encode_vlc_codeword(
            bw,
            abs_level - 1,
            PRORES_LEVEL_TO_CB[level_ctx.min(9) as usize],
        );

        // Encode sign
        bw.write_bit(if level < 0 { 1 } else { 0 });

        // Update context
        run = current_run as u32;
        level_ctx = abs_level;
        prev_pos = pos;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_signed_inverse() {
        // Test that from_signed is the inverse of to_signed from decoder
        fn to_signed(v: u32) -> i32 {
            ((v >> 1) as i32) ^ -((v as i32) & 1)
        }

        for i in -100..=100 {
            let unsigned = from_signed(i);
            let back = to_signed(unsigned);
            assert_eq!(back, i, "from_signed/to_signed roundtrip failed for {}", i);
        }
    }

    #[test]
    fn test_bitwriter_basic() {
        let mut bw = BitWriter::new();
        bw.write_bit(1);
        bw.write_bit(0);
        bw.write_bit(1);
        bw.write_bit(1);
        bw.write_bit(0);
        bw.write_bit(0);
        bw.write_bit(0);
        bw.write_bit(1);
        let data = bw.into_bytes();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0], 0b10110001);
    }

    #[test]
    fn test_bitwriter_write_bits() {
        let mut bw = BitWriter::new();
        bw.write_bits(0b11010, 5);
        bw.write_bits(0b110, 3);
        let data = bw.into_bytes();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0], 0b11010110);
    }

    #[test]
    fn test_bitwriter_partial_byte() {
        let mut bw = BitWriter::new();
        bw.write_bits(0b101, 3);
        let data = bw.into_bytes();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0], 0b10100000);
    }

    #[test]
    fn test_vlc_simple_values() {
        // Test encoding small values with a simple codebook
        let codebook: u8 = 0x04; // rice_order=0, exp_order=1, switch_bits=0

        let mut bw = BitWriter::new();
        encode_vlc_codeword(&mut bw, 0, codebook);
        // Value 0 with rice_order=0: should be just "1"
        let data = bw.into_bytes();
        // First bit should be 1 (terminator), rest padded
        assert_eq!(data[0] & 0x80, 0x80);
    }

    #[test]
    fn test_encode_dc_simple() {
        let mut coeffs = vec![0i16; 128]; // 2 blocks
        coeffs[0] = 100; // First DC
        coeffs[64] = 100; // Second DC (same, so diff = 0)

        let mut bw = BitWriter::new();
        encode_dc_coeffs(&mut bw, &coeffs, 2);
        let data = bw.into_bytes();

        // Should have encoded something
        assert!(!data.is_empty());
    }
}
