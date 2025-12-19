//! VP9 Entropy Coding (Coefficient Decoding)
//!
//! VP9 uses context-adaptive binary arithmetic coding for coefficient decoding.
//! The context depends on:
//! - Transform size
//! - Plane type (Y or UV)
//! - Coefficient position (band)
//! - Whether previous coefficient was zero

use super::range_coder::RangeCoder;
use super::tables::{TxSize, COEF_BANDS_4X4, COEF_BANDS_8X8, DEFAULT_SCAN_4X4, DEFAULT_SCAN_8X8};

// =============================================================================
// Token Definitions
// =============================================================================

/// Coefficient tokens
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Token {
    Zero = 0,      // 0
    One = 1,       // 1
    Two = 2,       // 2
    Three = 3,     // 3
    Four = 4,      // 4
    Cat1 = 5,      // 5-6 (category 1)
    Cat2 = 6,      // 7-10 (category 2)
    Cat3 = 7,      // 11-18 (category 3)
    Cat4 = 8,      // 19-34 (category 4)
    Cat5 = 9,      // 35-66 (category 5)
    Cat6 = 10,     // 67+ (category 6)
    EobToken = 11, // End of block
}

/// Token tree for coefficient decoding
/// Binary tree: positive values are next node index, negative are -token
const TOKEN_TREE: [i8; 22] = [
    -11, 2, // EOB or continue?
    -0, 4, // Zero or non-zero?
    -1, 6, // One or larger?
    8, 12, // Two-Four or larger?
    -2, 10, // Two or Three/Four?
    -3, -4, // Three or Four
    14, 18, // Cat1-2 or Cat3+?
    -5, 16, // Cat1 or Cat2?
    -6, -7, // Cat1 value or Cat2 value (placeholder indices)
    -8, 20, // Cat3-4 or Cat5-6?
    -9, -10, // Cat5 or Cat6
];

/// Extra bits for each category
const CAT_EXTRA_BITS: [u8; 6] = [1, 2, 3, 4, 5, 14]; // Cat1-Cat6

/// Base values for each category
const CAT_BASE_VALUES: [i16; 6] = [5, 7, 11, 19, 35, 67];

// =============================================================================
// Probability Tables
// =============================================================================

/// Default coefficient probabilities
/// Indexed by [tx_size][plane_type][ref_type][band][context][prob_index]
/// This is a simplified version - full VP9 has much larger tables
pub type CoefProbs = [[[[[[u8; 3]; 6]; 6]; 2]; 2]; 4];

/// Get default coefficient probabilities
pub fn default_coef_probs() -> CoefProbs {
    // Initialize with reasonable default values
    // In practice, these would be loaded from the bitstream or defaults
    let mut probs = [[[[[[128u8; 3]; 6]; 6]; 2]; 2]; 4];

    // Set some typical probability values
    // [eob, zero, one]
    for tx_size in 0..4 {
        for plane in 0..2 {
            for ref_type in 0..2 {
                for band in 0..6 {
                    for ctx in 0..6 {
                        // EOB probability decreases with band
                        probs[tx_size][plane][ref_type][band][ctx][0] =
                            (240 - band as u8 * 20).max(20);
                        // Zero probability
                        probs[tx_size][plane][ref_type][band][ctx][1] = 128;
                        // One probability
                        probs[tx_size][plane][ref_type][band][ctx][2] = 160;
                    }
                }
            }
        }
    }

    probs
}

// =============================================================================
// Coefficient Decoding
// =============================================================================

/// Coefficient decoder context
pub struct CoefDecoder {
    /// Probability tables
    probs: CoefProbs,
}

impl CoefDecoder {
    /// Create a new coefficient decoder with default probabilities
    pub fn new() -> Self {
        CoefDecoder {
            probs: default_coef_probs(),
        }
    }

    /// Reset to default probabilities
    pub fn reset(&mut self) {
        self.probs = default_coef_probs();
    }

    /// Decode a block of coefficients
    ///
    /// Returns (coefficients, eob) where eob is the end-of-block position
    pub fn decode_block(
        &self,
        reader: &mut RangeCoder,
        tx_size: TxSize,
        plane: usize,
        is_inter: bool,
        dc_context: usize,
    ) -> (Vec<i16>, usize) {
        let num_coeffs = tx_size.num_coeffs();
        let mut coeffs = vec![0i16; num_coeffs];

        let tx_idx = tx_size as usize;
        let plane_type = if plane == 0 { 0 } else { 1 };
        let ref_type = if is_inter { 1 } else { 0 };

        let scan = get_scan_order(tx_size);
        let bands = get_band_table(tx_size);

        let mut eob = 0;
        let mut context = dc_context.min(5);

        for i in 0..num_coeffs {
            let scan_idx = if i < scan.len() { scan[i] as usize } else { i };
            let band = if scan_idx < bands.len() {
                bands[scan_idx] as usize
            } else {
                5
            };
            let band = band.min(5);
            let ctx = context.min(5);

            let probs = &self.probs[tx_idx][plane_type][ref_type][band][ctx];

            // Check for EOB
            if i > 0 {
                let eob_prob = probs[0];
                if reader.read_bool(eob_prob) {
                    // EOB - stop decoding
                    break;
                }
            }

            // Decode coefficient
            let coeff = self.decode_coeff(reader, probs);

            if coeff != 0 {
                coeffs[scan_idx] = coeff;
                eob = i + 1;
                context = 2; // Non-zero context
            } else {
                context = if context == 0 { 0 } else { 1 }; // Zero context
            }
        }

        (coeffs, eob)
    }

    /// Decode a single coefficient value
    fn decode_coeff(&self, reader: &mut RangeCoder, probs: &[u8; 3]) -> i16 {
        // Check if zero
        if reader.read_bool(probs[1]) {
            return 0;
        }

        // Non-zero - decode magnitude
        let magnitude = self.decode_magnitude(reader, probs[2]);

        // Read sign
        let sign = reader.read_bit();

        if sign {
            -magnitude
        } else {
            magnitude
        }
    }

    /// Decode coefficient magnitude
    fn decode_magnitude(&self, reader: &mut RangeCoder, one_prob: u8) -> i16 {
        // Check if 1
        if reader.read_bool(one_prob) {
            return 1;
        }

        // Check for small values (2, 3, 4)
        if reader.read_bool(170) {
            if reader.read_bool(140) {
                return 2;
            } else if reader.read_bool(128) {
                return 3;
            } else {
                return 4;
            }
        }

        // Larger values - use categories
        if reader.read_bool(140) {
            // Cat 1-2
            if reader.read_bool(128) {
                // Cat 1 (5-6)
                let extra = reader.read_literal(1) as i16;
                return 5 + extra;
            } else {
                // Cat 2 (7-10)
                let extra = reader.read_literal(2) as i16;
                return 7 + extra;
            }
        }

        // Cat 3+ - use more bits
        if reader.read_bool(140) {
            if reader.read_bool(128) {
                // Cat 3 (11-18)
                let extra = reader.read_literal(3) as i16;
                return 11 + extra;
            } else {
                // Cat 4 (19-34)
                let extra = reader.read_literal(4) as i16;
                return 19 + extra;
            }
        }

        // Cat 5-6
        if reader.read_bool(128) {
            // Cat 5 (35-66)
            let extra = reader.read_literal(5) as i16;
            35 + extra
        } else {
            // Cat 6 (67+)
            let extra = reader.read_literal(14) as i16;
            67 + extra
        }
    }

    /// Decode a 4x4 block
    pub fn decode_4x4(
        &self,
        reader: &mut RangeCoder,
        plane: usize,
        is_inter: bool,
        dc_context: usize,
    ) -> ([i16; 16], usize) {
        let (coeffs, eob) = self.decode_block(reader, TxSize::Tx4x4, plane, is_inter, dc_context);
        let mut result = [0i16; 16];
        for (i, &c) in coeffs.iter().enumerate().take(16) {
            result[i] = c;
        }
        (result, eob)
    }

    /// Decode an 8x8 block
    pub fn decode_8x8(
        &self,
        reader: &mut RangeCoder,
        plane: usize,
        is_inter: bool,
        dc_context: usize,
    ) -> ([i16; 64], usize) {
        let (coeffs, eob) = self.decode_block(reader, TxSize::Tx8x8, plane, is_inter, dc_context);
        let mut result = [0i16; 64];
        for (i, &c) in coeffs.iter().enumerate().take(64) {
            result[i] = c;
        }
        (result, eob)
    }

    /// Decode a 16x16 block
    pub fn decode_16x16(
        &self,
        reader: &mut RangeCoder,
        plane: usize,
        is_inter: bool,
        dc_context: usize,
    ) -> ([i16; 256], usize) {
        let (coeffs, eob) = self.decode_block(reader, TxSize::Tx16x16, plane, is_inter, dc_context);
        let mut result = [0i16; 256];
        for (i, &c) in coeffs.iter().enumerate().take(256) {
            result[i] = c;
        }
        (result, eob)
    }

    /// Decode a 32x32 block
    pub fn decode_32x32(
        &self,
        reader: &mut RangeCoder,
        plane: usize,
        is_inter: bool,
        dc_context: usize,
    ) -> ([i16; 1024], usize) {
        let (coeffs, eob) = self.decode_block(reader, TxSize::Tx32x32, plane, is_inter, dc_context);
        let mut result = [0i16; 1024];
        for (i, &c) in coeffs.iter().enumerate().take(1024) {
            result[i] = c;
        }
        (result, eob)
    }
}

impl Default for CoefDecoder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Scan Order and Band Tables
// =============================================================================

/// Get scan order for transform size
fn get_scan_order(tx_size: TxSize) -> &'static [u8] {
    match tx_size {
        TxSize::Tx4x4 => &DEFAULT_SCAN_4X4,
        TxSize::Tx8x8 => &DEFAULT_SCAN_8X8,
        _ => &DEFAULT_SCAN_8X8, // Use 8x8 as fallback
    }
}

/// Get band table for transform size
fn get_band_table(tx_size: TxSize) -> &'static [u8] {
    match tx_size {
        TxSize::Tx4x4 => &COEF_BANDS_4X4,
        TxSize::Tx8x8 => &COEF_BANDS_8X8,
        _ => &COEF_BANDS_8X8, // Use 8x8 as fallback
    }
}

// =============================================================================
// Context Calculation
// =============================================================================

/// Calculate DC coefficient context from neighbors
pub fn calc_dc_context(above_nonzero: bool, left_nonzero: bool) -> usize {
    (above_nonzero as usize) + (left_nonzero as usize)
}

/// Calculate context from previous coefficient
pub fn calc_context_from_prev(prev_coeff: i16) -> usize {
    if prev_coeff == 0 {
        0
    } else if prev_coeff.abs() == 1 {
        1
    } else {
        2
    }
}

/// Token context for coefficient decoding
#[derive(Clone, Default)]
pub struct TokenContext {
    /// Above non-zero flags (one per 4x4 block column)
    pub above: Vec<bool>,
    /// Left non-zero flags (one per 4x4 block row)
    pub left: Vec<bool>,
}

impl TokenContext {
    /// Create new token context for given dimensions
    pub fn new(width_mi: usize, height_mi: usize) -> Self {
        TokenContext {
            above: vec![false; width_mi * 2], // 2 per mi unit for 4x4 granularity
            left: vec![false; height_mi * 2],
        }
    }

    /// Get context for position
    pub fn get_context(&self, mi_col: usize, mi_row: usize, plane: usize) -> usize {
        let scale = if plane == 0 { 2 } else { 1 };
        let col = mi_col * scale;
        let row = mi_row * scale;

        let above = col < self.above.len() && self.above[col];
        let left = row < self.left.len() && self.left[row];

        calc_dc_context(above, left)
    }

    /// Update context after decoding a block
    pub fn update(
        &mut self,
        mi_col: usize,
        mi_row: usize,
        plane: usize,
        has_coeffs: bool,
        size: usize,
    ) {
        let scale = if plane == 0 { 2 } else { 1 };
        let col_start = mi_col * scale;
        let row_start = mi_row * scale;
        let units = size / 4;

        for i in 0..units {
            if col_start + i < self.above.len() {
                self.above[col_start + i] = has_coeffs;
            }
            if row_start + i < self.left.len() {
                self.left[row_start + i] = has_coeffs;
            }
        }
    }

    /// Clear left context for new superblock row
    pub fn clear_left(&mut self) {
        self.left.fill(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coef_decoder_creation() {
        let decoder = CoefDecoder::new();
        // Just verify it creates without panicking
        assert!(true);
    }

    #[test]
    fn test_calc_dc_context() {
        assert_eq!(calc_dc_context(false, false), 0);
        assert_eq!(calc_dc_context(true, false), 1);
        assert_eq!(calc_dc_context(false, true), 1);
        assert_eq!(calc_dc_context(true, true), 2);
    }

    #[test]
    fn test_calc_context_from_prev() {
        assert_eq!(calc_context_from_prev(0), 0);
        assert_eq!(calc_context_from_prev(1), 1);
        assert_eq!(calc_context_from_prev(-1), 1);
        assert_eq!(calc_context_from_prev(5), 2);
        assert_eq!(calc_context_from_prev(-100), 2);
    }

    #[test]
    fn test_token_context() {
        let mut ctx = TokenContext::new(10, 10);

        // Initial context should be 0
        assert_eq!(ctx.get_context(0, 0, 0), 0);

        // Update and check
        ctx.update(0, 0, 0, true, 8);
        assert_eq!(ctx.get_context(1, 0, 0), 1);
    }

    #[test]
    fn test_scan_order_sizes() {
        assert_eq!(get_scan_order(TxSize::Tx4x4).len(), 16);
        assert_eq!(get_scan_order(TxSize::Tx8x8).len(), 64);
    }

    #[test]
    fn test_band_table_sizes() {
        assert_eq!(get_band_table(TxSize::Tx4x4).len(), 16);
        assert_eq!(get_band_table(TxSize::Tx8x8).len(), 64);
    }
}
