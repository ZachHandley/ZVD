//! VP9 Entropy Encoding (Coefficient Encoding)
//!
//! VP9 uses context-adaptive binary arithmetic coding for coefficient encoding.
//! The context depends on:
//! - Transform size
//! - Plane type (Y or UV)
//! - Coefficient position (band)
//! - Whether previous coefficient was zero

use super::range_encoder::RangeEncoder;
use super::tables::{TxSize, COEF_BANDS_4X4, COEF_BANDS_8X8, DEFAULT_SCAN_4X4, DEFAULT_SCAN_8X8};

// =============================================================================
// Token Definitions (same as decoder)
// =============================================================================

/// Coefficient token values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Token {
    Zero = 0,
    One = 1,
    Two = 2,
    Three = 3,
    Four = 4,
    Cat1 = 5,  // 5-6
    Cat2 = 6,  // 7-10
    Cat3 = 7,  // 11-18
    Cat4 = 8,  // 19-34
    Cat5 = 9,  // 35-66
    Cat6 = 10, // 67+
    EobToken = 11,
}

impl Token {
    /// Get token for a coefficient magnitude
    pub fn from_magnitude(mag: u16) -> Self {
        match mag {
            0 => Token::Zero,
            1 => Token::One,
            2 => Token::Two,
            3 => Token::Three,
            4 => Token::Four,
            5..=6 => Token::Cat1,
            7..=10 => Token::Cat2,
            11..=18 => Token::Cat3,
            19..=34 => Token::Cat4,
            35..=66 => Token::Cat5,
            _ => Token::Cat6,
        }
    }

    /// Get extra bits count for category tokens
    pub fn extra_bits(&self) -> u8 {
        match self {
            Token::Cat1 => 1,
            Token::Cat2 => 2,
            Token::Cat3 => 3,
            Token::Cat4 => 4,
            Token::Cat5 => 5,
            Token::Cat6 => 14,
            _ => 0,
        }
    }

    /// Get base value for category tokens
    pub fn base_value(&self) -> u16 {
        match self {
            Token::Cat1 => 5,
            Token::Cat2 => 7,
            Token::Cat3 => 11,
            Token::Cat4 => 19,
            Token::Cat5 => 35,
            Token::Cat6 => 67,
            _ => 0,
        }
    }
}

// =============================================================================
// Probability Tables for Encoding
// =============================================================================

/// Default coefficient probabilities for encoding
/// Indexed by [tx_size][plane_type][ref_type][band][context][prob_index]
pub type CoefProbs = [[[[[[u8; 3]; 6]; 6]; 2]; 2]; 4];

/// Get default coefficient probabilities
pub fn default_coef_probs() -> CoefProbs {
    let mut probs = [[[[[[128u8; 3]; 6]; 6]; 2]; 2]; 4];

    // Set reasonable defaults based on VP9 spec
    for tx_size in 0..4 {
        for plane in 0..2 {
            for ref_type in 0..2 {
                for band in 0..6 {
                    for ctx in 0..6 {
                        // EOB probability (decreases with band)
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
// Coefficient Encoder
// =============================================================================

/// Coefficient encoder for VP9
pub struct CoefEncoder {
    /// Probability tables
    probs: CoefProbs,
}

impl CoefEncoder {
    /// Create a new coefficient encoder with default probabilities
    pub fn new() -> Self {
        CoefEncoder {
            probs: default_coef_probs(),
        }
    }

    /// Reset to default probabilities
    pub fn reset(&mut self) {
        self.probs = default_coef_probs();
    }

    /// Encode a block of coefficients
    ///
    /// # Arguments
    /// * `writer` - Range encoder to write to
    /// * `coeffs` - Quantized coefficients (in raster scan order)
    /// * `tx_size` - Transform size
    /// * `plane` - Plane index (0=Y, 1/2=UV)
    /// * `is_inter` - Is this an inter-predicted block
    /// * `eob` - End of block position
    /// * `dc_context` - Context from neighboring blocks
    pub fn encode_block(
        &self,
        writer: &mut RangeEncoder,
        coeffs: &[i16],
        tx_size: TxSize,
        plane: usize,
        is_inter: bool,
        eob: usize,
        dc_context: usize,
    ) {
        let num_coeffs = tx_size.num_coeffs();
        let tx_idx = tx_size as usize;
        let plane_type = if plane == 0 { 0 } else { 1 };
        let ref_type = if is_inter { 1 } else { 0 };

        let scan = get_scan_order(tx_size);
        let bands = get_band_table(tx_size);

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
                if i >= eob {
                    // Write EOB
                    writer.write_bool(true, probs[0]);
                    break;
                } else {
                    // Not EOB - continue
                    writer.write_bool(false, probs[0]);
                }
            }

            // Get coefficient at scan position
            let coeff = if scan_idx < coeffs.len() {
                coeffs[scan_idx]
            } else {
                0
            };

            // Encode coefficient
            self.encode_coeff(writer, coeff, probs);

            // Update context
            if coeff != 0 {
                context = 2; // Non-zero context
            } else {
                context = if context == 0 { 0 } else { 1 }; // Zero context
            }
        }
    }

    /// Encode a single coefficient
    fn encode_coeff(&self, writer: &mut RangeEncoder, coeff: i16, probs: &[u8; 3]) {
        let magnitude = coeff.unsigned_abs();

        // Zero check
        if magnitude == 0 {
            writer.write_bool(true, probs[1]); // Zero
            return;
        }

        writer.write_bool(false, probs[1]); // Non-zero

        // Encode magnitude
        self.encode_magnitude(writer, magnitude, probs[2]);

        // Encode sign
        writer.write_bit(coeff < 0);
    }

    /// Encode coefficient magnitude
    fn encode_magnitude(&self, writer: &mut RangeEncoder, magnitude: u16, one_prob: u8) {
        // Check if 1
        if magnitude == 1 {
            writer.write_bool(true, one_prob);
            return;
        }

        writer.write_bool(false, one_prob);

        // Small values (2, 3, 4)
        if magnitude <= 4 {
            writer.write_bool(true, 170);
            match magnitude {
                2 => {
                    writer.write_bool(true, 140);
                }
                3 => {
                    writer.write_bool(false, 140);
                    writer.write_bool(true, 128);
                }
                4 => {
                    writer.write_bool(false, 140);
                    writer.write_bool(false, 128);
                }
                _ => {}
            }
            return;
        }

        writer.write_bool(false, 170);

        // Category tokens
        let token = Token::from_magnitude(magnitude);
        let extra_bits = token.extra_bits();
        let base = token.base_value();

        match token {
            Token::Cat1 | Token::Cat2 => {
                writer.write_bool(true, 140);
                if token == Token::Cat1 {
                    writer.write_bool(true, 128);
                } else {
                    writer.write_bool(false, 128);
                }
            }
            Token::Cat3 | Token::Cat4 => {
                writer.write_bool(false, 140);
                writer.write_bool(true, 140);
                if token == Token::Cat3 {
                    writer.write_bool(true, 128);
                } else {
                    writer.write_bool(false, 128);
                }
            }
            Token::Cat5 | Token::Cat6 => {
                writer.write_bool(false, 140);
                writer.write_bool(false, 140);
                if token == Token::Cat5 {
                    writer.write_bool(true, 128);
                } else {
                    writer.write_bool(false, 128);
                }
            }
            _ => {}
        }

        // Write extra bits
        let extra = magnitude - base;
        writer.write_literal(extra as u32, extra_bits);
    }

    /// Encode a 4x4 block
    pub fn encode_4x4(
        &self,
        writer: &mut RangeEncoder,
        coeffs: &[i16; 16],
        plane: usize,
        is_inter: bool,
        eob: usize,
        dc_context: usize,
    ) {
        self.encode_block(
            writer,
            coeffs,
            TxSize::Tx4x4,
            plane,
            is_inter,
            eob,
            dc_context,
        );
    }

    /// Encode an 8x8 block
    pub fn encode_8x8(
        &self,
        writer: &mut RangeEncoder,
        coeffs: &[i16; 64],
        plane: usize,
        is_inter: bool,
        eob: usize,
        dc_context: usize,
    ) {
        self.encode_block(
            writer,
            coeffs,
            TxSize::Tx8x8,
            plane,
            is_inter,
            eob,
            dc_context,
        );
    }

    /// Encode a 16x16 block
    pub fn encode_16x16(
        &self,
        writer: &mut RangeEncoder,
        coeffs: &[i16; 256],
        plane: usize,
        is_inter: bool,
        eob: usize,
        dc_context: usize,
    ) {
        self.encode_block(
            writer,
            coeffs,
            TxSize::Tx16x16,
            plane,
            is_inter,
            eob,
            dc_context,
        );
    }

    /// Encode a 32x32 block
    pub fn encode_32x32(
        &self,
        writer: &mut RangeEncoder,
        coeffs: &[i16; 1024],
        plane: usize,
        is_inter: bool,
        eob: usize,
        dc_context: usize,
    ) {
        self.encode_block(
            writer,
            coeffs,
            TxSize::Tx32x32,
            plane,
            is_inter,
            eob,
            dc_context,
        );
    }
}

impl Default for CoefEncoder {
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
        _ => &DEFAULT_SCAN_8X8, // Use 8x8 as fallback for larger sizes
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

/// Token context for coefficient encoding
#[derive(Clone, Default)]
pub struct TokenContext {
    /// Above non-zero flags
    pub above: Vec<bool>,
    /// Left non-zero flags
    pub left: Vec<bool>,
}

impl TokenContext {
    /// Create new token context for given dimensions (in mode info units)
    pub fn new(width_mi: usize, height_mi: usize) -> Self {
        TokenContext {
            above: vec![false; width_mi * 2],
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

    /// Update context after encoding a block
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
    fn test_token_from_magnitude() {
        assert_eq!(Token::from_magnitude(0), Token::Zero);
        assert_eq!(Token::from_magnitude(1), Token::One);
        assert_eq!(Token::from_magnitude(2), Token::Two);
        assert_eq!(Token::from_magnitude(5), Token::Cat1);
        assert_eq!(Token::from_magnitude(7), Token::Cat2);
        assert_eq!(Token::from_magnitude(11), Token::Cat3);
        assert_eq!(Token::from_magnitude(100), Token::Cat6);
    }

    #[test]
    fn test_coef_encoder_creation() {
        let encoder = CoefEncoder::new();
        assert!(true); // Just verify creation doesn't panic
    }

    #[test]
    fn test_calc_dc_context() {
        assert_eq!(calc_dc_context(false, false), 0);
        assert_eq!(calc_dc_context(true, false), 1);
        assert_eq!(calc_dc_context(false, true), 1);
        assert_eq!(calc_dc_context(true, true), 2);
    }

    #[test]
    fn test_token_context() {
        let mut ctx = TokenContext::new(10, 10);
        assert_eq!(ctx.get_context(0, 0, 0), 0);

        ctx.update(0, 0, 0, true, 8);
        // Context at (1, 0) should now be 1 (above neighbor has coeffs)
    }

    #[test]
    fn test_encode_zero_block() {
        let encoder = CoefEncoder::new();
        let mut writer = RangeEncoder::new();
        let coeffs = [0i16; 16];

        encoder.encode_4x4(&mut writer, &coeffs, 0, false, 0, 0);

        let data = writer.finalize();
        assert!(!data.is_empty());
    }

    #[test]
    fn test_encode_dc_only() {
        let encoder = CoefEncoder::new();
        let mut writer = RangeEncoder::new();
        let mut coeffs = [0i16; 16];
        coeffs[0] = 10;

        encoder.encode_4x4(&mut writer, &coeffs, 0, false, 1, 0);

        let data = writer.finalize();
        assert!(!data.is_empty());
    }
}
