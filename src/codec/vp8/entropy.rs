//! VP8 entropy coding - coefficient decoding
//!
//! This module implements the coefficient token decoding used for
//! DCT coefficients in VP8.

use super::bool_decoder::BoolDecoder;
use super::tables::{
    CAT1_PROB, CAT2_PROB, CAT3_PROB, CAT4_PROB, CAT5_PROB, CAT6_PROB, COEFF_BANDS,
    COEFF_TOKEN_TREE, DEFAULT_COEFF_PROBS, NUM_BLOCK_TYPES, NUM_COEFF_BANDS, NUM_ENTROPY_NODES,
    NUM_PREV_COEFF_CONTEXTS, TOKEN_EXTRA_BITS, ZIGZAG_SCAN,
};

/// DCT coefficient token values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Token {
    DctEob = 0,   // End of block
    Dct0 = 1,     // Zero
    Dct1 = 2,     // +1 or -1
    Dct2 = 3,     // +2 or -2
    Dct3 = 4,     // +3 or -3
    Dct4 = 5,     // +4 or -4
    DctCat1 = 6,  // 5-6 (1 extra bit)
    DctCat2 = 7,  // 7-10 (2 extra bits)
    DctCat3 = 8,  // 11-18 (3 extra bits)
    DctCat4 = 9,  // 19-34 (4 extra bits)
    DctCat5 = 10, // 35-66 (5 extra bits)
    DctCat6 = 11, // 67-2048 (11 extra bits)
}

impl Token {
    fn from_u8(val: u8) -> Self {
        match val {
            0 => Token::DctEob,
            1 => Token::Dct0,
            2 => Token::Dct1,
            3 => Token::Dct2,
            4 => Token::Dct3,
            5 => Token::Dct4,
            6 => Token::DctCat1,
            7 => Token::DctCat2,
            8 => Token::DctCat3,
            9 => Token::DctCat4,
            10 => Token::DctCat5,
            11 => Token::DctCat6,
            _ => Token::DctEob,
        }
    }
}

/// Coefficient probabilities for the current frame
#[derive(Clone)]
pub struct CoeffProbs {
    pub probs:
        [[[[u8; NUM_ENTROPY_NODES]; NUM_PREV_COEFF_CONTEXTS]; NUM_COEFF_BANDS]; NUM_BLOCK_TYPES],
}

impl Default for CoeffProbs {
    fn default() -> Self {
        CoeffProbs {
            probs: DEFAULT_COEFF_PROBS,
        }
    }
}

impl CoeffProbs {
    /// Update coefficient probabilities from bitstream
    pub fn update(&mut self, bd: &mut BoolDecoder, update_probs: &[[[u8; 11]; 8]; 4]) {
        for block_type in 0..NUM_BLOCK_TYPES {
            for band in 0..NUM_COEFF_BANDS {
                for ctx in 0..NUM_PREV_COEFF_CONTEXTS {
                    for node in 0..NUM_ENTROPY_NODES {
                        let update_prob = update_probs[block_type][band][node];
                        if bd.read_bool(update_prob) {
                            self.probs[block_type][band][ctx][node] = bd.read_literal(8) as u8;
                        }
                    }
                }
            }
        }
    }
}

/// Decode a token from the coefficient tree
fn decode_token(bd: &mut BoolDecoder, probs: &[u8; NUM_ENTROPY_NODES]) -> Token {
    let mut node = 0usize;
    loop {
        let prob = probs[node];
        let bit = bd.read_bool(prob) as usize;
        let next = COEFF_TOKEN_TREE[node * 2 + bit];
        if next <= 0 {
            return Token::from_u8((-next) as u8);
        }
        node = next as usize;
    }
}

/// Read extra bits for coefficient categories
fn read_extra_bits(bd: &mut BoolDecoder, token: Token) -> i16 {
    match token {
        Token::DctCat1 => {
            let extra = bd.read_bool(CAT1_PROB[0]) as i16;
            5 + extra
        }
        Token::DctCat2 => {
            let mut extra = 0i16;
            for &prob in CAT2_PROB.iter() {
                extra = (extra << 1) | (bd.read_bool(prob) as i16);
            }
            7 + extra
        }
        Token::DctCat3 => {
            let mut extra = 0i16;
            for &prob in CAT3_PROB.iter() {
                extra = (extra << 1) | (bd.read_bool(prob) as i16);
            }
            11 + extra
        }
        Token::DctCat4 => {
            let mut extra = 0i16;
            for &prob in CAT4_PROB.iter() {
                extra = (extra << 1) | (bd.read_bool(prob) as i16);
            }
            19 + extra
        }
        Token::DctCat5 => {
            let mut extra = 0i16;
            for &prob in CAT5_PROB.iter() {
                extra = (extra << 1) | (bd.read_bool(prob) as i16);
            }
            35 + extra
        }
        Token::DctCat6 => {
            let mut extra = 0i16;
            for &prob in CAT6_PROB.iter() {
                extra = (extra << 1) | (bd.read_bool(prob) as i16);
            }
            67 + extra
        }
        _ => {
            let (_, base) = TOKEN_EXTRA_BITS[token as usize];
            base as i16
        }
    }
}

/// Decode DCT coefficients for a 4x4 block
///
/// # Arguments
/// * `bd` - Boolean decoder
/// * `probs` - Coefficient probabilities
/// * `block_type` - Block type (0-3)
/// * `start_at_dc` - If false, skip DC coefficient (for Y blocks when Y2 is present)
///
/// # Returns
/// * Tuple of (coefficients array, number of non-zero coefficients)
pub fn decode_block_coeffs(
    bd: &mut BoolDecoder,
    probs: &CoeffProbs,
    block_type: usize,
    start_at_dc: bool,
) -> ([i16; 16], usize) {
    let mut coeffs = [0i16; 16];
    let mut num_nonzero = 0usize;

    // Context: 0 = zero run, 1 = single non-zero, 2 = two+ non-zero
    let mut prev_coeff_ctx = 0usize;

    let start_idx = if start_at_dc { 0 } else { 1 };

    for i in start_idx..16 {
        let band = COEFF_BANDS[i] as usize;
        let probs_for_ctx = &probs.probs[block_type][band][prev_coeff_ctx];

        let token = decode_token(bd, probs_for_ctx);

        match token {
            Token::DctEob => break,
            Token::Dct0 => {
                // Zero coefficient, context stays 0
                prev_coeff_ctx = 0;
            }
            Token::Dct1 => {
                // +1 or -1
                let sign = bd.read_bool(128);
                coeffs[ZIGZAG_SCAN[i]] = if sign { -1 } else { 1 };
                num_nonzero += 1;
                prev_coeff_ctx = 1;
            }
            Token::Dct2 => {
                let sign = bd.read_bool(128);
                coeffs[ZIGZAG_SCAN[i]] = if sign { -2 } else { 2 };
                num_nonzero += 1;
                prev_coeff_ctx = 2;
            }
            Token::Dct3 => {
                let sign = bd.read_bool(128);
                coeffs[ZIGZAG_SCAN[i]] = if sign { -3 } else { 3 };
                num_nonzero += 1;
                prev_coeff_ctx = 2;
            }
            Token::Dct4 => {
                let sign = bd.read_bool(128);
                coeffs[ZIGZAG_SCAN[i]] = if sign { -4 } else { 4 };
                num_nonzero += 1;
                prev_coeff_ctx = 2;
            }
            _ => {
                // Category tokens
                let magnitude = read_extra_bits(bd, token);
                let sign = bd.read_bool(128);
                coeffs[ZIGZAG_SCAN[i]] = if sign { -magnitude } else { magnitude };
                num_nonzero += 1;
                prev_coeff_ctx = 2;
            }
        }
    }

    (coeffs, num_nonzero)
}

/// Decode coefficients for Y2 block (DC coefficients of 16 Y subblocks)
pub fn decode_y2_block(bd: &mut BoolDecoder, probs: &CoeffProbs) -> ([i16; 16], usize) {
    decode_block_coeffs(bd, probs, 3, true)
}

/// Decode coefficients for Y block (with or without Y2)
pub fn decode_y_block(
    bd: &mut BoolDecoder,
    probs: &CoeffProbs,
    has_y2: bool,
) -> ([i16; 16], usize) {
    let block_type = if has_y2 { 0 } else { 1 };
    let start_at_dc = !has_y2;
    decode_block_coeffs(bd, probs, block_type, start_at_dc)
}

/// Decode coefficients for UV block
pub fn decode_uv_block(bd: &mut BoolDecoder, probs: &CoeffProbs) -> ([i16; 16], usize) {
    decode_block_coeffs(bd, probs, 2, true)
}

/// Check if a block has all zero coefficients
pub fn has_nonzero_coeffs(coeffs: &[i16; 16]) -> bool {
    coeffs.iter().any(|&c| c != 0)
}

/// Get the "first band" coefficient position for a given block type
/// Used for determining initial probability context
pub fn get_first_coeff_band(block_type: usize, has_y2: bool) -> usize {
    if block_type == 0 && has_y2 {
        1 // Y block with Y2 starts at position 1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coeff_probs_default() {
        let probs = CoeffProbs::default();
        // Check that probabilities are initialized
        assert!(probs.probs[0][0][0][0] > 0);
    }

    #[test]
    fn test_token_from_u8() {
        assert_eq!(Token::from_u8(0), Token::DctEob);
        assert_eq!(Token::from_u8(1), Token::Dct0);
        assert_eq!(Token::from_u8(11), Token::DctCat6);
        assert_eq!(Token::from_u8(255), Token::DctEob); // Invalid maps to EOB
    }

    #[test]
    fn test_has_nonzero_coeffs() {
        let zero_block = [0i16; 16];
        assert!(!has_nonzero_coeffs(&zero_block));

        let mut nonzero_block = [0i16; 16];
        nonzero_block[5] = 10;
        assert!(has_nonzero_coeffs(&nonzero_block));
    }

    #[test]
    fn test_zigzag_scan() {
        // Verify zigzag order
        assert_eq!(ZIGZAG_SCAN[0], 0); // DC
        assert_eq!(ZIGZAG_SCAN[1], 1);
        assert_eq!(ZIGZAG_SCAN[2], 4);
        assert_eq!(ZIGZAG_SCAN[15], 15);
    }
}
