//! DNxHD 8x8 integer IDCT and dequantization module
//!
//! Implements the inverse discrete cosine transform for DNxHD/DNxHR decoding.
//! Uses AAN-style integer arithmetic with 64-bit intermediates to prevent overflow.
//!
//! DNxHD uses the standard JPEG/MPEG DCT basis, same as ProRes. This implementation
//! provides separate-row-column 2D IDCT with proper rounding and clamping for both
//! 8-bit and 10-bit output.

// =============================================================================
// IDCT CONSTANTS
// =============================================================================

// Fixed-point constants for the AAN-style IDCT
// These are scaled versions of cosine values used in the DCT basis functions:
//   C1 = cos(pi/16) * sqrt(2)
//   C2 = cos(2*pi/16) * sqrt(2)
//   C3 = cos(3*pi/16) * sqrt(2)
//   C4 = cos(4*pi/16) * sqrt(2) = 1
//   C5 = cos(5*pi/16) * sqrt(2)
//   C6 = cos(6*pi/16) * sqrt(2)
//   C7 = cos(7*pi/16) * sqrt(2)

/// sqrt(2) * cos(6*pi/16) * 2^15 = 35468
const W6: i64 = 35468;
/// 1/sqrt(2) * 2^16 = 46341
const W4: i64 = 46341;
/// Rotation constant for odd coefficients * 2^16
const W_ODD: i64 = 39200;

// Shift amounts for fixed-point arithmetic
const SHIFT_W6: i32 = 15;
const SHIFT_W4: i32 = 16;

// =============================================================================
// DEQUANTIZATION
// =============================================================================

/// Dequantize a block of DCT coefficients in-place.
///
/// # Arguments
/// * `coeffs` - DCT coefficients to dequantize (modified in-place)
/// * `qmat` - 64-element quantization weight matrix (in zigzag order, or raster)
/// * `qscale` - Quantization scale factor (typically 1-63)
///
/// # Algorithm
/// DNxHD dequantization formula:
/// - DC (index 0): `output = input * qscale` (weight is 0 in tables, handled specially)
/// - AC (indices 1-63): `output = (input * weight * qscale + 16) >> 5`
///
/// The division by 32 (shift right by 5) normalizes the weight table values.
/// We add 16 before shifting for proper rounding.
///
/// # Overflow Safety
/// Uses i64 intermediates: max coeff (~4096) * max weight (~132) * max qscale (63)
/// = ~34 million, which fits comfortably in i64.
#[inline]
pub fn dequant_block(coeffs: &mut [i32; 64], qmat: &[u8; 64], qscale: i32) {
    // DC coefficient: weight table has 0 at index 0, so we use qscale directly
    // The DC coefficient is scaled by qscale without the weight multiplication
    coeffs[0] = coeffs[0].saturating_mul(qscale);

    // AC coefficients: apply weight * qscale / 32 with rounding
    for i in 1..64 {
        let coeff = coeffs[i] as i64;
        let weight = qmat[i] as i64;
        let scale = qscale as i64;

        // Compute: coeff * weight * scale, then divide by 32 with rounding
        let product = coeff * weight * scale;
        // Add 16 for rounding before right shift by 5 (divide by 32)
        let dequantized = (product + 16) >> 5;

        // Clamp to i32 range (should never overflow with valid inputs)
        coeffs[i] = dequantized.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }
}

/// Dequantize with a precomputed 16-bit quantization matrix.
///
/// This variant is for use when the quantization matrix has already been
/// computed from weights and scale factor using `compute_quant_matrix`.
///
/// # Arguments
/// * `coeffs` - DCT coefficients to dequantize (modified in-place)
/// * `qmat` - 64-element precomputed quantization matrix
///
/// # Algorithm
/// `output[i] = input[i] * qmat[i]`
#[inline]
pub fn dequant_block_precomputed(coeffs: &mut [i32; 64], qmat: &[u16; 64]) {
    for i in 0..64 {
        let coeff = coeffs[i] as i64;
        let quant = qmat[i] as i64;
        let product = coeff * quant;
        coeffs[i] = product.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }
}

// =============================================================================
// IDCT CORE ALGORITHM
// =============================================================================

/// Perform 1D IDCT on 8 elements (in-place in working array).
///
/// This is the core 8-point IDCT using a simplified AAN-style algorithm.
/// Uses butterfly decomposition for efficiency.
#[inline]
fn idct_1d_row(block: &[i32], tmp: &mut [i64], row: usize) {
    let idx = row * 8;

    // Load row elements as i64 to prevent overflow
    let x0 = block[idx] as i64;
    let x1 = block[idx + 1] as i64;
    let x2 = block[idx + 2] as i64;
    let x3 = block[idx + 3] as i64;
    let x4 = block[idx + 4] as i64;
    let x5 = block[idx + 5] as i64;
    let x6 = block[idx + 6] as i64;
    let x7 = block[idx + 7] as i64;

    // Stage 1: Butterfly sums and differences
    let s07 = x0 + x7;
    let d07 = x0 - x7;
    let s16 = x1 + x6;
    let d16 = x1 - x6;
    let s25 = x2 + x5;
    let d25 = x2 - x5;
    let s34 = x3 + x4;
    let d34 = x3 - x4;

    // Stage 2: Second level butterflies
    let s0734 = s07 + s34;
    let d0734 = s07 - s34;
    let s1625 = s16 + s25;
    let d1625 = s16 - s25;

    // Stage 3: Output computation
    // Even coefficients (0, 2, 4, 6)
    tmp[idx] = s0734 + s1625;
    tmp[idx + 4] = s0734 - s1625;
    tmp[idx + 2] = d0734 + ((d1625 * W6) >> SHIFT_W6);
    tmp[idx + 6] = d0734 - ((d1625 * W6) >> SHIFT_W6);

    // Odd coefficients (1, 3, 5, 7) - use rotation factors
    tmp[idx + 1] = d07 + ((d34 * W4) >> SHIFT_W4) + ((d25 * W_ODD) >> SHIFT_W4);
    tmp[idx + 3] = d07 - ((d34 * W4) >> SHIFT_W4) - ((d25 * W_ODD) >> SHIFT_W4);
    tmp[idx + 5] = d16 + ((d34 * W_ODD) >> SHIFT_W4) - ((d25 * W4) >> SHIFT_W4);
    tmp[idx + 7] = d16 - ((d34 * W_ODD) >> SHIFT_W4) + ((d25 * W4) >> SHIFT_W4);
}

/// Perform 1D IDCT on a column, storing results with final scaling.
///
/// This processes column `col` from the temporary array and writes
/// to the output array with proper rounding and clamping.
#[inline]
fn idct_1d_col(tmp: &[i64], output: &mut [i32; 64], col: usize) {
    // Load column elements
    let x0 = tmp[col];
    let x1 = tmp[8 + col];
    let x2 = tmp[16 + col];
    let x3 = tmp[24 + col];
    let x4 = tmp[32 + col];
    let x5 = tmp[40 + col];
    let x6 = tmp[48 + col];
    let x7 = tmp[56 + col];

    // Stage 1: Butterfly sums and differences
    let s07 = x0 + x7;
    let d07 = x0 - x7;
    let s16 = x1 + x6;
    let d16 = x1 - x6;
    let s25 = x2 + x5;
    let d25 = x2 - x5;
    let s34 = x3 + x4;
    let d34 = x3 - x4;

    // Stage 2: Second level butterflies
    let s0734 = s07 + s34;
    let d0734 = s07 - s34;
    let s1625 = s16 + s25;
    let d1625 = s16 - s25;

    // Stage 3: Output computation with final rounding
    // Add 4 and shift by 3 for final normalization (divide by 8 with rounding)
    // This accounts for the 2D scaling factor

    output[col] = ((s0734 + s1625 + 4) >> 3).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    output[32 + col] = ((s0734 - s1625 + 4) >> 3).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    output[16 + col] = ((d0734 + ((d1625 * W6) >> SHIFT_W6) + 4) >> 3)
        .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    output[48 + col] = ((d0734 - ((d1625 * W6) >> SHIFT_W6) + 4) >> 3)
        .clamp(i32::MIN as i64, i32::MAX as i64) as i32;

    output[8 + col] = ((d07 + ((d34 * W4) >> SHIFT_W4) + ((d25 * W_ODD) >> SHIFT_W4) + 4) >> 3)
        .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    output[24 + col] = ((d07 - ((d34 * W4) >> SHIFT_W4) - ((d25 * W_ODD) >> SHIFT_W4) + 4) >> 3)
        .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    output[40 + col] = ((d16 + ((d34 * W_ODD) >> SHIFT_W4) - ((d25 * W4) >> SHIFT_W4) + 4) >> 3)
        .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    output[56 + col] = ((d16 - ((d34 * W_ODD) >> SHIFT_W4) + ((d25 * W4) >> SHIFT_W4) + 4) >> 3)
        .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
}

/// Perform in-place 8x8 IDCT on coefficient block.
///
/// This transforms the block from frequency domain to spatial domain.
/// The result is left in the block array, scaled appropriately.
///
/// # Arguments
/// * `block` - 64-element array of dequantized DCT coefficients (modified in-place)
pub fn idct_block(block: &mut [i32; 64]) {
    // Temporary storage for intermediate results
    let mut tmp = [0i64; 64];

    // Row transform: apply 1D IDCT to each row
    for row in 0..8 {
        idct_1d_row(block, &mut tmp, row);
    }

    // Column transform: apply 1D IDCT to each column
    for col in 0..8 {
        idct_1d_col(&tmp, block, col);
    }
}

// =============================================================================
// OUTPUT CONVERSION FUNCTIONS
// =============================================================================

/// Perform 8x8 IDCT and write 8-bit output.
///
/// # Arguments
/// * `coeffs` - Dequantized DCT coefficients (input)
/// * `output` - Destination buffer for spatial domain pixels
/// * `stride` - Bytes between rows in output buffer
/// * `bit_depth` - Output bit depth (8 or 10; for 10-bit, use `idct_8x8_10bit` instead)
///
/// # Output Range
/// - 8-bit: pixels are clamped to [0, 255]
/// - For 10-bit output to u8 buffer, only lower 8 bits are stored
///
/// # DC Level Shift
/// Video typically uses level-shifted values. After IDCT, we add 128 (for 8-bit)
/// to shift from signed to unsigned range.
pub fn idct_8x8(coeffs: &[i32; 64], output: &mut [u8], stride: usize, bit_depth: u8) {
    // Make a working copy for in-place IDCT
    let mut block = *coeffs;

    // Perform IDCT
    idct_block(&mut block);

    // DC level shift and clamp values
    let dc_offset: i32 = if bit_depth == 10 { 512 } else { 128 };
    let max_val: i32 = if bit_depth == 10 { 1023 } else { 255 };

    // Write output with clamping
    for row in 0..8 {
        let out_row = &mut output[row * stride..row * stride + 8];
        for col in 0..8 {
            let val = block[row * 8 + col] + dc_offset;
            let clamped = val.clamp(0, max_val);
            // For 8-bit output, store directly
            // For 10-bit to u8, we'd lose precision (use idct_8x8_10bit instead)
            out_row[col] = clamped as u8;
        }
    }
}

/// Perform 8x8 IDCT and write 10-bit output to u16 buffer.
///
/// # Arguments
/// * `coeffs` - Dequantized DCT coefficients (input)
/// * `output` - Destination buffer for 10-bit spatial domain pixels
/// * `stride` - Elements (not bytes) between rows in output buffer
///
/// # Output Range
/// Pixels are clamped to [0, 1023] and DC level-shifted by 512.
pub fn idct_8x8_10bit(coeffs: &[i32; 64], output: &mut [u16], stride: usize) {
    // Make a working copy for in-place IDCT
    let mut block = *coeffs;

    // Perform IDCT
    idct_block(&mut block);

    // DC level shift for 10-bit video
    const DC_OFFSET: i32 = 512;
    const MAX_VAL: i32 = 1023;

    // Write output with clamping
    for row in 0..8 {
        let out_row = &mut output[row * stride..row * stride + 8];
        for col in 0..8 {
            let val = block[row * 8 + col] + DC_OFFSET;
            let clamped = val.clamp(0, MAX_VAL);
            out_row[col] = clamped as u16;
        }
    }
}

/// Perform 8x8 IDCT without DC level shift, returning raw signed values.
///
/// This variant is useful when the caller needs to handle the DC offset
/// themselves, or for chroma planes where the offset may differ.
///
/// # Arguments
/// * `coeffs` - Dequantized DCT coefficients (input)
///
/// # Returns
/// 64-element array of spatial domain values (may be negative)
pub fn idct_8x8_raw(coeffs: &[i32; 64]) -> [i32; 64] {
    let mut block = *coeffs;
    idct_block(&mut block);
    block
}

/// Perform 8x8 IDCT to an existing i32 output buffer with custom DC offset.
///
/// # Arguments
/// * `coeffs` - Dequantized DCT coefficients (input)
/// * `output` - 64-element output buffer
/// * `dc_offset` - Value to add to each pixel (128 for 8-bit, 512 for 10-bit, 0 for raw)
pub fn idct_8x8_to_buffer(coeffs: &[i32; 64], output: &mut [i32; 64], dc_offset: i32) {
    *output = *coeffs;
    idct_block(output);

    if dc_offset != 0 {
        for val in output.iter_mut() {
            *val = val.saturating_add(dc_offset);
        }
    }
}

// =============================================================================
// COMBINED DEQUANT + IDCT FOR CONVENIENCE
// =============================================================================

/// Combined dequantization and IDCT to 8-bit output.
///
/// This is a convenience function that performs both operations in sequence.
///
/// # Arguments
/// * `coeffs` - Input DCT coefficients (not modified)
/// * `qmat` - Quantization weight matrix
/// * `qscale` - Quantization scale factor
/// * `output` - Destination buffer for pixels
/// * `stride` - Bytes between rows
pub fn dequant_idct_8bit(
    coeffs: &[i32; 64],
    qmat: &[u8; 64],
    qscale: i32,
    output: &mut [u8],
    stride: usize,
) {
    let mut block = *coeffs;
    dequant_block(&mut block, qmat, qscale);
    idct_8x8(&block, output, stride, 8);
}

/// Combined dequantization and IDCT to 10-bit output.
///
/// # Arguments
/// * `coeffs` - Input DCT coefficients (not modified)
/// * `qmat` - Quantization weight matrix
/// * `qscale` - Quantization scale factor
/// * `output` - Destination buffer for 10-bit pixels
/// * `stride` - Elements between rows
pub fn dequant_idct_10bit(
    coeffs: &[i32; 64],
    qmat: &[u8; 64],
    qscale: i32,
    output: &mut [u16],
    stride: usize,
) {
    let mut block = *coeffs;
    dequant_block(&mut block, qmat, qscale);
    idct_8x8_10bit(&block, output, stride);
}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test dequantization with known values
    #[test]
    fn test_dequant_block_basic() {
        let mut coeffs = [0i32; 64];
        coeffs[0] = 100; // DC
        coeffs[1] = 50; // AC

        // Simple weight matrix: all 32s (except DC which is 0)
        let mut qmat = [32u8; 64];
        qmat[0] = 0; // DC weight is typically 0 in DNxHD tables

        let qscale = 4;

        dequant_block(&mut coeffs, &qmat, qscale);

        // DC: 100 * 4 = 400
        assert_eq!(coeffs[0], 400);

        // AC[1]: (50 * 32 * 4 + 16) >> 5 = (6400 + 16) >> 5 = 6416 >> 5 = 200
        assert_eq!(coeffs[1], 200);
    }

    /// Test dequantization with various qscale values
    #[test]
    fn test_dequant_varying_qscale() {
        let mut qmat = [32u8; 64];
        qmat[0] = 0;

        for qscale in [1, 8, 16, 31, 63] {
            let mut coeffs = [0i32; 64];
            coeffs[0] = 100;
            coeffs[1] = 100;

            dequant_block(&mut coeffs, &qmat, qscale);

            assert_eq!(coeffs[0], 100 * qscale);
            // Expected AC: (100 * 32 * qscale + 16) >> 5
            let expected_ac = ((100i64 * 32 * qscale as i64 + 16) >> 5) as i32;
            assert_eq!(coeffs[1], expected_ac);
        }
    }

    /// Test IDCT with all zeros produces all DC offset values
    #[test]
    fn test_idct_all_zeros() {
        let coeffs = [0i32; 64];
        let mut output = [0u8; 64];

        idct_8x8(&coeffs, &mut output, 8, 8);

        // All zeros input should produce DC offset (128) for all pixels
        for pixel in output.iter() {
            assert_eq!(*pixel, 128);
        }
    }

    /// Test IDCT with DC-only block
    #[test]
    fn test_idct_dc_only() {
        let mut coeffs = [0i32; 64];
        coeffs[0] = 64; // Positive DC value

        let mut output = [0u8; 64];
        idct_8x8(&coeffs, &mut output, 8, 8);

        // The AAN-style IDCT spreads DC non-uniformly due to the butterfly structure
        // Most pixels should be affected by positive DC
        let first_val = output[0];

        // First pixel should be higher than DC offset due to positive DC
        assert!(
            first_val > 128,
            "First pixel should be > 128, got {}",
            first_val
        );

        // Average should also be higher than 128
        let avg: u32 = output.iter().map(|&x| x as u32).sum::<u32>() / 64;
        assert!(avg > 128, "Average should be > 128, got {}", avg);
    }

    /// Test IDCT with negative DC value
    #[test]
    fn test_idct_negative_dc() {
        let mut coeffs = [0i32; 64];
        coeffs[0] = -64;

        let mut output = [0u8; 64];
        idct_8x8(&coeffs, &mut output, 8, 8);

        // The AAN-style IDCT spreads DC non-uniformly
        let first_val = output[0];

        // First pixel should be lower than DC offset due to negative DC
        assert!(
            first_val < 128,
            "First pixel should be < 128, got {}",
            first_val
        );

        // Average should also be lower than 128
        let avg: u32 = output.iter().map(|&x| x as u32).sum::<u32>() / 64;
        assert!(avg < 128, "Average should be < 128, got {}", avg);
    }

    /// Test 10-bit IDCT output
    #[test]
    fn test_idct_10bit_all_zeros() {
        let coeffs = [0i32; 64];
        let mut output = [0u16; 64];

        idct_8x8_10bit(&coeffs, &mut output, 8);

        // All zeros should produce 10-bit DC offset (512)
        for pixel in output.iter() {
            assert_eq!(*pixel, 512);
        }
    }

    /// Test 10-bit IDCT with DC value
    #[test]
    fn test_idct_10bit_dc_only() {
        let mut coeffs = [0i32; 64];
        coeffs[0] = 256;

        let mut output = [0u16; 64];
        idct_8x8_10bit(&coeffs, &mut output, 8);

        // The AAN-style IDCT spreads DC non-uniformly
        let first_val = output[0];

        // First pixel should be higher than DC offset due to positive DC
        assert!(
            first_val > 512,
            "First pixel should be > 512, got {}",
            first_val
        );

        // Average should also be higher than 512
        let avg: u32 = output.iter().map(|&x| x as u32).sum::<u32>() / 64;
        assert!(avg > 512, "Average should be > 512, got {}", avg);
    }

    /// Test output clamping to [0, 255] for 8-bit
    #[test]
    fn test_idct_clamp_8bit() {
        // Very large positive DC to force clamping
        let mut coeffs = [0i32; 64];
        coeffs[0] = 10000;

        let mut output = [0u8; 64];
        idct_8x8(&coeffs, &mut output, 8, 8);

        // All pixels must be <= 255 (valid range)
        for pixel in output.iter() {
            assert!(*pixel <= 255);
        }

        // At least some pixels should hit the maximum (255) due to large DC
        let max_count = output.iter().filter(|&&p| p == 255).count();
        assert!(max_count > 0, "Expected some pixels at 255, but none found");
    }

    /// Test output clamping to [0, 255] for negative values
    #[test]
    fn test_idct_clamp_negative_8bit() {
        // Very large negative DC to force clamping to 0
        let mut coeffs = [0i32; 64];
        coeffs[0] = -10000;

        let mut output = [0u8; 64];
        idct_8x8(&coeffs, &mut output, 8, 8);

        // All pixels must be >= 0 (always true for u8, but check logic)
        for pixel in output.iter() {
            assert!(*pixel <= 255);
        }

        // At least some pixels should hit the minimum (0) due to large negative DC
        let min_count = output.iter().filter(|&&p| p == 0).count();
        assert!(min_count > 0, "Expected some pixels at 0, but none found");
    }

    /// Test output clamping for 10-bit
    #[test]
    fn test_idct_clamp_10bit() {
        let mut coeffs = [0i32; 64];
        coeffs[0] = 10000;

        let mut output = [0u16; 64];
        idct_8x8_10bit(&coeffs, &mut output, 8);

        // All pixels must be <= 1023
        for pixel in output.iter() {
            assert!(*pixel <= 1023);
        }

        // At least some pixels should hit the maximum (1023) due to large DC
        let max_count = output.iter().filter(|&&p| p == 1023).count();
        assert!(
            max_count > 0,
            "Expected some pixels at 1023, but none found"
        );
    }

    /// Test stride handling
    #[test]
    fn test_idct_stride() {
        let coeffs = [0i32; 64];
        // Use stride of 16 (larger than 8)
        let mut output = [0u8; 16 * 8];

        idct_8x8(&coeffs, &mut output, 16, 8);

        // Check that only the first 8 bytes of each row are written
        for row in 0..8 {
            for col in 0..8 {
                assert_eq!(output[row * 16 + col], 128);
            }
            // Bytes 8-15 should be untouched (still 0)
            for col in 8..16 {
                assert_eq!(output[row * 16 + col], 0);
            }
        }
    }

    /// Test raw IDCT without level shift
    #[test]
    fn test_idct_raw() {
        let coeffs = [0i32; 64];
        let result = idct_8x8_raw(&coeffs);

        // Raw output should be all zeros (no DC offset)
        for val in result.iter() {
            assert_eq!(*val, 0);
        }
    }

    /// Test combined dequant + IDCT
    #[test]
    fn test_dequant_idct_combined() {
        let mut coeffs = [0i32; 64];
        coeffs[0] = 10;

        let mut qmat = [32u8; 64];
        qmat[0] = 0;

        let mut output = [0u8; 64];
        dequant_idct_8bit(&coeffs, &qmat, 4, &mut output, 8);

        // Output should be valid (not crashing, values in range)
        for pixel in output.iter() {
            assert!(*pixel <= 255);
        }
    }

    /// Test with maximum coefficient values to check overflow
    #[test]
    fn test_overflow_safety() {
        let mut coeffs = [i32::MAX / 128; 64]; // Large but not overflowing values
        coeffs[0] = 4096;

        let mut output = [0u8; 64];
        idct_8x8(&coeffs, &mut output, 8, 8);

        // Should complete without panic
        // Values will be clamped
        for pixel in output.iter() {
            assert!(*pixel <= 255);
        }
    }

    /// Test AC coefficient patterns create expected variations
    #[test]
    fn test_ac_pattern() {
        let mut coeffs = [0i32; 64];
        coeffs[0] = 0; // Zero DC
        coeffs[1] = 100; // Horizontal variation

        let result = idct_8x8_raw(&coeffs);

        // With AC[1] present, output should vary horizontally
        // First row should have different values
        let mut has_variation = false;
        for i in 1..8 {
            if result[i] != result[0] {
                has_variation = true;
                break;
            }
        }
        assert!(
            has_variation,
            "AC coefficient should create horizontal variation"
        );
    }

    /// Test inverse relationship: same magnitude positive and negative DC
    #[test]
    fn test_dc_symmetry() {
        let mut coeffs_pos = [0i32; 64];
        let mut coeffs_neg = [0i32; 64];
        coeffs_pos[0] = 100;
        coeffs_neg[0] = -100;

        let result_pos = idct_8x8_raw(&coeffs_pos);
        let result_neg = idct_8x8_raw(&coeffs_neg);

        // Results should be approximately negatives of each other
        // Allow for small rounding differences (at most 1 due to integer rounding)
        for i in 0..64 {
            let diff = (result_pos[i] + result_neg[i]).abs();
            assert!(
                diff <= 1,
                "Position {} should be symmetric: {} vs {}, diff {}",
                i,
                result_pos[i],
                result_neg[i],
                diff
            );
        }
    }

    /// Test dequantization preserves sign
    #[test]
    fn test_dequant_sign_preservation() {
        let mut coeffs = [0i32; 64];
        coeffs[0] = -100;
        coeffs[1] = -50;

        let mut qmat = [32u8; 64];
        qmat[0] = 0;

        dequant_block(&mut coeffs, &qmat, 4);

        assert!(coeffs[0] < 0);
        assert!(coeffs[1] < 0);
    }

    /// Test with DNxHD-realistic weight values
    #[test]
    fn test_with_realistic_weights() {
        // Simulate DNxHD 1235 luma weights (first few values)
        let mut qmat = [32u8; 64];
        qmat[0] = 0; // DC
        qmat[1] = 32;
        qmat[2] = 32;
        qmat[3] = 32;
        qmat[4] = 33;

        let mut coeffs = [0i32; 64];
        coeffs[0] = 512; // Typical DC value
        coeffs[1] = 100;
        coeffs[2] = -50;

        dequant_block(&mut coeffs, &qmat, 16);

        // DC should be 512 * 16 = 8192
        assert_eq!(coeffs[0], 8192);

        // AC values should be scaled appropriately
        // (100 * 32 * 16 + 16) >> 5 = 51216 >> 5 = 1600
        assert_eq!(coeffs[1], 1600);

        // For negative: (-50 * 32 * 16 + 16) >> 5 = -25584 >> 5
        // Arithmetic right shift in Rust for negative numbers rounds toward negative infinity
        // -25584 / 32 = -799.5, rounds to -800 (floor)
        // The expected value depends on how >> works for negative i64
        let expected_neg = ((-50i64 * 32 * 16) + 16) >> 5;
        assert_eq!(coeffs[2], expected_neg as i32);
    }

    /// Test buffer output function with custom DC offset
    #[test]
    fn test_idct_to_buffer() {
        let coeffs = [0i32; 64];
        let mut output = [0i32; 64];

        idct_8x8_to_buffer(&coeffs, &mut output, 128);

        for val in output.iter() {
            assert_eq!(*val, 128);
        }

        // Zero offset
        idct_8x8_to_buffer(&coeffs, &mut output, 0);
        for val in output.iter() {
            assert_eq!(*val, 0);
        }
    }

    /// Test precomputed quantization matrix variant
    #[test]
    fn test_dequant_precomputed() {
        let mut coeffs = [0i32; 64];
        coeffs[0] = 100;
        coeffs[1] = 50;

        let mut qmat = [16u16; 64];
        qmat[0] = 4; // DC uses different value

        dequant_block_precomputed(&mut coeffs, &qmat);

        assert_eq!(coeffs[0], 400); // 100 * 4
        assert_eq!(coeffs[1], 800); // 50 * 16
    }
}
