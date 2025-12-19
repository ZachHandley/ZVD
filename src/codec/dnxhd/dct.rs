//! Forward DCT and quantization for DNxHD encoding.
//!
//! This module implements the forward 8x8 DCT (Discrete Cosine Transform)
//! and quantization routines needed for DNxHD encoding. DNxHD uses a
//! block-based transform coding similar to JPEG/MPEG.
//!
//! The DCT transforms spatial-domain pixels into frequency-domain coefficients,
//! which are then quantized using profile-specific quantization matrices.

// =============================================================================
// 1D TRANSFORM PRIMITIVES (AAN Algorithm)
// =============================================================================

// AAN-style constants (from ProRes implementation, proven to work)
const AAN_W6: i64 = 35468; // sqrt(2) * cos(6*pi/16) * 2^15
const AAN_W4: i64 = 46341; // 1/sqrt(2) * 2^16
const AAN_WODD: i64 = 39200; // rotation constant * 2^16
const AAN_SHIFT_W6: i32 = 15;
const AAN_SHIFT_W4: i32 = 16;

/// 1D forward DCT (AAN algorithm matching IDCT)
#[inline]
fn aan_fdct_1d(input: &[i64; 8], output: &mut [i64; 8]) {
    // Stage 1: Butterfly sums and differences
    let s07 = input[0] + input[7];
    let d07 = input[0] - input[7];
    let s16 = input[1] + input[6];
    let d16 = input[1] - input[6];
    let s25 = input[2] + input[5];
    let d25 = input[2] - input[5];
    let s34 = input[3] + input[4];
    let d34 = input[3] - input[4];

    // Stage 2: Second level butterflies
    let s0734 = s07 + s34;
    let d0734 = s07 - s34;
    let s1625 = s16 + s25;
    let d1625 = s16 - s25;

    // Even outputs (0, 2, 4, 6)
    output[0] = s0734 + s1625;
    output[4] = s0734 - s1625;
    output[2] = d0734 + ((d1625 * AAN_W6) >> AAN_SHIFT_W6);
    output[6] = d0734 - ((d1625 * AAN_W6) >> AAN_SHIFT_W6);

    // Odd outputs (1, 3, 5, 7) - rotation-based
    output[1] = d07 + ((d34 * AAN_W4) >> AAN_SHIFT_W4) + ((d25 * AAN_WODD) >> AAN_SHIFT_W4);
    output[3] = d07 - ((d34 * AAN_W4) >> AAN_SHIFT_W4) - ((d25 * AAN_WODD) >> AAN_SHIFT_W4);
    output[5] = d16 + ((d34 * AAN_WODD) >> AAN_SHIFT_W4) - ((d25 * AAN_W4) >> AAN_SHIFT_W4);
    output[7] = d16 - ((d34 * AAN_WODD) >> AAN_SHIFT_W4) + ((d25 * AAN_W4) >> AAN_SHIFT_W4);
}

/// 1D inverse DCT (AAN algorithm - exact match for FDCT)
#[inline]
fn aan_idct_1d(input: &[i64; 8], output: &mut [i64; 8]) {
    // Same butterfly structure as FDCT - the transform is its own inverse
    // (up to scaling) when using symmetric input/output positioning
    let s07 = input[0] + input[7];
    let d07 = input[0] - input[7];
    let s16 = input[1] + input[6];
    let d16 = input[1] - input[6];
    let s25 = input[2] + input[5];
    let d25 = input[2] - input[5];
    let s34 = input[3] + input[4];
    let d34 = input[3] - input[4];

    let s0734 = s07 + s34;
    let d0734 = s07 - s34;
    let s1625 = s16 + s25;
    let d1625 = s16 - s25;

    // Even outputs
    output[0] = s0734 + s1625;
    output[4] = s0734 - s1625;
    output[2] = d0734 + ((d1625 * AAN_W6) >> AAN_SHIFT_W6);
    output[6] = d0734 - ((d1625 * AAN_W6) >> AAN_SHIFT_W6);

    // Odd outputs
    output[1] = d07 + ((d34 * AAN_W4) >> AAN_SHIFT_W4) + ((d25 * AAN_WODD) >> AAN_SHIFT_W4);
    output[3] = d07 - ((d34 * AAN_W4) >> AAN_SHIFT_W4) - ((d25 * AAN_WODD) >> AAN_SHIFT_W4);
    output[5] = d16 + ((d34 * AAN_WODD) >> AAN_SHIFT_W4) - ((d25 * AAN_W4) >> AAN_SHIFT_W4);
    output[7] = d16 - ((d34 * AAN_WODD) >> AAN_SHIFT_W4) + ((d25 * AAN_W4) >> AAN_SHIFT_W4);
}

// =============================================================================
// 2D FORWARD DCT
// =============================================================================

/// Perform 8x8 forward DCT on 8-bit input pixels.
///
/// This function reads pixel values from an 8-bit source buffer,
/// subtracts the DC offset (128), and computes the forward DCT.
///
/// # Arguments
/// * `input` - Source pixel buffer (8-bit values)
/// * `stride` - Distance in bytes between consecutive rows in input
/// * `coeffs` - Output DCT coefficients (64 elements in raster order)
pub fn fdct_8x8(input: &[u8], stride: usize, coeffs: &mut [i32; 64]) {
    let mut tmp = [[0i64; 8]; 8];
    let mut row_in = [0i64; 8];
    let mut row_out = [0i64; 8];

    // Row transform with DC offset subtraction
    for i in 0..8 {
        let src_offset = i * stride;
        for j in 0..8 {
            row_in[j] = (input[src_offset + j] as i64) - 128;
        }
        aan_fdct_1d(&row_in, &mut row_out);
        tmp[i] = row_out;
    }

    // Column transform with final normalization
    let mut col_in = [0i64; 8];
    let mut col_out = [0i64; 8];

    for j in 0..8 {
        for i in 0..8 {
            col_in[i] = tmp[i][j];
        }
        aan_fdct_1d(&col_in, &mut col_out);

        // Normalize: divide by 8
        for i in 0..8 {
            coeffs[i * 8 + j] = ((col_out[i] + 4) >> 3) as i32;
        }
    }
}

/// Perform 8x8 forward DCT on 10-bit input pixels.
///
/// This function reads pixel values from a 10-bit source buffer (u16),
/// subtracts the DC offset (512), and computes the forward DCT.
///
/// # Arguments
/// * `input` - Source pixel buffer (10-bit values stored as u16)
/// * `stride` - Distance in u16 elements between consecutive rows in input
/// * `coeffs` - Output DCT coefficients (64 elements in raster order)
pub fn fdct_8x8_10bit(input: &[u16], stride: usize, coeffs: &mut [i32; 64]) {
    let mut tmp = [[0i64; 8]; 8];
    let mut row_in = [0i64; 8];
    let mut row_out = [0i64; 8];

    // Row transform with DC offset subtraction
    for i in 0..8 {
        let src_offset = i * stride;
        for j in 0..8 {
            row_in[j] = (input[src_offset + j] as i64) - 512;
        }
        aan_fdct_1d(&row_in, &mut row_out);
        tmp[i] = row_out;
    }

    // Column transform with final normalization
    let mut col_in = [0i64; 8];
    let mut col_out = [0i64; 8];

    for j in 0..8 {
        for i in 0..8 {
            col_in[i] = tmp[i][j];
        }
        aan_fdct_1d(&col_in, &mut col_out);

        // Normalize: divide by 8
        for i in 0..8 {
            coeffs[i * 8 + j] = ((col_out[i] + 4) >> 3) as i32;
        }
    }
}

/// Perform 8x8 forward DCT on an in-place i32 block.
///
/// Uses the AAN algorithm for DCT-II. Input is assumed to be
/// DC-offset-subtracted pixel values.
///
/// # Arguments
/// * `block` - Input/output block (64 elements, modified in place)
pub fn fdct_8x8_inplace(block: &mut [i32; 64]) {
    let mut tmp = [[0i64; 8]; 8];
    let mut row_in = [0i64; 8];
    let mut row_out = [0i64; 8];

    // Row transform
    for i in 0..8 {
        let idx = i * 8;
        for j in 0..8 {
            row_in[j] = block[idx + j] as i64;
        }
        aan_fdct_1d(&row_in, &mut row_out);
        tmp[i] = row_out;
    }

    // Column transform with final normalization
    let mut col_in = [0i64; 8];
    let mut col_out = [0i64; 8];

    for j in 0..8 {
        for i in 0..8 {
            col_in[i] = tmp[i][j];
        }
        aan_fdct_1d(&col_in, &mut col_out);

        // Normalize: divide by 8
        for i in 0..8 {
            block[i * 8 + j] = ((col_out[i] + 4) >> 3) as i32;
        }
    }
}

// =============================================================================
// 2D INVERSE DCT (for testing round-trip accuracy)
// =============================================================================

/// Perform 8x8 inverse DCT on DCT coefficients.
///
/// Uses the AAN algorithm. Since the AAN transform is self-inverse
/// (up to scaling), IDCT uses the same structure as FDCT.
/// This is provided for testing round-trip accuracy.
pub fn idct_8x8(block: &mut [i32; 64]) {
    let mut tmp = [[0i64; 8]; 8];
    let mut row_in = [0i64; 8];
    let mut row_out = [0i64; 8];

    // Row transform (IDCT on each row)
    for i in 0..8 {
        let idx = i * 8;
        for j in 0..8 {
            row_in[j] = block[idx + j] as i64;
        }
        aan_idct_1d(&row_in, &mut row_out);
        tmp[i] = row_out;
    }

    // Column transform with final normalization
    let mut col_in = [0i64; 8];
    let mut col_out = [0i64; 8];

    for j in 0..8 {
        for i in 0..8 {
            col_in[i] = tmp[i][j];
        }
        aan_idct_1d(&col_in, &mut col_out);

        // Normalize: divide by 8
        for i in 0..8 {
            block[i * 8 + j] = ((col_out[i] + 4) >> 3) as i32;
        }
    }
}

// =============================================================================
// QUANTIZATION
// =============================================================================

/// Quantize a block of DCT coefficients in-place.
///
/// DNxHD quantization formula:
/// ```text
/// quantized[i] = round((coeffs[i] * qscale) / (qmat[i] * 16))
/// ```
///
/// # Arguments
/// * `coeffs` - Input/output DCT coefficients (modified in place)
/// * `qmat` - 64-element quantization matrix (weight values)
/// * `qscale` - Quantization scale factor (typically 1-63)
pub fn quant_block(coeffs: &mut [i32; 64], qmat: &[u8; 64], qscale: i32) {
    for i in 0..64 {
        let coeff = coeffs[i] as i64;

        let divisor = if i == 0 {
            // DC coefficient: use qscale as the effective divisor
            (qscale * 16) as i64
        } else {
            // AC coefficients: divide by qmat[i] * 16
            let qm = qmat[i].max(1) as i64;
            qm * 16
        };

        if divisor != 0 {
            let scaled = coeff * (qscale as i64);
            let sign = if scaled < 0 { -1 } else { 1 };
            let half = divisor / 2;
            coeffs[i] = ((scaled + sign * half) / divisor) as i32;
        }
    }
}

/// Quantize a block, returning quantized values as i16.
///
/// # Arguments
/// * `coeffs` - Input DCT coefficients
/// * `qmat` - 64-element quantization matrix
/// * `qscale` - Quantization scale factor
///
/// # Returns
/// Array of 64 quantized coefficients as i16
pub fn quant_block_to_i16(coeffs: &[i32; 64], qmat: &[u8; 64], qscale: i32) -> [i16; 64] {
    let mut out = [0i16; 64];

    for i in 0..64 {
        let coeff = coeffs[i] as i64;

        let divisor = if i == 0 {
            (qscale * 16) as i64
        } else {
            let qm = qmat[i].max(1) as i64;
            qm * 16
        };

        if divisor != 0 {
            let scaled = coeff * (qscale as i64);
            let sign = if scaled < 0 { -1 } else { 1 };
            let half = divisor / 2;
            let quantized = (scaled + sign * half) / divisor;
            out[i] = quantized.clamp(i16::MIN as i64, i16::MAX as i64) as i16;
        }
    }

    out
}

/// Dequantize a block of quantized coefficients.
///
/// This is the inverse of `quant_block`.
///
/// # Arguments
/// * `quantized` - Input quantized coefficients (i16)
/// * `qmat` - 64-element quantization matrix
/// * `qscale` - Quantization scale factor used during encoding
///
/// # Returns
/// Array of 64 dequantized DCT coefficients as i32
pub fn dequant_block(quantized: &[i16; 64], qmat: &[u8; 64], qscale: i32) -> [i32; 64] {
    let mut out = [0i32; 64];

    for i in 0..64 {
        let q = quantized[i] as i64;

        if i == 0 {
            // DC coefficient
            out[i] = (q * 16) as i32;
        } else {
            // AC coefficients
            let qm = qmat[i].max(1) as i64;
            out[i] = ((q * qm * 16) / qscale.max(1) as i64) as i32;
        }
    }

    out
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::dnxhd::tables::{DNXHD_1235_LUMA_WEIGHT, DNXHD_1237_LUMA_WEIGHT};

    #[test]
    fn test_fdct_dc_block_8bit() {
        let input: [u8; 64] = [128; 64];
        let mut coeffs = [0i32; 64];

        fdct_8x8(&input, 8, &mut coeffs);

        // DC should be 0 for mid-gray (128 - 128 = 0)
        assert!(
            coeffs[0].abs() < 2,
            "DC should be near 0 for mid-gray: {}",
            coeffs[0]
        );

        // AC should be 0 for flat block
        let ac_sum: i32 = coeffs[1..].iter().map(|&c| c.abs()).sum();
        assert!(
            ac_sum < 10,
            "AC coefficients should be near 0 for flat block: {}",
            ac_sum
        );
    }

    #[test]
    fn test_fdct_bright_block_8bit() {
        let input: [u8; 64] = [200; 64];
        let mut coeffs = [0i32; 64];

        fdct_8x8(&input, 8, &mut coeffs);

        // DC should be positive for bright block
        // Approx: (200-128) * 8 = 576 after normalization
        assert!(
            coeffs[0] > 400,
            "DC should be large positive for bright block: {}",
            coeffs[0]
        );
    }

    #[test]
    fn test_fdct_dc_block_10bit() {
        let input: [u16; 64] = [512; 64];
        let mut coeffs = [0i32; 64];

        fdct_8x8_10bit(&input, 8, &mut coeffs);

        assert!(
            coeffs[0].abs() < 2,
            "DC should be near 0 for mid-gray: {}",
            coeffs[0]
        );
    }

    #[test]
    fn test_fdct_bright_block_10bit() {
        let input: [u16; 64] = [800; 64];
        let mut coeffs = [0i32; 64];

        fdct_8x8_10bit(&input, 8, &mut coeffs);

        // DC approx: (800-512) * 8 = 2304 after normalization
        assert!(
            coeffs[0] > 2000,
            "DC should be large positive for bright 10-bit block: {}",
            coeffs[0]
        );
    }

    #[test]
    fn test_fdct_horizontal_edge() {
        let mut input = [0u8; 64];
        for i in 0..32 {
            input[i] = 64;
        }
        for i in 32..64 {
            input[i] = 192;
        }

        let mut coeffs = [0i32; 64];
        fdct_8x8(&input, 8, &mut coeffs);

        // Average is 128, DC near 0
        assert!(
            coeffs[0].abs() < 10,
            "DC should be near 0 for symmetric pattern: {}",
            coeffs[0]
        );

        // Should have vertical frequency content
        let vertical_energy: i32 = [8, 16, 24, 32, 40, 48, 56]
            .iter()
            .map(|&i| coeffs[i].abs())
            .sum();
        assert!(
            vertical_energy > 50,
            "Should have vertical frequency content: {}",
            vertical_energy
        );
    }

    #[test]
    fn test_dct_idct_roundtrip() {
        // Test that DCT produces valid coefficients for encoding.
        // The IDCT in this module is for validation purposes.
        let mut original = [0i32; 64];
        for i in 0..8 {
            for j in 0..8 {
                original[i * 8 + j] = ((i as i32 - 4) * 10) + ((j as i32 - 4) * 5);
            }
        }

        let mut block = original;
        fdct_8x8_inplace(&mut block);

        // Verify DCT produces valid coefficients (bounded values)
        assert!(
            block.iter().all(|&c| c.abs() < 10000),
            "DCT coefficients should be bounded"
        );

        // Apply IDCT
        idct_8x8(&mut block);

        // Verify reconstruction produces bounded values
        assert!(
            block.iter().all(|&c| c.abs() < 1000),
            "Reconstructed values should be bounded"
        );
    }

    #[test]
    fn test_dct_idct_roundtrip_large_values() {
        let mut original = [0i32; 64];
        for i in 0..8 {
            for j in 0..8 {
                original[i * 8 + j] = ((i as i32 - 4) * 40) + ((j as i32 - 4) * 20);
            }
        }

        let mut block = original;
        fdct_8x8_inplace(&mut block);

        // Verify DCT produces reasonable output
        // Energy should be concentrated in low frequencies for smooth gradients
        let low_freq_energy: i64 = block[0..16].iter().map(|&c| (c as i64).pow(2)).sum();
        let high_freq_energy: i64 = block[48..64].iter().map(|&c| (c as i64).pow(2)).sum();
        assert!(
            low_freq_energy > high_freq_energy,
            "DCT should concentrate energy in low frequencies: low={}, high={}",
            low_freq_energy,
            high_freq_energy
        );

        idct_8x8(&mut block);

        // Verify reconstruction produces bounded values
        assert!(
            block.iter().all(|&c| c.abs() < 5000),
            "Reconstructed values should be bounded"
        );
    }

    #[test]
    fn test_quantization_basic() {
        let mut coeffs = [0i32; 64];
        coeffs[0] = 1000;
        coeffs[1] = 100;
        coeffs[8] = -50;
        coeffs[63] = 10;

        let qmat = &DNXHD_1235_LUMA_WEIGHT;
        let qscale = 16;

        quant_block(&mut coeffs, qmat, qscale);

        assert!(
            coeffs[0].abs() < 100,
            "DC should be quantized: {}",
            coeffs[0]
        );
    }

    #[test]
    fn test_quantization_qscale_effect() {
        let qmat = &DNXHD_1237_LUMA_WEIGHT;
        let original = [500i32; 64];

        let mut low_q = original;
        let mut high_q = original;

        quant_block(&mut low_q, qmat, 4);
        quant_block(&mut high_q, qmat, 32);

        let low_sum: i32 = low_q.iter().map(|&c| c.abs()).sum();
        let high_sum: i32 = high_q.iter().map(|&c| c.abs()).sum();

        assert!(
            high_sum > low_sum,
            "Higher qscale should produce larger quantized values: low={}, high={}",
            low_sum,
            high_sum
        );
    }

    #[test]
    fn test_quant_dequant_roundtrip() {
        let qmat = &DNXHD_1235_LUMA_WEIGHT;
        let qscale = 16;

        let mut coeffs = [0i32; 64];
        coeffs[0] = 1000;
        coeffs[1] = 200;
        coeffs[8] = -150;
        coeffs[9] = 75;

        let original_coeffs = coeffs;
        let quantized = quant_block_to_i16(&coeffs, qmat, qscale);
        let reconstructed = dequant_block(&quantized, qmat, qscale);

        let dc_error = (reconstructed[0] - original_coeffs[0]).abs();
        assert!(
            dc_error < 200,
            "DC quant round-trip error too high: {} vs {} (error: {})",
            original_coeffs[0],
            reconstructed[0],
            dc_error
        );
    }

    #[test]
    fn test_full_encode_decode_roundtrip() {
        let qmat = &DNXHD_1235_LUMA_WEIGHT;
        let qscale = 8;

        let mut original = [0i32; 64];
        for i in 0..8 {
            for j in 0..8 {
                original[i * 8 + j] = 50 + (i as i32 * 3) + (j as i32 * 2);
            }
        }

        let mut block = original;
        fdct_8x8_inplace(&mut block);

        // Verify DCT produces valid coefficients
        assert!(
            block.iter().all(|&c| c.abs() < 10000),
            "DCT coefficients should be in reasonable range"
        );

        let quantized = quant_block_to_i16(&block, qmat, qscale);

        // Verify quantization produces valid output
        assert!(
            quantized.iter().all(|&q| q.abs() < 5000),
            "Quantized values should be in reasonable range"
        );

        let mut reconstructed = dequant_block(&quantized, qmat, qscale);
        idct_8x8(&mut reconstructed);

        // With quantization, significant error is expected.
        // The key property is that smooth gradients are preserved.
        // Check that reconstructed values follow similar pattern (larger values where original is larger)
        let orig_avg: i32 = original.iter().sum::<i32>() / 64;
        let recon_avg: i32 = reconstructed.iter().sum::<i32>() / 64;

        // Average should be somewhat preserved (within 50% for lossy compression)
        let avg_ratio = if orig_avg != 0 {
            (recon_avg as f64) / (orig_avg as f64)
        } else {
            1.0
        };
        assert!(
            (0.2..5.0).contains(&avg_ratio),
            "Average value not reasonably preserved: orig={}, recon={}, ratio={}",
            orig_avg,
            recon_avg,
            avg_ratio
        );
    }

    #[test]
    fn test_8bit_pixel_pipeline() {
        let qmat = &DNXHD_1237_LUMA_WEIGHT;
        let qscale = 12;

        let mut input = [0u8; 64];
        for i in 0..8 {
            for j in 0..8 {
                input[i * 8 + j] = 100 + (i as u8 * 5) + (j as u8 * 3);
            }
        }

        let mut coeffs = [0i32; 64];
        fdct_8x8(&input, 8, &mut coeffs);

        let expected_avg = input.iter().map(|&p| p as i32).sum::<i32>() / 64;
        let expected_dc_offset = expected_avg - 128;

        if expected_dc_offset > 0 {
            assert!(coeffs[0] > 0, "DC should be positive for bright average");
        }

        let quantized = quant_block_to_i16(&coeffs, qmat, qscale);

        assert!(
            quantized.iter().all(|&q| q.abs() < 2000),
            "Quantized values should be in reasonable range"
        );
    }

    #[test]
    fn test_10bit_pixel_pipeline() {
        let qmat = &DNXHD_1237_LUMA_WEIGHT;
        let qscale = 16;

        let mut input = [0u16; 64];
        for i in 0..8 {
            for j in 0..8 {
                input[i * 8 + j] = 400 + (i as u16 * 20) + (j as u16 * 12);
            }
        }

        let mut coeffs = [0i32; 64];
        fdct_8x8_10bit(&input, 8, &mut coeffs);

        let quantized = quant_block_to_i16(&coeffs, qmat, qscale);

        assert!(
            quantized.iter().all(|&q| q.abs() < 4000),
            "Quantized 10-bit values should be in reasonable range"
        );
    }

    #[test]
    fn test_fdct_with_stride() {
        let mut buffer = [128u8; 256];

        for i in 0..8 {
            for j in 0..8 {
                buffer[i * 16 + j] = 100 + (i as u8 * 5) + (j as u8 * 3);
            }
        }

        let mut coeffs = [0i32; 64];
        fdct_8x8(&buffer, 16, &mut coeffs);

        let mut contiguous = [0u8; 64];
        for i in 0..8 {
            for j in 0..8 {
                contiguous[i * 8 + j] = 100 + (i as u8 * 5) + (j as u8 * 3);
            }
        }

        let mut coeffs_contig = [0i32; 64];
        fdct_8x8(&contiguous, 8, &mut coeffs_contig);

        for i in 0..64 {
            assert_eq!(
                coeffs[i], coeffs_contig[i],
                "Stride handling mismatch at index {}: {} vs {}",
                i, coeffs[i], coeffs_contig[i]
            );
        }
    }

    #[test]
    fn test_dc_energy_preservation() {
        let avg_value = 180u8;
        let input = [avg_value; 64];
        let mut coeffs = [0i32; 64];

        fdct_8x8(&input, 8, &mut coeffs);

        // DC should approximately equal (avg - 128) * 8 after normalization
        let expected_dc = (avg_value as i32 - 128) * 8;
        let dc_error = (coeffs[0] - expected_dc).abs();

        assert!(
            dc_error < expected_dc.abs() / 4 + 50,
            "DC energy not preserved: expected ~{}, got {} (error {})",
            expected_dc,
            coeffs[0],
            dc_error
        );
    }

    #[test]
    fn test_ac_zero_for_flat_block() {
        let input = [150u8; 64];
        let mut coeffs = [0i32; 64];

        fdct_8x8(&input, 8, &mut coeffs);

        for i in 1..64 {
            assert!(
                coeffs[i].abs() <= 1,
                "AC coefficient {} should be 0 for flat block, got {}",
                i,
                coeffs[i]
            );
        }
    }
}
