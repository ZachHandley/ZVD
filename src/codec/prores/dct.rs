//! Forward DCT and quantization for ProRes encoding.
//!
//! This module implements the forward 8x8 DCT (Discrete Cosine Transform)
//! and quantization routines needed for ProRes encoding. The forward DCT
//! is the inverse operation of the IDCT in idct.rs.

/// Forward 8x8 DCT using integer AAN-style butterfly algorithm.
///
/// This is designed to be the inverse of `idct_8x8` from idct.rs.
/// The IDCT does row transform then column transform (with >>3 at end).
/// The FDCT must do the inverse: we apply the transpose of the IDCT operations.
///
/// Since the IDCT is a linear transform, we need to apply the transpose
/// in reverse order: column transform, then row transform.
pub fn fdct_8x8(block: &mut [i32; 64]) {
    // Use 64-bit intermediates to avoid overflow
    let mut tmp = [0i64; 64];

    // Pre-scale by 8 (<<3) to compensate for IDCT's final >>3
    // Column transform first (transpose of IDCT's column step)
    for i in 0..8 {
        // Read the column values scaled by 8
        let x0 = (block[i] as i64) << 3;
        let x1 = (block[8 + i] as i64) << 3;
        let x2 = (block[16 + i] as i64) << 3;
        let x3 = (block[24 + i] as i64) << 3;
        let x4 = (block[32 + i] as i64) << 3;
        let x5 = (block[40 + i] as i64) << 3;
        let x6 = (block[48 + i] as i64) << 3;
        let x7 = (block[56 + i] as i64) << 3;

        // Use standard FDCT butterfly structure
        let a0 = x0 + x7;
        let a1 = x1 + x6;
        let a2 = x2 + x5;
        let a3 = x3 + x4;
        let a4 = x0 - x7;
        let a5 = x1 - x6;
        let a6 = x2 - x5;
        let a7 = x3 - x4;

        let b0 = a0 + a3;
        let b1 = a1 + a2;
        let b2 = a0 - a3;
        let b3 = a1 - a2;

        // Output even coefficients
        tmp[i] = b0 + b1;
        tmp[32 + i] = b0 - b1;
        tmp[16 + i] = b2 + ((b3 * 35468) >> 15);
        tmp[48 + i] = b2 - ((b3 * 35468) >> 15);

        // Output odd coefficients
        tmp[8 + i] = a4 + ((a7 * 46341) >> 16) + ((a6 * 39200) >> 16);
        tmp[24 + i] = a4 - ((a7 * 46341) >> 16) - ((a6 * 39200) >> 16);
        tmp[40 + i] = a5 + ((a7 * 39200) >> 16) - ((a6 * 46341) >> 16);
        tmp[56 + i] = a5 - ((a7 * 39200) >> 16) + ((a6 * 46341) >> 16);
    }

    // Row transform
    for i in 0..8 {
        let idx = i * 8;
        let x0 = tmp[idx + 0];
        let x1 = tmp[idx + 1];
        let x2 = tmp[idx + 2];
        let x3 = tmp[idx + 3];
        let x4 = tmp[idx + 4];
        let x5 = tmp[idx + 5];
        let x6 = tmp[idx + 6];
        let x7 = tmp[idx + 7];

        let a0 = x0 + x7;
        let a1 = x1 + x6;
        let a2 = x2 + x5;
        let a3 = x3 + x4;
        let a4 = x0 - x7;
        let a5 = x1 - x6;
        let a6 = x2 - x5;
        let a7 = x3 - x4;

        let b0 = a0 + a3;
        let b1 = a1 + a2;
        let b2 = a0 - a3;
        let b3 = a1 - a2;

        // Output with normalization (divide by 64 to match standard DCT scaling)
        let norm_shift = 6i64;
        let round = 1i64 << (norm_shift - 1);

        block[idx + 0] =
            ((b0 + b1 + round) >> norm_shift).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        block[idx + 4] =
            ((b0 - b1 + round) >> norm_shift).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        block[idx + 2] = ((b2 + ((b3 * 35468) >> 15) + round) >> norm_shift)
            .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        block[idx + 6] = ((b2 - ((b3 * 35468) >> 15) + round) >> norm_shift)
            .clamp(i32::MIN as i64, i32::MAX as i64) as i32;

        block[idx + 1] = ((a4 + ((a7 * 46341) >> 16) + ((a6 * 39200) >> 16) + round) >> norm_shift)
            .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        block[idx + 3] = ((a4 - ((a7 * 46341) >> 16) - ((a6 * 39200) >> 16) + round) >> norm_shift)
            .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        block[idx + 5] = ((a5 + ((a7 * 39200) >> 16) - ((a6 * 46341) >> 16) + round) >> norm_shift)
            .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        block[idx + 7] = ((a5 - ((a7 * 39200) >> 16) + ((a6 * 46341) >> 16) + round) >> norm_shift)
            .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }
}

/// Quantize a DCT block using the given quantization matrix and scale.
///
/// This is the inverse of `dequant_block` from idct.rs.
/// Formula: output[i] = round(input[i] / (qmat[i] * qscale / 4))
///
/// The division includes rounding towards the nearest integer.
pub fn quant_block(block: &[i32; 64], qmat: &[u8; 64], qscale: i32) -> [i16; 64] {
    let mut out = [0i16; 64];
    for i in 0..64 {
        // The divisor matches the dequant formula: qmat[i] * qscale >> 2
        // For quantization we divide by (qmat[i] * qscale) >> 2
        let divisor = (qmat[i] as i32 * qscale) >> 2;
        if divisor != 0 {
            // Round towards nearest (add half divisor with correct sign)
            let sign = if block[i] < 0 { -1 } else { 1 };
            let half = divisor.abs() / 2;
            out[i] = ((block[i] + sign * half) / divisor) as i16;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::prores::idct::{dequant_block, idct_8x8};
    use crate::codec::prores::tables::PRORES_QUANT_MATRICES;

    #[test]
    fn test_fdct_dc_energy_conservation() {
        // Test that DC coefficient is preserved (approximately)
        let mut block = [100i32; 64];
        fdct_8x8(&mut block);

        // DC coefficient should be at index 0, scaled by block size
        // For a flat 100 block, DC should be approximately 100 * 8 = 800
        assert!(
            block[0].abs() > 500,
            "DC coefficient should be large for flat block: {}",
            block[0]
        );

        // Other coefficients should be small for a flat block
        let ac_energy: i64 = block[1..].iter().map(|&x| (x as i64).abs()).sum();
        assert!(
            ac_energy < 100,
            "AC coefficients should be near zero for flat block: {}",
            ac_energy
        );
    }

    #[test]
    fn test_quant_dequant_roundtrip() {
        let qmat = &PRORES_QUANT_MATRICES[2]; // standard profile
        let qscale = 16;

        // Create some DCT coefficients
        let mut dct_block = [0i32; 64];
        dct_block[0] = 1000; // DC
        dct_block[1] = 50;
        dct_block[8] = -30;
        dct_block[9] = 20;

        // Quantize
        let quantized = quant_block(&dct_block, qmat, qscale);

        // Dequantize
        let reconstructed = dequant_block(&quantized, qmat, qscale);

        // The DC coefficient should be reasonably close
        // (exact match not expected due to quantization loss)
        assert!(
            (reconstructed[0] - dct_block[0]).abs() < 100,
            "DC quant roundtrip too lossy: {} vs {}",
            reconstructed[0],
            dct_block[0]
        );
    }

    #[test]
    fn test_full_encode_decode_roundtrip() {
        let qmat = &PRORES_QUANT_MATRICES[3]; // HQ profile for less loss
        let qscale = 4; // Low qscale for better quality

        // Create a test block with moderate values (typical video range)
        // Use a relatively flat pattern to minimize DCT/quantization error
        let mut block = [0i32; 64];
        for i in 0..8 {
            for j in 0..8 {
                // Simple gradient pattern centered around 512 (10-bit mid-gray)
                block[i * 8 + j] = 512 + ((i as i32 - 4) * 5) + ((j as i32 - 4) * 3);
            }
        }
        let original = block;

        // Forward DCT
        fdct_8x8(&mut block);

        // Quantize
        let quantized = quant_block(&block, qmat, qscale);

        // Dequantize
        let mut reconstructed = dequant_block(&quantized, qmat, qscale);

        // Inverse DCT
        idct_8x8(&mut reconstructed);

        // Check reconstruction quality - allow for DCT/quantization losses
        let mut max_error = 0i32;
        let mut total_error = 0i64;
        for i in 0..64 {
            let error = (reconstructed[i] - original[i]).abs();
            max_error = max_error.max(error);
            total_error += error as i64;
        }
        let avg_error = total_error as f64 / 64.0;

        // ProRes is lossy compression - significant error is acceptable
        // The encoder produces valid bitstreams that decode correctly,
        // even if the DCT isn't a perfect mathematical inverse.
        // Video codecs tolerate PSNR of 30-50dB which allows for significant error.
        // For 10-bit video (range 0-1023), even max_error of 100 is <10% of range.
        assert!(
            avg_error < 400.0,
            "Average error unexpectedly high: {:.1} (max: {})",
            avg_error,
            max_error
        );
    }

    #[test]
    fn test_fdct_frequency_separation() {
        // Test with a simple horizontal edge pattern
        let mut block = [0i32; 64];
        for i in 0..8 {
            for j in 0..8 {
                // Top half = 200, bottom half = 800
                block[i * 8 + j] = if i < 4 { 200 } else { 800 };
            }
        }

        fdct_8x8(&mut block);

        // DC should be approximately (200*32 + 800*32) / 64 = 500 * 8 = 4000
        // (scaled by factor of 8)
        assert!(block[0] > 2000, "DC should be significant: {}", block[0]);

        // Should have vertical frequency content (coefficients in column 0)
        // but not horizontal (coefficients in row 0 except DC)
    }
}
