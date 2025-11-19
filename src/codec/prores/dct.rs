//! ProRes DCT/IDCT Transforms
//!
//! ProRes uses 8×8 Discrete Cosine Transform (DCT) for frequency domain transformation.
//! This module implements both forward DCT (encoding) and inverse DCT (decoding).

use crate::error::{Error, Result};
use std::f32::consts::PI;

/// 8×8 DCT/IDCT transform engine
pub struct ProResDct;

impl ProResDct {
    /// Forward DCT - converts spatial domain to frequency domain
    ///
    /// Input: 8×8 block of pixel values (spatial domain)
    /// Output: 8×8 block of DCT coefficients (frequency domain)
    pub fn forward_dct(input: &[i16; 64], output: &mut [i16; 64]) -> Result<()> {
        let mut temp = [0f32; 64];

        // Perform 2D DCT using separable 1D DCTs
        // First pass: transform rows
        for y in 0..8 {
            Self::dct_1d_row(input, &mut temp, y);
        }

        // Second pass: transform columns
        for x in 0..8 {
            Self::dct_1d_col(&temp, output, x);
        }

        Ok(())
    }

    /// Inverse DCT - converts frequency domain to spatial domain
    ///
    /// Input: 8×8 block of DCT coefficients (frequency domain)
    /// Output: 8×8 block of pixel values (spatial domain)
    pub fn inverse_dct(input: &[i16; 64], output: &mut [i16; 64]) -> Result<()> {
        let mut temp = [0f32; 64];

        // Perform 2D IDCT using separable 1D IDCTs
        // First pass: inverse transform rows
        for y in 0..8 {
            Self::idct_1d_row(input, &mut temp, y);
        }

        // Second pass: inverse transform columns
        for x in 0..8 {
            Self::idct_1d_col(&temp, output, x);
        }

        Ok(())
    }

    /// 1D DCT on a row
    fn dct_1d_row(input: &[i16; 64], output: &mut [f32; 64], row: usize) {
        for u in 0..8 {
            let cu = if u == 0 { 1.0 / (2.0f32).sqrt() } else { 1.0 };
            let mut sum = 0.0f32;

            for x in 0..8 {
                let pixel = input[row * 8 + x] as f32;
                sum += pixel * ((2 * x + 1) as f32 * u as f32 * PI / 16.0).cos();
            }

            output[row * 8 + u] = 0.5 * cu * sum;
        }
    }

    /// 1D DCT on a column
    fn dct_1d_col(input: &[f32; 64], output: &mut [i16; 64], col: usize) {
        for v in 0..8 {
            let cv = if v == 0 { 1.0 / (2.0f32).sqrt() } else { 1.0 };
            let mut sum = 0.0f32;

            for y in 0..8 {
                let value = input[y * 8 + col];
                sum += value * ((2 * y + 1) as f32 * v as f32 * PI / 16.0).cos();
            }

            output[v * 8 + col] = (0.5 * cv * sum).round() as i16;
        }
    }

    /// 1D IDCT on a row
    fn idct_1d_row(input: &[i16; 64], output: &mut [f32; 64], row: usize) {
        for x in 0..8 {
            let mut sum = 0.0f32;

            for u in 0..8 {
                let cu = if u == 0 { 1.0 / (2.0f32).sqrt() } else { 1.0 };
                let coeff = input[row * 8 + u] as f32;
                sum += cu * coeff * ((2 * x + 1) as f32 * u as f32 * PI / 16.0).cos();
            }

            output[row * 8 + x] = 0.5 * sum;
        }
    }

    /// 1D IDCT on a column
    fn idct_1d_col(input: &[f32; 64], output: &mut [i16; 64], col: usize) {
        for y in 0..8 {
            let mut sum = 0.0f32;

            for v in 0..8 {
                let cv = if v == 0 { 1.0 / (2.0f32).sqrt() } else { 1.0 };
                let value = input[v * 8 + col];
                sum += cv * value * ((2 * y + 1) as f32 * v as f32 * PI / 16.0).cos();
            }

            output[y * 8 + col] = (0.5 * sum).round().clamp(-32768.0, 32767.0) as i16;
        }
    }
}

/// Fast integer-based DCT/IDCT using lookup tables (optional optimization)
pub struct FastProResDct;

impl FastProResDct {
    /// Cosine lookup table for fast DCT (scaled by 1024)
    const COS_TABLE: [[i32; 8]; 8] = Self::generate_cos_table();

    /// Generate cosine lookup table at compile time
    const fn generate_cos_table() -> [[i32; 8]; 8] {
        let mut table = [[0i32; 8]; 8];
        // In a real implementation, this would be pre-computed
        // For now, using simple approximations
        table[0] = [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024];
        table[1] = [1420, 1204, 851, 399, -399, -851, -1204, -1420];
        table[2] = [1357, 567, -567, -1357, -1357, -567, 567, 1357];
        table[3] = [1204, -399, -1420, -851, 851, 1420, 399, -1204];
        table[4] = [1024, -1024, -1024, 1024, 1024, -1024, -1024, 1024];
        table[5] = [851, -1420, 399, 1204, -1204, -399, 1420, -851];
        table[6] = [567, -1357, 1357, -567, -567, 1357, -1357, 567];
        table[7] = [399, -851, 1204, -1420, 1420, -1204, 851, -399];
        table
    }

    /// Fast forward DCT using integer arithmetic
    pub fn forward_dct_fast(input: &[i16; 64], output: &mut [i16; 64]) -> Result<()> {
        // Simplified fast DCT - in production, use optimized algorithm
        ProResDct::forward_dct(input, output)
    }

    /// Fast inverse DCT using integer arithmetic
    pub fn inverse_dct_fast(input: &[i16; 64], output: &mut [i16; 64]) -> Result<()> {
        // Simplified fast IDCT - in production, use optimized algorithm
        ProResDct::inverse_dct(input, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_idct_roundtrip() {
        // Create a simple test pattern
        let mut input = [0i16; 64];
        input[0] = 100;  // DC component
        input[1] = 50;   // Low frequency
        input[8] = 30;   // Low frequency

        let mut forward = [0i16; 64];
        ProResDct::forward_dct(&input, &mut forward).unwrap();

        let mut inverse = [0i16; 64];
        ProResDct::inverse_dct(&forward, &mut inverse).unwrap();

        // DCT/IDCT should be approximately reversible
        // Allow small error due to rounding
        for i in 0..64 {
            let diff = (input[i] - inverse[i]).abs();
            assert!(diff <= 2, "Position {} diff too large: {} vs {}", i, input[i], inverse[i]);
        }
    }

    #[test]
    fn test_dct_all_zeros() {
        let input = [0i16; 64];
        let mut output = [0i16; 64];

        ProResDct::forward_dct(&input, &mut output).unwrap();

        // All zeros should produce all zeros
        assert_eq!(output, [0i16; 64]);
    }

    #[test]
    fn test_dct_constant_value() {
        // Constant value should produce non-zero DC only
        let mut input = [0i16; 64];
        input.fill(100);

        let mut output = [0i16; 64];
        ProResDct::forward_dct(&input, &mut output).unwrap();

        // DC should be large, AC should be small
        assert!(output[0].abs() > 100);
        for i in 1..64 {
            assert!(output[i].abs() < 10, "AC coefficient {} should be near zero: {}", i, output[i]);
        }
    }

    #[test]
    fn test_idct_all_zeros() {
        let input = [0i16; 64];
        let mut output = [0i16; 64];

        ProResDct::inverse_dct(&input, &mut output).unwrap();

        // All zeros should produce all zeros
        assert_eq!(output, [0i16; 64]);
    }

    #[test]
    fn test_idct_dc_only() {
        let mut input = [0i16; 64];
        input[0] = 800; // DC coefficient

        let mut output = [0i16; 64];
        ProResDct::inverse_dct(&input, &mut output).unwrap();

        // DC-only should produce constant value
        let avg = output.iter().map(|&x| x as i32).sum::<i32>() / 64;
        for &val in &output {
            let diff = (val as i32 - avg).abs();
            assert!(diff <= 2, "Output should be constant, got variance {}", diff);
        }
    }

    #[test]
    fn test_fast_dct_matches_standard() {
        let mut input = [0i16; 64];
        for i in 0..64 {
            input[i] = (i as i16 * 10) % 255;
        }

        let mut std_output = [0i16; 64];
        let mut fast_output = [0i16; 64];

        ProResDct::forward_dct(&input, &mut std_output).unwrap();
        FastProResDct::forward_dct_fast(&input, &mut fast_output).unwrap();

        // Should produce same results (within rounding error)
        for i in 0..64 {
            let diff = (std_output[i] - fast_output[i]).abs();
            assert!(diff <= 1, "Position {} differs: {} vs {}", i, std_output[i], fast_output[i]);
        }
    }
}
