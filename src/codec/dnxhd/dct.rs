//! DNxHD DCT/IDCT transforms
//!
//! 8×8 forward and inverse Discrete Cosine Transform

use crate::error::Result;
use std::f32::consts::PI;

/// DNxHD DCT transformer
pub struct DnxhdDct;

impl DnxhdDct {
    /// Forward DCT (8×8)
    /// Transforms spatial domain block to frequency domain
    pub fn forward_dct(input: &[i16; 64], output: &mut [i16; 64]) -> Result<()> {
        let mut temp = [0f32; 64];

        // Forward DCT-II
        for v in 0..8 {
            for u in 0..8 {
                let mut sum = 0.0f32;

                for y in 0..8 {
                    for x in 0..8 {
                        let pixel = input[y * 8 + x] as f32;
                        let cos_u = ((2 * x + 1) as f32 * u as f32 * PI / 16.0).cos();
                        let cos_v = ((2 * y + 1) as f32 * v as f32 * PI / 16.0).cos();
                        sum += pixel * cos_u * cos_v;
                    }
                }

                let cu = if u == 0 { 1.0 / 2.0f32.sqrt() } else { 1.0 };
                let cv = if v == 0 { 1.0 / 2.0f32.sqrt() } else { 1.0 };
                temp[v * 8 + u] = 0.25 * cu * cv * sum;
            }
        }

        // Convert to i16
        for i in 0..64 {
            output[i] = temp[i].round() as i16;
        }

        Ok(())
    }

    /// Inverse DCT (8×8)
    /// Transforms frequency domain coefficients to spatial domain
    pub fn inverse_dct(input: &[i16; 64], output: &mut [i16; 64]) -> Result<()> {
        let mut temp = [0f32; 64];

        // Inverse DCT-II
        for y in 0..8 {
            for x in 0..8 {
                let mut sum = 0.0f32;

                for v in 0..8 {
                    for u in 0..8 {
                        let coeff = input[v * 8 + u] as f32;
                        let cu = if u == 0 { 1.0 / 2.0f32.sqrt() } else { 1.0 };
                        let cv = if v == 0 { 1.0 / 2.0f32.sqrt() } else { 1.0 };
                        let cos_u = ((2 * x + 1) as f32 * u as f32 * PI / 16.0).cos();
                        let cos_v = ((2 * y + 1) as f32 * v as f32 * PI / 16.0).cos();
                        sum += cu * cv * coeff * cos_u * cos_v;
                    }
                }

                temp[y * 8 + x] = 0.25 * sum;
            }
        }

        // Convert to i16
        for i in 0..64 {
            output[i] = temp[i].round().clamp(-2048.0, 2047.0) as i16;
        }

        Ok(())
    }
}

/// Fast integer-based DCT (optimized version)
pub struct FastDnxhdDct;

impl FastDnxhdDct {
    /// Fast forward DCT using integer arithmetic
    pub fn forward_dct(input: &[i16; 64], output: &mut [i16; 64]) -> Result<()> {
        // For now, use the floating-point version
        // TODO: Implement fast integer DCT
        DnxhdDct::forward_dct(input, output)
    }

    /// Fast inverse DCT using integer arithmetic
    pub fn inverse_dct(input: &[i16; 64], output: &mut [i16; 64]) -> Result<()> {
        // For now, use the floating-point version
        // TODO: Implement fast integer IDCT
        DnxhdDct::inverse_dct(input, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_dc_only() {
        let mut input = [0i16; 64];
        input[0] = 128; // DC component

        let mut dct_output = [0i16; 64];
        DnxhdDct::forward_dct(&input, &mut dct_output).unwrap();

        // DC coefficient should be significant
        assert_ne!(dct_output[0], 0);

        // Most AC coefficients should be small
        for i in 1..64 {
            assert!(dct_output[i].abs() < 100);
        }
    }

    #[test]
    fn test_dct_roundtrip() {
        let mut input = [0i16; 64];
        // Create a simple pattern
        for i in 0..64 {
            input[i] = ((i % 8) * 16) as i16;
        }

        let mut dct_output = [0i16; 64];
        DnxhdDct::forward_dct(&input, &mut dct_output).unwrap();

        let mut reconstructed = [0i16; 64];
        DnxhdDct::inverse_dct(&dct_output, &mut reconstructed).unwrap();

        // Should approximately reconstruct
        for i in 0..64 {
            let diff = (input[i] - reconstructed[i]).abs();
            assert!(diff < 5, "Position {}: diff={}", i, diff);
        }
    }

    #[test]
    fn test_dct_all_zero() {
        let input = [0i16; 64];
        let mut output = [1i16; 64]; // Initialize to non-zero

        DnxhdDct::forward_dct(&input, &mut output).unwrap();

        // All zeros should produce all zeros
        for i in 0..64 {
            assert_eq!(output[i], 0, "Position {} should be 0", i);
        }
    }

    #[test]
    fn test_fast_dct_roundtrip() {
        let mut input = [0i16; 64];
        for i in 0..64 {
            input[i] = (i as i16 * 4) % 128;
        }

        let mut dct_output = [0i16; 64];
        FastDnxhdDct::forward_dct(&input, &mut dct_output).unwrap();

        let mut reconstructed = [0i16; 64];
        FastDnxhdDct::inverse_dct(&dct_output, &mut reconstructed).unwrap();

        for i in 0..64 {
            let diff = (input[i] - reconstructed[i]).abs();
            assert!(diff < 5, "Position {}: diff={}", i, diff);
        }
    }
}
