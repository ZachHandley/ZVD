//! H.265/HEVC Transform and Quantization
//!
//! H.265 uses integer approximations of DCT and DST transforms.
//! Transform sizes: 4×4, 8×8, 16×16, 32×32
//!
//! For intra prediction, 4×4 luma blocks can use either:
//! - DST (Discrete Sine Transform) - better for directional content
//! - DCT (Discrete Cosine Transform) - general purpose

use crate::error::Result;

/// Transform size enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformSize {
    /// 4×4 transform
    Size4 = 4,
    /// 8×8 transform
    Size8 = 8,
    /// 16×16 transform
    Size16 = 16,
    /// 32×32 transform
    Size32 = 32,
}

impl TransformSize {
    /// Get transform size from log2 value
    pub fn from_log2(log2_size: u8) -> Option<Self> {
        match log2_size {
            2 => Some(TransformSize::Size4),
            3 => Some(TransformSize::Size8),
            4 => Some(TransformSize::Size16),
            5 => Some(TransformSize::Size32),
            _ => None,
        }
    }

    /// Get size in pixels
    pub fn size(&self) -> usize {
        *self as usize
    }
}

/// H.265 Transform processor
pub struct Transform {
    /// Bit depth
    bit_depth: u8,
}

impl Transform {
    /// Create a new transform processor
    pub fn new(bit_depth: u8) -> Self {
        Transform { bit_depth }
    }

    /// Inverse transform (coefficient domain ’ pixel domain)
    ///
    /// # Arguments
    /// * `coeffs` - Input coefficients (size × size)
    /// * `dst` - Output residual pixels (size × size)
    /// * `size` - Transform size
    /// * `use_dst` - Use DST instead of DCT (only for 4×4 luma intra)
    pub fn inverse_transform(
        &self,
        coeffs: &[i16],
        dst: &mut [i16],
        size: TransformSize,
        use_dst: bool,
    ) -> Result<()> {
        match size {
            TransformSize::Size4 => {
                if use_dst {
                    self.inverse_dst_4x4(coeffs, dst)
                } else {
                    self.inverse_dct_4x4(coeffs, dst)
                }
            }
            TransformSize::Size8 => {
                // Phase 8.2: 8×8 DCT not yet implemented
                // For now, zero out the destination
                dst[..64].fill(0);
                Ok(())
            }
            TransformSize::Size16 | TransformSize::Size32 => {
                // Phase 8.2: Larger transforms not yet implemented
                let n = size.size();
                dst[..n * n].fill(0);
                Ok(())
            }
        }
    }

    /// Inverse 4×4 DCT transform
    ///
    /// Uses H.265's integer approximation of DCT-II
    fn inverse_dct_4x4(&self, coeffs: &[i16], dst: &mut [i16]) -> Result<()> {
        // H.265 4×4 DCT matrix (transposed for column operations)
        // Each row represents a basis function
        const DCT4: [[i32; 4]; 4] = [
            [64,  64,  64,  64],
            [83,  36, -36, -83],
            [64, -64, -64,  64],
            [36, -83,  83, -36],
        ];

        let mut temp = [0i32; 16];

        // First pass: process rows (horizontal transform)
        for i in 0..4 {
            for j in 0..4 {
                let mut sum = 0i32;
                for k in 0..4 {
                    sum += DCT4[j][k] * coeffs[i * 4 + k] as i32;
                }
                temp[i * 4 + j] = sum;
            }
        }

        // Second pass: process columns (vertical transform)
        const SHIFT: i32 = 7 + 7; // Two passes, each with shift of 7
        let add = 1 << (SHIFT - 1); // Rounding offset

        for j in 0..4 {
            for i in 0..4 {
                let mut sum = 0i32;
                for k in 0..4 {
                    sum += DCT4[i][k] * temp[k * 4 + j];
                }
                // Add rounding offset and shift
                let result = (sum + add) >> SHIFT;
                // Clamp to i16 range
                dst[i * 4 + j] = result.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            }
        }

        Ok(())
    }

    /// Inverse 4×4 DST transform
    ///
    /// DST (Discrete Sine Transform) is used for 4×4 luma intra blocks in H.265.
    /// It's more efficient for directional (angular) content.
    fn inverse_dst_4x4(&self, coeffs: &[i16], dst: &mut [i16]) -> Result<()> {
        // H.265 4×4 DST matrix
        const DST4: [[i32; 4]; 4] = [
            [29,  55,  74,  84],
            [74,  74,   0, -74],
            [84, -29, -74,  55],
            [55, -84,  74, -29],
        ];

        let mut temp = [0i32; 16];

        // First pass: process rows
        for i in 0..4 {
            for j in 0..4 {
                let mut sum = 0i32;
                for k in 0..4 {
                    sum += DST4[j][k] * coeffs[i * 4 + k] as i32;
                }
                temp[i * 4 + j] = sum;
            }
        }

        // Second pass: process columns
        const SHIFT: i32 = 7 + 7;
        let add = 1 << (SHIFT - 1);

        for j in 0..4 {
            for i in 0..4 {
                let mut sum = 0i32;
                for k in 0..4 {
                    sum += DST4[i][k] * temp[k * 4 + j];
                }
                let result = (sum + add) >> SHIFT;
                dst[i * 4 + j] = result.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            }
        }

        Ok(())
    }

    /// Add residual to prediction (reconstruction)
    ///
    /// # Arguments
    /// * `pred` - Prediction pixels (u16, bit_depth range)
    /// * `residual` - Residual pixels (i16, signed)
    /// * `dst` - Output reconstructed pixels (u16, bit_depth range)
    /// * `size` - Block size
    pub fn reconstruct(
        &self,
        pred: &[u16],
        residual: &[i16],
        dst: &mut [u16],
        size: usize,
    ) -> Result<()> {
        let max_val = (1 << self.bit_depth) - 1;
        let n = size * size;

        for i in 0..n {
            let recon = (pred[i] as i32 + residual[i] as i32).clamp(0, max_val as i32);
            dst[i] = recon as u16;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_size_from_log2() {
        assert_eq!(TransformSize::from_log2(2), Some(TransformSize::Size4));
        assert_eq!(TransformSize::from_log2(3), Some(TransformSize::Size8));
        assert_eq!(TransformSize::from_log2(4), Some(TransformSize::Size16));
        assert_eq!(TransformSize::from_log2(5), Some(TransformSize::Size32));
        assert_eq!(TransformSize::from_log2(6), None);
    }

    #[test]
    fn test_dct_4x4_zero_coeffs() {
        let transform = Transform::new(8);
        let coeffs = [0i16; 16];
        let mut dst = [0i16; 16];

        transform.inverse_dct_4x4(&coeffs, &mut dst).unwrap();

        // Zero coefficients should produce zero output
        for &val in &dst {
            assert_eq!(val, 0);
        }
    }

    #[test]
    fn test_dct_4x4_dc_only() {
        let transform = Transform::new(8);

        // DC coefficient only (coefficient [0,0] = 64)
        let mut coeffs = [0i16; 16];
        coeffs[0] = 64;

        let mut dst = [0i16; 16];
        transform.inverse_dct_4x4(&coeffs, &mut dst).unwrap();

        // All pixels should be approximately the same (DC component)
        let avg = dst.iter().map(|&x| x as i32).sum::<i32>() / 16;
        for &val in &dst {
            // Allow small variation due to rounding
            assert!((val as i32 - avg).abs() <= 2, "Expected uniform DC, got variance");
        }
    }

    #[test]
    fn test_dst_4x4_zero_coeffs() {
        let transform = Transform::new(8);
        let coeffs = [0i16; 16];
        let mut dst = [0i16; 16];

        transform.inverse_dst_4x4(&coeffs, &mut dst).unwrap();

        for &val in &dst {
            assert_eq!(val, 0);
        }
    }

    #[test]
    fn test_inverse_transform_dispatch() {
        let transform = Transform::new(8);
        let coeffs = [0i16; 16];
        let mut dst = [0i16; 16];

        // Test DCT path
        transform.inverse_transform(&coeffs, &mut dst, TransformSize::Size4, false).unwrap();
        assert!(dst.iter().all(|&x| x == 0));

        // Test DST path
        transform.inverse_transform(&coeffs, &mut dst, TransformSize::Size4, true).unwrap();
        assert!(dst.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_reconstruct() {
        let transform = Transform::new(8);

        // Prediction: all 100
        let pred = [100u16; 16];

        // Residual: +10
        let residual = [10i16; 16];

        let mut dst = [0u16; 16];
        transform.reconstruct(&pred, &residual, &mut dst, 4).unwrap();

        // Output should be 110
        for &val in &dst {
            assert_eq!(val, 110);
        }
    }

    #[test]
    fn test_reconstruct_clamping() {
        let transform = Transform::new(8);

        // Prediction: near max value (255 for 8-bit)
        let pred = [250u16; 16];

        // Residual: +20 (would exceed 255)
        let residual = [20i16; 16];

        let mut dst = [0u16; 16];
        transform.reconstruct(&pred, &residual, &mut dst, 4).unwrap();

        // Should clamp to 255
        for &val in &dst {
            assert_eq!(val, 255);
        }
    }

    #[test]
    fn test_reconstruct_negative_residual() {
        let transform = Transform::new(8);

        let pred = [100u16; 16];
        let residual = [-50i16; 16];

        let mut dst = [0u16; 16];
        transform.reconstruct(&pred, &residual, &mut dst, 4).unwrap();

        // 100 - 50 = 50
        for &val in &dst {
            assert_eq!(val, 50);
        }
    }

    #[test]
    fn test_reconstruct_clamp_negative() {
        let transform = Transform::new(8);

        let pred = [10u16; 16];
        let residual = [-50i16; 16];

        let mut dst = [0u16; 16];
        transform.reconstruct(&pred, &residual, &mut dst, 4).unwrap();

        // 10 - 50 = -40, should clamp to 0
        for &val in &dst {
            assert_eq!(val, 0);
        }
    }
}
