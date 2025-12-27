//! Quantization and Dequantization for H.265/HEVC
//!
//! This module implements the quantization and dequantization (scaling) of
//! transform coefficients in H.265. Quantization reduces the precision of
//! coefficients to achieve compression, controlled by the QP (Quantization Parameter).
//!
//! # H.265 Quantization
//!
//! - **QP Range**: 0-51 (0 = highest quality, 51 = highest compression)
//! - **Scaling**: Uses multiplication + shift for efficiency
//! - **Transform Size Dependent**: Different scaling for different block sizes

use crate::error::{Error, Result};

/// Quantization and dequantization engine
pub struct Quantizer {
    /// Current quantization parameter (0-51)
    qp: u8,
    /// Bit depth (8, 10, or 12)
    bit_depth: u8,
}

impl Quantizer {
    /// Create a new quantizer with given QP
    pub fn new(qp: u8, bit_depth: u8) -> Result<Self> {
        if qp > 51 {
            return Err(Error::Codec(format!("Invalid QP: {}", qp)));
        }

        if bit_depth != 8 && bit_depth != 10 && bit_depth != 12 {
            return Err(Error::Codec(format!(
                "Invalid bit depth: {}",
                bit_depth
            )));
        }

        Ok(Self { qp, bit_depth })
    }

    /// Set quantization parameter
    pub fn set_qp(&mut self, qp: u8) -> Result<()> {
        if qp > 51 {
            return Err(Error::Codec(format!("Invalid QP: {}", qp)));
        }
        self.qp = qp;
        Ok(())
    }

    /// Get current QP
    pub fn qp(&self) -> u8 {
        self.qp
    }

    /// Dequantize (inverse quantize) transform coefficients
    ///
    /// This scales quantized coefficients back to their approximate original values.
    /// Used during decoding.
    pub fn dequantize(&self, coeffs: &[i16], output: &mut [i16], log2_size: u8) -> Result<()> {
        if coeffs.len() != output.len() {
            return Err(Error::Codec(
                "Coefficient arrays must be same length".to_string(),
            ));
        }

        let size = 1usize << log2_size;
        if coeffs.len() != size * size {
            return Err(Error::Codec(format!(
                "Expected {} coefficients for {}x{} block",
                size * size,
                size,
                size
            )));
        }

        // Get dequantization scale and shift
        let (scale, shift) = self.get_dequant_params(log2_size);

        // Dequantize each coefficient
        for i in 0..coeffs.len() {
            let level = coeffs[i] as i32;
            let dequant = (level * scale) >> shift;
            output[i] = dequant.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }

        Ok(())
    }

    /// Quantize transform coefficients
    ///
    /// This reduces the precision of coefficients for compression.
    /// Used during encoding.
    pub fn quantize(&self, coeffs: &[i16], output: &mut [i16], log2_size: u8) -> Result<()> {
        if coeffs.len() != output.len() {
            return Err(Error::Codec(
                "Coefficient arrays must be same length".to_string(),
            ));
        }

        let size = 1usize << log2_size;
        if coeffs.len() != size * size {
            return Err(Error::Codec(format!(
                "Expected {} coefficients for {}x{} block",
                size * size,
                size,
                size
            )));
        }

        // Get quantization scale and shift
        let (scale, shift) = self.get_quant_params(log2_size);
        let add = 1 << (shift - 1); // Rounding offset

        // Quantize each coefficient
        for i in 0..coeffs.len() {
            let level = coeffs[i] as i32;
            let sign = if level < 0 { -1 } else { 1 };
            let abs_level = level.abs();

            let quant = ((abs_level * scale + add) >> shift) * sign;
            output[i] = quant.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }

        Ok(())
    }

    /// Get dequantization parameters (scale, shift) for given transform size
    fn get_dequant_params(&self, log2_size: u8) -> (i32, i32) {
        // H.265 dequantization formula:
        // d[i][j] = c[i][j] * m[QP % 6][i][j] << (QP / 6)
        //
        // Simplified using scale and shift:
        // d[i][j] = (c[i][j] * scale) >> shift

        let qp_per = (self.qp / 6) as i32;
        let qp_rem = (self.qp % 6) as i32;

        // Dequant scaling table for QP % 6
        let scale = DEQUANT_SCALES[qp_rem as usize];

        // Shift depends on transform size and QP
        // Larger transforms need more shift
        let base_shift = 14; // Standard shift for 4×4
        let size_shift = if log2_size > 2 {
            (log2_size - 2) as i32
        } else {
            0
        };

        let shift = base_shift - qp_per + size_shift;

        (scale, shift)
    }

    /// Get quantization parameters (scale, shift) for given transform size
    fn get_quant_params(&self, log2_size: u8) -> (i32, i32) {
        let qp_per = (self.qp / 6) as i32;
        let qp_rem = (self.qp % 6) as i32;

        // Quant scaling table for QP % 6
        let scale = QUANT_SCALES[qp_rem as usize];

        // Shift depends on transform size and QP
        let base_shift = 15; // Standard shift for 4×4
        let size_shift = if log2_size > 2 {
            (log2_size - 2) as i32
        } else {
            0
        };

        let shift = base_shift + qp_per + size_shift;

        (scale, shift)
    }

    /// Calculate step size for given QP
    ///
    /// This gives an approximate quantization step size.
    /// Higher QP = larger step = more quantization = lower quality.
    pub fn step_size(&self) -> f64 {
        // H.265 step size formula: step = 2^(QP / 6)
        2.0_f64.powf(self.qp as f64 / 6.0)
    }
}

/// Dequantization scaling factors for QP % 6
///
/// These are derived from the H.265 specification and provide
/// the base scaling values for different QP remainders.
const DEQUANT_SCALES: [i32; 6] = [40, 45, 51, 57, 64, 72];

/// Quantization scaling factors for QP % 6
///
/// Reciprocals of dequant scales, used for forward quantization.
const QUANT_SCALES: [i32; 6] = [26214, 23302, 20560, 18396, 16384, 14564];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantizer_creation() {
        let quant = Quantizer::new(26, 8);
        assert!(quant.is_ok());
        assert_eq!(quant.unwrap().qp(), 26);
    }

    #[test]
    fn test_quantizer_invalid_qp() {
        let quant = Quantizer::new(52, 8);
        assert!(quant.is_err());
    }

    #[test]
    fn test_quantizer_invalid_bit_depth() {
        let quant = Quantizer::new(26, 9);
        assert!(quant.is_err());
    }

    #[test]
    fn test_set_qp() {
        let mut quant = Quantizer::new(26, 8).unwrap();
        assert!(quant.set_qp(30).is_ok());
        assert_eq!(quant.qp(), 30);
    }

    #[test]
    fn test_set_invalid_qp() {
        let mut quant = Quantizer::new(26, 8).unwrap();
        assert!(quant.set_qp(52).is_err());
    }

    #[test]
    fn test_dequantize_4x4() {
        let quant = Quantizer::new(26, 8).unwrap();
        let coeffs = vec![100i16; 16]; // 4×4 block
        let mut output = vec![0i16; 16];

        let result = quant.dequantize(&coeffs, &mut output, 2); // log2(4) = 2
        assert!(result.is_ok());

        // Output should be scaled up from input
        assert!(output[0] > coeffs[0]);
    }

    #[test]
    fn test_dequantize_8x8() {
        let quant = Quantizer::new(30, 8).unwrap();
        let coeffs = vec![50i16; 64]; // 8×8 block
        let mut output = vec![0i16; 64];

        let result = quant.dequantize(&coeffs, &mut output, 3); // log2(8) = 3
        assert!(result.is_ok());
    }

    #[test]
    fn test_quantize_4x4() {
        let quant = Quantizer::new(26, 8).unwrap();
        let coeffs = vec![1000i16; 16]; // 4×4 block
        let mut output = vec![0i16; 16];

        let result = quant.quantize(&coeffs, &mut output, 2);
        assert!(result.is_ok());

        // Output should be scaled down from input
        assert!(output[0].abs() < coeffs[0].abs());
    }

    #[test]
    fn test_quantize_negative_coeffs() {
        let quant = Quantizer::new(26, 8).unwrap();
        let coeffs = vec![-1000i16; 16];
        let mut output = vec![0i16; 16];

        let result = quant.quantize(&coeffs, &mut output, 2);
        assert!(result.is_ok());

        // Output should be negative
        assert!(output[0] < 0);
    }

    #[test]
    fn test_dequantize_invalid_size() {
        let quant = Quantizer::new(26, 8).unwrap();
        let coeffs = vec![100i16; 15]; // Wrong size
        let mut output = vec![0i16; 15];

        let result = quant.dequantize(&coeffs, &mut output, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantize_dequantize_round_trip() {
        let quant = Quantizer::new(20, 8).unwrap();
        let original = vec![500, -300, 200, -100, 50, -25, 10, -5, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut quantized = vec![0i16; 16];
        let mut reconstructed = vec![0i16; 16];

        quant.quantize(&original, &mut quantized, 2).unwrap();
        quant
            .dequantize(&quantized, &mut reconstructed, 2)
            .unwrap();

        // Reconstructed should be close to original (with quantization loss)
        for i in 0..16 {
            let diff = (original[i] - reconstructed[i]).abs();
            // Allow some quantization error
            assert!(diff < original[i].abs() / 2 + 100);
        }
    }

    #[test]
    fn test_step_size_increases_with_qp() {
        let quant_low = Quantizer::new(10, 8).unwrap();
        let quant_high = Quantizer::new(40, 8).unwrap();

        assert!(quant_high.step_size() > quant_low.step_size());
    }

    #[test]
    fn test_qp_0_has_smallest_step() {
        let quant = Quantizer::new(0, 8).unwrap();
        let step = quant.step_size();
        assert!(step >= 1.0 && step < 2.0);
    }

    #[test]
    fn test_qp_51_has_largest_step() {
        let quant = Quantizer::new(51, 8).unwrap();
        let step = quant.step_size();
        assert!(step > 100.0);
    }

    #[test]
    fn test_different_bit_depths() {
        for bit_depth in [8, 10, 12] {
            let quant = Quantizer::new(26, bit_depth);
            assert!(quant.is_ok());
        }
    }

    #[test]
    fn test_all_valid_qp_values() {
        for qp in 0..=51 {
            let quant = Quantizer::new(qp, 8);
            assert!(quant.is_ok());
        }
    }
}
