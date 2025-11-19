//! Transform and Quantization Decision for H.265/HEVC Encoder
//!
//! This module implements transform and quantization optimization for encoding.
//!
//! # Transform Decision
//!
//! For each residual block, encoder can choose:
//! - **Apply Transform**: DCT/DST → Quantize
//! - **Transform Skip**: Quantize directly (for screen content)
//!
//! # Quantization Optimization
//!
//! - **Rate-Distortion Optimized Quantization (RDOQ)**
//! - Tests different quantized levels for each coefficient
//! - Minimizes: Distortion + λ × Rate
//!
//! # Process
//!
//! 1. Compute residual (orig - prediction)
//! 2. Test with transform vs transform skip
//! 3. Quantize coefficients
//! 4. (Optional) RDOQ: optimize quantized levels
//! 5. Inverse quantize + inverse transform
//! 6. Calculate RD cost
//! 7. Select best option

use crate::codec::h265::transform::{Transform, TransformSize};
use crate::codec::h265::quant::Quantizer;
use crate::codec::h265::rdo::{RdCost, RdoDecision, DistortionMetric, RateEstimator};
use crate::error::{Error, Result};

/// Transform decision result
#[derive(Debug, Clone)]
pub struct TransformResult {
    /// Use transform skip?
    pub transform_skip: bool,
    /// Quantized coefficients
    pub coeffs: Vec<i16>,
    /// Reconstructed residual
    pub recon: Vec<i16>,
    /// RD cost
    pub cost: RdCost,
    /// Number of non-zero coefficients
    pub num_nonzero: usize,
}

/// Transform decision engine
pub struct TransformDecision {
    /// Transform engine
    transform: Transform,
    /// Quantizer
    quantizer: Quantizer,
    /// RDO decision maker
    rdo: RdoDecision,
    /// Transform size
    size: TransformSize,
}

impl TransformDecision {
    /// Create new transform decision engine
    pub fn new(qp: u8, size: TransformSize, bit_depth: u8) -> Result<Self> {
        let transform = Transform::new(bit_depth)?;
        let quantizer = Quantizer::new(qp, bit_depth)?;
        let rdo = RdoDecision::new(qp, DistortionMetric::SSE);

        Ok(Self {
            transform,
            quantizer,
            rdo,
            size,
        })
    }

    /// Decide whether to use transform or transform skip
    ///
    /// Tests both options and selects better RD cost
    pub fn decide_transform(
        &mut self,
        residual: &[i16],
        width: usize,
        height: usize,
    ) -> Result<TransformResult> {
        // Test with transform
        let transform_result = self.test_with_transform(residual, width, height)?;

        // Test transform skip (for 4×4 only, typically)
        let skip_enabled = width == 4 && height == 4;
        let skip_result = if skip_enabled {
            Some(self.test_transform_skip(residual, width, height)?)
        } else {
            None
        };

        // Select best
        match skip_result {
            Some(skip) if skip.cost.is_better_than(&transform_result.cost) => Ok(skip),
            _ => Ok(transform_result),
        }
    }

    /// Test coding with transform
    fn test_with_transform(
        &mut self,
        residual: &[i16],
        width: usize,
        height: usize,
    ) -> Result<TransformResult> {
        // Forward transform
        let mut coeffs = vec![0i16; width * height];
        self.transform.forward_dct(residual, &mut coeffs, self.size)?;

        // Quantize
        let mut quant_coeffs = vec![0i16; width * height];
        self.quantizer.quantize(&coeffs, &mut quant_coeffs, self.size.to_log2())?;

        // Count non-zero
        let num_nonzero = quant_coeffs.iter().filter(|&&c| c != 0).count();

        // Dequantize
        let mut dequant_coeffs = vec![0i16; width * height];
        self.quantizer.dequantize(&quant_coeffs, &mut dequant_coeffs, self.size.to_log2())?;

        // Inverse transform
        let mut recon = vec![0i16; width * height];
        self.transform.inverse_dct(&dequant_coeffs, &mut recon, self.size)?;

        // Estimate rate
        let rate = RateEstimator::estimate_residual_bits(&quant_coeffs);

        // Calculate distortion (MSE between original residual and reconstructed)
        let distortion = Self::calculate_residual_distortion(residual, &recon, width, height);

        let cost = RdCost::new(distortion, rate, self.rdo.lambda());

        Ok(TransformResult {
            transform_skip: false,
            coeffs: quant_coeffs,
            recon,
            cost,
            num_nonzero,
        })
    }

    /// Test coding with transform skip
    fn test_transform_skip(
        &mut self,
        residual: &[i16],
        width: usize,
        height: usize,
    ) -> Result<TransformResult> {
        // Directly quantize residual (no transform)
        let mut quant_coeffs = vec![0i16; width * height];
        self.quantizer.quantize(residual, &mut quant_coeffs, self.size.to_log2())?;

        let num_nonzero = quant_coeffs.iter().filter(|&&c| c != 0).count();

        // Dequantize
        let mut recon = vec![0i16; width * height];
        self.quantizer.dequantize(&quant_coeffs, &mut recon, self.size.to_log2())?;

        // Estimate rate (+ 1 bit for transform_skip flag)
        let rate = RateEstimator::estimate_residual_bits(&quant_coeffs) + 1;

        let distortion = Self::calculate_residual_distortion(residual, &recon, width, height);

        let cost = RdCost::new(distortion, rate, self.rdo.lambda());

        Ok(TransformResult {
            transform_skip: true,
            coeffs: quant_coeffs,
            recon,
            cost,
            num_nonzero,
        })
    }

    /// Calculate distortion between original and reconstructed residual
    fn calculate_residual_distortion(orig: &[i16], recon: &[i16], width: usize, height: usize) -> u64 {
        let mut sse = 0u64;
        for i in 0..(width * height) {
            let diff = orig[i] as i32 - recon[i] as i32;
            sse += (diff * diff) as u64;
        }
        sse
    }
}

/// Rate-Distortion Optimized Quantization (RDOQ)
pub struct RdoqOptimizer {
    /// Lambda for optimization
    lambda: f64,
}

impl RdoqOptimizer {
    /// Create new RDOQ optimizer
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }

    /// Optimize quantized coefficient levels
    ///
    /// For each coefficient, tests level ± 1 and selects best
    pub fn optimize_levels(
        &self,
        coeffs: &[i16],
        quant_coeffs: &mut [i16],
    ) -> Result<()> {
        for i in 0..coeffs.len() {
            if quant_coeffs[i] == 0 {
                continue; // Skip zero coefficients
            }

            let original_level = quant_coeffs[i];

            // Test level - 1
            let level_minus = original_level - original_level.signum();

            // Test level + 1
            let level_plus = original_level + original_level.signum();

            // Calculate costs (simplified - would need actual distortion)
            let cost_original = self.calculate_level_cost(original_level);
            let cost_minus = self.calculate_level_cost(level_minus);
            let cost_plus = self.calculate_level_cost(level_plus);

            // Select best
            if cost_minus < cost_original && cost_minus < cost_plus {
                quant_coeffs[i] = level_minus;
            } else if cost_plus < cost_original {
                quant_coeffs[i] = level_plus;
            }
        }

        Ok(())
    }

    /// Calculate RD cost for a coefficient level
    fn calculate_level_cost(&self, level: i16) -> f64 {
        // Simplified: just penalize large levels
        let abs_level = level.abs() as f64;
        abs_level * abs_level + self.lambda * abs_level.log2()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_decision_creation() {
        let td = TransformDecision::new(24, TransformSize::Size4x4, 8);
        assert!(td.is_ok());
    }

    #[test]
    fn test_calculate_residual_distortion_zero() {
        let residual = vec![10i16; 16];
        let distortion = TransformDecision::calculate_residual_distortion(&residual, &residual, 4, 4);
        assert_eq!(distortion, 0);
    }

    #[test]
    fn test_calculate_residual_distortion() {
        let orig = vec![10i16; 16];
        let recon = vec![12i16; 16];
        let distortion = TransformDecision::calculate_residual_distortion(&orig, &recon, 4, 4);
        assert_eq!(distortion, 4 * 16); // (2^2) * 16
    }

    #[test]
    fn test_transform_result_creation() {
        let result = TransformResult {
            transform_skip: false,
            coeffs: vec![10, 5, 0, 0],
            recon: vec![9, 6, 0, 0],
            cost: RdCost::new(100, 10, 1.0),
            num_nonzero: 2,
        };

        assert_eq!(result.transform_skip, false);
        assert_eq!(result.num_nonzero, 2);
    }

    #[test]
    fn test_rdoq_optimizer_creation() {
        let rdoq = RdoqOptimizer::new(1.5);
        assert_eq!(rdoq.lambda, 1.5);
    }

    #[test]
    fn test_rdoq_calculate_level_cost() {
        let rdoq = RdoqOptimizer::new(1.0);
        let cost_0 = rdoq.calculate_level_cost(0);
        let cost_1 = rdoq.calculate_level_cost(1);
        let cost_10 = rdoq.calculate_level_cost(10);

        assert_eq!(cost_0, 0.0);
        assert!(cost_1 > 0.0);
        assert!(cost_10 > cost_1);
    }

    #[test]
    fn test_rdoq_optimize_levels() {
        let rdoq = RdoqOptimizer::new(1.0);
        let coeffs = vec![100i16; 16];
        let mut quant_coeffs = vec![10i16; 16];

        let result = rdoq.optimize_levels(&coeffs, &mut quant_coeffs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_decide_transform_4x4() {
        let mut td = TransformDecision::new(24, TransformSize::Size4x4, 8).unwrap();
        let residual = vec![10i16; 16];

        let result = td.decide_transform(&residual, 4, 4);
        assert!(result.is_ok());

        let result = result.unwrap();
        // Should pick either transform or skip
        assert!(result.coeffs.len() == 16);
    }

    #[test]
    fn test_decide_transform_8x8() {
        let mut td = TransformDecision::new(24, TransformSize::Size8x8, 8).unwrap();
        let residual = vec![10i16; 64];

        let result = td.decide_transform(&residual, 8, 8);
        assert!(result.is_ok());

        let result = result.unwrap();
        // Should use transform (no skip for 8×8)
        assert_eq!(result.transform_skip, false);
    }

    #[test]
    fn test_transform_result_num_nonzero() {
        let result = TransformResult {
            transform_skip: false,
            coeffs: vec![10, 0, 5, 0, 0, 3, 0, 0],
            recon: vec![9, 0, 6, 0, 0, 3, 0, 0],
            cost: RdCost::new(100, 10, 1.0),
            num_nonzero: 3,
        };

        assert_eq!(result.num_nonzero, 3);
        let actual_nonzero = result.coeffs.iter().filter(|&&c| c != 0).count();
        assert_eq!(actual_nonzero, 3);
    }
}
