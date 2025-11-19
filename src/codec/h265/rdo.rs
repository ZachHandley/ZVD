//! Rate-Distortion Optimization (RDO) for H.265/HEVC Encoder
//!
//! This module implements the core RDO framework that drives encoding decisions.
//! RDO is the heart of modern video encoders - it balances quality (distortion)
//! against bitrate (rate) to achieve optimal compression.
//!
//! # Rate-Distortion Theory
//!
//! The encoder minimizes: **Cost = Distortion + λ × Rate**
//!
//! - **Distortion**: Difference between original and reconstructed (SSE, SAD, SATD)
//! - **Rate**: Number of bits required to encode
//! - **λ (Lambda)**: Lagrange multiplier derived from QP
//!
//! # RDO Process
//!
//! For each block, the encoder:
//! 1. Tests multiple coding modes (intra/inter, block sizes, etc.)
//! 2. Calculates distortion for each mode
//! 3. Estimates rate (bits) for each mode
//! 4. Computes RD cost: D + λR
//! 5. Selects mode with minimum cost

use crate::error::{Error, Result};

/// Distortion metric type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistortionMetric {
    /// Sum of Absolute Differences
    SAD,
    /// Sum of Squared Errors
    SSE,
    /// Sum of Absolute Transformed Differences (Hadamard)
    SATD,
}

/// Rate-distortion cost
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RdCost {
    /// Distortion value
    pub distortion: u64,
    /// Rate (bits) estimate
    pub rate: u32,
    /// Total RD cost (distortion + lambda * rate)
    pub cost: u64,
}

impl RdCost {
    /// Create a new RD cost
    pub fn new(distortion: u64, rate: u32, lambda: f64) -> Self {
        let cost = distortion + (lambda * rate as f64) as u64;
        Self {
            distortion,
            rate,
            cost,
        }
    }

    /// Create maximum cost (for initialization)
    pub fn max() -> Self {
        Self {
            distortion: u64::MAX,
            rate: u32::MAX,
            cost: u64::MAX,
        }
    }

    /// Check if this cost is better (lower) than another
    pub fn is_better_than(&self, other: &RdCost) -> bool {
        self.cost < other.cost
    }
}

/// Distortion calculator
pub struct DistortionCalc;

impl DistortionCalc {
    /// Calculate Sum of Absolute Differences (SAD)
    ///
    /// Fast distortion metric, good for motion estimation
    pub fn calculate_sad(
        orig: &[u16],
        recon: &[u16],
        width: usize,
        height: usize,
        orig_stride: usize,
        recon_stride: usize,
    ) -> u64 {
        let mut sad = 0u64;

        for y in 0..height {
            for x in 0..width {
                let orig_idx = y * orig_stride + x;
                let recon_idx = y * recon_stride + x;
                sad += (orig[orig_idx] as i32 - recon[recon_idx] as i32).abs() as u64;
            }
        }

        sad
    }

    /// Calculate Sum of Squared Errors (SSE)
    ///
    /// Standard distortion metric, correlates well with PSNR
    pub fn calculate_sse(
        orig: &[u16],
        recon: &[u16],
        width: usize,
        height: usize,
        orig_stride: usize,
        recon_stride: usize,
    ) -> u64 {
        let mut sse = 0u64;

        for y in 0..height {
            for x in 0..width {
                let orig_idx = y * orig_stride + x;
                let recon_idx = y * recon_stride + x;
                let diff = orig[orig_idx] as i32 - recon[recon_idx] as i32;
                sse += (diff * diff) as u64;
            }
        }

        sse
    }

    /// Calculate Sum of Absolute Transformed Differences (SATD)
    ///
    /// Uses Hadamard transform, better for texture/detail
    pub fn calculate_satd_4x4(orig: &[u16], recon: &[u16], stride: usize) -> u64 {
        let mut diff = [0i32; 16];

        // Calculate differences
        for y in 0..4 {
            for x in 0..4 {
                let idx = y * stride + x;
                diff[y * 4 + x] = orig[idx] as i32 - recon[idx] as i32;
            }
        }

        // Hadamard transform
        Self::hadamard_4x4(&mut diff);

        // Sum absolute values
        diff.iter().map(|&d| d.abs() as u64).sum()
    }

    /// 4×4 Hadamard transform (in-place)
    fn hadamard_4x4(block: &mut [i32; 16]) {
        // Horizontal
        for y in 0..4 {
            let i = y * 4;
            let a0 = block[i] + block[i + 2];
            let a1 = block[i + 1] + block[i + 3];
            let a2 = block[i] - block[i + 2];
            let a3 = block[i + 1] - block[i + 3];

            block[i] = a0 + a1;
            block[i + 1] = a2 + a3;
            block[i + 2] = a0 - a1;
            block[i + 3] = a2 - a3;
        }

        // Vertical
        for x in 0..4 {
            let a0 = block[x] + block[x + 8];
            let a1 = block[x + 4] + block[x + 12];
            let a2 = block[x] - block[x + 8];
            let a3 = block[x + 4] - block[x + 12];

            block[x] = a0 + a1;
            block[x + 4] = a2 + a3;
            block[x + 8] = a0 - a1;
            block[x + 12] = a2 - a3;
        }
    }
}

/// Lambda calculator for RDO
pub struct LambdaCalc;

impl LambdaCalc {
    /// Calculate lambda from QP
    ///
    /// Based on: λ = 0.85 × 2^((QP-12)/3)
    pub fn calculate_lambda(qp: u8) -> f64 {
        let qp_clamped = qp.min(51) as f64;
        0.85 * 2.0_f64.powf((qp_clamped - 12.0) / 3.0)
    }

    /// Calculate lambda for motion estimation (typically sqrt of RDO lambda)
    pub fn calculate_lambda_me(qp: u8) -> f64 {
        Self::calculate_lambda(qp).sqrt()
    }

    /// Calculate lambda for mode decision
    pub fn calculate_lambda_mode(qp: u8) -> f64 {
        Self::calculate_lambda(qp)
    }
}

/// Rate estimator
pub struct RateEstimator;

impl RateEstimator {
    /// Estimate bits for MVD (Motion Vector Difference)
    ///
    /// Rough estimate based on MVD magnitude
    pub fn estimate_mvd_bits(mvd_x: i16, mvd_y: i16) -> u32 {
        let abs_x = mvd_x.abs() as u32;
        let abs_y = mvd_y.abs() as u32;

        // Exponential Golomb coding estimate
        let bits_x = if abs_x == 0 {
            1
        } else {
            2 * (32 - abs_x.leading_zeros()) + 1
        };

        let bits_y = if abs_y == 0 {
            1
        } else {
            2 * (32 - abs_y.leading_zeros()) + 1
        };

        bits_x + bits_y
    }

    /// Estimate bits for residual coefficients
    ///
    /// Based on number of non-zero coefficients
    pub fn estimate_residual_bits(coeffs: &[i16]) -> u32 {
        let num_nonzero = coeffs.iter().filter(|&&c| c != 0).count();

        // Rough estimate: 2 bits per non-zero coeff + overhead
        (num_nonzero * 2 + 4) as u32
    }

    /// Estimate bits for mode selection
    pub fn estimate_mode_bits(is_intra: bool, is_skip: bool) -> u32 {
        if is_skip {
            1 // Skip mode: 1 bit
        } else if is_intra {
            8 // Intra mode: ~8 bits average
        } else {
            6 // Inter mode: ~6 bits average
        }
    }
}

/// RDO decision maker
pub struct RdoDecision {
    /// Quantization parameter
    qp: u8,
    /// Lambda for cost calculation
    lambda: f64,
    /// Distortion metric to use
    metric: DistortionMetric,
}

impl RdoDecision {
    /// Create a new RDO decision maker
    pub fn new(qp: u8, metric: DistortionMetric) -> Self {
        let lambda = LambdaCalc::calculate_lambda(qp);
        Self { qp, lambda, metric }
    }

    /// Calculate RD cost for a coding mode
    pub fn calculate_cost(
        &self,
        orig: &[u16],
        recon: &[u16],
        width: usize,
        height: usize,
        stride: usize,
        rate: u32,
    ) -> RdCost {
        let distortion = match self.metric {
            DistortionMetric::SAD => {
                DistortionCalc::calculate_sad(orig, recon, width, height, stride, stride)
            }
            DistortionMetric::SSE => {
                DistortionCalc::calculate_sse(orig, recon, width, height, stride, stride)
            }
            DistortionMetric::SATD => {
                // For SATD, we'd need to process in 4×4 blocks
                DistortionCalc::calculate_sse(orig, recon, width, height, stride, stride)
            }
        };

        RdCost::new(distortion, rate, self.lambda)
    }

    /// Get current lambda
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Get current QP
    pub fn qp(&self) -> u8 {
        self.qp
    }
}

/// PSNR calculator (for quality measurement)
pub struct PsnrCalc;

impl PsnrCalc {
    /// Calculate PSNR from SSE
    ///
    /// PSNR = 10 × log10(MAX² / MSE)
    pub fn calculate_psnr(sse: u64, num_pixels: usize, bit_depth: u8) -> f64 {
        if sse == 0 {
            return f64::INFINITY;
        }

        let max_val = (1u64 << bit_depth) - 1;
        let mse = sse as f64 / num_pixels as f64;
        let psnr = 10.0 * ((max_val * max_val) as f64 / mse).log10();

        psnr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rd_cost_creation() {
        let cost = RdCost::new(100, 50, 1.0);
        assert_eq!(cost.distortion, 100);
        assert_eq!(cost.rate, 50);
        assert_eq!(cost.cost, 150); // 100 + 1.0 * 50
    }

    #[test]
    fn test_rd_cost_comparison() {
        let cost1 = RdCost::new(100, 50, 1.0);
        let cost2 = RdCost::new(200, 30, 1.0);

        assert!(cost1.is_better_than(&cost2));
        assert!(!cost2.is_better_than(&cost1));
    }

    #[test]
    fn test_rd_cost_max() {
        let max_cost = RdCost::max();
        let normal_cost = RdCost::new(100, 50, 1.0);

        assert!(normal_cost.is_better_than(&max_cost));
    }

    #[test]
    fn test_calculate_sad_identical() {
        let orig = vec![100u16; 64];
        let recon = vec![100u16; 64];

        let sad = DistortionCalc::calculate_sad(&orig, &recon, 8, 8, 8, 8);
        assert_eq!(sad, 0);
    }

    #[test]
    fn test_calculate_sad() {
        let orig = vec![100u16; 64];
        let recon = vec![105u16; 64];

        let sad = DistortionCalc::calculate_sad(&orig, &recon, 8, 8, 8, 8);
        assert_eq!(sad, 5 * 64); // 5 difference per pixel
    }

    #[test]
    fn test_calculate_sse_identical() {
        let orig = vec![100u16; 64];
        let recon = vec![100u16; 64];

        let sse = DistortionCalc::calculate_sse(&orig, &recon, 8, 8, 8, 8);
        assert_eq!(sse, 0);
    }

    #[test]
    fn test_calculate_sse() {
        let orig = vec![100u16; 64];
        let recon = vec![105u16; 64];

        let sse = DistortionCalc::calculate_sse(&orig, &recon, 8, 8, 8, 8);
        assert_eq!(sse, 25 * 64); // 5² per pixel
    }

    #[test]
    fn test_calculate_satd_4x4_identical() {
        let orig = vec![100u16; 16];
        let recon = vec![100u16; 16];

        let satd = DistortionCalc::calculate_satd_4x4(&orig, &recon, 4);
        assert_eq!(satd, 0);
    }

    #[test]
    fn test_lambda_calculation() {
        let lambda_qp0 = LambdaCalc::calculate_lambda(0);
        let lambda_qp51 = LambdaCalc::calculate_lambda(51);

        assert!(lambda_qp0 > 0.0);
        assert!(lambda_qp51 > lambda_qp0); // Higher QP → higher lambda
    }

    #[test]
    fn test_lambda_me() {
        let lambda = LambdaCalc::calculate_lambda(24);
        let lambda_me = LambdaCalc::calculate_lambda_me(24);

        // ME lambda should be sqrt of RDO lambda
        assert!((lambda_me * lambda_me - lambda).abs() < 0.01);
    }

    #[test]
    fn test_estimate_mvd_bits_zero() {
        let bits = RateEstimator::estimate_mvd_bits(0, 0);
        assert_eq!(bits, 2); // 1 bit per component
    }

    #[test]
    fn test_estimate_mvd_bits() {
        let bits_small = RateEstimator::estimate_mvd_bits(1, 1);
        let bits_large = RateEstimator::estimate_mvd_bits(100, 100);

        assert!(bits_large > bits_small);
    }

    #[test]
    fn test_estimate_residual_bits_all_zero() {
        let coeffs = vec![0i16; 64];
        let bits = RateEstimator::estimate_residual_bits(&coeffs);
        assert_eq!(bits, 4); // Just overhead
    }

    #[test]
    fn test_estimate_residual_bits() {
        let mut coeffs = vec![0i16; 64];
        coeffs[0] = 10;
        coeffs[1] = 20;

        let bits = RateEstimator::estimate_residual_bits(&coeffs);
        assert_eq!(bits, 2 * 2 + 4); // 2 non-zero * 2 bits + overhead
    }

    #[test]
    fn test_estimate_mode_bits() {
        let skip_bits = RateEstimator::estimate_mode_bits(false, true);
        let intra_bits = RateEstimator::estimate_mode_bits(true, false);
        let inter_bits = RateEstimator::estimate_mode_bits(false, false);

        assert_eq!(skip_bits, 1);
        assert_eq!(intra_bits, 8);
        assert_eq!(inter_bits, 6);
    }

    #[test]
    fn test_rdo_decision_creation() {
        let rdo = RdoDecision::new(24, DistortionMetric::SSE);
        assert_eq!(rdo.qp(), 24);
        assert!(rdo.lambda() > 0.0);
    }

    #[test]
    fn test_rdo_decision_calculate_cost() {
        let rdo = RdoDecision::new(24, DistortionMetric::SAD);
        let orig = vec![100u16; 64];
        let recon = vec![105u16; 64];

        let cost = rdo.calculate_cost(&orig, &recon, 8, 8, 8, 10);

        assert_eq!(cost.distortion, 5 * 64);
        assert_eq!(cost.rate, 10);
        assert!(cost.cost > cost.distortion); // Should include rate cost
    }

    #[test]
    fn test_psnr_calculation() {
        let sse = 100;
        let num_pixels = 64;
        let psnr = PsnrCalc::calculate_psnr(sse, num_pixels, 8);

        assert!(psnr > 0.0);
        assert!(psnr < 100.0); // Reasonable PSNR range
    }

    #[test]
    fn test_psnr_zero_sse() {
        let psnr = PsnrCalc::calculate_psnr(0, 64, 8);
        assert!(psnr.is_infinite());
    }

    #[test]
    fn test_hadamard_4x4_dc_only() {
        let mut block = [4i32; 16]; // DC only
        DistortionCalc::hadamard_4x4(&mut block);

        // DC coefficient should be sum of all
        assert_eq!(block[0], 64); // 16 * 4
        // AC coefficients should be 0
        assert!(block[1..].iter().all(|&x| x == 0));
    }
}
