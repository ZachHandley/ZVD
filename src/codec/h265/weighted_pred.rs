//! Weighted Prediction for H.265/HEVC
//!
//! This module implements weighted prediction for inter-frame coding, which applies
//! weights and offsets to motion-compensated predictions to handle brightness changes.
//!
//! # Overview
//!
//! Weighted prediction is used to:
//! - Handle fade transitions (fade to black/white)
//! - Compensate for scene brightness changes
//! - Improve compression efficiency in varying lighting
//!
//! # Weighted Prediction Modes
//!
//! - **Explicit**: Encoder signals weights and offsets in slice header
//! - **Implicit**: Weights derived from temporal distances (B-frames)
//! - **Default**: No weighting (weight=1, offset=0)
//!
//! # Formula
//!
//! ```text
//! weighted_pred = Clip((pred * weight + (offset << shift)) >> shift)
//! ```

use crate::error::{Error, Result};

/// Weight parameters for a reference picture
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightParams {
    /// Weight value (signed)
    pub weight: i16,
    /// Offset value (signed)
    pub offset: i16,
    /// Bit shift for normalization
    pub shift: u8,
}

impl WeightParams {
    /// Create new weight parameters
    pub fn new(weight: i16, offset: i16, shift: u8) -> Self {
        Self {
            weight,
            offset,
            shift,
        }
    }

    /// Create default weight parameters (no weighting)
    pub fn default() -> Self {
        Self {
            weight: 1 << 6, // 64 in 6-bit precision (weight = 1.0)
            offset: 0,
            shift: 6,
        }
    }

    /// Check if this is default weighting (no effect)
    pub fn is_default(&self) -> bool {
        self.weight == (1 << self.shift) && self.offset == 0
    }
}

/// Weighted prediction engine
pub struct WeightedPredictor {
    /// Bit depth
    bit_depth: u8,
    /// Maximum sample value
    max_val: u16,
}

impl WeightedPredictor {
    /// Create a new weighted predictor
    pub fn new(bit_depth: u8) -> Result<Self> {
        if bit_depth != 8 && bit_depth != 10 && bit_depth != 12 {
            return Err(Error::InvalidData(format!(
                "Invalid bit depth: {}",
                bit_depth
            )));
        }

        let max_val = (1 << bit_depth) - 1;

        Ok(Self { bit_depth, max_val })
    }

    /// Apply weighted prediction to luma samples
    ///
    /// Formula: weighted = Clip((pred * weight + (offset << shift)) >> shift)
    pub fn apply_weights_luma(
        &self,
        pred: &[u16],
        dst: &mut [u16],
        params: WeightParams,
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<()> {
        if params.is_default() {
            // No weighting, just copy
            for y in 0..height {
                for x in 0..width {
                    let idx = y * stride + x;
                    dst[idx] = pred[idx];
                }
            }
            return Ok(());
        }

        let shift = params.shift;
        let weight = params.weight as i32;
        let offset = (params.offset as i32) << shift;

        for y in 0..height {
            for x in 0..width {
                let idx = y * stride + x;
                let pred_val = pred[idx] as i32;

                let weighted = (pred_val * weight + offset) >> shift;
                dst[idx] = weighted.clamp(0, self.max_val as i32) as u16;
            }
        }

        Ok(())
    }

    /// Apply bi-directional weighted prediction
    ///
    /// Combines L0 and L1 predictions with their respective weights
    pub fn apply_weights_bipred(
        &self,
        pred_l0: &[u16],
        pred_l1: &[u16],
        dst: &mut [u16],
        params_l0: WeightParams,
        params_l1: WeightParams,
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<()> {
        let shift = params_l0.shift.max(params_l1.shift);
        let weight_l0 = params_l0.weight as i32;
        let weight_l1 = params_l1.weight as i32;
        let offset = (params_l0.offset as i32 + params_l1.offset as i32) << shift;

        for y in 0..height {
            for x in 0..width {
                let idx = y * stride + x;
                let p0 = pred_l0[idx] as i32;
                let p1 = pred_l1[idx] as i32;

                let weighted = ((p0 * weight_l0 + p1 * weight_l1 + offset) >> (shift + 1))
                    .clamp(0, self.max_val as i32);
                dst[idx] = weighted as u16;
            }
        }

        Ok(())
    }

    /// Apply implicit weighted prediction (for B-frames)
    ///
    /// Weights derived from temporal distances
    pub fn apply_implicit_weights(
        &self,
        pred_l0: &[u16],
        pred_l1: &[u16],
        dst: &mut [u16],
        dist_l0: i32,
        dist_l1: i32,
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<()> {
        // Calculate implicit weights based on temporal distance
        let total_dist = dist_l0 + dist_l1;
        if total_dist == 0 {
            return Err(Error::InvalidData("Total distance is zero".to_string()));
        }

        let weight_l0 = ((dist_l1 << 8) / total_dist) as i32;
        let weight_l1 = ((dist_l0 << 8) / total_dist) as i32;

        for y in 0..height {
            for x in 0..width {
                let idx = y * stride + x;
                let p0 = pred_l0[idx] as i32;
                let p1 = pred_l1[idx] as i32;

                let weighted = ((p0 * weight_l0 + p1 * weight_l1 + 128) >> 8)
                    .clamp(0, self.max_val as i32);
                dst[idx] = weighted as u16;
            }
        }

        Ok(())
    }
}

/// Implicit weight calculator
pub struct ImplicitWeightCalc;

impl ImplicitWeightCalc {
    /// Calculate implicit weights from temporal distances
    ///
    /// Returns (weight_l0, weight_l1) in 8-bit fixed-point
    pub fn calculate(dist_l0: i32, dist_l1: i32) -> Result<(i32, i32)> {
        let total = dist_l0 + dist_l1;
        if total == 0 {
            return Err(Error::InvalidData("Total distance is zero".to_string()));
        }

        // Fixed-point with 8-bit fractional
        let weight_l0 = (dist_l1 << 8) / total;
        let weight_l1 = (dist_l0 << 8) / total;

        Ok((weight_l0, weight_l1))
    }

    /// Check if implicit weighting should be used
    pub fn should_use_implicit(dist_l0: i32, dist_l1: i32) -> bool {
        dist_l0 > 0 && dist_l1 > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_params_creation() {
        let params = WeightParams::new(64, 0, 6);
        assert_eq!(params.weight, 64);
        assert_eq!(params.offset, 0);
        assert_eq!(params.shift, 6);
    }

    #[test]
    fn test_weight_params_default() {
        let params = WeightParams::default();
        assert!(params.is_default());
        assert_eq!(params.weight, 64);
        assert_eq!(params.offset, 0);
    }

    #[test]
    fn test_weighted_predictor_creation() {
        let wp = WeightedPredictor::new(8);
        assert!(wp.is_ok());
    }

    #[test]
    fn test_weighted_predictor_invalid_bit_depth() {
        let wp = WeightedPredictor::new(9);
        assert!(wp.is_err());
    }

    #[test]
    fn test_apply_weights_default() {
        let wp = WeightedPredictor::new(8).unwrap();
        let pred = vec![100u16; 64];
        let mut dst = vec![0u16; 64];
        let params = WeightParams::default();

        wp.apply_weights_luma(&pred, &mut dst, params, 8, 8, 8)
            .unwrap();

        assert_eq!(dst, pred);
    }

    #[test]
    fn test_apply_weights_half() {
        let wp = WeightedPredictor::new(8).unwrap();
        let pred = vec![100u16; 64];
        let mut dst = vec![0u16; 64];
        let params = WeightParams::new(32, 0, 6); // 0.5 weight

        wp.apply_weights_luma(&pred, &mut dst, params, 8, 8, 8)
            .unwrap();

        assert!(dst[0] <= 50);
    }

    #[test]
    fn test_apply_weights_clipping() {
        let wp = WeightedPredictor::new(8).unwrap();
        let pred = vec![250u16; 64];
        let mut dst = vec![0u16; 64];
        let params = WeightParams::new(128, 0, 6); // 2.0 weight

        wp.apply_weights_luma(&pred, &mut dst, params, 8, 8, 8)
            .unwrap();

        assert_eq!(dst[0], 255);
    }

    #[test]
    fn test_apply_weights_bipred() {
        let wp = WeightedPredictor::new(8).unwrap();
        let pred_l0 = vec![100u16; 64];
        let pred_l1 = vec![120u16; 64];
        let mut dst = vec![0u16; 64];
        let params_l0 = WeightParams::new(32, 0, 6);
        let params_l1 = WeightParams::new(32, 0, 6);

        wp.apply_weights_bipred(&pred_l0, &pred_l1, &mut dst, params_l0, params_l1, 8, 8, 8)
            .unwrap();

        assert!(dst[0] >= 105 && dst[0] <= 115);
    }

    #[test]
    fn test_apply_implicit_weights() {
        let wp = WeightedPredictor::new(8).unwrap();
        let pred_l0 = vec![100u16; 64];
        let pred_l1 = vec![140u16; 64];
        let mut dst = vec![0u16; 64];

        wp.apply_implicit_weights(&pred_l0, &pred_l1, &mut dst, 1, 1, 8, 8, 8)
            .unwrap();

        assert!(dst[0] >= 115 && dst[0] <= 125);
    }

    #[test]
    fn test_implicit_weight_calc() {
        let (w0, w1) = ImplicitWeightCalc::calculate(1, 1).unwrap();
        assert_eq!(w0, 128);
        assert_eq!(w1, 128);
    }

    #[test]
    fn test_implicit_weight_calc_zero_distance() {
        let result = ImplicitWeightCalc::calculate(0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_implicit_weight_should_use() {
        assert!(ImplicitWeightCalc::should_use_implicit(1, 1));
        assert!(!ImplicitWeightCalc::should_use_implicit(0, 1));
    }
}
