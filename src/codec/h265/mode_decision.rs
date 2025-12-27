//! Mode Decision for H.265/HEVC Encoder
//!
//! This module implements the high-level mode decision process that chooses
//! between different coding modes to minimize rate-distortion cost.
//!
//! # Mode Decision Process
//!
//! For each CU, the encoder tests:
//! 1. **Skip Mode**: Copy from reference (0 bits for residual)
//! 2. **Inter Mode**: Motion estimation + residual
//! 3. **Intra Mode**: Spatial prediction + residual
//!
//! And selects the mode with lowest RD cost.
//!
//! # Decision Hierarchy
//!
//! ```text
//! CU (64×64)
//!   ├─ Skip?
//!   ├─ Inter? (test merge, test AMVP)
//!   ├─ Intra? (test all 35 modes)
//!   └─ Split to smaller CUs? (recursive)
//! ```

use crate::codec::h265::{IntraMode, PredMode};
use crate::codec::h265::rdo::{RdCost, RdoDecision, DistortionMetric, RateEstimator};
use crate::codec::h265::encoder_intra::{IntraModeSelector, IntraModeResult};
use crate::codec::h265::me::{MotionEstimator, MeResult, SearchRange};
use crate::codec::h265::mv::MotionVector;
use crate::error::{Error, Result};

/// Coding mode choice
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodingMode {
    /// Skip mode (no residual)
    Skip,
    /// Inter mode with motion vector
    Inter,
    /// Intra mode
    Intra,
}

/// Mode decision result
#[derive(Debug, Clone)]
pub struct ModeDecisionResult {
    /// Selected coding mode
    pub mode: CodingMode,
    /// Pred mode (Intra/Inter)
    pub pred_mode: PredMode,
    /// Intra mode (if intra)
    pub intra_mode: Option<IntraMode>,
    /// Motion vector (if inter)
    pub mv: Option<MotionVector>,
    /// MV predictor (if inter)
    pub mvp: Option<MotionVector>,
    /// RD cost of selected mode
    pub cost: RdCost,
    /// Prediction samples
    pub prediction: Vec<u16>,
}

/// Mode decision engine
pub struct ModeDecision {
    /// RDO decision maker
    rdo: RdoDecision,
    /// Block width
    width: usize,
    /// Block height
    height: usize,
    /// QP
    qp: u8,
}

impl ModeDecision {
    /// Create a new mode decision engine
    pub fn new(qp: u8, width: usize, height: usize) -> Self {
        let rdo = RdoDecision::new(qp, DistortionMetric::SSE);

        Self {
            rdo,
            width,
            height,
            qp,
        }
    }

    /// Decide best coding mode for a CU
    ///
    /// Tests skip, inter, and intra modes
    pub fn decide_mode(
        &mut self,
        orig: &[u16],
        orig_stride: usize,
        // In real implementation would need reference frames, neighbors, etc.
    ) -> Result<ModeDecisionResult> {
        let mut best_cost = RdCost::max();
        let mut best_result = None;

        // 1. Test Skip mode (if reference available)
        // Skipped for now - needs reference frame

        // 2. Test Inter mode (if reference available)
        // Skipped for now - needs reference frame

        // 3. Test Intra mode (always available)
        let intra_result = self.test_intra_mode(orig, orig_stride)?;

        if intra_result.cost.is_better_than(&best_cost) {
            best_cost = intra_result.cost;
            best_result = Some(intra_result);
        }

        // Return best mode
        best_result.ok_or_else(|| Error::Codec("No valid mode found".to_string()))
    }

    /// Test intra coding mode
    fn test_intra_mode(
        &mut self,
        orig: &[u16],
        orig_stride: usize,
    ) -> Result<ModeDecisionResult> {
        // For now, just test Planar mode as placeholder
        // Real implementation would use IntraModeSelector

        let prediction = vec![128u16; self.width * self.height]; // DC-like prediction
        let rate = RateEstimator::estimate_mode_bits(true, false);

        let cost = self.rdo.calculate_cost(
            orig,
            &prediction,
            self.width,
            self.height,
            orig_stride,
            rate,
        );

        Ok(ModeDecisionResult {
            mode: CodingMode::Intra,
            pred_mode: PredMode::Intra,
            intra_mode: Some(IntraMode::Planar),
            mv: None,
            mvp: None,
            cost,
            prediction,
        })
    }

    /// Test skip mode
    fn test_skip_mode(
        &self,
        orig: &[u16],
        orig_stride: usize,
        prediction: &[u16],
    ) -> Result<ModeDecisionResult> {
        // Skip mode: 0 residual, minimal bits
        let rate = RateEstimator::estimate_mode_bits(false, true);

        let cost = self.rdo.calculate_cost(
            orig,
            prediction,
            self.width,
            self.height,
            orig_stride,
            rate,
        );

        Ok(ModeDecisionResult {
            mode: CodingMode::Skip,
            pred_mode: PredMode::Inter,
            intra_mode: None,
            mv: Some(MotionVector::zero()),
            mvp: Some(MotionVector::zero()),
            cost,
            prediction: prediction.to_vec(),
        })
    }

    /// Test inter mode with motion estimation
    fn test_inter_mode(
        &self,
        orig: &[u16],
        orig_stride: usize,
        me_result: MeResult,
        prediction: &[u16],
    ) -> Result<ModeDecisionResult> {
        // Inter mode: ME cost + residual bits
        let mode_bits = RateEstimator::estimate_mode_bits(false, false);
        let mvd_bits = RateEstimator::estimate_mvd_bits(
            me_result.mv.x - me_result.mvp.x,
            me_result.mv.y - me_result.mvp.y,
        );

        // Estimate residual (would actually compute transform + quant)
        let residual_bits = 20; // Placeholder

        let total_rate = mode_bits + mvd_bits + residual_bits;

        let cost = self.rdo.calculate_cost(
            orig,
            prediction,
            self.width,
            self.height,
            orig_stride,
            total_rate,
        );

        Ok(ModeDecisionResult {
            mode: CodingMode::Inter,
            pred_mode: PredMode::Inter,
            intra_mode: None,
            mv: Some(me_result.mv),
            mvp: Some(me_result.mvp),
            cost,
            prediction: prediction.to_vec(),
        })
    }
}

/// CU split decision
pub struct SplitDecision {
    /// RDO decision maker
    rdo: RdoDecision,
}

impl SplitDecision {
    /// Create new split decision engine
    pub fn new(qp: u8) -> Self {
        let rdo = RdoDecision::new(qp, DistortionMetric::SSE);
        Self { rdo }
    }

    /// Decide whether to split CU
    ///
    /// Compares cost of coding as single CU vs 4 smaller CUs
    pub fn should_split(&self, current_cost: RdCost, split_cost: RdCost) -> bool {
        split_cost.is_better_than(&current_cost)
    }

    /// Calculate split overhead (bits to signal split)
    pub fn split_overhead(&self) -> u32 {
        1 // 1 bit to signal split
    }
}

/// Early termination decisions
pub struct EarlyTermination;

impl EarlyTermination {
    /// Check if skip mode is good enough (early termination)
    pub fn is_skip_sufficient(skip_cost: RdCost, threshold: u64) -> bool {
        skip_cost.distortion < threshold
    }

    /// Check if we can skip testing larger CU sizes
    pub fn can_skip_larger_sizes(current_cost: RdCost, parent_cost: RdCost) -> bool {
        // If current size is much better, don't test larger
        current_cost.cost < parent_cost.cost / 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mode_decision_creation() {
        let md = ModeDecision::new(24, 16, 16);
        assert_eq!(md.width, 16);
        assert_eq!(md.height, 16);
        assert_eq!(md.qp, 24);
    }

    #[test]
    fn test_decide_mode_intra() {
        let mut md = ModeDecision::new(24, 16, 16);
        let orig = vec![128u16; 256];

        let result = md.decide_mode(&orig, 16);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.mode, CodingMode::Intra);
        assert_eq!(result.pred_mode, PredMode::Intra);
        assert!(result.intra_mode.is_some());
    }

    #[test]
    fn test_coding_mode_enum() {
        assert_eq!(CodingMode::Skip, CodingMode::Skip);
        assert_ne!(CodingMode::Skip, CodingMode::Inter);
    }

    #[test]
    fn test_split_decision_creation() {
        let split = SplitDecision::new(24);
        assert_eq!(split.split_overhead(), 1);
    }

    #[test]
    fn test_should_split() {
        let split = SplitDecision::new(24);
        let current = RdCost::new(1000, 50, 1.0);
        let split_cost = RdCost::new(800, 52, 1.0);

        assert!(split.should_split(current, split_cost));
    }

    #[test]
    fn test_should_not_split() {
        let split = SplitDecision::new(24);
        let current = RdCost::new(800, 50, 1.0);
        let split_cost = RdCost::new(1000, 52, 1.0);

        assert!(!split.should_split(current, split_cost));
    }

    #[test]
    fn test_is_skip_sufficient() {
        let skip_cost = RdCost::new(100, 1, 1.0);
        assert!(EarlyTermination::is_skip_sufficient(skip_cost, 200));
        assert!(!EarlyTermination::is_skip_sufficient(skip_cost, 50));
    }

    #[test]
    fn test_can_skip_larger_sizes() {
        let current = RdCost::new(500, 50, 1.0);
        let parent = RdCost::new(2000, 60, 1.0);

        assert!(EarlyTermination::can_skip_larger_sizes(current, parent));
    }

    #[test]
    fn test_mode_decision_result_creation() {
        let result = ModeDecisionResult {
            mode: CodingMode::Intra,
            pred_mode: PredMode::Intra,
            intra_mode: Some(IntraMode::Planar),
            mv: None,
            mvp: None,
            cost: RdCost::new(1000, 50, 1.0),
            prediction: vec![128; 256],
        };

        assert_eq!(result.mode, CodingMode::Intra);
        assert_eq!(result.intra_mode, Some(IntraMode::Planar));
        assert!(result.mv.is_none());
    }

    #[test]
    fn test_test_skip_mode() {
        let md = ModeDecision::new(24, 16, 16);
        let orig = vec![128u16; 256];
        let prediction = vec![128u16; 256];

        let result = md.test_skip_mode(&orig, 16, &prediction);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.mode, CodingMode::Skip);
        assert_eq!(result.cost.distortion, 0); // Perfect match
    }

    #[test]
    fn test_test_inter_mode() {
        let md = ModeDecision::new(24, 16, 16);
        let orig = vec![128u16; 256];
        let prediction = vec![130u16; 256];

        let me_result = MeResult {
            mv: MotionVector::new(16, 8),
            mvp: MotionVector::new(12, 6),
            cost: 1000,
            sad: 900,
        };

        let result = md.test_inter_mode(&orig, 16, me_result, &prediction);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.mode, CodingMode::Inter);
        assert!(result.mv.is_some());
    }
}
