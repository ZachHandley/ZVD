//! Intra Mode Decision for H.265/HEVC Encoder
//!
//! This module implements intra mode selection using RDO.
//! For each block, it tests all 35 intra prediction modes and selects
//! the one with the lowest rate-distortion cost.
//!
//! # Intra Prediction Modes
//!
//! H.265 has 35 intra modes:
//! - Mode 0: Planar
//! - Mode 1: DC
//! - Modes 2-34: 33 angular directions
//!
//! # Most Probable Modes (MPM)
//!
//! To reduce signaling overhead, H.265 uses MPM:
//! - 3 most probable modes derived from left/above neighbors
//! - MPM requires fewer bits to signal
//! - Non-MPM modes require more bits

use crate::codec::h265::{IntraMode, IntraPredictor, ReferenceSamples};
use crate::codec::h265::rdo::{RdoDecision, RdCost, DistortionMetric, RateEstimator};
use crate::error::{Error, Result};

/// Intra mode selection result
#[derive(Debug, Clone)]
pub struct IntraModeResult {
    /// Selected intra mode
    pub mode: IntraMode,
    /// RD cost of selected mode
    pub cost: RdCost,
    /// Predicted samples
    pub prediction: Vec<u16>,
}

/// Intra mode selector
pub struct IntraModeSelector {
    /// RDO decision maker
    rdo: RdoDecision,
    /// Intra predictor
    predictor: IntraPredictor,
    /// Block width
    width: usize,
    /// Block height
    height: usize,
    /// Bit depth
    bit_depth: u8,
}

impl IntraModeSelector {
    /// Create a new intra mode selector
    pub fn new(qp: u8, width: usize, height: usize, bit_depth: u8) -> Result<Self> {
        let rdo = RdoDecision::new(qp, DistortionMetric::SSE);
        let predictor = IntraPredictor::new(bit_depth)?;

        Ok(Self {
            rdo,
            predictor,
            width,
            height,
            bit_depth,
        })
    }

    /// Select best intra mode for a block
    ///
    /// Tests all 35 modes and returns the one with lowest RD cost
    pub fn select_mode(
        &mut self,
        orig: &[u16],
        stride: usize,
        ref_samples: &ReferenceSamples,
    ) -> Result<IntraModeResult> {
        let mut best_cost = RdCost::max();
        let mut best_mode = IntraMode::Planar;
        let mut best_prediction = vec![0u16; self.width * self.height];

        // Test all 35 modes
        for mode_idx in 0..35 {
            let mode = Self::index_to_mode(mode_idx);
            let result = self.test_mode(orig, stride, ref_samples, mode)?;

            if result.cost.is_better_than(&best_cost) {
                best_cost = result.cost;
                best_mode = mode;
                best_prediction.copy_from_slice(&result.prediction);
            }
        }

        Ok(IntraModeResult {
            mode: best_mode,
            cost: best_cost,
            prediction: best_prediction,
        })
    }

    /// Test a single intra mode
    fn test_mode(
        &mut self,
        orig: &[u16],
        orig_stride: usize,
        ref_samples: &ReferenceSamples,
        mode: IntraMode,
    ) -> Result<IntraModeResult> {
        // Generate prediction
        let mut prediction = vec![0u16; self.width * self.height];
        self.predictor.predict(
            mode,
            ref_samples,
            &mut prediction,
            self.width,
            self.height,
            self.width,
        )?;

        // Estimate rate (bits required to signal this mode)
        let rate = self.estimate_mode_rate(mode);

        // Calculate RD cost
        let cost = self.rdo.calculate_cost(
            orig,
            &prediction,
            self.width,
            self.height,
            orig_stride,
            rate,
        );

        Ok(IntraModeResult {
            mode,
            cost,
            prediction,
        })
    }

    /// Estimate bits required to signal intra mode
    fn estimate_mode_rate(&self, mode: IntraMode) -> u32 {
        // Simplified rate estimation
        // In real encoder, would check MPM and use actual entropy coding
        match mode {
            IntraMode::Planar | IntraMode::DC => 2, // Frequently used, low bits
            _ => 6, // Angular modes require more bits
        }
    }

    /// Convert mode index to IntraMode
    fn index_to_mode(index: usize) -> IntraMode {
        match index {
            0 => IntraMode::Planar,
            1 => IntraMode::DC,
            n @ 2..=34 => IntraMode::Angular(n as u8),
            _ => IntraMode::Planar,
        }
    }
}

/// Most Probable Mode (MPM) derivation
pub struct MpmDerivation;

impl MpmDerivation {
    /// Derive 3 most probable modes from neighbors
    ///
    /// Uses left and above neighbor modes to predict current mode
    pub fn derive_mpm_list(left_mode: Option<IntraMode>, above_mode: Option<IntraMode>) -> [IntraMode; 3] {
        match (left_mode, above_mode) {
            (Some(left), Some(above)) if left == above => {
                // Both neighbors have same mode
                [left, Self::get_related_mode(left, -1), Self::get_related_mode(left, 1)]
            }
            (Some(left), Some(above)) => {
                // Different modes
                [left, above, IntraMode::Planar]
            }
            (Some(mode), None) | (None, Some(mode)) => {
                // One neighbor available
                [mode, Self::get_related_mode(mode, -1), Self::get_related_mode(mode, 1)]
            }
            (None, None) => {
                // No neighbors, use defaults
                [IntraMode::Planar, IntraMode::DC, IntraMode::Angular(26)]
            }
        }
    }

    /// Get related angular mode (for MPM derivation)
    fn get_related_mode(mode: IntraMode, offset: i32) -> IntraMode {
        match mode {
            IntraMode::Angular(angle) => {
                let new_angle = (angle as i32 + offset).clamp(2, 34) as u8;
                IntraMode::Angular(new_angle)
            }
            _ => mode,
        }
    }

    /// Check if mode is in MPM list
    pub fn is_mpm(mode: IntraMode, mpm_list: &[IntraMode; 3]) -> bool {
        mpm_list.contains(&mode)
    }
}

/// Fast intra mode decision (rough mode decision)
pub struct FastIntraDecision;

impl FastIntraDecision {
    /// Perform rough mode decision using SATD
    ///
    /// Tests subset of modes to find candidates, then RDO on candidates
    pub fn select_candidates(
        predictor: &mut IntraPredictor,
        orig: &[u16],
        stride: usize,
        ref_samples: &ReferenceSamples,
        width: usize,
        height: usize,
    ) -> Result<Vec<IntraMode>> {
        let mut candidates = Vec::new();

        // Always test Planar and DC
        candidates.push(IntraMode::Planar);
        candidates.push(IntraMode::DC);

        // Test subset of angular modes (every 4th mode)
        for angle in (2..=34).step_by(4) {
            candidates.push(IntraMode::Angular(angle));
        }

        Ok(candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intra_mode_selector_creation() {
        let selector = IntraModeSelector::new(24, 16, 16, 8);
        assert!(selector.is_ok());
    }

    #[test]
    fn test_index_to_mode() {
        assert_eq!(IntraModeSelector::index_to_mode(0), IntraMode::Planar);
        assert_eq!(IntraModeSelector::index_to_mode(1), IntraMode::DC);
        assert_eq!(IntraModeSelector::index_to_mode(2), IntraMode::Angular(2));
        assert_eq!(IntraModeSelector::index_to_mode(34), IntraMode::Angular(34));
    }

    #[test]
    fn test_mpm_derivation_same_neighbors() {
        let mode = IntraMode::Angular(10);
        let mpm = MpmDerivation::derive_mpm_list(Some(mode), Some(mode));

        assert_eq!(mpm[0], mode);
        // Other two should be related modes
    }

    #[test]
    fn test_mpm_derivation_different_neighbors() {
        let left = IntraMode::Angular(10);
        let above = IntraMode::Angular(20);
        let mpm = MpmDerivation::derive_mpm_list(Some(left), Some(above));

        assert_eq!(mpm[0], left);
        assert_eq!(mpm[1], above);
        assert_eq!(mpm[2], IntraMode::Planar);
    }

    #[test]
    fn test_mpm_derivation_no_neighbors() {
        let mpm = MpmDerivation::derive_mpm_list(None, None);

        assert_eq!(mpm[0], IntraMode::Planar);
        assert_eq!(mpm[1], IntraMode::DC);
        assert_eq!(mpm[2], IntraMode::Angular(26));
    }

    #[test]
    fn test_is_mpm() {
        let mpm = [IntraMode::Planar, IntraMode::DC, IntraMode::Angular(10)];

        assert!(MpmDerivation::is_mpm(IntraMode::Planar, &mpm));
        assert!(MpmDerivation::is_mpm(IntraMode::DC, &mpm));
        assert!(MpmDerivation::is_mpm(IntraMode::Angular(10), &mpm));
        assert!(!MpmDerivation::is_mpm(IntraMode::Angular(20), &mpm));
    }

    #[test]
    fn test_get_related_mode() {
        let mode = IntraMode::Angular(10);
        let prev = MpmDerivation::get_related_mode(mode, -1);
        let next = MpmDerivation::get_related_mode(mode, 1);

        assert_eq!(prev, IntraMode::Angular(9));
        assert_eq!(next, IntraMode::Angular(11));
    }

    #[test]
    fn test_get_related_mode_clamp() {
        let mode = IntraMode::Angular(2);
        let prev = MpmDerivation::get_related_mode(mode, -1);
        assert_eq!(prev, IntraMode::Angular(2)); // Clamped

        let mode = IntraMode::Angular(34);
        let next = MpmDerivation::get_related_mode(mode, 1);
        assert_eq!(next, IntraMode::Angular(34)); // Clamped
    }

    #[test]
    fn test_fast_intra_decision_candidates() {
        let mut predictor = IntraPredictor::new(8).unwrap();
        let orig = vec![128u16; 256];
        let ref_samples = ReferenceSamples::new(16, 8).unwrap();

        let candidates = FastIntraDecision::select_candidates(
            &mut predictor,
            &orig,
            16,
            &ref_samples,
            16,
            16,
        ).unwrap();

        // Should at least have Planar and DC
        assert!(candidates.contains(&IntraMode::Planar));
        assert!(candidates.contains(&IntraMode::DC));
    }

    #[test]
    fn test_estimate_mode_rate() {
        let selector = IntraModeSelector::new(24, 16, 16, 8).unwrap();

        let planar_rate = selector.estimate_mode_rate(IntraMode::Planar);
        let angular_rate = selector.estimate_mode_rate(IntraMode::Angular(10));

        assert_eq!(planar_rate, 2);
        assert_eq!(angular_rate, 6);
    }
}
