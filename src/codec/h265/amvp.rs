//! AMVP (Advanced Motion Vector Prediction) for H.265/HEVC
//!
//! This module implements motion vector prediction from spatial and temporal neighbors.
//! AMVP is one of the two inter prediction modes in H.265 (the other being merge mode).
//!
//! # AMVP Overview
//!
//! AMVP predicts motion vectors using:
//! - **Spatial candidates**: MVs from neighboring blocks (A0, A1, B0, B1, B2)
//! - **Temporal candidates**: MVs from co-located block in reference picture
//! - **MV scaling**: Adjusts MVs for different reference picture distances
//!
//! # Spatial Neighbors
//!
//! ```text
//!     B2   B1   B0
//!      +---+---+
//!      |       |
//!   A1 +   X   |
//!      |       |
//!   A0 +-------+
//! ```
//!
//! # Process
//!
//! 1. Derive spatial MV candidates (left and above)
//! 2. Derive temporal MV candidate (co-located picture)
//! 3. Scale MVs if referencing different pictures
//! 4. Remove duplicates
//! 5. Insert zero MV if needed (to ensure at least 2 candidates)

use crate::codec::h265::mv::{MotionVector, MvCandidate, PredictionList};
use crate::error::{Error, Result};

/// AMVP (Advanced Motion Vector Prediction) derivation engine
pub struct AmvpDerivation {
    /// Current PU position X
    pub_x: usize,
    /// Current PU position Y
    pub_y: usize,
    /// Current PU width
    pub_width: usize,
    /// Current PU height
    pub_height: usize,
    /// Current reference index
    ref_idx: u8,
    /// Prediction list (L0 or L1)
    pred_list: PredictionList,
}

impl AmvpDerivation {
    /// Create a new AMVP derivation context
    pub fn new(
        pub_x: usize,
        pub_y: usize,
        pub_width: usize,
        pub_height: usize,
        ref_idx: u8,
        pred_list: PredictionList,
    ) -> Self {
        Self {
            pub_x,
            pub_y,
            pub_width,
            pub_height,
            ref_idx,
            pred_list,
        }
    }

    /// Derive AMVP candidate list
    ///
    /// Returns up to 2 candidates for AMVP mode.
    pub fn derive_candidates(&self) -> Result<Vec<MotionVector>> {
        let mut candidates = Vec::new();

        // 1. Derive spatial candidates
        if let Some(mv_a) = self.derive_spatial_a()? {
            candidates.push(mv_a);
        }

        if let Some(mv_b) = self.derive_spatial_b()? {
            // Check for duplicates before adding
            if candidates.is_empty() || !self.is_duplicate(&mv_b, &candidates) {
                candidates.push(mv_b);
            }
        }

        // 2. If we have 2 candidates, we're done
        if candidates.len() >= 2 {
            return Ok(candidates);
        }

        // 3. Derive temporal candidate if needed
        if let Some(mv_t) = self.derive_temporal()? {
            if !self.is_duplicate(&mv_t, &candidates) {
                candidates.push(mv_t);
            }
        }

        // 4. If still less than 2 candidates, add zero MV
        while candidates.len() < 2 {
            candidates.push(MotionVector::zero());
        }

        Ok(candidates)
    }

    /// Derive spatial candidate from left neighbors (A0, A1)
    fn derive_spatial_a(&self) -> Result<Option<MotionVector>> {
        // Try A0 first (directly left, bottom position)
        if self.pub_x > 0 {
            let a0_x = self.pub_x - 1;
            let a0_y = self.pub_y + self.pub_height - 1;

            if let Some(mv) = self.get_neighbor_mv(a0_x, a0_y)? {
                return Ok(Some(mv));
            }
        }

        // Try A1 (left, above bottom position)
        if self.pub_x > 0 && self.pub_height > 4 {
            let a1_x = self.pub_x - 1;
            let a1_y = self.pub_y + self.pub_height - 4; // 4 pixels above bottom

            if let Some(mv) = self.get_neighbor_mv(a1_x, a1_y)? {
                return Ok(Some(mv));
            }
        }

        Ok(None)
    }

    /// Derive spatial candidate from above neighbors (B0, B1, B2)
    fn derive_spatial_b(&self) -> Result<Option<MotionVector>> {
        // Try B0 first (directly above, right position)
        if self.pub_y > 0 {
            let b0_x = self.pub_x + self.pub_width - 1;
            let b0_y = self.pub_y - 1;

            if let Some(mv) = self.get_neighbor_mv(b0_x, b0_y)? {
                return Ok(Some(mv));
            }
        }

        // Try B1 (above, left of right position)
        if self.pub_y > 0 && self.pub_width > 4 {
            let b1_x = self.pub_x + self.pub_width - 4; // 4 pixels left of right edge
            let b1_y = self.pub_y - 1;

            if let Some(mv) = self.get_neighbor_mv(b1_x, b1_y)? {
                return Ok(Some(mv));
            }
        }

        // Try B2 (above-left corner)
        if self.pub_x > 0 && self.pub_y > 0 {
            let b2_x = self.pub_x - 1;
            let b2_y = self.pub_y - 1;

            if let Some(mv) = self.get_neighbor_mv(b2_x, b2_y)? {
                return Ok(Some(mv));
            }
        }

        Ok(None)
    }

    /// Derive temporal candidate from co-located picture
    fn derive_temporal(&self) -> Result<Option<MotionVector>> {
        // Get co-located position (center-right of current PU)
        let col_x = self.pub_x + self.pub_width;
        let col_y = self.pub_y + (self.pub_height / 2);

        // In a real implementation, this would fetch from the temporal MV field
        // For now, we return None to indicate temporal MV not available
        // This would be implemented when we have DPB (Decoded Picture Buffer)
        Ok(None)
    }

    /// Get MV from neighbor position (stub for now)
    ///
    /// In full implementation, this would query the MotionVectorField
    fn get_neighbor_mv(&self, _x: usize, _y: usize) -> Result<Option<MotionVector>> {
        // Stub: In real implementation, would fetch from MotionVectorField
        // For now, return None to indicate no MV available
        Ok(None)
    }

    /// Check if MV is duplicate of any in candidate list
    fn is_duplicate(&self, mv: &MotionVector, candidates: &[MotionVector]) -> bool {
        candidates.iter().any(|c| c.x == mv.x && c.y == mv.y)
    }

    /// Scale motion vector for different reference picture distance
    ///
    /// When a neighbor's MV references a different picture, we scale it
    /// based on the temporal distance ratio.
    pub fn scale_mv(
        &self,
        mv: MotionVector,
        neighbor_ref_poc: i32,
        current_poc: i32,
        target_ref_poc: i32,
    ) -> MotionVector {
        // Calculate temporal distances
        let cur_diff = current_poc - target_ref_poc;
        let neighbor_diff = current_poc - neighbor_ref_poc;

        if neighbor_diff == 0 {
            return mv; // No scaling needed
        }

        // Scale MV proportionally
        // scaled_mv = mv * (cur_diff / neighbor_diff)
        let scale_factor = (cur_diff << 8) / neighbor_diff; // Fixed-point with 8-bit fractional
        mv.scale(scale_factor, 8)
    }
}

/// AMVP candidate list manager
pub struct AmvpCandidateList {
    /// List of MV predictors
    candidates: Vec<MotionVector>,
}

impl AmvpCandidateList {
    /// Create a new AMVP candidate list
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
        }
    }

    /// Add a candidate to the list
    pub fn add(&mut self, mv: MotionVector) {
        if self.candidates.len() < 2 {
            self.candidates.push(mv);
        }
    }

    /// Get candidate at index
    pub fn get(&self, index: usize) -> Result<MotionVector> {
        self.candidates
            .get(index)
            .copied()
            .ok_or_else(|| Error::InvalidData(format!("Invalid AMVP index: {}", index)))
    }

    /// Get number of candidates
    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    /// Check if list is empty
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    /// Check if list is full (2 candidates)
    pub fn is_full(&self) -> bool {
        self.candidates.len() >= 2
    }

    /// Clear the list
    pub fn clear(&mut self) {
        self.candidates.clear();
    }

    /// Build candidate list from derivation
    pub fn build(
        pub_x: usize,
        pub_y: usize,
        pub_width: usize,
        pub_height: usize,
        ref_idx: u8,
        pred_list: PredictionList,
    ) -> Result<Self> {
        let derivation = AmvpDerivation::new(pub_x, pub_y, pub_width, pub_height, ref_idx, pred_list);
        let candidates = derivation.derive_candidates()?;

        Ok(Self { candidates })
    }
}

impl Default for AmvpCandidateList {
    fn default() -> Self {
        Self::new()
    }
}

/// Spatial neighbor position helper
pub struct SpatialNeighborHelper;

impl SpatialNeighborHelper {
    /// Get A0 position (left, bottom)
    pub fn get_a0_position(pub_x: usize, pub_y: usize, pub_height: usize) -> Option<(usize, usize)> {
        if pub_x > 0 {
            Some((pub_x - 1, pub_y + pub_height - 1))
        } else {
            None
        }
    }

    /// Get A1 position (left, above bottom)
    pub fn get_a1_position(pub_x: usize, pub_y: usize, pub_height: usize) -> Option<(usize, usize)> {
        if pub_x > 0 && pub_height > 4 {
            Some((pub_x - 1, pub_y + pub_height - 4))
        } else {
            None
        }
    }

    /// Get B0 position (above, right)
    pub fn get_b0_position(pub_x: usize, pub_y: usize, pub_width: usize) -> Option<(usize, usize)> {
        if pub_y > 0 {
            Some((pub_x + pub_width - 1, pub_y - 1))
        } else {
            None
        }
    }

    /// Get B1 position (above, left of right)
    pub fn get_b1_position(pub_x: usize, pub_y: usize, pub_width: usize) -> Option<(usize, usize)> {
        if pub_y > 0 && pub_width > 4 {
            Some((pub_x + pub_width - 4, pub_y - 1))
        } else {
            None
        }
    }

    /// Get B2 position (above-left corner)
    pub fn get_b2_position(pub_x: usize, pub_y: usize) -> Option<(usize, usize)> {
        if pub_x > 0 && pub_y > 0 {
            Some((pub_x - 1, pub_y - 1))
        } else {
            None
        }
    }

    /// Get co-located temporal position
    pub fn get_temporal_position(
        pub_x: usize,
        pub_y: usize,
        pub_width: usize,
        pub_height: usize,
    ) -> (usize, usize) {
        // H.265 uses center-right position for temporal MV
        (pub_x + pub_width, pub_y + pub_height / 2)
    }
}

/// Temporal MV scaling calculator
pub struct TemporalScaler;

impl TemporalScaler {
    /// Calculate scaling factor for temporal MV
    ///
    /// Returns fixed-point scale factor with 8-bit fractional part
    pub fn calculate_scale_factor(
        current_poc: i32,
        target_ref_poc: i32,
        neighbor_ref_poc: i32,
    ) -> Option<i32> {
        let cur_diff = current_poc - target_ref_poc;
        let neighbor_diff = current_poc - neighbor_ref_poc;

        if neighbor_diff == 0 {
            return None; // Cannot scale, division by zero
        }

        // Fixed-point scale: (cur_diff / neighbor_diff) * 256
        Some((cur_diff << 8) / neighbor_diff)
    }

    /// Scale motion vector using precomputed scale factor
    pub fn scale_mv(mv: MotionVector, scale_factor: i32) -> MotionVector {
        mv.scale(scale_factor, 8)
    }

    /// Scale motion vector directly from POC values
    pub fn scale_mv_from_poc(
        mv: MotionVector,
        current_poc: i32,
        target_ref_poc: i32,
        neighbor_ref_poc: i32,
    ) -> MotionVector {
        if let Some(scale_factor) = Self::calculate_scale_factor(current_poc, target_ref_poc, neighbor_ref_poc) {
            Self::scale_mv(mv, scale_factor)
        } else {
            mv // Return unscaled if cannot scale
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amvp_derivation_creation() {
        let amvp = AmvpDerivation::new(32, 32, 16, 16, 0, PredictionList::L0);
        assert_eq!(amvp.pub_x, 32);
        assert_eq!(amvp.pub_y, 32);
        assert_eq!(amvp.pub_width, 16);
        assert_eq!(amvp.pub_height, 16);
    }

    #[test]
    fn test_amvp_derive_candidates_minimum() {
        let amvp = AmvpDerivation::new(32, 32, 16, 16, 0, PredictionList::L0);
        let candidates = amvp.derive_candidates().unwrap();

        // Should always have at least 2 candidates (with zero MVs if needed)
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn test_amvp_zero_mv_insertion() {
        let amvp = AmvpDerivation::new(0, 0, 16, 16, 0, PredictionList::L0);
        let candidates = amvp.derive_candidates().unwrap();

        // At position (0,0), no spatial neighbors available, should get zero MVs
        assert_eq!(candidates.len(), 2);
        assert!(candidates[0].is_zero());
        assert!(candidates[1].is_zero());
    }

    #[test]
    fn test_amvp_is_duplicate() {
        let amvp = AmvpDerivation::new(32, 32, 16, 16, 0, PredictionList::L0);
        let mv1 = MotionVector::new(16, 8);
        let mv2 = MotionVector::new(16, 8);
        let mv3 = MotionVector::new(20, 10);

        let candidates = vec![mv1];
        assert!(amvp.is_duplicate(&mv2, &candidates));
        assert!(!amvp.is_duplicate(&mv3, &candidates));
    }

    #[test]
    fn test_amvp_scale_mv_same_ref() {
        let amvp = AmvpDerivation::new(32, 32, 16, 16, 0, PredictionList::L0);
        let mv = MotionVector::new(16, 8);

        // Same reference, no scaling
        let scaled = amvp.scale_mv(mv, 10, 20, 10);
        assert_eq!(scaled, mv);
    }

    #[test]
    fn test_amvp_scale_mv_different_ref() {
        let amvp = AmvpDerivation::new(32, 32, 16, 16, 0, PredictionList::L0);
        let mv = MotionVector::new(16, 8);

        // Scale MV: current_poc=20, target_ref=10, neighbor_ref=5
        // cur_diff = 20-10 = 10, neighbor_diff = 20-5 = 15
        // scale = 10/15 = 2/3
        let scaled = amvp.scale_mv(mv, 5, 20, 10);

        // Should be scaled down
        assert!(scaled.x.abs() <= mv.x.abs());
        assert!(scaled.y.abs() <= mv.y.abs());
    }

    #[test]
    fn test_amvp_candidate_list() {
        let mut list = AmvpCandidateList::new();
        assert!(list.is_empty());
        assert!(!list.is_full());

        let mv1 = MotionVector::new(16, 8);
        list.add(mv1);
        assert_eq!(list.len(), 1);
        assert!(!list.is_full());

        let mv2 = MotionVector::new(20, 10);
        list.add(mv2);
        assert_eq!(list.len(), 2);
        assert!(list.is_full());

        // Adding more should not increase size
        let mv3 = MotionVector::new(24, 12);
        list.add(mv3);
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_amvp_candidate_list_get() {
        let mut list = AmvpCandidateList::new();
        let mv1 = MotionVector::new(16, 8);
        let mv2 = MotionVector::new(20, 10);

        list.add(mv1);
        list.add(mv2);

        assert_eq!(list.get(0).unwrap(), mv1);
        assert_eq!(list.get(1).unwrap(), mv2);
        assert!(list.get(2).is_err());
    }

    #[test]
    fn test_amvp_candidate_list_build() {
        let list = AmvpCandidateList::build(32, 32, 16, 16, 0, PredictionList::L0).unwrap();
        assert_eq!(list.len(), 2); // Always returns 2 candidates
    }

    #[test]
    fn test_spatial_neighbor_a0() {
        let pos = SpatialNeighborHelper::get_a0_position(32, 32, 16);
        assert_eq!(pos, Some((31, 47))); // (x-1, y+height-1)
    }

    #[test]
    fn test_spatial_neighbor_a0_edge() {
        let pos = SpatialNeighborHelper::get_a0_position(0, 32, 16);
        assert_eq!(pos, None); // At left edge, no left neighbor
    }

    #[test]
    fn test_spatial_neighbor_a1() {
        let pos = SpatialNeighborHelper::get_a1_position(32, 32, 16);
        assert_eq!(pos, Some((31, 44))); // (x-1, y+height-4)
    }

    #[test]
    fn test_spatial_neighbor_a1_small_block() {
        let pos = SpatialNeighborHelper::get_a1_position(32, 32, 4);
        assert_eq!(pos, None); // Block too small
    }

    #[test]
    fn test_spatial_neighbor_b0() {
        let pos = SpatialNeighborHelper::get_b0_position(32, 32, 16);
        assert_eq!(pos, Some((47, 31))); // (x+width-1, y-1)
    }

    #[test]
    fn test_spatial_neighbor_b0_edge() {
        let pos = SpatialNeighborHelper::get_b0_position(32, 0, 16);
        assert_eq!(pos, None); // At top edge
    }

    #[test]
    fn test_spatial_neighbor_b1() {
        let pos = SpatialNeighborHelper::get_b1_position(32, 32, 16);
        assert_eq!(pos, Some((44, 31))); // (x+width-4, y-1)
    }

    #[test]
    fn test_spatial_neighbor_b1_small_block() {
        let pos = SpatialNeighborHelper::get_b1_position(32, 32, 4);
        assert_eq!(pos, None);
    }

    #[test]
    fn test_spatial_neighbor_b2() {
        let pos = SpatialNeighborHelper::get_b2_position(32, 32);
        assert_eq!(pos, Some((31, 31))); // (x-1, y-1)
    }

    #[test]
    fn test_spatial_neighbor_b2_corner() {
        let pos = SpatialNeighborHelper::get_b2_position(0, 0);
        assert_eq!(pos, None);
    }

    #[test]
    fn test_temporal_position() {
        let pos = SpatialNeighborHelper::get_temporal_position(32, 32, 16, 16);
        assert_eq!(pos, (48, 40)); // (x+width, y+height/2)
    }

    #[test]
    fn test_temporal_scaler_same_ref() {
        let scale = TemporalScaler::calculate_scale_factor(20, 10, 10);
        assert_eq!(scale, None); // Same ref, no scaling
    }

    #[test]
    fn test_temporal_scaler_factor() {
        let scale = TemporalScaler::calculate_scale_factor(20, 10, 5);
        assert!(scale.is_some());

        // cur_diff = 20-10 = 10, neighbor_diff = 20-5 = 15
        // scale = (10 << 8) / 15 = 2560 / 15 ≈ 170
        let factor = scale.unwrap();
        assert!(factor > 0);
        assert!(factor < 256); // Should be less than 1.0 in fixed-point
    }

    #[test]
    fn test_temporal_scaler_scale_mv() {
        let mv = MotionVector::new(30, 20);
        let scale_factor = 128; // 0.5 in 8-bit fixed-point

        let scaled = TemporalScaler::scale_mv(mv, scale_factor);
        assert_eq!(scaled.x, 15); // 30 * 0.5
        assert_eq!(scaled.y, 10); // 20 * 0.5
    }

    #[test]
    fn test_temporal_scaler_scale_mv_from_poc() {
        let mv = MotionVector::new(20, 10);

        // POCs: current=20, target_ref=10, neighbor_ref=5
        // scale = 10/15 = 2/3
        let scaled = TemporalScaler::scale_mv_from_poc(mv, 20, 10, 5);

        // Should be scaled down
        assert!(scaled.x.abs() <= mv.x.abs());
        assert!(scaled.y.abs() <= mv.y.abs());
    }

    #[test]
    fn test_temporal_scaler_identity() {
        let mv = MotionVector::new(16, 8);
        let scale_factor = 256; // 1.0 in 8-bit fixed-point

        let scaled = TemporalScaler::scale_mv(mv, scale_factor);
        assert_eq!(scaled.x, 16);
        assert_eq!(scaled.y, 8);
    }

    #[test]
    fn test_temporal_scaler_double() {
        let mv = MotionVector::new(8, 4);
        let scale_factor = 512; // 2.0 in 8-bit fixed-point

        let scaled = TemporalScaler::scale_mv(mv, scale_factor);
        assert_eq!(scaled.x, 16);
        assert_eq!(scaled.y, 8);
    }

    #[test]
    fn test_amvp_candidate_list_clear() {
        let mut list = AmvpCandidateList::new();
        list.add(MotionVector::new(10, 5));
        list.add(MotionVector::new(20, 10));
        assert_eq!(list.len(), 2);

        list.clear();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn test_spatial_neighbor_all_positions() {
        // Test that all neighbor positions can be calculated for a valid block
        let x = 64;
        let y = 64;
        let width = 32;
        let height = 32;

        assert!(SpatialNeighborHelper::get_a0_position(x, y, height).is_some());
        assert!(SpatialNeighborHelper::get_a1_position(x, y, height).is_some());
        assert!(SpatialNeighborHelper::get_b0_position(x, y, width).is_some());
        assert!(SpatialNeighborHelper::get_b1_position(x, y, width).is_some());
        assert!(SpatialNeighborHelper::get_b2_position(x, y).is_some());
    }

    #[test]
    fn test_amvp_list_default() {
        let list = AmvpCandidateList::default();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }
}
