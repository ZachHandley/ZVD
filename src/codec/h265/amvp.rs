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

use crate::codec::h265::mv::{MotionVector, MvCandidate, MotionVectorField, PredictionList};
use crate::error::{Error, Result};

/// AMVP (Advanced Motion Vector Prediction) derivation engine
///
/// This struct holds context for deriving AMVP candidates for a prediction unit.
/// It requires access to the motion vector field of the current picture to fetch
/// neighbor MVs.
pub struct AmvpDerivation<'a> {
    /// Current PU position X (in pixels)
    pub_x: usize,
    /// Current PU position Y (in pixels)
    pub_y: usize,
    /// Current PU width (in pixels)
    pub_width: usize,
    /// Current PU height (in pixels)
    pub_height: usize,
    /// Current reference index we're looking for
    ref_idx: u8,
    /// Prediction list (L0 or L1)
    pred_list: PredictionList,
    /// Reference to the motion vector field for the current picture
    mv_field: Option<&'a MotionVectorField>,
    /// Current picture POC (for temporal scaling)
    current_poc: i32,
    /// Reference picture POCs for L0 (for temporal scaling)
    ref_poc_l0: Vec<i32>,
    /// Reference picture POCs for L1 (for temporal scaling)
    ref_poc_l1: Vec<i32>,
}

impl<'a> AmvpDerivation<'a> {
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
            mv_field: None,
            current_poc: 0,
            ref_poc_l0: Vec::new(),
            ref_poc_l1: Vec::new(),
        }
    }

    /// Create a new AMVP derivation context with motion vector field
    pub fn with_mv_field(
        pub_x: usize,
        pub_y: usize,
        pub_width: usize,
        pub_height: usize,
        ref_idx: u8,
        pred_list: PredictionList,
        mv_field: &'a MotionVectorField,
    ) -> Self {
        Self {
            pub_x,
            pub_y,
            pub_width,
            pub_height,
            ref_idx,
            pred_list,
            mv_field: Some(mv_field),
            current_poc: 0,
            ref_poc_l0: Vec::new(),
            ref_poc_l1: Vec::new(),
        }
    }

    /// Set POC information for temporal MV scaling
    pub fn set_poc_info(
        &mut self,
        current_poc: i32,
        ref_poc_l0: Vec<i32>,
        ref_poc_l1: Vec<i32>,
    ) {
        self.current_poc = current_poc;
        self.ref_poc_l0 = ref_poc_l0;
        self.ref_poc_l1 = ref_poc_l1;
    }

    /// Convert pixel coordinates to 4x4 block coordinates
    #[inline]
    fn pixel_to_block(pixel_pos: usize) -> usize {
        pixel_pos / 4
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
        let _col_x = self.pub_x + self.pub_width;
        let _col_y = self.pub_y + (self.pub_height / 2);

        // In a real implementation, this would fetch from the temporal MV field
        // (co-located picture in the DPB). For now, we return None to indicate
        // temporal MV not available. This would be implemented when we have full
        // DPB (Decoded Picture Buffer) support with co-located picture tracking.
        Ok(None)
    }

    /// Get MV from neighbor position
    ///
    /// Fetches the motion vector from the MV field at the given pixel position.
    /// Converts pixel coordinates to 4x4 block coordinates and retrieves the MV
    /// for the appropriate prediction list.
    ///
    /// Returns the MV if:
    /// 1. The neighbor block has inter prediction mode
    /// 2. The neighbor uses the same prediction list (or has a compatible MV)
    /// 3. After optional scaling for different reference pictures
    fn get_neighbor_mv(&self, x: usize, y: usize) -> Result<Option<MotionVector>> {
        let mv_field = match self.mv_field {
            Some(field) => field,
            None => return Ok(None), // No MV field available
        };

        // Convert pixel coordinates to 4x4 block coordinates
        let block_x = Self::pixel_to_block(x);
        let block_y = Self::pixel_to_block(y);

        // First, try to get MV from the same prediction list
        let (mv_opt, neighbor_ref_idx) = match self.pred_list {
            PredictionList::L0 => {
                if let Some((mv, ref_idx)) = mv_field.get_mv_l0(block_x, block_y) {
                    (Some(mv), ref_idx)
                } else {
                    (None, 0)
                }
            }
            PredictionList::L1 => {
                if let Some((mv, ref_idx)) = mv_field.get_mv_l1(block_x, block_y) {
                    (Some(mv), ref_idx)
                } else {
                    (None, 0)
                }
            }
        };

        if let Some(mv) = mv_opt {
            // Check if the MV is valid (non-zero or explicitly set)
            // A zero MV with ref_idx 0 might still be valid if block is inter-coded

            // If reference indices match, return MV directly
            if neighbor_ref_idx == self.ref_idx {
                return Ok(Some(mv));
            }

            // Reference indices differ - need to scale the MV
            // Get the POC for scaling
            let target_ref_poc = self.get_ref_poc(self.ref_idx, self.pred_list);
            let neighbor_ref_poc = self.get_ref_poc(neighbor_ref_idx, self.pred_list);

            if let (Some(target_poc), Some(neighbor_poc)) = (target_ref_poc, neighbor_ref_poc) {
                let scaled_mv = self.scale_mv(mv, neighbor_poc, self.current_poc, target_poc);
                return Ok(Some(scaled_mv));
            }

            // If we can't get POC info for scaling, still return the MV
            // (this is a fallback for when POC info is not available)
            return Ok(Some(mv));
        }

        // If same list didn't have MV, try the other list and scale
        let (other_mv_opt, other_ref_idx) = match self.pred_list {
            PredictionList::L0 => {
                // Try L1
                if let Some((mv, ref_idx)) = mv_field.get_mv_l1(block_x, block_y) {
                    (Some(mv), ref_idx)
                } else {
                    (None, 0)
                }
            }
            PredictionList::L1 => {
                // Try L0
                if let Some((mv, ref_idx)) = mv_field.get_mv_l0(block_x, block_y) {
                    (Some(mv), ref_idx)
                } else {
                    (None, 0)
                }
            }
        };

        if let Some(mv) = other_mv_opt {
            // Get POCs for cross-list scaling
            let target_ref_poc = self.get_ref_poc(self.ref_idx, self.pred_list);
            let other_list = match self.pred_list {
                PredictionList::L0 => PredictionList::L1,
                PredictionList::L1 => PredictionList::L0,
            };
            let neighbor_ref_poc = self.get_ref_poc(other_ref_idx, other_list);

            if let (Some(target_poc), Some(neighbor_poc)) = (target_ref_poc, neighbor_ref_poc) {
                let scaled_mv = self.scale_mv(mv, neighbor_poc, self.current_poc, target_poc);
                return Ok(Some(scaled_mv));
            }

            // Fallback: return unscaled MV if POC info unavailable
            return Ok(Some(mv));
        }

        Ok(None)
    }

    /// Get reference picture POC for a given ref_idx and list
    fn get_ref_poc(&self, ref_idx: u8, list: PredictionList) -> Option<i32> {
        let ref_pocs = match list {
            PredictionList::L0 => &self.ref_poc_l0,
            PredictionList::L1 => &self.ref_poc_l1,
        };
        ref_pocs.get(ref_idx as usize).copied()
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
            .ok_or_else(|| Error::Codec(format!("Invalid AMVP index: {}", index)))
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

    /// Build candidate list from derivation (without MV field - legacy compatibility)
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

    /// Build candidate list from derivation with MV field
    pub fn build_with_mv_field<'a>(
        pub_x: usize,
        pub_y: usize,
        pub_width: usize,
        pub_height: usize,
        ref_idx: u8,
        pred_list: PredictionList,
        mv_field: &'a MotionVectorField,
    ) -> Result<Self> {
        let derivation = AmvpDerivation::with_mv_field(
            pub_x, pub_y, pub_width, pub_height, ref_idx, pred_list, mv_field
        );
        let candidates = derivation.derive_candidates()?;

        Ok(Self { candidates })
    }

    /// Build candidate list with full POC info for proper MV scaling
    pub fn build_with_poc_info<'a>(
        pub_x: usize,
        pub_y: usize,
        pub_width: usize,
        pub_height: usize,
        ref_idx: u8,
        pred_list: PredictionList,
        mv_field: &'a MotionVectorField,
        current_poc: i32,
        ref_poc_l0: Vec<i32>,
        ref_poc_l1: Vec<i32>,
    ) -> Result<Self> {
        let mut derivation = AmvpDerivation::with_mv_field(
            pub_x, pub_y, pub_width, pub_height, ref_idx, pred_list, mv_field
        );
        derivation.set_poc_info(current_poc, ref_poc_l0, ref_poc_l1);
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
    fn test_amvp_derivation_with_mv_field() {
        let mv_field = MotionVectorField::new(64, 64);
        let amvp = AmvpDerivation::with_mv_field(32, 32, 16, 16, 0, PredictionList::L0, &mv_field);
        assert_eq!(amvp.pub_x, 32);
        assert!(amvp.mv_field.is_some());
    }

    #[test]
    fn test_amvp_derive_candidates_minimum() {
        let amvp = AmvpDerivation::new(32, 32, 16, 16, 0, PredictionList::L0);
        let candidates = amvp.derive_candidates().unwrap();

        // Should always have at least 2 candidates (with zero MVs if needed)
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn test_amvp_derive_candidates_with_mv_field() {
        let mut mv_field = MotionVectorField::new(64, 64);
        // Set MV for neighbor A0 (at pixel 31, 47 -> block 7, 11)
        mv_field.set_mv_l0(7, 11, MotionVector::new(16, 8), 0).unwrap();

        let amvp = AmvpDerivation::with_mv_field(32, 32, 16, 16, 0, PredictionList::L0, &mv_field);
        let candidates = amvp.derive_candidates().unwrap();

        // Should have the MV from A0 and one zero MV
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0], MotionVector::new(16, 8));
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
    fn test_amvp_get_neighbor_mv_with_field() {
        let mut mv_field = MotionVectorField::new(64, 64);
        // Set MV at block position (4, 4) which corresponds to pixels 16-19, 16-19
        mv_field.set_mv_l0(4, 4, MotionVector::new(20, 10), 0).unwrap();

        let amvp = AmvpDerivation::with_mv_field(20, 20, 8, 8, 0, PredictionList::L0, &mv_field);

        // Try to get MV at pixel (17, 17) which should be block (4, 4)
        let mv = amvp.get_neighbor_mv(17, 17).unwrap();
        assert!(mv.is_some());
        assert_eq!(mv.unwrap(), MotionVector::new(20, 10));
    }

    #[test]
    fn test_amvp_get_neighbor_mv_cross_list() {
        let mut mv_field = MotionVectorField::new(64, 64);
        // Set MV only in L1 list
        mv_field.set_mv_l1(4, 4, MotionVector::new(20, 10), 0).unwrap();

        // Request from L0 - should fall back to L1
        let amvp = AmvpDerivation::with_mv_field(20, 20, 8, 8, 0, PredictionList::L0, &mv_field);
        let mv = amvp.get_neighbor_mv(17, 17).unwrap();
        assert!(mv.is_some());
        assert_eq!(mv.unwrap(), MotionVector::new(20, 10));
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
    fn test_amvp_candidate_list_build_with_mv_field() {
        let mut mv_field = MotionVectorField::new(64, 64);
        // Set MVs for neighbors
        mv_field.set_mv_l0(7, 11, MotionVector::new(16, 8), 0).unwrap(); // A0
        mv_field.set_mv_l0(11, 7, MotionVector::new(20, 10), 0).unwrap(); // B0

        let list = AmvpCandidateList::build_with_mv_field(
            32, 32, 16, 16, 0, PredictionList::L0, &mv_field
        ).unwrap();

        assert_eq!(list.len(), 2);
        // First candidate should be from A0
        assert_eq!(list.get(0).unwrap(), MotionVector::new(16, 8));
        // Second candidate should be from B0
        assert_eq!(list.get(1).unwrap(), MotionVector::new(20, 10));
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
        // scale = (10 << 8) / 15 = 2560 / 15 = 170
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

    #[test]
    fn test_pixel_to_block_conversion() {
        assert_eq!(AmvpDerivation::<'_>::pixel_to_block(0), 0);
        assert_eq!(AmvpDerivation::<'_>::pixel_to_block(3), 0);
        assert_eq!(AmvpDerivation::<'_>::pixel_to_block(4), 1);
        assert_eq!(AmvpDerivation::<'_>::pixel_to_block(7), 1);
        assert_eq!(AmvpDerivation::<'_>::pixel_to_block(8), 2);
        assert_eq!(AmvpDerivation::<'_>::pixel_to_block(31), 7);
        assert_eq!(AmvpDerivation::<'_>::pixel_to_block(32), 8);
    }

    #[test]
    fn test_amvp_with_poc_info() {
        let mut mv_field = MotionVectorField::new(64, 64);
        mv_field.set_mv_l0(7, 11, MotionVector::new(16, 8), 1).unwrap(); // Different ref_idx

        let list = AmvpCandidateList::build_with_poc_info(
            32, 32, 16, 16,
            0,  // We want ref_idx 0
            PredictionList::L0,
            &mv_field,
            20, // current_poc
            vec![10, 5], // ref_poc_l0: ref 0 is at POC 10, ref 1 is at POC 5
            vec![], // ref_poc_l1
        ).unwrap();

        assert_eq!(list.len(), 2);
        // The MV should be scaled because neighbor has ref_idx 1 but we want ref_idx 0
        // neighbor_ref_poc = 5, target_ref_poc = 10
        // cur_diff = 20-10 = 10, neighbor_diff = 20-5 = 15
        // scale = 10/15 = 2/3
        let scaled = list.get(0).unwrap();
        // 16 * (10/15) = 10.67 -> 10 (truncated)
        // 8 * (10/15) = 5.33 -> 5 (truncated)
        assert_eq!(scaled.x, 10);
        assert_eq!(scaled.y, 5);
    }
}
