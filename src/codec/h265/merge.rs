//! Merge Mode for H.265/HEVC
//!
//! This module implements merge mode for inter prediction, where the PU directly
//! inherits motion information from spatial and temporal neighbors without signaling
//! any motion vector differential.
//!
//! # Merge Mode vs AMVP
//!
//! - **Merge**: Inherits complete MV info (no differential), up to 5 candidates
//! - **AMVP**: Uses MV predictor + differential, up to 2 candidates
//!
//! # Merge Candidate Types
//!
//! 1. **Spatial candidates**: From neighboring blocks (A0, A1, B0, B1, B2)
//! 2. **Temporal candidates**: From co-located picture (collocated)
//! 3. **Combined bi-predictive**: Combine L0 from one neighbor, L1 from another
//! 4. **Zero MV**: Fill remaining slots with zero MVs
//!
//! # Merge List Construction
//!
//! ```text
//!     B2   B1   B0
//!      +---+---+
//!      |       |
//!   A1 +   X   |  (X = current PU)
//!      |       |
//!   A0 +-------+
//! ```
//!
//! Priority order: A1 → B1 → B0 → A0 → B2 → Temporal → Combined → Zero

use crate::codec::h265::mv::{MotionVector, MergeCandidate, PredictionFlag, PredictionList};
use crate::error::{Error, Result};

/// Merge mode candidate derivation engine
pub struct MergeDerivation {
    /// Current PU position X
    pub_x: usize,
    /// Current PU position Y
    pub_y: usize,
    /// Current PU width
    pub_width: usize,
    /// Current PU height
    pub_height: usize,
    /// Maximum merge candidates (typically 5)
    max_candidates: usize,
}

impl MergeDerivation {
    /// Create a new merge derivation context
    pub fn new(
        pub_x: usize,
        pub_y: usize,
        pub_width: usize,
        pub_height: usize,
        max_candidates: usize,
    ) -> Self {
        Self {
            pub_x,
            pub_y,
            pub_width,
            pub_height,
            max_candidates: max_candidates.min(5), // H.265 spec maximum
        }
    }

    /// Derive merge candidate list
    ///
    /// Returns up to 5 candidates for merge mode.
    pub fn derive_candidates(&self) -> Result<Vec<MergeCandidate>> {
        let mut candidates = Vec::new();

        // 1. Spatial candidates (check A1, B1, B0, A0, B2 in order)
        self.add_spatial_candidates(&mut candidates)?;

        // 2. If we have enough candidates, we're done
        if candidates.len() >= self.max_candidates {
            candidates.truncate(self.max_candidates);
            return Ok(candidates);
        }

        // 3. Temporal candidate
        if let Some(temporal) = self.derive_temporal()? {
            if !self.is_duplicate(&temporal, &candidates) {
                candidates.push(temporal);
            }
        }

        // 4. Combined bi-predictive candidates (if we have spatial L0 and L1)
        if candidates.len() < self.max_candidates {
            self.add_combined_candidates(&mut candidates)?;
        }

        // 5. Fill remaining with zero MV
        while candidates.len() < self.max_candidates {
            candidates.push(MergeCandidate::new(
                MotionVector::zero(),
                MotionVector::zero(),
                0,
                0,
                PredictionFlag::L0,
            ));
        }

        Ok(candidates)
    }

    /// Add spatial merge candidates
    fn add_spatial_candidates(&self, candidates: &mut Vec<MergeCandidate>) -> Result<()> {
        // Priority order per H.265 spec: A1 → B1 → B0 → A0 → B2

        // A1 (left, above bottom)
        if let Some(cand) = self.get_neighbor_a1()? {
            if !self.is_duplicate(&cand, candidates) {
                candidates.push(cand);
                if candidates.len() >= self.max_candidates {
                    return Ok(());
                }
            }
        }

        // B1 (above, left of right)
        if let Some(cand) = self.get_neighbor_b1()? {
            if !self.is_duplicate(&cand, candidates) {
                candidates.push(cand);
                if candidates.len() >= self.max_candidates {
                    return Ok(());
                }
            }
        }

        // B0 (above, right)
        if let Some(cand) = self.get_neighbor_b0()? {
            if !self.is_duplicate(&cand, candidates) {
                candidates.push(cand);
                if candidates.len() >= self.max_candidates {
                    return Ok(());
                }
            }
        }

        // A0 (left, bottom)
        if let Some(cand) = self.get_neighbor_a0()? {
            if !self.is_duplicate(&cand, candidates) {
                candidates.push(cand);
                if candidates.len() >= self.max_candidates {
                    return Ok(());
                }
            }
        }

        // B2 (above-left corner)
        if let Some(cand) = self.get_neighbor_b2()? {
            if !self.is_duplicate(&cand, candidates) {
                candidates.push(cand);
            }
        }

        Ok(())
    }

    /// Get merge candidate from A0 position (left, bottom)
    fn get_neighbor_a0(&self) -> Result<Option<MergeCandidate>> {
        if self.pub_x == 0 {
            return Ok(None);
        }
        // In real implementation, would query MotionVectorField
        // For now, stub returns None
        Ok(None)
    }

    /// Get merge candidate from A1 position (left, above bottom)
    fn get_neighbor_a1(&self) -> Result<Option<MergeCandidate>> {
        if self.pub_x == 0 || self.pub_height <= 4 {
            return Ok(None);
        }
        Ok(None)
    }

    /// Get merge candidate from B0 position (above, right)
    fn get_neighbor_b0(&self) -> Result<Option<MergeCandidate>> {
        if self.pub_y == 0 {
            return Ok(None);
        }
        Ok(None)
    }

    /// Get merge candidate from B1 position (above, left of right)
    fn get_neighbor_b1(&self) -> Result<Option<MergeCandidate>> {
        if self.pub_y == 0 || self.pub_width <= 4 {
            return Ok(None);
        }
        Ok(None)
    }

    /// Get merge candidate from B2 position (above-left corner)
    fn get_neighbor_b2(&self) -> Result<Option<MergeCandidate>> {
        if self.pub_x == 0 || self.pub_y == 0 {
            return Ok(None);
        }
        Ok(None)
    }

    /// Derive temporal merge candidate
    fn derive_temporal(&self) -> Result<Option<MergeCandidate>> {
        // Would query co-located picture's MV field
        // Stub for now
        Ok(None)
    }

    /// Add combined bi-predictive candidates
    ///
    /// Creates new candidates by combining L0 from one spatial candidate
    /// and L1 from another spatial candidate.
    fn add_combined_candidates(&self, candidates: &mut Vec<MergeCandidate>) -> Result<()> {
        if candidates.len() >= self.max_candidates {
            return Ok(());
        }

        // Try to find candidates with L0 and L1
        let mut l0_cands = Vec::new();
        let mut l1_cands = Vec::new();

        for cand in candidates.iter() {
            if cand.pred_flag == PredictionFlag::L0 || cand.pred_flag == PredictionFlag::Bi {
                l0_cands.push(cand.mv_l0);
            }
            if cand.pred_flag == PredictionFlag::L1 || cand.pred_flag == PredictionFlag::Bi {
                l1_cands.push(cand.mv_l1);
            }
        }

        // Create combined candidates
        for (i, &mv_l0) in l0_cands.iter().enumerate() {
            for (j, &mv_l1) in l1_cands.iter().enumerate() {
                if i != j && candidates.len() < self.max_candidates {
                    let combined = MergeCandidate::new(mv_l0, mv_l1, 0, 0, PredictionFlag::Bi);
                    if !self.is_duplicate(&combined, candidates) {
                        candidates.push(combined);
                        if candidates.len() >= self.max_candidates {
                            return Ok(());
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if candidate is duplicate
    fn is_duplicate(&self, cand: &MergeCandidate, candidates: &[MergeCandidate]) -> bool {
        candidates.iter().any(|c| {
            c.mv_l0 == cand.mv_l0
                && c.mv_l1 == cand.mv_l1
                && c.ref_idx_l0 == cand.ref_idx_l0
                && c.ref_idx_l1 == cand.ref_idx_l1
                && c.pred_flag == cand.pred_flag
        })
    }
}

/// Merge candidate list manager
pub struct MergeCandidateListBuilder {
    /// Maximum number of candidates
    max_candidates: usize,
}

impl MergeCandidateListBuilder {
    /// Create a new merge candidate list builder
    pub fn new(max_candidates: usize) -> Self {
        Self {
            max_candidates: max_candidates.min(5),
        }
    }

    /// Build merge candidate list for a PU
    pub fn build(
        &self,
        pub_x: usize,
        pub_y: usize,
        pub_width: usize,
        pub_height: usize,
    ) -> Result<Vec<MergeCandidate>> {
        let derivation = MergeDerivation::new(
            pub_x,
            pub_y,
            pub_width,
            pub_height,
            self.max_candidates,
        );
        derivation.derive_candidates()
    }
}

impl Default for MergeCandidateListBuilder {
    fn default() -> Self {
        Self::new(5) // H.265 default
    }
}

/// Merge mode utilities
pub struct MergeUtils;

impl MergeUtils {
    /// Check if two merge candidates are identical
    pub fn are_identical(a: &MergeCandidate, b: &MergeCandidate) -> bool {
        a.mv_l0 == b.mv_l0
            && a.mv_l1 == b.mv_l1
            && a.ref_idx_l0 == b.ref_idx_l0
            && a.ref_idx_l1 == b.ref_idx_l1
            && a.pred_flag == b.pred_flag
    }

    /// Create a zero MV merge candidate
    pub fn zero_candidate() -> MergeCandidate {
        MergeCandidate::new(
            MotionVector::zero(),
            MotionVector::zero(),
            0,
            0,
            PredictionFlag::L0,
        )
    }

    /// Create a bi-predictive candidate from L0 and L1 candidates
    pub fn combine_candidates(
        l0_cand: &MergeCandidate,
        l1_cand: &MergeCandidate,
    ) -> MergeCandidate {
        MergeCandidate::new(
            l0_cand.mv_l0,
            l1_cand.mv_l1,
            l0_cand.ref_idx_l0,
            l1_cand.ref_idx_l1,
            PredictionFlag::Bi,
        )
    }

    /// Check if candidate has L0 prediction
    pub fn has_l0(cand: &MergeCandidate) -> bool {
        cand.pred_flag == PredictionFlag::L0 || cand.pred_flag == PredictionFlag::Bi
    }

    /// Check if candidate has L1 prediction
    pub fn has_l1(cand: &MergeCandidate) -> bool {
        cand.pred_flag == PredictionFlag::L1 || cand.pred_flag == PredictionFlag::Bi
    }

    /// Check if candidate is bi-predictive
    pub fn is_bi_pred(cand: &MergeCandidate) -> bool {
        cand.pred_flag == PredictionFlag::Bi
    }
}

/// Merge candidate pruning
///
/// H.265 uses redundancy checking to remove duplicate candidates
pub struct MergePruning;

impl MergePruning {
    /// Prune duplicate candidates from list
    pub fn prune_duplicates(candidates: &mut Vec<MergeCandidate>) {
        let mut unique = Vec::new();

        for cand in candidates.drain(..) {
            if !unique.iter().any(|u| MergeUtils::are_identical(&cand, u)) {
                unique.push(cand);
            }
        }

        *candidates = unique;
    }

    /// Remove candidates with same MV but different ref_idx
    ///
    /// This is a more aggressive pruning used in some encoder implementations
    pub fn prune_similar_mv(candidates: &mut Vec<MergeCandidate>) {
        let mut unique = Vec::new();

        for cand in candidates.drain(..) {
            let is_similar = unique.iter().any(|u: &MergeCandidate| {
                u.mv_l0 == cand.mv_l0 && u.mv_l1 == cand.mv_l1
            });

            if !is_similar {
                unique.push(cand);
            }
        }

        *candidates = unique;
    }
}

/// Spatial neighbor positions for merge mode
pub struct MergeSpatialNeighbors;

impl MergeSpatialNeighbors {
    /// Get all spatial neighbor positions in priority order
    ///
    /// Returns: [(A1), (B1), (B0), (A0), (B2)]
    pub fn get_all_positions(
        pub_x: usize,
        pub_y: usize,
        pub_width: usize,
        pub_height: usize,
    ) -> Vec<Option<(usize, usize)>> {
        vec![
            Self::get_a1(pub_x, pub_y, pub_height),
            Self::get_b1(pub_x, pub_y, pub_width),
            Self::get_b0(pub_x, pub_y, pub_width),
            Self::get_a0(pub_x, pub_y, pub_height),
            Self::get_b2(pub_x, pub_y),
        ]
    }

    /// A0: Left-bottom
    pub fn get_a0(pub_x: usize, pub_y: usize, pub_height: usize) -> Option<(usize, usize)> {
        if pub_x > 0 {
            Some((pub_x - 1, pub_y + pub_height - 1))
        } else {
            None
        }
    }

    /// A1: Left-above-bottom
    pub fn get_a1(pub_x: usize, pub_y: usize, pub_height: usize) -> Option<(usize, usize)> {
        if pub_x > 0 && pub_height > 4 {
            Some((pub_x - 1, pub_y + pub_height - 4))
        } else {
            None
        }
    }

    /// B0: Above-right
    pub fn get_b0(pub_x: usize, pub_y: usize, pub_width: usize) -> Option<(usize, usize)> {
        if pub_y > 0 {
            Some((pub_x + pub_width - 1, pub_y - 1))
        } else {
            None
        }
    }

    /// B1: Above-left-of-right
    pub fn get_b1(pub_x: usize, pub_y: usize, pub_width: usize) -> Option<(usize, usize)> {
        if pub_y > 0 && pub_width > 4 {
            Some((pub_x + pub_width - 4, pub_y - 1))
        } else {
            None
        }
    }

    /// B2: Above-left corner
    pub fn get_b2(pub_x: usize, pub_y: usize) -> Option<(usize, usize)> {
        if pub_x > 0 && pub_y > 0 {
            Some((pub_x - 1, pub_y - 1))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_derivation_creation() {
        let merge = MergeDerivation::new(32, 32, 16, 16, 5);
        assert_eq!(merge.pub_x, 32);
        assert_eq!(merge.pub_y, 32);
        assert_eq!(merge.max_candidates, 5);
    }

    #[test]
    fn test_merge_derivation_max_candidates_limit() {
        let merge = MergeDerivation::new(32, 32, 16, 16, 10);
        assert_eq!(merge.max_candidates, 5); // Clamped to spec maximum
    }

    #[test]
    fn test_merge_derive_candidates_minimum() {
        let merge = MergeDerivation::new(32, 32, 16, 16, 5);
        let candidates = merge.derive_candidates().unwrap();

        // Should have up to 5 candidates
        assert!(candidates.len() <= 5);
        assert!(!candidates.is_empty());
    }

    #[test]
    fn test_merge_zero_fill() {
        let merge = MergeDerivation::new(0, 0, 16, 16, 5);
        let candidates = merge.derive_candidates().unwrap();

        // At (0,0), no spatial neighbors, should fill with zeros
        assert_eq!(candidates.len(), 5);
        // All should be zero MVs
        assert!(candidates.iter().all(|c| c.mv_l0.is_zero()));
    }

    #[test]
    fn test_merge_is_duplicate() {
        let merge = MergeDerivation::new(32, 32, 16, 16, 5);
        let cand1 = MergeCandidate::new(
            MotionVector::new(16, 8),
            MotionVector::zero(),
            0,
            0,
            PredictionFlag::L0,
        );
        let cand2 = MergeCandidate::new(
            MotionVector::new(16, 8),
            MotionVector::zero(),
            0,
            0,
            PredictionFlag::L0,
        );
        let cand3 = MergeCandidate::new(
            MotionVector::new(20, 10),
            MotionVector::zero(),
            0,
            0,
            PredictionFlag::L0,
        );

        let candidates = vec![cand1];
        assert!(merge.is_duplicate(&cand2, &candidates));
        assert!(!merge.is_duplicate(&cand3, &candidates));
    }

    #[test]
    fn test_merge_candidate_list_builder() {
        let builder = MergeCandidateListBuilder::new(5);
        let candidates = builder.build(32, 32, 16, 16).unwrap();

        assert!(candidates.len() <= 5);
        assert!(!candidates.is_empty());
    }

    #[test]
    fn test_merge_candidate_list_builder_default() {
        let builder = MergeCandidateListBuilder::default();
        assert_eq!(builder.max_candidates, 5);
    }

    #[test]
    fn test_merge_utils_are_identical() {
        let cand1 = MergeCandidate::new(
            MotionVector::new(16, 8),
            MotionVector::zero(),
            0,
            0,
            PredictionFlag::L0,
        );
        let cand2 = MergeCandidate::new(
            MotionVector::new(16, 8),
            MotionVector::zero(),
            0,
            0,
            PredictionFlag::L0,
        );
        let cand3 = MergeCandidate::new(
            MotionVector::new(20, 10),
            MotionVector::zero(),
            0,
            0,
            PredictionFlag::L0,
        );

        assert!(MergeUtils::are_identical(&cand1, &cand2));
        assert!(!MergeUtils::are_identical(&cand1, &cand3));
    }

    #[test]
    fn test_merge_utils_zero_candidate() {
        let cand = MergeUtils::zero_candidate();
        assert!(cand.mv_l0.is_zero());
        assert!(cand.mv_l1.is_zero());
        assert_eq!(cand.pred_flag, PredictionFlag::L0);
    }

    #[test]
    fn test_merge_utils_combine_candidates() {
        let l0_cand = MergeCandidate::new(
            MotionVector::new(16, 8),
            MotionVector::zero(),
            0,
            0,
            PredictionFlag::L0,
        );
        let l1_cand = MergeCandidate::new(
            MotionVector::zero(),
            MotionVector::new(20, 10),
            0,
            1,
            PredictionFlag::L1,
        );

        let combined = MergeUtils::combine_candidates(&l0_cand, &l1_cand);
        assert_eq!(combined.mv_l0.x, 16);
        assert_eq!(combined.mv_l1.x, 20);
        assert_eq!(combined.pred_flag, PredictionFlag::Bi);
    }

    #[test]
    fn test_merge_utils_has_l0() {
        let l0_cand = MergeCandidate::new(
            MotionVector::new(16, 8),
            MotionVector::zero(),
            0,
            0,
            PredictionFlag::L0,
        );
        let bi_cand = MergeCandidate::new(
            MotionVector::new(16, 8),
            MotionVector::new(20, 10),
            0,
            0,
            PredictionFlag::Bi,
        );
        let l1_cand = MergeCandidate::new(
            MotionVector::zero(),
            MotionVector::new(20, 10),
            0,
            0,
            PredictionFlag::L1,
        );

        assert!(MergeUtils::has_l0(&l0_cand));
        assert!(MergeUtils::has_l0(&bi_cand));
        assert!(!MergeUtils::has_l0(&l1_cand));
    }

    #[test]
    fn test_merge_utils_has_l1() {
        let l0_cand = MergeCandidate::new(
            MotionVector::new(16, 8),
            MotionVector::zero(),
            0,
            0,
            PredictionFlag::L0,
        );
        let bi_cand = MergeCandidate::new(
            MotionVector::new(16, 8),
            MotionVector::new(20, 10),
            0,
            0,
            PredictionFlag::Bi,
        );
        let l1_cand = MergeCandidate::new(
            MotionVector::zero(),
            MotionVector::new(20, 10),
            0,
            0,
            PredictionFlag::L1,
        );

        assert!(!MergeUtils::has_l1(&l0_cand));
        assert!(MergeUtils::has_l1(&bi_cand));
        assert!(MergeUtils::has_l1(&l1_cand));
    }

    #[test]
    fn test_merge_utils_is_bi_pred() {
        let l0_cand = MergeCandidate::new(
            MotionVector::new(16, 8),
            MotionVector::zero(),
            0,
            0,
            PredictionFlag::L0,
        );
        let bi_cand = MergeCandidate::new(
            MotionVector::new(16, 8),
            MotionVector::new(20, 10),
            0,
            0,
            PredictionFlag::Bi,
        );

        assert!(!MergeUtils::is_bi_pred(&l0_cand));
        assert!(MergeUtils::is_bi_pred(&bi_cand));
    }

    #[test]
    fn test_merge_pruning_duplicates() {
        let cand1 = MergeCandidate::new(
            MotionVector::new(16, 8),
            MotionVector::zero(),
            0,
            0,
            PredictionFlag::L0,
        );
        let cand2 = MergeCandidate::new(
            MotionVector::new(16, 8),
            MotionVector::zero(),
            0,
            0,
            PredictionFlag::L0,
        );
        let cand3 = MergeCandidate::new(
            MotionVector::new(20, 10),
            MotionVector::zero(),
            0,
            0,
            PredictionFlag::L0,
        );

        let mut candidates = vec![cand1, cand2, cand3];
        MergePruning::prune_duplicates(&mut candidates);

        assert_eq!(candidates.len(), 2); // cand1 and cand3 only
    }

    #[test]
    fn test_merge_pruning_similar_mv() {
        let cand1 = MergeCandidate::new(
            MotionVector::new(16, 8),
            MotionVector::zero(),
            0,
            0,
            PredictionFlag::L0,
        );
        let cand2 = MergeCandidate::new(
            MotionVector::new(16, 8),
            MotionVector::zero(),
            1,
            0,
            PredictionFlag::L0,
        ); // Different ref_idx
        let cand3 = MergeCandidate::new(
            MotionVector::new(20, 10),
            MotionVector::zero(),
            0,
            0,
            PredictionFlag::L0,
        );

        let mut candidates = vec![cand1, cand2, cand3];
        MergePruning::prune_similar_mv(&mut candidates);

        assert_eq!(candidates.len(), 2); // Same MV pruned
    }

    #[test]
    fn test_merge_spatial_neighbors_a0() {
        let pos = MergeSpatialNeighbors::get_a0(32, 32, 16);
        assert_eq!(pos, Some((31, 47)));
    }

    #[test]
    fn test_merge_spatial_neighbors_a0_edge() {
        let pos = MergeSpatialNeighbors::get_a0(0, 32, 16);
        assert_eq!(pos, None);
    }

    #[test]
    fn test_merge_spatial_neighbors_all() {
        let positions = MergeSpatialNeighbors::get_all_positions(64, 64, 32, 32);
        assert_eq!(positions.len(), 5);

        // A1
        assert_eq!(positions[0], Some((63, 92)));
        // B1
        assert_eq!(positions[1], Some((92, 63)));
        // B0
        assert_eq!(positions[2], Some((95, 63)));
        // A0
        assert_eq!(positions[3], Some((63, 95)));
        // B2
        assert_eq!(positions[4], Some((63, 63)));
    }

    #[test]
    fn test_merge_spatial_neighbors_corner() {
        let positions = MergeSpatialNeighbors::get_all_positions(0, 0, 16, 16);

        // All should be None at corner
        assert!(positions.iter().all(|p| p.is_none()));
    }

    #[test]
    fn test_merge_derivation_combine_candidates() {
        let merge = MergeDerivation::new(32, 32, 16, 16, 5);
        let mut candidates = vec![
            MergeCandidate::new(
                MotionVector::new(16, 8),
                MotionVector::zero(),
                0,
                0,
                PredictionFlag::L0,
            ),
            MergeCandidate::new(
                MotionVector::zero(),
                MotionVector::new(20, 10),
                0,
                1,
                PredictionFlag::L1,
            ),
        ];

        merge.add_combined_candidates(&mut candidates).unwrap();

        // Should have original 2 + at least 1 combined
        assert!(candidates.len() >= 3);
    }

    #[test]
    fn test_merge_builder_custom_max() {
        let builder = MergeCandidateListBuilder::new(3);
        let candidates = builder.build(32, 32, 16, 16).unwrap();

        assert!(candidates.len() <= 3);
    }
}
