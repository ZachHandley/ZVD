//! Motion Vector structures and prediction for H.265/HEVC
//!
//! This module implements motion vector representation, storage, and prediction
//! mechanisms used in H.265 inter-frame coding.
//!
//! # Motion Vectors in H.265
//!
//! Motion vectors point from the current block to a similar block in a reference picture.
//! They have 1/4-pixel precision and can reference multiple pictures.
//!
//! # Prediction Modes
//!
//! - **AMVP (Advanced Motion Vector Prediction)**: Predicts MV from spatial/temporal neighbors
//! - **Merge Mode**: Inherits MV and reference index from neighbors
//!
//! # Fractional Pixel Precision
//!
//! Motion vectors use 1/4-pixel precision:
//! - Integer positions: Direct sample copy
//! - 1/2-pixel positions: 8-tap interpolation filter (luma), 4-tap (chroma)
//! - 1/4-pixel positions: Interpolation between integer and 1/2-pixel samples

use crate::error::{Error, Result};

/// Motion vector with 1/4-pixel precision
///
/// Stored as (x, y) offsets in quarter-pixel units.
/// Example: MV (4, 8) means 1 pixel right, 2 pixels down
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MotionVector {
    /// Horizontal component (1/4-pixel units)
    pub x: i16,
    /// Vertical component (1/4-pixel units)
    pub y: i16,
}

impl MotionVector {
    /// Create a new motion vector
    pub fn new(x: i16, y: i16) -> Self {
        Self { x, y }
    }

    /// Create a zero motion vector
    pub fn zero() -> Self {
        Self { x: 0, y: 0 }
    }

    /// Check if this is a zero motion vector
    pub fn is_zero(&self) -> bool {
        self.x == 0 && self.y == 0
    }

    /// Get integer pixel part (divide by 4)
    pub fn integer_x(&self) -> i16 {
        self.x >> 2
    }

    /// Get integer pixel part (divide by 4)
    pub fn integer_y(&self) -> i16 {
        self.y >> 2
    }

    /// Get fractional part (0-3)
    pub fn frac_x(&self) -> u8 {
        (self.x & 3) as u8
    }

    /// Get fractional part (0-3)
    pub fn frac_y(&self) -> u8 {
        (self.y & 3) as u8
    }

    /// Scale motion vector by given factor
    ///
    /// Used for temporal scaling when reference pictures are at different distances
    pub fn scale(&self, factor: i32, shift: u8) -> Self {
        let x = ((self.x as i32 * factor) >> shift) as i16;
        let y = ((self.y as i32 * factor) >> shift) as i16;
        Self { x, y }
    }

    /// Clip motion vector to valid range
    pub fn clip(&self, min_x: i16, max_x: i16, min_y: i16, max_y: i16) -> Self {
        Self {
            x: self.x.clamp(min_x, max_x),
            y: self.y.clamp(min_y, max_y),
        }
    }

    /// Calculate median of three motion vectors
    ///
    /// Used in motion vector prediction
    pub fn median3(a: Self, b: Self, c: Self) -> Self {
        Self {
            x: median3(a.x, b.x, c.x),
            y: median3(a.y, b.y, c.y),
        }
    }
}

/// Motion vector candidate for prediction
#[derive(Debug, Clone, Copy)]
pub struct MvCandidate {
    /// Motion vector
    pub mv: MotionVector,
    /// Reference picture index (0 = most recent, 1 = second most recent, etc.)
    pub ref_idx: u8,
    /// Prediction list (L0 or L1 for B-frames)
    pub pred_list: PredictionList,
}

impl MvCandidate {
    /// Create a new MV candidate
    pub fn new(mv: MotionVector, ref_idx: u8, pred_list: PredictionList) -> Self {
        Self {
            mv,
            ref_idx,
            pred_list,
        }
    }

    /// Create an unavailable candidate (zero MV, invalid ref)
    pub fn unavailable() -> Self {
        Self {
            mv: MotionVector::zero(),
            ref_idx: 0,
            pred_list: PredictionList::L0,
        }
    }
}

/// Prediction list identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionList {
    /// List 0 (forward prediction)
    L0 = 0,
    /// List 1 (backward prediction, B-frames only)
    L1 = 1,
}

/// Spatial neighboring position for MV prediction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialNeighbor {
    /// Left neighbor (A0)
    Left = 0,
    /// Above neighbor (B0)
    Above = 1,
    /// Above-right neighbor (B1)
    AboveRight = 2,
    /// Below-left neighbor (A1)
    BelowLeft = 3,
    /// Above-left neighbor (B2)
    AboveLeft = 4,
}

/// Motion vector predictor for AMVP mode
pub struct MvPredictor {
    /// Available spatial candidates
    spatial_candidates: Vec<MvCandidate>,
    /// Available temporal candidates
    temporal_candidates: Vec<MvCandidate>,
}

impl MvPredictor {
    /// Create a new MV predictor
    pub fn new() -> Self {
        Self {
            spatial_candidates: Vec::new(),
            temporal_candidates: Vec::new(),
        }
    }

    /// Add a spatial candidate
    pub fn add_spatial(&mut self, candidate: MvCandidate) {
        self.spatial_candidates.push(candidate);
    }

    /// Add a temporal candidate
    pub fn add_temporal(&mut self, candidate: MvCandidate) {
        self.temporal_candidates.push(candidate);
    }

    /// Get predictor at given index
    ///
    /// Returns the MV predictor for AMVP mode.
    /// Index selects from available candidates (up to 2 for AMVP).
    pub fn get_predictor(&self, index: usize) -> Result<MotionVector> {
        // Combine spatial and temporal candidates
        let mut candidates = self.spatial_candidates.clone();
        candidates.extend(self.temporal_candidates.iter());

        if candidates.is_empty() {
            return Ok(MotionVector::zero());
        }

        // AMVP provides up to 2 candidates
        let idx = index.min(candidates.len() - 1);
        Ok(candidates[idx].mv)
    }

    /// Get number of available candidates
    pub fn num_candidates(&self) -> usize {
        self.spatial_candidates.len() + self.temporal_candidates.len()
    }

    /// Clear all candidates
    pub fn clear(&mut self) {
        self.spatial_candidates.clear();
        self.temporal_candidates.clear();
    }
}

/// Merge mode candidate
///
/// In merge mode, the PU inherits both MV and reference index from a neighbor
#[derive(Debug, Clone, Copy)]
pub struct MergeCandidate {
    /// Motion vector for L0
    pub mv_l0: MotionVector,
    /// Motion vector for L1 (B-frames)
    pub mv_l1: MotionVector,
    /// Reference index for L0
    pub ref_idx_l0: u8,
    /// Reference index for L1 (B-frames)
    pub ref_idx_l1: u8,
    /// Prediction direction
    pub pred_flag: PredictionFlag,
}

impl MergeCandidate {
    /// Create a new merge candidate
    pub fn new(
        mv_l0: MotionVector,
        mv_l1: MotionVector,
        ref_idx_l0: u8,
        ref_idx_l1: u8,
        pred_flag: PredictionFlag,
    ) -> Self {
        Self {
            mv_l0,
            mv_l1,
            ref_idx_l0,
            ref_idx_l1,
            pred_flag,
        }
    }

    /// Create an unavailable merge candidate
    pub fn unavailable() -> Self {
        Self {
            mv_l0: MotionVector::zero(),
            mv_l1: MotionVector::zero(),
            ref_idx_l0: 0,
            ref_idx_l1: 0,
            pred_flag: PredictionFlag::L0,
        }
    }
}

/// Prediction flag for inter prediction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionFlag {
    /// Predict from L0 only
    L0 = 0,
    /// Predict from L1 only
    L1 = 1,
    /// Bi-directional prediction (L0 + L1)
    Bi = 2,
}

/// Merge candidate list for merge mode
pub struct MergeCandidateList {
    /// List of merge candidates
    candidates: Vec<MergeCandidate>,
    /// Maximum number of candidates
    max_candidates: usize,
}

impl MergeCandidateList {
    /// Create a new merge candidate list
    pub fn new(max_candidates: usize) -> Self {
        Self {
            candidates: Vec::new(),
            max_candidates: max_candidates.min(5), // H.265 allows up to 5
        }
    }

    /// Add a merge candidate
    pub fn add(&mut self, candidate: MergeCandidate) {
        if self.candidates.len() < self.max_candidates {
            self.candidates.push(candidate);
        }
    }

    /// Get candidate at index
    pub fn get(&self, index: usize) -> Result<MergeCandidate> {
        self.candidates
            .get(index)
            .copied()
            .ok_or_else(|| Error::codec(format!("Invalid merge index: {}", index)))
    }

    /// Get number of candidates
    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    /// Check if list is empty
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    /// Check if list is full
    pub fn is_full(&self) -> bool {
        self.candidates.len() >= self.max_candidates
    }

    /// Clear the list
    pub fn clear(&mut self) {
        self.candidates.clear();
    }
}

/// Calculate median of three values
fn median3(a: i16, b: i16, c: i16) -> i16 {
    if a > b {
        if b > c {
            b
        } else if a > c {
            c
        } else {
            a
        }
    } else {
        if a > c {
            a
        } else if b > c {
            c
        } else {
            b
        }
    }
}

/// Motion vector field for a picture
///
/// Stores motion vectors for each block in the picture
#[derive(Debug, Clone)]
pub struct MotionVectorField {
    /// Width in 4x4 blocks
    width_in_4x4: usize,
    /// Height in 4x4 blocks
    height_in_4x4: usize,
    /// Motion vectors for L0
    mvs_l0: Vec<MotionVector>,
    /// Motion vectors for L1
    mvs_l1: Vec<MotionVector>,
    /// Reference indices for L0
    ref_idx_l0: Vec<i8>,
    /// Reference indices for L1
    ref_idx_l1: Vec<i8>,
    /// Prediction flags per block
    pred_flags: Vec<PredictionFlag>,
}

impl MotionVectorField {
    /// Create a new motion vector field
    pub fn new(width: usize, height: usize) -> Self {
        let width_in_4x4 = (width + 3) / 4;
        let height_in_4x4 = (height + 3) / 4;
        let num_blocks = width_in_4x4 * height_in_4x4;

        Self {
            width_in_4x4,
            height_in_4x4,
            mvs_l0: vec![MotionVector::zero(); num_blocks],
            mvs_l1: vec![MotionVector::zero(); num_blocks],
            ref_idx_l0: vec![-1; num_blocks], // -1 indicates unavailable
            ref_idx_l1: vec![-1; num_blocks],
            pred_flags: vec![PredictionFlag::L0; num_blocks],
        }
    }

    /// Get width in 4x4 blocks
    pub fn width_in_blocks(&self) -> usize {
        self.width_in_4x4
    }

    /// Get height in 4x4 blocks
    pub fn height_in_blocks(&self) -> usize {
        self.height_in_4x4
    }

    /// Set motion vector for a block (L0 only)
    pub fn set_mv_l0(&mut self, x: usize, y: usize, mv: MotionVector, ref_idx: i8) -> Result<()> {
        let idx = self.get_index(x, y)?;
        self.mvs_l0[idx] = mv;
        self.ref_idx_l0[idx] = ref_idx;
        self.pred_flags[idx] = PredictionFlag::L0;
        Ok(())
    }

    /// Set motion vector for L1 only
    pub fn set_mv_l1(&mut self, x: usize, y: usize, mv: MotionVector, ref_idx: i8) -> Result<()> {
        let idx = self.get_index(x, y)?;
        self.mvs_l1[idx] = mv;
        self.ref_idx_l1[idx] = ref_idx;
        self.pred_flags[idx] = PredictionFlag::L1;
        Ok(())
    }

    /// Set motion vectors for bi-prediction
    pub fn set_mv_bi(
        &mut self,
        x: usize,
        y: usize,
        mv_l0: MotionVector,
        mv_l1: MotionVector,
        ref_idx_l0: i8,
        ref_idx_l1: i8,
    ) -> Result<()> {
        let idx = self.get_index(x, y)?;
        self.mvs_l0[idx] = mv_l0;
        self.mvs_l1[idx] = mv_l1;
        self.ref_idx_l0[idx] = ref_idx_l0;
        self.ref_idx_l1[idx] = ref_idx_l1;
        self.pred_flags[idx] = PredictionFlag::Bi;
        Ok(())
    }

    /// Get motion vector for L0
    pub fn get_mv_l0(&self, x: usize, y: usize) -> Option<(MotionVector, i8)> {
        let idx = self.get_index(x, y).ok()?;
        let ref_idx = self.ref_idx_l0[idx];
        if ref_idx < 0 {
            return None; // Not available
        }
        Some((self.mvs_l0[idx], ref_idx))
    }

    /// Get motion vector for L1
    pub fn get_mv_l1(&self, x: usize, y: usize) -> Option<(MotionVector, i8)> {
        let idx = self.get_index(x, y).ok()?;
        let ref_idx = self.ref_idx_l1[idx];
        if ref_idx < 0 {
            return None; // Not available
        }
        Some((self.mvs_l1[idx], ref_idx))
    }

    /// Get prediction flag for a block
    pub fn get_pred_flag(&self, x: usize, y: usize) -> Option<PredictionFlag> {
        let idx = self.get_index(x, y).ok()?;
        Some(self.pred_flags[idx])
    }

    /// Check if a block has valid motion information
    pub fn is_inter(&self, x: usize, y: usize) -> bool {
        if let Ok(idx) = self.get_index(x, y) {
            self.ref_idx_l0[idx] >= 0 || self.ref_idx_l1[idx] >= 0
        } else {
            false
        }
    }

    /// Get merge candidate from a block position
    ///
    /// Returns None if the block is not inter-coded or out of bounds
    pub fn get_merge_candidate(&self, x: usize, y: usize) -> Option<MergeCandidate> {
        let idx = self.get_index(x, y).ok()?;

        let ref_l0 = self.ref_idx_l0[idx];
        let ref_l1 = self.ref_idx_l1[idx];

        // Block must be inter-coded (at least one valid reference)
        if ref_l0 < 0 && ref_l1 < 0 {
            return None;
        }

        let pred_flag = self.pred_flags[idx];
        let mv_l0 = self.mvs_l0[idx];
        let mv_l1 = self.mvs_l1[idx];

        Some(MergeCandidate::new(
            mv_l0,
            mv_l1,
            ref_l0.max(0) as u8,
            ref_l1.max(0) as u8,
            pred_flag,
        ))
    }

    /// Get merge candidate from pixel position (converts to 4x4 block coordinates)
    pub fn get_merge_candidate_at_pixel(&self, px: usize, py: usize) -> Option<MergeCandidate> {
        let block_x = px / 4;
        let block_y = py / 4;
        self.get_merge_candidate(block_x, block_y)
    }

    /// Get block index
    fn get_index(&self, x: usize, y: usize) -> Result<usize> {
        if x >= self.width_in_4x4 || y >= self.height_in_4x4 {
            return Err(Error::codec(format!(
                "Block position out of bounds: ({}, {})",
                x, y
            )));
        }
        Ok(y * self.width_in_4x4 + x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_vector_creation() {
        let mv = MotionVector::new(16, -8);
        assert_eq!(mv.x, 16);
        assert_eq!(mv.y, -8);
    }

    #[test]
    fn test_motion_vector_zero() {
        let mv = MotionVector::zero();
        assert!(mv.is_zero());
        assert_eq!(mv.x, 0);
        assert_eq!(mv.y, 0);
    }

    #[test]
    fn test_motion_vector_integer_parts() {
        let mv = MotionVector::new(17, -10); // 4.25 pixels, -2.5 pixels
        assert_eq!(mv.integer_x(), 4);
        assert_eq!(mv.integer_y(), -2);
    }

    #[test]
    fn test_motion_vector_fractional_parts() {
        let mv = MotionVector::new(17, 10); // .25 pixels in x, .5 pixels in y
        assert_eq!(mv.frac_x(), 1); // 17 & 3 = 1
        assert_eq!(mv.frac_y(), 2); // 10 & 3 = 2
    }

    #[test]
    fn test_motion_vector_scale() {
        let mv = MotionVector::new(16, 8);
        let scaled = mv.scale(2, 1); // Multiply by 2, shift by 1 = multiply by 1
        assert_eq!(scaled.x, 16);
        assert_eq!(scaled.y, 8);

        let scaled2 = mv.scale(4, 1); // Multiply by 4, shift by 1 = multiply by 2
        assert_eq!(scaled2.x, 32);
        assert_eq!(scaled2.y, 16);
    }

    #[test]
    fn test_motion_vector_clip() {
        let mv = MotionVector::new(100, -100);
        let clipped = mv.clip(-50, 50, -60, 60);
        assert_eq!(clipped.x, 50);
        assert_eq!(clipped.y, -60);
    }

    #[test]
    fn test_median3_function() {
        assert_eq!(median3(1, 2, 3), 2);
        assert_eq!(median3(3, 2, 1), 2);
        assert_eq!(median3(1, 3, 2), 2);
        assert_eq!(median3(5, 5, 5), 5);
    }

    #[test]
    fn test_motion_vector_median() {
        let a = MotionVector::new(10, 20);
        let b = MotionVector::new(20, 10);
        let c = MotionVector::new(15, 15);
        let median = MotionVector::median3(a, b, c);
        assert_eq!(median.x, 15);
        assert_eq!(median.y, 15);
    }

    #[test]
    fn test_mv_candidate_creation() {
        let mv = MotionVector::new(8, 4);
        let candidate = MvCandidate::new(mv, 0, PredictionList::L0);
        assert_eq!(candidate.mv, mv);
        assert_eq!(candidate.ref_idx, 0);
        assert_eq!(candidate.pred_list, PredictionList::L0);
    }

    #[test]
    fn test_mv_predictor() {
        let mut predictor = MvPredictor::new();
        assert_eq!(predictor.num_candidates(), 0);

        let candidate = MvCandidate::new(MotionVector::new(16, 8), 0, PredictionList::L0);
        predictor.add_spatial(candidate);
        assert_eq!(predictor.num_candidates(), 1);

        let mv = predictor.get_predictor(0).unwrap();
        assert_eq!(mv.x, 16);
        assert_eq!(mv.y, 8);
    }

    #[test]
    fn test_mv_predictor_empty() {
        let predictor = MvPredictor::new();
        let mv = predictor.get_predictor(0).unwrap();
        assert!(mv.is_zero()); // Should return zero MV when empty
    }

    #[test]
    fn test_merge_candidate_creation() {
        let candidate = MergeCandidate::new(
            MotionVector::new(8, 4),
            MotionVector::new(-8, -4),
            0,
            1,
            PredictionFlag::Bi,
        );
        assert_eq!(candidate.mv_l0.x, 8);
        assert_eq!(candidate.mv_l1.x, -8);
        assert_eq!(candidate.pred_flag, PredictionFlag::Bi);
    }

    #[test]
    fn test_merge_candidate_list() {
        let mut list = MergeCandidateList::new(5);
        assert!(list.is_empty());
        assert!(!list.is_full());

        let candidate = MergeCandidate::unavailable();
        list.add(candidate);
        assert_eq!(list.len(), 1);
        assert!(!list.is_empty());

        let retrieved = list.get(0).unwrap();
        assert_eq!(retrieved.mv_l0, candidate.mv_l0);
    }

    #[test]
    fn test_merge_candidate_list_max() {
        let mut list = MergeCandidateList::new(2);
        list.add(MergeCandidate::unavailable());
        list.add(MergeCandidate::unavailable());
        assert!(list.is_full());

        // Adding more should not increase size
        list.add(MergeCandidate::unavailable());
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_merge_candidate_list_invalid_index() {
        let list = MergeCandidateList::new(5);
        let result = list.get(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_motion_vector_field() {
        let mut field = MotionVectorField::new(64, 64);
        let mv = MotionVector::new(16, 8);

        field.set_mv_l0(0, 0, mv, 0).unwrap();
        let (retrieved_mv, ref_idx) = field.get_mv_l0(0, 0).unwrap();
        assert_eq!(retrieved_mv, mv);
        assert_eq!(ref_idx, 0);
    }

    #[test]
    fn test_motion_vector_field_l1() {
        let mut field = MotionVectorField::new(64, 64);
        let mv = MotionVector::new(-16, -8);

        field.set_mv_l1(5, 3, mv, 1).unwrap();
        let (retrieved_mv, ref_idx) = field.get_mv_l1(5, 3).unwrap();
        assert_eq!(retrieved_mv, mv);
        assert_eq!(ref_idx, 1);
    }

    #[test]
    fn test_motion_vector_field_bounds() {
        let mut field = MotionVectorField::new(64, 64);
        let mv = MotionVector::new(8, 4);

        // Out of bounds should error
        let result = field.set_mv_l0(100, 100, mv, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_prediction_flag_values() {
        assert_eq!(PredictionFlag::L0 as u8, 0);
        assert_eq!(PredictionFlag::L1 as u8, 1);
        assert_eq!(PredictionFlag::Bi as u8, 2);
    }

    #[test]
    fn test_prediction_list_values() {
        assert_eq!(PredictionList::L0 as u8, 0);
        assert_eq!(PredictionList::L1 as u8, 1);
    }
}
