//! Motion Estimation (ME) for H.265/HEVC Encoder
//!
//! This module implements motion vector search algorithms to find the best
//! motion vector for inter prediction.
//!
//! # Motion Estimation Process
//!
//! 1. Start from predictor (AMVP or merge candidate)
//! 2. Integer-pixel search (full search or diamond search)
//! 3. Fractional-pixel refinement (1/2-pixel, then 1/4-pixel)
//! 4. Calculate final MV and cost
//!
//! # Search Algorithms
//!
//! - **Full Search**: Exhaustive search in range, accurate but slow
//! - **Diamond Search**: Fast hierarchical pattern search
//! - **AMVP-based**: Start from predicted MV, refine locally
//!
//! # Cost Function
//!
//! **Cost = SAD + λ_ME × MVD_bits**
//!
//! Where:
//! - SAD: Sum of Absolute Differences
//! - λ_ME: Motion estimation lambda (sqrt of RDO lambda)
//! - MVD_bits: Bits to code MV difference from predictor

use crate::codec::h265::mv::MotionVector;
use crate::codec::h265::rdo::{DistortionCalc, LambdaCalc, RateEstimator};
use crate::error::{Error, Result};

/// Motion estimation result
#[derive(Debug, Clone, Copy)]
pub struct MeResult {
    /// Best motion vector found (1/4-pixel precision)
    pub mv: MotionVector,
    /// Motion vector predictor used
    pub mvp: MotionVector,
    /// Best cost (SAD + lambda * bits)
    pub cost: u64,
    /// SAD distortion only
    pub sad: u64,
}

/// Motion estimation search range
#[derive(Debug, Clone, Copy)]
pub struct SearchRange {
    /// Horizontal search range (±pixels)
    pub range_x: i16,
    /// Vertical search range (±pixels)
    pub range_y: i16,
}

impl SearchRange {
    /// Create a new search range
    pub fn new(range_x: i16, range_y: i16) -> Self {
        Self { range_x, range_y }
    }

    /// Standard search range for P-frames
    pub fn standard_p() -> Self {
        Self::new(64, 64)
    }

    /// Fast search range (smaller)
    pub fn fast() -> Self {
        Self::new(32, 32)
    }
}

/// Motion estimator
pub struct MotionEstimator {
    /// Lambda for motion estimation
    lambda: f64,
    /// Search range
    range: SearchRange,
    /// Block width
    width: usize,
    /// Block height
    height: usize,
}

impl MotionEstimator {
    /// Create a new motion estimator
    pub fn new(qp: u8, range: SearchRange, width: usize, height: usize) -> Self {
        let lambda = LambdaCalc::calculate_lambda_me(qp);

        Self {
            lambda,
            range,
            width,
            height,
        }
    }

    /// Perform full search motion estimation
    ///
    /// Exhaustively tests all positions in search range
    pub fn full_search(
        &self,
        cur_block: &[u16],
        cur_stride: usize,
        ref_frame: &[u16],
        ref_stride: usize,
        ref_width: usize,
        ref_height: usize,
        center_x: usize,
        center_y: usize,
        mvp: MotionVector,
    ) -> Result<MeResult> {
        let mut best_cost = u64::MAX;
        let mut best_mv = mvp;
        let mut best_sad = u64::MAX;

        // Search all positions in range
        for dy in -self.range.range_y..=self.range.range_y {
            for dx in -self.range.range_x..=self.range.range_x {
                let search_x = center_x as i32 + dx as i32;
                let search_y = center_y as i32 + dy as i32;

                // Check bounds
                if search_x < 0
                    || search_y < 0
                    || search_x + self.width as i32 > ref_width as i32
                    || search_y + self.height as i32 > ref_height as i32
                {
                    continue;
                }

                // Calculate SAD
                let sad = self.calculate_sad(
                    cur_block,
                    cur_stride,
                    ref_frame,
                    ref_stride,
                    search_x as usize,
                    search_y as usize,
                );

                // Calculate MV and MVD
                let mv = MotionVector::new(dx * 4, dy * 4); // Convert to 1/4-pixel
                let mvd_x = mv.x - mvp.x;
                let mvd_y = mv.y - mvp.y;

                // Calculate cost
                let bits = RateEstimator::estimate_mvd_bits(mvd_x, mvd_y);
                let cost = sad + (self.lambda * bits as f64) as u64;

                if cost < best_cost {
                    best_cost = cost;
                    best_mv = mv;
                    best_sad = sad;
                }
            }
        }

        Ok(MeResult {
            mv: best_mv,
            mvp,
            cost: best_cost,
            sad: best_sad,
        })
    }

    /// Perform diamond search
    ///
    /// Fast hierarchical pattern-based search
    pub fn diamond_search(
        &self,
        cur_block: &[u16],
        cur_stride: usize,
        ref_frame: &[u16],
        ref_stride: usize,
        ref_width: usize,
        ref_height: usize,
        center_x: usize,
        center_y: usize,
        mvp: MotionVector,
    ) -> Result<MeResult> {
        // Start from predictor position
        let mut best_x = mvp.integer_x() as i32;
        let mut best_y = mvp.integer_y() as i32;

        let mut best_sad = self.calculate_sad_at(
            cur_block,
            cur_stride,
            ref_frame,
            ref_stride,
            ref_width,
            ref_height,
            center_x,
            center_y,
            best_x,
            best_y,
        );

        let mut step_size = 8; // Start with large step

        // Large diamond pattern
        while step_size >= 1 {
            let improved = self.diamond_step(
                cur_block,
                cur_stride,
                ref_frame,
                ref_stride,
                ref_width,
                ref_height,
                center_x,
                center_y,
                &mut best_x,
                &mut best_y,
                &mut best_sad,
                step_size,
            );

            if !improved {
                step_size /= 2; // Reduce step size
            }
        }

        // Convert to MV
        let best_mv = MotionVector::new(best_x as i16 * 4, best_y as i16 * 4);
        let mvd_x = best_mv.x - mvp.x;
        let mvd_y = best_mv.y - mvp.y;
        let bits = RateEstimator::estimate_mvd_bits(mvd_x, mvd_y);
        let cost = best_sad + (self.lambda * bits as f64) as u64;

        Ok(MeResult {
            mv: best_mv,
            mvp,
            cost,
            sad: best_sad,
        })
    }

    /// Single diamond search step
    fn diamond_step(
        &self,
        cur_block: &[u16],
        cur_stride: usize,
        ref_frame: &[u16],
        ref_stride: usize,
        ref_width: usize,
        ref_height: usize,
        center_x: usize,
        center_y: usize,
        best_x: &mut i32,
        best_y: &mut i32,
        best_sad: &mut u64,
        step: i32,
    ) -> bool {
        let diamond_pattern = [(0, -step), (-step, 0), (step, 0), (0, step)];

        let mut improved = false;

        for (dx, dy) in diamond_pattern.iter() {
            let test_x = *best_x + dx;
            let test_y = *best_y + dy;

            let sad = self.calculate_sad_at(
                cur_block,
                cur_stride,
                ref_frame,
                ref_stride,
                ref_width,
                ref_height,
                center_x,
                center_y,
                test_x,
                test_y,
            );

            if sad < *best_sad {
                *best_x = test_x;
                *best_y = test_y;
                *best_sad = sad;
                improved = true;
            }
        }

        improved
    }

    /// Calculate SAD at specific MV position
    fn calculate_sad_at(
        &self,
        cur_block: &[u16],
        cur_stride: usize,
        ref_frame: &[u16],
        ref_stride: usize,
        ref_width: usize,
        ref_height: usize,
        center_x: usize,
        center_y: usize,
        mv_x: i32,
        mv_y: i32,
    ) -> u64 {
        let ref_x = center_x as i32 + mv_x;
        let ref_y = center_y as i32 + mv_y;

        // Check bounds
        if ref_x < 0
            || ref_y < 0
            || ref_x + self.width as i32 > ref_width as i32
            || ref_y + self.height as i32 > ref_height as i32
        {
            return u64::MAX;
        }

        self.calculate_sad(
            cur_block,
            cur_stride,
            ref_frame,
            ref_stride,
            ref_x as usize,
            ref_y as usize,
        )
    }

    /// Calculate SAD between current block and reference block
    fn calculate_sad(
        &self,
        cur_block: &[u16],
        cur_stride: usize,
        ref_frame: &[u16],
        ref_stride: usize,
        ref_x: usize,
        ref_y: usize,
    ) -> u64 {
        let mut sad = 0u64;

        for y in 0..self.height {
            for x in 0..self.width {
                let cur_idx = y * cur_stride + x;
                let ref_idx = (ref_y + y) * ref_stride + (ref_x + x);

                sad += (cur_block[cur_idx] as i32 - ref_frame[ref_idx] as i32).abs() as u64;
            }
        }

        sad
    }
}

/// Fractional-pixel motion estimation refinement
pub struct SubpelRefinement {
    /// Lambda for cost calculation
    lambda: f64,
}

impl SubpelRefinement {
    /// Create new subpel refinement
    pub fn new(qp: u8) -> Self {
        let lambda = LambdaCalc::calculate_lambda_me(qp);
        Self { lambda }
    }

    /// Refine integer MV to 1/4-pixel precision
    ///
    /// Tests 1/2-pixel positions around best integer MV,
    /// then 1/4-pixel positions around best 1/2-pixel MV
    pub fn refine(
        &self,
        integer_mv: MotionVector,
        _mvp: MotionVector,
        // In real implementation would need MC predictor and reference frame
    ) -> MotionVector {
        // Placeholder: just return integer MV
        // Real implementation would test half-pel and quarter-pel positions
        integer_mv
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_range_creation() {
        let range = SearchRange::new(64, 64);
        assert_eq!(range.range_x, 64);
        assert_eq!(range.range_y, 64);
    }

    #[test]
    fn test_search_range_presets() {
        let standard = SearchRange::standard_p();
        assert_eq!(standard.range_x, 64);

        let fast = SearchRange::fast();
        assert_eq!(fast.range_x, 32);
    }

    #[test]
    fn test_motion_estimator_creation() {
        let me = MotionEstimator::new(24, SearchRange::fast(), 16, 16);
        assert!(me.lambda > 0.0);
        assert_eq!(me.width, 16);
        assert_eq!(me.height, 16);
    }

    #[test]
    fn test_calculate_sad_identical() {
        let me = MotionEstimator::new(24, SearchRange::fast(), 8, 8);
        let block = vec![128u16; 64];

        let sad = me.calculate_sad(&block, 8, &block, 8, 0, 0);
        assert_eq!(sad, 0);
    }

    #[test]
    fn test_calculate_sad() {
        let me = MotionEstimator::new(24, SearchRange::fast(), 8, 8);
        let cur_block = vec![100u16; 64];
        let ref_block = vec![105u16; 64];

        let sad = me.calculate_sad(&cur_block, 8, &ref_block, 8, 0, 0);
        assert_eq!(sad, 5 * 64);
    }

    #[test]
    fn test_me_result_creation() {
        let result = MeResult {
            mv: MotionVector::new(16, 8),
            mvp: MotionVector::new(12, 6),
            cost: 1000,
            sad: 900,
        };

        assert_eq!(result.mv.x, 16);
        assert_eq!(result.cost, 1000);
    }

    #[test]
    fn test_diamond_search_same_block() {
        let me = MotionEstimator::new(24, SearchRange::fast(), 16, 16);
        let block = vec![128u16; 256];
        let mvp = MotionVector::zero();

        let result = me.diamond_search(&block, 16, &block, 16, 64, 64, 0, 0, mvp);
        assert!(result.is_ok());

        let result = result.unwrap();
        // Should find MV at (0,0) since blocks are identical
        assert_eq!(result.sad, 0);
    }

    #[test]
    fn test_subpel_refinement() {
        let refine = SubpelRefinement::new(24);
        let int_mv = MotionVector::new(16, 8);
        let mvp = MotionVector::zero();

        let refined = refine.refine(int_mv, mvp);
        // Placeholder implementation just returns input
        assert_eq!(refined, int_mv);
    }
}
