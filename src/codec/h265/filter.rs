//! In-loop filters for H.265/HEVC
//!
//! This module implements the in-loop filters used in H.265 to improve
//! reconstructed picture quality:
//!
//! - **Deblocking Filter**: Reduces blocking artifacts at block boundaries
//! - **SAO (Sample Adaptive Offset)**: Reduces ringing and banding artifacts
//!
//! These filters are applied in the decoding loop, so their output is used
//! for both display and motion compensation reference.

use crate::error::{Error, Result};

/// Boundary strength values for deblocking filter
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryStrength {
    /// No filtering
    None = 0,
    /// Weak filtering
    Weak = 1,
    /// Strong filtering (intra blocks or large motion difference)
    Strong = 2,
}

/// Deblocking filter for H.265
pub struct DeblockingFilter {
    /// Bit depth (8, 10, or 12)
    bit_depth: u8,
    /// Beta offset for threshold calculation
    beta_offset: i8,
    /// Tc (threshold) offset
    tc_offset: i8,
}

impl DeblockingFilter {
    /// Create a new deblocking filter
    pub fn new(bit_depth: u8) -> Result<Self> {
        if bit_depth != 8 && bit_depth != 10 && bit_depth != 12 {
            return Err(Error::InvalidData(format!(
                "Invalid bit depth: {}",
                bit_depth
            )));
        }

        Ok(Self {
            bit_depth,
            beta_offset: 0,
            tc_offset: 0,
        })
    }

    /// Set beta offset (-6 to 6)
    pub fn set_beta_offset(&mut self, offset: i8) -> Result<()> {
        if offset < -6 || offset > 6 {
            return Err(Error::InvalidData(format!(
                "Beta offset must be -6 to 6, got {}",
                offset
            )));
        }
        self.beta_offset = offset;
        Ok(())
    }

    /// Set tc offset (-6 to 6)
    pub fn set_tc_offset(&mut self, offset: i8) -> Result<()> {
        if offset < -6 || offset > 6 {
            return Err(Error::InvalidData(format!(
                "Tc offset must be -6 to 6, got {}",
                offset
            )));
        }
        self.tc_offset = offset;
        Ok(())
    }

    /// Apply deblocking filter to a vertical edge
    ///
    /// Filters the boundary between two 4-pixel columns.
    /// `samples` contains pixels on both sides of the boundary.
    /// p3 p2 p1 p0 | q0 q1 q2 q3 (vertical edge)
    pub fn filter_vertical_edge(
        &self,
        samples: &mut [u16],
        stride: usize,
        qp: u8,
        bs: BoundaryStrength,
    ) -> Result<()> {
        if bs == BoundaryStrength::None {
            return Ok(());
        }

        // Check if we have enough samples (need 4 on each side, 4 rows minimum)
        if samples.len() < stride * 4 {
            return Err(Error::InvalidData("Insufficient samples for filtering".to_string()));
        }

        // Get thresholds
        let beta = self.get_beta(qp)?;
        let tc = self.get_tc(qp, bs)?;

        // Filter each row
        for y in 0..4.min(samples.len() / stride) {
            let offset = y * stride;

            // Get pixels: p3 p2 p1 p0 | q0 q1 q2 q3
            if offset + 7 >= samples.len() {
                continue;
            }

            let p3 = samples[offset] as i32;
            let p2 = samples[offset + 1] as i32;
            let p1 = samples[offset + 2] as i32;
            let p0 = samples[offset + 3] as i32;
            let q0 = samples[offset + 4] as i32;
            let q1 = samples[offset + 5] as i32;
            let q2 = samples[offset + 6] as i32;
            let q3 = samples[offset + 7] as i32;

            // Decision to filter based on gradient
            let dp0 = (p2 - 2 * p1 + p0).abs();
            let dq0 = (q2 - 2 * q1 + q0).abs();
            let dp = dp0 + dq0;
            let d = (p0 - q0).abs();

            if d >= beta || dp >= (beta >> 2) {
                continue; // Skip filtering for this position
            }

            // Determine strong or weak filtering
            let use_strong = bs == BoundaryStrength::Strong
                && (d < (beta >> 2))
                && (p0 - p3).abs() < (beta >> 3)
                && (q0 - q3).abs() < (beta >> 3);

            if use_strong {
                // Strong filtering
                let p0_new = (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3;
                let q0_new = (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3;

                samples[offset + 3] = self.clip(p0_new);
                samples[offset + 4] = self.clip(q0_new);

                // Filter p1, q1 if gradient is small enough
                let dp1 = (p2 - p1).abs();
                let dq1 = (q2 - q1).abs();

                if dp1 < (beta + (beta >> 1)) >> 3 {
                    let p1_new = (p2 + p1 + p0 + q0 + 2) >> 2;
                    samples[offset + 2] = self.clip(p1_new);
                }

                if dq1 < (beta + (beta >> 1)) >> 3 {
                    let q1_new = (p0 + q0 + q1 + q2 + 2) >> 2;
                    samples[offset + 5] = self.clip(q1_new);
                }
            } else {
                // Weak filtering
                let delta = ((9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4)
                    .clamp(-tc, tc);

                let p0_new = p0 + delta;
                let q0_new = q0 - delta;

                samples[offset + 3] = self.clip(p0_new);
                samples[offset + 4] = self.clip(q0_new);
            }
        }

        Ok(())
    }

    /// Apply deblocking filter to a horizontal edge
    ///
    /// Filters the boundary between two 4-pixel rows.
    pub fn filter_horizontal_edge(
        &self,
        samples: &mut [u16],
        stride: usize,
        qp: u8,
        bs: BoundaryStrength,
    ) -> Result<()> {
        if bs == BoundaryStrength::None {
            return Ok(());
        }

        // Need at least 8 rows for filtering
        if samples.len() < stride * 8 {
            return Err(Error::InvalidData("Insufficient samples for filtering".to_string()));
        }

        let beta = self.get_beta(qp)?;
        let tc = self.get_tc(qp, bs)?;

        // Filter each column
        for x in 0..4.min(stride) {
            // Get pixels in column
            // p3 (row 0)
            // p2 (row 1)
            // p1 (row 2)
            // p0 (row 3)
            // -- boundary --
            // q0 (row 4)
            // q1 (row 5)
            // q2 (row 6)
            // q3 (row 7)

            if x + stride * 7 >= samples.len() {
                continue;
            }

            let p3 = samples[x] as i32;
            let p2 = samples[x + stride] as i32;
            let p1 = samples[x + stride * 2] as i32;
            let p0 = samples[x + stride * 3] as i32;
            let q0 = samples[x + stride * 4] as i32;
            let q1 = samples[x + stride * 5] as i32;
            let q2 = samples[x + stride * 6] as i32;
            let q3 = samples[x + stride * 7] as i32;

            // Decision to filter
            let dp0 = (p2 - 2 * p1 + p0).abs();
            let dq0 = (q2 - 2 * q1 + q0).abs();
            let dp = dp0 + dq0;
            let d = (p0 - q0).abs();

            if d >= beta || dp >= (beta >> 2) {
                continue;
            }

            // Determine filtering strength
            let use_strong = bs == BoundaryStrength::Strong
                && (d < (beta >> 2))
                && (p0 - p3).abs() < (beta >> 3)
                && (q0 - q3).abs() < (beta >> 3);

            if use_strong {
                // Strong filtering
                let p0_new = (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3;
                let q0_new = (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3;

                samples[x + stride * 3] = self.clip(p0_new);
                samples[x + stride * 4] = self.clip(q0_new);
            } else {
                // Weak filtering
                let delta = ((9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4)
                    .clamp(-tc, tc);

                let p0_new = p0 + delta;
                let q0_new = q0 - delta;

                samples[x + stride * 3] = self.clip(p0_new);
                samples[x + stride * 4] = self.clip(q0_new);
            }
        }

        Ok(())
    }

    /// Get beta threshold value based on QP
    fn get_beta(&self, qp: u8) -> Result<i32> {
        let qp_adjusted = (qp as i32 + self.beta_offset as i32).clamp(0, 51) as usize;

        if qp_adjusted >= BETA_TABLE.len() {
            return Err(Error::InvalidData(format!("Invalid QP: {}", qp)));
        }

        let beta = BETA_TABLE[qp_adjusted];

        // Scale for bit depth
        let shift = self.bit_depth - 8;
        Ok((beta as i32) << shift)
    }

    /// Get tc (threshold) value based on QP and boundary strength
    fn get_tc(&self, qp: u8, bs: BoundaryStrength) -> Result<i32> {
        let qp_adjusted = (qp as i32 + self.tc_offset as i32 + 2).clamp(0, 53) as usize;

        if qp_adjusted >= TC_TABLE.len() {
            return Err(Error::InvalidData(format!("Invalid QP: {}", qp)));
        }

        let tc = TC_TABLE[qp_adjusted];

        // Scale for boundary strength and bit depth
        let tc_scaled = if bs == BoundaryStrength::Strong {
            tc + 2
        } else {
            tc
        };

        let shift = self.bit_depth - 8;
        Ok((tc_scaled as i32) << shift)
    }

    /// Clip value to valid sample range
    #[inline]
    fn clip(&self, value: i32) -> u16 {
        let max = (1 << self.bit_depth) - 1;
        value.clamp(0, max) as u16
    }
}

/// Beta table for deblocking filter threshold calculation
/// Indexed by QP (0-51)
const BETA_TABLE: [u8; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24,
    26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56,
    58, 60, 62, 64,
];

/// Tc (threshold) table for deblocking filter
/// Indexed by QP + offset (0-53)
const TC_TABLE: [u8; 54] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3,
    3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 13,
    14, 16, 18, 20, 22, 24,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deblocking_filter_creation() {
        let filter = DeblockingFilter::new(8);
        assert!(filter.is_ok());
    }

    #[test]
    fn test_deblocking_filter_invalid_bit_depth() {
        let filter = DeblockingFilter::new(9);
        assert!(filter.is_err());
    }

    #[test]
    fn test_set_beta_offset() {
        let mut filter = DeblockingFilter::new(8).unwrap();
        assert!(filter.set_beta_offset(3).is_ok());
        assert!(filter.set_beta_offset(-6).is_ok());
        assert!(filter.set_beta_offset(6).is_ok());
        assert!(filter.set_beta_offset(7).is_err());
    }

    #[test]
    fn test_set_tc_offset() {
        let mut filter = DeblockingFilter::new(8).unwrap();
        assert!(filter.set_tc_offset(3).is_ok());
        assert!(filter.set_tc_offset(-6).is_ok());
        assert!(filter.set_tc_offset(6).is_ok());
        assert!(filter.set_tc_offset(7).is_err());
    }

    #[test]
    fn test_filter_vertical_edge_no_filtering() {
        let filter = DeblockingFilter::new(8).unwrap();
        let mut samples = vec![128u16; 32]; // 4x8 samples
        let original = samples.clone();

        let result = filter.filter_vertical_edge(
            &mut samples,
            8,
            26,
            BoundaryStrength::None,
        );

        assert!(result.is_ok());
        assert_eq!(samples, original); // Should be unchanged
    }

    #[test]
    fn test_filter_vertical_edge_weak() {
        let filter = DeblockingFilter::new(8).unwrap();
        // Create samples with a boundary discontinuity
        let mut samples = vec![
            100, 110, 120, 130, 140, 150, 160, 170, // Row 0
            100, 110, 120, 130, 140, 150, 160, 170, // Row 1
            100, 110, 120, 130, 140, 150, 160, 170, // Row 2
            100, 110, 120, 130, 140, 150, 160, 170, // Row 3
        ];

        let result = filter.filter_vertical_edge(
            &mut samples,
            8,
            26,
            BoundaryStrength::Weak,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_filter_horizontal_edge_no_filtering() {
        let filter = DeblockingFilter::new(8).unwrap();
        let mut samples = vec![128u16; 64]; // 8x8 samples
        let original = samples.clone();

        let result = filter.filter_horizontal_edge(
            &mut samples,
            8,
            26,
            BoundaryStrength::None,
        );

        assert!(result.is_ok());
        assert_eq!(samples, original);
    }

    #[test]
    fn test_filter_horizontal_edge_strong() {
        let filter = DeblockingFilter::new(8).unwrap();
        let mut samples = vec![128u16; 64]; // 8x8 samples

        let result = filter.filter_horizontal_edge(
            &mut samples,
            8,
            20,
            BoundaryStrength::Strong,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_boundary_strength_values() {
        assert_eq!(BoundaryStrength::None as u8, 0);
        assert_eq!(BoundaryStrength::Weak as u8, 1);
        assert_eq!(BoundaryStrength::Strong as u8, 2);
    }

    #[test]
    fn test_beta_table_length() {
        assert_eq!(BETA_TABLE.len(), 52);
    }

    #[test]
    fn test_tc_table_length() {
        assert_eq!(TC_TABLE.len(), 54);
    }

    #[test]
    fn test_beta_increases_with_qp() {
        let filter = DeblockingFilter::new(8).unwrap();
        let beta_low = filter.get_beta(10).unwrap();
        let beta_high = filter.get_beta(40).unwrap();
        assert!(beta_high > beta_low);
    }

    #[test]
    fn test_tc_increases_with_qp() {
        let filter = DeblockingFilter::new(8).unwrap();
        let tc_low = filter.get_tc(10, BoundaryStrength::Weak).unwrap();
        let tc_high = filter.get_tc(40, BoundaryStrength::Weak).unwrap();
        assert!(tc_high >= tc_low);
    }

    #[test]
    fn test_clip_within_range() {
        let filter = DeblockingFilter::new(8).unwrap();
        assert_eq!(filter.clip(100), 100);
        assert_eq!(filter.clip(255), 255);
    }

    #[test]
    fn test_clip_clamps_high() {
        let filter = DeblockingFilter::new(8).unwrap();
        assert_eq!(filter.clip(300), 255);
    }

    #[test]
    fn test_clip_clamps_low() {
        let filter = DeblockingFilter::new(8).unwrap();
        assert_eq!(filter.clip(-10), 0);
    }

    #[test]
    fn test_clip_10_bit() {
        let filter = DeblockingFilter::new(10).unwrap();
        assert_eq!(filter.clip(1023), 1023);
        assert_eq!(filter.clip(1500), 1023);
    }

    #[test]
    fn test_different_bit_depths() {
        for bit_depth in [8, 10, 12] {
            let filter = DeblockingFilter::new(bit_depth);
            assert!(filter.is_ok());
        }
    }
}
