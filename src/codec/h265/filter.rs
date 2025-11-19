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

/// SAO (Sample Adaptive Offset) type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaoType {
    /// No SAO filtering
    None = 0,
    /// Band offset mode
    BandOffset = 1,
    /// Edge offset mode
    EdgeOffset = 2,
}

/// SAO edge offset class
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaoEdgeClass {
    /// Horizontal edge (0°)
    Horizontal = 0,
    /// Vertical edge (90°)
    Vertical = 1,
    /// Diagonal 135° edge
    Diagonal135 = 2,
    /// Diagonal 45° edge
    Diagonal45 = 3,
}

/// SAO edge offset category
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EdgeCategory {
    /// Pixel < both neighbors
    Valley = 0,
    /// Pixel < one neighbor
    Concave = 1,
    /// Pixel == neighbors (no offset)
    None = 2,
    /// Pixel > one neighbor
    Convex = 3,
    /// Pixel > both neighbors
    Peak = 4,
}

/// SAO filter parameters
#[derive(Debug, Clone)]
pub struct SaoParams {
    /// SAO type
    pub sao_type: SaoType,
    /// Edge offset class (for EdgeOffset type)
    pub edge_class: SaoEdgeClass,
    /// Offsets for each category/band
    pub offsets: [i8; 4],
    /// Band position (for BandOffset type)
    pub band_position: u8,
}

impl Default for SaoParams {
    fn default() -> Self {
        Self {
            sao_type: SaoType::None,
            edge_class: SaoEdgeClass::Horizontal,
            offsets: [0; 4],
            band_position: 0,
        }
    }
}

/// SAO (Sample Adaptive Offset) filter for H.265
pub struct SaoFilter {
    /// Bit depth (8, 10, or 12)
    bit_depth: u8,
}

impl SaoFilter {
    /// Create a new SAO filter
    pub fn new(bit_depth: u8) -> Result<Self> {
        if bit_depth != 8 && bit_depth != 10 && bit_depth != 12 {
            return Err(Error::InvalidData(format!(
                "Invalid bit depth: {}",
                bit_depth
            )));
        }

        Ok(Self { bit_depth })
    }

    /// Apply SAO filter to a CTU
    pub fn apply(
        &self,
        samples: &mut [u16],
        width: usize,
        height: usize,
        stride: usize,
        params: &SaoParams,
    ) -> Result<()> {
        match params.sao_type {
            SaoType::None => Ok(()),
            SaoType::BandOffset => self.apply_band_offset(samples, width, height, stride, params),
            SaoType::EdgeOffset => self.apply_edge_offset(samples, width, height, stride, params),
        }
    }

    /// Apply band offset SAO
    ///
    /// Divides sample values into 32 bands and applies offsets to 4 consecutive bands.
    fn apply_band_offset(
        &self,
        samples: &mut [u16],
        width: usize,
        height: usize,
        stride: usize,
        params: &SaoParams,
    ) -> Result<()> {
        // H.265 uses 32 bands for sample values
        let num_bands = 32;
        let band_shift = self.bit_depth - 5; // Map to 0-31 range
        let start_band = params.band_position as usize;

        for y in 0..height {
            for x in 0..width {
                let idx = y * stride + x;
                if idx >= samples.len() {
                    continue;
                }

                let sample = samples[idx];
                let band_idx = (sample >> band_shift) as usize;

                // Check if sample is in one of the 4 offset bands
                if band_idx >= start_band && band_idx < start_band + 4 {
                    let offset_idx = band_idx - start_band;
                    let offset = params.offsets[offset_idx] as i32;

                    let new_sample = (sample as i32 + offset).clamp(0, (1 << self.bit_depth) - 1);
                    samples[idx] = new_sample as u16;
                }
            }
        }

        Ok(())
    }

    /// Apply edge offset SAO
    ///
    /// Classifies pixels based on edge category and applies offsets.
    fn apply_edge_offset(
        &self,
        samples: &mut [u16],
        width: usize,
        height: usize,
        stride: usize,
        params: &SaoParams,
    ) -> Result<()> {
        // Get neighbor offsets based on edge class
        let (offset1, offset2) = match params.edge_class {
            SaoEdgeClass::Horizontal => (-(stride as isize), stride as isize),    // top, bottom
            SaoEdgeClass::Vertical => (-1isize, 1isize),                           // left, right
            SaoEdgeClass::Diagonal135 => (-(stride as isize) - 1, stride as isize + 1), // top-left, bottom-right
            SaoEdgeClass::Diagonal45 => (-(stride as isize) + 1, stride as isize - 1),  // top-right, bottom-left
        };

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = y * stride + x;
                if idx >= samples.len() {
                    continue;
                }

                // Get neighbor indices
                let idx1 = (idx as isize + offset1) as usize;
                let idx2 = (idx as isize + offset2) as usize;

                if idx1 >= samples.len() || idx2 >= samples.len() {
                    continue;
                }

                let sample = samples[idx] as i32;
                let neighbor1 = samples[idx1] as i32;
                let neighbor2 = samples[idx2] as i32;

                // Classify edge category
                let category = self.classify_edge(sample, neighbor1, neighbor2);

                if category != EdgeCategory::None {
                    let offset = params.offsets[category as usize] as i32;
                    let new_sample = (sample + offset).clamp(0, (1 << self.bit_depth) - 1);
                    samples[idx] = new_sample as u16;
                }
            }
        }

        Ok(())
    }

    /// Classify edge category based on pixel and its neighbors
    fn classify_edge(&self, sample: i32, neighbor1: i32, neighbor2: i32) -> EdgeCategory {
        let sign1 = (sample - neighbor1).signum();
        let sign2 = (sample - neighbor2).signum();

        match (sign1, sign2) {
            (-1, -1) => EdgeCategory::Valley,   // < both neighbors
            (-1, 0) | (0, -1) | (-1, 1) | (1, -1) => EdgeCategory::Concave, // < one neighbor
            (0, 0) => EdgeCategory::None,       // == neighbors
            (1, 0) | (0, 1) | (1, -1) | (-1, 1) => EdgeCategory::Convex,    // > one neighbor
            (1, 1) => EdgeCategory::Peak,       // > both neighbors
            _ => EdgeCategory::None,
        }
    }

    /// Clip value to valid sample range
    #[inline]
    fn clip(&self, value: i32) -> u16 {
        let max = (1 << self.bit_depth) - 1;
        value.clamp(0, max) as u16
    }
}

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

    // SAO Filter Tests

    #[test]
    fn test_sao_filter_creation() {
        let filter = SaoFilter::new(8);
        assert!(filter.is_ok());
    }

    #[test]
    fn test_sao_filter_invalid_bit_depth() {
        let filter = SaoFilter::new(9);
        assert!(filter.is_err());
    }

    #[test]
    fn test_sao_params_default() {
        let params = SaoParams::default();
        assert_eq!(params.sao_type, SaoType::None);
        assert_eq!(params.offsets, [0; 4]);
    }

    #[test]
    fn test_sao_none_no_change() {
        let filter = SaoFilter::new(8).unwrap();
        let mut samples = vec![128u16; 64]; // 8×8 block
        let original = samples.clone();

        let params = SaoParams::default(); // Type::None
        let result = filter.apply(&mut samples, 8, 8, 8, &params);

        assert!(result.is_ok());
        assert_eq!(samples, original);
    }

    #[test]
    fn test_sao_band_offset() {
        let filter = SaoFilter::new(8).unwrap();
        let mut samples = vec![128u16; 64]; // 8×8 block

        let params = SaoParams {
            sao_type: SaoType::BandOffset,
            edge_class: SaoEdgeClass::Horizontal,
            offsets: [5, -3, 2, -1],
            band_position: 4, // Band 4-7
        };

        let result = filter.apply(&mut samples, 8, 8, 8, &params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sao_edge_offset_horizontal() {
        let filter = SaoFilter::new(8).unwrap();
        let mut samples = vec![128u16; 64]; // 8×8 block

        let params = SaoParams {
            sao_type: SaoType::EdgeOffset,
            edge_class: SaoEdgeClass::Horizontal,
            offsets: [2, 1, 0, -1],
            band_position: 0,
        };

        let result = filter.apply(&mut samples, 8, 8, 8, &params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sao_edge_offset_vertical() {
        let filter = SaoFilter::new(8).unwrap();
        let mut samples = vec![100u16; 100]; // 10×10 block

        let params = SaoParams {
            sao_type: SaoType::EdgeOffset,
            edge_class: SaoEdgeClass::Vertical,
            offsets: [3, 1, 0, -2],
            band_position: 0,
        };

        let result = filter.apply(&mut samples, 10, 10, 10, &params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sao_edge_offset_diagonal135() {
        let filter = SaoFilter::new(8).unwrap();
        let mut samples = vec![128u16; 64];

        let params = SaoParams {
            sao_type: SaoType::EdgeOffset,
            edge_class: SaoEdgeClass::Diagonal135,
            offsets: [2, 1, 0, -1],
            band_position: 0,
        };

        let result = filter.apply(&mut samples, 8, 8, 8, &params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sao_edge_offset_diagonal45() {
        let filter = SaoFilter::new(8).unwrap();
        let mut samples = vec![128u16; 64];

        let params = SaoParams {
            sao_type: SaoType::EdgeOffset,
            edge_class: SaoEdgeClass::Diagonal45,
            offsets: [2, 1, 0, -1],
            band_position: 0,
        };

        let result = filter.apply(&mut samples, 8, 8, 8, &params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sao_edge_classification() {
        let filter = SaoFilter::new(8).unwrap();

        // Valley: sample < both neighbors
        assert_eq!(filter.classify_edge(100, 110, 120), EdgeCategory::Valley);

        // Peak: sample > both neighbors
        assert_eq!(filter.classify_edge(150, 110, 120), EdgeCategory::Peak);

        // None: sample == neighbors
        assert_eq!(filter.classify_edge(100, 100, 100), EdgeCategory::None);
    }

    #[test]
    fn test_sao_band_offset_10bit() {
        let filter = SaoFilter::new(10).unwrap();
        let mut samples = vec![512u16; 64]; // 10-bit samples

        let params = SaoParams {
            sao_type: SaoType::BandOffset,
            edge_class: SaoEdgeClass::Horizontal,
            offsets: [10, -5, 3, -2],
            band_position: 8,
        };

        let result = filter.apply(&mut samples, 8, 8, 8, &params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sao_type_values() {
        assert_eq!(SaoType::None as u8, 0);
        assert_eq!(SaoType::BandOffset as u8, 1);
        assert_eq!(SaoType::EdgeOffset as u8, 2);
    }

    #[test]
    fn test_sao_edge_class_values() {
        assert_eq!(SaoEdgeClass::Horizontal as u8, 0);
        assert_eq!(SaoEdgeClass::Vertical as u8, 1);
        assert_eq!(SaoEdgeClass::Diagonal135 as u8, 2);
        assert_eq!(SaoEdgeClass::Diagonal45 as u8, 3);
    }

    #[test]
    fn test_sao_clip_8bit() {
        let filter = SaoFilter::new(8).unwrap();
        assert_eq!(filter.clip(100), 100);
        assert_eq!(filter.clip(255), 255);
        assert_eq!(filter.clip(300), 255);
        assert_eq!(filter.clip(-10), 0);
    }

    #[test]
    fn test_sao_clip_10bit() {
        let filter = SaoFilter::new(10).unwrap();
        assert_eq!(filter.clip(1023), 1023);
        assert_eq!(filter.clip(1500), 1023);
        assert_eq!(filter.clip(-5), 0);
    }
}
