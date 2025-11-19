//! Motion Compensation for H.265/HEVC
//!
//! This module implements motion compensation with fractional-pixel interpolation
//! for inter-frame prediction in H.265.
//!
//! # Interpolation
//!
//! H.265 uses separable 2D interpolation:
//! - **Luma**: 8-tap filters for 1/2-pixel, 7-tap for 1/4-pixel positions
//! - **Chroma**: 4-tap filters for all fractional positions
//! - **Precision**: 1/4-pixel for luma and chroma
//!
//! # Process
//!
//! 1. Apply horizontal filter to generate intermediate values
//! 2. Apply vertical filter to intermediate values
//! 3. For bi-directional prediction, average L0 and L1 predictions
//!
//! # Filter Coefficients
//!
//! Filters are designed to minimize aliasing while maintaining sharpness.
//! Different filters are used for different fractional positions.

use crate::codec::h265::mv::{MotionVector, PredictionFlag};
use crate::error::{Error, Result};

/// Motion compensation engine
pub struct MotionCompensator {
    /// Bit depth (8, 10, or 12)
    bit_depth: u8,
    /// Maximum sample value
    max_val: u16,
}

impl MotionCompensator {
    /// Create a new motion compensator
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

    /// Perform motion compensation for a luma block
    ///
    /// # Arguments
    ///
    /// * `ref_pic` - Reference picture samples
    /// * `ref_stride` - Stride of reference picture
    /// * `mv` - Motion vector (1/4-pixel precision)
    /// * `dst` - Destination buffer
    /// * `dst_stride` - Destination stride
    /// * `width` - Block width
    /// * `height` - Block height
    pub fn mc_luma(
        &self,
        ref_pic: &[u16],
        ref_stride: usize,
        mv: MotionVector,
        dst: &mut [u16],
        dst_stride: usize,
        width: usize,
        height: usize,
    ) -> Result<()> {
        let frac_x = mv.frac_x();
        let frac_y = mv.frac_y();

        // Integer position offset
        let int_x = mv.integer_x() as isize;
        let int_y = mv.integer_y() as isize;

        if frac_x == 0 && frac_y == 0 {
            // Integer position: direct copy
            self.copy_block(ref_pic, ref_stride, int_x, int_y, dst, dst_stride, width, height)?;
        } else if frac_y == 0 {
            // Horizontal interpolation only
            self.luma_h_interp(
                ref_pic, ref_stride, int_x, int_y, frac_x, dst, dst_stride, width, height,
            )?;
        } else if frac_x == 0 {
            // Vertical interpolation only
            self.luma_v_interp(
                ref_pic, ref_stride, int_x, int_y, frac_y, dst, dst_stride, width, height,
            )?;
        } else {
            // 2D interpolation (horizontal then vertical)
            self.luma_hv_interp(
                ref_pic, ref_stride, int_x, int_y, frac_x, frac_y, dst, dst_stride, width,
                height,
            )?;
        }

        Ok(())
    }

    /// Perform motion compensation for a chroma block
    pub fn mc_chroma(
        &self,
        ref_pic: &[u16],
        ref_stride: usize,
        mv: MotionVector,
        dst: &mut [u16],
        dst_stride: usize,
        width: usize,
        height: usize,
    ) -> Result<()> {
        // Chroma MVs are half the luma MVs (4:2:0 subsampling)
        let frac_x = ((mv.x / 2) & 7) as u8; // 1/8-pixel precision for chroma
        let frac_y = ((mv.y / 2) & 7) as u8;

        let int_x = (mv.x / 2) >> 3; // Divide by 8
        let int_y = (mv.y / 2) >> 3;

        if frac_x == 0 && frac_y == 0 {
            // Integer position
            self.copy_block(
                ref_pic,
                ref_stride,
                int_x as isize,
                int_y as isize,
                dst,
                dst_stride,
                width,
                height,
            )?;
        } else if frac_y == 0 {
            // Horizontal only
            self.chroma_h_interp(
                ref_pic,
                ref_stride,
                int_x as isize,
                int_y as isize,
                frac_x,
                dst,
                dst_stride,
                width,
                height,
            )?;
        } else if frac_x == 0 {
            // Vertical only
            self.chroma_v_interp(
                ref_pic,
                ref_stride,
                int_x as isize,
                int_y as isize,
                frac_y,
                dst,
                dst_stride,
                width,
                height,
            )?;
        } else {
            // 2D interpolation
            self.chroma_hv_interp(
                ref_pic,
                ref_stride,
                int_x as isize,
                int_y as isize,
                frac_x,
                frac_y,
                dst,
                dst_stride,
                width,
                height,
            )?;
        }

        Ok(())
    }

    /// Bi-directional prediction: average L0 and L1 predictions
    pub fn mc_bipred(
        &self,
        pred_l0: &[u16],
        pred_l1: &[u16],
        dst: &mut [u16],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<()> {
        for y in 0..height {
            for x in 0..width {
                let idx = y * stride + x;
                let p0 = pred_l0[idx] as u32;
                let p1 = pred_l1[idx] as u32;
                let avg = ((p0 + p1 + 1) >> 1) as u16; // Round to nearest
                dst[idx] = avg.min(self.max_val);
            }
        }
        Ok(())
    }

    /// Copy block without interpolation
    fn copy_block(
        &self,
        src: &[u16],
        src_stride: usize,
        x: isize,
        y: isize,
        dst: &mut [u16],
        dst_stride: usize,
        width: usize,
        height: usize,
    ) -> Result<()> {
        for dy in 0..height {
            let src_y = (y + dy as isize) as usize;
            let src_offset = src_y * src_stride + x as usize;
            let dst_offset = dy * dst_stride;

            for dx in 0..width {
                dst[dst_offset + dx] = src[src_offset + dx];
            }
        }
        Ok(())
    }

    /// Luma horizontal interpolation
    fn luma_h_interp(
        &self,
        ref_pic: &[u16],
        ref_stride: usize,
        int_x: isize,
        int_y: isize,
        frac_x: u8,
        dst: &mut [u16],
        dst_stride: usize,
        width: usize,
        height: usize,
    ) -> Result<()> {
        let filter = &LUMA_FILTER[frac_x as usize];
        let shift = IF_FILTER_SHIFT;
        let offset = 1 << (shift - 1);

        for y in 0..height {
            let src_y = (int_y + y as isize) as usize;
            for x in 0..width {
                let src_x = (int_x + x as isize) as usize;
                let mut sum = 0i32;

                // 8-tap filter
                for i in 0..8 {
                    let src_idx = src_y * ref_stride + src_x + i - 3;
                    sum += ref_pic[src_idx] as i32 * filter[i] as i32;
                }

                let val = ((sum + offset) >> shift).clamp(0, self.max_val as i32) as u16;
                dst[y * dst_stride + x] = val;
            }
        }

        Ok(())
    }

    /// Luma vertical interpolation
    fn luma_v_interp(
        &self,
        ref_pic: &[u16],
        ref_stride: usize,
        int_x: isize,
        int_y: isize,
        frac_y: u8,
        dst: &mut [u16],
        dst_stride: usize,
        width: usize,
        height: usize,
    ) -> Result<()> {
        let filter = &LUMA_FILTER[frac_y as usize];
        let shift = IF_FILTER_SHIFT;
        let offset = 1 << (shift - 1);

        for y in 0..height {
            for x in 0..width {
                let src_x = (int_x + x as isize) as usize;
                let src_y = (int_y + y as isize) as usize;
                let mut sum = 0i32;

                // 8-tap filter
                for i in 0..8 {
                    let src_idx = (src_y + i - 3) * ref_stride + src_x;
                    sum += ref_pic[src_idx] as i32 * filter[i] as i32;
                }

                let val = ((sum + offset) >> shift).clamp(0, self.max_val as i32) as u16;
                dst[y * dst_stride + x] = val;
            }
        }

        Ok(())
    }

    /// Luma 2D interpolation (horizontal then vertical)
    fn luma_hv_interp(
        &self,
        ref_pic: &[u16],
        ref_stride: usize,
        int_x: isize,
        int_y: isize,
        frac_x: u8,
        frac_y: u8,
        dst: &mut [u16],
        dst_stride: usize,
        width: usize,
        height: usize,
    ) -> Result<()> {
        // Intermediate buffer for horizontal filtering
        let mut temp = vec![0i16; (width + 7) * (height + 7)];
        let temp_stride = width + 7;

        let h_filter = &LUMA_FILTER[frac_x as usize];
        let v_filter = &LUMA_FILTER[frac_y as usize];

        // Horizontal filtering to temp buffer
        for y in 0..height + 7 {
            let src_y = (int_y + y as isize - 3) as usize;
            for x in 0..width {
                let src_x = (int_x + x as isize) as usize;
                let mut sum = 0i32;

                for i in 0..8 {
                    let src_idx = src_y * ref_stride + src_x + i - 3;
                    sum += ref_pic[src_idx] as i32 * h_filter[i] as i32;
                }

                temp[y * temp_stride + x] = (sum >> IF_FILTER_SHIFT) as i16;
            }
        }

        // Vertical filtering from temp to destination
        let offset = 1 << (IF_FILTER_SHIFT - 1);
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0i32;

                for i in 0..8 {
                    let temp_idx = (y + i) * temp_stride + x;
                    sum += temp[temp_idx] as i32 * v_filter[i] as i32;
                }

                let val = ((sum + offset) >> IF_FILTER_SHIFT).clamp(0, self.max_val as i32) as u16;
                dst[y * dst_stride + x] = val;
            }
        }

        Ok(())
    }

    /// Chroma horizontal interpolation
    fn chroma_h_interp(
        &self,
        ref_pic: &[u16],
        ref_stride: usize,
        int_x: isize,
        int_y: isize,
        frac_x: u8,
        dst: &mut [u16],
        dst_stride: usize,
        width: usize,
        height: usize,
    ) -> Result<()> {
        let filter = &CHROMA_FILTER[frac_x as usize];
        let shift = IF_FILTER_SHIFT;
        let offset = 1 << (shift - 1);

        for y in 0..height {
            let src_y = (int_y + y as isize) as usize;
            for x in 0..width {
                let src_x = (int_x + x as isize) as usize;
                let mut sum = 0i32;

                // 4-tap filter
                for i in 0..4 {
                    let src_idx = src_y * ref_stride + src_x + i - 1;
                    sum += ref_pic[src_idx] as i32 * filter[i] as i32;
                }

                let val = ((sum + offset) >> shift).clamp(0, self.max_val as i32) as u16;
                dst[y * dst_stride + x] = val;
            }
        }

        Ok(())
    }

    /// Chroma vertical interpolation
    fn chroma_v_interp(
        &self,
        ref_pic: &[u16],
        ref_stride: usize,
        int_x: isize,
        int_y: isize,
        frac_y: u8,
        dst: &mut [u16],
        dst_stride: usize,
        width: usize,
        height: usize,
    ) -> Result<()> {
        let filter = &CHROMA_FILTER[frac_y as usize];
        let shift = IF_FILTER_SHIFT;
        let offset = 1 << (shift - 1);

        for y in 0..height {
            for x in 0..width {
                let src_x = (int_x + x as isize) as usize;
                let src_y = (int_y + y as isize) as usize;
                let mut sum = 0i32;

                // 4-tap filter
                for i in 0..4 {
                    let src_idx = (src_y + i - 1) * ref_stride + src_x;
                    sum += ref_pic[src_idx] as i32 * filter[i] as i32;
                }

                let val = ((sum + offset) >> shift).clamp(0, self.max_val as i32) as u16;
                dst[y * dst_stride + x] = val;
            }
        }

        Ok(())
    }

    /// Chroma 2D interpolation
    fn chroma_hv_interp(
        &self,
        ref_pic: &[u16],
        ref_stride: usize,
        int_x: isize,
        int_y: isize,
        frac_x: u8,
        frac_y: u8,
        dst: &mut [u16],
        dst_stride: usize,
        width: usize,
        height: usize,
    ) -> Result<()> {
        // Intermediate buffer
        let mut temp = vec![0i16; (width + 3) * (height + 3)];
        let temp_stride = width + 3;

        let h_filter = &CHROMA_FILTER[frac_x as usize];
        let v_filter = &CHROMA_FILTER[frac_y as usize];

        // Horizontal filtering
        for y in 0..height + 3 {
            let src_y = (int_y + y as isize - 1) as usize;
            for x in 0..width {
                let src_x = (int_x + x as isize) as usize;
                let mut sum = 0i32;

                for i in 0..4 {
                    let src_idx = src_y * ref_stride + src_x + i - 1;
                    sum += ref_pic[src_idx] as i32 * h_filter[i] as i32;
                }

                temp[y * temp_stride + x] = (sum >> IF_FILTER_SHIFT) as i16;
            }
        }

        // Vertical filtering
        let offset = 1 << (IF_FILTER_SHIFT - 1);
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0i32;

                for i in 0..4 {
                    let temp_idx = (y + i) * temp_stride + x;
                    sum += temp[temp_idx] as i32 * v_filter[i] as i32;
                }

                let val = ((sum + offset) >> IF_FILTER_SHIFT).clamp(0, self.max_val as i32) as u16;
                dst[y * dst_stride + x] = val;
            }
        }

        Ok(())
    }
}

/// Interpolation filter bit shift
const IF_FILTER_SHIFT: i32 = 6;

/// Luma interpolation filter coefficients (8-tap)
///
/// Index 0 = integer position (not used, just copy)
/// Index 1-3 = 1/4, 1/2, 3/4 pixel positions
const LUMA_FILTER: [[i8; 8]; 4] = [
    [0, 0, 0, 64, 0, 0, 0, 0],     // Integer (0/4)
    [-1, 4, -10, 58, 17, -5, 1, 0], // 1/4 pixel
    [-1, 4, -11, 40, 40, -11, 4, -1], // 1/2 pixel
    [0, 1, -5, 17, 58, -10, 4, -1], // 3/4 pixel
];

/// Chroma interpolation filter coefficients (4-tap)
///
/// For 1/8-pixel positions (chroma has 1/8-pixel precision)
const CHROMA_FILTER: [[i8; 4]; 8] = [
    [0, 64, 0, 0],    // 0/8
    [-2, 58, 10, -2], // 1/8
    [-4, 54, 16, -2], // 2/8 (1/4)
    [-6, 46, 28, -4], // 3/8
    [-4, 36, 36, -4], // 4/8 (1/2)
    [-4, 28, 46, -6], // 5/8
    [-2, 16, 54, -4], // 6/8 (3/4)
    [-2, 10, 58, -2], // 7/8
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_compensator_creation() {
        let mc = MotionCompensator::new(8);
        assert!(mc.is_ok());
    }

    #[test]
    fn test_motion_compensator_invalid_bit_depth() {
        let mc = MotionCompensator::new(9);
        assert!(mc.is_err());
    }

    #[test]
    fn test_copy_block_integer_position() {
        let mc = MotionCompensator::new(8).unwrap();
        let ref_pic = vec![128u16; 256]; // 16×16 reference
        let mut dst = vec![0u16; 64]; // 8×8 destination

        let mv = MotionVector::new(0, 0); // Integer position
        let result = mc.mc_luma(&ref_pic, 16, mv, &mut dst, 8, 8, 8);
        assert!(result.is_ok());

        // All samples should be 128
        assert!(dst.iter().all(|&v| v == 128));
    }

    #[test]
    fn test_luma_h_interp_half_pixel() {
        let mc = MotionCompensator::new(8).unwrap();
        let mut ref_pic = vec![0u16; 256];

        // Create a pattern: left side = 0, right side = 255
        for y in 0..16 {
            for x in 0..16 {
                ref_pic[y * 16 + x] = if x < 8 { 0 } else { 255 };
            }
        }

        let mut dst = vec![0u16; 64];
        let mv = MotionVector::new(2, 0); // 1/2 pixel in x direction
        mc.mc_luma(&ref_pic, 16, mv, &mut dst, 8, 8, 8).unwrap();

        // At edge, should be interpolated (not exactly 0 or 255)
        let edge_val = dst[0];
        assert!(edge_val > 0 && edge_val < 255);
    }

    #[test]
    fn test_luma_v_interp_half_pixel() {
        let mc = MotionCompensator::new(8).unwrap();
        let mut ref_pic = vec![0u16; 256];

        // Top half = 0, bottom half = 255
        for y in 0..16 {
            for x in 0..16 {
                ref_pic[y * 16 + x] = if y < 8 { 0 } else { 255 };
            }
        }

        let mut dst = vec![0u16; 64];
        let mv = MotionVector::new(0, 2); // 1/2 pixel in y direction
        mc.mc_luma(&ref_pic, 16, mv, &mut dst, 8, 8, 8).unwrap();

        // Should be interpolated
        assert!(dst[0] > 0 && dst[0] < 255);
    }

    #[test]
    fn test_luma_hv_interp() {
        let mc = MotionCompensator::new(8).unwrap();
        let ref_pic = vec![128u16; 256];
        let mut dst = vec![0u16; 64];

        let mv = MotionVector::new(2, 2); // 1/2 pixel in both directions
        mc.mc_luma(&ref_pic, 16, mv, &mut dst, 8, 8, 8).unwrap();

        // Uniform source should produce uniform output
        assert!(dst.iter().all(|&v| v.abs_diff(128) < 5));
    }

    #[test]
    fn test_chroma_interp_integer() {
        let mc = MotionCompensator::new(8).unwrap();
        let ref_pic = vec![64u16; 256];
        let mut dst = vec![0u16; 64];

        let mv = MotionVector::new(0, 0); // Integer position
        mc.mc_chroma(&ref_pic, 16, mv, &mut dst, 8, 8, 8)
            .unwrap();

        assert!(dst.iter().all(|&v| v == 64));
    }

    #[test]
    fn test_chroma_h_interp() {
        let mc = MotionCompensator::new(8).unwrap();
        let mut ref_pic = vec![0u16; 256];

        // Gradient pattern
        for y in 0..16 {
            for x in 0..16 {
                ref_pic[y * 16 + x] = (x * 16) as u16;
            }
        }

        let mut dst = vec![0u16; 64];
        let mv = MotionVector::new(4, 0); // 1/2 pixel in x (chroma)
        mc.mc_chroma(&ref_pic, 16, mv, &mut dst, 8, 8, 8)
            .unwrap();

        // Should have interpolated values
        assert!(dst[0] > 0);
    }

    #[test]
    fn test_chroma_v_interp() {
        let mc = MotionCompensator::new(8).unwrap();
        let mut ref_pic = vec![0u16; 256];

        for y in 0..16 {
            for x in 0..16 {
                ref_pic[y * 16 + x] = (y * 16) as u16;
            }
        }

        let mut dst = vec![0u16; 64];
        let mv = MotionVector::new(0, 4); // 1/2 pixel in y
        mc.mc_chroma(&ref_pic, 16, mv, &mut dst, 8, 8, 8)
            .unwrap();

        assert!(dst[0] > 0);
    }

    #[test]
    fn test_chroma_hv_interp() {
        let mc = MotionCompensator::new(8).unwrap();
        let ref_pic = vec![100u16; 256];
        let mut dst = vec![0u16; 64];

        let mv = MotionVector::new(4, 4); // 1/2 pixel in both
        mc.mc_chroma(&ref_pic, 16, mv, &mut dst, 8, 8, 8)
            .unwrap();

        // Uniform source -> uniform output
        assert!(dst.iter().all(|&v| v.abs_diff(100) < 5));
    }

    #[test]
    fn test_bipred_averaging() {
        let mc = MotionCompensator::new(8).unwrap();
        let pred_l0 = vec![100u16; 64];
        let pred_l1 = vec![200u16; 64];
        let mut dst = vec![0u16; 64];

        mc.mc_bipred(&pred_l0, &pred_l1, &mut dst, 8, 8, 8)
            .unwrap();

        // Should average to 150
        assert!(dst.iter().all(|&v| v == 150));
    }

    #[test]
    fn test_bipred_same_prediction() {
        let mc = MotionCompensator::new(8).unwrap();
        let pred = vec![128u16; 64];
        let mut dst = vec![0u16; 64];

        mc.mc_bipred(&pred, &pred, &mut dst, 8, 8, 8).unwrap();

        // Same predictions should produce same output
        assert!(dst.iter().all(|&v| v == 128));
    }

    #[test]
    fn test_bipred_clipping() {
        let mc = MotionCompensator::new(8).unwrap();
        let pred_l0 = vec![255u16; 64];
        let pred_l1 = vec![255u16; 64];
        let mut dst = vec![0u16; 64];

        mc.mc_bipred(&pred_l0, &pred_l1, &mut dst, 8, 8, 8)
            .unwrap();

        // Should be clipped to max_val (255 for 8-bit)
        assert!(dst.iter().all(|&v| v == 255));
    }

    #[test]
    fn test_luma_filter_coefficients_sum() {
        // All filter coefficients should sum to 64 for DC preservation
        for filter in &LUMA_FILTER {
            let sum: i32 = filter.iter().map(|&c| c as i32).sum();
            assert_eq!(sum, 64);
        }
    }

    #[test]
    fn test_chroma_filter_coefficients_sum() {
        for filter in &CHROMA_FILTER {
            let sum: i32 = filter.iter().map(|&c| c as i32).sum();
            assert_eq!(sum, 64);
        }
    }

    #[test]
    fn test_luma_filter_symmetry() {
        // 1/2-pixel filter should be symmetric
        let half_pixel = &LUMA_FILTER[2];
        assert_eq!(half_pixel[0], half_pixel[7]);
        assert_eq!(half_pixel[1], half_pixel[6]);
        assert_eq!(half_pixel[2], half_pixel[5]);
        assert_eq!(half_pixel[3], half_pixel[4]);
    }

    #[test]
    fn test_motion_compensator_10bit() {
        let mc = MotionCompensator::new(10).unwrap();
        assert_eq!(mc.bit_depth, 10);
        assert_eq!(mc.max_val, 1023);
    }

    #[test]
    fn test_motion_compensator_12bit() {
        let mc = MotionCompensator::new(12).unwrap();
        assert_eq!(mc.bit_depth, 12);
        assert_eq!(mc.max_val, 4095);
    }

    #[test]
    fn test_mc_luma_quarter_pixel() {
        let mc = MotionCompensator::new(8).unwrap();
        let ref_pic = vec![128u16; 256];
        let mut dst = vec![0u16; 64];

        // 1/4 pixel position
        let mv = MotionVector::new(1, 1);
        let result = mc.mc_luma(&ref_pic, 16, mv, &mut dst, 8, 8, 8);
        assert!(result.is_ok());

        // Uniform source should stay uniform
        assert!(dst.iter().all(|&v| v.abs_diff(128) < 5));
    }

    #[test]
    fn test_mc_luma_three_quarter_pixel() {
        let mc = MotionCompensator::new(8).unwrap();
        let ref_pic = vec![100u16; 256];
        let mut dst = vec![0u16; 64];

        // 3/4 pixel position
        let mv = MotionVector::new(3, 3);
        let result = mc.mc_luma(&ref_pic, 16, mv, &mut dst, 8, 8, 8);
        assert!(result.is_ok());

        assert!(dst.iter().all(|&v| v.abs_diff(100) < 5));
    }

    #[test]
    fn test_copy_block() {
        let mc = MotionCompensator::new(8).unwrap();
        let mut src = vec![0u16; 256];

        // Fill with pattern
        for i in 0..256 {
            src[i] = i as u16;
        }

        let mut dst = vec![0u16; 64];
        mc.copy_block(&src, 16, 4, 4, &mut dst, 8, 8, 8)
            .unwrap();

        // Check copied values
        for y in 0..8 {
            for x in 0..8 {
                let expected = (y + 4) * 16 + (x + 4);
                assert_eq!(dst[y * 8 + x], expected as u16);
            }
        }
    }

    #[test]
    fn test_if_filter_shift() {
        assert_eq!(IF_FILTER_SHIFT, 6);
    }
}
