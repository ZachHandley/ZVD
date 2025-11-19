//! H.265/HEVC Intra Prediction
//!
//! Intra prediction predicts block pixels from neighboring already-decoded pixels.
//! H.265 has 35 intra prediction modes:
//! - Mode 0: Planar (smooth gradients)
//! - Mode 1: DC (average)
//! - Modes 2-34: Angular/directional prediction

use super::ctu::IntraMode;
use crate::error::{Error, Result};

/// Reference samples for intra prediction
///
/// For a block of size NxN, we need:
/// - 2N+1 reference samples from the left (including top-left corner)
/// - 2N reference samples from the top
/// - N reference samples from top-right (for angular modes)
#[derive(Debug, Clone)]
pub struct ReferenceSamples {
    /// Left reference samples (bottom to top), including top-left corner
    /// Length: 2*N + 1
    pub left: Vec<u16>,

    /// Top reference samples (left to right), NOT including top-left
    /// Length: 2*N
    pub top: Vec<u16>,

    /// Block size (N)
    pub size: usize,

    /// Bit depth
    pub bit_depth: u8,
}

impl ReferenceSamples {
    /// Create reference samples for a block
    pub fn new(size: usize, bit_depth: u8) -> Self {
        ReferenceSamples {
            left: vec![0; 2 * size + 1],
            top: vec![0; 2 * size],
            size,
            bit_depth,
        }
    }

    /// Get top-left corner pixel
    pub fn top_left(&self) -> u16 {
        self.left[2 * self.size] // Top-left is the last element of left[]
    }

    /// Get left pixel at position (0 = bottom-left, size-1 = adjacent to block)
    pub fn get_left(&self, idx: usize) -> u16 {
        if idx < self.left.len() {
            self.left[idx]
        } else {
            0
        }
    }

    /// Get top pixel at position (0 = left-most, size-1 = right-most of block)
    pub fn get_top(&self, idx: usize) -> u16 {
        if idx < self.top.len() {
            self.top[idx]
        } else {
            0
        }
    }

    /// Fill with a constant value (for testing)
    pub fn fill_constant(&mut self, value: u16) {
        self.left.fill(value);
        self.top.fill(value);
    }

    /// Fill with gradient (for testing)
    pub fn fill_gradient(&mut self) {
        let max_val = (1 << self.bit_depth) - 1;
        let step = max_val / (2 * self.size) as u16;

        for i in 0..self.left.len() {
            self.left[i] = (i as u16 * step).min(max_val);
        }

        for i in 0..self.top.len() {
            self.top[i] = (i as u16 * step).min(max_val);
        }
    }
}

/// Intra predictor for H.265
pub struct IntraPredictor {
    /// Bit depth (8, 10, or 12)
    pub bit_depth: u8,
}

impl IntraPredictor {
    /// Create a new intra predictor
    pub fn new(bit_depth: u8) -> Self {
        IntraPredictor { bit_depth }
    }

    /// Perform intra prediction
    ///
    /// # Arguments
    /// * `mode` - Intra prediction mode
    /// * `refs` - Reference samples
    /// * `dst` - Destination buffer (will be filled with predicted pixels)
    /// * `stride` - Stride of destination buffer
    pub fn predict(
        &self,
        mode: IntraMode,
        refs: &ReferenceSamples,
        dst: &mut [u16],
        stride: usize,
    ) -> Result<()> {
        let size = refs.size;

        // Ensure destination is large enough
        if dst.len() < size * stride {
            return Err(Error::codec("Destination buffer too small for intra prediction"));
        }

        match mode {
            IntraMode::Planar => self.predict_planar(refs, dst, stride),
            IntraMode::Dc => self.predict_dc(refs, dst, stride),
            IntraMode::Angular(angle) => self.predict_angular(refs, dst, stride, angle),
        }
    }

    /// Planar prediction (Mode 0)
    ///
    /// Creates smooth gradients by bilinear interpolation between
    /// top, left, top-right, and bottom-left reference pixels.
    fn predict_planar(&self, refs: &ReferenceSamples, dst: &mut [u16], stride: usize) -> Result<()> {
        let n = refs.size;

        // Get corner reference pixels
        let top_right = refs.get_top(n);  // Pixel to the right of top-right corner
        let bottom_left = refs.get_left(0); // Bottom-left pixel

        for y in 0..n {
            for x in 0..n {
                // Horizontal gradient from left to top-right
                let left_val = refs.get_left(n - 1 - y) as i32;
                let horiz = ((n - 1 - x) as i32 * left_val + (x + 1) as i32 * top_right as i32) as i32;

                // Vertical gradient from top to bottom-left
                let top_val = refs.get_top(x) as i32;
                let vert = ((n - 1 - y) as i32 * top_val + (y + 1) as i32 * bottom_left as i32) as i32;

                // Average horizontal and vertical, with rounding
                let pred = ((horiz + vert + n as i32) >> (n.trailing_zeros() + 1)) as u16;

                // Clamp to bit depth
                let max_val = (1 << self.bit_depth) - 1;
                dst[y * stride + x] = pred.min(max_val);
            }
        }

        Ok(())
    }

    /// DC prediction (Mode 1)
    ///
    /// Predicts all pixels as the average of available reference pixels.
    fn predict_dc(&self, refs: &ReferenceSamples, dst: &mut [u16], stride: usize) -> Result<()> {
        let n = refs.size;

        // Calculate average of top and left reference pixels
        let mut sum = 0u32;
        let mut count = 0u32;

        // Sum top reference pixels
        for i in 0..n {
            sum += refs.get_top(i) as u32;
            count += 1;
        }

        // Sum left reference pixels (adjacent to block)
        for i in 0..n {
            sum += refs.get_left(n - 1 - i) as u32;
            count += 1;
        }

        // Calculate average (DC value)
        let dc_val = if count > 0 {
            ((sum + (count / 2)) / count) as u16
        } else {
            1 << (self.bit_depth - 1) // Mid-gray if no references
        };

        // Fill entire block with DC value
        for y in 0..n {
            for x in 0..n {
                dst[y * stride + x] = dc_val;
            }
        }

        Ok(())
    }

    /// Angular prediction (Modes 2-34)
    ///
    /// For Phase 8.2, we implement simplified angular prediction.
    /// Full implementation would include:
    /// - 33 different angles
    /// - Fractional sample interpolation
    /// - Reference sample filtering
    fn predict_angular(&self, refs: &ReferenceSamples, dst: &mut [u16], stride: usize, angle: u8) -> Result<()> {
        let n = refs.size;

        // For Phase 8.2, implement only vertical (mode 26) and horizontal (mode 10)
        // as simplified angular modes
        match angle {
            10 => self.predict_horizontal(refs, dst, stride),
            26 => self.predict_vertical(refs, dst, stride),
            _ => {
                // For other angular modes, fall back to DC for now
                // Full implementation in Phase 8.3
                self.predict_dc(refs, dst, stride)
            }
        }
    }

    /// Horizontal prediction (Angular mode 10)
    ///
    /// Each row is filled with the left reference pixel for that row.
    fn predict_horizontal(&self, refs: &ReferenceSamples, dst: &mut [u16], stride: usize) -> Result<()> {
        let n = refs.size;

        for y in 0..n {
            let ref_val = refs.get_left(n - 1 - y);
            for x in 0..n {
                dst[y * stride + x] = ref_val;
            }
        }

        Ok(())
    }

    /// Vertical prediction (Angular mode 26)
    ///
    /// Each column is filled with the top reference pixel for that column.
    fn predict_vertical(&self, refs: &ReferenceSamples, dst: &mut [u16], stride: usize) -> Result<()> {
        let n = refs.size;

        for y in 0..n {
            for x in 0..n {
                dst[y * stride + x] = refs.get_top(x);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reference_samples_creation() {
        let refs = ReferenceSamples::new(8, 8);
        assert_eq!(refs.left.len(), 17);  // 2*8 + 1
        assert_eq!(refs.top.len(), 16);   // 2*8
        assert_eq!(refs.size, 8);
    }

    #[test]
    fn test_reference_samples_fill_constant() {
        let mut refs = ReferenceSamples::new(4, 8);
        refs.fill_constant(100);

        assert_eq!(refs.get_left(0), 100);
        assert_eq!(refs.get_top(0), 100);
        assert_eq!(refs.top_left(), 100);
    }

    #[test]
    fn test_dc_prediction_uniform() {
        let predictor = IntraPredictor::new(8);
        let mut refs = ReferenceSamples::new(4, 8);

        // Fill references with constant value 128
        refs.fill_constant(128);

        let mut dst = vec![0u16; 16];
        predictor.predict_dc(&refs, &mut dst, 4).unwrap();

        // All pixels should be 128
        for &pixel in &dst {
            assert_eq!(pixel, 128);
        }
    }

    #[test]
    fn test_dc_prediction_average() {
        let predictor = IntraPredictor::new(8);
        let mut refs = ReferenceSamples::new(2, 8);

        // Top: [100, 100]
        refs.top[0] = 100;
        refs.top[1] = 100;

        // Left: [200, 200] (indices 1 and 2, as index 0 is bottom-most)
        refs.left[0] = 200;
        refs.left[1] = 200;

        let mut dst = vec![0u16; 4];
        predictor.predict_dc(&refs, &mut dst, 2).unwrap();

        // DC = (100 + 100 + 200 + 200) / 4 = 150
        for &pixel in &dst {
            assert_eq!(pixel, 150);
        }
    }

    #[test]
    fn test_vertical_prediction() {
        let predictor = IntraPredictor::new(8);
        let mut refs = ReferenceSamples::new(4, 8);

        // Set top reference: [10, 20, 30, 40]
        for i in 0..4 {
            refs.top[i] = (i as u16 + 1) * 10;
        }

        let mut dst = vec![0u16; 16];
        predictor.predict_vertical(&refs, &mut dst, 4).unwrap();

        // Each column should be constant
        for y in 0..4 {
            assert_eq!(dst[y * 4 + 0], 10);  // Column 0
            assert_eq!(dst[y * 4 + 1], 20);  // Column 1
            assert_eq!(dst[y * 4 + 2], 30);  // Column 2
            assert_eq!(dst[y * 4 + 3], 40);  // Column 3
        }
    }

    #[test]
    fn test_horizontal_prediction() {
        let predictor = IntraPredictor::new(8);
        let mut refs = ReferenceSamples::new(4, 8);

        // Set left reference: [40, 30, 20, 10] (bottom to top)
        refs.left[0] = 40;
        refs.left[1] = 30;
        refs.left[2] = 20;
        refs.left[3] = 10;

        let mut dst = vec![0u16; 16];
        predictor.predict_horizontal(&refs, &mut dst, 4).unwrap();

        // Each row should be constant
        // Row 0 uses left[3] = 10
        // Row 1 uses left[2] = 20
        // Row 2 uses left[1] = 30
        // Row 3 uses left[0] = 40
        for x in 0..4 {
            assert_eq!(dst[0 * 4 + x], 10);  // Row 0
            assert_eq!(dst[1 * 4 + x], 20);  // Row 1
            assert_eq!(dst[2 * 4 + x], 30);  // Row 2
            assert_eq!(dst[3 * 4 + x], 40);  // Row 3
        }
    }

    #[test]
    fn test_planar_prediction_uniform() {
        let predictor = IntraPredictor::new(8);
        let mut refs = ReferenceSamples::new(4, 8);

        // Uniform reference (all 100)
        refs.fill_constant(100);

        let mut dst = vec![0u16; 16];
        predictor.predict_planar(&refs, &mut dst, 4).unwrap();

        // With uniform references, planar should produce uniform output
        for &pixel in &dst {
            assert!(pixel >= 95 && pixel <= 105, "Planar with uniform refs should be ~100, got {}", pixel);
        }
    }

    #[test]
    fn test_predict_with_mode() {
        let predictor = IntraPredictor::new(8);
        let mut refs = ReferenceSamples::new(4, 8);
        refs.fill_constant(50);

        let mut dst = vec![0u16; 16];

        // Test DC mode
        predictor.predict(IntraMode::Dc, &refs, &mut dst, 4).unwrap();
        assert_eq!(dst[0], 50);

        // Test Planar mode
        dst.fill(0);
        predictor.predict(IntraMode::Planar, &refs, &mut dst, 4).unwrap();
        assert!(dst[0] > 0);

        // Test Angular mode (vertical)
        dst.fill(0);
        predictor.predict(IntraMode::Angular(26), &refs, &mut dst, 4).unwrap();
        assert_eq!(dst[0], 50);
    }
}
