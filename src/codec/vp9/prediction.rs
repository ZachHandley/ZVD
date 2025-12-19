//! VP9 Intra and Inter Prediction
//!
//! VP9 supports 10 intra prediction modes:
//! - DC_PRED: Average of above and left neighbors
//! - V_PRED: Vertical (copy from above)
//! - H_PRED: Horizontal (copy from left)
//! - D45_PRED: 45-degree diagonal (down-right)
//! - D135_PRED: 135-degree diagonal (down-left)
//! - D117_PRED: 117-degree
//! - D153_PRED: 153-degree
//! - D207_PRED: 207-degree (horizontal-up)
//! - D63_PRED: 63-degree (vertical-right)
//! - TM_PRED: True motion (gradient prediction)
//!
//! Inter prediction uses 8-tap filtering with 1/8-pel precision.

use super::tables::{
    InterpFilter, IntraMode, MotionVector, BILINEAR_FILTERS, SUBPEL_FILTERS_REGULAR,
    SUBPEL_FILTERS_SHARP, SUBPEL_FILTERS_SMOOTH,
};

// =============================================================================
// Intra Prediction
// =============================================================================

/// Intra predictor with reference samples
pub struct IntraPredictor<'a> {
    /// Above reference samples (block_size + block_size for diagonal modes)
    pub above: &'a [u8],
    /// Left reference samples
    pub left: &'a [u8],
    /// Top-left corner sample
    pub top_left: u8,
    /// Above samples available
    pub have_above: bool,
    /// Left samples available
    pub have_left: bool,
}

impl<'a> IntraPredictor<'a> {
    /// Create a new intra predictor
    pub fn new(
        above: &'a [u8],
        left: &'a [u8],
        top_left: u8,
        have_above: bool,
        have_left: bool,
    ) -> Self {
        IntraPredictor {
            above,
            left,
            top_left,
            have_above,
            have_left,
        }
    }

    /// Perform prediction for the given mode
    pub fn predict(&self, mode: IntraMode, output: &mut [u8], stride: usize, size: usize) {
        match mode {
            IntraMode::DcPred => self.predict_dc(output, stride, size),
            IntraMode::VPred => self.predict_v(output, stride, size),
            IntraMode::HPred => self.predict_h(output, stride, size),
            IntraMode::D45Pred => self.predict_d45(output, stride, size),
            IntraMode::D135Pred => self.predict_d135(output, stride, size),
            IntraMode::D117Pred => self.predict_d117(output, stride, size),
            IntraMode::D153Pred => self.predict_d153(output, stride, size),
            IntraMode::D207Pred => self.predict_d207(output, stride, size),
            IntraMode::D63Pred => self.predict_d63(output, stride, size),
            IntraMode::TmPred => self.predict_tm(output, stride, size),
        }
    }

    /// DC prediction - average of above and left neighbors
    fn predict_dc(&self, output: &mut [u8], stride: usize, size: usize) {
        let dc = if self.have_above && self.have_left {
            let above_sum: u32 = self.above[..size].iter().map(|&x| x as u32).sum();
            let left_sum: u32 = self.left[..size].iter().map(|&x| x as u32).sum();
            ((above_sum + left_sum + size as u32) / (2 * size as u32)) as u8
        } else if self.have_above {
            let sum: u32 = self.above[..size].iter().map(|&x| x as u32).sum();
            ((sum + (size / 2) as u32) / size as u32) as u8
        } else if self.have_left {
            let sum: u32 = self.left[..size].iter().map(|&x| x as u32).sum();
            ((sum + (size / 2) as u32) / size as u32) as u8
        } else {
            128 // Default mid-gray
        };

        for y in 0..size {
            for x in 0..size {
                output[y * stride + x] = dc;
            }
        }
    }

    /// Vertical prediction - copy from above row
    fn predict_v(&self, output: &mut [u8], stride: usize, size: usize) {
        for y in 0..size {
            for x in 0..size {
                output[y * stride + x] = self.above[x];
            }
        }
    }

    /// Horizontal prediction - copy from left column
    fn predict_h(&self, output: &mut [u8], stride: usize, size: usize) {
        for y in 0..size {
            let val = self.left[y];
            for x in 0..size {
                output[y * stride + x] = val;
            }
        }
    }

    /// True motion prediction - gradient prediction
    fn predict_tm(&self, output: &mut [u8], stride: usize, size: usize) {
        for y in 0..size {
            for x in 0..size {
                let pred = self.left[y] as i16 + self.above[x] as i16 - self.top_left as i16;
                output[y * stride + x] = pred.clamp(0, 255) as u8;
            }
        }
    }

    /// D45 prediction - 45-degree diagonal (down-right)
    fn predict_d45(&self, output: &mut [u8], stride: usize, size: usize) {
        for y in 0..size {
            for x in 0..size {
                let idx = x + y;
                if idx < 2 * size - 1 {
                    // Average of diagonal neighbors
                    let a = self.above.get(idx).copied().unwrap_or(128) as u16;
                    let b = self.above.get(idx + 1).copied().unwrap_or(a as u8) as u16;
                    let c = self.above.get(idx + 2).copied().unwrap_or(b as u8) as u16;
                    output[y * stride + x] = ((a + 2 * b + c + 2) >> 2) as u8;
                } else {
                    output[y * stride + x] = self.above.get(2 * size - 1).copied().unwrap_or(128);
                }
            }
        }
    }

    /// D135 prediction - 135-degree diagonal (down-left)
    fn predict_d135(&self, output: &mut [u8], stride: usize, size: usize) {
        // Build extended reference array
        let mut ref_samples = vec![128u8; 2 * size + 1];

        // Fill left samples (reversed)
        for i in 0..size.min(self.left.len()) {
            ref_samples[size - 1 - i] = self.left[i];
        }

        // Top-left
        ref_samples[size] = self.top_left;

        // Fill above samples
        for i in 0..size.min(self.above.len()) {
            ref_samples[size + 1 + i] = self.above[i];
        }

        for y in 0..size {
            for x in 0..size {
                let idx = size + x - y;
                let a = ref_samples.get(idx.wrapping_sub(1)).copied().unwrap_or(128) as u16;
                let b = ref_samples.get(idx).copied().unwrap_or(128) as u16;
                let c = ref_samples.get(idx + 1).copied().unwrap_or(128) as u16;
                output[y * stride + x] = ((a + 2 * b + c + 2) >> 2) as u8;
            }
        }
    }

    /// D117 prediction - 117-degree
    fn predict_d117(&self, output: &mut [u8], stride: usize, size: usize) {
        // Build extended reference
        let mut ref_samples = vec![128u8; 2 * size + 1];

        for i in 0..size.min(self.left.len()) {
            ref_samples[size - 1 - i] = self.left[i];
        }
        ref_samples[size] = self.top_left;
        for i in 0..size.min(self.above.len()) {
            ref_samples[size + 1 + i] = self.above[i];
        }

        for y in 0..size {
            for x in 0..size {
                let zy = 2 * y as i32;
                let zx = 2 * x as i32;
                let idx = (size as i32 + zx - zy) as usize;

                let val = if (zx - zy) >= -1 {
                    let base = idx >> 1;
                    if (zx - zy) & 1 == 0 {
                        let a = ref_samples.get(base).copied().unwrap_or(128) as u16;
                        let b = ref_samples.get(base + 1).copied().unwrap_or(128) as u16;
                        ((a + b + 1) >> 1) as u8
                    } else {
                        let a = ref_samples.get(base).copied().unwrap_or(128) as u16;
                        let b = ref_samples.get(base + 1).copied().unwrap_or(128) as u16;
                        let c = ref_samples.get(base + 2).copied().unwrap_or(128) as u16;
                        ((a + 2 * b + c + 2) >> 2) as u8
                    }
                } else {
                    let idx = size - y;
                    let a = ref_samples.get(idx.wrapping_sub(1)).copied().unwrap_or(128) as u16;
                    let b = ref_samples.get(idx).copied().unwrap_or(128) as u16;
                    let c = ref_samples.get(idx + 1).copied().unwrap_or(128) as u16;
                    ((a + 2 * b + c + 2) >> 2) as u8
                };

                output[y * stride + x] = val;
            }
        }
    }

    /// D153 prediction - 153-degree
    fn predict_d153(&self, output: &mut [u8], stride: usize, size: usize) {
        let mut ref_samples = vec![128u8; 2 * size + 1];

        for i in 0..size.min(self.left.len()) {
            ref_samples[size - 1 - i] = self.left[i];
        }
        ref_samples[size] = self.top_left;
        for i in 0..size.min(self.above.len()) {
            ref_samples[size + 1 + i] = self.above[i];
        }

        for y in 0..size {
            for x in 0..size {
                let zy = 2 * y as i32;
                let zx = 2 * x as i32;

                let val = if zy > zx {
                    let idx = (size as i32 - ((zy - zx) >> 1) - 1) as usize;
                    if (zy - zx) & 1 == 0 {
                        let a = ref_samples.get(idx).copied().unwrap_or(128) as u16;
                        let b = ref_samples.get(idx + 1).copied().unwrap_or(128) as u16;
                        ((a + b + 1) >> 1) as u8
                    } else {
                        let a = ref_samples.get(idx).copied().unwrap_or(128) as u16;
                        let b = ref_samples.get(idx + 1).copied().unwrap_or(128) as u16;
                        let c = ref_samples.get(idx + 2).copied().unwrap_or(128) as u16;
                        ((a + 2 * b + c + 2) >> 2) as u8
                    }
                } else {
                    let idx = size + x - y;
                    let a = ref_samples.get(idx.wrapping_sub(1)).copied().unwrap_or(128) as u16;
                    let b = ref_samples.get(idx).copied().unwrap_or(128) as u16;
                    let c = ref_samples.get(idx + 1).copied().unwrap_or(128) as u16;
                    ((a + 2 * b + c + 2) >> 2) as u8
                };

                output[y * stride + x] = val;
            }
        }
    }

    /// D207 prediction - 207-degree (horizontal-up)
    fn predict_d207(&self, output: &mut [u8], stride: usize, size: usize) {
        for y in 0..size {
            for x in 0..size {
                let z = x as i32 + 2 * y as i32;

                let val = if z < (2 * size - 2) as i32 {
                    let idx = (z >> 1) as usize;
                    if z & 1 == 0 {
                        let a = self.left.get(idx).copied().unwrap_or(128) as u16;
                        let b = self.left.get(idx + 1).copied().unwrap_or(128) as u16;
                        ((a + b + 1) >> 1) as u8
                    } else {
                        let a = self.left.get(idx).copied().unwrap_or(128) as u16;
                        let b = self.left.get(idx + 1).copied().unwrap_or(128) as u16;
                        let c = self.left.get(idx + 2).copied().unwrap_or(128) as u16;
                        ((a + 2 * b + c + 2) >> 2) as u8
                    }
                } else if z == (2 * size - 2) as i32 {
                    let a = self.left.get(size - 1).copied().unwrap_or(128) as u16;
                    let b = self.left.get(size - 1).copied().unwrap_or(128) as u16;
                    ((a + b + 1) >> 1) as u8
                } else {
                    self.left.get(size - 1).copied().unwrap_or(128)
                };

                output[y * stride + x] = val;
            }
        }
    }

    /// D63 prediction - 63-degree (vertical-right)
    fn predict_d63(&self, output: &mut [u8], stride: usize, size: usize) {
        for y in 0..size {
            for x in 0..size {
                let z = x as i32 + 2 * y as i32;

                let val = if z < (2 * size - 2) as i32 {
                    let idx = (x + (y >> 1)) as usize;
                    if y & 1 == 0 {
                        let a = self.above.get(idx).copied().unwrap_or(128) as u16;
                        let b = self.above.get(idx + 1).copied().unwrap_or(128) as u16;
                        ((a + b + 1) >> 1) as u8
                    } else {
                        let a = self.above.get(idx).copied().unwrap_or(128) as u16;
                        let b = self.above.get(idx + 1).copied().unwrap_or(128) as u16;
                        let c = self.above.get(idx + 2).copied().unwrap_or(128) as u16;
                        ((a + 2 * b + c + 2) >> 2) as u8
                    }
                } else {
                    self.above.get(2 * size - 1).copied().unwrap_or(128)
                };

                output[y * stride + x] = val;
            }
        }
    }
}

// =============================================================================
// Inter Prediction
// =============================================================================

/// Frame buffer for reference frames
#[derive(Clone)]
pub struct FrameBuffer {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub y_stride: usize,
    pub uv_stride: usize,
}

impl FrameBuffer {
    /// Create a new frame buffer
    pub fn new(width: u32, height: u32) -> Self {
        // Align to 64-byte boundary for SIMD
        let y_stride = ((width as usize + 63) / 64) * 64;
        let uv_stride = ((width as usize / 2 + 31) / 32) * 32;
        let height_aligned = ((height as usize + 63) / 64) * 64;

        FrameBuffer {
            width,
            height,
            y: vec![128; y_stride * height_aligned],
            u: vec![128; uv_stride * (height_aligned / 2)],
            v: vec![128; uv_stride * (height_aligned / 2)],
            y_stride,
            uv_stride,
        }
    }

    /// Copy from another frame buffer
    pub fn copy_from(&mut self, other: &FrameBuffer) {
        self.y.copy_from_slice(&other.y);
        self.u.copy_from_slice(&other.u);
        self.v.copy_from_slice(&other.v);
    }

    /// Get pixel with boundary clamping
    #[inline]
    pub fn get_pixel_y(&self, x: i32, y: i32) -> u8 {
        let cx = x.clamp(0, self.width as i32 - 1) as usize;
        let cy = y.clamp(0, self.height as i32 - 1) as usize;
        self.y[cy * self.y_stride + cx]
    }

    /// Get chroma pixel with boundary clamping
    #[inline]
    pub fn get_pixel_uv(&self, plane: &[u8], x: i32, y: i32) -> u8 {
        let cx = x.clamp(0, (self.width / 2) as i32 - 1) as usize;
        let cy = y.clamp(0, (self.height / 2) as i32 - 1) as usize;
        plane[cy * self.uv_stride + cx]
    }
}

/// Get interpolation filter kernel
pub fn get_filter(filter_type: InterpFilter, subpel: usize) -> &'static [i16; 8] {
    match filter_type {
        InterpFilter::EightTap => &SUBPEL_FILTERS_REGULAR[subpel],
        InterpFilter::EightTapSmooth => &SUBPEL_FILTERS_SMOOTH[subpel],
        InterpFilter::EightTapSharp => &SUBPEL_FILTERS_SHARP[subpel],
        InterpFilter::Bilinear | InterpFilter::Switchable => &SUBPEL_FILTERS_REGULAR[subpel],
    }
}

/// Perform 8-tap horizontal filtering
fn filter_horizontal(
    src: &FrameBuffer,
    src_x: i32,
    src_y: i32,
    width: usize,
    height: usize,
    subpel_x: usize,
    filter_type: InterpFilter,
    output: &mut [i16],
    out_stride: usize,
) {
    let filter = get_filter(filter_type, subpel_x);

    for y in 0..height {
        let sy = src_y + y as i32;
        for x in 0..width {
            let mut sum = 0i32;
            for (k, &coeff) in filter.iter().enumerate() {
                let sx = src_x + x as i32 + k as i32 - 3;
                sum += src.get_pixel_y(sx, sy) as i32 * coeff as i32;
            }
            output[y * out_stride + x] = ((sum + 64) >> 7) as i16;
        }
    }
}

/// Perform 8-tap vertical filtering
fn filter_vertical(
    src: &[i16],
    src_stride: usize,
    width: usize,
    height: usize,
    subpel_y: usize,
    filter_type: InterpFilter,
    output: &mut [u8],
    out_stride: usize,
) {
    let filter = get_filter(filter_type, subpel_y);

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0i32;
            for (k, &coeff) in filter.iter().enumerate() {
                let sy = y + k;
                let src_val = if sy < src.len() / src_stride {
                    src[sy * src_stride + x]
                } else {
                    0
                };
                sum += src_val as i32 * coeff as i32;
            }
            output[y * out_stride + x] = ((sum + 64) >> 7).clamp(0, 255) as u8;
        }
    }
}

/// Perform motion compensation for a luma block
pub fn motion_compensate_luma(
    ref_frame: &FrameBuffer,
    mv: MotionVector,
    block_x: usize,
    block_y: usize,
    width: usize,
    height: usize,
    filter_type: InterpFilter,
    output: &mut [u8],
    out_stride: usize,
) {
    // MV is in 1/8-pel units
    let full_x = block_x as i32 + (mv.col >> 3) as i32;
    let full_y = block_y as i32 + (mv.row >> 3) as i32;
    let subpel_x = (mv.col & 7) as usize;
    let subpel_y = (mv.row & 7) as usize;

    // Convert to filter table index (0-15 for 1/8 pel positions)
    let filter_x = subpel_x * 2;
    let filter_y = subpel_y * 2;

    if filter_x == 0 && filter_y == 0 {
        // Full-pixel copy
        for y in 0..height {
            for x in 0..width {
                output[y * out_stride + x] =
                    ref_frame.get_pixel_y(full_x + x as i32, full_y + y as i32);
            }
        }
    } else if filter_y == 0 {
        // Horizontal-only filtering
        let filter = get_filter(filter_type, filter_x);
        for y in 0..height {
            let sy = full_y + y as i32;
            for x in 0..width {
                let mut sum = 0i32;
                for (k, &coeff) in filter.iter().enumerate() {
                    let sx = full_x + x as i32 + k as i32 - 3;
                    sum += ref_frame.get_pixel_y(sx, sy) as i32 * coeff as i32;
                }
                output[y * out_stride + x] = ((sum + 64) >> 7).clamp(0, 255) as u8;
            }
        }
    } else if filter_x == 0 {
        // Vertical-only filtering
        let filter = get_filter(filter_type, filter_y);
        for y in 0..height {
            for x in 0..width {
                let sx = full_x + x as i32;
                let mut sum = 0i32;
                for (k, &coeff) in filter.iter().enumerate() {
                    let sy = full_y + y as i32 + k as i32 - 3;
                    sum += ref_frame.get_pixel_y(sx, sy) as i32 * coeff as i32;
                }
                output[y * out_stride + x] = ((sum + 64) >> 7).clamp(0, 255) as u8;
            }
        }
    } else {
        // 2D filtering: horizontal first, then vertical
        let temp_height = height + 7;
        let mut temp = vec![0i16; temp_height * width];

        // Horizontal pass
        filter_horizontal(
            ref_frame,
            full_x,
            full_y - 3,
            width,
            temp_height,
            filter_x,
            filter_type,
            &mut temp,
            width,
        );

        // Vertical pass
        filter_vertical(
            &temp,
            width,
            width,
            height,
            filter_y,
            filter_type,
            output,
            out_stride,
        );
    }
}

/// Perform motion compensation for a chroma block
pub fn motion_compensate_chroma(
    ref_plane: &[u8],
    ref_stride: usize,
    ref_width: u32,
    ref_height: u32,
    mv: MotionVector,
    block_x: usize,
    block_y: usize,
    width: usize,
    height: usize,
    output: &mut [u8],
    out_stride: usize,
) {
    // Chroma MV is in 1/16-pel units (luma MV / 2)
    let mv_x = mv.col as i32;
    let mv_y = mv.row as i32;

    let full_x = block_x as i32 + (mv_x >> 4);
    let full_y = block_y as i32 + (mv_y >> 4);
    let subpel_x = (mv_x & 15) as usize;
    let subpel_y = (mv_y & 15) as usize;

    let max_x = ref_width as i32 - 1;
    let max_y = ref_height as i32 - 1;

    if subpel_x == 0 && subpel_y == 0 {
        // Full-pixel copy
        for y in 0..height {
            let sy = (full_y + y as i32).clamp(0, max_y) as usize;
            for x in 0..width {
                let sx = (full_x + x as i32).clamp(0, max_x) as usize;
                output[y * out_stride + x] = ref_plane[sy * ref_stride + sx];
            }
        }
    } else {
        // Bilinear interpolation for chroma
        let filter_h = &BILINEAR_FILTERS[subpel_x];
        let filter_v = &BILINEAR_FILTERS[subpel_y];

        for y in 0..height {
            for x in 0..width {
                let mut sum = 0i32;

                for fy in 0..2 {
                    let sy = (full_y + y as i32 + fy).clamp(0, max_y) as usize;
                    for fx in 0..2 {
                        let sx = (full_x + x as i32 + fx).clamp(0, max_x) as usize;
                        let pixel = ref_plane[sy * ref_stride + sx] as i32;
                        sum += pixel * filter_h[fx as usize] as i32 * filter_v[fy as usize] as i32;
                    }
                }

                output[y * out_stride + x] = ((sum + 8192) >> 14).clamp(0, 255) as u8;
            }
        }
    }
}

/// Compound prediction - average of two predictions
pub fn compound_predict(
    pred1: &[u8],
    pred2: &[u8],
    output: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) {
    for y in 0..height {
        for x in 0..width {
            let idx = y * stride + x;
            let p1 = pred1[idx] as u16;
            let p2 = pred2[idx] as u16;
            output[idx] = ((p1 + p2 + 1) >> 1) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dc_prediction() {
        let above = [100u8; 8];
        let left = [100u8; 8];

        let predictor = IntraPredictor::new(&above, &left, 100, true, true);

        let mut output = vec![0u8; 64];
        predictor.predict_dc(&mut output, 8, 8);

        // All values should be 100
        for &v in &output {
            assert_eq!(v, 100);
        }
    }

    #[test]
    fn test_v_prediction() {
        let above = [50u8, 60, 70, 80, 90, 100, 110, 120];
        let left = [0u8; 8];

        let predictor = IntraPredictor::new(&above, &left, 128, true, true);

        let mut output = vec![0u8; 64];
        predictor.predict_v(&mut output, 8, 8);

        // Each column should match the above value
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(output[y * 8 + x], above[x]);
            }
        }
    }

    #[test]
    fn test_h_prediction() {
        let above = [0u8; 8];
        let left = [50u8, 60, 70, 80, 90, 100, 110, 120];

        let predictor = IntraPredictor::new(&above, &left, 128, true, true);

        let mut output = vec![0u8; 64];
        predictor.predict_h(&mut output, 8, 8);

        // Each row should match the left value
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(output[y * 8 + x], left[y]);
            }
        }
    }

    #[test]
    fn test_tm_prediction() {
        let above = [100u8; 8];
        let left = [100u8; 8];

        let predictor = IntraPredictor::new(&above, &left, 100, true, true);

        let mut output = vec![0u8; 64];
        predictor.predict_tm(&mut output, 8, 8);

        // TM: left + above - top_left = 100 + 100 - 100 = 100
        for &v in &output {
            assert_eq!(v, 100);
        }
    }

    #[test]
    fn test_frame_buffer_creation() {
        let fb = FrameBuffer::new(1920, 1080);
        assert!(fb.y.len() >= 1920 * 1080);
        assert!(fb.u.len() >= 960 * 540);
    }

    #[test]
    fn test_compound_predict() {
        let pred1 = [100u8; 16];
        let pred2 = [200u8; 16];
        let mut output = [0u8; 16];

        compound_predict(&pred1, &pred2, &mut output, 4, 4, 4);

        // (100 + 200 + 1) >> 1 = 150
        for &v in &output {
            assert_eq!(v, 150);
        }
    }
}
