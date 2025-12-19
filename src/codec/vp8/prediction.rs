//! VP8 intra and inter prediction
//!
//! This module implements all prediction modes used in VP8:
//! - 16x16 and 8x8 intra prediction (DC, V, H, TM)
//! - 4x4 intra prediction (10 modes)
//! - Inter prediction with motion compensation

use super::tables::{IntraMode16x16, IntraMode4x4, MotionVector, BILINEAR_FILTERS, SUBPEL_FILTERS};

/// Perform 16x16 intra prediction
///
/// # Arguments
/// * `mode` - Prediction mode (DC, V, H, or TM)
/// * `above` - 16 pixels from the row above
/// * `left` - 16 pixels from the column to the left
/// * `above_left` - Corner pixel at (-1, -1)
/// * `above_available` - Whether above pixels are available
/// * `left_available` - Whether left pixels are available
/// * `output` - Output buffer (16x16 = 256 bytes)
/// * `stride` - Output buffer stride
pub fn predict_16x16(
    mode: IntraMode16x16,
    above: &[u8],
    left: &[u8],
    above_left: u8,
    above_available: bool,
    left_available: bool,
    output: &mut [u8],
    stride: usize,
) {
    match mode {
        IntraMode16x16::DcPred => {
            let dc = if above_available && left_available {
                let sum: u32 = above[..16].iter().map(|&x| x as u32).sum::<u32>()
                    + left[..16].iter().map(|&x| x as u32).sum::<u32>();
                ((sum + 16) >> 5) as u8
            } else if above_available {
                let sum: u32 = above[..16].iter().map(|&x| x as u32).sum();
                ((sum + 8) >> 4) as u8
            } else if left_available {
                let sum: u32 = left[..16].iter().map(|&x| x as u32).sum();
                ((sum + 8) >> 4) as u8
            } else {
                128
            };

            for y in 0..16 {
                for x in 0..16 {
                    output[y * stride + x] = dc;
                }
            }
        }
        IntraMode16x16::VPred => {
            for y in 0..16 {
                output[y * stride..y * stride + 16].copy_from_slice(&above[..16]);
            }
        }
        IntraMode16x16::HPred => {
            for y in 0..16 {
                let val = left[y];
                for x in 0..16 {
                    output[y * stride + x] = val;
                }
            }
        }
        IntraMode16x16::TmPred => {
            for y in 0..16 {
                for x in 0..16 {
                    let val = left[y] as i16 + above[x] as i16 - above_left as i16;
                    output[y * stride + x] = val.clamp(0, 255) as u8;
                }
            }
        }
    }
}

/// Perform 8x8 chroma intra prediction (same modes as 16x16)
pub fn predict_8x8_chroma(
    mode: IntraMode16x16,
    above: &[u8],
    left: &[u8],
    above_left: u8,
    above_available: bool,
    left_available: bool,
    output: &mut [u8],
    stride: usize,
) {
    match mode {
        IntraMode16x16::DcPred => {
            let dc = if above_available && left_available {
                let sum: u32 = above[..8].iter().map(|&x| x as u32).sum::<u32>()
                    + left[..8].iter().map(|&x| x as u32).sum::<u32>();
                ((sum + 8) >> 4) as u8
            } else if above_available {
                let sum: u32 = above[..8].iter().map(|&x| x as u32).sum();
                ((sum + 4) >> 3) as u8
            } else if left_available {
                let sum: u32 = left[..8].iter().map(|&x| x as u32).sum();
                ((sum + 4) >> 3) as u8
            } else {
                128
            };

            for y in 0..8 {
                for x in 0..8 {
                    output[y * stride + x] = dc;
                }
            }
        }
        IntraMode16x16::VPred => {
            for y in 0..8 {
                output[y * stride..y * stride + 8].copy_from_slice(&above[..8]);
            }
        }
        IntraMode16x16::HPred => {
            for y in 0..8 {
                let val = left[y];
                for x in 0..8 {
                    output[y * stride + x] = val;
                }
            }
        }
        IntraMode16x16::TmPred => {
            for y in 0..8 {
                for x in 0..8 {
                    let val = left[y] as i16 + above[x] as i16 - above_left as i16;
                    output[y * stride + x] = val.clamp(0, 255) as u8;
                }
            }
        }
    }
}

/// Perform 4x4 intra prediction
///
/// # Arguments
/// * `mode` - One of 10 4x4 intra prediction modes
/// * `above` - 8 pixels (4 above + 4 above-right)
/// * `left` - 4 pixels from left column
/// * `above_left` - Corner pixel
/// * `output` - Output buffer (4x4 = 16 bytes)
/// * `stride` - Output buffer stride
pub fn predict_4x4(
    mode: IntraMode4x4,
    above: &[u8],
    left: &[u8],
    above_left: u8,
    output: &mut [u8],
    stride: usize,
) {
    match mode {
        IntraMode4x4::BDcPred => {
            let sum: u32 = above[..4].iter().map(|&x| x as u32).sum::<u32>()
                + left[..4].iter().map(|&x| x as u32).sum::<u32>();
            let dc = ((sum + 4) >> 3) as u8;
            for y in 0..4 {
                for x in 0..4 {
                    output[y * stride + x] = dc;
                }
            }
        }
        IntraMode4x4::BTmPred => {
            for y in 0..4 {
                for x in 0..4 {
                    let val = left[y] as i16 + above[x] as i16 - above_left as i16;
                    output[y * stride + x] = val.clamp(0, 255) as u8;
                }
            }
        }
        IntraMode4x4::BVePred => {
            // Vertical with smoothing
            for x in 0..4 {
                let avg = if x == 0 {
                    (above_left as u16 + 2 * above[0] as u16 + above[1] as u16 + 2) >> 2
                } else if x == 3 {
                    (above[2] as u16 + 2 * above[3] as u16 + above[4] as u16 + 2) >> 2
                } else {
                    (above[x - 1] as u16 + 2 * above[x] as u16 + above[x + 1] as u16 + 2) >> 2
                };
                for y in 0..4 {
                    output[y * stride + x] = avg as u8;
                }
            }
        }
        IntraMode4x4::BHePred => {
            // Horizontal with smoothing
            for y in 0..4 {
                let avg = if y == 0 {
                    (above_left as u16 + 2 * left[0] as u16 + left[1] as u16 + 2) >> 2
                } else if y == 3 {
                    (left[2] as u16 + 3 * left[3] as u16 + 2) >> 2
                } else {
                    (left[y - 1] as u16 + 2 * left[y] as u16 + left[y + 1] as u16 + 2) >> 2
                };
                for x in 0..4 {
                    output[y * stride + x] = avg as u8;
                }
            }
        }
        IntraMode4x4::BLdPred => {
            // Down-left diagonal (45 degrees)
            for y in 0..4 {
                for x in 0..4 {
                    let idx = x + y;
                    let val = if idx < 6 {
                        (above[idx] as u16 + 2 * above[idx + 1] as u16 + above[idx + 2] as u16 + 2)
                            >> 2
                    } else {
                        above[7] as u16
                    };
                    output[y * stride + x] = val as u8;
                }
            }
        }
        IntraMode4x4::BRdPred => {
            // Down-right diagonal
            let mut edge = [0u8; 9];
            edge[0] = left[3];
            edge[1] = left[2];
            edge[2] = left[1];
            edge[3] = left[0];
            edge[4] = above_left;
            edge[5..9].copy_from_slice(&above[..4]);

            for y in 0..4 {
                for x in 0..4 {
                    let idx = 4 - y + x;
                    let val =
                        (edge[idx - 1] as u16 + 2 * edge[idx] as u16 + edge[idx + 1] as u16 + 2)
                            >> 2;
                    output[y * stride + x] = val as u8;
                }
            }
        }
        IntraMode4x4::BVrPred => {
            // Vertical-right
            let mut edge = [0u8; 9];
            edge[0] = left[3];
            edge[1] = left[2];
            edge[2] = left[1];
            edge[3] = left[0];
            edge[4] = above_left;
            edge[5..9].copy_from_slice(&above[..4]);

            for y in 0..4 {
                for x in 0..4 {
                    let zy = 2 * y as i32;
                    let zx = 2 * x as i32;
                    let idx = zx - zy;
                    let val = if idx >= -1 {
                        let base = (4 + (idx >> 1)) as usize;
                        if idx & 1 == 0 {
                            (edge[base] as u16 + edge[base + 1] as u16 + 1) >> 1
                        } else {
                            (edge[base] as u16
                                + 2 * edge[base + 1] as u16
                                + edge[base + 2] as u16
                                + 2)
                                >> 2
                        }
                    } else {
                        let base = (4 - y) as usize;
                        (edge[base - 1] as u16 + 2 * edge[base] as u16 + edge[base + 1] as u16 + 2)
                            >> 2
                    };
                    output[y * stride + x] = val as u8;
                }
            }
        }
        IntraMode4x4::BVlPred => {
            // Vertical-left
            for y in 0..4 {
                for x in 0..4 {
                    let idx = x + (y >> 1);
                    let val = if y & 1 == 0 {
                        (above[idx] as u16 + above[idx + 1] as u16 + 1) >> 1
                    } else {
                        (above[idx] as u16 + 2 * above[idx + 1] as u16 + above[idx + 2] as u16 + 2)
                            >> 2
                    };
                    output[y * stride + x] = val as u8;
                }
            }
        }
        IntraMode4x4::BHdPred => {
            // Horizontal-down
            let mut edge = [0u8; 9];
            edge[0] = left[3];
            edge[1] = left[2];
            edge[2] = left[1];
            edge[3] = left[0];
            edge[4] = above_left;
            edge[5..9].copy_from_slice(&above[..4]);

            for y in 0..4 {
                for x in 0..4 {
                    let idx = 2 * y as i32 - x as i32;
                    let val = if idx >= 0 {
                        let base = (4 - (idx >> 1) - 1) as usize;
                        if idx & 1 == 0 {
                            (edge[base] as u16 + edge[base + 1] as u16 + 1) >> 1
                        } else {
                            (edge[base] as u16
                                + 2 * edge[base + 1] as u16
                                + edge[base + 2] as u16
                                + 2)
                                >> 2
                        }
                    } else {
                        let base = (4 + x) as usize;
                        (edge[base - 1] as u16 + 2 * edge[base] as u16 + edge[base + 1] as u16 + 2)
                            >> 2
                    };
                    output[y * stride + x] = val as u8;
                }
            }
        }
        IntraMode4x4::BHuPred => {
            // Horizontal-up
            for y in 0..4 {
                for x in 0..4 {
                    let z = x as i32 + 2 * y as i32;
                    let val = if z < 6 {
                        let idx = z >> 1;
                        if z & 1 == 0 {
                            (left[idx as usize] as u16 + left[idx as usize + 1] as u16 + 1) >> 1
                        } else {
                            (left[idx as usize] as u16
                                + 2 * left[idx as usize + 1] as u16
                                + left[(idx as usize + 2).min(3)] as u16
                                + 2)
                                >> 2
                        }
                    } else if z == 6 {
                        (left[3] as u16 + left[3] as u16 + 1) >> 1
                    } else {
                        left[3] as u16
                    };
                    output[y * stride + x] = val as u8;
                }
            }
        }
    }
}

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
    /// Create a new frame buffer with given dimensions
    pub fn new(width: u32, height: u32) -> Self {
        // Align to macroblock boundaries
        let mb_width = (width + 15) & !15;
        let mb_height = (height + 15) & !15;

        let y_stride = mb_width as usize;
        let uv_stride = (mb_width as usize) >> 1;

        let y_size = y_stride * mb_height as usize;
        let uv_size = uv_stride * (mb_height as usize >> 1);

        FrameBuffer {
            width,
            height,
            y: vec![128; y_size], // Initialize to mid-gray
            u: vec![128; uv_size],
            v: vec![128; uv_size],
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
}

/// Perform motion compensation for a luma block
///
/// # Arguments
/// * `ref_frame` - Reference frame buffer
/// * `mv` - Motion vector in 1/4 pixel units
/// * `block_x` - Block X position in pixels
/// * `block_y` - Block Y position in pixels
/// * `block_size` - Block size (4, 8, or 16)
/// * `output` - Output buffer
/// * `output_stride` - Output buffer stride
pub fn motion_compensate_luma(
    ref_frame: &FrameBuffer,
    mv: MotionVector,
    block_x: usize,
    block_y: usize,
    block_size: usize,
    output: &mut [u8],
    output_stride: usize,
) {
    // MV is in 1/4 pixel units
    let full_x = block_x as i32 + (mv.x as i32 >> 2);
    let full_y = block_y as i32 + (mv.y as i32 >> 2);
    let subpel_x = (mv.x & 3) as usize;
    let subpel_y = (mv.y & 3) as usize;

    // Convert 1/4 pel to 1/8 pel for filter lookup
    let filter_x = subpel_x * 2;
    let filter_y = subpel_y * 2;

    if filter_x == 0 && filter_y == 0 {
        // Full-pixel copy
        copy_block(
            ref_frame,
            full_x,
            full_y,
            block_size,
            output,
            output_stride,
            true,
        );
    } else if filter_y == 0 {
        // Horizontal-only interpolation
        filter_horizontal_block(
            ref_frame,
            full_x,
            full_y,
            block_size,
            filter_x,
            output,
            output_stride,
        );
    } else if filter_x == 0 {
        // Vertical-only interpolation
        filter_vertical_block(
            ref_frame,
            full_x,
            full_y,
            block_size,
            filter_y,
            output,
            output_stride,
        );
    } else {
        // 2D interpolation: horizontal first, then vertical
        let mut temp = vec![0i16; (block_size + 5) * block_size];

        // Horizontal filtering into temp buffer
        filter_horizontal_to_temp(
            ref_frame,
            full_x,
            full_y - 2,
            block_size,
            block_size + 5,
            filter_x,
            &mut temp,
            block_size,
        );

        // Vertical filtering from temp to output
        filter_vertical_from_temp(
            &temp,
            block_size,
            block_size,
            filter_y,
            output,
            output_stride,
        );
    }
}

/// Copy a block from reference frame with boundary clamping
fn copy_block(
    ref_frame: &FrameBuffer,
    src_x: i32,
    src_y: i32,
    block_size: usize,
    output: &mut [u8],
    output_stride: usize,
    is_luma: bool,
) {
    let (src, src_stride, max_x, max_y) = if is_luma {
        (
            &ref_frame.y,
            ref_frame.y_stride,
            ref_frame.width as i32 - 1,
            ref_frame.height as i32 - 1,
        )
    } else {
        (
            &ref_frame.u, // Caller will handle U/V
            ref_frame.uv_stride,
            (ref_frame.width as i32 >> 1) - 1,
            (ref_frame.height as i32 >> 1) - 1,
        )
    };

    for y in 0..block_size {
        let sy = (src_y + y as i32).clamp(0, max_y) as usize;
        for x in 0..block_size {
            let sx = (src_x + x as i32).clamp(0, max_x) as usize;
            output[y * output_stride + x] = src[sy * src_stride + sx];
        }
    }
}

/// Horizontal-only 6-tap filtering
fn filter_horizontal_block(
    ref_frame: &FrameBuffer,
    src_x: i32,
    src_y: i32,
    block_size: usize,
    filter_x: usize,
    output: &mut [u8],
    output_stride: usize,
) {
    let filter = &SUBPEL_FILTERS[filter_x];
    let max_x = ref_frame.width as i32 - 1;
    let max_y = ref_frame.height as i32 - 1;

    for y in 0..block_size {
        let sy = (src_y + y as i32).clamp(0, max_y) as usize;
        for x in 0..block_size {
            let mut sum = 0i32;
            for (i, &coeff) in filter.iter().enumerate() {
                let sx = (src_x + x as i32 + i as i32 - 2).clamp(0, max_x) as usize;
                sum += ref_frame.y[sy * ref_frame.y_stride + sx] as i32 * coeff as i32;
            }
            output[y * output_stride + x] = ((sum + 64) >> 7).clamp(0, 255) as u8;
        }
    }
}

/// Vertical-only 6-tap filtering
fn filter_vertical_block(
    ref_frame: &FrameBuffer,
    src_x: i32,
    src_y: i32,
    block_size: usize,
    filter_y: usize,
    output: &mut [u8],
    output_stride: usize,
) {
    let filter = &SUBPEL_FILTERS[filter_y];
    let max_x = ref_frame.width as i32 - 1;
    let max_y = ref_frame.height as i32 - 1;

    for y in 0..block_size {
        for x in 0..block_size {
            let sx = (src_x + x as i32).clamp(0, max_x) as usize;
            let mut sum = 0i32;
            for (i, &coeff) in filter.iter().enumerate() {
                let sy = (src_y + y as i32 + i as i32 - 2).clamp(0, max_y) as usize;
                sum += ref_frame.y[sy * ref_frame.y_stride + sx] as i32 * coeff as i32;
            }
            output[y * output_stride + x] = ((sum + 64) >> 7).clamp(0, 255) as u8;
        }
    }
}

/// Horizontal filtering to temporary buffer (for 2D interpolation)
fn filter_horizontal_to_temp(
    ref_frame: &FrameBuffer,
    src_x: i32,
    src_y: i32,
    width: usize,
    height: usize,
    filter_x: usize,
    temp: &mut [i16],
    temp_stride: usize,
) {
    let filter = &SUBPEL_FILTERS[filter_x];
    let max_x = ref_frame.width as i32 - 1;
    let max_y = ref_frame.height as i32 - 1;

    for y in 0..height {
        let sy = (src_y + y as i32).clamp(0, max_y) as usize;
        for x in 0..width {
            let mut sum = 0i32;
            for (i, &coeff) in filter.iter().enumerate() {
                let sx = (src_x + x as i32 + i as i32 - 2).clamp(0, max_x) as usize;
                sum += ref_frame.y[sy * ref_frame.y_stride + sx] as i32 * coeff as i32;
            }
            temp[y * temp_stride + x] = ((sum + 64) >> 7) as i16;
        }
    }
}

/// Vertical filtering from temporary buffer (for 2D interpolation)
fn filter_vertical_from_temp(
    temp: &[i16],
    temp_stride: usize,
    block_size: usize,
    filter_y: usize,
    output: &mut [u8],
    output_stride: usize,
) {
    let filter = &SUBPEL_FILTERS[filter_y];
    let height = temp.len() / temp_stride;

    for y in 0..block_size {
        for x in 0..block_size {
            let mut sum = 0i32;
            for (i, &coeff) in filter.iter().enumerate() {
                let ty = (y + i).min(height - 1);
                sum += temp[ty * temp_stride + x] as i32 * coeff as i32;
            }
            output[y * output_stride + x] = ((sum + 64) >> 7).clamp(0, 255) as u8;
        }
    }
}

/// Perform motion compensation for chroma (U or V)
pub fn motion_compensate_chroma(
    ref_plane: &[u8],
    ref_stride: usize,
    ref_width: u32,
    ref_height: u32,
    mv: MotionVector,
    block_x: usize,
    block_y: usize,
    block_size: usize,
    output: &mut [u8],
    output_stride: usize,
) {
    // Chroma MV is in 1/8 pixel units (luma MV / 2)
    let mv_x = mv.x as i32;
    let mv_y = mv.y as i32;

    let full_x = block_x as i32 + (mv_x >> 3);
    let full_y = block_y as i32 + (mv_y >> 3);
    let subpel_x = (mv_x & 7) as usize;
    let subpel_y = (mv_y & 7) as usize;

    let max_x = ref_width as i32 - 1;
    let max_y = ref_height as i32 - 1;

    if subpel_x == 0 && subpel_y == 0 {
        // Full-pixel copy
        for y in 0..block_size {
            let sy = (full_y + y as i32).clamp(0, max_y) as usize;
            for x in 0..block_size {
                let sx = (full_x + x as i32).clamp(0, max_x) as usize;
                output[y * output_stride + x] = ref_plane[sy * ref_stride + sx];
            }
        }
    } else {
        // Bilinear interpolation for chroma
        let filter_h = &BILINEAR_FILTERS[subpel_x];
        let filter_v = &BILINEAR_FILTERS[subpel_y];

        for y in 0..block_size {
            for x in 0..block_size {
                let mut sum = 0i32;

                for fy in 0..2 {
                    let sy = (full_y + y as i32 + fy).clamp(0, max_y) as usize;
                    for fx in 0..2 {
                        let sx = (full_x + x as i32 + fx).clamp(0, max_x) as usize;
                        let pixel = ref_plane[sy * ref_stride + sx] as i32;
                        sum += pixel * filter_h[fx as usize] as i32 * filter_v[fy as usize] as i32;
                    }
                }

                output[y * output_stride + x] = ((sum + 8192) >> 14).clamp(0, 255) as u8;
            }
        }
    }
}

/// Compute chroma MV from 4 luma subblock MVs (averaged)
pub fn compute_chroma_mv(luma_mvs: &[MotionVector; 4]) -> MotionVector {
    let sum_x: i32 = luma_mvs.iter().map(|mv| mv.x as i32).sum();
    let sum_y: i32 = luma_mvs.iter().map(|mv| mv.y as i32).sum();

    // Average with rounding, result in 1/8 pel for chroma
    MotionVector {
        x: ((sum_x + 2) >> 2) as i16,
        y: ((sum_y + 2) >> 2) as i16,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict_16x16_dc() {
        let above = [100u8; 16];
        let left = [100u8; 16];
        let mut output = [0u8; 256];

        predict_16x16(
            IntraMode16x16::DcPred,
            &above,
            &left,
            100,
            true,
            true,
            &mut output,
            16,
        );

        // All values should be 100
        for &val in &output {
            assert_eq!(val, 100);
        }
    }

    #[test]
    fn test_predict_16x16_v() {
        let above = [50u8; 16];
        let left = [100u8; 16];
        let mut output = [0u8; 256];

        predict_16x16(
            IntraMode16x16::VPred,
            &above,
            &left,
            75,
            true,
            true,
            &mut output,
            16,
        );

        // All rows should equal above
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(output[y * 16 + x], 50);
            }
        }
    }

    #[test]
    fn test_predict_16x16_h() {
        let above = [100u8; 16];
        let mut left = [0u8; 16];
        for i in 0..16 {
            left[i] = (i * 10) as u8;
        }
        let mut output = [0u8; 256];

        predict_16x16(
            IntraMode16x16::HPred,
            &above,
            &left,
            0,
            true,
            true,
            &mut output,
            16,
        );

        // Each row should be filled with corresponding left value
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(output[y * 16 + x], (y * 10) as u8);
            }
        }
    }

    #[test]
    fn test_frame_buffer_creation() {
        let fb = FrameBuffer::new(1920, 1080);
        assert!(fb.y.len() >= 1920 * 1080);
        assert!(fb.u.len() >= 960 * 540);
    }

    #[test]
    fn test_chroma_mv_computation() {
        let mvs = [
            MotionVector::new(4, 8),
            MotionVector::new(8, 12),
            MotionVector::new(4, 8),
            MotionVector::new(8, 12),
        ];

        let chroma_mv = compute_chroma_mv(&mvs);
        // Average of (4+8+4+8)/4 = 6, (8+12+8+12)/4 = 10
        assert_eq!(chroma_mv.x, 6);
        assert_eq!(chroma_mv.y, 10);
    }
}
