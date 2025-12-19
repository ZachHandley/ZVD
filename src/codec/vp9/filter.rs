//! VP9 Loop Filter (Deblocking)
//!
//! The loop filter removes blocking artifacts that occur at transform block
//! and prediction block boundaries. VP9 uses an adaptive filter that varies
//! based on edge properties.

use super::frame::LoopFilterParams;
use super::tables::RefFrame;

/// Filter information for a single edge
#[derive(Debug, Clone, Copy)]
pub struct EdgeFilterInfo {
    /// Filter level for this edge
    pub level: u8,
    /// Filter limit values
    pub limit: u8,
    pub blimit: u8,
    pub thresh: u8,
    /// Skip filtering
    pub skip: bool,
}

impl Default for EdgeFilterInfo {
    fn default() -> Self {
        EdgeFilterInfo {
            level: 0,
            limit: 0,
            blimit: 0,
            thresh: 0,
            skip: true,
        }
    }
}

/// Calculate filter level for a block edge
pub fn calc_filter_level(params: &LoopFilterParams, ref_frame: RefFrame, is_inter: bool) -> u8 {
    let base = params.level;

    if base == 0 {
        return 0;
    }

    let mut level = base as i32;

    // Apply reference frame delta
    if params.delta_enabled {
        let ref_idx = match ref_frame {
            RefFrame::None => 0,
            RefFrame::Last => 1,
            RefFrame::Golden => 2,
            RefFrame::AltRef => 3,
        };
        level += params.ref_deltas[ref_idx] as i32;

        // Apply mode delta (0 for zero_mv, 1 for non-zero)
        if is_inter {
            level += params.mode_deltas[1] as i32;
        } else {
            level += params.mode_deltas[0] as i32;
        }
    }

    level.clamp(0, 63) as u8
}

/// Calculate filter limit values from level and sharpness
pub fn calc_filter_limits(level: u8, sharpness: u8) -> (u8, u8, u8) {
    if level == 0 {
        return (0, 0, 0);
    }

    // Interior limit (limit)
    let mut interior_limit = level;
    if sharpness > 0 {
        interior_limit >>= if sharpness > 4 { 2 } else { 1 };
        let sharpness_limit = 9u8.saturating_sub(sharpness);
        interior_limit = interior_limit.min(sharpness_limit);
    }
    interior_limit = interior_limit.max(1);

    // Block limit (blimit) = 2 * level + interior_limit
    let block_limit = (2 * level as u16 + interior_limit as u16).min(255) as u8;

    // Threshold for high edge variance
    let thresh = if level >= 40 {
        2
    } else if level >= 20 {
        1
    } else {
        0
    };

    (interior_limit, block_limit, thresh)
}

/// Check if edge should be filtered
#[inline]
fn should_filter(p: &[i16], q: &[i16], limit: i16, blimit: i16) -> bool {
    // Check flatness condition
    let p1 = p[1];
    let p0 = p[0];
    let q0 = q[0];
    let q1 = q[1];

    // Edge test: abs(p0 - q0) * 2 + abs(p1 - q1) / 2 <= blimit
    let edge_diff = (p0 - q0).abs() * 2 + (p1 - q1).abs() / 2;
    if edge_diff > blimit {
        return false;
    }

    // Interior tests
    (p1 - p0).abs() <= limit && (q1 - q0).abs() <= limit
}

/// Check for high edge variance (determines filter width)
#[inline]
fn has_high_edge_variance(p: &[i16], q: &[i16], thresh: i16) -> bool {
    (p[1] - p[0]).abs() > thresh || (q[1] - q[0]).abs() > thresh
}

/// Check if we can use wide filter (flat region)
#[inline]
fn is_flat(p: &[i16], q: &[i16], thresh: i16) -> bool {
    // Check if region is flat enough for wide filter
    if p.len() < 4 || q.len() < 4 {
        return false;
    }

    (p[3] - p[0]).abs() <= thresh
        && (p[2] - p[0]).abs() <= thresh
        && (p[1] - p[0]).abs() <= thresh
        && (q[0] - q[1]).abs() <= thresh
        && (q[0] - q[2]).abs() <= thresh
        && (q[0] - q[3]).abs() <= thresh
}

/// 4-tap filter for normal edges
fn filter4(p: &mut [i16; 2], q: &mut [i16; 2]) {
    let p1 = p[1];
    let p0 = p[0];
    let q0 = q[0];
    let q1 = q[1];

    // Calculate filter value
    let filter_value = clamp_filter(3 * (q0 - p0) + (p1 - q1));

    // Apply filter
    let filter1 = (filter_value + 4) >> 3;
    let filter2 = (filter_value + 3) >> 3;

    p[0] = clamp_pixel(p0 + filter2);
    q[0] = clamp_pixel(q0 - filter1);

    // Adjust outer pixels
    let outer = (filter1 + 1) >> 1;
    p[1] = clamp_pixel(p1 + outer);
    q[1] = clamp_pixel(q1 - outer);
}

/// 8-tap filter for flat regions
fn filter8(p: &mut [i16; 4], q: &mut [i16; 4]) {
    let p3 = p[3];
    let p2 = p[2];
    let p1 = p[1];
    let p0 = p[0];
    let q0 = q[0];
    let q1 = q[1];
    let q2 = q[2];
    let q3 = q[3];

    // 7-tap filter
    p[2] = ((p3 + p3 + p3 + 2 * p2 + p1 + p0 + q0 + 4) >> 3) as i16;
    p[1] = ((p3 + p3 + p2 + 2 * p1 + p0 + q0 + q1 + 4) >> 3) as i16;
    p[0] = ((p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4) >> 3) as i16;
    q[0] = ((p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4) >> 3) as i16;
    q[1] = ((p1 + p0 + q0 + 2 * q1 + q2 + q3 + q3 + 4) >> 3) as i16;
    q[2] = ((p0 + q0 + q1 + 2 * q2 + q3 + q3 + q3 + 4) >> 3) as i16;
}

/// 16-tap filter for very flat regions (superblock boundaries)
fn filter16(p: &mut [i16; 8], q: &mut [i16; 8]) {
    // 15-tap filter using wider neighborhood
    let p7 = p[7];
    let p6 = p[6];
    let p5 = p[5];
    let p4 = p[4];
    let p3 = p[3];
    let p2 = p[2];
    let p1 = p[1];
    let p0 = p[0];
    let q0 = q[0];
    let q1 = q[1];
    let q2 = q[2];
    let q3 = q[3];
    let q4 = q[4];
    let q5 = q[5];
    let q6 = q[6];
    let q7 = q[7];

    p[6] = ((p7 * 7 + p6 * 2 + p5 + p4 + p3 + p2 + p1 + p0 + q0 + 8) >> 4) as i16;
    p[5] = ((p7 * 6 + p6 + p5 * 2 + p4 + p3 + p2 + p1 + p0 + q0 + q1 + 8) >> 4) as i16;
    p[4] = ((p7 * 5 + p6 + p5 + p4 * 2 + p3 + p2 + p1 + p0 + q0 + q1 + q2 + 8) >> 4) as i16;
    p[3] = ((p7 * 4 + p6 + p5 + p4 + p3 * 2 + p2 + p1 + p0 + q0 + q1 + q2 + q3 + 8) >> 4) as i16;
    p[2] =
        ((p7 * 3 + p6 + p5 + p4 + p3 + p2 * 2 + p1 + p0 + q0 + q1 + q2 + q3 + q4 + 8) >> 4) as i16;
    p[1] = ((p7 * 2 + p6 + p5 + p4 + p3 + p2 + p1 * 2 + p0 + q0 + q1 + q2 + q3 + q4 + q5 + 8) >> 4)
        as i16;
    p[0] = ((p7 + p6 + p5 + p4 + p3 + p2 + p1 + p0 * 2 + q0 + q1 + q2 + q3 + q4 + q5 + q6 + 8) >> 4)
        as i16;
    q[0] = ((p6 + p5 + p4 + p3 + p2 + p1 + p0 + q0 * 2 + q1 + q2 + q3 + q4 + q5 + q6 + q7 + 8) >> 4)
        as i16;
    q[1] = ((p5 + p4 + p3 + p2 + p1 + p0 + q0 + q1 * 2 + q2 + q3 + q4 + q5 + q6 + q7 * 2 + 8) >> 4)
        as i16;
    q[2] =
        ((p4 + p3 + p2 + p1 + p0 + q0 + q1 + q2 * 2 + q3 + q4 + q5 + q6 + q7 * 3 + 8) >> 4) as i16;
    q[3] = ((p3 + p2 + p1 + p0 + q0 + q1 + q2 + q3 * 2 + q4 + q5 + q6 + q7 * 4 + 8) >> 4) as i16;
    q[4] = ((p2 + p1 + p0 + q0 + q1 + q2 + q3 + q4 * 2 + q5 + q6 + q7 * 5 + 8) >> 4) as i16;
    q[5] = ((p1 + p0 + q0 + q1 + q2 + q3 + q4 + q5 * 2 + q6 + q7 * 6 + 8) >> 4) as i16;
    q[6] = ((p0 + q0 + q1 + q2 + q3 + q4 + q5 + q6 * 2 + q7 * 7 + 8) >> 4) as i16;
}

#[inline]
fn clamp_filter(value: i16) -> i16 {
    value.clamp(-128, 127)
}

#[inline]
fn clamp_pixel(value: i16) -> i16 {
    value.clamp(0, 255)
}

/// Apply loop filter to a vertical edge
pub fn filter_vertical_edge(
    pixels: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    height: usize,
    level: u8,
    sharpness: u8,
) {
    if level == 0 {
        return;
    }

    let (limit, blimit, thresh) = calc_filter_limits(level, sharpness);
    let limit = limit as i16;
    let blimit = blimit as i16;
    let thresh = thresh as i16;

    for row in 0..height {
        let row_start = (y + row) * stride;

        // Get pixels around the edge (4 on each side)
        let mut p = [0i16; 4];
        let mut q = [0i16; 4];

        for i in 0..4 {
            if x >= i + 1 {
                p[i] = pixels[row_start + x - i - 1] as i16;
            }
            if x + i < stride {
                q[i] = pixels[row_start + x + i] as i16;
            }
        }

        // Check if we should filter
        if !should_filter(&p[..2], &q[..2], limit, blimit) {
            continue;
        }

        // Determine filter type
        let hev = has_high_edge_variance(&p[..2], &q[..2], thresh);

        if hev {
            // Use 4-tap filter
            let mut p2 = [p[1], p[0]];
            let mut q2 = [q[0], q[1]];
            filter4(&mut p2, &mut q2);
            p[1] = p2[0];
            p[0] = p2[1];
            q[0] = q2[0];
            q[1] = q2[1];
        } else if is_flat(&p, &q, 1) {
            // Use 8-tap filter
            filter8(&mut p, &mut q);
        } else {
            // Use 4-tap filter
            let mut p2 = [p[1], p[0]];
            let mut q2 = [q[0], q[1]];
            filter4(&mut p2, &mut q2);
            p[1] = p2[0];
            p[0] = p2[1];
            q[0] = q2[0];
            q[1] = q2[1];
        }

        // Write back
        for i in 0..4 {
            if x >= i + 1 {
                pixels[row_start + x - i - 1] = p[i] as u8;
            }
            if x + i < stride {
                pixels[row_start + x + i] = q[i] as u8;
            }
        }
    }
}

/// Apply loop filter to a horizontal edge
pub fn filter_horizontal_edge(
    pixels: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    width: usize,
    level: u8,
    sharpness: u8,
) {
    if level == 0 {
        return;
    }

    let (limit, blimit, thresh) = calc_filter_limits(level, sharpness);
    let limit = limit as i16;
    let blimit = blimit as i16;
    let thresh = thresh as i16;

    for col in 0..width {
        let col_x = x + col;

        // Get pixels around the edge
        let mut p = [0i16; 4];
        let mut q = [0i16; 4];

        for i in 0..4 {
            if y >= i + 1 {
                p[i] = pixels[(y - i - 1) * stride + col_x] as i16;
            }
            if y + i < pixels.len() / stride {
                q[i] = pixels[(y + i) * stride + col_x] as i16;
            }
        }

        // Check if we should filter
        if !should_filter(&p[..2], &q[..2], limit, blimit) {
            continue;
        }

        let hev = has_high_edge_variance(&p[..2], &q[..2], thresh);

        if hev {
            let mut p2 = [p[1], p[0]];
            let mut q2 = [q[0], q[1]];
            filter4(&mut p2, &mut q2);
            p[1] = p2[0];
            p[0] = p2[1];
            q[0] = q2[0];
            q[1] = q2[1];
        } else if is_flat(&p, &q, 1) {
            filter8(&mut p, &mut q);
        } else {
            let mut p2 = [p[1], p[0]];
            let mut q2 = [q[0], q[1]];
            filter4(&mut p2, &mut q2);
            p[1] = p2[0];
            p[0] = p2[1];
            q[0] = q2[0];
            q[1] = q2[1];
        }

        // Write back
        for i in 0..4 {
            if y >= i + 1 {
                pixels[(y - i - 1) * stride + col_x] = p[i] as u8;
            }
            if y + i < pixels.len() / stride {
                pixels[(y + i) * stride + col_x] = q[i] as u8;
            }
        }
    }
}

/// Apply loop filter to entire frame
pub fn apply_loop_filter(
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    y_stride: usize,
    uv_stride: usize,
    width: u32,
    height: u32,
    params: &LoopFilterParams,
) {
    if params.level == 0 {
        return;
    }

    let level = params.level;
    let sharpness = params.sharpness;

    // Filter 8x8 block boundaries for luma
    let block_size = 8usize;

    // Vertical edges
    for block_y in 0..(height as usize / block_size) {
        for block_x in 1..(width as usize / block_size) {
            let x = block_x * block_size;
            let y = block_y * block_size;
            filter_vertical_edge(y_plane, y_stride, x, y, block_size, level, sharpness);
        }
    }

    // Horizontal edges
    for block_y in 1..(height as usize / block_size) {
        for block_x in 0..(width as usize / block_size) {
            let x = block_x * block_size;
            let y = block_y * block_size;
            filter_horizontal_edge(y_plane, y_stride, x, y, block_size, level, sharpness);
        }
    }

    // Chroma (half resolution)
    let chroma_block = 4usize;
    let chroma_width = width / 2;
    let chroma_height = height / 2;

    // U plane
    for block_y in 0..(chroma_height as usize / chroma_block) {
        for block_x in 1..(chroma_width as usize / chroma_block) {
            let x = block_x * chroma_block;
            let y = block_y * chroma_block;
            filter_vertical_edge(u_plane, uv_stride, x, y, chroma_block, level, sharpness);
        }
    }
    for block_y in 1..(chroma_height as usize / chroma_block) {
        for block_x in 0..(chroma_width as usize / chroma_block) {
            let x = block_x * chroma_block;
            let y = block_y * chroma_block;
            filter_horizontal_edge(u_plane, uv_stride, x, y, chroma_block, level, sharpness);
        }
    }

    // V plane
    for block_y in 0..(chroma_height as usize / chroma_block) {
        for block_x in 1..(chroma_width as usize / chroma_block) {
            let x = block_x * chroma_block;
            let y = block_y * chroma_block;
            filter_vertical_edge(v_plane, uv_stride, x, y, chroma_block, level, sharpness);
        }
    }
    for block_y in 1..(chroma_height as usize / chroma_block) {
        for block_x in 0..(chroma_width as usize / chroma_block) {
            let x = block_x * chroma_block;
            let y = block_y * chroma_block;
            filter_horizontal_edge(v_plane, uv_stride, x, y, chroma_block, level, sharpness);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calc_filter_limits() {
        let (limit, blimit, thresh) = calc_filter_limits(30, 0);
        assert!(limit > 0);
        assert!(blimit > limit);
        assert_eq!(thresh, 1);

        let (limit, blimit, thresh) = calc_filter_limits(50, 0);
        assert_eq!(thresh, 2);

        let (limit, blimit, thresh) = calc_filter_limits(10, 0);
        assert_eq!(thresh, 0);
    }

    #[test]
    fn test_filter_level_zero() {
        let (limit, blimit, thresh) = calc_filter_limits(0, 0);
        assert_eq!(limit, 0);
        assert_eq!(blimit, 0);
        assert_eq!(thresh, 0);
    }

    #[test]
    fn test_sharpness_effect() {
        let (limit1, _, _) = calc_filter_limits(30, 0);
        let (limit2, _, _) = calc_filter_limits(30, 4);
        let (limit3, _, _) = calc_filter_limits(30, 7);

        // Higher sharpness should result in lower limit
        assert!(limit1 >= limit2);
        assert!(limit2 >= limit3);
    }

    #[test]
    fn test_should_filter() {
        // Smooth edge should be filtered
        let p = [100i16, 102];
        let q = [98i16, 96];
        assert!(should_filter(&p, &q, 10, 30));

        // Sharp edge should not be filtered
        let p = [100i16, 150];
        let q = [50i16, 0];
        assert!(!should_filter(&p, &q, 10, 30));
    }

    #[test]
    fn test_filter4() {
        let mut p = [100i16, 105];
        let mut q = [95i16, 90];

        filter4(&mut p, &mut q);

        // Values should be brought closer together
        assert!((p[0] - q[0]).abs() < (105 - 95));
    }
}
