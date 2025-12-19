//! VP8 loop filter implementation
//!
//! The loop filter is an in-loop deblocking filter that removes blocking
//! artifacts at macroblock and subblock boundaries. VP8 supports two filter
//! types: simple and normal.

use super::tables::{FilterType, RefFrame};

/// Loop filter parameters
#[derive(Debug, Clone, Copy, Default)]
pub struct LoopFilterParams {
    pub filter_level: u8,
    pub sharpness: u8,
    pub filter_type: FilterType,
    pub mode_ref_lf_delta_enabled: bool,
    pub ref_lf_deltas: [i8; 4],
    pub mode_lf_deltas: [i8; 4],
}

/// Macroblock information needed for loop filtering
#[derive(Debug, Clone, Copy, Default)]
pub struct MacroblockFilterInfo {
    pub ref_frame: RefFrame,
    pub is_intra: bool,
    pub is_skip: bool,
    pub has_nonzero_y2: bool,
    pub segment_id: u8,
    pub filter_level_override: Option<u8>,
}

/// Filter limits computed from level and sharpness
#[derive(Debug, Clone, Copy)]
pub struct FilterLimits {
    pub blimit: i32,     // Block edge limit
    pub limit: i32,      // Interior limit
    pub hev_thresh: i32, // High edge variance threshold
}

impl FilterLimits {
    /// Calculate filter limits from level and sharpness
    pub fn from_level_sharpness(level: u8, sharpness: u8) -> Self {
        if level == 0 {
            return FilterLimits {
                blimit: 0,
                limit: 0,
                hev_thresh: 0,
            };
        }

        // Block edge limit
        let blimit = if sharpness > 0 {
            let limit = if sharpness > 4 {
                ((level + 2) as i32).min((9 - sharpness as i32).max(1))
            } else {
                ((level + 2) as i32).min(9 - sharpness as i32)
            };
            limit.max(1)
        } else {
            ((level + 2) as i32 * 2).max(1)
        };

        // Interior limit
        let limit = if sharpness > 0 {
            let shift = if sharpness > 4 { 2 } else { 1 };
            let l = ((level as i32) >> shift).max(1);
            l.min(9 - sharpness as i32)
        } else {
            level as i32
        };

        // High edge variance threshold
        let hev_thresh = if level >= 40 {
            2
        } else if level >= 15 {
            1
        } else {
            0
        };

        FilterLimits {
            blimit,
            limit: limit.max(1),
            hev_thresh,
        }
    }
}

/// Clamp signed value to [-128, 127] range
#[inline]
fn clamp_signed(val: i32) -> i32 {
    val.clamp(-128, 127)
}

/// Common filter function used by both simple and normal filters
#[inline]
fn filter_common(p1: i32, p0: i32, q0: i32, q1: i32, hev: bool) -> (i32, i32, i32, i32) {
    let p1_out;
    let q1_out;

    // Calculate filter value
    let a = clamp_signed(clamp_signed(p1 - q1) + 3 * (q0 - p0));

    // Filter for pixels adjacent to edge
    let a1 = clamp_signed((a + 4) >> 3);
    let a2 = clamp_signed((a + 3) >> 3);

    let p0_out = (p0 + a2).clamp(0, 255);
    let q0_out = (q0 - a1).clamp(0, 255);

    if hev {
        // High edge variance: only filter adjacent pixels
        p1_out = p1;
        q1_out = q1;
    } else {
        // Low edge variance: filter outer pixels too
        let a3 = (a1 + 1) >> 1;
        p1_out = (p1 + a3).clamp(0, 255);
        q1_out = (q1 - a3).clamp(0, 255);
    }

    (p1_out, p0_out, q0_out, q1_out)
}

/// Apply simple loop filter to a single edge
///
/// The simple filter only uses 2 pixels on each side of the edge.
pub fn simple_loop_filter(p: &mut [u8], offset: usize, stride: usize, blimit: i32, count: usize) {
    for i in 0..count {
        let idx = offset + i * stride;
        if idx + 3 >= p.len() {
            break;
        }

        let p1 = p[idx] as i32;
        let p0 = p[idx + 1] as i32;
        let q0 = p[idx + 2] as i32;
        let q1 = p[idx + 3] as i32;

        // Filter mask: check if edge is "flat enough" to filter
        let mask = (p0 - q0).abs() * 2 + ((p1 - q1).abs() >> 1);
        if mask > blimit {
            continue;
        }

        // Apply filter
        let a = clamp_signed(clamp_signed(p1 - q1) + 3 * (q0 - p0));
        let a1 = clamp_signed((a + 4) >> 3);
        let a2 = clamp_signed((a + 3) >> 3);

        p[idx + 1] = (p0 + a2).clamp(0, 255) as u8;
        p[idx + 2] = (q0 - a1).clamp(0, 255) as u8;
    }
}

/// Apply normal loop filter with high edge variance detection
///
/// The normal filter uses 4 pixels on each side of the edge and includes
/// additional filtering for edges with high variance.
pub fn normal_loop_filter(
    p: &mut [u8],
    offset: usize,
    stride: usize,
    limits: &FilterLimits,
    count: usize,
) {
    for i in 0..count {
        let idx = offset + i * stride;
        if idx + 7 >= p.len() || idx < 3 {
            continue;
        }

        // Read 4 pixels on each side of edge
        let p3 = p[idx - 3] as i32;
        let p2 = p[idx - 2] as i32;
        let p1 = p[idx - 1] as i32;
        let p0 = p[idx] as i32;
        let q0 = p[idx + 1] as i32;
        let q1 = p[idx + 2] as i32;
        let q2 = p[idx + 3] as i32;
        let q3 = p[idx + 4] as i32;

        // Interior mask: check for smooth interior
        let interior_mask = (p3 - p2)
            .abs()
            .max((p2 - p1).abs())
            .max((p1 - p0).abs())
            .max((q1 - q0).abs())
            .max((q2 - q1).abs())
            .max((q3 - q2).abs());

        if interior_mask > limits.limit {
            continue;
        }

        // Edge mask
        let edge_diff = (p0 - q0).abs() * 2 + ((p1 - q1).abs() >> 1);
        if edge_diff > limits.blimit {
            continue;
        }

        // High edge variance check
        let hev = (p1 - p0).abs() > limits.hev_thresh || (q1 - q0).abs() > limits.hev_thresh;

        // Apply filter
        let (new_p1, new_p0, new_q0, new_q1) = filter_common(p1, p0, q0, q1, hev);

        p[idx - 1] = new_p1 as u8;
        p[idx] = new_p0 as u8;
        p[idx + 1] = new_q0 as u8;
        p[idx + 2] = new_q1 as u8;
    }
}

/// Apply loop filter to vertical edges (between macroblocks/subblocks)
pub fn filter_vertical_edge(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    height: usize,
    limits: &FilterLimits,
    filter_type: FilterType,
) {
    if limits.blimit == 0 {
        return;
    }

    match filter_type {
        FilterType::Simple => {
            // For simple filter, work on 2 pixels each side
            let offset = y * stride + x - 1;
            simple_loop_filter(plane, offset, stride, limits.blimit, height);
        }
        FilterType::Normal => {
            // For normal filter, work on 4 pixels each side
            if x >= 3 {
                let offset = y * stride + x;
                normal_loop_filter(plane, offset, stride, limits, height);
            }
        }
    }
}

/// Apply loop filter to horizontal edges (between macroblocks/subblocks)
pub fn filter_horizontal_edge(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    width: usize,
    limits: &FilterLimits,
    filter_type: FilterType,
) {
    if limits.blimit == 0 {
        return;
    }

    match filter_type {
        FilterType::Simple => {
            // Horizontal edges with vertical step
            for i in 0..width {
                let idx = (y - 1) * stride + x + i;
                if idx + stride * 3 >= plane.len() || y < 1 {
                    continue;
                }

                let p1 = plane[idx] as i32;
                let p0 = plane[idx + stride] as i32;
                let q0 = plane[idx + stride * 2] as i32;
                let q1 = plane[idx + stride * 3] as i32;

                let mask = (p0 - q0).abs() * 2 + ((p1 - q1).abs() >> 1);
                if mask > limits.blimit {
                    continue;
                }

                let a = clamp_signed(clamp_signed(p1 - q1) + 3 * (q0 - p0));
                let a1 = clamp_signed((a + 4) >> 3);
                let a2 = clamp_signed((a + 3) >> 3);

                plane[idx + stride] = (p0 + a2).clamp(0, 255) as u8;
                plane[idx + stride * 2] = (q0 - a1).clamp(0, 255) as u8;
            }
        }
        FilterType::Normal => {
            // Normal filter on horizontal edge
            if y >= 3 {
                for i in 0..width {
                    let idx = y * stride + x + i;
                    if idx + stride * 4 >= plane.len() {
                        continue;
                    }

                    let p3 = plane[idx - stride * 3] as i32;
                    let p2 = plane[idx - stride * 2] as i32;
                    let p1 = plane[idx - stride] as i32;
                    let p0 = plane[idx] as i32;
                    let q0 = plane[idx + stride] as i32;
                    let q1 = plane[idx + stride * 2] as i32;
                    let q2 = plane[idx + stride * 3] as i32;
                    let q3 = plane[idx + stride * 4] as i32;

                    let interior_mask = (p3 - p2)
                        .abs()
                        .max((p2 - p1).abs())
                        .max((p1 - p0).abs())
                        .max((q1 - q0).abs())
                        .max((q2 - q1).abs())
                        .max((q3 - q2).abs());

                    if interior_mask > limits.limit {
                        continue;
                    }

                    let edge_diff = (p0 - q0).abs() * 2 + ((p1 - q1).abs() >> 1);
                    if edge_diff > limits.blimit {
                        continue;
                    }

                    let hev =
                        (p1 - p0).abs() > limits.hev_thresh || (q1 - q0).abs() > limits.hev_thresh;

                    let (new_p1, new_p0, new_q0, new_q1) = filter_common(p1, p0, q0, q1, hev);

                    plane[idx - stride] = new_p1 as u8;
                    plane[idx] = new_p0 as u8;
                    plane[idx + stride] = new_q0 as u8;
                    plane[idx + stride * 2] = new_q1 as u8;
                }
            }
        }
    }
}

/// Calculate the effective filter level for a macroblock
pub fn calculate_mb_filter_level(params: &LoopFilterParams, mb_info: &MacroblockFilterInfo) -> u8 {
    let mut level = if let Some(override_level) = mb_info.filter_level_override {
        override_level as i32
    } else {
        params.filter_level as i32
    };

    if params.mode_ref_lf_delta_enabled {
        // Reference frame adjustment
        let ref_delta = params.ref_lf_deltas[mb_info.ref_frame as usize];
        level += ref_delta as i32;

        // Mode adjustment
        let mode_idx = if mb_info.is_intra {
            0 // Intra
        } else if mb_info.is_skip {
            1 // Skip/Zero MV
        } else {
            2 // Normal inter
        };
        level += params.mode_lf_deltas[mode_idx] as i32;
    }

    level.clamp(0, 63) as u8
}

/// Apply loop filter to an entire frame
pub fn apply_loop_filter(
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    y_stride: usize,
    uv_stride: usize,
    width: u32,
    height: u32,
    params: &LoopFilterParams,
    mb_info: &[MacroblockFilterInfo],
) {
    if params.filter_level == 0 {
        return;
    }

    let mb_width = (width + 15) >> 4;
    let mb_height = (height + 15) >> 4;

    // Process each macroblock row
    for mb_y in 0..mb_height {
        for mb_x in 0..mb_width {
            let mb_idx = (mb_y * mb_width + mb_x) as usize;
            let info = if mb_idx < mb_info.len() {
                mb_info[mb_idx]
            } else {
                MacroblockFilterInfo::default()
            };

            let level = calculate_mb_filter_level(params, &info);
            if level == 0 {
                continue;
            }

            let limits = FilterLimits::from_level_sharpness(level, params.sharpness);

            let y_x = (mb_x * 16) as usize;
            let y_y = (mb_y * 16) as usize;
            let uv_x = (mb_x * 8) as usize;
            let uv_y = (mb_y * 8) as usize;

            // Filter vertical edges (left edge of macroblock)
            if mb_x > 0 {
                filter_vertical_edge(y_plane, y_stride, y_x, y_y, 16, &limits, params.filter_type);
                filter_vertical_edge(
                    u_plane,
                    uv_stride,
                    uv_x,
                    uv_y,
                    8,
                    &limits,
                    params.filter_type,
                );
                filter_vertical_edge(
                    v_plane,
                    uv_stride,
                    uv_x,
                    uv_y,
                    8,
                    &limits,
                    params.filter_type,
                );
            }

            // Filter horizontal edges (top edge of macroblock)
            if mb_y > 0 {
                filter_horizontal_edge(
                    y_plane,
                    y_stride,
                    y_x,
                    y_y,
                    16,
                    &limits,
                    params.filter_type,
                );
                filter_horizontal_edge(
                    u_plane,
                    uv_stride,
                    uv_x,
                    uv_y,
                    8,
                    &limits,
                    params.filter_type,
                );
                filter_horizontal_edge(
                    v_plane,
                    uv_stride,
                    uv_x,
                    uv_y,
                    8,
                    &limits,
                    params.filter_type,
                );
            }

            // Filter internal 4x4 block edges for Y plane (if normal filter)
            if params.filter_type == FilterType::Normal {
                // Internal vertical edges
                for sub_x in 1..4 {
                    let x = y_x + sub_x * 4;
                    filter_vertical_edge(
                        y_plane,
                        y_stride,
                        x,
                        y_y,
                        16,
                        &limits,
                        params.filter_type,
                    );
                }

                // Internal horizontal edges
                for sub_y in 1..4 {
                    let y = y_y + sub_y * 4;
                    filter_horizontal_edge(
                        y_plane,
                        y_stride,
                        y_x,
                        y,
                        16,
                        &limits,
                        params.filter_type,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_limits() {
        let limits = FilterLimits::from_level_sharpness(40, 0);
        assert!(limits.blimit > 0);
        assert!(limits.limit > 0);
        assert_eq!(limits.hev_thresh, 2);

        let limits_low = FilterLimits::from_level_sharpness(10, 0);
        assert_eq!(limits_low.hev_thresh, 0);
    }

    #[test]
    fn test_filter_limits_zero_level() {
        let limits = FilterLimits::from_level_sharpness(0, 0);
        assert_eq!(limits.blimit, 0);
        assert_eq!(limits.limit, 0);
        assert_eq!(limits.hev_thresh, 0);
    }

    #[test]
    fn test_clamp_signed() {
        assert_eq!(clamp_signed(150), 127);
        assert_eq!(clamp_signed(-150), -128);
        assert_eq!(clamp_signed(50), 50);
    }

    #[test]
    fn test_calculate_mb_filter_level() {
        let params = LoopFilterParams {
            filter_level: 40,
            sharpness: 0,
            filter_type: FilterType::Normal,
            mode_ref_lf_delta_enabled: false,
            ref_lf_deltas: [0; 4],
            mode_lf_deltas: [0; 4],
        };

        let info = MacroblockFilterInfo::default();
        let level = calculate_mb_filter_level(&params, &info);
        assert_eq!(level, 40);
    }

    #[test]
    fn test_simple_loop_filter() {
        // Create a test pattern with an edge
        let mut data = vec![100u8, 100, 150, 150, 100, 100, 150, 150];
        simple_loop_filter(&mut data, 0, 4, 10, 2);
        // After filtering, the edge should be smoother
        // (exact values depend on filter implementation)
    }
}
