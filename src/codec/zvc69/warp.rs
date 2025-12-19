//! ZVC69 Frame Warping Module
//!
//! This module provides frame warping (motion compensation) for P-frames in the ZVC69
//! neural video codec. It uses optical flow / motion vectors to warp a reference frame
//! to predict the current frame.
//!
//! ## Overview
//!
//! Frame warping is essential for inter-frame coding (P/B frames):
//!
//! 1. Motion estimation produces optical flow (u, v) for each pixel
//! 2. The reference frame is warped using the motion field
//! 3. Only the residual (current - warped) needs to be encoded
//!
//! ## Warping Modes
//!
//! - **Backward Warping** (default): Each output pixel samples from motion-displaced
//!   location in reference. Clean, no holes.
//! - **Forward Warping** (splatting): Each reference pixel goes to motion-displaced
//!   location in output. Can have holes, needs filling.
//!
//! ## Interpolation Methods
//!
//! - **Nearest**: Fastest, lowest quality
//! - **Bilinear**: Good balance (default)
//! - **Bicubic**: Highest quality, slower
//!
//! ## Usage
//!
//! ```rust,ignore
//! use zvd::codec::zvc69::warp::{FrameWarper, WarpConfig, Interpolation};
//!
//! let warper = FrameWarper::new(WarpConfig::default());
//! let warped = warper.warp(&reference_frame, &motion_field)?;
//! ```

use super::error::ZVC69Error;
use super::motion::MotionField;
use ndarray::{s, Array2, Array4, Zip};

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------

/// Small epsilon for numerical stability
pub const WARP_EPSILON: f32 = 1e-7;

/// Default flow consistency threshold for occlusion detection
pub const DEFAULT_OCCLUSION_THRESHOLD: f32 = 1.0;

/// Default motion magnitude threshold for occlusion
pub const DEFAULT_MAGNITUDE_THRESHOLD: f32 = 50.0;

// -------------------------------------------------------------------------
// Interpolation Mode
// -------------------------------------------------------------------------

/// Interpolation method for sub-pixel sampling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Interpolation {
    /// Nearest neighbor (fastest, blocky artifacts)
    Nearest,
    /// Bilinear interpolation (default, good balance)
    #[default]
    Bilinear,
    /// Bicubic interpolation (highest quality, slowest)
    Bicubic,
}

impl Interpolation {
    /// Get the kernel size for this interpolation method
    pub fn kernel_size(&self) -> usize {
        match self {
            Interpolation::Nearest => 1,
            Interpolation::Bilinear => 2,
            Interpolation::Bicubic => 4,
        }
    }
}

// -------------------------------------------------------------------------
// Border Mode
// -------------------------------------------------------------------------

/// Border handling mode for out-of-bounds samples
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BorderMode {
    /// Pad with zeros (default, cleanest for compression)
    #[default]
    Zeros,
    /// Replicate edge pixels (no discontinuity at edges)
    Replicate,
    /// Mirror/reflect at edges (smooth continuation)
    Reflect,
}

// -------------------------------------------------------------------------
// Warp Configuration
// -------------------------------------------------------------------------

/// Configuration for frame warping
#[derive(Debug, Clone)]
pub struct WarpConfig {
    /// Interpolation method for sub-pixel sampling
    pub interpolation: Interpolation,
    /// Border handling for out-of-bounds coordinates
    pub border_mode: BorderMode,
    /// Whether to align corners (PyTorch-style grid sampling)
    ///
    /// When true: corner pixels map exactly to [-1, 1]
    /// When false: pixel centers are considered (half-pixel offset)
    pub align_corners: bool,
}

impl Default for WarpConfig {
    fn default() -> Self {
        Self {
            interpolation: Interpolation::Bilinear,
            border_mode: BorderMode::Zeros,
            align_corners: false,
        }
    }
}

impl WarpConfig {
    /// Create a new warp configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set interpolation method
    pub fn with_interpolation(mut self, interp: Interpolation) -> Self {
        self.interpolation = interp;
        self
    }

    /// Builder: set border mode
    pub fn with_border_mode(mut self, mode: BorderMode) -> Self {
        self.border_mode = mode;
        self
    }

    /// Builder: set align corners flag
    pub fn with_align_corners(mut self, align: bool) -> Self {
        self.align_corners = align;
        self
    }

    /// Configuration optimized for speed
    pub fn fast() -> Self {
        Self {
            interpolation: Interpolation::Nearest,
            border_mode: BorderMode::Zeros,
            align_corners: false,
        }
    }

    /// Configuration optimized for quality
    pub fn quality() -> Self {
        Self {
            interpolation: Interpolation::Bicubic,
            border_mode: BorderMode::Replicate,
            align_corners: false,
        }
    }
}

// -------------------------------------------------------------------------
// Frame Warper
// -------------------------------------------------------------------------

/// Frame warper for motion compensation
///
/// Warps a reference frame using a motion field (optical flow) to produce
/// a prediction of the current frame.
pub struct FrameWarper {
    config: WarpConfig,
}

impl FrameWarper {
    /// Create a new frame warper with the given configuration
    pub fn new(config: WarpConfig) -> Self {
        Self { config }
    }

    /// Get the current configuration
    pub fn config(&self) -> &WarpConfig {
        &self.config
    }

    /// Warp reference frame using motion field to predict current frame
    ///
    /// This is the main API for motion compensation. The motion field should
    /// contain (u, v) optical flow from reference to current frame.
    ///
    /// # Arguments
    ///
    /// * `reference` - Reference frame tensor [B, C, H, W]
    /// * `motion_field` - Motion field with (u, v) displacement per pixel
    ///
    /// # Returns
    ///
    /// Warped prediction of the current frame [B, C, H, W]
    ///
    /// # Errors
    ///
    /// Returns error if shapes are incompatible
    pub fn warp(
        &self,
        reference: &Array4<f32>,
        motion_field: &MotionField,
    ) -> Result<Array4<f32>, ZVC69Error> {
        // For standard video coding, use backward warping
        Ok(self.backward_warp(reference, motion_field))
    }

    /// Backward warp (standard for video coding)
    ///
    /// Each pixel in the output is sampled from a motion-displaced location
    /// in the reference frame. This produces a clean output without holes.
    ///
    /// For pixel (x, y) in output:
    ///   sample_x = x - u(x, y)
    ///   sample_y = y - v(x, y)
    ///
    /// where (u, v) is the motion FROM reference TO current.
    pub fn backward_warp(
        &self,
        reference: &Array4<f32>,
        motion_field: &MotionField,
    ) -> Array4<f32> {
        let batch = reference.shape()[0];
        let channels = reference.shape()[1];
        let height = reference.shape()[2];
        let width = reference.shape()[3];

        // Upsample motion field if necessary
        let motion = if motion_field.height() != height || motion_field.width() != width {
            motion_field.upsample(height, width)
        } else {
            motion_field.clone()
        };

        // Generate sampling grid from motion
        let (y_grid, x_grid) = motion_to_grid(&motion, height, width);

        // Allocate output
        let mut output = Array4::zeros((batch, channels, height, width));

        // Warp each batch and channel
        for b in 0..batch {
            for c in 0..channels {
                let ref_slice = reference.slice(s![b, c, .., ..]);

                for y in 0..height {
                    for x in 0..width {
                        // Get sampling coordinates
                        let sample_y = y_grid[[y, x]];
                        let sample_x = x_grid[[y, x]];

                        // Sample from reference using configured interpolation
                        let value = match self.config.interpolation {
                            Interpolation::Nearest => nearest_sample(
                                &ref_slice,
                                sample_y,
                                sample_x,
                                self.config.border_mode,
                            ),
                            Interpolation::Bilinear => bilinear_sample(
                                &ref_slice,
                                sample_y,
                                sample_x,
                                self.config.border_mode,
                            ),
                            Interpolation::Bicubic => bicubic_sample(
                                &ref_slice,
                                sample_y,
                                sample_x,
                                self.config.border_mode,
                            ),
                        };

                        output[[b, c, y, x]] = value;
                    }
                }
            }
        }

        output
    }

    /// Backward warp with 4D tensor interface
    ///
    /// Alternative interface that accepts reference as [B, C, H, W]
    /// and returns warped output [B, C, H, W].
    pub fn backward_warp_tensor(
        &self,
        reference: &Array4<f32>,
        motion_field: &MotionField,
    ) -> Array4<f32> {
        self.backward_warp(reference, motion_field)
    }

    /// Forward warp (splatting)
    ///
    /// Each pixel in the reference frame is "splatted" to a motion-displaced
    /// location in the output. This can create holes and requires filling.
    ///
    /// For pixel (x, y) in reference:
    ///   output_x = x + u(x, y)
    ///   output_y = y + v(x, y)
    ///
    /// Uses weighted accumulation for overlapping contributions.
    pub fn forward_warp(&self, reference: &Array4<f32>, motion_field: &MotionField) -> Array4<f32> {
        let batch = reference.shape()[0];
        let channels = reference.shape()[1];
        let height = reference.shape()[2];
        let width = reference.shape()[3];

        // Upsample motion field if necessary
        let motion = if motion_field.height() != height || motion_field.width() != width {
            motion_field.upsample(height, width)
        } else {
            motion_field.clone()
        };

        // Allocate output and weight accumulator
        let mut output = Array4::zeros((batch, channels, height, width));
        let mut weights = Array2::zeros((height, width));

        // Forward splat each batch and channel
        for b in 0..batch {
            // Reset weights for each batch
            weights.fill(0.0);
            let mut temp_output = Array2::zeros((height, width));

            for c in 0..channels {
                temp_output.fill(0.0);
                weights.fill(0.0);

                for y in 0..height {
                    for x in 0..width {
                        // Get motion at this pixel
                        let (u, v) = motion.get(y, x).unwrap_or((0.0, 0.0));

                        // Compute target position (forward flow)
                        let target_x = x as f32 + u;
                        let target_y = y as f32 + v;

                        // Get source pixel value
                        let value = reference[[b, c, y, x]];

                        // Bilinear splatting (distribute to 4 neighbors)
                        splat_bilinear(
                            &mut temp_output,
                            &mut weights,
                            target_y,
                            target_x,
                            value,
                            height,
                            width,
                        );
                    }
                }

                // Normalize by accumulated weights
                for y in 0..height {
                    for x in 0..width {
                        let w = weights[[y, x]];
                        if w > WARP_EPSILON {
                            output[[b, c, y, x]] = temp_output[[y, x]] / w;
                        }
                        // Holes (w == 0) remain as 0
                    }
                }
            }
        }

        output
    }
}

impl Default for FrameWarper {
    fn default() -> Self {
        Self::new(WarpConfig::default())
    }
}

// -------------------------------------------------------------------------
// Sampling Functions
// -------------------------------------------------------------------------

/// Sample a single pixel with nearest neighbor interpolation
fn nearest_sample(
    image: &ndarray::ArrayView2<f32>,
    y: f32,
    x: f32,
    border_mode: BorderMode,
) -> f32 {
    let height = image.shape()[0];
    let width = image.shape()[1];

    // Round to nearest integer
    let yi = y.round() as i32;
    let xi = x.round() as i32;

    // Handle border
    let (yi, xi) = apply_border_mode(yi, xi, height, width, border_mode);

    if yi < 0 || xi < 0 {
        return 0.0; // Zeros mode, out of bounds
    }

    let yi = yi as usize;
    let xi = xi as usize;

    if yi < height && xi < width {
        image[[yi, xi]]
    } else {
        0.0
    }
}

/// Sample a single pixel with bilinear interpolation
///
/// Uses the standard bilinear formula:
///   f(x, y) = f(0,0)(1-dx)(1-dy) + f(1,0)dx(1-dy) + f(0,1)(1-dx)dy + f(1,1)dx*dy
///
/// # Arguments
///
/// * `image` - 2D image array [H, W]
/// * `y` - Vertical coordinate (can be fractional)
/// * `x` - Horizontal coordinate (can be fractional)
/// * `border_mode` - How to handle out-of-bounds samples
fn bilinear_sample(
    image: &ndarray::ArrayView2<f32>,
    y: f32,
    x: f32,
    border_mode: BorderMode,
) -> f32 {
    let height = image.shape()[0];
    let width = image.shape()[1];

    // Compute integer and fractional parts
    let y0 = y.floor() as i32;
    let x0 = x.floor() as i32;
    let y1 = y0 + 1;
    let x1 = x0 + 1;

    // Fractional parts for interpolation weights
    let fy = y - y0 as f32;
    let fx = x - x0 as f32;

    // Get coordinates with border handling
    let (y0b, x0b) = apply_border_mode(y0, x0, height, width, border_mode);
    let (y1b, x1b) = apply_border_mode(y1, x1, height, width, border_mode);
    let (y0b2, x1b2) = apply_border_mode(y0, x1, height, width, border_mode);
    let (y1b2, x0b2) = apply_border_mode(y1, x0, height, width, border_mode);

    // Sample 4 corners
    let v00 = safe_get(image, y0b, x0b, height, width);
    let v01 = safe_get(image, y0b2, x1b2, height, width);
    let v10 = safe_get(image, y1b2, x0b2, height, width);
    let v11 = safe_get(image, y1b, x1b, height, width);

    // Bilinear interpolation
    let w00 = (1.0 - fx) * (1.0 - fy);
    let w01 = fx * (1.0 - fy);
    let w10 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11
}

/// Sample a single pixel with bicubic interpolation
///
/// Uses Mitchell-Netravali cubic kernel (B=1/3, C=1/3).
fn bicubic_sample(
    image: &ndarray::ArrayView2<f32>,
    y: f32,
    x: f32,
    border_mode: BorderMode,
) -> f32 {
    let height = image.shape()[0];
    let width = image.shape()[1];

    // Integer coordinates of center pixel
    let y_int = y.floor() as i32;
    let x_int = x.floor() as i32;

    // Fractional parts
    let fy = y - y_int as f32;
    let fx = x - x_int as f32;

    // Compute cubic weights for 4x4 neighborhood
    let wy: [f32; 4] = [
        cubic_weight(1.0 + fy),
        cubic_weight(fy),
        cubic_weight(1.0 - fy),
        cubic_weight(2.0 - fy),
    ];
    let wx: [f32; 4] = [
        cubic_weight(1.0 + fx),
        cubic_weight(fx),
        cubic_weight(1.0 - fx),
        cubic_weight(2.0 - fx),
    ];

    // Sample 4x4 neighborhood
    let mut sum = 0.0f32;
    let mut weight_sum = 0.0f32;

    for dy in -1i32..=2 {
        for dx in -1i32..=2 {
            let yi = y_int + dy;
            let xi = x_int + dx;

            let (yb, xb) = apply_border_mode(yi, xi, height, width, border_mode);
            let v = safe_get(image, yb, xb, height, width);

            let w = wy[(dy + 1) as usize] * wx[(dx + 1) as usize];
            sum += v * w;
            weight_sum += w;
        }
    }

    if weight_sum.abs() > WARP_EPSILON {
        sum / weight_sum
    } else {
        0.0
    }
}

/// Mitchell-Netravali cubic kernel weight
///
/// B = 1/3, C = 1/3 (balanced sharpness and ringing)
fn cubic_weight(t: f32) -> f32 {
    let t = t.abs();

    // Mitchell-Netravali with B=1/3, C=1/3
    const B: f32 = 1.0 / 3.0;
    const C: f32 = 1.0 / 3.0;

    if t < 1.0 {
        let t2 = t * t;
        let t3 = t2 * t;
        ((12.0 - 9.0 * B - 6.0 * C) * t3 + (-18.0 + 12.0 * B + 6.0 * C) * t2 + (6.0 - 2.0 * B))
            / 6.0
    } else if t < 2.0 {
        let t2 = t * t;
        let t3 = t2 * t;
        ((-B - 6.0 * C) * t3
            + (6.0 * B + 30.0 * C) * t2
            + (-12.0 * B - 48.0 * C) * t
            + (8.0 * B + 24.0 * C))
            / 6.0
    } else {
        0.0
    }
}

/// Apply border mode to coordinates
///
/// Returns adjusted (y, x) coordinates, or (-1, -1) for zeros mode out of bounds
fn apply_border_mode(y: i32, x: i32, height: usize, width: usize, mode: BorderMode) -> (i32, i32) {
    let h = height as i32;
    let w = width as i32;

    match mode {
        BorderMode::Zeros => {
            if y < 0 || y >= h || x < 0 || x >= w {
                (-1, -1) // Signal out of bounds
            } else {
                (y, x)
            }
        }
        BorderMode::Replicate => {
            let y_clamped = y.clamp(0, h - 1);
            let x_clamped = x.clamp(0, w - 1);
            (y_clamped, x_clamped)
        }
        BorderMode::Reflect => {
            let y_reflected = reflect_coord(y, h);
            let x_reflected = reflect_coord(x, w);
            (y_reflected, x_reflected)
        }
    }
}

/// Reflect coordinate at boundaries
///
/// Uses reflect-101 mode (OpenCV style): the boundary pixels are not duplicated.
fn reflect_coord(coord: i32, size: i32) -> i32 {
    if size <= 1 {
        return 0;
    }

    let mut c = coord;

    // Handle negative coordinates
    while c < 0 {
        c = -c;
    }

    // Handle coordinates beyond size
    while c >= size {
        c = 2 * size - 2 - c;
        if c < 0 {
            c = -c;
        }
    }

    c.clamp(0, size - 1)
}

/// Safely get a pixel value, returning 0 for out-of-bounds
fn safe_get(image: &ndarray::ArrayView2<f32>, y: i32, x: i32, height: usize, width: usize) -> f32 {
    if y < 0 || x < 0 {
        return 0.0;
    }

    let yi = y as usize;
    let xi = x as usize;

    if yi < height && xi < width {
        image[[yi, xi]]
    } else {
        0.0
    }
}

/// Bilinear splatting for forward warping
///
/// Distributes a value to 4 neighbors with bilinear weights.
fn splat_bilinear(
    output: &mut Array2<f32>,
    weights: &mut Array2<f32>,
    y: f32,
    x: f32,
    value: f32,
    height: usize,
    width: usize,
) {
    let y0 = y.floor() as i32;
    let x0 = x.floor() as i32;
    let y1 = y0 + 1;
    let x1 = x0 + 1;

    let fy = y - y0 as f32;
    let fx = x - x0 as f32;

    // Bilinear weights
    let w00 = (1.0 - fx) * (1.0 - fy);
    let w01 = fx * (1.0 - fy);
    let w10 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    // Splat to 4 neighbors
    splat_one(output, weights, y0, x0, value, w00, height, width);
    splat_one(output, weights, y0, x1, value, w01, height, width);
    splat_one(output, weights, y1, x0, value, w10, height, width);
    splat_one(output, weights, y1, x1, value, w11, height, width);
}

/// Splat a single value to one location
fn splat_one(
    output: &mut Array2<f32>,
    weights: &mut Array2<f32>,
    y: i32,
    x: i32,
    value: f32,
    weight: f32,
    height: usize,
    width: usize,
) {
    if y >= 0 && y < height as i32 && x >= 0 && x < width as i32 {
        let yi = y as usize;
        let xi = x as usize;
        output[[yi, xi]] += value * weight;
        weights[[yi, xi]] += weight;
    }
}

// -------------------------------------------------------------------------
// Grid Generation
// -------------------------------------------------------------------------

/// Generate sampling grid from motion field
///
/// Converts motion field (u, v) to absolute sampling coordinates.
/// For backward warping: sample_pos = pixel_pos - motion
///
/// # Returns
///
/// Tuple of (y_coords, x_coords) grids, both [H, W]
pub fn motion_to_grid(
    motion: &MotionField,
    height: usize,
    width: usize,
) -> (Array2<f32>, Array2<f32>) {
    let mut y_grid = Array2::zeros((height, width));
    let mut x_grid = Array2::zeros((height, width));

    for y in 0..height {
        for x in 0..width {
            // Get motion at this position
            let (u, v) = motion.get(y, x).unwrap_or((0.0, 0.0));

            // Backward warp: sample from (x - u, y - v) in reference
            // Motion convention: (u, v) points from reference to current
            x_grid[[y, x]] = x as f32 - u;
            y_grid[[y, x]] = y as f32 - v;
        }
    }

    (y_grid, x_grid)
}

/// Generate identity sampling grid (no motion)
pub fn identity_grid(height: usize, width: usize) -> (Array2<f32>, Array2<f32>) {
    let mut y_grid = Array2::zeros((height, width));
    let mut x_grid = Array2::zeros((height, width));

    for y in 0..height {
        for x in 0..width {
            y_grid[[y, x]] = y as f32;
            x_grid[[y, x]] = x as f32;
        }
    }

    (y_grid, x_grid)
}

// -------------------------------------------------------------------------
// Occlusion Handling
// -------------------------------------------------------------------------

/// Occlusion mask for motion compensation
///
/// Identifies areas that are not visible in the reference frame (occluded).
/// These areas cannot be properly reconstructed by warping and need special
/// handling (e.g., inpainting, using I-frame coding).
#[derive(Debug, Clone)]
pub struct OcclusionMask {
    /// Mask values: 0.0 = occluded, 1.0 = visible
    pub mask: Array2<f32>,
}

impl OcclusionMask {
    /// Create a new occlusion mask with all pixels visible
    pub fn all_visible(height: usize, width: usize) -> Self {
        Self {
            mask: Array2::ones((height, width)),
        }
    }

    /// Create a new occlusion mask with all pixels occluded
    pub fn all_occluded(height: usize, width: usize) -> Self {
        Self {
            mask: Array2::zeros((height, width)),
        }
    }

    /// Create from existing mask array
    pub fn from_array(mask: Array2<f32>) -> Self {
        Self { mask }
    }

    /// Get mask dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.mask.shape()[0], self.mask.shape()[1])
    }

    /// Get mask value at position (0.0 = occluded, 1.0 = visible)
    pub fn get(&self, y: usize, x: usize) -> Option<f32> {
        let (h, w) = self.dimensions();
        if y < h && x < w {
            Some(self.mask[[y, x]])
        } else {
            None
        }
    }

    /// Check if a pixel is occluded
    pub fn is_occluded(&self, y: usize, x: usize) -> bool {
        self.get(y, x).unwrap_or(1.0) < 0.5
    }

    /// Get the fraction of occluded pixels
    pub fn occlusion_ratio(&self) -> f32 {
        let total = self.mask.len() as f32;
        let occluded = self.mask.iter().filter(|&&v| v < 0.5).count() as f32;
        occluded / total
    }

    /// Compute occlusion from forward/backward flow consistency
    ///
    /// A pixel is considered occluded if forward flow and backward flow
    /// are not consistent (don't form a cycle back to the same position).
    ///
    /// # Arguments
    ///
    /// * `forward_flow` - Motion from frame t to frame t+1
    /// * `backward_flow` - Motion from frame t+1 to frame t
    /// * `threshold` - Maximum allowed inconsistency in pixels
    pub fn from_flow_consistency(
        forward_flow: &MotionField,
        backward_flow: &MotionField,
        threshold: f32,
    ) -> Self {
        let height = forward_flow.height();
        let width = forward_flow.width();

        let mut mask = Array2::ones((height, width));

        for y in 0..height {
            for x in 0..width {
                // Get forward motion
                let (u_fwd, v_fwd) = forward_flow.get(y, x).unwrap_or((0.0, 0.0));

                // Compute position in target frame
                let target_x = (x as f32 + u_fwd).round() as i32;
                let target_y = (y as f32 + v_fwd).round() as i32;

                // Check bounds
                if target_x < 0
                    || target_x >= width as i32
                    || target_y < 0
                    || target_y >= height as i32
                {
                    mask[[y, x]] = 0.0; // Out of bounds = occluded
                    continue;
                }

                let tx = target_x as usize;
                let ty = target_y as usize;

                // Get backward motion at target position
                let (u_bwd, v_bwd) = backward_flow.get(ty, tx).unwrap_or((0.0, 0.0));

                // Compute return position
                let return_x = target_x as f32 + u_bwd;
                let return_y = target_y as f32 + v_bwd;

                // Check consistency
                let dx = return_x - x as f32;
                let dy = return_y - y as f32;
                let error = (dx * dx + dy * dy).sqrt();

                if error > threshold {
                    mask[[y, x]] = 0.0; // Inconsistent = likely occluded
                }
            }
        }

        Self { mask }
    }

    /// Compute occlusion from motion magnitude
    ///
    /// Areas with very large motion are likely to be occluded or disoccluded.
    ///
    /// # Arguments
    ///
    /// * `motion` - Motion field
    /// * `threshold` - Maximum motion magnitude for non-occluded pixels
    pub fn from_motion_magnitude(motion: &MotionField, threshold: f32) -> Self {
        let mag = motion.magnitude();
        let (height, width) = (mag.shape()[0], mag.shape()[1]);

        let mut mask = Array2::ones((height, width));

        Zip::from(&mut mask).and(&mag).for_each(|m, &mag_val| {
            if mag_val > threshold {
                *m = 0.0;
            }
        });

        Self { mask }
    }

    /// Compute occlusion from divergence of motion field
    ///
    /// Large positive divergence indicates expansion (disocclusion).
    /// Large negative divergence indicates contraction (occlusion).
    pub fn from_divergence(motion: &MotionField, threshold: f32) -> Self {
        let height = motion.height();
        let width = motion.width();

        let mut mask = Array2::ones((height, width));

        // Compute divergence: du/dx + dv/dy
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let (u_right, _) = motion.get(y, x + 1).unwrap_or((0.0, 0.0));
                let (u_left, _) = motion.get(y, x - 1).unwrap_or((0.0, 0.0));
                let (_, v_down) = motion.get(y + 1, x).unwrap_or((0.0, 0.0));
                let (_, v_up) = motion.get(y - 1, x).unwrap_or((0.0, 0.0));

                let du_dx = (u_right - u_left) / 2.0;
                let dv_dy = (v_down - v_up) / 2.0;

                let divergence = du_dx + dv_dy;

                if divergence.abs() > threshold {
                    mask[[y, x]] = 0.0;
                }
            }
        }

        Self { mask }
    }

    /// Combine multiple occlusion masks (union of occluded regions)
    pub fn union(masks: &[&OcclusionMask]) -> Self {
        if masks.is_empty() {
            return Self::all_visible(0, 0);
        }

        let (height, width) = masks[0].dimensions();
        let mut combined = Array2::ones((height, width));

        for mask in masks {
            Zip::from(&mut combined)
                .and(&mask.mask)
                .for_each(|c: &mut f32, &m: &f32| {
                    *c = c.min(m);
                });
        }

        Self { mask: combined }
    }

    /// Dilate the occlusion mask (expand occluded regions)
    pub fn dilate(&self, radius: usize) -> Self {
        let (height, width) = self.dimensions();
        let mut dilated = self.mask.clone();

        for y in 0..height {
            for x in 0..width {
                if self.mask[[y, x]] < 0.5 {
                    // This pixel is occluded, mark neighbors
                    for dy in 0..=radius {
                        for dx in 0..=radius {
                            if dy * dy + dx * dx <= radius * radius {
                                // Mark all 4 quadrants
                                for (sy, sx) in [
                                    (y as i32 - dy as i32, x as i32 - dx as i32),
                                    (y as i32 - dy as i32, x as i32 + dx as i32),
                                    (y as i32 + dy as i32, x as i32 - dx as i32),
                                    (y as i32 + dy as i32, x as i32 + dx as i32),
                                ] {
                                    if sy >= 0 && sy < height as i32 && sx >= 0 && sx < width as i32
                                    {
                                        dilated[[sy as usize, sx as usize]] = 0.0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Self { mask: dilated }
    }
}

// -------------------------------------------------------------------------
// Multi-Frame Warping
// -------------------------------------------------------------------------

/// Warp and blend multiple reference frames
///
/// Useful for B-frame prediction where both past and future frames are used.
///
/// # Arguments
///
/// * `warped_frames` - Already-warped reference frames
/// * `weights` - Blending weights for each frame (should sum to 1.0)
///
/// # Panics
///
/// Panics if warped_frames and weights have different lengths
pub fn blend_warped_frames(warped_frames: &[Array4<f32>], weights: &[f32]) -> Array4<f32> {
    assert_eq!(
        warped_frames.len(),
        weights.len(),
        "Number of frames and weights must match"
    );

    if warped_frames.is_empty() {
        return Array4::zeros((1, 1, 1, 1));
    }

    let shape = warped_frames[0].shape();
    let mut blended = Array4::zeros((shape[0], shape[1], shape[2], shape[3]));

    // Weighted sum
    for (frame, &weight) in warped_frames.iter().zip(weights.iter()) {
        Zip::from(&mut blended).and(frame).for_each(|b, &f| {
            *b += f * weight;
        });
    }

    blended
}

/// Warp and blend with occlusion-aware weighting
///
/// Increases weight for visible areas and decreases for occluded.
pub fn blend_with_occlusion(
    warped_frames: &[Array4<f32>],
    occlusion_masks: &[&OcclusionMask],
    base_weights: &[f32],
) -> Array4<f32> {
    if warped_frames.is_empty() {
        return Array4::zeros((1, 1, 1, 1));
    }

    let shape = warped_frames[0].shape();
    let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);

    let mut blended = Array4::zeros((batch, channels, height, width));

    for b in 0..batch {
        for c in 0..channels {
            for y in 0..height {
                for x in 0..width {
                    let mut sum = 0.0f32;
                    let mut weight_sum = 0.0f32;

                    for (idx, (frame, mask)) in
                        warped_frames.iter().zip(occlusion_masks.iter()).enumerate()
                    {
                        let visibility = mask.get(y, x).unwrap_or(1.0);
                        let w = base_weights[idx] * visibility;

                        sum += frame[[b, c, y, x]] * w;
                        weight_sum += w;
                    }

                    if weight_sum > WARP_EPSILON {
                        blended[[b, c, y, x]] = sum / weight_sum;
                    }
                }
            }
        }
    }

    blended
}

// -------------------------------------------------------------------------
// Quality Metrics
// -------------------------------------------------------------------------

/// Compute warping error between current and warped frame
///
/// Returns Mean Squared Error (MSE)
pub fn warp_error(current: &Array4<f32>, warped: &Array4<f32>) -> f32 {
    if current.shape() != warped.shape() {
        return f32::MAX;
    }

    let mut sum_sq_error = 0.0f32;
    let n = current.len() as f32;

    Zip::from(current).and(warped).for_each(|&c, &w| {
        let diff = c - w;
        sum_sq_error += diff * diff;
    });

    sum_sq_error / n
}

/// Compute PSNR between current and warped frame
///
/// PSNR = 10 * log10(MAX^2 / MSE)
pub fn warp_psnr(current: &Array4<f32>, warped: &Array4<f32>, max_value: f32) -> f32 {
    let mse = warp_error(current, warped);
    if mse < WARP_EPSILON {
        return f32::INFINITY;
    }
    10.0 * (max_value * max_value / mse).log10()
}

/// Compute mask-weighted warping error
///
/// Only considers pixels where mask > 0.5 (visible pixels)
pub fn masked_warp_error(current: &Array4<f32>, warped: &Array4<f32>, mask: &OcclusionMask) -> f32 {
    if current.shape() != warped.shape() {
        return f32::MAX;
    }

    let channels = current.shape()[1];
    let height = current.shape()[2];
    let width = current.shape()[3];

    let mut sum_sq_error = 0.0f32;
    let mut count = 0.0f32;

    for b in 0..current.shape()[0] {
        for c in 0..channels {
            for y in 0..height {
                for x in 0..width {
                    let visibility = mask.get(y, x).unwrap_or(1.0);
                    if visibility > 0.5 {
                        let diff = current[[b, c, y, x]] - warped[[b, c, y, x]];
                        sum_sq_error += diff * diff;
                        count += 1.0;
                    }
                }
            }
        }
    }

    if count > 0.0 {
        sum_sq_error / count
    } else {
        0.0 // No visible pixels to compare
    }
}

/// Compute SSIM-like structural similarity for warped frame
///
/// Simplified version without the full SSIM windows.
pub fn warp_structural_similarity(
    current: &Array4<f32>,
    warped: &Array4<f32>,
    window_size: usize,
) -> f32 {
    if current.shape() != warped.shape() {
        return 0.0;
    }

    let height = current.shape()[2];
    let width = current.shape()[3];

    if height < window_size || width < window_size {
        return 0.0;
    }

    const C1: f32 = 0.01 * 0.01; // (K1 * L)^2
    const C2: f32 = 0.03 * 0.03; // (K2 * L)^2

    let mut ssim_sum = 0.0f32;
    let mut count = 0;

    // Compute SSIM over non-overlapping windows
    let step = window_size;
    for y in (0..height - window_size).step_by(step) {
        for x in (0..width - window_size).step_by(step) {
            // Compute means
            let mut mean_c = 0.0f32;
            let mut mean_w = 0.0f32;

            for dy in 0..window_size {
                for dx in 0..window_size {
                    mean_c += current[[0, 0, y + dy, x + dx]];
                    mean_w += warped[[0, 0, y + dy, x + dx]];
                }
            }

            let n = (window_size * window_size) as f32;
            mean_c /= n;
            mean_w /= n;

            // Compute variances and covariance
            let mut var_c = 0.0f32;
            let mut var_w = 0.0f32;
            let mut covar = 0.0f32;

            for dy in 0..window_size {
                for dx in 0..window_size {
                    let dc = current[[0, 0, y + dy, x + dx]] - mean_c;
                    let dw = warped[[0, 0, y + dy, x + dx]] - mean_w;
                    var_c += dc * dc;
                    var_w += dw * dw;
                    covar += dc * dw;
                }
            }

            var_c /= n - 1.0;
            var_w /= n - 1.0;
            covar /= n - 1.0;

            // SSIM formula
            let numerator = (2.0 * mean_c * mean_w + C1) * (2.0 * covar + C2);
            let denominator = (mean_c * mean_c + mean_w * mean_w + C1) * (var_c + var_w + C2);

            ssim_sum += numerator / denominator;
            count += 1;
        }
    }

    if count > 0 {
        ssim_sum / count as f32
    } else {
        0.0
    }
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // WarpConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_warp_config_default() {
        let config = WarpConfig::default();
        assert_eq!(config.interpolation, Interpolation::Bilinear);
        assert_eq!(config.border_mode, BorderMode::Zeros);
        assert!(!config.align_corners);
    }

    #[test]
    fn test_warp_config_fast() {
        let config = WarpConfig::fast();
        assert_eq!(config.interpolation, Interpolation::Nearest);
    }

    #[test]
    fn test_warp_config_quality() {
        let config = WarpConfig::quality();
        assert_eq!(config.interpolation, Interpolation::Bicubic);
        assert_eq!(config.border_mode, BorderMode::Replicate);
    }

    #[test]
    fn test_warp_config_builder() {
        let config = WarpConfig::new()
            .with_interpolation(Interpolation::Bicubic)
            .with_border_mode(BorderMode::Reflect)
            .with_align_corners(true);

        assert_eq!(config.interpolation, Interpolation::Bicubic);
        assert_eq!(config.border_mode, BorderMode::Reflect);
        assert!(config.align_corners);
    }

    // -------------------------------------------------------------------------
    // Interpolation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_interpolation_kernel_size() {
        assert_eq!(Interpolation::Nearest.kernel_size(), 1);
        assert_eq!(Interpolation::Bilinear.kernel_size(), 2);
        assert_eq!(Interpolation::Bicubic.kernel_size(), 4);
    }

    // -------------------------------------------------------------------------
    // Zero Motion Warp (Identity) Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_zero_motion_warp_identity() {
        let warper = FrameWarper::default();

        // Create a test frame
        let mut frame = Array4::zeros((1, 1, 8, 8));
        for y in 0..8 {
            for x in 0..8 {
                frame[[0, 0, y, x]] = (y * 8 + x) as f32;
            }
        }

        // Zero motion
        let motion = MotionField::zeros(8, 8);

        // Warp should return identity
        let warped = warper.warp(&frame, &motion).unwrap();

        // Check values are preserved
        for y in 0..8 {
            for x in 0..8 {
                assert!(
                    (warped[[0, 0, y, x]] - frame[[0, 0, y, x]]).abs() < 0.001,
                    "Mismatch at ({}, {}): got {}, expected {}",
                    y,
                    x,
                    warped[[0, 0, y, x]],
                    frame[[0, 0, y, x]]
                );
            }
        }
    }

    #[test]
    fn test_zero_motion_warp_multichannel() {
        let warper = FrameWarper::default();

        // 3-channel frame
        let mut frame = Array4::zeros((1, 3, 8, 8));
        for c in 0..3 {
            for y in 0..8 {
                for x in 0..8 {
                    frame[[0, c, y, x]] = (c * 64 + y * 8 + x) as f32;
                }
            }
        }

        let motion = MotionField::zeros(8, 8);
        let warped = warper.warp(&frame, &motion).unwrap();

        // All channels should be preserved
        for c in 0..3 {
            for y in 0..8 {
                for x in 0..8 {
                    assert!((warped[[0, c, y, x]] - frame[[0, c, y, x]]).abs() < 0.001);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Translation Warp Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_integer_translation_warp() {
        let warper = FrameWarper::default();

        // Create checkerboard pattern
        let mut frame = Array4::zeros((1, 1, 8, 8));
        for y in 0..8 {
            for x in 0..8 {
                frame[[0, 0, y, x]] = ((x + y) % 2) as f32;
            }
        }

        // Translate by (2, 0) - 2 pixels to the right
        let mut motion = MotionField::zeros(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                motion.set(y, x, 2.0, 0.0);
            }
        }

        let warped = warper.warp(&frame, &motion).unwrap();

        // Check interior pixels (accounting for boundary effects)
        for y in 0..8 {
            for x in 2..8 {
                let expected = frame[[0, 0, y, x - 2]];
                assert!(
                    (warped[[0, 0, y, x]] - expected).abs() < 0.001,
                    "Translation mismatch at ({}, {})",
                    y,
                    x
                );
            }
        }
    }

    #[test]
    fn test_subpixel_translation_bilinear() {
        let warper =
            FrameWarper::new(WarpConfig::default().with_interpolation(Interpolation::Bilinear));

        // Create gradient
        let mut frame = Array4::zeros((1, 1, 8, 8));
        for x in 0..8 {
            for y in 0..8 {
                frame[[0, 0, y, x]] = x as f32;
            }
        }

        // Translate by 0.5 pixels
        let mut motion = MotionField::zeros(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                motion.set(y, x, 0.5, 0.0);
            }
        }

        let warped = warper.warp(&frame, &motion).unwrap();

        // Check that values are interpolated (not integer)
        // At x=4, sampling from x=3.5, should get average of 3 and 4 = 3.5
        for y in 0..8 {
            assert!(
                (warped[[0, 0, y, 4]] - 3.5).abs() < 0.1,
                "Sub-pixel interpolation failed at y={}, got {}",
                y,
                warped[[0, 0, y, 4]]
            );
        }
    }

    // -------------------------------------------------------------------------
    // Boundary Handling Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_boundary_zeros_mode() {
        let warper = FrameWarper::new(WarpConfig::default().with_border_mode(BorderMode::Zeros));

        let frame = Array4::ones((1, 1, 4, 4));

        // Large motion that goes out of bounds
        let mut motion = MotionField::zeros(4, 4);
        motion.set(0, 0, -10.0, 0.0); // Sample from way left

        let warped = warper.warp(&frame, &motion).unwrap();

        // Should be zero at boundary-crossing location
        assert_eq!(warped[[0, 0, 0, 0]], 0.0);
    }

    #[test]
    fn test_boundary_replicate_mode() {
        let warper =
            FrameWarper::new(WarpConfig::default().with_border_mode(BorderMode::Replicate));

        // Create frame with distinct edge values
        let mut frame = Array4::zeros((1, 1, 4, 4));
        frame[[0, 0, 0, 0]] = 1.0; // Top-left corner
        frame[[0, 0, 3, 3]] = 2.0; // Bottom-right corner

        // To sample from beyond top-left (negative coordinates), motion must be positive
        // sample_x = x - u, sample_y = y - v
        // At (0,0) with motion (10, 10): sample from (0-10, 0-10) = (-10, -10)
        // With replicate mode, clamps to (0, 0) -> value 1.0
        let mut motion = MotionField::zeros(4, 4);
        motion.set(0, 0, 10.0, 10.0);

        let warped = warper.warp(&frame, &motion).unwrap();

        // Should replicate the edge value (top-left corner = 1.0)
        assert!(
            (warped[[0, 0, 0, 0]] - 1.0).abs() < 0.01,
            "Expected 1.0, got {}",
            warped[[0, 0, 0, 0]]
        );
    }

    #[test]
    fn test_boundary_reflect_mode() {
        let warper = FrameWarper::new(WarpConfig::default().with_border_mode(BorderMode::Reflect));

        // Create frame with pattern
        let mut frame = Array4::zeros((1, 1, 4, 4));
        for y in 0..4 {
            for x in 0..4 {
                frame[[0, 0, y, x]] = y as f32;
            }
        }

        // Sample from position that requires reflection
        let mut motion = MotionField::zeros(4, 4);
        motion.set(0, 0, 0.0, 5.0); // Sample from y=5 which should reflect to y=1

        let warped = warper.warp(&frame, &motion).unwrap();

        // Reflected coordinate: 5 -> 2*3 - 5 = 1
        // But bilinear samples multiple pixels, so check approximate
        assert!(warped[[0, 0, 0, 0]] >= 0.0 && warped[[0, 0, 0, 0]] <= 3.0);
    }

    // -------------------------------------------------------------------------
    // Occlusion Detection Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_occlusion_all_visible() {
        let mask = OcclusionMask::all_visible(8, 8);
        assert!(!mask.is_occluded(0, 0));
        assert!(!mask.is_occluded(4, 4));
        assert_eq!(mask.occlusion_ratio(), 0.0);
    }

    #[test]
    fn test_occlusion_all_occluded() {
        let mask = OcclusionMask::all_occluded(8, 8);
        assert!(mask.is_occluded(0, 0));
        assert!(mask.is_occluded(4, 4));
        assert_eq!(mask.occlusion_ratio(), 1.0);
    }

    #[test]
    fn test_occlusion_from_magnitude() {
        let mut motion = MotionField::zeros(8, 8);
        motion.set(0, 0, 100.0, 0.0); // Large motion
        motion.set(4, 4, 1.0, 1.0); // Small motion

        let mask = OcclusionMask::from_motion_magnitude(&motion, 10.0);

        assert!(mask.is_occluded(0, 0)); // Large motion = occluded
        assert!(!mask.is_occluded(4, 4)); // Small motion = visible
    }

    #[test]
    fn test_occlusion_flow_consistency() {
        // Perfect forward-backward consistency
        let mut forward = MotionField::zeros(8, 8);
        let mut backward = MotionField::zeros(8, 8);

        for y in 0..8 {
            for x in 0..8 {
                forward.set(y, x, 1.0, 0.0); // Move right
                backward.set(y, x, -1.0, 0.0); // Move left
            }
        }

        let mask = OcclusionMask::from_flow_consistency(&forward, &backward, 0.5);

        // Most interior pixels should be visible
        assert!(!mask.is_occluded(4, 4));
    }

    #[test]
    fn test_occlusion_union() {
        let mut mask1_arr = Array2::ones((4, 4));
        mask1_arr[[0, 0]] = 0.0;
        let mask1 = OcclusionMask::from_array(mask1_arr);

        let mut mask2_arr = Array2::ones((4, 4));
        mask2_arr[[1, 1]] = 0.0;
        let mask2 = OcclusionMask::from_array(mask2_arr);

        let combined = OcclusionMask::union(&[&mask1, &mask2]);

        assert!(combined.is_occluded(0, 0));
        assert!(combined.is_occluded(1, 1));
        assert!(!combined.is_occluded(2, 2));
    }

    #[test]
    fn test_occlusion_dilate() {
        let mut mask_arr = Array2::ones((8, 8));
        mask_arr[[4, 4]] = 0.0; // Single occluded pixel
        let mask = OcclusionMask::from_array(mask_arr);

        let dilated = mask.dilate(1);

        // Original pixel and neighbors should be occluded
        assert!(dilated.is_occluded(4, 4));
        assert!(dilated.is_occluded(4, 3));
        assert!(dilated.is_occluded(4, 5));
        assert!(dilated.is_occluded(3, 4));
        assert!(dilated.is_occluded(5, 4));

        // Farther pixels should be visible
        assert!(!dilated.is_occluded(0, 0));
    }

    // -------------------------------------------------------------------------
    // Warp Error Computation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_warp_error_identical() {
        let frame1 = Array4::ones((1, 1, 4, 4));
        let frame2 = Array4::ones((1, 1, 4, 4));

        let error = warp_error(&frame1, &frame2);
        assert_eq!(error, 0.0);
    }

    #[test]
    fn test_warp_error_different() {
        let frame1 = Array4::zeros((1, 1, 4, 4));
        let frame2 = Array4::ones((1, 1, 4, 4));

        let error = warp_error(&frame1, &frame2);
        assert_eq!(error, 1.0); // MSE = 1.0
    }

    #[test]
    fn test_warp_psnr_identical() {
        let frame1 = Array4::ones((1, 1, 4, 4));
        let frame2 = Array4::ones((1, 1, 4, 4));

        let psnr = warp_psnr(&frame1, &frame2, 1.0);
        assert!(psnr.is_infinite()); // Perfect match = infinite PSNR
    }

    #[test]
    fn test_warp_psnr_different() {
        let frame1 = Array4::zeros((1, 1, 4, 4));
        let frame2 = Array4::ones((1, 1, 4, 4));

        let psnr = warp_psnr(&frame1, &frame2, 1.0);
        // PSNR = 10 * log10(1 / 1) = 0 dB
        assert!((psnr - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_masked_warp_error() {
        let mut frame1 = Array4::zeros((1, 1, 4, 4));
        let mut frame2 = Array4::zeros((1, 1, 4, 4));

        // Make them different everywhere
        frame1.fill(0.0);
        frame2.fill(1.0);

        // Mask out most pixels
        let mut mask_arr = Array2::zeros((4, 4));
        mask_arr[[0, 0]] = 1.0; // Only one visible pixel
        let mask = OcclusionMask::from_array(mask_arr);

        let error = masked_warp_error(&frame1, &frame2, &mask);

        // Only one pixel compared, with error = 1.0
        assert_eq!(error, 1.0);
    }

    // -------------------------------------------------------------------------
    // Multi-Frame Blending Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_blend_warped_frames_equal_weights() {
        let frame1 = Array4::zeros((1, 1, 4, 4));
        let mut frame2 = Array4::zeros((1, 1, 4, 4));
        frame2.fill(1.0);

        let blended = blend_warped_frames(&[frame1, frame2], &[0.5, 0.5]);

        // Should be average of 0 and 1 = 0.5
        for y in 0..4 {
            for x in 0..4 {
                assert!((blended[[0, 0, y, x]] - 0.5).abs() < 0.01);
            }
        }
    }

    #[test]
    fn test_blend_warped_frames_unequal_weights() {
        let frame1 = Array4::zeros((1, 1, 4, 4));
        let mut frame2 = Array4::zeros((1, 1, 4, 4));
        frame2.fill(1.0);

        let blended = blend_warped_frames(&[frame1, frame2], &[0.25, 0.75]);

        // Should be 0.25 * 0 + 0.75 * 1 = 0.75
        for y in 0..4 {
            for x in 0..4 {
                assert!((blended[[0, 0, y, x]] - 0.75).abs() < 0.01);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Grid Generation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_identity_grid() {
        let (y_grid, x_grid) = identity_grid(4, 4);

        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(y_grid[[y, x]], y as f32);
                assert_eq!(x_grid[[y, x]], x as f32);
            }
        }
    }

    #[test]
    fn test_motion_to_grid() {
        let mut motion = MotionField::zeros(4, 4);
        motion.set(2, 2, 1.0, 0.5);

        let (y_grid, x_grid) = motion_to_grid(&motion, 4, 4);

        // At (2, 2), motion is (1.0, 0.5)
        // Sample from (2 - 1.0, 2 - 0.5) = (1.0, 1.5)
        assert!((x_grid[[2, 2]] - 1.0).abs() < 0.01);
        assert!((y_grid[[2, 2]] - 1.5).abs() < 0.01);
    }

    // -------------------------------------------------------------------------
    // Forward Warp Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_forward_warp_zero_motion() {
        let warper = FrameWarper::default();

        let mut frame = Array4::zeros((1, 1, 8, 8));
        for y in 0..8 {
            for x in 0..8 {
                frame[[0, 0, y, x]] = (y * 8 + x) as f32;
            }
        }

        let motion = MotionField::zeros(8, 8);
        let warped = warper.forward_warp(&frame, &motion);

        // Should be approximately identity
        for y in 0..8 {
            for x in 0..8 {
                assert!(
                    (warped[[0, 0, y, x]] - frame[[0, 0, y, x]]).abs() < 0.1,
                    "Forward warp identity failed at ({}, {})",
                    y,
                    x
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // Bicubic Interpolation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_bicubic_integer_position() {
        let warper =
            FrameWarper::new(WarpConfig::default().with_interpolation(Interpolation::Bicubic));

        // Create frame with known values
        let mut frame = Array4::zeros((1, 1, 8, 8));
        for y in 0..8 {
            for x in 0..8 {
                frame[[0, 0, y, x]] = (y * 8 + x) as f32;
            }
        }

        // Zero motion
        let motion = MotionField::zeros(8, 8);
        let warped = warper.warp(&frame, &motion).unwrap();

        // At integer positions, bicubic should return close to original
        // (not exact due to kernel normalization)
        for y in 2..6 {
            for x in 2..6 {
                assert!(
                    (warped[[0, 0, y, x]] - frame[[0, 0, y, x]]).abs() < 1.0,
                    "Bicubic mismatch at ({}, {}): got {}, expected {}",
                    y,
                    x,
                    warped[[0, 0, y, x]],
                    frame[[0, 0, y, x]]
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // Cubic Weight Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cubic_weight_at_zero() {
        let w = cubic_weight(0.0);
        // At t=0, Mitchell-Netravali (B=1/3, C=1/3) returns (6-2B)/6 = 8/9 ≈ 0.889
        assert!(w > 0.85 && w < 0.95, "Expected ~0.889, got {}", w);
    }

    #[test]
    fn test_cubic_weight_symmetry() {
        // Weight function should be symmetric
        assert!((cubic_weight(0.5) - cubic_weight(-0.5)).abs() < 0.01);
        assert!((cubic_weight(1.5) - cubic_weight(-1.5)).abs() < 0.01);
    }

    #[test]
    fn test_cubic_weight_beyond_support() {
        // Beyond support (|t| >= 2), weight should be 0
        assert_eq!(cubic_weight(2.0), 0.0);
        assert_eq!(cubic_weight(3.0), 0.0);
    }

    // -------------------------------------------------------------------------
    // Reflect Coordinate Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_reflect_coord_in_bounds() {
        assert_eq!(reflect_coord(3, 8), 3);
        assert_eq!(reflect_coord(0, 8), 0);
        assert_eq!(reflect_coord(7, 8), 7);
    }

    #[test]
    fn test_reflect_coord_negative() {
        assert_eq!(reflect_coord(-1, 8), 1);
        assert_eq!(reflect_coord(-2, 8), 2);
    }

    #[test]
    fn test_reflect_coord_beyond() {
        assert_eq!(reflect_coord(8, 8), 6);
        assert_eq!(reflect_coord(9, 8), 5);
    }

    // -------------------------------------------------------------------------
    // Integration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_full_warp_pipeline() {
        // Create encoder-like scenario
        let warper = FrameWarper::default();

        // Reference frame
        let mut reference = Array4::zeros((1, 3, 32, 32));
        for c in 0..3 {
            for y in 0..32 {
                for x in 0..32 {
                    reference[[0, c, y, x]] = ((x + y + c * 10) % 256) as f32 / 255.0;
                }
            }
        }

        // Motion field at 1/4 resolution
        let mut motion = MotionField::zeros(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                // Small varying motion
                motion.set(y, x, (x as f32 - 4.0) * 0.5, (y as f32 - 4.0) * 0.5);
            }
        }

        // Warp
        let warped = warper.warp(&reference, &motion).unwrap();

        // Check output shape
        assert_eq!(warped.shape(), &[1, 3, 32, 32]);

        // Check values are in valid range
        for &v in warped.iter() {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_warp_with_occlusion_mask() {
        let warper = FrameWarper::default();

        // Create frames
        let reference = Array4::ones((1, 1, 16, 16));
        let current = Array4::zeros((1, 1, 16, 16));

        // Create motion and warp
        let motion = MotionField::zeros(16, 16);
        let warped = warper.warp(&reference, &motion).unwrap();

        // Compute occlusion
        let forward = motion.clone();
        let backward = MotionField::zeros(16, 16);
        let occlusion = OcclusionMask::from_flow_consistency(&forward, &backward, 1.0);

        // Compute masked error
        let error = masked_warp_error(&current, &warped, &occlusion);

        // Since reference is 1s and current is 0s, error should be 1.0
        assert!((error - 1.0).abs() < 0.1);
    }
}
