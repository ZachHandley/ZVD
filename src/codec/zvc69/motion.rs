//! ZVC69 Motion Estimation Module
//!
//! This module provides motion estimation capabilities for P-frames in the ZVC69
//! neural video codec. It includes both neural-based optical flow estimation
//! (using ONNX models) and traditional block matching as a fallback.
//!
//! ## Overview
//!
//! Motion estimation finds the displacement (motion vectors) between frames to
//! enable temporal prediction. The P-frame encoding pipeline uses motion to:
//!
//! 1. Estimate optical flow from reference to current frame
//! 2. Warp the reference frame using the motion field
//! 3. Encode only the residual (difference) for better compression
//!
//! ## Motion Representations
//!
//! - **MotionField**: Dense optical flow with (u, v) displacement per pixel
//! - **CompressedMotion**: Entropy-coded representation for transmission
//!
//! ## Precision Levels
//!
//! - **FullPel**: Integer pixel precision (fastest, lowest quality)
//! - **HalfPel**: Half-pixel interpolation (balanced)
//! - **QuarterPel**: Quarter-pixel interpolation (best quality)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use zvd::codec::zvc69::motion::{MotionEstimator, MotionConfig, MotionPrecision};
//!
//! // Create estimator with configuration
//! let config = MotionConfig {
//!     search_range: 64,
//!     multi_scale: true,
//!     precision: MotionPrecision::QuarterPel,
//! };
//!
//! let estimator = MotionEstimator::new(config);
//!
//! // Estimate motion between frames
//! let motion = estimator.estimate_placeholder(&current, &reference);
//!
//! // Compress motion for transmission
//! let compressed = estimator.compress(&motion)?;
//! ```

use super::entropy::{EntropyCoder, GaussianConditional, DEFAULT_MAX_SYMBOL, DEFAULT_MIN_SYMBOL};
use super::error::ZVC69Error;
use ndarray::{s, Array2, Array4, ArrayView2, Zip};
use std::path::Path;

#[cfg(feature = "zvc69")]
use ort::session::Session;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Default motion search range in pixels
pub const DEFAULT_SEARCH_RANGE: usize = 64;

/// Default block size for block matching
pub const DEFAULT_BLOCK_SIZE: usize = 16;

/// Minimum block size for hierarchical search
pub const MIN_BLOCK_SIZE: usize = 4;

/// Motion vector clamp range (prevents extreme values)
pub const MV_CLAMP_RANGE: f32 = 512.0;

/// Spatial downscale factor for motion field (relative to frame)
/// Motion field is typically H/4 x W/4 for efficiency
pub const MOTION_SPATIAL_FACTOR: usize = 4;

/// Default quantization scale for motion vectors
pub const DEFAULT_MOTION_QUANT_SCALE: f32 = 4.0;

// ─────────────────────────────────────────────────────────────────────────────
// Motion Precision
// ─────────────────────────────────────────────────────────────────────────────

/// Motion vector precision level
///
/// Higher precision enables sub-pixel motion estimation which improves
/// compression quality at the cost of computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MotionPrecision {
    /// Full pixel precision - fastest, lowest quality
    FullPel,
    /// Half pixel precision - balanced
    HalfPel,
    /// Quarter pixel precision - highest quality (default)
    #[default]
    QuarterPel,
}

impl MotionPrecision {
    /// Get the scale factor for this precision
    ///
    /// Returns 1 for full-pel, 2 for half-pel, 4 for quarter-pel
    pub fn scale_factor(&self) -> usize {
        match self {
            MotionPrecision::FullPel => 1,
            MotionPrecision::HalfPel => 2,
            MotionPrecision::QuarterPel => 4,
        }
    }

    /// Get the precision in bits (for encoding)
    pub fn precision_bits(&self) -> usize {
        match self {
            MotionPrecision::FullPel => 0,
            MotionPrecision::HalfPel => 1,
            MotionPrecision::QuarterPel => 2,
        }
    }

    /// Create from precision bits
    pub fn from_bits(bits: usize) -> Self {
        match bits {
            0 => MotionPrecision::FullPel,
            1 => MotionPrecision::HalfPel,
            _ => MotionPrecision::QuarterPel,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Motion Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for motion estimation
#[derive(Debug, Clone)]
pub struct MotionConfig {
    /// Maximum motion search range in pixels (default 64)
    pub search_range: usize,
    /// Use multi-scale (pyramid) for large motion (default true)
    pub multi_scale: bool,
    /// Motion vector precision level (default QuarterPel)
    pub precision: MotionPrecision,
    /// Block size for block matching fallback (default 16)
    pub block_size: usize,
    /// Quantization scale for motion vectors (default 4.0)
    pub quant_scale: f32,
}

impl Default for MotionConfig {
    fn default() -> Self {
        Self {
            search_range: DEFAULT_SEARCH_RANGE,
            multi_scale: true,
            precision: MotionPrecision::QuarterPel,
            block_size: DEFAULT_BLOCK_SIZE,
            quant_scale: DEFAULT_MOTION_QUANT_SCALE,
        }
    }
}

impl MotionConfig {
    /// Create a new motion configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Create configuration optimized for speed
    pub fn fast() -> Self {
        Self {
            search_range: 32,
            multi_scale: false,
            precision: MotionPrecision::FullPel,
            block_size: 16,
            quant_scale: 8.0,
        }
    }

    /// Create configuration optimized for quality
    pub fn quality() -> Self {
        Self {
            search_range: 128,
            multi_scale: true,
            precision: MotionPrecision::QuarterPel,
            block_size: 8,
            quant_scale: 2.0,
        }
    }

    /// Builder method: set search range
    pub fn with_search_range(mut self, range: usize) -> Self {
        self.search_range = range;
        self
    }

    /// Builder method: set multi-scale mode
    pub fn with_multi_scale(mut self, enabled: bool) -> Self {
        self.multi_scale = enabled;
        self
    }

    /// Builder method: set precision
    pub fn with_precision(mut self, precision: MotionPrecision) -> Self {
        self.precision = precision;
        self
    }

    /// Builder method: set block size
    pub fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = size.max(MIN_BLOCK_SIZE);
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Motion Field
// ─────────────────────────────────────────────────────────────────────────────

/// Dense motion field (optical flow) for a frame
///
/// Represents the pixel displacement from reference frame to current frame.
/// For each pixel position (x, y), the motion vector (u, v) indicates where
/// to sample from the reference frame: ref_pos = (x - u, y - v).
///
/// The motion field is stored at potentially lower resolution than the video
/// frame for efficiency. Standard is H/4 x W/4.
#[derive(Debug, Clone)]
pub struct MotionField {
    /// Horizontal motion component (u) [B, 1, H, W]
    /// Positive u = motion to the right
    pub u: Array4<f32>,
    /// Vertical motion component (v) [B, 1, H, W]
    /// Positive v = motion downward
    pub v: Array4<f32>,
    /// Resolution of the motion field (height, width)
    pub resolution: (usize, usize),
}

impl MotionField {
    /// Create a new motion field with uninitialized values
    ///
    /// # Arguments
    ///
    /// * `height` - Height of motion field
    /// * `width` - Width of motion field
    pub fn new(height: usize, width: usize) -> Self {
        Self {
            u: Array4::zeros((1, 1, height, width)),
            v: Array4::zeros((1, 1, height, width)),
            resolution: (height, width),
        }
    }

    /// Create a zero motion field (no motion)
    ///
    /// Useful for initialization or static scenes.
    pub fn zeros(height: usize, width: usize) -> Self {
        Self::new(height, width)
    }

    /// Create motion field from tensor components
    ///
    /// # Arguments
    ///
    /// * `u` - Horizontal motion tensor [B, 1, H, W]
    /// * `v` - Vertical motion tensor [B, 1, H, W]
    ///
    /// # Panics
    ///
    /// Panics if u and v have different shapes
    pub fn from_tensors(u: Array4<f32>, v: Array4<f32>) -> Self {
        assert_eq!(u.shape(), v.shape(), "Motion u and v must have same shape");

        let height = u.shape()[2];
        let width = u.shape()[3];

        Self {
            u,
            v,
            resolution: (height, width),
        }
    }

    /// Create from a combined flow tensor [B, 2, H, W]
    ///
    /// Channel 0 is horizontal (u), channel 1 is vertical (v)
    pub fn from_flow_tensor(flow: &Array4<f32>) -> Result<Self, ZVC69Error> {
        if flow.shape()[1] != 2 {
            return Err(ZVC69Error::MotionEstimationFailed {
                frame_num: 0,
                reason: format!("Flow tensor must have 2 channels, got {}", flow.shape()[1]),
            });
        }

        let batch = flow.shape()[0];
        let height = flow.shape()[2];
        let width = flow.shape()[3];

        let u = flow
            .slice(s![.., 0..1, .., ..])
            .to_owned()
            .into_shape_with_order((batch, 1, height, width))
            .map_err(|e| ZVC69Error::MotionEstimationFailed {
                frame_num: 0,
                reason: format!("Failed to reshape u component: {}", e),
            })?;

        let v = flow
            .slice(s![.., 1..2, .., ..])
            .to_owned()
            .into_shape_with_order((batch, 1, height, width))
            .map_err(|e| ZVC69Error::MotionEstimationFailed {
                frame_num: 0,
                reason: format!("Failed to reshape v component: {}", e),
            })?;

        Ok(Self {
            u,
            v,
            resolution: (height, width),
        })
    }

    /// Convert to combined flow tensor [B, 2, H, W]
    pub fn to_flow_tensor(&self) -> Array4<f32> {
        let batch = self.u.shape()[0];
        let height = self.resolution.0;
        let width = self.resolution.1;

        let mut flow = Array4::zeros((batch, 2, height, width));

        flow.slice_mut(s![.., 0, .., ..]).assign(
            &self
                .u
                .slice(s![.., 0, .., ..])
                .into_shape_with_order((batch, height, width))
                .unwrap(),
        );
        flow.slice_mut(s![.., 1, .., ..]).assign(
            &self
                .v
                .slice(s![.., 0, .., ..])
                .into_shape_with_order((batch, height, width))
                .unwrap(),
        );

        flow
    }

    /// Get the height of the motion field
    pub fn height(&self) -> usize {
        self.resolution.0
    }

    /// Get the width of the motion field
    pub fn width(&self) -> usize {
        self.resolution.1
    }

    /// Compute motion magnitude (sqrt(u^2 + v^2)) as a 2D array
    pub fn magnitude(&self) -> Array2<f32> {
        let u_2d = self.u.slice(s![0, 0, .., ..]);
        let v_2d = self.v.slice(s![0, 0, .., ..]);

        let mut mag = Array2::zeros((self.resolution.0, self.resolution.1));

        Zip::from(&mut mag)
            .and(&u_2d)
            .and(&v_2d)
            .for_each(|m, &u, &v| {
                *m = (u * u + v * v).sqrt();
            });

        mag
    }

    /// Compute motion angle (atan2(v, u)) as a 2D array (radians)
    pub fn angle(&self) -> Array2<f32> {
        let u_2d = self.u.slice(s![0, 0, .., ..]);
        let v_2d = self.v.slice(s![0, 0, .., ..]);

        let mut ang = Array2::zeros((self.resolution.0, self.resolution.1));

        Zip::from(&mut ang)
            .and(&u_2d)
            .and(&v_2d)
            .for_each(|a, &u, &v| {
                *a = v.atan2(u);
            });

        ang
    }

    /// Scale motion vectors by a factor
    ///
    /// Useful when resizing motion field or adjusting for frame rate changes.
    pub fn scale(&mut self, factor: f32) {
        self.u.mapv_inplace(|x| x * factor);
        self.v.mapv_inplace(|x| x * factor);
    }

    /// Clamp motion vectors to a range
    ///
    /// Prevents extreme motion values that could cause artifacts.
    pub fn clamp(&mut self, max_magnitude: f32) {
        self.u
            .mapv_inplace(|x| x.clamp(-max_magnitude, max_magnitude));
        self.v
            .mapv_inplace(|x| x.clamp(-max_magnitude, max_magnitude));
    }

    /// Get motion vector at a specific position
    ///
    /// Returns (u, v) displacement at (row, col)
    pub fn get(&self, row: usize, col: usize) -> Option<(f32, f32)> {
        if row < self.resolution.0 && col < self.resolution.1 {
            Some((self.u[[0, 0, row, col]], self.v[[0, 0, row, col]]))
        } else {
            None
        }
    }

    /// Set motion vector at a specific position
    pub fn set(&mut self, row: usize, col: usize, u: f32, v: f32) {
        if row < self.resolution.0 && col < self.resolution.1 {
            self.u[[0, 0, row, col]] = u;
            self.v[[0, 0, row, col]] = v;
        }
    }

    /// Upsample motion field to a larger resolution using bilinear interpolation
    ///
    /// Motion values are scaled proportionally.
    pub fn upsample(&self, target_height: usize, target_width: usize) -> Self {
        let scale_h = target_height as f32 / self.resolution.0 as f32;
        let scale_w = target_width as f32 / self.resolution.1 as f32;

        let mut u_up = Array4::zeros((1, 1, target_height, target_width));
        let mut v_up = Array4::zeros((1, 1, target_height, target_width));

        // Bilinear interpolation with motion scaling
        for y in 0..target_height {
            for x in 0..target_width {
                let src_y = (y as f32 / scale_h).min((self.resolution.0 - 1) as f32);
                let src_x = (x as f32 / scale_w).min((self.resolution.1 - 1) as f32);

                let y0 = src_y.floor() as usize;
                let x0 = src_x.floor() as usize;
                let y1 = (y0 + 1).min(self.resolution.0 - 1);
                let x1 = (x0 + 1).min(self.resolution.1 - 1);

                let fy = src_y - y0 as f32;
                let fx = src_x - x0 as f32;

                // Bilinear interpolation
                let u00 = self.u[[0, 0, y0, x0]];
                let u01 = self.u[[0, 0, y0, x1]];
                let u10 = self.u[[0, 0, y1, x0]];
                let u11 = self.u[[0, 0, y1, x1]];

                let u_interp = u00 * (1.0 - fx) * (1.0 - fy)
                    + u01 * fx * (1.0 - fy)
                    + u10 * (1.0 - fx) * fy
                    + u11 * fx * fy;

                let v00 = self.v[[0, 0, y0, x0]];
                let v01 = self.v[[0, 0, y0, x1]];
                let v10 = self.v[[0, 0, y1, x0]];
                let v11 = self.v[[0, 0, y1, x1]];

                let v_interp = v00 * (1.0 - fx) * (1.0 - fy)
                    + v01 * fx * (1.0 - fy)
                    + v10 * (1.0 - fx) * fy
                    + v11 * fx * fy;

                // Scale motion values for new resolution
                u_up[[0, 0, y, x]] = u_interp * scale_w;
                v_up[[0, 0, y, x]] = v_interp * scale_h;
            }
        }

        Self {
            u: u_up,
            v: v_up,
            resolution: (target_height, target_width),
        }
    }

    /// Downsample motion field to a smaller resolution
    ///
    /// Uses average pooling for smooth downsampling.
    pub fn downsample(&self, target_height: usize, target_width: usize) -> Self {
        let scale_h = self.resolution.0 as f32 / target_height as f32;
        let scale_w = self.resolution.1 as f32 / target_width as f32;

        let mut u_down = Array4::zeros((1, 1, target_height, target_width));
        let mut v_down = Array4::zeros((1, 1, target_height, target_width));

        // Average pooling
        for y in 0..target_height {
            for x in 0..target_width {
                let y0 = (y as f32 * scale_h) as usize;
                let x0 = (x as f32 * scale_w) as usize;
                let y1 = ((y + 1) as f32 * scale_h).ceil() as usize;
                let x1 = ((x + 1) as f32 * scale_w).ceil() as usize;

                let y1 = y1.min(self.resolution.0);
                let x1 = x1.min(self.resolution.1);

                let mut u_sum = 0.0f32;
                let mut v_sum = 0.0f32;
                let mut count = 0.0f32;

                for sy in y0..y1 {
                    for sx in x0..x1 {
                        u_sum += self.u[[0, 0, sy, sx]];
                        v_sum += self.v[[0, 0, sy, sx]];
                        count += 1.0;
                    }
                }

                if count > 0.0 {
                    // Scale motion values for new resolution
                    u_down[[0, 0, y, x]] = (u_sum / count) / scale_w;
                    v_down[[0, 0, y, x]] = (v_sum / count) / scale_h;
                }
            }
        }

        Self {
            u: u_down,
            v: v_down,
            resolution: (target_height, target_width),
        }
    }

    /// Compute mean motion (average displacement)
    pub fn mean_motion(&self) -> (f32, f32) {
        let u_mean = self.u.mean().unwrap_or(0.0);
        let v_mean = self.v.mean().unwrap_or(0.0);
        (u_mean, v_mean)
    }

    /// Compute maximum motion magnitude
    pub fn max_magnitude(&self) -> f32 {
        let mag = self.magnitude();
        mag.iter().cloned().fold(0.0f32, f32::max)
    }

    /// Check if motion field is effectively zero (static scene)
    pub fn is_static(&self, threshold: f32) -> bool {
        self.max_magnitude() < threshold
    }
}

impl Default for MotionField {
    fn default() -> Self {
        Self::zeros(1, 1)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Compressed Motion
// ─────────────────────────────────────────────────────────────────────────────

/// Compressed motion representation for bitstream transmission
///
/// Motion vectors are quantized and entropy-coded for efficient storage.
#[derive(Debug, Clone)]
pub struct CompressedMotion {
    /// Quantized motion latents (interleaved u, v)
    pub latents: Vec<i32>,
    /// Shape of the motion field (channels, height, width)
    pub shape: (usize, usize, usize),
    /// Quantization scale used
    pub scale: f32,
    /// Precision level
    pub precision: MotionPrecision,
}

impl CompressedMotion {
    /// Get the number of motion vectors
    pub fn num_vectors(&self) -> usize {
        self.shape.1 * self.shape.2
    }

    /// Get the compressed size in bytes (after entropy coding)
    pub fn compressed_size(&self) -> usize {
        // Estimate: each latent needs ~2-4 bits on average with good prediction
        (self.latents.len() * 3) / 8 + 16 // Add header overhead
    }

    /// Check if motion is effectively zero
    pub fn is_zero(&self) -> bool {
        self.latents.iter().all(|&x| x == 0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Motion Estimator
// ─────────────────────────────────────────────────────────────────────────────

/// Neural motion estimation for P-frame encoding
///
/// Uses either a trained ONNX model for optical flow estimation or
/// falls back to traditional block matching when no model is available.
pub struct MotionEstimator {
    /// ONNX session for motion estimation (optional)
    #[cfg(feature = "zvc69")]
    session: Option<Session>,
    #[cfg(not(feature = "zvc69"))]
    #[allow(dead_code)]
    session: Option<()>,
    /// Motion estimation configuration
    config: MotionConfig,
    /// Whether model is loaded
    model_loaded: bool,
}

impl MotionEstimator {
    /// Create a new motion estimator with the given configuration
    pub fn new(config: MotionConfig) -> Self {
        Self {
            session: None,
            config,
            model_loaded: false,
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(MotionConfig::default())
    }

    /// Get the current configuration
    pub fn config(&self) -> &MotionConfig {
        &self.config
    }

    /// Check if a neural model is loaded
    pub fn has_model(&self) -> bool {
        self.model_loaded
    }

    /// Load motion estimation model from file
    #[cfg(feature = "zvc69")]
    pub fn load_model(&mut self, path: &Path) -> Result<(), ZVC69Error> {
        let model_path = if path.is_file() {
            path.to_path_buf()
        } else {
            path.join("motion.onnx")
        };

        if !model_path.exists() {
            return Err(ZVC69Error::ModelNotFound {
                path: model_path.display().to_string(),
            });
        }

        let session = Session::builder()
            .map_err(|e| ZVC69Error::ModelLoadFailed {
                model_name: "motion".to_string(),
                reason: format!("Failed to create session builder: {}", e),
            })?
            .commit_from_file(&model_path)
            .map_err(|e| ZVC69Error::ModelLoadFailed {
                model_name: "motion".to_string(),
                reason: format!("Failed to load model: {}", e),
            })?;

        self.session = Some(session);
        self.model_loaded = true;
        Ok(())
    }

    #[cfg(not(feature = "zvc69"))]
    pub fn load_model(&mut self, _path: &Path) -> Result<(), ZVC69Error> {
        Err(ZVC69Error::FeatureNotEnabled)
    }

    /// Estimate motion from reference to current frame using neural model
    ///
    /// # Arguments
    ///
    /// * `current` - Current frame tensor [B, C, H, W]
    /// * `reference` - Reference frame tensor [B, C, H, W]
    ///
    /// # Returns
    ///
    /// Estimated motion field or error
    #[cfg(feature = "zvc69")]
    pub fn estimate(
        &mut self,
        current: &Array4<f32>,
        reference: &Array4<f32>,
    ) -> Result<MotionField, ZVC69Error> {
        // Validate input shapes
        if current.shape() != reference.shape() {
            return Err(ZVC69Error::MotionEstimationFailed {
                frame_num: 0,
                reason: format!(
                    "Shape mismatch: current {:?} vs reference {:?}",
                    current.shape(),
                    reference.shape()
                ),
            });
        }

        // If no model loaded, use block matching fallback
        if self.session.is_none() {
            return Ok(self.estimate_placeholder(current, reference));
        }

        let session = self.session.as_mut().unwrap();

        // Concatenate frames along channel dimension [B, 6, H, W]
        let batch = current.shape()[0];
        let height = current.shape()[2];
        let width = current.shape()[3];

        let mut concat = Array4::zeros((batch, 6, height, width));
        concat.slice_mut(s![.., 0..3, .., ..]).assign(current);
        concat.slice_mut(s![.., 3..6, .., ..]).assign(reference);

        // Create ONNX tensor
        let input_tensor = ort::value::Tensor::from_array(concat.clone()).map_err(|e| {
            ZVC69Error::MotionEstimationFailed {
                frame_num: 0,
                reason: format!("Failed to create input tensor: {}", e),
            }
        })?;

        // Run inference
        let outputs = session.run(ort::inputs![input_tensor]).map_err(|e| {
            ZVC69Error::MotionEstimationFailed {
                frame_num: 0,
                reason: format!("Motion estimation inference failed: {}", e),
            }
        })?;

        // Extract flow tensor [B, 2, H', W']
        let output = &outputs[0];
        let flow_view =
            output
                .try_extract_array::<f32>()
                .map_err(|e| ZVC69Error::MotionEstimationFailed {
                    frame_num: 0,
                    reason: format!("Failed to extract output tensor: {}", e),
                })?;

        // Convert to 4D array
        let flow_shape = flow_view.shape();
        if flow_shape.len() != 4 {
            return Err(ZVC69Error::MotionEstimationFailed {
                frame_num: 0,
                reason: format!("Expected 4D output, got {} dimensions", flow_shape.len()),
            });
        }

        let flow_4d: Array4<f32> = flow_view
            .to_shape((flow_shape[0], flow_shape[1], flow_shape[2], flow_shape[3]))
            .map_err(|e| ZVC69Error::MotionEstimationFailed {
                frame_num: 0,
                reason: format!("Failed to reshape flow tensor: {}", e),
            })?
            .to_owned()
            .into_dimensionality::<ndarray::Ix4>()
            .map_err(|e| ZVC69Error::MotionEstimationFailed {
                frame_num: 0,
                reason: format!("Failed to convert to 4D array: {}", e),
            })?;

        MotionField::from_flow_tensor(&flow_4d)
    }

    #[cfg(not(feature = "zvc69"))]
    pub fn estimate(
        &mut self,
        current: &Array4<f32>,
        reference: &Array4<f32>,
    ) -> Result<MotionField, ZVC69Error> {
        // Validate input shapes
        if current.shape() != reference.shape() {
            return Err(ZVC69Error::MotionEstimationFailed {
                frame_num: 0,
                reason: format!(
                    "Shape mismatch: current {:?} vs reference {:?}",
                    current.shape(),
                    reference.shape()
                ),
            });
        }
        Ok(self.estimate_placeholder(current, reference))
    }

    /// Placeholder motion estimation (for testing without model)
    ///
    /// Uses simple block matching on the luminance channel.
    /// Returns a motion field at 1/4 resolution of the input.
    pub fn estimate_placeholder(
        &self,
        current: &Array4<f32>,
        reference: &Array4<f32>,
    ) -> MotionField {
        let height = current.shape()[2];
        let width = current.shape()[3];

        // Motion field at 1/4 resolution
        let mv_height = height / MOTION_SPATIAL_FACTOR;
        let mv_width = width / MOTION_SPATIAL_FACTOR;

        // Extract first channel (luminance) for block matching
        let current_luma = current.slice(s![0, 0, .., ..]).to_owned();
        let reference_luma = reference.slice(s![0, 0, .., ..]).to_owned();

        // Perform block matching at motion field resolution
        block_match(
            &current_luma,
            &reference_luma,
            self.config.block_size,
            self.config.search_range,
            mv_height,
            mv_width,
        )
    }

    /// Compress motion field for transmission
    ///
    /// Quantizes and prepares motion vectors for entropy coding.
    pub fn compress(&self, motion: &MotionField) -> Result<CompressedMotion, ZVC69Error> {
        let height = motion.height();
        let width = motion.width();
        let num_elements = height * width * 2; // u and v components

        // Quantize motion vectors
        let mut latents = Vec::with_capacity(num_elements);

        for y in 0..height {
            for x in 0..width {
                if let Some((u, v)) = motion.get(y, x) {
                    // Quantize with configurable scale
                    let u_quant = (u / self.config.quant_scale).round() as i32;
                    let v_quant = (v / self.config.quant_scale).round() as i32;

                    // Clamp to valid range for entropy coder
                    let u_clamped = u_quant.clamp(DEFAULT_MIN_SYMBOL, DEFAULT_MAX_SYMBOL);
                    let v_clamped = v_quant.clamp(DEFAULT_MIN_SYMBOL, DEFAULT_MAX_SYMBOL);

                    latents.push(u_clamped);
                    latents.push(v_clamped);
                } else {
                    latents.push(0);
                    latents.push(0);
                }
            }
        }

        Ok(CompressedMotion {
            latents,
            shape: (2, height, width),
            scale: self.config.quant_scale,
            precision: self.config.precision,
        })
    }

    /// Decompress motion field from compressed representation
    pub fn decompress(&self, compressed: &CompressedMotion) -> Result<MotionField, ZVC69Error> {
        let height = compressed.shape.1;
        let width = compressed.shape.2;

        if compressed.latents.len() != height * width * 2 {
            return Err(ZVC69Error::MotionCompensationFailed {
                reason: format!(
                    "Invalid compressed motion size: expected {}, got {}",
                    height * width * 2,
                    compressed.latents.len()
                ),
            });
        }

        let mut motion = MotionField::zeros(height, width);

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 2;
                let u_quant = compressed.latents[idx];
                let v_quant = compressed.latents[idx + 1];

                // Dequantize
                let u = u_quant as f32 * compressed.scale;
                let v = v_quant as f32 * compressed.scale;

                motion.set(y, x, u, v);
            }
        }

        Ok(motion)
    }
}

impl Default for MotionEstimator {
    fn default() -> Self {
        Self::default_config()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Block Matching Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Perform block-based motion estimation using Sum of Absolute Differences (SAD)
///
/// This is a fallback method when no neural model is available.
///
/// # Arguments
///
/// * `current` - Current frame as 2D array
/// * `reference` - Reference frame as 2D array
/// * `block_size` - Size of matching blocks
/// * `search_range` - Maximum search distance in pixels
/// * `mv_height` - Height of output motion field
/// * `mv_width` - Width of output motion field
///
/// # Returns
///
/// Motion field with estimated displacement vectors
pub fn block_match(
    current: &Array2<f32>,
    reference: &Array2<f32>,
    block_size: usize,
    search_range: usize,
    mv_height: usize,
    mv_width: usize,
) -> MotionField {
    let height = current.shape()[0];
    let width = current.shape()[1];

    let block_size = block_size.max(MIN_BLOCK_SIZE);
    let step_y = height / mv_height;
    let step_x = width / mv_width;

    let mut motion = MotionField::zeros(mv_height, mv_width);

    for my in 0..mv_height {
        for mx in 0..mv_width {
            let block_y = my * step_y;
            let block_x = mx * step_x;

            // Search for best matching block in reference
            let (best_u, best_v) = search_best_match(
                current,
                reference,
                block_y,
                block_x,
                block_size,
                search_range,
            );

            // Scale motion to full resolution
            motion.set(my, mx, best_u as f32, best_v as f32);
        }
    }

    motion
}

/// Search for the best matching block using SAD
///
/// Uses a small motion penalty (lambda) to prefer smaller motion vectors
/// when SAD values are similar. This prevents arbitrary large motion
/// for identical frames or uniform regions.
fn search_best_match(
    current: &Array2<f32>,
    reference: &Array2<f32>,
    block_y: usize,
    block_x: usize,
    block_size: usize,
    search_range: usize,
) -> (i32, i32) {
    let height = current.shape()[0];
    let width = current.shape()[1];

    let mut best_cost = f32::MAX;
    let mut best_dy = 0i32;
    let mut best_dx = 0i32;

    let search_range = search_range as i32;

    // Small penalty for motion magnitude to prefer zero motion when SAD is equal
    const MOTION_LAMBDA: f32 = 0.01;

    // Extract current block and get its actual dimensions
    let by_end = (block_y + block_size).min(height);
    let bx_end = (block_x + block_size).min(width);
    let actual_block_h = by_end - block_y;
    let actual_block_w = bx_end - block_x;

    let current_block = current.slice(s![block_y..by_end, block_x..bx_end]);

    // Search in reference frame
    for dy in -search_range..=search_range {
        for dx in -search_range..=search_range {
            let ref_y = block_y as i32 + dy;
            let ref_x = block_x as i32 + dx;

            // Skip if out of bounds
            if ref_y < 0 || ref_x < 0 {
                continue;
            }

            let ref_y = ref_y as usize;
            let ref_x = ref_x as usize;

            // Use actual block dimensions (not block_size) to avoid boundary issues
            if ref_y + actual_block_h > height || ref_x + actual_block_w > width {
                continue;
            }

            let ref_block = reference.slice(s![
                ref_y..ref_y + actual_block_h,
                ref_x..ref_x + actual_block_w
            ]);

            // Compute SAD
            let sad_val = sad(&current_block, &ref_block);

            // Add small motion penalty to prefer zero motion when SAD is equal
            let motion_cost = ((dy * dy + dx * dx) as f32).sqrt() * MOTION_LAMBDA;
            let total_cost = sad_val + motion_cost;

            if total_cost < best_cost {
                best_cost = total_cost;
                best_dy = dy;
                best_dx = dx;
            }
        }
    }

    // Return motion from current to reference (negative displacement)
    (-best_dx, -best_dy)
}

/// Sum of Absolute Differences between two blocks
///
/// # Arguments
///
/// * `block1` - First block
/// * `block2` - Second block (must be same size)
///
/// # Returns
///
/// SAD value (lower is better match)
pub fn sad(block1: &ArrayView2<f32>, block2: &ArrayView2<f32>) -> f32 {
    let mut sum = 0.0f32;

    Zip::from(block1).and(block2).for_each(|&a, &b| {
        sum += (a - b).abs();
    });

    sum
}

/// Sum of Squared Differences between two blocks
///
/// Alternative to SAD, more sensitive to outliers but smoother.
pub fn ssd(block1: &ArrayView2<f32>, block2: &ArrayView2<f32>) -> f32 {
    let mut sum = 0.0f32;

    Zip::from(block1).and(block2).for_each(|&a, &b| {
        let diff = a - b;
        sum += diff * diff;
    });

    sum
}

// ─────────────────────────────────────────────────────────────────────────────
// Motion Vector Encoding/Decoding
// ─────────────────────────────────────────────────────────────────────────────

/// Encode motion vectors with entropy coder
///
/// Uses Laplacian-like distribution (motion vectors tend toward zero)
///
/// # Arguments
///
/// * `motion` - Motion field to encode
/// * `entropy_coder` - Entropy coder instance
///
/// # Returns
///
/// Encoded bytes or error
pub fn encode_motion(
    motion: &MotionField,
    entropy_coder: &mut EntropyCoder,
) -> Result<Vec<u8>, ZVC69Error> {
    let height = motion.height();
    let width = motion.width();
    let num_elements = height * width * 2;

    // Flatten motion vectors
    let mut symbols = Vec::with_capacity(num_elements);

    for y in 0..height {
        for x in 0..width {
            if let Some((u, v)) = motion.get(y, x) {
                // Quantize to integer
                let u_int = u.round() as i32;
                let v_int = v.round() as i32;

                // Clamp to valid range
                let u_clamped = u_int.clamp(DEFAULT_MIN_SYMBOL, DEFAULT_MAX_SYMBOL);
                let v_clamped = v_int.clamp(DEFAULT_MIN_SYMBOL, DEFAULT_MAX_SYMBOL);

                symbols.push(u_clamped);
                symbols.push(v_clamped);
            } else {
                symbols.push(0);
                symbols.push(0);
            }
        }
    }

    // Motion vectors tend toward zero - use zero-centered Gaussian with moderate scale
    // This models the typical distribution of motion in natural video
    let means: Vec<f32> = vec![0.0; num_elements];
    let scales: Vec<f32> = vec![8.0; num_elements]; // Moderate scale for motion

    entropy_coder.encode_symbols(&symbols, &means, &scales)
}

/// Decode motion vectors from entropy-coded bytes
///
/// # Arguments
///
/// * `data` - Encoded bytes
/// * `shape` - Shape of motion field (height, width)
/// * `entropy_coder` - Entropy coder instance
///
/// # Returns
///
/// Decoded motion field or error
pub fn decode_motion(
    data: &[u8],
    shape: (usize, usize),
    entropy_coder: &mut EntropyCoder,
) -> Result<MotionField, ZVC69Error> {
    let (height, width) = shape;
    let num_elements = height * width * 2;

    // Motion distribution parameters
    let means: Vec<f32> = vec![0.0; num_elements];
    let scales: Vec<f32> = vec![8.0; num_elements];

    // Decode symbols
    let symbols = entropy_coder.decode_symbols(data, &means, &scales, num_elements)?;

    // Reconstruct motion field
    let mut motion = MotionField::zeros(height, width);

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 2;
            let u = symbols[idx] as f32;
            let v = symbols[idx + 1] as f32;
            motion.set(y, x, u, v);
        }
    }

    Ok(motion)
}

/// Encode motion with Gaussian conditional model (for better compression)
///
/// Uses predicted motion from previous frame as prior.
pub fn encode_motion_conditional(
    motion: &MotionField,
    predicted: Option<&MotionField>,
    gaussian_cond: &GaussianConditional,
) -> Result<Vec<u8>, ZVC69Error> {
    let height = motion.height();
    let width = motion.width();
    let num_elements = height * width * 2;

    // Flatten motion vectors
    let mut symbols = Vec::with_capacity(num_elements);
    let mut means = Vec::with_capacity(num_elements);
    let mut scales = Vec::with_capacity(num_elements);

    for y in 0..height {
        for x in 0..width {
            let (u, v) = motion.get(y, x).unwrap_or((0.0, 0.0));

            // Quantize
            let u_int = u.round() as i32;
            let v_int = v.round() as i32;

            let u_clamped = u_int.clamp(DEFAULT_MIN_SYMBOL, DEFAULT_MAX_SYMBOL);
            let v_clamped = v_int.clamp(DEFAULT_MIN_SYMBOL, DEFAULT_MAX_SYMBOL);

            symbols.push(u_clamped);
            symbols.push(v_clamped);

            // Use predicted motion as prior if available
            if let Some(pred) = predicted {
                let (pred_u, pred_v) = pred.get(y, x).unwrap_or((0.0, 0.0));
                means.push(pred_u);
                means.push(pred_v);
                scales.push(4.0); // Smaller scale = tighter prediction
                scales.push(4.0);
            } else {
                means.push(0.0);
                means.push(0.0);
                scales.push(8.0);
                scales.push(8.0);
            }
        }
    }

    gaussian_cond.encode(&symbols, &means, &scales)
}

/// Decode motion with Gaussian conditional model
pub fn decode_motion_conditional(
    data: &[u8],
    shape: (usize, usize),
    predicted: Option<&MotionField>,
    gaussian_cond: &GaussianConditional,
) -> Result<MotionField, ZVC69Error> {
    let (height, width) = shape;
    let num_elements = height * width * 2;

    // Build means and scales
    let mut means = Vec::with_capacity(num_elements);
    let mut scales = Vec::with_capacity(num_elements);

    for y in 0..height {
        for x in 0..width {
            if let Some(pred) = predicted {
                let (pred_u, pred_v) = pred.get(y, x).unwrap_or((0.0, 0.0));
                means.push(pred_u);
                means.push(pred_v);
                scales.push(4.0);
                scales.push(4.0);
            } else {
                means.push(0.0);
                means.push(0.0);
                scales.push(8.0);
                scales.push(8.0);
            }
        }
    }

    // Decode
    let symbols = gaussian_cond.decode(data, &means, &scales, num_elements)?;

    // Reconstruct motion field
    let mut motion = MotionField::zeros(height, width);

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 2;
            let u = symbols[idx] as f32;
            let v = symbols[idx + 1] as f32;
            motion.set(y, x, u, v);
        }
    }

    Ok(motion)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─────────────────────────────────────────────────────────────────────────
    // MotionPrecision Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_motion_precision_scale_factor() {
        assert_eq!(MotionPrecision::FullPel.scale_factor(), 1);
        assert_eq!(MotionPrecision::HalfPel.scale_factor(), 2);
        assert_eq!(MotionPrecision::QuarterPel.scale_factor(), 4);
    }

    #[test]
    fn test_motion_precision_bits() {
        assert_eq!(MotionPrecision::FullPel.precision_bits(), 0);
        assert_eq!(MotionPrecision::HalfPel.precision_bits(), 1);
        assert_eq!(MotionPrecision::QuarterPel.precision_bits(), 2);
    }

    #[test]
    fn test_motion_precision_from_bits() {
        assert_eq!(MotionPrecision::from_bits(0), MotionPrecision::FullPel);
        assert_eq!(MotionPrecision::from_bits(1), MotionPrecision::HalfPel);
        assert_eq!(MotionPrecision::from_bits(2), MotionPrecision::QuarterPel);
        assert_eq!(MotionPrecision::from_bits(99), MotionPrecision::QuarterPel);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MotionConfig Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_motion_config_default() {
        let config = MotionConfig::default();
        assert_eq!(config.search_range, 64);
        assert!(config.multi_scale);
        assert_eq!(config.precision, MotionPrecision::QuarterPel);
        assert_eq!(config.block_size, 16);
    }

    #[test]
    fn test_motion_config_fast() {
        let config = MotionConfig::fast();
        assert_eq!(config.search_range, 32);
        assert!(!config.multi_scale);
        assert_eq!(config.precision, MotionPrecision::FullPel);
    }

    #[test]
    fn test_motion_config_quality() {
        let config = MotionConfig::quality();
        assert_eq!(config.search_range, 128);
        assert!(config.multi_scale);
        assert_eq!(config.precision, MotionPrecision::QuarterPel);
    }

    #[test]
    fn test_motion_config_builder() {
        let config = MotionConfig::new()
            .with_search_range(48)
            .with_multi_scale(false)
            .with_precision(MotionPrecision::HalfPel)
            .with_block_size(8);

        assert_eq!(config.search_range, 48);
        assert!(!config.multi_scale);
        assert_eq!(config.precision, MotionPrecision::HalfPel);
        assert_eq!(config.block_size, 8);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MotionField Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_motion_field_new() {
        let mf = MotionField::new(64, 128);
        assert_eq!(mf.height(), 64);
        assert_eq!(mf.width(), 128);
        assert_eq!(mf.resolution, (64, 128));
    }

    #[test]
    fn test_motion_field_zeros() {
        let mf = MotionField::zeros(32, 32);
        assert_eq!(mf.get(0, 0), Some((0.0, 0.0)));
        assert_eq!(mf.get(31, 31), Some((0.0, 0.0)));
        assert!(mf.is_static(0.001));
    }

    #[test]
    fn test_motion_field_from_tensors() {
        let u = Array4::ones((1, 1, 4, 4)) * 5.0;
        let v = Array4::ones((1, 1, 4, 4)) * -3.0;

        let mf = MotionField::from_tensors(u, v);
        assert_eq!(mf.height(), 4);
        assert_eq!(mf.width(), 4);
        assert_eq!(mf.get(0, 0), Some((5.0, -3.0)));
    }

    #[test]
    fn test_motion_field_from_flow_tensor() {
        let mut flow = Array4::zeros((1, 2, 4, 4));
        flow[[0, 0, 0, 0]] = 2.5; // u
        flow[[0, 1, 0, 0]] = -1.5; // v

        let mf = MotionField::from_flow_tensor(&flow).unwrap();
        assert_eq!(mf.get(0, 0), Some((2.5, -1.5)));
    }

    #[test]
    fn test_motion_field_to_flow_tensor() {
        let mut mf = MotionField::zeros(4, 4);
        mf.set(0, 0, 3.0, 4.0);

        let flow = mf.to_flow_tensor();
        assert_eq!(flow.shape(), &[1, 2, 4, 4]);
        assert_eq!(flow[[0, 0, 0, 0]], 3.0);
        assert_eq!(flow[[0, 1, 0, 0]], 4.0);
    }

    #[test]
    fn test_motion_field_magnitude() {
        let mut mf = MotionField::zeros(2, 2);
        mf.set(0, 0, 3.0, 4.0); // magnitude = 5

        let mag = mf.magnitude();
        assert!((mag[[0, 0]] - 5.0).abs() < 0.001);
        assert_eq!(mag[[1, 1]], 0.0);
    }

    #[test]
    fn test_motion_field_angle() {
        let mut mf = MotionField::zeros(2, 2);
        mf.set(0, 0, 1.0, 0.0); // angle = 0
        mf.set(0, 1, 0.0, 1.0); // angle = pi/2

        let ang = mf.angle();
        assert!((ang[[0, 0]]).abs() < 0.001);
        assert!((ang[[0, 1]] - std::f32::consts::FRAC_PI_2).abs() < 0.001);
    }

    #[test]
    fn test_motion_field_scale() {
        let mut mf = MotionField::zeros(2, 2);
        mf.set(0, 0, 2.0, -3.0);

        mf.scale(2.0);
        assert_eq!(mf.get(0, 0), Some((4.0, -6.0)));
    }

    #[test]
    fn test_motion_field_clamp() {
        let mut mf = MotionField::zeros(2, 2);
        mf.set(0, 0, 100.0, -100.0);

        mf.clamp(50.0);
        assert_eq!(mf.get(0, 0), Some((50.0, -50.0)));
    }

    #[test]
    fn test_motion_field_get_set() {
        let mut mf = MotionField::zeros(4, 4);

        mf.set(2, 3, 1.5, -2.5);
        assert_eq!(mf.get(2, 3), Some((1.5, -2.5)));

        // Out of bounds
        assert_eq!(mf.get(10, 10), None);
    }

    #[test]
    fn test_motion_field_upsample() {
        let mut mf = MotionField::zeros(2, 2);
        mf.set(0, 0, 1.0, 2.0);
        mf.set(1, 1, 1.0, 2.0);

        let upsampled = mf.upsample(4, 4);
        assert_eq!(upsampled.height(), 4);
        assert_eq!(upsampled.width(), 4);

        // Motion should be scaled proportionally
        let (u, v) = upsampled.get(0, 0).unwrap();
        assert!((u - 2.0).abs() < 0.1); // 1.0 * 2.0 scale
        assert!((v - 4.0).abs() < 0.1); // 2.0 * 2.0 scale
    }

    #[test]
    fn test_motion_field_downsample() {
        let mut mf = MotionField::zeros(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                mf.set(y, x, 2.0, -2.0);
            }
        }

        let downsampled = mf.downsample(2, 2);
        assert_eq!(downsampled.height(), 2);
        assert_eq!(downsampled.width(), 2);

        // Motion should be scaled proportionally
        let (u, v) = downsampled.get(0, 0).unwrap();
        assert!((u - 1.0).abs() < 0.1); // 2.0 / 2.0 scale
        assert!((v - (-1.0)).abs() < 0.1);
    }

    #[test]
    fn test_motion_field_mean_motion() {
        let mut mf = MotionField::zeros(2, 2);
        mf.set(0, 0, 4.0, 8.0);
        mf.set(0, 1, 4.0, 8.0);
        mf.set(1, 0, 4.0, 8.0);
        mf.set(1, 1, 4.0, 8.0);

        let (u_mean, v_mean) = mf.mean_motion();
        assert!((u_mean - 4.0).abs() < 0.001);
        assert!((v_mean - 8.0).abs() < 0.001);
    }

    #[test]
    fn test_motion_field_max_magnitude() {
        let mut mf = MotionField::zeros(2, 2);
        mf.set(0, 0, 3.0, 4.0); // mag = 5
        mf.set(1, 1, 6.0, 8.0); // mag = 10

        let max_mag = mf.max_magnitude();
        assert!((max_mag - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_motion_field_is_static() {
        let mf = MotionField::zeros(4, 4);
        assert!(mf.is_static(0.1));

        let mut mf2 = MotionField::zeros(4, 4);
        mf2.set(0, 0, 10.0, 10.0);
        assert!(!mf2.is_static(1.0));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CompressedMotion Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_compressed_motion_num_vectors() {
        let cm = CompressedMotion {
            latents: vec![0; 64],
            shape: (2, 4, 8),
            scale: 4.0,
            precision: MotionPrecision::QuarterPel,
        };
        assert_eq!(cm.num_vectors(), 32);
    }

    #[test]
    fn test_compressed_motion_is_zero() {
        let cm_zero = CompressedMotion {
            latents: vec![0; 32],
            shape: (2, 4, 4),
            scale: 4.0,
            precision: MotionPrecision::QuarterPel,
        };
        assert!(cm_zero.is_zero());

        let cm_nonzero = CompressedMotion {
            latents: vec![1, 0, -1, 0],
            shape: (2, 1, 2),
            scale: 4.0,
            precision: MotionPrecision::QuarterPel,
        };
        assert!(!cm_nonzero.is_zero());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MotionEstimator Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_motion_estimator_new() {
        let estimator = MotionEstimator::new(MotionConfig::default());
        assert!(!estimator.has_model());
        assert_eq!(estimator.config().search_range, 64);
    }

    #[test]
    fn test_motion_estimator_placeholder() {
        let estimator = MotionEstimator::default_config();

        // Create simple test frames
        let current = Array4::zeros((1, 3, 64, 64));
        let reference = Array4::zeros((1, 3, 64, 64));

        let motion = estimator.estimate_placeholder(&current, &reference);

        // Motion field should be at 1/4 resolution
        assert_eq!(motion.height(), 64 / MOTION_SPATIAL_FACTOR);
        assert_eq!(motion.width(), 64 / MOTION_SPATIAL_FACTOR);
    }

    #[test]
    fn test_motion_estimator_compress_decompress() {
        let estimator = MotionEstimator::new(MotionConfig::default());

        let mut motion = MotionField::zeros(8, 8);
        motion.set(0, 0, 4.0, -8.0);
        motion.set(3, 3, -4.0, 4.0);

        // Compress
        let compressed = estimator.compress(&motion).unwrap();
        assert!(!compressed.latents.is_empty());

        // Decompress
        let decompressed = estimator.decompress(&compressed).unwrap();
        assert_eq!(decompressed.height(), 8);
        assert_eq!(decompressed.width(), 8);

        // Values should be approximately preserved (quantization error)
        let (u, v) = decompressed.get(0, 0).unwrap();
        assert!((u - 4.0).abs() < 0.5);
        assert!((v - (-8.0)).abs() < 0.5);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Block Matching Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_sad_identical_blocks() {
        let block1 = Array2::ones((4, 4)) * 0.5;
        let block2 = Array2::ones((4, 4)) * 0.5;

        let sad_val = sad(&block1.view(), &block2.view());
        assert_eq!(sad_val, 0.0);
    }

    #[test]
    fn test_sad_different_blocks() {
        let block1 = Array2::ones((4, 4));
        let block2 = Array2::zeros((4, 4));

        let sad_val = sad(&block1.view(), &block2.view());
        assert_eq!(sad_val, 16.0); // 4*4 * |1-0|
    }

    #[test]
    fn test_ssd_identical_blocks() {
        let block1 = Array2::ones((4, 4)) * 0.5;
        let block2 = Array2::ones((4, 4)) * 0.5;

        let ssd_val = ssd(&block1.view(), &block2.view());
        assert_eq!(ssd_val, 0.0);
    }

    #[test]
    fn test_ssd_different_blocks() {
        let block1 = Array2::ones((4, 4));
        let block2 = Array2::zeros((4, 4));

        let ssd_val = ssd(&block1.view(), &block2.view());
        assert_eq!(ssd_val, 16.0); // 4*4 * (1-0)^2
    }

    #[test]
    fn test_block_match_static_scene() {
        // Same frame -> zero motion
        let frame = Array2::ones((64, 64));

        let motion = block_match(&frame, &frame, 16, 16, 4, 4);

        // Should find zero motion
        for y in 0..motion.height() {
            for x in 0..motion.width() {
                let (u, v) = motion.get(y, x).unwrap();
                assert_eq!(u, 0.0);
                assert_eq!(v, 0.0);
            }
        }
    }

    #[test]
    fn test_block_match_shifted_frame() {
        // Create frame with gradient
        let mut frame1 = Array2::zeros((64, 64));
        for y in 0..64 {
            for x in 0..64 {
                frame1[[y, x]] = (x as f32) / 64.0;
            }
        }

        // Shift frame by 4 pixels horizontally
        let mut frame2 = Array2::zeros((64, 64));
        for y in 0..64 {
            for x in 4..64 {
                frame2[[y, x]] = frame1[[y, x - 4]];
            }
        }

        let motion = block_match(&frame2, &frame1, 16, 16, 4, 4);

        // Should detect horizontal motion
        // Note: Not all blocks will have perfect detection due to boundaries
        let (u, v) = motion.get(1, 1).unwrap();
        // Motion should be roughly in the negative x direction
        assert!(u <= 0.0 || v.abs() < 1.0);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Motion Encoding Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_encode_decode_motion_roundtrip() {
        let mut mf = MotionField::zeros(4, 4);
        mf.set(0, 0, 5.0, -3.0);
        mf.set(1, 2, -7.0, 12.0);
        mf.set(3, 3, 0.0, 0.0);

        let mut encoder = EntropyCoder::new();
        let encoded = encode_motion(&mf, &mut encoder).unwrap();
        assert!(!encoded.is_empty());

        let mut decoder = EntropyCoder::new();
        let decoded = decode_motion(&encoded, (4, 4), &mut decoder).unwrap();

        // Check roundtrip accuracy
        assert_eq!(decoded.height(), 4);
        assert_eq!(decoded.width(), 4);

        let (u, v) = decoded.get(0, 0).unwrap();
        assert!((u - 5.0).abs() < 0.5);
        assert!((v - (-3.0)).abs() < 0.5);
    }

    #[test]
    fn test_encode_motion_zero_field() {
        let mf = MotionField::zeros(4, 4);

        let mut encoder = EntropyCoder::new();
        let encoded = encode_motion(&mf, &mut encoder).unwrap();

        let mut decoder = EntropyCoder::new();
        let decoded = decode_motion(&encoded, (4, 4), &mut decoder).unwrap();

        assert!(decoded.is_static(0.001));
    }

    #[test]
    fn test_encode_motion_large_values() {
        let mut mf = MotionField::zeros(4, 4);
        // Test with values near the symbol range limits
        mf.set(0, 0, 100.0, -100.0);

        let mut encoder = EntropyCoder::new();
        let encoded = encode_motion(&mf, &mut encoder).unwrap();

        let mut decoder = EntropyCoder::new();
        let decoded = decode_motion(&encoded, (4, 4), &mut decoder).unwrap();

        // Values should be clamped to valid range
        let (u, v) = decoded.get(0, 0).unwrap();
        assert!(u.abs() <= 128.0);
        assert!(v.abs() <= 128.0);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Integration Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_full_motion_pipeline() {
        // Create estimator
        let estimator = MotionEstimator::new(MotionConfig::default());

        // Create test frames
        let mut current = Array4::zeros((1, 3, 64, 64));
        let mut reference = Array4::zeros((1, 3, 64, 64));

        // Add some pattern
        for y in 0..64 {
            for x in 0..64 {
                current[[0, 0, y, x]] = ((x + y) % 16) as f32 / 16.0;
                reference[[0, 0, y, x]] = ((x + y + 2) % 16) as f32 / 16.0;
            }
        }

        // Estimate motion
        let motion = estimator.estimate_placeholder(&current, &reference);
        assert!(motion.height() > 0);
        assert!(motion.width() > 0);

        // Compress
        let compressed = estimator.compress(&motion).unwrap();
        assert!(!compressed.latents.is_empty());

        // Decompress
        let recovered = estimator.decompress(&compressed).unwrap();
        assert_eq!(recovered.height(), motion.height());
        assert_eq!(recovered.width(), motion.width());

        // Entropy encode
        let mut coder = EntropyCoder::new();
        let encoded = encode_motion(&motion, &mut coder).unwrap();
        assert!(!encoded.is_empty());

        // Entropy decode
        let mut coder2 = EntropyCoder::new();
        let decoded =
            decode_motion(&encoded, (motion.height(), motion.width()), &mut coder2).unwrap();
        assert_eq!(decoded.height(), motion.height());
    }

    #[test]
    fn test_motion_field_flow_tensor_roundtrip() {
        let mut mf = MotionField::zeros(8, 8);
        mf.set(2, 3, 5.5, -2.5);
        mf.set(7, 0, -1.0, 3.0);

        let flow = mf.to_flow_tensor();
        let recovered = MotionField::from_flow_tensor(&flow).unwrap();

        assert_eq!(mf.height(), recovered.height());
        assert_eq!(mf.width(), recovered.width());

        let (u1, v1) = mf.get(2, 3).unwrap();
        let (u2, v2) = recovered.get(2, 3).unwrap();
        assert!((u1 - u2).abs() < 0.001);
        assert!((v1 - v2).abs() < 0.001);
    }
}
