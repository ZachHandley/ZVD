//! ZVC69 Residual Coding Module
//!
//! This module provides residual coding for P-frames in the ZVC69 neural video codec.
//! Residuals represent the difference between the current frame and the motion-compensated
//! prediction, enabling efficient inter-frame compression.
//!
//! ## Overview
//!
//! In neural video codecs, P-frame compression works as follows:
//!
//! 1. **Motion Estimation**: Find motion vectors from reference to current frame
//! 2. **Motion Compensation**: Warp reference frame using motion vectors
//! 3. **Residual Computation**: Current - Predicted = Residual
//! 4. **Residual Encoding**: Compress residual using a (smaller) neural network
//! 5. **Entropy Coding**: Encode quantized residual latents
//!
//! ## Key Features
//!
//! - **Smaller Network**: Residual encoder uses fewer channels than I-frame encoder
//!   (96 vs 192 latent channels) since residuals have lower complexity
//! - **Skip Mode**: Regions with near-zero residual can be skipped entirely
//! - **Adaptive Quantization**: Quantization can vary based on motion magnitude
//!   and content complexity
//!
//! ## Usage
//!
//! ```rust,ignore
//! use zvd::codec::zvc69::residual::{Residual, ResidualEncoder, ResidualConfig};
//! use ndarray::Array4;
//!
//! // Compute residual
//! let current = Array4::zeros((1, 3, 256, 256));
//! let predicted = Array4::zeros((1, 3, 256, 256));
//! let residual = Residual::compute(&current, &predicted);
//!
//! // Check if residual can be skipped
//! if should_skip_residual(&residual, 0.01) {
//!     // Use skip mode - no residual data needed
//! } else {
//!     // Encode residual
//!     let encoder = ResidualEncoder::new(ResidualConfig::default());
//!     let compressed = encoder.encode_placeholder(&residual);
//! }
//!
//! // Reconstruct current frame
//! let reconstructed = Residual::reconstruct(&residual, &predicted);
//! ```

use super::entropy::{EntropyCoder, FactorizedPrior};
use super::error::ZVC69Error;
use super::quantize::{quality_to_scale, quantize_scaled};
use ndarray::{Array2, Array4};
use std::path::Path;

#[cfg(feature = "zvc69")]
use ort::session::Session;
#[cfg(feature = "zvc69")]
use std::sync::Mutex;

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------

/// Default number of latent channels for residual encoder (smaller than I-frame)
pub const DEFAULT_RESIDUAL_LATENT_CHANNELS: usize = 96;

/// Default number of hyperprior channels for residual encoder
pub const DEFAULT_RESIDUAL_HYPERPRIOR_CHANNELS: usize = 64;

/// Default skip threshold (energy below this is considered near-zero)
pub const DEFAULT_SKIP_THRESHOLD: f32 = 0.001;

/// Default block size for skip mask computation
pub const DEFAULT_SKIP_BLOCK_SIZE: usize = 16;

/// Default sparsity threshold (fraction of elements below threshold)
pub const DEFAULT_SPARSITY_THRESHOLD: f32 = 0.01;

/// Spatial downsampling factor for residual latents
pub const RESIDUAL_LATENT_SPATIAL_FACTOR: usize = 8;

/// Spatial downsampling factor for residual hyperprior
pub const RESIDUAL_HYPERPRIOR_SPATIAL_FACTOR: usize = 32;

// -------------------------------------------------------------------------
// ResidualStats
// -------------------------------------------------------------------------

/// Statistics about a residual tensor
///
/// These statistics are useful for:
/// - Determining if skip mode can be used
/// - Adaptive quantization decisions
/// - Rate-distortion optimization
/// - Quality monitoring
#[derive(Debug, Clone, Copy, Default)]
pub struct ResidualStats {
    /// Mean of residual values
    pub mean: f32,
    /// Standard deviation of residual values
    pub std: f32,
    /// Total energy (sum of squares)
    pub energy: f32,
    /// Fraction of elements near zero (below threshold)
    pub sparsity: f32,
    /// Maximum absolute value
    pub max_abs: f32,
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// Number of elements
    pub num_elements: usize,
}

impl ResidualStats {
    /// Compute statistics from residual data
    pub fn compute(data: &Array4<f32>, sparsity_threshold: f32) -> Self {
        let n = data.len();
        if n == 0 {
            return Self::default();
        }

        let n_f32 = n as f32;

        // Compute mean
        let sum: f32 = data.iter().sum();
        let mean = sum / n_f32;

        // Compute variance and energy
        let mut variance_sum = 0.0f32;
        let mut energy = 0.0f32;
        let mut max_abs = 0.0f32;
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        let mut near_zero_count = 0usize;

        for &x in data.iter() {
            let diff = x - mean;
            variance_sum += diff * diff;
            energy += x * x;
            let abs_x = x.abs();
            if abs_x > max_abs {
                max_abs = abs_x;
            }
            if x < min_val {
                min_val = x;
            }
            if x > max_val {
                max_val = x;
            }
            if abs_x < sparsity_threshold {
                near_zero_count += 1;
            }
        }

        let variance = variance_sum / n_f32;
        let std = variance.sqrt();
        let sparsity = near_zero_count as f32 / n_f32;

        ResidualStats {
            mean,
            std,
            energy,
            sparsity,
            max_abs,
            min: min_val,
            max: max_val,
            num_elements: n,
        }
    }

    /// Check if residual is suitable for skip mode
    ///
    /// Returns true if the residual has very low energy and high sparsity
    pub fn can_skip(&self, energy_threshold: f32, sparsity_threshold: f32) -> bool {
        // Skip if energy is very low OR if most elements are near zero
        let normalized_energy = if self.num_elements > 0 {
            self.energy / self.num_elements as f32
        } else {
            0.0
        };

        normalized_energy < energy_threshold || self.sparsity > sparsity_threshold
    }

    /// Estimate bits per element based on statistics
    ///
    /// Uses Gaussian entropy formula: H = 0.5 * log2(2*pi*e*sigma^2)
    pub fn estimate_bits_per_element(&self) -> f32 {
        if self.std < 0.001 {
            // Very low variance - near zero bits
            0.1
        } else {
            // Gaussian entropy approximation
            0.5 * (2.0 * std::f32::consts::PI * std::f32::consts::E * self.std * self.std).log2()
        }
    }
}

// -------------------------------------------------------------------------
// Residual
// -------------------------------------------------------------------------

/// Residual tensor representing the difference between current and predicted frames
///
/// The residual is computed as: `Residual = Current - Predicted`
/// At the decoder: `Current = Residual + Predicted`
///
/// For neural video codecs, residuals typically have:
/// - Lower energy than the original frame
/// - Higher sparsity (many near-zero values)
/// - Lower entropy (easier to compress)
#[derive(Debug, Clone)]
pub struct Residual {
    /// Residual data tensor [B, C, H, W]
    pub data: Array4<f32>,
    /// Shape information (batch, channels, height, width)
    pub shape: (usize, usize, usize, usize),
}

impl Residual {
    /// Create a new residual from data tensor
    pub fn new(data: Array4<f32>) -> Self {
        let shape = data.dim();
        Residual {
            data,
            shape: (shape.0, shape.1, shape.2, shape.3),
        }
    }

    /// Create a zero residual with given shape
    pub fn zeros(batch: usize, channels: usize, height: usize, width: usize) -> Self {
        let data = Array4::zeros((batch, channels, height, width));
        Residual {
            data,
            shape: (batch, channels, height, width),
        }
    }

    /// Compute residual from current and predicted frames
    ///
    /// # Arguments
    ///
    /// * `current` - Current frame tensor [B, C, H, W]
    /// * `predicted` - Motion-compensated prediction [B, C, H, W]
    ///
    /// # Returns
    ///
    /// Residual tensor (current - predicted)
    ///
    /// # Panics
    ///
    /// Panics if shapes don't match
    pub fn compute(current: &Array4<f32>, predicted: &Array4<f32>) -> Self {
        assert_eq!(
            current.dim(),
            predicted.dim(),
            "Current and predicted frame shapes must match"
        );

        let data = current - predicted;
        Self::new(data)
    }

    /// Compute residual with error checking (no panic)
    pub fn try_compute(current: &Array4<f32>, predicted: &Array4<f32>) -> Result<Self, ZVC69Error> {
        if current.dim() != predicted.dim() {
            return Err(ZVC69Error::ResidualError {
                reason: format!(
                    "Shape mismatch: current {:?} vs predicted {:?}",
                    current.dim(),
                    predicted.dim()
                ),
            });
        }

        let data = current - predicted;
        Ok(Self::new(data))
    }

    /// Reconstruct current frame from residual and predicted
    ///
    /// # Arguments
    ///
    /// * `residual` - Residual tensor (from decode)
    /// * `predicted` - Motion-compensated prediction
    ///
    /// # Returns
    ///
    /// Reconstructed current frame
    pub fn reconstruct(residual: &Self, predicted: &Array4<f32>) -> Array4<f32> {
        &residual.data + predicted
    }

    /// Reconstruct with error checking
    pub fn try_reconstruct(
        residual: &Self,
        predicted: &Array4<f32>,
    ) -> Result<Array4<f32>, ZVC69Error> {
        if residual.shape != predicted.dim().into() {
            return Err(ZVC69Error::ResidualError {
                reason: format!(
                    "Shape mismatch: residual {:?} vs predicted {:?}",
                    residual.shape,
                    predicted.dim()
                ),
            });
        }

        Ok(&residual.data + predicted)
    }

    /// Get statistics about the residual
    pub fn stats(&self) -> ResidualStats {
        ResidualStats::compute(&self.data, DEFAULT_SPARSITY_THRESHOLD)
    }

    /// Get statistics with custom sparsity threshold
    pub fn stats_with_threshold(&self, sparsity_threshold: f32) -> ResidualStats {
        ResidualStats::compute(&self.data, sparsity_threshold)
    }

    /// Clamp residual values to valid range
    pub fn clamp(&mut self, min: f32, max: f32) {
        self.data.mapv_inplace(|x| x.clamp(min, max));
    }

    /// Return clamped copy
    pub fn clamped(&self, min: f32, max: f32) -> Self {
        Self::new(self.data.mapv(|x| x.clamp(min, max)))
    }

    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.shape.0
    }

    /// Get number of channels
    pub fn channels(&self) -> usize {
        self.shape.1
    }

    /// Get height
    pub fn height(&self) -> usize {
        self.shape.2
    }

    /// Get width
    pub fn width(&self) -> usize {
        self.shape.3
    }

    /// Get total number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if residual is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Flatten residual data to 1D vector
    pub fn flatten(&self) -> Vec<f32> {
        self.data.iter().copied().collect()
    }

    /// Check if residual is near zero (skip candidate)
    pub fn is_near_zero(&self, threshold: f32) -> bool {
        let stats = self.stats();
        stats.can_skip(threshold * threshold, 0.95)
    }

    /// Scale residual by a factor
    pub fn scale(&mut self, factor: f32) {
        self.data.mapv_inplace(|x| x * factor);
    }

    /// Return scaled copy
    pub fn scaled(&self, factor: f32) -> Self {
        Self::new(self.data.mapv(|x| x * factor))
    }

    /// Add another residual
    pub fn add(&mut self, other: &Residual) {
        self.data = &self.data + &other.data;
    }
}

// -------------------------------------------------------------------------
// ResidualConfig
// -------------------------------------------------------------------------

/// Configuration for residual encoder/decoder
#[derive(Debug, Clone)]
pub struct ResidualConfig {
    /// Number of latent channels (default: 96, smaller than I-frame's 192)
    pub latent_channels: usize,
    /// Number of hyperprior channels (default: 64)
    pub hyperprior_channels: usize,
    /// Quality scale for quantization
    pub quality_scale: f32,
    /// Skip threshold for near-zero residuals
    pub skip_threshold: f32,
    /// Block size for skip mask computation
    pub skip_block_size: usize,
    /// Enable adaptive quantization
    pub adaptive_quant: bool,
    /// Minimum quantization scale (for high-motion regions)
    pub min_quant_scale: f32,
    /// Maximum quantization scale (for low-motion regions)
    pub max_quant_scale: f32,
}

impl Default for ResidualConfig {
    fn default() -> Self {
        Self {
            latent_channels: DEFAULT_RESIDUAL_LATENT_CHANNELS,
            hyperprior_channels: DEFAULT_RESIDUAL_HYPERPRIOR_CHANNELS,
            quality_scale: 1.0,
            skip_threshold: DEFAULT_SKIP_THRESHOLD,
            skip_block_size: DEFAULT_SKIP_BLOCK_SIZE,
            adaptive_quant: true,
            min_quant_scale: 0.25,
            max_quant_scale: 4.0,
        }
    }
}

impl ResidualConfig {
    /// Create new config with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set latent channels
    pub fn with_latent_channels(mut self, channels: usize) -> Self {
        self.latent_channels = channels;
        self
    }

    /// Set hyperprior channels
    pub fn with_hyperprior_channels(mut self, channels: usize) -> Self {
        self.hyperprior_channels = channels;
        self
    }

    /// Set quality scale
    pub fn with_quality_scale(mut self, scale: f32) -> Self {
        self.quality_scale = scale;
        self
    }

    /// Set quality level (1-8)
    pub fn with_quality(mut self, quality: u8) -> Self {
        self.quality_scale = quality_to_scale(quality);
        self
    }

    /// Set skip threshold
    pub fn with_skip_threshold(mut self, threshold: f32) -> Self {
        self.skip_threshold = threshold;
        self
    }

    /// Set skip block size
    pub fn with_skip_block_size(mut self, size: usize) -> Self {
        self.skip_block_size = size;
        self
    }

    /// Enable/disable adaptive quantization
    pub fn with_adaptive_quant(mut self, enable: bool) -> Self {
        self.adaptive_quant = enable;
        self
    }

    /// Create config optimized for speed
    pub fn fast() -> Self {
        Self {
            latent_channels: 64,
            hyperprior_channels: 48,
            quality_scale: 1.5,
            skip_threshold: 0.005, // More aggressive skip
            skip_block_size: 32,   // Larger blocks
            adaptive_quant: false,
            min_quant_scale: 0.5,
            max_quant_scale: 2.0,
        }
    }

    /// Create config optimized for quality
    pub fn quality() -> Self {
        Self {
            latent_channels: 128,
            hyperprior_channels: 96,
            quality_scale: 0.5,
            skip_threshold: 0.0001, // Less aggressive skip
            skip_block_size: 8,     // Smaller blocks
            adaptive_quant: true,
            min_quant_scale: 0.125,
            max_quant_scale: 8.0,
        }
    }
}

// -------------------------------------------------------------------------
// CompressedResidual
// -------------------------------------------------------------------------

/// Compressed residual representation for bitstream
///
/// Contains quantized latents and hyperprior data ready for entropy coding.
#[derive(Debug, Clone)]
pub struct CompressedResidual {
    /// Quantized residual latents (main data)
    pub latents: Vec<i32>,
    /// Quantized hyperprior latents (side information)
    pub hyperprior: Vec<i32>,
    /// Shape of latent tensor (channels, height, width)
    pub latent_shape: (usize, usize, usize),
    /// Shape of hyperprior tensor (channels, height, width)
    pub hyperprior_shape: (usize, usize, usize),
    /// Quantization scale used
    pub quant_scale: f32,
    /// Whether this is a skip frame (all zeros)
    pub is_skip: bool,
}

impl CompressedResidual {
    /// Create a new compressed residual
    pub fn new(
        latents: Vec<i32>,
        hyperprior: Vec<i32>,
        latent_shape: (usize, usize, usize),
        hyperprior_shape: (usize, usize, usize),
        quant_scale: f32,
    ) -> Self {
        Self {
            latents,
            hyperprior,
            latent_shape,
            hyperprior_shape,
            quant_scale,
            is_skip: false,
        }
    }

    /// Create a skip (zero) compressed residual
    pub fn skip(
        latent_shape: (usize, usize, usize),
        hyperprior_shape: (usize, usize, usize),
    ) -> Self {
        Self {
            latents: Vec::new(),
            hyperprior: Vec::new(),
            latent_shape,
            hyperprior_shape,
            quant_scale: 1.0,
            is_skip: true,
        }
    }

    /// Get number of latent elements
    pub fn num_latents(&self) -> usize {
        self.latent_shape.0 * self.latent_shape.1 * self.latent_shape.2
    }

    /// Get number of hyperprior elements
    pub fn num_hyperprior(&self) -> usize {
        self.hyperprior_shape.0 * self.hyperprior_shape.1 * self.hyperprior_shape.2
    }

    /// Entropy encode to bytes
    ///
    /// Uses the factorized prior for hyperprior and Gaussian conditional for latents.
    pub fn to_bytes(&self, entropy: &mut EntropyCoder) -> Result<Vec<u8>, ZVC69Error> {
        if self.is_skip {
            // Skip frame: just encode a marker
            return Ok(vec![0u8; 1]); // Single byte marker for skip
        }

        let mut result = Vec::new();

        // Encode skip flag (0 = not skip)
        result.push(1u8);

        // Encode hyperprior with factorized prior
        let hp_prior = FactorizedPrior::new(self.hyperprior_shape.0);
        let hp_bytes = hp_prior.encode(&self.hyperprior)?;

        // Write hyperprior length and data
        result.extend_from_slice(&(hp_bytes.len() as u32).to_le_bytes());
        result.extend_from_slice(&hp_bytes);

        // For latents, we need means and scales from hyperprior decoder
        // In placeholder mode, use zero mean and fixed scale
        let num_latents = self.num_latents();
        let means = vec![0.0f32; num_latents];
        let scales = vec![2.0f32; num_latents]; // Moderate scale for residuals

        let latent_bytes = entropy.encode_symbols(&self.latents, &means, &scales)?;

        // Write latent length and data
        result.extend_from_slice(&(latent_bytes.len() as u32).to_le_bytes());
        result.extend_from_slice(&latent_bytes);

        // Write metadata
        result.extend_from_slice(&(self.latent_shape.0 as u16).to_le_bytes());
        result.extend_from_slice(&(self.latent_shape.1 as u16).to_le_bytes());
        result.extend_from_slice(&(self.latent_shape.2 as u16).to_le_bytes());
        result.extend_from_slice(&(self.hyperprior_shape.0 as u16).to_le_bytes());
        result.extend_from_slice(&(self.hyperprior_shape.1 as u16).to_le_bytes());
        result.extend_from_slice(&(self.hyperprior_shape.2 as u16).to_le_bytes());
        result.extend_from_slice(&self.quant_scale.to_le_bytes());

        Ok(result)
    }

    /// Entropy decode from bytes
    pub fn from_bytes(
        data: &[u8],
        latent_shape: (usize, usize, usize),
        hyperprior_shape: (usize, usize, usize),
        entropy: &mut EntropyCoder,
    ) -> Result<Self, ZVC69Error> {
        if data.is_empty() {
            return Err(ZVC69Error::EntropyDecodingFailed {
                reason: "Empty compressed residual data".to_string(),
            });
        }

        // Check skip flag
        if data[0] == 0 {
            return Ok(Self::skip(latent_shape, hyperprior_shape));
        }

        let mut offset = 1;

        // Read hyperprior
        if data.len() < offset + 4 {
            return Err(ZVC69Error::EntropyDecodingFailed {
                reason: "Truncated hyperprior length".to_string(),
            });
        }
        let hp_len = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        if data.len() < offset + hp_len {
            return Err(ZVC69Error::EntropyDecodingFailed {
                reason: "Truncated hyperprior data".to_string(),
            });
        }
        let hp_bytes = &data[offset..offset + hp_len];
        offset += hp_len;

        let hp_prior = FactorizedPrior::new(hyperprior_shape.0);
        let num_hp = hyperprior_shape.0 * hyperprior_shape.1 * hyperprior_shape.2;
        let hyperprior = hp_prior.decode(hp_bytes, num_hp)?;

        // Read latents
        if data.len() < offset + 4 {
            return Err(ZVC69Error::EntropyDecodingFailed {
                reason: "Truncated latent length".to_string(),
            });
        }
        let latent_len = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        if data.len() < offset + latent_len {
            return Err(ZVC69Error::EntropyDecodingFailed {
                reason: "Truncated latent data".to_string(),
            });
        }
        let latent_bytes = &data[offset..offset + latent_len];
        offset += latent_len;

        let num_latents = latent_shape.0 * latent_shape.1 * latent_shape.2;
        let means = vec![0.0f32; num_latents];
        let scales = vec![2.0f32; num_latents];
        let latents = entropy.decode_symbols(latent_bytes, &means, &scales, num_latents)?;

        // Read metadata
        if data.len() < offset + 16 {
            // Use provided shapes if metadata missing
            return Ok(Self::new(
                latents,
                hyperprior,
                latent_shape,
                hyperprior_shape,
                1.0,
            ));
        }

        let quant_scale_offset = offset + 12;
        let quant_scale = if data.len() >= quant_scale_offset + 4 {
            f32::from_le_bytes([
                data[quant_scale_offset],
                data[quant_scale_offset + 1],
                data[quant_scale_offset + 2],
                data[quant_scale_offset + 3],
            ])
        } else {
            1.0
        };

        Ok(Self::new(
            latents,
            hyperprior,
            latent_shape,
            hyperprior_shape,
            quant_scale,
        ))
    }

    /// Estimate size in bits
    pub fn estimate_bits(&self) -> usize {
        if self.is_skip {
            return 8; // 1 byte marker
        }

        // Rough estimate: ~2 bits per latent, ~4 bits per hyperprior
        // Plus overhead
        let latent_bits = self.latents.len() * 2;
        let hp_bits = self.hyperprior.len() * 4;
        let overhead = 64; // Headers, shapes, etc.

        latent_bits + hp_bits + overhead
    }
}

// -------------------------------------------------------------------------
// ResidualEncoder
// -------------------------------------------------------------------------

/// Neural residual encoder for P-frame compression
///
/// Uses a smaller neural network than the I-frame encoder since residuals
/// typically have lower complexity than full frames.
#[cfg(feature = "zvc69")]
pub struct ResidualEncoder {
    /// ONNX session for residual analysis transform
    session: Option<Mutex<Session>>,
    /// ONNX session for hyperprior encoder
    hyperprior_session: Option<Mutex<Session>>,
    /// Configuration
    config: ResidualConfig,
}

#[cfg(not(feature = "zvc69"))]
pub struct ResidualEncoder {
    /// Configuration
    config: ResidualConfig,
}

impl ResidualEncoder {
    /// Create a new residual encoder
    pub fn new(config: ResidualConfig) -> Self {
        #[cfg(feature = "zvc69")]
        {
            Self {
                session: None,
                hyperprior_session: None,
                config,
            }
        }
        #[cfg(not(feature = "zvc69"))]
        {
            Self { config }
        }
    }

    /// Load neural model from path
    #[cfg(feature = "zvc69")]
    pub fn load_model(&mut self, path: &Path) -> Result<(), ZVC69Error> {
        use ort::session::Session;

        let model_path = path.join("residual_encoder.onnx");
        if !model_path.exists() {
            return Err(ZVC69Error::ModelNotFound {
                path: model_path.to_string_lossy().to_string(),
            });
        }

        let session = Session::builder()
            .map_err(|e| ZVC69Error::ModelLoadFailed {
                model_name: "residual_encoder".to_string(),
                reason: e.to_string(),
            })?
            .commit_from_file(&model_path)
            .map_err(|e| ZVC69Error::ModelLoadFailed {
                model_name: "residual_encoder".to_string(),
                reason: e.to_string(),
            })?;

        self.session = Some(Mutex::new(session));

        // Also try to load hyperprior encoder
        let hp_path = path.join("residual_hyperprior_enc.onnx");
        if hp_path.exists() {
            let hp_session = Session::builder()
                .map_err(|e| ZVC69Error::ModelLoadFailed {
                    model_name: "residual_hyperprior_enc".to_string(),
                    reason: e.to_string(),
                })?
                .commit_from_file(&hp_path)
                .map_err(|e| ZVC69Error::ModelLoadFailed {
                    model_name: "residual_hyperprior_enc".to_string(),
                    reason: e.to_string(),
                })?;

            self.hyperprior_session = Some(Mutex::new(hp_session));
        }

        Ok(())
    }

    #[cfg(not(feature = "zvc69"))]
    pub fn load_model(&mut self, _path: &Path) -> Result<(), ZVC69Error> {
        Err(ZVC69Error::FeatureNotEnabled)
    }

    /// Check if model is loaded
    pub fn has_model(&self) -> bool {
        #[cfg(feature = "zvc69")]
        {
            self.session.is_some()
        }
        #[cfg(not(feature = "zvc69"))]
        {
            false
        }
    }

    /// Get configuration
    pub fn config(&self) -> &ResidualConfig {
        &self.config
    }

    /// Encode residual to compressed representation (neural)
    #[cfg(feature = "zvc69")]
    pub fn encode(&self, residual: &Residual) -> Result<CompressedResidual, ZVC69Error> {
        // Check for skip mode first
        if should_skip_residual(residual, self.config.skip_threshold) {
            let (_, _, h, w) = residual.shape;
            let latent_h = h / RESIDUAL_LATENT_SPATIAL_FACTOR;
            let latent_w = w / RESIDUAL_LATENT_SPATIAL_FACTOR;
            let hp_h = h / RESIDUAL_HYPERPRIOR_SPATIAL_FACTOR;
            let hp_w = w / RESIDUAL_HYPERPRIOR_SPATIAL_FACTOR;

            return Ok(CompressedResidual::skip(
                (self.config.latent_channels, latent_h, latent_w),
                (self.config.hyperprior_channels, hp_h, hp_w),
            ));
        }

        if self.session.is_none() {
            return Ok(self.encode_placeholder(residual));
        }

        // Full neural encoding would go here
        // For now, fall back to placeholder
        Ok(self.encode_placeholder(residual))
    }

    #[cfg(not(feature = "zvc69"))]
    pub fn encode(&self, residual: &Residual) -> Result<CompressedResidual, ZVC69Error> {
        // Check for skip mode first
        if should_skip_residual(residual, self.config.skip_threshold) {
            let (_, _, h, w) = residual.shape;
            let latent_h = h / RESIDUAL_LATENT_SPATIAL_FACTOR;
            let latent_w = w / RESIDUAL_LATENT_SPATIAL_FACTOR;
            let hp_h = h / RESIDUAL_HYPERPRIOR_SPATIAL_FACTOR;
            let hp_w = w / RESIDUAL_HYPERPRIOR_SPATIAL_FACTOR;

            return Ok(CompressedResidual::skip(
                (self.config.latent_channels, latent_h, latent_w),
                (self.config.hyperprior_channels, hp_h, hp_w),
            ));
        }

        Ok(self.encode_placeholder(residual))
    }

    /// Placeholder encoding (for testing without neural model)
    ///
    /// Simulates neural encoding by:
    /// 1. Downsampling residual
    /// 2. Quantizing with quality scale
    /// 3. Creating synthetic hyperprior
    pub fn encode_placeholder(&self, residual: &Residual) -> CompressedResidual {
        let (batch, channels, height, width) = residual.shape;

        // Compute latent and hyperprior dimensions
        let latent_h =
            (height + RESIDUAL_LATENT_SPATIAL_FACTOR - 1) / RESIDUAL_LATENT_SPATIAL_FACTOR;
        let latent_w =
            (width + RESIDUAL_LATENT_SPATIAL_FACTOR - 1) / RESIDUAL_LATENT_SPATIAL_FACTOR;
        let hp_h =
            (height + RESIDUAL_HYPERPRIOR_SPATIAL_FACTOR - 1) / RESIDUAL_HYPERPRIOR_SPATIAL_FACTOR;
        let hp_w =
            (width + RESIDUAL_HYPERPRIOR_SPATIAL_FACTOR - 1) / RESIDUAL_HYPERPRIOR_SPATIAL_FACTOR;

        // Check for skip mode
        if should_skip_residual(residual, self.config.skip_threshold) {
            return CompressedResidual::skip(
                (self.config.latent_channels, latent_h, latent_w),
                (self.config.hyperprior_channels, hp_h, hp_w),
            );
        }

        // Create downsampled "latent" representation
        // This is a simplified simulation - real encoder uses neural network
        let mut latent_data =
            Array4::zeros((batch, self.config.latent_channels, latent_h, latent_w));

        // Simple averaging pooling simulation
        for b in 0..batch {
            for c in 0..self.config.latent_channels.min(channels) {
                for y in 0..latent_h {
                    for x in 0..latent_w {
                        let mut sum = 0.0f32;
                        let mut count = 0;

                        let src_c = c % channels;
                        for dy in 0..RESIDUAL_LATENT_SPATIAL_FACTOR {
                            for dx in 0..RESIDUAL_LATENT_SPATIAL_FACTOR {
                                let sy = y * RESIDUAL_LATENT_SPATIAL_FACTOR + dy;
                                let sx = x * RESIDUAL_LATENT_SPATIAL_FACTOR + dx;
                                if sy < height && sx < width {
                                    sum += residual.data[[b, src_c, sy, sx]];
                                    count += 1;
                                }
                            }
                        }

                        if count > 0 {
                            latent_data[[b, c, y, x]] = sum / count as f32;
                        }
                    }
                }
            }
        }

        // Quantize latents
        let quantized = quantize_scaled(&latent_data, self.config.quality_scale);
        let latents: Vec<i32> = quantized.iter().copied().collect();

        // Create synthetic hyperprior (further downsampled)
        let hp_data = Array4::zeros((batch, self.config.hyperprior_channels, hp_h, hp_w));
        let hp_quantized = quantize_scaled(&hp_data, self.config.quality_scale);
        let hyperprior: Vec<i32> = hp_quantized.iter().copied().collect();

        CompressedResidual::new(
            latents,
            hyperprior,
            (self.config.latent_channels, latent_h, latent_w),
            (self.config.hyperprior_channels, hp_h, hp_w),
            self.config.quality_scale,
        )
    }
}

// -------------------------------------------------------------------------
// ResidualDecoder
// -------------------------------------------------------------------------

/// Neural residual decoder for P-frame decompression
#[cfg(feature = "zvc69")]
pub struct ResidualDecoder {
    /// ONNX session for residual synthesis transform
    session: Option<Mutex<Session>>,
    /// ONNX session for hyperprior decoder
    hyperprior_session: Option<Mutex<Session>>,
    /// Configuration
    config: ResidualConfig,
}

#[cfg(not(feature = "zvc69"))]
pub struct ResidualDecoder {
    /// Configuration
    config: ResidualConfig,
}

impl ResidualDecoder {
    /// Create a new residual decoder
    pub fn new(config: ResidualConfig) -> Self {
        #[cfg(feature = "zvc69")]
        {
            Self {
                session: None,
                hyperprior_session: None,
                config,
            }
        }
        #[cfg(not(feature = "zvc69"))]
        {
            Self { config }
        }
    }

    /// Load neural model from path
    #[cfg(feature = "zvc69")]
    pub fn load_model(&mut self, path: &Path) -> Result<(), ZVC69Error> {
        use ort::session::Session;

        let model_path = path.join("residual_decoder.onnx");
        if !model_path.exists() {
            return Err(ZVC69Error::ModelNotFound {
                path: model_path.to_string_lossy().to_string(),
            });
        }

        let session = Session::builder()
            .map_err(|e| ZVC69Error::ModelLoadFailed {
                model_name: "residual_decoder".to_string(),
                reason: e.to_string(),
            })?
            .commit_from_file(&model_path)
            .map_err(|e| ZVC69Error::ModelLoadFailed {
                model_name: "residual_decoder".to_string(),
                reason: e.to_string(),
            })?;

        self.session = Some(Mutex::new(session));

        // Also try to load hyperprior decoder
        let hp_path = path.join("residual_hyperprior_dec.onnx");
        if hp_path.exists() {
            let hp_session = Session::builder()
                .map_err(|e| ZVC69Error::ModelLoadFailed {
                    model_name: "residual_hyperprior_dec".to_string(),
                    reason: e.to_string(),
                })?
                .commit_from_file(&hp_path)
                .map_err(|e| ZVC69Error::ModelLoadFailed {
                    model_name: "residual_hyperprior_dec".to_string(),
                    reason: e.to_string(),
                })?;

            self.hyperprior_session = Some(Mutex::new(hp_session));
        }

        Ok(())
    }

    #[cfg(not(feature = "zvc69"))]
    pub fn load_model(&mut self, _path: &Path) -> Result<(), ZVC69Error> {
        Err(ZVC69Error::FeatureNotEnabled)
    }

    /// Check if model is loaded
    pub fn has_model(&self) -> bool {
        #[cfg(feature = "zvc69")]
        {
            self.session.is_some()
        }
        #[cfg(not(feature = "zvc69"))]
        {
            false
        }
    }

    /// Get configuration
    pub fn config(&self) -> &ResidualConfig {
        &self.config
    }

    /// Decode compressed residual (neural)
    #[cfg(feature = "zvc69")]
    pub fn decode(
        &self,
        compressed: &CompressedResidual,
        output_height: usize,
        output_width: usize,
    ) -> Result<Residual, ZVC69Error> {
        if compressed.is_skip {
            return Ok(Residual::zeros(1, 3, output_height, output_width));
        }

        if self.session.is_none() {
            return Ok(self.decode_placeholder(compressed, output_height, output_width));
        }

        // Full neural decoding would go here
        // For now, fall back to placeholder
        Ok(self.decode_placeholder(compressed, output_height, output_width))
    }

    #[cfg(not(feature = "zvc69"))]
    pub fn decode(
        &self,
        compressed: &CompressedResidual,
        output_height: usize,
        output_width: usize,
    ) -> Result<Residual, ZVC69Error> {
        if compressed.is_skip {
            return Ok(Residual::zeros(1, 3, output_height, output_width));
        }

        Ok(self.decode_placeholder(compressed, output_height, output_width))
    }

    /// Placeholder decoding (for testing without neural model)
    pub fn decode_placeholder(
        &self,
        compressed: &CompressedResidual,
        output_height: usize,
        output_width: usize,
    ) -> Residual {
        if compressed.is_skip {
            return Residual::zeros(1, 3, output_height, output_width);
        }

        let (latent_c, latent_h, latent_w) = compressed.latent_shape;

        // Reshape latents to 4D tensor
        let latent_data: Vec<f32> = compressed.latents.iter().map(|&x| x as f32).collect();

        // Create output residual by upsampling
        let mut output = Array4::zeros((1, 3, output_height, output_width));

        // Simple nearest-neighbor upsampling simulation
        for c in 0..3 {
            for y in 0..output_height {
                for x in 0..output_width {
                    let src_c = c % latent_c;
                    let src_y = (y * latent_h) / output_height;
                    let src_x = (x * latent_w) / output_width;
                    let src_idx = src_c * latent_h * latent_w + src_y * latent_w + src_x;

                    if src_idx < latent_data.len() {
                        output[[0, c, y, x]] = latent_data[src_idx] * compressed.quant_scale;
                    }
                }
            }
        }

        Residual::new(output)
    }
}

// -------------------------------------------------------------------------
// Skip Mode Functions
// -------------------------------------------------------------------------

/// Determine if residual can be skipped (near zero)
///
/// Skip mode is used when the motion-compensated prediction is so good
/// that the residual is negligible. This saves bits by encoding just
/// a skip flag instead of residual data.
///
/// # Arguments
///
/// * `residual` - Residual to check
/// * `threshold` - Energy threshold (mean squared error)
///
/// # Returns
///
/// True if residual can be skipped
pub fn should_skip_residual(residual: &Residual, threshold: f32) -> bool {
    let stats = residual.stats();

    // Skip if normalized energy is below threshold
    let normalized_energy = if stats.num_elements > 0 {
        stats.energy / stats.num_elements as f32
    } else {
        return true; // Empty residual
    };

    normalized_energy < threshold
}

/// Create skip mask for selective residual coding
///
/// Creates a 2D binary mask indicating which blocks can be skipped.
/// This enables block-level skip decisions rather than frame-level.
///
/// # Arguments
///
/// * `residual` - Residual to analyze
/// * `block_size` - Size of blocks for skip decision
/// * `threshold` - Energy threshold per block
///
/// # Returns
///
/// 2D boolean array where true = skip this block
pub fn compute_skip_mask(residual: &Residual, block_size: usize, threshold: f32) -> Array2<bool> {
    let (_, _, height, width) = residual.shape;

    let num_blocks_h = (height + block_size - 1) / block_size;
    let num_blocks_w = (width + block_size - 1) / block_size;

    let mut mask = Array2::from_elem((num_blocks_h, num_blocks_w), false);

    for by in 0..num_blocks_h {
        for bx in 0..num_blocks_w {
            let y_start = by * block_size;
            let x_start = bx * block_size;
            let y_end = (y_start + block_size).min(height);
            let x_end = (x_start + block_size).min(width);

            // Compute block energy
            let mut energy = 0.0f32;
            let mut count = 0usize;

            for c in 0..residual.channels() {
                for y in y_start..y_end {
                    for x in x_start..x_end {
                        let val = residual.data[[0, c, y, x]];
                        energy += val * val;
                        count += 1;
                    }
                }
            }

            // Normalize and check threshold
            let normalized_energy = if count > 0 {
                energy / count as f32
            } else {
                0.0
            };

            mask[[by, bx]] = normalized_energy < threshold;
        }
    }

    mask
}

/// Count skippable blocks in a skip mask
pub fn count_skip_blocks(mask: &Array2<bool>) -> usize {
    mask.iter().filter(|&&x| x).count()
}

/// Get skip ratio (fraction of skippable blocks)
pub fn skip_ratio(mask: &Array2<bool>) -> f32 {
    let total = mask.len();
    if total == 0 {
        return 0.0;
    }
    count_skip_blocks(mask) as f32 / total as f32
}

// -------------------------------------------------------------------------
// Adaptive Quantization
// -------------------------------------------------------------------------

/// Adaptive quantization based on content and motion
///
/// Adjusts quantization scale per-element or per-block based on:
/// - Local residual magnitude (higher residual = finer quantization)
/// - Motion magnitude (higher motion = coarser quantization to save bits)
///
/// # Arguments
///
/// * `residual` - Residual to quantize
/// * `base_scale` - Base quantization scale
/// * `motion_magnitude` - Optional motion magnitude map [H, W]
///
/// # Returns
///
/// Tuple of (quantized values, effective scale used)
pub fn adaptive_quantize(
    residual: &Residual,
    base_scale: f32,
    motion_magnitude: Option<&Array2<f32>>,
) -> (Vec<i32>, f32) {
    let stats = residual.stats();

    // Adjust scale based on residual statistics
    let mut effective_scale = base_scale;

    // High-energy residuals need finer quantization
    let energy_factor = if stats.std > 0.0 {
        (1.0 + stats.std.log2().max(-2.0).min(2.0)) * 0.5
    } else {
        1.0
    };
    effective_scale *= energy_factor;

    // Adjust based on motion (if provided)
    if let Some(motion) = motion_magnitude {
        let mean_motion = motion.mean().unwrap_or(0.0);
        // Higher motion -> coarser quantization (save bits)
        let motion_factor = 1.0 + (mean_motion / 100.0).min(1.0);
        effective_scale *= motion_factor;
    }

    // Clamp to valid range
    effective_scale = effective_scale.clamp(0.1, 10.0);

    // Quantize
    let quantized: Vec<i32> = residual
        .data
        .iter()
        .map(|&x| (x / effective_scale).round() as i32)
        .collect();

    (quantized, effective_scale)
}

/// Per-block adaptive quantization
///
/// Computes different quantization scales for different blocks based on
/// local content characteristics.
pub fn adaptive_quantize_blockwise(
    residual: &Residual,
    base_scale: f32,
    block_size: usize,
    motion_magnitude: Option<&Array2<f32>>,
) -> (Vec<i32>, Array2<f32>) {
    let (_, _, height, width) = residual.shape;

    let num_blocks_h = (height + block_size - 1) / block_size;
    let num_blocks_w = (width + block_size - 1) / block_size;

    let mut scale_map = Array2::from_elem((num_blocks_h, num_blocks_w), base_scale);
    let mut quantized = Vec::with_capacity(residual.len());

    for by in 0..num_blocks_h {
        for bx in 0..num_blocks_w {
            let y_start = by * block_size;
            let x_start = bx * block_size;
            let y_end = (y_start + block_size).min(height);
            let x_end = (x_start + block_size).min(width);

            // Compute block statistics
            let mut energy = 0.0f32;
            let mut count = 0;

            for c in 0..residual.channels() {
                for y in y_start..y_end {
                    for x in x_start..x_end {
                        let val = residual.data[[0, c, y, x]];
                        energy += val * val;
                        count += 1;
                    }
                }
            }

            let block_std = if count > 0 {
                (energy / count as f32).sqrt()
            } else {
                0.0
            };

            // Compute adaptive scale for this block
            let mut block_scale = base_scale;

            // Adjust based on local energy
            let energy_factor = if block_std > 0.0 {
                (1.0 + block_std.log2().max(-2.0).min(2.0)) * 0.5
            } else {
                1.0
            };
            block_scale *= energy_factor;

            // Adjust based on motion (if provided)
            if let Some(motion) = motion_magnitude {
                let motion_y = (by * block_size + block_size / 2).min(motion.nrows() - 1);
                let motion_x = (bx * block_size + block_size / 2).min(motion.ncols() - 1);
                let local_motion = motion[[motion_y, motion_x]];
                let motion_factor = 1.0 + (local_motion / 100.0).min(1.0);
                block_scale *= motion_factor;
            }

            block_scale = block_scale.clamp(0.1, 10.0);
            scale_map[[by, bx]] = block_scale;
        }
    }

    // Now quantize using per-block scales
    for c in 0..residual.channels() {
        for y in 0..height {
            for x in 0..width {
                let by = y / block_size;
                let bx = x / block_size;
                let scale = scale_map[[by.min(num_blocks_h - 1), bx.min(num_blocks_w - 1)]];
                let val = residual.data[[0, c, y, x]];
                quantized.push((val / scale).round() as i32);
            }
        }
    }

    (quantized, scale_map)
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // ResidualStats Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_residual_stats_zero() {
        let data = Array4::zeros((1, 3, 16, 16));
        let stats = ResidualStats::compute(&data, 0.01);

        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.std, 0.0);
        assert_eq!(stats.energy, 0.0);
        assert_eq!(stats.sparsity, 1.0); // All zeros are below threshold
        assert_eq!(stats.max_abs, 0.0);
    }

    #[test]
    fn test_residual_stats_nonzero() {
        let data = Array4::from_shape_fn((1, 1, 4, 4), |(_, _, y, x)| {
            (y as f32 - 1.5) * (x as f32 - 1.5)
        });
        let stats = ResidualStats::compute(&data, 0.01);

        assert!(stats.energy > 0.0);
        assert!(stats.max_abs > 0.0);
        assert!(stats.sparsity < 1.0);
    }

    #[test]
    fn test_residual_stats_can_skip() {
        // Very low energy residual
        let data = Array4::from_elem((1, 3, 16, 16), 0.001);
        let stats = ResidualStats::compute(&data, 0.01);

        assert!(stats.can_skip(0.01, 0.9));

        // High energy residual
        let data = Array4::from_elem((1, 3, 16, 16), 10.0);
        let stats = ResidualStats::compute(&data, 0.01);

        assert!(!stats.can_skip(0.01, 0.9));
    }

    // -----------------------------------------------------------------------
    // Residual Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_residual_compute() {
        let current = Array4::from_elem((1, 3, 16, 16), 1.0);
        let predicted = Array4::from_elem((1, 3, 16, 16), 0.5);

        let residual = Residual::compute(&current, &predicted);

        assert_eq!(residual.shape, (1, 3, 16, 16));
        assert!((residual.data[[0, 0, 0, 0]] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_residual_reconstruct() {
        let current = Array4::from_shape_fn((1, 3, 8, 8), |(_, c, y, x)| {
            (c * 64 + y * 8 + x) as f32 / 100.0
        });
        let predicted = Array4::from_elem((1, 3, 8, 8), 0.25);

        let residual = Residual::compute(&current, &predicted);
        let reconstructed = Residual::reconstruct(&residual, &predicted);

        // Should exactly match current
        for ((a, b), _) in current.iter().zip(reconstructed.iter()).zip(0..) {
            assert!((a - b).abs() < 1e-5, "Mismatch at element");
        }
    }

    #[test]
    fn test_residual_zero() {
        let frame = Array4::from_elem((1, 3, 32, 32), 0.5);
        let residual = Residual::compute(&frame, &frame);

        assert!(residual.is_near_zero(0.001));
        let stats = residual.stats();
        assert_eq!(stats.energy, 0.0);
    }

    #[test]
    fn test_residual_clamp() {
        let data = Array4::from_shape_fn((1, 1, 4, 4), |(_, _, y, x)| {
            ((y * 4 + x) as f32 - 8.0) * 2.0 // Range: -16 to 14
        });
        let mut residual = Residual::new(data);

        residual.clamp(-5.0, 5.0);

        assert!(residual.data.iter().all(|&x| x >= -5.0 && x <= 5.0));
    }

    #[test]
    fn test_residual_scale() {
        let data = Array4::from_elem((1, 1, 4, 4), 2.0);
        let residual = Residual::new(data);

        let scaled = residual.scaled(0.5);

        assert!((scaled.data[[0, 0, 0, 0]] - 1.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // ResidualConfig Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_residual_config_default() {
        let config = ResidualConfig::default();

        assert_eq!(config.latent_channels, DEFAULT_RESIDUAL_LATENT_CHANNELS);
        assert_eq!(
            config.hyperprior_channels,
            DEFAULT_RESIDUAL_HYPERPRIOR_CHANNELS
        );
        assert!(config.adaptive_quant);
    }

    #[test]
    fn test_residual_config_builder() {
        let config = ResidualConfig::new()
            .with_latent_channels(128)
            .with_quality(5)
            .with_skip_threshold(0.005);

        assert_eq!(config.latent_channels, 128);
        assert!(config.quality_scale < 1.0); // Q5 is high quality
        assert_eq!(config.skip_threshold, 0.005);
    }

    #[test]
    fn test_residual_config_fast() {
        let config = ResidualConfig::fast();

        assert!(config.latent_channels < DEFAULT_RESIDUAL_LATENT_CHANNELS);
        assert!(config.quality_scale > 1.0); // Coarser quantization
    }

    #[test]
    fn test_residual_config_quality() {
        let config = ResidualConfig::quality();

        assert!(config.latent_channels > DEFAULT_RESIDUAL_LATENT_CHANNELS);
        assert!(config.quality_scale < 1.0); // Finer quantization
    }

    // -----------------------------------------------------------------------
    // CompressedResidual Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compressed_residual_skip() {
        let compressed = CompressedResidual::skip((96, 4, 4), (64, 1, 1));

        assert!(compressed.is_skip);
        assert!(compressed.latents.is_empty());
        assert!(compressed.hyperprior.is_empty());
    }

    #[test]
    fn test_compressed_residual_estimate_bits() {
        let compressed =
            CompressedResidual::new(vec![0i32; 100], vec![0i32; 10], (96, 4, 4), (64, 1, 1), 1.0);

        let bits = compressed.estimate_bits();
        assert!(bits > 0);

        // Skip should have minimal bits
        let skip = CompressedResidual::skip((96, 4, 4), (64, 1, 1));
        let skip_bits = skip.estimate_bits();
        assert!(skip_bits < bits);
    }

    // -----------------------------------------------------------------------
    // ResidualEncoder/Decoder Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_residual_encoder_placeholder() {
        let config = ResidualConfig::default();
        let encoder = ResidualEncoder::new(config);

        let residual = Residual::zeros(1, 3, 64, 64);
        let compressed = encoder.encode_placeholder(&residual);

        // Should be skip mode for zero residual
        assert!(compressed.is_skip);
    }

    #[test]
    fn test_residual_encoder_nonzero() {
        let config = ResidualConfig::default().with_skip_threshold(0.0);
        let encoder = ResidualEncoder::new(config);

        let data = Array4::from_elem((1, 3, 64, 64), 1.0);
        let residual = Residual::new(data);
        let compressed = encoder.encode_placeholder(&residual);

        assert!(!compressed.is_skip);
        assert!(!compressed.latents.is_empty());
    }

    #[test]
    fn test_residual_decoder_placeholder() {
        let config = ResidualConfig::default();
        let decoder = ResidualDecoder::new(config);

        let compressed = CompressedResidual::skip((96, 8, 8), (64, 2, 2));
        let decoded = decoder.decode_placeholder(&compressed, 64, 64);

        assert_eq!(decoded.shape, (1, 3, 64, 64));
        // Skip should produce zero residual
        assert!(decoded.is_near_zero(0.001));
    }

    #[test]
    fn test_residual_encode_decode_roundtrip() {
        let config = ResidualConfig::default().with_skip_threshold(0.0);
        let encoder = ResidualEncoder::new(config.clone());
        let decoder = ResidualDecoder::new(config);

        // Create non-zero residual
        let data = Array4::from_shape_fn((1, 3, 32, 32), |(_, c, y, x)| {
            ((c * 32 * 32 + y * 32 + x) as f32 / 1000.0 - 1.5)
        });
        let original = Residual::new(data);

        let compressed = encoder.encode_placeholder(&original);
        let decoded = decoder.decode_placeholder(&compressed, 32, 32);

        // Shapes should match
        assert_eq!(decoded.height(), original.height());
        assert_eq!(decoded.width(), original.width());
    }

    // -----------------------------------------------------------------------
    // Skip Mode Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_should_skip_residual_zero() {
        let residual = Residual::zeros(1, 3, 16, 16);
        assert!(should_skip_residual(&residual, 0.001));
    }

    #[test]
    fn test_should_skip_residual_large() {
        let data = Array4::from_elem((1, 3, 16, 16), 10.0);
        let residual = Residual::new(data);
        assert!(!should_skip_residual(&residual, 0.001));
    }

    #[test]
    fn test_compute_skip_mask() {
        // Create residual with some zero and non-zero blocks
        let mut data = Array4::zeros((1, 3, 32, 32));
        // Fill bottom-right quadrant with non-zero
        for c in 0..3 {
            for y in 16..32 {
                for x in 16..32 {
                    data[[0, c, y, x]] = 1.0;
                }
            }
        }
        let residual = Residual::new(data);

        let mask = compute_skip_mask(&residual, 16, 0.001);

        assert_eq!(mask.dim(), (2, 2));
        assert!(mask[[0, 0]]); // Top-left: zero -> skip
        assert!(mask[[0, 1]]); // Top-right: zero -> skip
        assert!(mask[[1, 0]]); // Bottom-left: zero -> skip
        assert!(!mask[[1, 1]]); // Bottom-right: non-zero -> don't skip
    }

    #[test]
    fn test_skip_ratio() {
        let mask = Array2::from_shape_vec((2, 2), vec![true, true, false, false]).unwrap();

        assert!((skip_ratio(&mask) - 0.5).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Adaptive Quantization Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_adaptive_quantize_basic() {
        let data = Array4::from_elem((1, 1, 4, 4), 2.0);
        let residual = Residual::new(data);

        let (quantized, scale) = adaptive_quantize(&residual, 1.0, None);

        assert_eq!(quantized.len(), 16);
        assert!(scale > 0.0);
    }

    #[test]
    fn test_adaptive_quantize_with_motion() {
        let data = Array4::from_elem((1, 1, 4, 4), 2.0);
        let residual = Residual::new(data);

        let motion = Array2::from_elem((4, 4), 50.0); // High motion

        let (_, scale_with_motion) = adaptive_quantize(&residual, 1.0, Some(&motion));
        let (_, scale_no_motion) = adaptive_quantize(&residual, 1.0, None);

        // High motion should increase scale (coarser quantization)
        assert!(scale_with_motion > scale_no_motion);
    }

    #[test]
    fn test_adaptive_quantize_blockwise() {
        let mut data = Array4::zeros((1, 1, 32, 32));
        // High energy in one quadrant
        for y in 0..16 {
            for x in 0..16 {
                data[[0, 0, y, x]] = 10.0;
            }
        }
        let residual = Residual::new(data);

        let (quantized, scale_map) = adaptive_quantize_blockwise(&residual, 1.0, 16, None);

        assert_eq!(quantized.len(), 32 * 32);
        assert_eq!(scale_map.dim(), (2, 2));

        // High-energy block should have different scale
        // (exact relationship depends on implementation)
    }

    // -----------------------------------------------------------------------
    // Entropy Coding Integration Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compressed_residual_entropy_roundtrip() {
        let config = ResidualConfig::default().with_skip_threshold(0.0);
        let encoder = ResidualEncoder::new(config);

        let data = Array4::from_elem((1, 3, 32, 32), 0.5);
        let residual = Residual::new(data);
        let compressed = encoder.encode_placeholder(&residual);

        // Encode to bytes
        let mut entropy = EntropyCoder::new();
        let bytes = compressed.to_bytes(&mut entropy).unwrap();

        assert!(!bytes.is_empty());

        // Decode from bytes
        let decoded = CompressedResidual::from_bytes(
            &bytes,
            compressed.latent_shape,
            compressed.hyperprior_shape,
            &mut entropy,
        )
        .unwrap();

        assert_eq!(decoded.latent_shape, compressed.latent_shape);
        assert_eq!(decoded.hyperprior_shape, compressed.hyperprior_shape);
    }

    #[test]
    fn test_compressed_residual_skip_entropy() {
        let compressed = CompressedResidual::skip((96, 4, 4), (64, 1, 1));

        let mut entropy = EntropyCoder::new();
        let bytes = compressed.to_bytes(&mut entropy).unwrap();

        // Skip should be very small
        assert_eq!(bytes.len(), 1);

        let decoded = CompressedResidual::from_bytes(
            &bytes,
            compressed.latent_shape,
            compressed.hyperprior_shape,
            &mut entropy,
        )
        .unwrap();

        assert!(decoded.is_skip);
    }

    // -----------------------------------------------------------------------
    // Quality vs Bitrate Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_quality_affects_bitrate() {
        let data = Array4::from_shape_fn((1, 3, 32, 32), |(_, c, y, x)| {
            ((c * 32 * 32 + y * 32 + x) as f32 / 500.0 - 3.0)
        });
        let residual = Residual::new(data);

        // Low quality (coarse quantization)
        let config_low = ResidualConfig::new()
            .with_quality(2)
            .with_skip_threshold(0.0);
        let encoder_low = ResidualEncoder::new(config_low);
        let compressed_low = encoder_low.encode_placeholder(&residual);

        // High quality (fine quantization)
        let config_high = ResidualConfig::new()
            .with_quality(7)
            .with_skip_threshold(0.0);
        let encoder_high = ResidualEncoder::new(config_high);
        let compressed_high = encoder_high.encode_placeholder(&residual);

        // Higher quality should have larger latent values (less quantization)
        let max_low = compressed_low
            .latents
            .iter()
            .map(|&x| x.abs())
            .max()
            .unwrap_or(0);
        let max_high = compressed_high
            .latents
            .iter()
            .map(|&x| x.abs())
            .max()
            .unwrap_or(0);

        // Higher quality = finer quantization = larger integer values
        assert!(max_high >= max_low);
    }
}
