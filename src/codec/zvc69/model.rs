//! ZVC69 Neural Network Model Loading and Inference
//!
//! This module provides ONNX model loading and inference capabilities for the ZVC69
//! neural video codec using the `ort` crate (ONNX Runtime bindings).
//!
//! ## Architecture
//!
//! The ZVC69 codec uses four neural network models:
//!
//! - **Encoder (Analysis Transform)**: Converts input frames to latent representations
//! - **Decoder (Synthesis Transform)**: Reconstructs frames from latent representations
//! - **Hyperprior Encoder**: Compresses latents to hyperprior side information
//! - **Hyperprior Decoder**: Produces entropy model parameters (means, scales)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use zvd::codec::zvc69::model::{NeuralModel, NeuralModelConfig, Device};
//! use std::path::Path;
//!
//! // Load models from directory
//! let model = NeuralModel::load(Path::new("models/zvc69/"))?;
//!
//! // Or load with custom configuration
//! let config = NeuralModelConfig::default()
//!     .with_device(Device::Cuda(0))
//!     .with_optimization_level(GraphOptimizationLevel::Level3);
//! let model = NeuralModel::load_with_config(Path::new("models/"), config)?;
//!
//! // Run inference
//! let latents = model.encode(&input_tensor)?;
//! let reconstructed = model.decode(&latents)?;
//! ```
//!
//! ## Model Files
//!
//! Expected model file names in the model directory:
//! - `encoder.onnx` - Analysis transform network
//! - `decoder.onnx` - Synthesis transform network
//! - `hyperprior_enc.onnx` - Hyperprior encoder
//! - `hyperprior_dec.onnx` - Hyperprior decoder

use super::error::ZVC69Error;
use crate::codec::frame::VideoFrame;
use crate::util::PixelFormat;
use ndarray::{Array4, ArrayView4, Ix4};
use std::path::Path;

#[cfg(feature = "zvc69")]
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
};
#[cfg(feature = "zvc69")]
use std::sync::Mutex;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Default number of latent channels in the main encoder
pub const DEFAULT_LATENT_CHANNELS: usize = 192;

/// Default number of hyperprior channels
pub const DEFAULT_HYPERPRIOR_CHANNELS: usize = 128;

/// Default number of inference threads (0 = auto)
pub const DEFAULT_NUM_THREADS: usize = 0;

/// Spatial downsampling factor for main latents (H/16, W/16)
pub const LATENT_SPATIAL_FACTOR: usize = 16;

/// Spatial downsampling factor for hyperprior latents (H/64, W/64)
pub const HYPERPRIOR_SPATIAL_FACTOR: usize = 64;

/// Model file names
pub const ENCODER_MODEL_NAME: &str = "encoder.onnx";
pub const DECODER_MODEL_NAME: &str = "decoder.onnx";
pub const HYPERPRIOR_ENC_MODEL_NAME: &str = "hyperprior_enc.onnx";
pub const HYPERPRIOR_DEC_MODEL_NAME: &str = "hyperprior_dec.onnx";

// ─────────────────────────────────────────────────────────────────────────────
// Device Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Compute device for neural network inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// CPU inference (default fallback)
    Cpu,
    /// CUDA GPU inference with specified device index
    Cuda(usize),
}

impl Default for Device {
    fn default() -> Self {
        Device::Cpu
    }
}

impl Device {
    /// Check if this device is a GPU
    pub fn is_gpu(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }

    /// Get the device index (0 for CPU)
    pub fn index(&self) -> usize {
        match self {
            Device::Cpu => 0,
            Device::Cuda(idx) => *idx,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Model Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for neural model loading and inference
#[derive(Debug, Clone)]
pub struct NeuralModelConfig {
    /// Number of latent channels in the encoder output
    pub latent_channels: usize,

    /// Number of hyperprior channels
    pub hyperprior_channels: usize,

    /// Graph optimization level for ONNX Runtime
    pub optimization_level: OptimizationLevel,

    /// Number of inference threads (0 = auto-detect)
    pub num_threads: usize,

    /// Compute device (CPU or CUDA)
    pub device: Device,

    /// Enable FP16 inference (for supported GPUs)
    pub use_fp16: bool,

    /// Enable memory pattern optimization
    pub enable_memory_pattern: bool,
}

/// Wrapper for GraphOptimizationLevel to avoid exposing ort types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// Disable all optimizations
    Disable,
    /// Basic optimizations (constant folding, redundant node elimination)
    Basic,
    /// Extended optimizations (includes Basic + more aggressive optimizations)
    Extended,
    /// All optimizations (includes Extended + layout optimizations)
    All,
}

impl Default for OptimizationLevel {
    fn default() -> Self {
        OptimizationLevel::All
    }
}

#[cfg(feature = "zvc69")]
impl From<OptimizationLevel> for GraphOptimizationLevel {
    fn from(level: OptimizationLevel) -> Self {
        match level {
            OptimizationLevel::Disable => GraphOptimizationLevel::Disable,
            OptimizationLevel::Basic => GraphOptimizationLevel::Level1,
            OptimizationLevel::Extended => GraphOptimizationLevel::Level2,
            OptimizationLevel::All => GraphOptimizationLevel::Level3,
        }
    }
}

impl Default for NeuralModelConfig {
    fn default() -> Self {
        NeuralModelConfig {
            latent_channels: DEFAULT_LATENT_CHANNELS,
            hyperprior_channels: DEFAULT_HYPERPRIOR_CHANNELS,
            optimization_level: OptimizationLevel::All,
            num_threads: DEFAULT_NUM_THREADS,
            device: Device::Cpu,
            use_fp16: false,
            enable_memory_pattern: true,
        }
    }
}

impl NeuralModelConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the compute device
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Set the optimization level
    pub fn with_optimization_level(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }

    /// Set the number of inference threads
    pub fn with_num_threads(mut self, threads: usize) -> Self {
        self.num_threads = threads;
        self
    }

    /// Enable FP16 inference
    pub fn with_fp16(mut self, enable: bool) -> Self {
        self.use_fp16 = enable;
        self
    }

    /// Set the number of latent channels
    pub fn with_latent_channels(mut self, channels: usize) -> Self {
        self.latent_channels = channels;
        self
    }

    /// Set the number of hyperprior channels
    pub fn with_hyperprior_channels(mut self, channels: usize) -> Self {
        self.hyperprior_channels = channels;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Data Structures
// ─────────────────────────────────────────────────────────────────────────────

/// Compressed latent representation from the encoder
///
/// The latent tensor has shape [B, C, H/16, W/16] where:
/// - B = batch size (typically 1)
/// - C = latent channels (typically 192)
/// - H/16, W/16 = spatially downsampled dimensions
#[derive(Debug, Clone)]
pub struct Latents {
    /// Main latent tensor [B, C, H/16, W/16]
    pub y: Array4<f32>,

    /// Original shape information (batch, channels, height, width)
    pub shape: (usize, usize, usize, usize),
}

impl Latents {
    /// Create new latents from an ndarray
    pub fn new(y: Array4<f32>) -> Self {
        let shape = y.dim();
        Latents {
            y,
            shape: (shape.0, shape.1, shape.2, shape.3),
        }
    }

    /// Get the batch size
    pub fn batch_size(&self) -> usize {
        self.shape.0
    }

    /// Get the number of channels
    pub fn channels(&self) -> usize {
        self.shape.1
    }

    /// Get the spatial height
    pub fn height(&self) -> usize {
        self.shape.2
    }

    /// Get the spatial width
    pub fn width(&self) -> usize {
        self.shape.3
    }

    /// Get total number of elements
    pub fn len(&self) -> usize {
        self.y.len()
    }

    /// Check if latents are empty
    pub fn is_empty(&self) -> bool {
        self.y.is_empty()
    }

    /// Get a view of the underlying data
    pub fn view(&self) -> ArrayView4<f32> {
        self.y.view()
    }

    /// Flatten latents for entropy coding
    pub fn flatten(&self) -> Vec<f32> {
        self.y.iter().copied().collect()
    }
}

/// Hyperprior side information
///
/// The hyperprior tensor has shape [B, C, H/64, W/64] where:
/// - B = batch size
/// - C = hyperprior channels (typically 128)
/// - H/64, W/64 = further spatially downsampled dimensions
#[derive(Debug, Clone)]
pub struct Hyperprior {
    /// Hyperprior latent tensor [B, C, H/64, W/64]
    pub z: Array4<f32>,
}

impl Hyperprior {
    /// Create new hyperprior from an ndarray
    pub fn new(z: Array4<f32>) -> Self {
        Hyperprior { z }
    }

    /// Get the shape of the hyperprior tensor
    pub fn shape(&self) -> (usize, usize, usize, usize) {
        let dim = self.z.dim();
        (dim.0, dim.1, dim.2, dim.3)
    }

    /// Get a view of the underlying data
    pub fn view(&self) -> ArrayView4<f32> {
        self.z.view()
    }

    /// Flatten hyperprior for entropy coding
    pub fn flatten(&self) -> Vec<f32> {
        self.z.iter().copied().collect()
    }
}

/// Entropy model parameters from hyperprior decoder
///
/// These parameters (means and scales) define the Gaussian distributions
/// used for entropy coding of the main latents.
#[derive(Debug, Clone)]
pub struct EntropyParams {
    /// Predicted means for each latent element [B, C, H/16, W/16]
    pub means: Array4<f32>,

    /// Predicted scales (standard deviations) [B, C, H/16, W/16]
    pub scales: Array4<f32>,
}

impl EntropyParams {
    /// Create new entropy parameters
    pub fn new(means: Array4<f32>, scales: Array4<f32>) -> Self {
        EntropyParams { means, scales }
    }

    /// Get the shape (should match latent shape)
    pub fn shape(&self) -> (usize, usize, usize, usize) {
        let dim = self.means.dim();
        (dim.0, dim.1, dim.2, dim.3)
    }

    /// Flatten means for entropy coding
    pub fn flatten_means(&self) -> Vec<f32> {
        self.means.iter().copied().collect()
    }

    /// Flatten scales for entropy coding
    pub fn flatten_scales(&self) -> Vec<f32> {
        self.scales.iter().copied().collect()
    }

    /// Validate that means and scales have the same shape
    pub fn validate(&self) -> Result<(), ZVC69Error> {
        if self.means.dim() != self.scales.dim() {
            return Err(ZVC69Error::InvalidModelArchitecture {
                details: format!(
                    "Entropy params shape mismatch: means {:?} vs scales {:?}",
                    self.means.dim(),
                    self.scales.dim()
                ),
            });
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Neural Model
// ─────────────────────────────────────────────────────────────────────────────

/// Neural network model for ZVC69 codec
///
/// This struct manages the four ONNX sessions required for the neural codec:
/// - Encoder (analysis transform)
/// - Decoder (synthesis transform)
/// - Hyperprior encoder
/// - Hyperprior decoder
#[cfg(feature = "zvc69")]
pub struct NeuralModel {
    /// Analysis transform: image -> latents
    encoder: Mutex<Session>,

    /// Synthesis transform: latents -> image
    decoder: Mutex<Session>,

    /// Hyperprior encoder: latents -> hyperprior
    hyperprior_enc: Mutex<Session>,

    /// Hyperprior decoder: hyperprior -> entropy parameters
    hyperprior_dec: Mutex<Session>,

    /// Model configuration
    config: NeuralModelConfig,
}

#[cfg(feature = "zvc69")]
impl NeuralModel {
    /// Load models from a directory containing ONNX files
    ///
    /// Expects the following files in the directory:
    /// - `encoder.onnx`
    /// - `decoder.onnx`
    /// - `hyperprior_enc.onnx`
    /// - `hyperprior_dec.onnx`
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Path to directory containing model files
    ///
    /// # Returns
    ///
    /// A new `NeuralModel` instance or an error if loading fails
    pub fn load(model_dir: &Path) -> Result<Self, ZVC69Error> {
        Self::load_with_config(model_dir, NeuralModelConfig::default())
    }

    /// Load models with custom configuration
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Path to directory containing model files
    /// * `config` - Model configuration
    ///
    /// # Returns
    ///
    /// A new `NeuralModel` instance or an error if loading fails
    pub fn load_with_config(
        model_dir: &Path,
        config: NeuralModelConfig,
    ) -> Result<Self, ZVC69Error> {
        // Check directory exists
        if !model_dir.exists() {
            return Err(ZVC69Error::ModelNotFound {
                path: model_dir.to_string_lossy().to_string(),
            });
        }

        // Build paths to model files
        let encoder_path = model_dir.join(ENCODER_MODEL_NAME);
        let decoder_path = model_dir.join(DECODER_MODEL_NAME);
        let hyperprior_enc_path = model_dir.join(HYPERPRIOR_ENC_MODEL_NAME);
        let hyperprior_dec_path = model_dir.join(HYPERPRIOR_DEC_MODEL_NAME);

        // Verify all files exist
        for (name, path) in [
            ("encoder", &encoder_path),
            ("decoder", &decoder_path),
            ("hyperprior_enc", &hyperprior_enc_path),
            ("hyperprior_dec", &hyperprior_dec_path),
        ] {
            if !path.exists() {
                return Err(ZVC69Error::ModelNotFound {
                    path: path.to_string_lossy().to_string(),
                });
            }
        }

        // Load each model
        let encoder = Self::create_session(&encoder_path, &config)?;
        let decoder = Self::create_session(&decoder_path, &config)?;
        let hyperprior_enc = Self::create_session(&hyperprior_enc_path, &config)?;
        let hyperprior_dec = Self::create_session(&hyperprior_dec_path, &config)?;

        Ok(NeuralModel {
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
            hyperprior_enc: Mutex::new(hyperprior_enc),
            hyperprior_dec: Mutex::new(hyperprior_dec),
            config,
        })
    }

    /// Load models from embedded bytes
    ///
    /// This is useful for bundling models with the application binary.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Encoder model bytes
    /// * `decoder` - Decoder model bytes
    /// * `hyperprior_enc` - Hyperprior encoder model bytes
    /// * `hyperprior_dec` - Hyperprior decoder model bytes
    ///
    /// # Returns
    ///
    /// A new `NeuralModel` instance or an error if loading fails
    pub fn load_from_bytes(
        encoder_bytes: &[u8],
        decoder_bytes: &[u8],
        hyperprior_enc_bytes: &[u8],
        hyperprior_dec_bytes: &[u8],
    ) -> Result<Self, ZVC69Error> {
        Self::load_from_bytes_with_config(
            encoder_bytes,
            decoder_bytes,
            hyperprior_enc_bytes,
            hyperprior_dec_bytes,
            NeuralModelConfig::default(),
        )
    }

    /// Load models from bytes with custom configuration
    pub fn load_from_bytes_with_config(
        encoder_bytes: &[u8],
        decoder_bytes: &[u8],
        hyperprior_enc_bytes: &[u8],
        hyperprior_dec_bytes: &[u8],
        config: NeuralModelConfig,
    ) -> Result<Self, ZVC69Error> {
        let encoder = Self::create_session_from_bytes(encoder_bytes, &config)?;
        let decoder = Self::create_session_from_bytes(decoder_bytes, &config)?;
        let hyperprior_enc = Self::create_session_from_bytes(hyperprior_enc_bytes, &config)?;
        let hyperprior_dec = Self::create_session_from_bytes(hyperprior_dec_bytes, &config)?;

        Ok(NeuralModel {
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
            hyperprior_enc: Mutex::new(hyperprior_enc),
            hyperprior_dec: Mutex::new(hyperprior_dec),
            config,
        })
    }

    /// Create an ONNX session from a file path
    fn create_session(path: &Path, config: &NeuralModelConfig) -> Result<Session, ZVC69Error> {
        let mut builder = Session::builder().map_err(|e| ZVC69Error::ModelLoadFailed {
            model_name: path.to_string_lossy().to_string(),
            reason: format!("Failed to create session builder: {}", e),
        })?;

        // Configure optimization level
        builder = builder
            .with_optimization_level(config.optimization_level.into())
            .map_err(|e| ZVC69Error::ModelLoadFailed {
                model_name: path.to_string_lossy().to_string(),
                reason: format!("Failed to set optimization level: {}", e),
            })?;

        // Configure threads if specified
        if config.num_threads > 0 {
            builder = builder
                .with_intra_threads(config.num_threads)
                .map_err(|e| ZVC69Error::ModelLoadFailed {
                    model_name: path.to_string_lossy().to_string(),
                    reason: format!("Failed to set thread count: {}", e),
                })?;
        }

        // Configure execution providers based on device
        builder = Self::configure_execution_providers(builder, config).map_err(|e| {
            ZVC69Error::ModelLoadFailed {
                model_name: path.to_string_lossy().to_string(),
                reason: format!("Failed to configure execution providers: {}", e),
            }
        })?;

        // Load model from file
        builder
            .commit_from_file(path)
            .map_err(|e| ZVC69Error::ModelLoadFailed {
                model_name: path.to_string_lossy().to_string(),
                reason: format!("Failed to load model: {}", e),
            })
    }

    /// Create an ONNX session from bytes
    fn create_session_from_bytes(
        bytes: &[u8],
        config: &NeuralModelConfig,
    ) -> Result<Session, ZVC69Error> {
        let mut builder = Session::builder().map_err(|e| ZVC69Error::ModelLoadFailed {
            model_name: "<bytes>".to_string(),
            reason: format!("Failed to create session builder: {}", e),
        })?;

        // Configure optimization level
        builder = builder
            .with_optimization_level(config.optimization_level.into())
            .map_err(|e| ZVC69Error::ModelLoadFailed {
                model_name: "<bytes>".to_string(),
                reason: format!("Failed to set optimization level: {}", e),
            })?;

        // Configure threads if specified
        if config.num_threads > 0 {
            builder = builder
                .with_intra_threads(config.num_threads)
                .map_err(|e| ZVC69Error::ModelLoadFailed {
                    model_name: "<bytes>".to_string(),
                    reason: format!("Failed to set thread count: {}", e),
                })?;
        }

        // Configure execution providers based on device
        builder = Self::configure_execution_providers(builder, config).map_err(|e| {
            ZVC69Error::ModelLoadFailed {
                model_name: "<bytes>".to_string(),
                reason: format!("Failed to configure execution providers: {}", e),
            }
        })?;

        // Load model from memory
        builder
            .commit_from_memory(bytes)
            .map_err(|e| ZVC69Error::ModelLoadFailed {
                model_name: "<bytes>".to_string(),
                reason: format!("Failed to load model from bytes: {}", e),
            })
    }

    /// Configure execution providers based on device setting
    fn configure_execution_providers(
        builder: ort::session::builder::SessionBuilder,
        config: &NeuralModelConfig,
    ) -> Result<ort::session::builder::SessionBuilder, ort::Error> {
        match config.device {
            Device::Cpu => {
                builder.with_execution_providers([CPUExecutionProvider::default().build()])
            }
            Device::Cuda(device_id) => builder.with_execution_providers([
                CUDAExecutionProvider::default()
                    .with_device_id(device_id as i32)
                    .build(),
                CPUExecutionProvider::default().build(), // Fallback
            ]),
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Inference Methods
    // ─────────────────────────────────────────────────────────────────────────

    /// Helper function to extract a 4D tensor from session outputs
    fn extract_output_tensor_4d(
        outputs: &ort::session::SessionOutputs,
        output_name: &str,
    ) -> Result<Array4<f32>, ZVC69Error> {
        let output = outputs
            .get(output_name)
            .ok_or_else(|| ZVC69Error::InferenceFailed {
                reason: format!("Output '{}' not found", output_name),
            })?;

        // Use try_extract_array() to get an ndarray view
        let array_view =
            output
                .try_extract_array::<f32>()
                .map_err(|e| ZVC69Error::InferenceFailed {
                    reason: format!("Failed to extract tensor: {}", e),
                })?;

        let dims = array_view.shape();
        if dims.len() != 4 {
            return Err(ZVC69Error::InferenceFailed {
                reason: format!("Expected 4D output, got {} dimensions", dims.len()),
            });
        }

        // Convert to owned Array4
        array_view
            .to_shape((dims[0], dims[1], dims[2], dims[3]))
            .map_err(|e| ZVC69Error::InferenceFailed {
                reason: format!("Failed to reshape tensor: {}", e),
            })?
            .to_owned()
            .into_dimensionality::<Ix4>()
            .map_err(|e| ZVC69Error::InferenceFailed {
                reason: format!("Failed to convert to 4D array: {}", e),
            })
    }

    /// Get the first output name from a session (Mutex-wrapped)
    fn first_output_name_from_mutex(session_mutex: &Mutex<Session>) -> Result<String, ZVC69Error> {
        let session = session_mutex
            .lock()
            .map_err(|_| ZVC69Error::InferenceFailed {
                reason: "Failed to acquire session lock".to_string(),
            })?;
        session
            .outputs
            .first()
            .map(|o| o.name.clone())
            .ok_or_else(|| ZVC69Error::InferenceFailed {
                reason: "Session has no outputs".to_string(),
            })
    }

    /// Get all output names from a session (Mutex-wrapped)
    fn output_names_from_mutex(session_mutex: &Mutex<Session>) -> Result<Vec<String>, ZVC69Error> {
        let session = session_mutex
            .lock()
            .map_err(|_| ZVC69Error::InferenceFailed {
                reason: "Failed to acquire session lock".to_string(),
            })?;
        Ok(session.outputs.iter().map(|o| o.name.clone()).collect())
    }

    /// Run encoder (analysis transform): image -> latents
    ///
    /// Converts an input image tensor to its latent representation.
    ///
    /// # Arguments
    ///
    /// * `image` - Input tensor with shape [B, 3, H, W] in range [0, 1] or [-1, 1]
    ///
    /// # Returns
    ///
    /// Latent representation with shape [B, C, H/16, W/16]
    pub fn encode(&self, image: &Array4<f32>) -> Result<Latents, ZVC69Error> {
        // Validate input shape
        let (_batch, channels, _height, _width) = image.dim();
        if channels != 3 {
            return Err(ZVC69Error::InvalidModelArchitecture {
                details: format!("Encoder expects 3 input channels, got {}", channels),
            });
        }

        // Create input tensor (clone to owned array)
        let input = ort::value::Tensor::from_array(image.clone()).map_err(|e| {
            ZVC69Error::InferenceFailed {
                reason: format!("Failed to create input tensor: {}", e),
            }
        })?;

        // Get output name before locking session
        let output_name = Self::first_output_name_from_mutex(&self.encoder)?;

        // Lock session and run inference
        let mut session = self
            .encoder
            .lock()
            .map_err(|_| ZVC69Error::InferenceFailed {
                reason: "Failed to acquire encoder session lock".to_string(),
            })?;

        let outputs =
            session
                .run(ort::inputs![input])
                .map_err(|e| ZVC69Error::InferenceFailed {
                    reason: format!("Encoder inference failed: {}", e),
                })?;

        // Extract output tensor using first output name
        let output_array = Self::extract_output_tensor_4d(&outputs, &output_name)?;

        Ok(Latents::new(output_array))
    }

    /// Run decoder (synthesis transform): latents -> image
    ///
    /// Reconstructs an image from its latent representation.
    ///
    /// # Arguments
    ///
    /// * `latents` - Latent representation from encoder or entropy decoder
    ///
    /// # Returns
    ///
    /// Reconstructed image tensor with shape [B, 3, H, W]
    pub fn decode(&self, latents: &Latents) -> Result<Array4<f32>, ZVC69Error> {
        // Create input tensor (clone to owned array)
        let input = ort::value::Tensor::from_array(latents.y.clone()).map_err(|e| {
            ZVC69Error::InferenceFailed {
                reason: format!("Failed to create input tensor: {}", e),
            }
        })?;

        // Get output name before locking session
        let output_name = Self::first_output_name_from_mutex(&self.decoder)?;

        // Lock session and run inference
        let mut session = self
            .decoder
            .lock()
            .map_err(|_| ZVC69Error::InferenceFailed {
                reason: "Failed to acquire decoder session lock".to_string(),
            })?;

        let outputs =
            session
                .run(ort::inputs![input])
                .map_err(|e| ZVC69Error::InferenceFailed {
                    reason: format!("Decoder inference failed: {}", e),
                })?;

        // Extract output tensor using first output name
        let output_array = Self::extract_output_tensor_4d(&outputs, &output_name)?;

        Ok(output_array)
    }

    /// Run hyperprior encoder: latents -> hyperprior
    ///
    /// Compresses the latent representation to hyperprior side information.
    ///
    /// # Arguments
    ///
    /// * `latents` - Latent tensor from the main encoder [B, C, H/16, W/16]
    ///
    /// # Returns
    ///
    /// Hyperprior representation [B, C_hp, H/64, W/64]
    pub fn encode_hyperprior(&self, latents: &Array4<f32>) -> Result<Hyperprior, ZVC69Error> {
        // Create input tensor (clone to owned array)
        let input = ort::value::Tensor::from_array(latents.clone()).map_err(|e| {
            ZVC69Error::InferenceFailed {
                reason: format!("Failed to create input tensor: {}", e),
            }
        })?;

        // Get output name before locking session
        let output_name = Self::first_output_name_from_mutex(&self.hyperprior_enc)?;

        // Lock session and run inference
        let mut session = self
            .hyperprior_enc
            .lock()
            .map_err(|_| ZVC69Error::InferenceFailed {
                reason: "Failed to acquire hyperprior encoder session lock".to_string(),
            })?;

        let outputs =
            session
                .run(ort::inputs![input])
                .map_err(|e| ZVC69Error::InferenceFailed {
                    reason: format!("Hyperprior encoder inference failed: {}", e),
                })?;

        // Extract output tensor using first output name
        let output_array = Self::extract_output_tensor_4d(&outputs, &output_name)?;

        Ok(Hyperprior::new(output_array))
    }

    /// Run hyperprior decoder: hyperprior -> entropy parameters
    ///
    /// Produces the entropy model parameters (means and scales) from hyperprior.
    ///
    /// # Arguments
    ///
    /// * `hyperprior` - Hyperprior representation from hyperprior encoder
    ///
    /// # Returns
    ///
    /// Entropy parameters (means and scales) for Gaussian entropy model
    pub fn decode_hyperprior(&self, hyperprior: &Hyperprior) -> Result<EntropyParams, ZVC69Error> {
        // Create input tensor (clone to owned array)
        let input = ort::value::Tensor::from_array(hyperprior.z.clone()).map_err(|e| {
            ZVC69Error::InferenceFailed {
                reason: format!("Failed to create input tensor: {}", e),
            }
        })?;

        // Get output names before locking session
        let output_names = Self::output_names_from_mutex(&self.hyperprior_dec)?;

        // Lock session and run inference
        let mut session = self
            .hyperprior_dec
            .lock()
            .map_err(|_| ZVC69Error::InferenceFailed {
                reason: "Failed to acquire hyperprior decoder session lock".to_string(),
            })?;

        let outputs =
            session
                .run(ort::inputs![input])
                .map_err(|e| ZVC69Error::InferenceFailed {
                    reason: format!("Hyperprior decoder inference failed: {}", e),
                })?;

        // The hyperprior decoder typically outputs a tensor with 2*C channels
        // where the first C channels are means and the second C channels are scales
        // Or it outputs two separate tensors for means and scales
        if output_names.len() >= 2 {
            // Two separate output tensors
            let means = Self::extract_output_tensor_4d(&outputs, &output_names[0])?;
            let scales = Self::extract_output_tensor_4d(&outputs, &output_names[1])?;

            Ok(EntropyParams::new(means, scales))
        } else if output_names.len() == 1 {
            // Single output with concatenated means and scales
            let combined = Self::extract_output_tensor_4d(&outputs, &output_names[0])?;

            let shape = combined.dim();
            if shape.1 % 2 != 0 {
                return Err(ZVC69Error::InferenceFailed {
                    reason: format!(
                        "Expected even number of channels for mean/scale split, got {}",
                        shape.1
                    ),
                });
            }

            // Assume channels are split in half: first half means, second half scales
            let half_channels = shape.1 / 2;

            // Split along channel dimension
            let means = combined
                .slice(ndarray::s![.., ..half_channels, .., ..])
                .to_owned();
            let scales = combined
                .slice(ndarray::s![.., half_channels.., .., ..])
                .to_owned();

            Ok(EntropyParams::new(means, scales))
        } else {
            Err(ZVC69Error::InferenceFailed {
                reason: "Hyperprior decoder produced no outputs".to_string(),
            })
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Accessors
    // ─────────────────────────────────────────────────────────────────────────

    /// Get the model configuration
    pub fn config(&self) -> &NeuralModelConfig {
        &self.config
    }

    /// Get the device being used
    pub fn device(&self) -> Device {
        self.config.device
    }

    /// Change the device for a model (requires reloading)
    ///
    /// Note: This returns a new model instance as sessions cannot be migrated.
    pub fn with_device(self, _device: Device) -> Result<Self, ZVC69Error> {
        // Sessions cannot be migrated to a new device after creation
        // This would require reloading the models
        Err(ZVC69Error::NotSupported(
            "Device change after model loading is not supported. Reload the model with the desired device.".to_string(),
        ))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stub Implementation (when zvc69 feature is disabled)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(not(feature = "zvc69"))]
pub struct NeuralModel {
    config: NeuralModelConfig,
}

#[cfg(not(feature = "zvc69"))]
impl NeuralModel {
    pub fn load(_model_dir: &Path) -> Result<Self, ZVC69Error> {
        Err(ZVC69Error::FeatureNotEnabled)
    }

    pub fn load_with_config(
        _model_dir: &Path,
        _config: NeuralModelConfig,
    ) -> Result<Self, ZVC69Error> {
        Err(ZVC69Error::FeatureNotEnabled)
    }

    pub fn load_from_bytes(
        _encoder: &[u8],
        _decoder: &[u8],
        _hyperprior_enc: &[u8],
        _hyperprior_dec: &[u8],
    ) -> Result<Self, ZVC69Error> {
        Err(ZVC69Error::FeatureNotEnabled)
    }

    pub fn encode(&self, _image: &Array4<f32>) -> Result<Latents, ZVC69Error> {
        Err(ZVC69Error::FeatureNotEnabled)
    }

    pub fn decode(&self, _latents: &Latents) -> Result<Array4<f32>, ZVC69Error> {
        Err(ZVC69Error::FeatureNotEnabled)
    }

    pub fn encode_hyperprior(&self, _latents: &Array4<f32>) -> Result<Hyperprior, ZVC69Error> {
        Err(ZVC69Error::FeatureNotEnabled)
    }

    pub fn decode_hyperprior(&self, _hyperprior: &Hyperprior) -> Result<EntropyParams, ZVC69Error> {
        Err(ZVC69Error::FeatureNotEnabled)
    }

    pub fn config(&self) -> &NeuralModelConfig {
        &self.config
    }

    pub fn device(&self) -> Device {
        self.config.device
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper Functions: Tensor/Frame Conversion
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a VideoFrame to a tensor for neural network input
///
/// Converts the video frame to a 4D tensor with shape [1, 3, H, W]
/// and normalizes pixel values to the range [0, 1].
///
/// # Arguments
///
/// * `frame` - Video frame to convert
///
/// # Returns
///
/// 4D tensor suitable for neural network input
pub fn image_to_tensor(frame: &VideoFrame) -> Result<Array4<f32>, ZVC69Error> {
    let width = frame.width as usize;
    let height = frame.height as usize;

    match frame.format {
        PixelFormat::RGB24 => {
            // RGB packed format: RGBRGBRGB...
            if frame.data.is_empty() || frame.data[0].is_empty() {
                return Err(ZVC69Error::EmptyFrame);
            }

            let data = frame.data[0].as_slice();
            let expected_size = width * height * 3;
            if data.len() < expected_size {
                return Err(ZVC69Error::InvalidConfig {
                    reason: format!(
                        "RGB24 frame data too small: {} < {}",
                        data.len(),
                        expected_size
                    ),
                });
            }

            // Create tensor [1, 3, H, W]
            let mut tensor = Array4::<f32>::zeros((1, 3, height, width));

            for y in 0..height {
                for x in 0..width {
                    let idx = (y * width + x) * 3;
                    tensor[[0, 0, y, x]] = data[idx] as f32 / 255.0; // R
                    tensor[[0, 1, y, x]] = data[idx + 1] as f32 / 255.0; // G
                    tensor[[0, 2, y, x]] = data[idx + 2] as f32 / 255.0; // B
                }
            }

            Ok(tensor)
        }
        PixelFormat::YUV420P => {
            // YUV420P: Y plane, then U plane (quarter size), then V plane (quarter size)
            if frame.data.len() < 3 {
                return Err(ZVC69Error::InvalidConfig {
                    reason: "YUV420P frame requires 3 planes".to_string(),
                });
            }

            let y_data = frame.data[0].as_slice();
            let u_data = frame.data[1].as_slice();
            let v_data = frame.data[2].as_slice();

            let uv_width = width / 2;
            let uv_height = height / 2;

            // Convert YUV to RGB and create tensor
            let mut tensor = Array4::<f32>::zeros((1, 3, height, width));

            for py in 0..height {
                for px in 0..width {
                    let y_idx = py * width + px;
                    let uv_idx = (py / 2) * uv_width + (px / 2);

                    let y = y_data.get(y_idx).copied().unwrap_or(128) as f32;
                    let u = u_data.get(uv_idx).copied().unwrap_or(128) as f32 - 128.0;
                    let v = v_data.get(uv_idx).copied().unwrap_or(128) as f32 - 128.0;

                    // BT.601 YUV to RGB conversion
                    let r = (y + 1.402 * v).clamp(0.0, 255.0);
                    let g = (y - 0.344136 * u - 0.714136 * v).clamp(0.0, 255.0);
                    let b = (y + 1.772 * u).clamp(0.0, 255.0);

                    tensor[[0, 0, py, px]] = r / 255.0;
                    tensor[[0, 1, py, px]] = g / 255.0;
                    tensor[[0, 2, py, px]] = b / 255.0;
                }
            }

            Ok(tensor)
        }
        PixelFormat::YUV444P => {
            // YUV444P: Full resolution for all planes
            if frame.data.len() < 3 {
                return Err(ZVC69Error::InvalidConfig {
                    reason: "YUV444P frame requires 3 planes".to_string(),
                });
            }

            let y_data = frame.data[0].as_slice();
            let u_data = frame.data[1].as_slice();
            let v_data = frame.data[2].as_slice();

            let mut tensor = Array4::<f32>::zeros((1, 3, height, width));

            for py in 0..height {
                for px in 0..width {
                    let idx = py * width + px;

                    let y = y_data.get(idx).copied().unwrap_or(128) as f32;
                    let u = u_data.get(idx).copied().unwrap_or(128) as f32 - 128.0;
                    let v = v_data.get(idx).copied().unwrap_or(128) as f32 - 128.0;

                    // BT.601 YUV to RGB conversion
                    let r = (y + 1.402 * v).clamp(0.0, 255.0);
                    let g = (y - 0.344136 * u - 0.714136 * v).clamp(0.0, 255.0);
                    let b = (y + 1.772 * u).clamp(0.0, 255.0);

                    tensor[[0, 0, py, px]] = r / 255.0;
                    tensor[[0, 1, py, px]] = g / 255.0;
                    tensor[[0, 2, py, px]] = b / 255.0;
                }
            }

            Ok(tensor)
        }
        _ => Err(ZVC69Error::UnsupportedPixelFormat {
            format: format!("{:?}", frame.format),
        }),
    }
}

/// Convert a tensor back to a VideoFrame
///
/// Converts a 4D tensor [1, 3, H, W] back to a VideoFrame in RGB24 format.
/// Values are expected to be in the range [0, 1].
///
/// # Arguments
///
/// * `tensor` - 4D tensor from neural network output
///
/// # Returns
///
/// VideoFrame in RGB24 format
pub fn tensor_to_image(tensor: &Array4<f32>) -> Result<VideoFrame, ZVC69Error> {
    let shape = tensor.dim();
    if shape.1 != 3 {
        return Err(ZVC69Error::InvalidModelArchitecture {
            details: format!("Expected 3 channels, got {}", shape.1),
        });
    }

    let height = shape.2 as u32;
    let width = shape.3 as u32;

    // Create RGB24 data
    let mut rgb_data = vec![0u8; (width * height * 3) as usize];

    for py in 0..shape.2 {
        for px in 0..shape.3 {
            let idx = (py * shape.3 + px) * 3;
            rgb_data[idx] = (tensor[[0, 0, py, px]].clamp(0.0, 1.0) * 255.0) as u8; // R
            rgb_data[idx + 1] = (tensor[[0, 1, py, px]].clamp(0.0, 1.0) * 255.0) as u8; // G
            rgb_data[idx + 2] = (tensor[[0, 2, py, px]].clamp(0.0, 1.0) * 255.0) as u8;
            // B
        }
    }

    let mut frame = VideoFrame::new(width, height, PixelFormat::RGB24);
    frame.data = vec![crate::util::Buffer::from_vec(rgb_data)];
    frame.linesize = vec![(width * 3) as usize];

    Ok(frame)
}

/// Normalize image tensor to [-1, 1] range
///
/// Many neural networks expect input in the [-1, 1] range instead of [0, 1].
///
/// # Arguments
///
/// * `tensor` - Mutable reference to tensor with values in [0, 1]
pub fn normalize_image(tensor: &mut Array4<f32>) {
    tensor.mapv_inplace(|x| x * 2.0 - 1.0);
}

/// Denormalize image tensor from [-1, 1] back to [0, 1]
///
/// # Arguments
///
/// * `tensor` - Mutable reference to tensor with values in [-1, 1]
pub fn denormalize_image(tensor: &mut Array4<f32>) {
    tensor.mapv_inplace(|x| (x + 1.0) / 2.0);
}

/// Normalize image tensor with ImageNet mean and std
///
/// Common normalization for models pretrained on ImageNet.
///
/// # Arguments
///
/// * `tensor` - Mutable reference to tensor with values in [0, 1]
pub fn normalize_imagenet(tensor: &mut Array4<f32>) {
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    let shape = tensor.dim();
    for c in 0..3 {
        for h in 0..shape.2 {
            for w in 0..shape.3 {
                tensor[[0, c, h, w]] = (tensor[[0, c, h, w]] - mean[c]) / std[c];
            }
        }
    }
}

/// Denormalize image tensor from ImageNet normalization back to [0, 1]
///
/// # Arguments
///
/// * `tensor` - Mutable reference to tensor with ImageNet normalization
pub fn denormalize_imagenet(tensor: &mut Array4<f32>) {
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    let shape = tensor.dim();
    for c in 0..3 {
        for h in 0..shape.2 {
            for w in 0..shape.3 {
                tensor[[0, c, h, w]] = tensor[[0, c, h, w]] * std[c] + mean[c];
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_default() {
        let device = Device::default();
        assert_eq!(device, Device::Cpu);
        assert!(!device.is_gpu());
        assert_eq!(device.index(), 0);
    }

    #[test]
    fn test_device_cuda() {
        let device = Device::Cuda(1);
        assert!(device.is_gpu());
        assert_eq!(device.index(), 1);
    }

    #[test]
    fn test_model_config_builder() {
        let config = NeuralModelConfig::new()
            .with_device(Device::Cuda(0))
            .with_optimization_level(OptimizationLevel::Extended)
            .with_num_threads(4)
            .with_fp16(true)
            .with_latent_channels(256);

        assert_eq!(config.device, Device::Cuda(0));
        assert_eq!(config.optimization_level, OptimizationLevel::Extended);
        assert_eq!(config.num_threads, 4);
        assert!(config.use_fp16);
        assert_eq!(config.latent_channels, 256);
    }

    #[test]
    fn test_latents_creation() {
        let y = Array4::<f32>::zeros((1, 192, 8, 8));
        let latents = Latents::new(y);

        assert_eq!(latents.batch_size(), 1);
        assert_eq!(latents.channels(), 192);
        assert_eq!(latents.height(), 8);
        assert_eq!(latents.width(), 8);
        assert_eq!(latents.len(), 192 * 8 * 8);
    }

    #[test]
    fn test_hyperprior_creation() {
        let z = Array4::<f32>::zeros((1, 128, 2, 2));
        let hyperprior = Hyperprior::new(z);

        let shape = hyperprior.shape();
        assert_eq!(shape, (1, 128, 2, 2));
    }

    #[test]
    fn test_entropy_params_validation() {
        let means = Array4::<f32>::zeros((1, 192, 8, 8));
        let scales = Array4::<f32>::ones((1, 192, 8, 8));
        let params = EntropyParams::new(means, scales);

        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_entropy_params_shape_mismatch() {
        let means = Array4::<f32>::zeros((1, 192, 8, 8));
        let scales = Array4::<f32>::ones((1, 128, 8, 8)); // Wrong channels
        let params = EntropyParams::new(means, scales);

        assert!(params.validate().is_err());
    }

    #[test]
    fn test_normalize_denormalize() {
        let mut tensor = Array4::<f32>::from_elem((1, 3, 2, 2), 0.5);

        normalize_image(&mut tensor);
        // 0.5 * 2.0 - 1.0 = 0.0
        assert!((tensor[[0, 0, 0, 0]] - 0.0).abs() < 1e-6);

        denormalize_image(&mut tensor);
        // (0.0 + 1.0) / 2.0 = 0.5
        assert!((tensor[[0, 0, 0, 0]] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_to_image_basic() {
        let tensor = Array4::<f32>::from_elem((1, 3, 4, 4), 0.5);
        let frame = tensor_to_image(&tensor).unwrap();

        assert_eq!(frame.width, 4);
        assert_eq!(frame.height, 4);
        assert_eq!(frame.format, PixelFormat::RGB24);
        assert_eq!(frame.data.len(), 1);
        assert_eq!(frame.data[0].len(), 4 * 4 * 3);

        // Check pixel value (0.5 * 255 = 127 or 128)
        let data = frame.data[0].as_slice();
        assert!(data[0] >= 127 && data[0] <= 128);
    }

    #[test]
    fn test_optimization_level_default() {
        let level = OptimizationLevel::default();
        assert_eq!(level, OptimizationLevel::All);
    }

    #[cfg(not(feature = "zvc69"))]
    #[test]
    fn test_neural_model_feature_disabled() {
        use std::path::PathBuf;
        let result = NeuralModel::load(&PathBuf::from("models/"));
        assert!(matches!(result, Err(ZVC69Error::FeatureNotEnabled)));
    }
}
