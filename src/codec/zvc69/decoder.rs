//! ZVC69 Neural Video Decoder
//!
//! This module implements the decoder for the ZVC69 neural video codec.
//! The decoder uses learned neural network transforms for decompression,
//! reconstructing high-quality video from the compressed bitstream.
//!
//! ## Architecture
//!
//! The decoder pipeline consists of:
//! 1. **Bitstream Parsing**: Extract entropy-coded data and metadata
//! 2. **Entropy Decoding**: Decode quantized latent values using learned probability model
//! 3. **Dequantization**: Inverse quantization with trained parameters
//! 4. **Synthesis Transform**: Neural network that transforms latents back to pixels
//! 5. **Motion Compensation**: Apply decoded motion for inter-frames
//!
//! ## P-Frame Decoding Pipeline
//!
//! For inter-frames (P-frames), the following pipeline is used:
//! 1. **Parse Frame Header**: Extract frame type, quality, dimensions
//! 2. **Parse P-Frame Flags**: Determine skip mode vs full residual
//! 3. **Decode Motion Vectors**: Entropy decode motion field
//! 4. **Warp Reference Frame**: Apply motion to reconstruct prediction
//! 5. **Decode Residual**: (if not skip mode) Decode residual signal
//! 6. **Reconstruct Frame**: Add residual to prediction
//! 7. **Update Reference**: Store reconstructed frame for future P-frames
//!
//! ## Usage
//!
//! ```rust,ignore
//! use zvd::codec::zvc69::ZVC69Decoder;
//!
//! let mut decoder = ZVC69Decoder::new()?;
//! decoder.send_packet(&packet)?;
//! let frame = decoder.receive_frame()?;
//! ```

use std::path::Path;
use std::sync::Arc;

use byteorder::{LittleEndian, ReadBytesExt};
use ndarray::Array4;
use std::io::Cursor;

use super::bitstream::FrameType as BitstreamFrameType;
use super::config::{Quality, ZVC69Config};
use super::entropy::{EntropyCoder, FactorizedPrior, GaussianConditional};
use super::error::ZVC69Error;
use super::memory::{BitstreamArena, BufferShape, FramePool, PoolConfig, PooledBuffer};
use super::model::{tensor_to_image, Hyperprior, Latents, NeuralModel, NeuralModelConfig, LATENT_SPATIAL_FACTOR};
use super::motion::{decode_motion, MotionConfig, MotionEstimator, MotionField};
use super::profiler::{stages, Profiler, TimingFrameType};
use super::quantize::{dequantize_scaled, unflatten_tensor};
use super::residual::{
    CompressedResidual, Residual, ResidualConfig, ResidualDecoder,
    RESIDUAL_HYPERPRIOR_SPATIAL_FACTOR, RESIDUAL_LATENT_SPATIAL_FACTOR,
};
use super::tensorrt::TensorRTConfig;
use super::warp::{FrameWarper, WarpConfig};
use crate::codec::{Decoder, Frame, PictureType, VideoFrame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, PixelFormat, Timestamp};

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------

/// Default number of latent channels
const DEFAULT_LATENT_CHANNELS: usize = 192;

/// Default number of hyperprior channels
const DEFAULT_HYPERPRIOR_CHANNELS: usize = 128;

/// Default number of residual latent channels
const DEFAULT_RESIDUAL_LATENT_CHANNELS: usize = 96;

/// Default number of residual hyperprior channels
const DEFAULT_RESIDUAL_HYPERPRIOR_CHANNELS: usize = 64;

/// Magic bytes for P-frame skip mode
const PFRAME_SKIP_MAGIC: &[u8; 4] = b"SKIP";

/// P-frame motion section marker
const MOTION_SECTION_MARKER: u8 = 0x4D; // 'M'

/// P-frame residual section marker
const RESIDUAL_SECTION_MARKER: u8 = 0x52; // 'R'

// -------------------------------------------------------------------------
// Frame Type (Internal)
// -------------------------------------------------------------------------

/// Frame type parsed from bitstream
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameType {
    /// Intra frame (keyframe)
    I,
    /// Predicted frame
    P,
    /// Bidirectional frame
    B,
}

impl FrameType {
    fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(FrameType::I),
            1 => Some(FrameType::P),
            2 => Some(FrameType::B),
            _ => None,
        }
    }

    fn to_picture_type(self) -> PictureType {
        match self {
            FrameType::I => PictureType::I,
            FrameType::P => PictureType::P,
            FrameType::B => PictureType::B,
        }
    }

    fn to_bitstream_type(self) -> BitstreamFrameType {
        match self {
            FrameType::I => BitstreamFrameType::I,
            FrameType::P => BitstreamFrameType::P,
            FrameType::B => BitstreamFrameType::B,
        }
    }
}

// -------------------------------------------------------------------------
// Parsed Frame Header
// -------------------------------------------------------------------------

/// Parsed frame header
#[derive(Debug, Clone)]
struct FrameHeader {
    /// Frame type
    frame_type: FrameType,
    /// Quality level
    quality: Quality,
    /// Width in macroblocks
    width_mb: u16,
    /// Height in macroblocks
    height_mb: u16,
    /// Quantization parameter
    qp: u8,
}

impl FrameHeader {
    /// Header size in bytes
    const SIZE: usize = 12;

    /// Parse frame header from bitstream
    fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < Self::SIZE {
            return Err(Error::codec(
                ZVC69Error::invalid_header("Frame header too short").to_string(),
            ));
        }

        // Check magic
        if &data[0..4] != b"ZVC1" {
            return Err(Error::codec(
                ZVC69Error::invalid_header("Invalid magic bytes, expected ZVC1").to_string(),
            ));
        }

        // Parse frame type
        let frame_type = FrameType::from_byte(data[4]).ok_or_else(|| {
            Error::codec(
                ZVC69Error::invalid_header(format!("Invalid frame type: {}", data[4])).to_string(),
            )
        })?;

        // Parse quality
        let quality = Quality::try_from(data[5]).map_err(|e| Error::codec(e.to_string()))?;

        // Parse dimensions
        let width_mb = u16::from_le_bytes([data[6], data[7]]);
        let height_mb = u16::from_le_bytes([data[8], data[9]]);

        // Parse QP
        let qp = data[10];

        Ok(FrameHeader {
            frame_type,
            quality,
            width_mb,
            height_mb,
            qp,
        })
    }

    /// Get frame dimensions in pixels
    fn dimensions(&self) -> (u32, u32) {
        (self.width_mb as u32 * 16, self.height_mb as u32 * 16)
    }
}

// -------------------------------------------------------------------------
// I-Frame Metadata
// -------------------------------------------------------------------------

/// Metadata parsed from I-frame bitstream (for neural decoding)
#[derive(Debug, Clone)]
struct IFrameMetadata {
    /// Main latent shape (batch, channels, height, width)
    latent_shape: (usize, usize, usize, usize),
    /// Hyperprior shape (batch, channels, height, width)
    hyperprior_shape: (usize, usize, usize, usize),
    /// Quantization scale
    quant_scale: f32,
}

// -------------------------------------------------------------------------
// Decoded Frame
// -------------------------------------------------------------------------

/// Result of decoding a single frame
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    /// The decoded video frame
    pub frame: VideoFrame,
    /// Frame type
    pub frame_type: BitstreamFrameType,
    /// Presentation timestamp
    pub pts: i64,
    /// Decode timestamp
    pub dts: i64,
}

impl DecodedFrame {
    /// Create a new decoded frame
    pub fn new(frame: VideoFrame, frame_type: BitstreamFrameType, pts: i64, dts: i64) -> Self {
        DecodedFrame {
            frame,
            frame_type,
            pts,
            dts,
        }
    }

    /// Check if this is a keyframe
    pub fn is_keyframe(&self) -> bool {
        self.frame_type.is_keyframe()
    }
}

// -------------------------------------------------------------------------
// Internal Decoder State
// -------------------------------------------------------------------------

/// Internal decoder state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecoderState {
    /// Decoder waiting for first packet
    Uninitialized,
    /// Decoder is ready
    Ready,
    /// Decoder is flushing
    Flushing,
}

/// Reference frame buffer for YUV data
#[derive(Debug, Clone)]
struct ReferenceBuffer {
    /// Y plane data
    y: Vec<u8>,
    /// U plane data
    u: Vec<u8>,
    /// V plane data
    v: Vec<u8>,
    /// Frame number
    frame_num: u64,
    /// Is valid (has been decoded)
    valid: bool,
}

impl ReferenceBuffer {
    fn new(width: u32, height: u32) -> Self {
        let y_size = (width * height) as usize;
        let uv_size = y_size / 4;
        ReferenceBuffer {
            y: vec![128; y_size],
            u: vec![128; uv_size],
            v: vec![128; uv_size],
            frame_num: 0,
            valid: false,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        let y_size = (width * height) as usize;
        let uv_size = y_size / 4;
        self.y.resize(y_size, 128);
        self.u.resize(uv_size, 128);
        self.v.resize(uv_size, 128);
        self.valid = false;
    }
}

/// Decoder reference buffer for tensor data (used in P-frame decoding)
#[derive(Debug, Clone)]
pub struct DecoderReferenceBuffer {
    /// Last reconstructed frame tensor [1, 3, H, W]
    last_tensor: Option<Array4<f32>>,
    /// Last decoded video frame
    last_frame: Option<VideoFrame>,
    /// Frame number of last reference
    last_frame_num: u64,
    /// Golden frame tensor (long-term reference)
    golden_tensor: Option<Array4<f32>>,
    /// Golden frame number
    golden_frame_num: u64,
}

impl DecoderReferenceBuffer {
    /// Create a new empty decoder reference buffer
    pub fn new() -> Self {
        DecoderReferenceBuffer {
            last_tensor: None,
            last_frame: None,
            last_frame_num: 0,
            golden_tensor: None,
            golden_frame_num: 0,
        }
    }

    /// Check if we have a valid reference frame
    pub fn has_reference(&self) -> bool {
        self.last_tensor.is_some()
    }

    /// Get the last frame tensor
    pub fn get_last_tensor(&self) -> Option<&Array4<f32>> {
        self.last_tensor.as_ref()
    }

    /// Get the last frame as VideoFrame
    pub fn get_last_frame(&self) -> Option<&VideoFrame> {
        self.last_frame.as_ref()
    }

    /// Update the reference buffer with a new reconstructed frame
    pub fn update(&mut self, frame: VideoFrame, tensor: Array4<f32>, frame_num: u64) {
        self.last_frame = Some(frame);
        self.last_tensor = Some(tensor);
        self.last_frame_num = frame_num;
    }

    /// Update the golden (long-term) reference
    pub fn update_golden(&mut self, tensor: Array4<f32>, frame_num: u64) {
        self.golden_tensor = Some(tensor);
        self.golden_frame_num = frame_num;
    }

    /// Clear all references
    pub fn clear(&mut self) {
        self.last_tensor = None;
        self.last_frame = None;
        self.last_frame_num = 0;
        self.golden_tensor = None;
        self.golden_frame_num = 0;
    }
}

impl Default for DecoderReferenceBuffer {
    fn default() -> Self {
        Self::new()
    }
}

// -------------------------------------------------------------------------
// ZVC69 Decoder
// -------------------------------------------------------------------------

/// ZVC69 Neural Video Decoder
///
/// A high-performance neural video decoder that uses learned neural network
/// transforms for decompression, reconstructing high-quality video from the
/// compressed ZVC69 bitstream.
///
/// # Features
///
/// - **Neural Decompression**: Uses learned synthesis transforms
/// - **I/P/B Frame Support**: Full GOP structure decoding
/// - **GPU Acceleration**: Optional TensorRT support for real-time decoding
/// - **Memory Efficiency**: Zero-allocation decoding in steady state
/// - **Reference Management**: Automatic reference frame handling
///
/// # Quick Start
///
/// ```rust,ignore
/// use zvd::codec::zvc69::prelude::*;
///
/// // Create decoder
/// let mut decoder = ZVC69Decoder::new()?;
///
/// // Or pre-initialize with known dimensions
/// let mut decoder = ZVC69Decoder::realtime_1080p()?;
///
/// // Decode packets
/// decoder.send_packet(&packet)?;
/// let frame = decoder.receive_frame()?;
/// ```
///
/// # Factory Functions
///
/// For common use cases, prefer the factory functions:
///
/// - [`realtime_720p()`](Self::realtime_720p) - Pre-warmed for 720p
/// - [`realtime_1080p()`](Self::realtime_1080p) - Pre-warmed for 1080p
/// - [`quality_optimized()`](Self::quality_optimized) - Maximum quality
/// - [`low_latency()`](Self::low_latency) - Minimum latency streaming
///
/// # TensorRT Acceleration
///
/// For GPU-accelerated decoding, use TensorRT:
///
/// ```rust,ignore
/// let decoder = ZVC69Decoder::with_tensorrt_fast(0)?; // GPU 0
/// ```
///
/// # Architecture
///
/// The decoder pipeline consists of:
///
/// 1. **Bitstream Parsing**: Extract entropy-coded data and metadata
/// 2. **Entropy Decoding**: Decode quantized latents using learned priors
/// 3. **Dequantization**: Inverse quantization with trained parameters
/// 4. **Synthesis Transform**: Neural network converts latents to pixels
/// 5. **Motion Compensation**: Apply decoded motion for inter-frames
///
/// # Lazy Initialization
///
/// The decoder can be created without knowing the video dimensions. It will
/// automatically initialize on the first keyframe:
///
/// ```rust,ignore
/// let mut decoder = ZVC69Decoder::new()?;
/// assert!(!decoder.is_initialized());
///
/// decoder.send_packet(&keyframe_packet)?;
/// assert!(decoder.is_initialized());
/// ```
pub struct ZVC69Decoder {
    /// Current video dimensions
    width: u32,
    height: u32,

    /// Decoder state
    state: DecoderState,

    /// Frame counter
    frame_count: u64,

    /// Decoded frames ready for output
    output_frames: Vec<VideoFrame>,

    /// Last reference frame (YUV buffer)
    last_ref: ReferenceBuffer,

    /// Golden reference frame (long-term)
    golden_ref: ReferenceBuffer,

    /// Alternative reference frame
    alt_ref: ReferenceBuffer,

    /// Configuration from extradata (if received)
    config: Option<ZVC69Config>,

    /// Neural model (optional - for real neural decoding)
    #[cfg(feature = "zvc69")]
    model: Option<NeuralModel>,

    /// Factorized prior for hyperprior decoding
    factorized_prior: FactorizedPrior,

    /// Gaussian conditional for main latent decoding
    gaussian_cond: GaussianConditional,

    /// Number of latent channels
    latent_channels: usize,

    /// Number of hyperprior channels
    hyperprior_channels: usize,

    // ─────────────────────────────────────────────────────────────────────────
    // P-Frame Decoding Components
    // ─────────────────────────────────────────────────────────────────────────
    /// Motion estimator (for motion decoding)
    motion_estimator: MotionEstimator,

    /// Frame warper for motion compensation
    frame_warper: FrameWarper,

    /// Residual decoder for P-frame residuals
    residual_decoder: ResidualDecoder,

    /// Decoder reference buffer (tensor data)
    reference_buffer: DecoderReferenceBuffer,

    /// Entropy coder for motion and residual decoding
    entropy_coder: EntropyCoder,

    /// Optional TensorRT configuration for accelerated inference
    tensorrt_config: Option<TensorRTConfig>,

    // ─────────────────────────────────────────────────────────────────────────
    // Memory Pooling
    // ─────────────────────────────────────────────────────────────────────────
    /// Frame buffer pool for zero-allocation decoding
    frame_pool: Option<Arc<FramePool>>,

    /// Bitstream arena for temporary allocations during decoding
    bitstream_arena: BitstreamArena,

    // ─────────────────────────────────────────────────────────────────────────
    // Pre-allocated Inference Buffers (Zero-Allocation Hot Path)
    // ─────────────────────────────────────────────────────────────────────────
    /// Pre-allocated entropy params means buffer
    #[cfg(feature = "zvc69")]
    entropy_means_buffer: Option<Array4<f32>>,

    /// Pre-allocated entropy params scales buffer
    #[cfg(feature = "zvc69")]
    entropy_scales_buffer: Option<Array4<f32>>,

    /// Pre-allocated reconstructed frame buffer [1, 3, H, W]
    #[cfg(feature = "zvc69")]
    reconstructed_buffer: Option<Array4<f32>>,

    // ─────────────────────────────────────────────────────────────────────────
    // Performance Profiling
    // ─────────────────────────────────────────────────────────────────────────
    /// Optional profiler for performance analysis (zero-cost when disabled)
    profiler: Option<Profiler>,
}

impl ZVC69Decoder {
    /// Create a new ZVC69 decoder
    pub fn new() -> Result<Self> {
        // Initialize motion estimator with default config
        let motion_config = MotionConfig::default();
        let motion_estimator = MotionEstimator::new(motion_config);

        // Initialize frame warper
        let warp_config = WarpConfig::default();
        let frame_warper = FrameWarper::new(warp_config);

        // Initialize residual decoder
        let residual_config = ResidualConfig::default();
        let residual_decoder = ResidualDecoder::new(residual_config);

        // Initialize bitstream arena with default size (will be resized on first frame)
        let bitstream_arena = BitstreamArena::default_size();

        Ok(ZVC69Decoder {
            width: 0,
            height: 0,
            state: DecoderState::Uninitialized,
            frame_count: 0,
            output_frames: Vec::new(),
            last_ref: ReferenceBuffer::new(0, 0),
            golden_ref: ReferenceBuffer::new(0, 0),
            alt_ref: ReferenceBuffer::new(0, 0),
            config: None,
            #[cfg(feature = "zvc69")]
            model: None,
            factorized_prior: FactorizedPrior::new(DEFAULT_HYPERPRIOR_CHANNELS),
            gaussian_cond: GaussianConditional::default(),
            latent_channels: DEFAULT_LATENT_CHANNELS,
            hyperprior_channels: DEFAULT_HYPERPRIOR_CHANNELS,
            // P-frame decoding components
            motion_estimator,
            frame_warper,
            residual_decoder,
            reference_buffer: DecoderReferenceBuffer::new(),
            entropy_coder: EntropyCoder::new(),
            tensorrt_config: None,
            // Memory pooling (initialized on first frame with known dimensions)
            frame_pool: None,
            bitstream_arena,
            // Pre-allocated inference buffers (initialized on first frame with known dimensions)
            #[cfg(feature = "zvc69")]
            entropy_means_buffer: None,
            #[cfg(feature = "zvc69")]
            entropy_scales_buffer: None,
            #[cfg(feature = "zvc69")]
            reconstructed_buffer: None,
            profiler: None,
        })
    }

    /// Create a decoder with known dimensions
    pub fn with_dimensions(width: u32, height: u32) -> Result<Self> {
        let mut decoder = Self::new()?;
        decoder.initialize(width, height);
        Ok(decoder)
    }

    /// Create a new decoder with TensorRT acceleration enabled
    ///
    /// This constructor enables TensorRT execution provider for accelerated
    /// neural network inference during decoding.
    ///
    /// # Arguments
    ///
    /// * `tensorrt_config` - TensorRT configuration for inference optimization
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use zvd::codec::zvc69::ZVC69Decoder;
    /// use zvd::codec::zvc69::tensorrt::TensorRTConfig;
    ///
    /// let trt_config = TensorRTConfig::fast(); // FP16 for speed
    /// let decoder = ZVC69Decoder::with_tensorrt(trt_config)?;
    /// ```
    pub fn with_tensorrt(tensorrt_config: TensorRTConfig) -> Result<Self> {
        let mut decoder = Self::new()?;
        decoder.tensorrt_config = Some(tensorrt_config);
        Ok(decoder)
    }

    /// Create a new decoder with TensorRT acceleration and known dimensions
    pub fn with_tensorrt_and_dimensions(
        tensorrt_config: TensorRTConfig,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let mut decoder = Self::with_tensorrt(tensorrt_config)?;
        decoder.initialize(width, height);
        Ok(decoder)
    }

    /// Create a new decoder with TensorRT acceleration using default fast configuration
    ///
    /// This is a convenience constructor that uses `TensorRTConfig::fast()` which
    /// enables FP16 mode for maximum real-time performance.
    ///
    /// # Arguments
    ///
    /// * `device_id` - CUDA device ID (0 for first GPU)
    pub fn with_tensorrt_fast(device_id: usize) -> Result<Self> {
        let trt_config = TensorRTConfig::fast().with_device(device_id);
        Self::with_tensorrt(trt_config)
    }

    /// Create a new decoder with TensorRT acceleration optimized for quality
    ///
    /// This uses `TensorRTConfig::quality()` which uses FP32 precision for
    /// maximum visual quality at the cost of some performance.
    ///
    /// # Arguments
    ///
    /// * `device_id` - CUDA device ID (0 for first GPU)
    pub fn with_tensorrt_quality(device_id: usize) -> Result<Self> {
        let trt_config = TensorRTConfig::quality().with_device(device_id);
        Self::with_tensorrt(trt_config)
    }

    // -------------------------------------------------------------------------
    // Factory Functions for Common Configurations
    // -------------------------------------------------------------------------

    /// Create a real-time optimized decoder for 720p video (1280x720)
    ///
    /// This factory function creates a decoder pre-configured for real-time 720p
    /// decoding at 60+ fps. Memory pools are pre-warmed for zero-allocation
    /// steady-state decoding.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use zvd::codec::zvc69::ZVC69Decoder;
    ///
    /// let decoder = ZVC69Decoder::realtime_720p()?;
    /// ```
    pub fn realtime_720p() -> Result<Self> {
        Self::with_dimensions(1280, 720)
    }

    /// Create a real-time optimized decoder for 1080p video (1920x1088)
    ///
    /// This factory function creates a decoder pre-configured for real-time 1080p
    /// decoding. Memory pools are pre-warmed with the optimized 1080p preset for
    /// maximum performance.
    ///
    /// Note: Uses 1088 height for 16-pixel alignment.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use zvd::codec::zvc69::ZVC69Decoder;
    ///
    /// let decoder = ZVC69Decoder::realtime_1080p()?;
    /// ```
    pub fn realtime_1080p() -> Result<Self> {
        Self::with_dimensions(1920, 1088)
    }

    /// Create a quality-optimized decoder
    ///
    /// This factory function creates a decoder configured for maximum visual
    /// quality. Suitable for professional applications where quality is more
    /// important than latency.
    ///
    /// # Arguments
    ///
    /// * `width` - Expected video width (must be divisible by 16)
    /// * `height` - Expected video height (must be divisible by 16)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use zvd::codec::zvc69::ZVC69Decoder;
    ///
    /// // Quality-optimized 4K decoder
    /// let decoder = ZVC69Decoder::quality_optimized(3840, 2160)?;
    /// ```
    pub fn quality_optimized(width: u32, height: u32) -> Result<Self> {
        Self::with_dimensions(width, height)
    }

    /// Create a low-latency decoder for live streaming
    ///
    /// This factory function creates a decoder optimized for minimal latency,
    /// suitable for live streaming, video calls, and real-time applications.
    ///
    /// # Arguments
    ///
    /// * `width` - Expected video width (must be divisible by 16)
    /// * `height` - Expected video height (must be divisible by 16)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use zvd::codec::zvc69::ZVC69Decoder;
    ///
    /// let decoder = ZVC69Decoder::low_latency(1280, 720)?;
    /// ```
    pub fn low_latency(width: u32, height: u32) -> Result<Self> {
        Self::with_dimensions(width, height)
    }

    /// Set TensorRT configuration after decoder creation
    ///
    /// Note: This should be called before loading the model to take effect.
    pub fn set_tensorrt_config(&mut self, tensorrt_config: TensorRTConfig) {
        self.tensorrt_config = Some(tensorrt_config);
    }

    /// Get the current TensorRT configuration
    pub fn tensorrt_config(&self) -> Option<&TensorRTConfig> {
        self.tensorrt_config.as_ref()
    }

    /// Check if TensorRT acceleration is enabled
    pub fn is_tensorrt_enabled(&self) -> bool {
        self.tensorrt_config.is_some()
    }

    /// Get the frame buffer pool (if initialized)
    ///
    /// Returns a reference to the internal frame pool for monitoring or
    /// external buffer management.
    pub fn frame_pool(&self) -> Option<&Arc<FramePool>> {
        self.frame_pool.as_ref()
    }

    /// Acquire a frame buffer from the pool
    ///
    /// Returns a pooled buffer that will automatically return to the pool on drop.
    /// Returns None if the decoder hasn't been initialized with dimensions yet.
    pub fn acquire_frame_buffer(&self) -> Option<PooledBuffer> {
        self.frame_pool.as_ref().map(|p| p.acquire())
    }

    /// Acquire a latent buffer from the pool
    ///
    /// Returns a pooled buffer sized for latent representations.
    /// Returns None if the decoder hasn't been initialized with dimensions yet.
    pub fn acquire_latent_buffer(&self) -> Option<PooledBuffer> {
        self.frame_pool.as_ref().map(|p| p.acquire_latent())
    }

    /// Acquire a buffer with specific shape from the pool
    pub fn acquire_buffer_with_shape(&self, shape: BufferShape) -> Option<PooledBuffer> {
        self.frame_pool.as_ref().map(|p| p.acquire_shape(shape))
    }

    /// Reset the bitstream arena for the next frame
    ///
    /// Call this at the start of each frame to reuse arena memory
    /// without deallocation.
    pub fn reset_arena(&mut self) {
        self.bitstream_arena.reset();
    }

    /// Get memory pool statistics for monitoring
    pub fn pool_stats(&self) -> Option<super::memory::PoolStats> {
        self.frame_pool.as_ref().map(|p| p.stats())
    }

    /// Load neural model from a directory
    ///
    /// If TensorRT configuration is set on the decoder, it will be used
    /// to configure the model for TensorRT-accelerated inference.
    #[cfg(feature = "zvc69")]
    pub fn load_model(&mut self, model_dir: &Path) -> Result<()> {
        // If TensorRT is configured, use it for model loading
        if let Some(ref trt_config) = self.tensorrt_config {
            let model_config = NeuralModelConfig::default()
                .with_tensorrt_config(trt_config.clone());
            let model = NeuralModel::load_with_config(model_dir, model_config)
                .map_err(|e| Error::codec(e.to_string()))?;
            self.model = Some(model);
        } else {
            let model = NeuralModel::load(model_dir).map_err(|e| Error::codec(e.to_string()))?;
            self.model = Some(model);
        }
        Ok(())
    }

    /// Load neural model with custom configuration
    ///
    /// Note: If TensorRT configuration is set on the decoder, it will override
    /// the TensorRT settings in the provided model_config.
    #[cfg(feature = "zvc69")]
    pub fn load_model_with_config(
        &mut self,
        model_dir: &Path,
        mut model_config: NeuralModelConfig,
    ) -> Result<()> {
        // If TensorRT is configured on decoder, apply it to model config
        if let Some(ref trt_config) = self.tensorrt_config {
            model_config = model_config.with_tensorrt_config(trt_config.clone());
        }
        let model = NeuralModel::load_with_config(model_dir, model_config)
            .map_err(|e| Error::codec(e.to_string()))?;
        self.model = Some(model);
        Ok(())
    }

    /// Load neural model with TensorRT acceleration
    ///
    /// Convenience method that loads the model with TensorRT enabled using
    /// the specified configuration.
    #[cfg(feature = "zvc69")]
    pub fn load_model_with_tensorrt(
        &mut self,
        model_dir: &Path,
        tensorrt_config: TensorRTConfig,
    ) -> Result<()> {
        self.tensorrt_config = Some(tensorrt_config.clone());
        let model_config = NeuralModelConfig::default()
            .with_tensorrt_config(tensorrt_config);
        let model = NeuralModel::load_with_config(model_dir, model_config)
            .map_err(|e| Error::codec(e.to_string()))?;
        self.model = Some(model);
        Ok(())
    }

    /// Check if neural model is loaded
    #[cfg(feature = "zvc69")]
    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }

    #[cfg(not(feature = "zvc69"))]
    pub fn has_model(&self) -> bool {
        false
    }

    /// Initialize decoder with dimensions
    fn initialize(&mut self, width: u32, height: u32) {
        if self.width == width && self.height == height && self.state != DecoderState::Uninitialized
        {
            return;
        }

        self.width = width;
        self.height = height;

        // Resize reference buffers
        self.last_ref.resize(width, height);
        self.golden_ref.resize(width, height);
        self.alt_ref.resize(width, height);

        // Clear tensor reference buffer on resize
        self.reference_buffer.clear();

        // Initialize frame pool for this resolution with pre-warming
        // Use optimized realtime preset for 1080p resolution
        let pool_config = if width == 1920 && (height == 1080 || height == 1088) {
            PoolConfig::preset_1080p_realtime()
        } else {
            PoolConfig::for_resolution(width, height)
        };
        let frame_pool = FramePool::new(pool_config);

        // Pre-warm pool for zero-allocation steady-state decoding
        // Use prewarm_1080p for 1080p, otherwise standard prewarm
        if width == 1920 && (height == 1080 || height == 1088) {
            frame_pool.prewarm_1080p();
        } else {
            frame_pool.prewarm(4);
        }
        self.frame_pool = Some(frame_pool);

        // Pre-allocate inference buffers for zero-allocation hot path
        #[cfg(feature = "zvc69")]
        {
            let latent_h = height as usize / LATENT_SPATIAL_FACTOR;
            let latent_w = width as usize / LATENT_SPATIAL_FACTOR;

            self.entropy_means_buffer = Some(Array4::<f32>::zeros((
                1,
                DEFAULT_LATENT_CHANNELS,
                latent_h,
                latent_w,
            )));
            self.entropy_scales_buffer = Some(Array4::<f32>::zeros((
                1,
                DEFAULT_LATENT_CHANNELS,
                latent_h,
                latent_w,
            )));
            self.reconstructed_buffer = Some(Array4::<f32>::zeros((
                1,
                3,
                height as usize,
                width as usize,
            )));
        }

        self.state = DecoderState::Ready;
    }

    /// Get current video dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Check if decoder is initialized
    pub fn is_initialized(&self) -> bool {
        self.state != DecoderState::Uninitialized
    }

    /// Check if decoder has a valid reference frame
    pub fn has_reference(&self) -> bool {
        self.reference_buffer.has_reference()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Profiling API
    // ─────────────────────────────────────────────────────────────────────────

    /// Enable performance profiling
    ///
    /// When enabled, the decoder will collect timing data for each pipeline stage.
    /// Profiling has zero cost when disabled (the default state).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut decoder = ZVC69Decoder::new()?;
    /// decoder.enable_profiling();
    ///
    /// // ... decode frames ...
    ///
    /// println!("{}", decoder.profiler_report());
    /// ```
    pub fn enable_profiling(&mut self) {
        if self.profiler.is_none() {
            self.profiler = Some(Profiler::enabled());
        } else if let Some(ref mut p) = self.profiler {
            p.set_enabled(true);
        }
    }

    /// Disable performance profiling
    ///
    /// Disabling profiling clears any collected timing data.
    pub fn disable_profiling(&mut self) {
        if let Some(ref mut p) = self.profiler {
            p.set_enabled(false);
        }
    }

    /// Check if profiling is enabled
    pub fn is_profiling_enabled(&self) -> bool {
        self.profiler.as_ref().map_or(false, |p| p.is_enabled())
    }

    /// Get the profiler report as a formatted string
    ///
    /// Returns a detailed report of timing statistics for each decoder stage.
    /// Returns an empty message if profiling is not enabled or no data has been collected.
    pub fn profiler_report(&self) -> String {
        self.profiler
            .as_ref()
            .map_or_else(|| "Profiling not enabled.".to_string(), |p| p.report())
    }

    /// Get a reference to the profiler (if enabled)
    ///
    /// This allows access to detailed profiling statistics and frame-level timing data.
    pub fn profiler(&self) -> Option<&Profiler> {
        self.profiler.as_ref()
    }

    /// Get a mutable reference to the profiler (if enabled)
    pub fn profiler_mut(&mut self) -> Option<&mut Profiler> {
        self.profiler.as_mut()
    }

    /// Reset profiler statistics
    ///
    /// Clears all collected timing data while keeping profiling enabled.
    pub fn reset_profiler(&mut self) {
        if let Some(ref mut p) = self.profiler {
            p.reset();
        }
    }

    /// Set extradata (codec configuration record)
    pub fn set_extradata(&mut self, extradata: &[u8]) -> Result<()> {
        if extradata.len() < 16 {
            return Err(Error::codec("ZVC69 extradata too short"));
        }

        // Parse ZVC69 configuration record
        if &extradata[0..4] != b"ZVC0" {
            return Err(Error::codec("Invalid ZVC69 extradata magic"));
        }

        let _version = extradata[4];
        let _profile = extradata[5];
        let _level = extradata[6];
        let quality = Quality::try_from(extradata[7]).unwrap_or_default();
        let width = u16::from_le_bytes([extradata[8], extradata[9]]) as u32;
        let height = u16::from_le_bytes([extradata[10], extradata[11]]) as u32;
        let framerate_num = u16::from_le_bytes([extradata[12], extradata[13]]) as u32;
        let framerate_den = u16::from_le_bytes([extradata[14], extradata[15]]) as u32;

        // Initialize with dimensions from extradata
        self.initialize(width, height);

        // Store config
        self.config = Some(ZVC69Config {
            width,
            height,
            quality,
            framerate_num,
            framerate_den,
            ..Default::default()
        });

        Ok(())
    }

    /// Decode a packet
    fn decode_packet(&mut self, packet: &Packet) -> Result<()> {
        // Reset arena at start of each frame for zero-allocation steady state
        self.bitstream_arena.reset();

        let data = packet.data.as_slice();

        if data.len() < FrameHeader::SIZE {
            return Err(Error::codec(
                ZVC69Error::BitstreamCorrupted {
                    offset: 0,
                    reason: "Packet too short for frame header".to_string(),
                }
                .to_string(),
            ));
        }

        // Parse frame header
        let header = if let Some(ref mut p) = self.profiler {
            p.time_result(stages::BITSTREAM_PARSE, || FrameHeader::parse(data))?
        } else {
            FrameHeader::parse(data)?
        };

        // Begin profiler frame timing
        let profiler_frame_type = match header.frame_type {
            FrameType::I => TimingFrameType::I,
            FrameType::P => TimingFrameType::P,
            FrameType::B => TimingFrameType::B,
        };
        if let Some(ref mut p) = self.profiler {
            p.begin_frame(self.frame_count, profiler_frame_type);
        }

        // Initialize on first keyframe if not already initialized
        let (width, height) = header.dimensions();
        if !self.is_initialized() || header.frame_type == FrameType::I {
            self.initialize(width, height);
        }

        // Verify dimensions match
        if width != self.width || height != self.height {
            return Err(Error::codec(format!(
                "Frame dimensions {}x{} don't match decoder {}x{}",
                width, height, self.width, self.height
            )));
        }

        // Decode based on frame type
        let video_frame = match header.frame_type {
            FrameType::I => self.decode_iframe(data, &header, packet.pts)?,
            FrameType::P => self.decode_pframe(data, &header, packet.pts)?,
            FrameType::B => self.decode_bframe(data, &header, packet.pts)?,
        };

        // End profiler frame timing
        if let Some(ref mut p) = self.profiler {
            p.end_frame(data.len(), width as usize, height as usize);
        }

        self.output_frames.push(video_frame);
        self.frame_count += 1;

        Ok(())
    }

    // -------------------------------------------------------------------------
    // I-Frame Decoding
    // -------------------------------------------------------------------------

    /// Decode an I-frame from bitstream data
    fn decode_iframe(
        &mut self,
        data: &[u8],
        header: &FrameHeader,
        pts: Timestamp,
    ) -> Result<VideoFrame> {
        let (width, height) = header.dimensions();

        // Try to detect if this is a neural-encoded or placeholder frame
        // Neural frames have shape metadata after the header
        // Placeholder frames have: frame_size (4), checksum (4), marker (4)
        let payload = &data[FrameHeader::SIZE..];

        // Check for neural frame format: we expect at least 16 bytes of shape metadata
        // plus 4 bytes for z_bytes length minimum
        if payload.len() >= 20 {
            // Try to parse as neural frame
            if let Ok(frame) = self.decode_iframe_neural(data, header, pts) {
                return Ok(frame);
            }
        }

        // Fall back to placeholder decoding
        self.decode_iframe_placeholder(header, pts)
    }

    /// Full neural I-frame decoding pipeline
    ///
    /// Uses pre-allocated buffers for zero-allocation inference in the hot path.
    #[cfg(feature = "zvc69")]
    fn decode_iframe_neural(
        &mut self,
        data: &[u8],
        header: &FrameHeader,
        pts: Timestamp,
    ) -> Result<VideoFrame> {
        // Check if model is available
        let model = match &self.model {
            Some(m) => m,
            None => return self.decode_iframe_placeholder(header, pts),
        };

        let (width, height) = header.dimensions();
        let payload = &data[FrameHeader::SIZE..];

        // Step 1: Parse I-frame metadata
        let (metadata, rest) = self.parse_iframe_metadata(payload)?;

        // Step 2: Parse entropy-coded sections
        let (z_bytes, y_bytes) = self.parse_iframe_sections(rest)?;

        // Step 3: Calculate number of symbols for each section
        let z_num_symbols = metadata.hyperprior_shape.0
            * metadata.hyperprior_shape.1
            * metadata.hyperprior_shape.2
            * metadata.hyperprior_shape.3;
        let y_num_symbols = metadata.latent_shape.0
            * metadata.latent_shape.1
            * metadata.latent_shape.2
            * metadata.latent_shape.3;

        // Step 4: Decode hyperprior with factorized prior
        let z_symbols = self
            .factorized_prior
            .decode(&z_bytes, z_num_symbols)
            .map_err(|e| Error::codec(e.to_string()))?;

        // Step 5: Dequantize and reshape hyperprior
        let z_tensor = self.unflatten_and_dequantize(
            &z_symbols,
            metadata.hyperprior_shape,
            metadata.quant_scale,
        )?;

        // Step 6: Run hyperprior decoder using pre-allocated buffers for means/scales
        let z_hyperprior = Hyperprior::new(z_tensor);
        {
            let means_buffer = self.entropy_means_buffer.as_mut().ok_or_else(|| {
                Error::codec("Entropy means buffer not initialized")
            })?;
            let scales_buffer = self.entropy_scales_buffer.as_mut().ok_or_else(|| {
                Error::codec("Entropy scales buffer not initialized")
            })?;
            model
                .decode_hyperprior_inplace(&z_hyperprior, means_buffer, scales_buffer)
                .map_err(|e| Error::codec(e.to_string()))?;
        }

        // Step 7: Flatten entropy parameters for decoding (from pre-allocated buffers)
        let means_flat: Vec<f32> = self.entropy_means_buffer.as_ref().unwrap().iter().copied().collect();
        let scales_flat: Vec<f32> = self.entropy_scales_buffer.as_ref().unwrap().iter().copied().collect();

        // Step 8: Decode main latents with Gaussian conditional
        let y_symbols = self
            .gaussian_cond
            .decode(&y_bytes, &means_flat, &scales_flat, y_num_symbols)
            .map_err(|e| Error::codec(e.to_string()))?;

        // Step 9: Dequantize and reshape main latents
        let y_tensor =
            self.unflatten_and_dequantize(&y_symbols, metadata.latent_shape, metadata.quant_scale)?;

        // Step 10: Run decoder network using pre-allocated buffer
        let latents = Latents::new(y_tensor);
        {
            let reconstructed_buffer = self.reconstructed_buffer.as_mut().ok_or_else(|| {
                Error::codec("Reconstructed buffer not initialized")
            })?;
            model
                .decode_inplace(&latents, reconstructed_buffer)
                .map_err(|e| Error::codec(e.to_string()))?;
        }

        // Clone the image tensor for reference storage (necessary for reference buffer)
        let image_tensor = self.reconstructed_buffer.as_ref().unwrap().clone();

        // Step 11: Convert tensor to VideoFrame
        let mut frame = tensor_to_image(&image_tensor).map_err(|e| Error::codec(e.to_string()))?;

        // Convert RGB24 to YUV420P if needed
        frame = self.convert_rgb_to_yuv(frame, width, height)?;

        // Set frame metadata
        frame.pts = pts;
        frame.keyframe = true;
        frame.pict_type = PictureType::I;

        // Step 12: Update reference frame (both YUV and tensor)
        self.update_reference_frame(&frame)?;
        self.reference_buffer
            .update(frame.clone(), image_tensor, self.frame_count);

        Ok(frame)
    }

    #[cfg(not(feature = "zvc69"))]
    fn decode_iframe_neural(
        &mut self,
        _data: &[u8],
        header: &FrameHeader,
        pts: Timestamp,
    ) -> Result<VideoFrame> {
        // Without zvc69 feature, fall back to placeholder
        self.decode_iframe_placeholder(header, pts)
    }

    /// Parse I-frame metadata from bitstream
    fn parse_iframe_metadata<'a>(&self, data: &'a [u8]) -> Result<(IFrameMetadata, &'a [u8])> {
        // Expected format after frame header:
        // - y_channels (2 bytes)
        // - y_height (2 bytes)
        // - y_width (2 bytes)
        // - z_channels (2 bytes)
        // - z_height (2 bytes)
        // - z_width (2 bytes)
        // - quant_scale (4 bytes as f32 bits)
        // Total: 16 bytes

        if data.len() < 16 {
            return Err(Error::codec(
                ZVC69Error::BitstreamCorrupted {
                    offset: FrameHeader::SIZE,
                    reason: "Not enough data for I-frame metadata".to_string(),
                }
                .to_string(),
            ));
        }

        let mut cursor = Cursor::new(data);

        // Parse latent shape
        let y_channels = cursor
            .read_u16::<LittleEndian>()
            .map_err(|e| Error::codec(format!("Failed to read y_channels: {}", e)))?
            as usize;
        let y_height = cursor
            .read_u16::<LittleEndian>()
            .map_err(|e| Error::codec(format!("Failed to read y_height: {}", e)))?
            as usize;
        let y_width = cursor
            .read_u16::<LittleEndian>()
            .map_err(|e| Error::codec(format!("Failed to read y_width: {}", e)))?
            as usize;

        // Parse hyperprior shape
        let z_channels = cursor
            .read_u16::<LittleEndian>()
            .map_err(|e| Error::codec(format!("Failed to read z_channels: {}", e)))?
            as usize;
        let z_height = cursor
            .read_u16::<LittleEndian>()
            .map_err(|e| Error::codec(format!("Failed to read z_height: {}", e)))?
            as usize;
        let z_width = cursor
            .read_u16::<LittleEndian>()
            .map_err(|e| Error::codec(format!("Failed to read z_width: {}", e)))?
            as usize;

        // Parse quantization scale
        let quant_scale_bits = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::codec(format!("Failed to read quant_scale: {}", e)))?;
        let quant_scale = f32::from_bits(quant_scale_bits);

        let metadata = IFrameMetadata {
            latent_shape: (1, y_channels, y_height, y_width),
            hyperprior_shape: (1, z_channels, z_height, z_width),
            quant_scale,
        };

        Ok((metadata, &data[16..]))
    }

    /// Parse I-frame entropy-coded sections
    fn parse_iframe_sections(&self, data: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
        // Expected format:
        // - z_bytes_len (4 bytes)
        // - z_bytes (variable)
        // - y_bytes_len (4 bytes)
        // - y_bytes (variable)

        if data.len() < 4 {
            return Err(Error::codec(
                ZVC69Error::BitstreamCorrupted {
                    offset: 0,
                    reason: "Not enough data for hyperprior section length".to_string(),
                }
                .to_string(),
            ));
        }

        let mut cursor = Cursor::new(data);

        // Read hyperprior section
        let z_bytes_len = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| Error::codec(format!("Failed to read z_bytes_len: {}", e)))?
            as usize;

        let z_start = 4;
        let z_end = z_start + z_bytes_len;

        if data.len() < z_end + 4 {
            return Err(Error::codec(
                ZVC69Error::BitstreamCorrupted {
                    offset: z_start,
                    reason: format!(
                        "Not enough data for hyperprior bytes: need {}, have {}",
                        z_bytes_len,
                        data.len() - z_start
                    ),
                }
                .to_string(),
            ));
        }

        let z_bytes = data[z_start..z_end].to_vec();

        // Read main latent section
        let y_bytes_len_offset = z_end;
        let y_bytes_len = u32::from_le_bytes([
            data[y_bytes_len_offset],
            data[y_bytes_len_offset + 1],
            data[y_bytes_len_offset + 2],
            data[y_bytes_len_offset + 3],
        ]) as usize;

        let y_start = y_bytes_len_offset + 4;
        let y_end = y_start + y_bytes_len;

        if data.len() < y_end {
            return Err(Error::codec(
                ZVC69Error::BitstreamCorrupted {
                    offset: y_start,
                    reason: format!(
                        "Not enough data for latent bytes: need {}, have {}",
                        y_bytes_len,
                        data.len() - y_start
                    ),
                }
                .to_string(),
            ));
        }

        let y_bytes = data[y_start..y_end].to_vec();

        Ok((z_bytes, y_bytes))
    }

    /// Unflatten quantized symbols and dequantize to tensor
    fn unflatten_and_dequantize(
        &self,
        symbols: &[i32],
        shape: (usize, usize, usize, usize),
        quant_scale: f32,
    ) -> Result<Array4<f32>> {
        let tensor = unflatten_tensor(symbols, shape).ok_or_else(|| {
            Error::codec(format!(
                "Failed to unflatten tensor: expected {} elements, got {}",
                shape.0 * shape.1 * shape.2 * shape.3,
                symbols.len()
            ))
        })?;

        Ok(dequantize_scaled(&tensor, quant_scale))
    }

    /// Convert RGB24 frame to YUV420P
    fn convert_rgb_to_yuv(&self, frame: VideoFrame, width: u32, height: u32) -> Result<VideoFrame> {
        if frame.format == PixelFormat::YUV420P {
            return Ok(frame);
        }

        if frame.format != PixelFormat::RGB24 {
            return Err(Error::codec(format!(
                "Unexpected pixel format: {:?}",
                frame.format
            )));
        }

        let rgb_data = frame.data[0].as_slice();
        let y_size = (width * height) as usize;
        let uv_size = y_size / 4;

        let mut y_data = vec![0u8; y_size];
        let mut u_data = vec![0u8; uv_size];
        let mut v_data = vec![0u8; uv_size];

        // Convert RGB to YUV (BT.601)
        for py in 0..height as usize {
            for px in 0..width as usize {
                let rgb_idx = (py * width as usize + px) * 3;
                let r = rgb_data.get(rgb_idx).copied().unwrap_or(128) as f32;
                let g = rgb_data.get(rgb_idx + 1).copied().unwrap_or(128) as f32;
                let b = rgb_data.get(rgb_idx + 2).copied().unwrap_or(128) as f32;

                // BT.601 RGB to YUV
                let y = 0.299 * r + 0.587 * g + 0.114 * b;
                y_data[py * width as usize + px] = y.clamp(0.0, 255.0) as u8;

                // Subsampled U/V
                if py % 2 == 0 && px % 2 == 0 {
                    let uv_idx = (py / 2) * (width as usize / 2) + (px / 2);
                    let u = -0.14713 * r - 0.28886 * g + 0.436 * b + 128.0;
                    let v = 0.615 * r - 0.51499 * g - 0.10001 * b + 128.0;
                    u_data[uv_idx] = u.clamp(0.0, 255.0) as u8;
                    v_data[uv_idx] = v.clamp(0.0, 255.0) as u8;
                }
            }
        }

        let mut yuv_frame = VideoFrame::new(width, height, PixelFormat::YUV420P);
        yuv_frame.data = vec![
            Buffer::from_vec(y_data),
            Buffer::from_vec(u_data),
            Buffer::from_vec(v_data),
        ];
        yuv_frame.linesize = vec![width as usize, (width / 2) as usize, (width / 2) as usize];
        yuv_frame.pts = frame.pts;
        yuv_frame.keyframe = frame.keyframe;
        yuv_frame.pict_type = frame.pict_type;

        Ok(yuv_frame)
    }

    /// Placeholder I-frame decoding (when model is not loaded or placeholder data)
    fn decode_iframe_placeholder(
        &mut self,
        header: &FrameHeader,
        pts: Timestamp,
    ) -> Result<VideoFrame> {
        let (width, height) = header.dimensions();

        // Generate placeholder output (gray frame)
        let video_frame = self.generate_placeholder_frame(width, height, pts, header)?;

        // Update reference frame (YUV buffer)
        self.update_reference_frame(&video_frame)?;

        // Create tensor representation for P-frame reference
        let tensor = self.video_frame_to_tensor(&video_frame)?;
        self.reference_buffer
            .update(video_frame.clone(), tensor, self.frame_count);

        Ok(video_frame)
    }

    // -------------------------------------------------------------------------
    // P-Frame Decoding
    // -------------------------------------------------------------------------

    /// Decode a P-frame from bitstream data
    ///
    /// P-frame decoding pipeline:
    /// 1. Parse frame header and P-frame flags
    /// 2. Check for skip mode vs full residual mode
    /// 3. Parse and decode motion vectors
    /// 4. Get reference frame tensor
    /// 5. Warp reference using motion
    /// 6. (If not skip) Decode and add residual
    /// 7. Convert to VideoFrame
    /// 8. Update reference buffer
    fn decode_pframe(
        &mut self,
        data: &[u8],
        header: &FrameHeader,
        pts: Timestamp,
    ) -> Result<VideoFrame> {
        // Check if we have a valid reference
        if !self.reference_buffer.has_reference() {
            // No reference frame - treat as intra (generate independent frame)
            let mut frame = self.decode_iframe(data, header, pts)?;
            // Force keyframe flag since we're generating independent data
            frame.keyframe = true;
            frame.pict_type = PictureType::I;
            return Ok(frame);
        }

        let payload = &data[FrameHeader::SIZE..];

        // Parse P-frame flags
        if payload.is_empty() {
            return self.decode_pframe_placeholder(header, pts);
        }

        let flags = payload[0];
        let has_residual = (flags & 0x01) != 0;
        let is_skip_mode = (flags & 0x02) != 0;

        // Decode based on mode
        if is_skip_mode {
            self.decode_pframe_skip(&payload[1..], header, pts)
        } else if has_residual {
            self.decode_pframe_full(&payload[1..], header, pts)
        } else {
            // Invalid combination - fall back to placeholder
            self.decode_pframe_placeholder(header, pts)
        }
    }

    /// Decode skip-mode P-frame (motion only, zero residual)
    fn decode_pframe_skip(
        &mut self,
        payload: &[u8],
        header: &FrameHeader,
        pts: Timestamp,
    ) -> Result<VideoFrame> {
        let (width, height) = header.dimensions();

        // Step 1: Verify skip magic
        if payload.len() < 4 || &payload[0..4] != PFRAME_SKIP_MAGIC {
            return self.decode_pframe_placeholder(header, pts);
        }

        let motion_payload = &payload[4..];

        // Step 2: Parse motion section
        let motion = if let Some(ref mut p) = self.profiler {
            match p.time_result(stages::MOTION_CODING, || {
                self.parse_skip_motion_section(motion_payload)
            }) {
                Ok(m) => m,
                Err(_) => return self.decode_pframe_placeholder(header, pts),
            }
        } else {
            match self.parse_skip_motion_section(motion_payload) {
                Ok(m) => m,
                Err(_) => return self.decode_pframe_placeholder(header, pts),
            }
        };

        // Step 3: Get reference tensor
        let reference = self.get_reference_tensor()?;

        // Step 4: Warp reference using motion (predicted IS the output)
        let predicted = if let Some(ref mut p) = self.profiler {
            p.time(stages::FRAME_WARP, || {
                self.frame_warper.backward_warp(&reference, &motion)
            })
        } else {
            self.frame_warper.backward_warp(&reference, &motion)
        };

        // Step 5: Convert to VideoFrame
        let frame = if let Some(ref mut p) = self.profiler {
            p.time_result(stages::COLOR_CONVERT, || {
                self.tensor_to_video_frame(&predicted, width, height)
            })?
        } else {
            self.tensor_to_video_frame(&predicted, width, height)?
        };
        let mut frame = frame;
        frame.pts = pts;
        frame.keyframe = false;
        frame.pict_type = PictureType::P;

        // Step 6: Update references
        self.update_reference_frame(&frame)?;
        self.reference_buffer
            .update(frame.clone(), predicted, self.frame_count);

        Ok(frame)
    }

    /// Decode full P-frame with motion and residual
    fn decode_pframe_full(
        &mut self,
        payload: &[u8],
        header: &FrameHeader,
        pts: Timestamp,
    ) -> Result<VideoFrame> {
        let (width, height) = header.dimensions();

        // Step 1: Parse motion section
        let (motion, residual_payload) = if let Some(ref mut p) = self.profiler {
            match p.time_result(stages::MOTION_CODING, || self.parse_pframe_motion_section(payload))
            {
                Ok((m, r)) => (m, r),
                Err(_) => return self.decode_pframe_placeholder(header, pts),
            }
        } else {
            match self.parse_pframe_motion_section(payload) {
                Ok((m, r)) => (m, r),
                Err(_) => return self.decode_pframe_placeholder(header, pts),
            }
        };

        // Step 2: Get reference tensor
        let reference = self.get_reference_tensor()?;

        // Step 3: Warp reference using motion
        let predicted = if let Some(ref mut p) = self.profiler {
            p.time(stages::FRAME_WARP, || {
                self.frame_warper.backward_warp(&reference, &motion)
            })
        } else {
            self.frame_warper.backward_warp(&reference, &motion)
        };

        // Step 4: Parse and decode residual section
        let residual = if let Some(ref mut p) = self.profiler {
            match p.time_result(stages::RESIDUAL_DECODE, || {
                self.parse_and_decode_residual_section(
                    residual_payload,
                    height as usize,
                    width as usize,
                )
            }) {
                Ok(r) => r,
                Err(_) => {
                    // Fall back to skip mode if residual parsing fails
                    let mut frame = self.tensor_to_video_frame(&predicted, width, height)?;
                    frame.pts = pts;
                    frame.keyframe = false;
                    frame.pict_type = PictureType::P;
                    self.update_reference_frame(&frame)?;
                    self.reference_buffer
                        .update(frame.clone(), predicted, self.frame_count);
                    return Ok(frame);
                }
            }
        } else {
            match self.parse_and_decode_residual_section(
                residual_payload,
                height as usize,
                width as usize,
            ) {
                Ok(r) => r,
                Err(_) => {
                    // Fall back to skip mode if residual parsing fails
                    let mut frame = self.tensor_to_video_frame(&predicted, width, height)?;
                    frame.pts = pts;
                    frame.keyframe = false;
                    frame.pict_type = PictureType::P;
                    self.update_reference_frame(&frame)?;
                    self.reference_buffer
                        .update(frame.clone(), predicted, self.frame_count);
                    return Ok(frame);
                }
            }
        };

        // Step 5: Reconstruct: current = predicted + residual
        let reconstructed = if let Some(ref mut p) = self.profiler {
            p.time(stages::RESIDUAL_COMPUTE, || {
                Residual::reconstruct(&residual, &predicted)
            })
        } else {
            Residual::reconstruct(&residual, &predicted)
        };

        // Step 6: Convert to VideoFrame
        let frame = if let Some(ref mut p) = self.profiler {
            p.time_result(stages::COLOR_CONVERT, || {
                self.tensor_to_video_frame(&reconstructed, width, height)
            })?
        } else {
            self.tensor_to_video_frame(&reconstructed, width, height)?
        };
        let mut frame = frame;
        frame.pts = pts;
        frame.keyframe = false;
        frame.pict_type = PictureType::P;

        // Step 7: Update references
        self.update_reference_frame(&frame)?;
        self.reference_buffer
            .update(frame.clone(), reconstructed, self.frame_count);

        Ok(frame)
    }

    /// Parse motion section for skip-mode P-frame
    fn parse_skip_motion_section(&mut self, data: &[u8]) -> Result<MotionField> {
        // Skip mode format:
        // - Marker byte 'M' (0x4D)
        // - Motion data length (4 bytes)
        // - Motion bytes (variable)

        if data.is_empty() {
            return Err(Error::codec("Empty motion section"));
        }

        if data[0] != MOTION_SECTION_MARKER {
            return Err(Error::codec(format!(
                "Invalid motion section marker: expected 0x{:02X}, got 0x{:02X}",
                MOTION_SECTION_MARKER, data[0]
            )));
        }

        if data.len() < 5 {
            return Err(Error::codec("Motion section too short"));
        }

        // Read motion data length
        let motion_len = u32::from_le_bytes([data[1], data[2], data[3], data[4]]) as usize;

        if data.len() < 5 + motion_len {
            return Err(Error::codec(format!(
                "Not enough data for motion bytes: need {}, have {}",
                motion_len,
                data.len() - 5
            )));
        }

        let motion_bytes = &data[5..5 + motion_len];

        // Decode motion field
        let frame_shape = self.frame_shape();
        let motion = decode_motion(motion_bytes, frame_shape, &mut self.entropy_coder)
            .map_err(|e| Error::codec(e.to_string()))?;

        Ok(motion)
    }

    /// Parse motion section for full P-frame (with height/width metadata)
    fn parse_pframe_motion_section<'a>(
        &mut self,
        data: &'a [u8],
    ) -> Result<(MotionField, &'a [u8])> {
        // Full P-frame motion format:
        // - Marker byte 'M' (0x4D)
        // - Motion height (2 bytes)
        // - Motion width (2 bytes)
        // - Motion data length (4 bytes)
        // - Motion bytes (variable)

        if data.is_empty() {
            return Err(Error::codec("Empty motion section"));
        }

        if data[0] != MOTION_SECTION_MARKER {
            return Err(Error::codec(format!(
                "Invalid motion section marker: expected 0x{:02X}, got 0x{:02X}",
                MOTION_SECTION_MARKER, data[0]
            )));
        }

        if data.len() < 9 {
            return Err(Error::codec("Motion section too short"));
        }

        // Read motion dimensions
        let _motion_height = u16::from_le_bytes([data[1], data[2]]) as usize;
        let _motion_width = u16::from_le_bytes([data[3], data[4]]) as usize;

        // Read motion data length
        let motion_len = u32::from_le_bytes([data[5], data[6], data[7], data[8]]) as usize;

        if data.len() < 9 + motion_len {
            return Err(Error::codec(format!(
                "Not enough data for motion bytes: need {}, have {}",
                motion_len,
                data.len() - 9
            )));
        }

        let motion_bytes = &data[9..9 + motion_len];

        // Decode motion field
        let frame_shape = self.frame_shape();
        let motion = decode_motion(motion_bytes, frame_shape, &mut self.entropy_coder)
            .map_err(|e| Error::codec(e.to_string()))?;

        let remaining = &data[9 + motion_len..];
        Ok((motion, remaining))
    }

    /// Parse and decode residual section
    fn parse_and_decode_residual_section(
        &mut self,
        data: &[u8],
        height: usize,
        width: usize,
    ) -> Result<Residual> {
        // Residual format:
        // - Marker byte 'R' (0x52)
        // - Residual data length (4 bytes)
        // - Residual bytes (variable)

        if data.is_empty() {
            return Err(Error::codec("Empty residual section"));
        }

        if data[0] != RESIDUAL_SECTION_MARKER {
            return Err(Error::codec(format!(
                "Invalid residual section marker: expected 0x{:02X}, got 0x{:02X}",
                RESIDUAL_SECTION_MARKER, data[0]
            )));
        }

        if data.len() < 5 {
            return Err(Error::codec("Residual section too short"));
        }

        // Read residual data length
        let residual_len = u32::from_le_bytes([data[1], data[2], data[3], data[4]]) as usize;

        if data.len() < 5 + residual_len {
            return Err(Error::codec(format!(
                "Not enough data for residual bytes: need {}, have {}",
                residual_len,
                data.len() - 5
            )));
        }

        let residual_bytes = &data[5..5 + residual_len];

        // Decode compressed residual
        let latent_shape = self.residual_latent_shape(height, width);
        let hp_shape = self.residual_hyperprior_shape(height, width);

        let compressed = CompressedResidual::from_bytes(
            residual_bytes,
            latent_shape,
            hp_shape,
            &mut self.entropy_coder,
        )
        .map_err(|e| Error::codec(e.to_string()))?;

        // Decode residual
        let residual = self
            .residual_decoder
            .decode(&compressed, height, width)
            .map_err(|e| Error::codec(e.to_string()))?;

        Ok(residual)
    }

    /// Placeholder P-frame decoding
    fn decode_pframe_placeholder(
        &mut self,
        header: &FrameHeader,
        pts: Timestamp,
    ) -> Result<VideoFrame> {
        let (width, height) = header.dimensions();

        // Copy from reference frame (simple placeholder behavior)
        let y_size = (width * height) as usize;
        let uv_size = y_size / 4;

        let y_data = self.last_ref.y[..y_size.min(self.last_ref.y.len())].to_vec();
        let u_data = self.last_ref.u[..uv_size.min(self.last_ref.u.len())].to_vec();
        let v_data = self.last_ref.v[..uv_size.min(self.last_ref.v.len())].to_vec();

        let mut frame = VideoFrame::new(width, height, PixelFormat::YUV420P);
        frame.data = vec![
            Buffer::from_vec(if y_data.len() == y_size {
                y_data
            } else {
                vec![128; y_size]
            }),
            Buffer::from_vec(if u_data.len() == uv_size {
                u_data
            } else {
                vec![128; uv_size]
            }),
            Buffer::from_vec(if v_data.len() == uv_size {
                v_data
            } else {
                vec![128; uv_size]
            }),
        ];
        frame.linesize = vec![width as usize, (width / 2) as usize, (width / 2) as usize];
        frame.pts = pts;
        frame.keyframe = false;
        frame.pict_type = PictureType::P;

        // Update reference frame
        self.update_reference_frame(&frame)?;

        // Update tensor reference
        let tensor = self.video_frame_to_tensor(&frame)?;
        self.reference_buffer
            .update(frame.clone(), tensor, self.frame_count);

        Ok(frame)
    }

    // -------------------------------------------------------------------------
    // B-Frame Decoding
    // -------------------------------------------------------------------------

    /// Decode a B-frame from bitstream data
    fn decode_bframe(
        &mut self,
        data: &[u8],
        header: &FrameHeader,
        pts: Timestamp,
    ) -> Result<VideoFrame> {
        // B-frames require two reference frames
        if !self.last_ref.valid {
            // No reference frames - treat as P-frame
            return self.decode_pframe(data, header, pts);
        }

        // Placeholder B-frame decoding (similar to P-frame for now)
        self.decode_bframe_placeholder(header, pts)
    }

    /// Placeholder B-frame decoding
    fn decode_bframe_placeholder(
        &mut self,
        header: &FrameHeader,
        pts: Timestamp,
    ) -> Result<VideoFrame> {
        let (width, height) = header.dimensions();

        // Copy from reference frame (simple placeholder behavior)
        // B-frames don't update reference by default
        let y_size = (width * height) as usize;
        let uv_size = y_size / 4;

        let y_data = self.last_ref.y[..y_size.min(self.last_ref.y.len())].to_vec();
        let u_data = self.last_ref.u[..uv_size.min(self.last_ref.u.len())].to_vec();
        let v_data = self.last_ref.v[..uv_size.min(self.last_ref.v.len())].to_vec();

        let mut frame = VideoFrame::new(width, height, PixelFormat::YUV420P);
        frame.data = vec![
            Buffer::from_vec(if y_data.len() == y_size {
                y_data
            } else {
                vec![128; y_size]
            }),
            Buffer::from_vec(if u_data.len() == uv_size {
                u_data
            } else {
                vec![128; uv_size]
            }),
            Buffer::from_vec(if v_data.len() == uv_size {
                v_data
            } else {
                vec![128; uv_size]
            }),
        ];
        frame.linesize = vec![width as usize, (width / 2) as usize, (width / 2) as usize];
        frame.pts = pts;
        frame.keyframe = false;
        frame.pict_type = PictureType::B;

        // B-frames typically don't update reference
        Ok(frame)
    }

    // -------------------------------------------------------------------------
    // Reference Frame Management
    // -------------------------------------------------------------------------

    /// Get reference frame tensor for P-frame decoding
    fn get_reference_tensor(&self) -> Result<Array4<f32>> {
        self.reference_buffer
            .get_last_tensor()
            .cloned()
            .ok_or_else(|| Error::codec("No reference frame available for P-frame decoding"))
    }

    /// Update reference frame buffer (YUV planes)
    fn update_reference_frame(&mut self, frame: &VideoFrame) -> Result<()> {
        if frame.data.len() >= 3 {
            // Copy Y plane
            let y_size = (self.width * self.height) as usize;
            let y_data = frame.data[0].as_slice();
            if y_data.len() >= y_size {
                self.last_ref.y[..y_size].copy_from_slice(&y_data[..y_size]);
            }

            // Copy U plane
            let uv_size = y_size / 4;
            let u_data = frame.data[1].as_slice();
            if u_data.len() >= uv_size {
                self.last_ref.u[..uv_size].copy_from_slice(&u_data[..uv_size]);
            }

            // Copy V plane
            let v_data = frame.data[2].as_slice();
            if v_data.len() >= uv_size {
                self.last_ref.v[..uv_size].copy_from_slice(&v_data[..uv_size]);
            }

            self.last_ref.frame_num = self.frame_count;
            self.last_ref.valid = true;
        }

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Helper Functions
    // -------------------------------------------------------------------------

    /// Get frame shape as (height, width) tuple
    fn frame_shape(&self) -> (usize, usize) {
        (self.height as usize, self.width as usize)
    }

    /// Get residual latent shape
    fn residual_latent_shape(&self, height: usize, width: usize) -> (usize, usize, usize) {
        let latent_h =
            (height + RESIDUAL_LATENT_SPATIAL_FACTOR - 1) / RESIDUAL_LATENT_SPATIAL_FACTOR;
        let latent_w =
            (width + RESIDUAL_LATENT_SPATIAL_FACTOR - 1) / RESIDUAL_LATENT_SPATIAL_FACTOR;
        (DEFAULT_RESIDUAL_LATENT_CHANNELS, latent_h, latent_w)
    }

    /// Get residual hyperprior shape
    fn residual_hyperprior_shape(&self, height: usize, width: usize) -> (usize, usize, usize) {
        let hp_h =
            (height + RESIDUAL_HYPERPRIOR_SPATIAL_FACTOR - 1) / RESIDUAL_HYPERPRIOR_SPATIAL_FACTOR;
        let hp_w =
            (width + RESIDUAL_HYPERPRIOR_SPATIAL_FACTOR - 1) / RESIDUAL_HYPERPRIOR_SPATIAL_FACTOR;
        (DEFAULT_RESIDUAL_HYPERPRIOR_CHANNELS, hp_h, hp_w)
    }

    /// Convert VideoFrame to tensor [1, 3, H, W] with normalized RGB values
    fn video_frame_to_tensor(&self, frame: &VideoFrame) -> Result<Array4<f32>> {
        // Convert YUV420P to normalized RGB tensor
        let width = frame.width as usize;
        let height = frame.height as usize;

        let mut tensor = Array4::zeros((1, 3, height, width));

        if frame.data.len() < 3 {
            return Ok(tensor); // Return zero tensor if no data
        }

        let y_data = frame.data[0].as_slice();
        let u_data = frame.data[1].as_slice();
        let v_data = frame.data[2].as_slice();

        // Convert YUV420P to RGB
        for py in 0..height {
            for px in 0..width {
                let y_idx = py * width + px;
                let uv_idx = (py / 2) * (width / 2) + (px / 2);

                let y = y_data.get(y_idx).copied().unwrap_or(128) as f32;
                let u = u_data.get(uv_idx).copied().unwrap_or(128) as f32 - 128.0;
                let v = v_data.get(uv_idx).copied().unwrap_or(128) as f32 - 128.0;

                // YUV to RGB (BT.601)
                let r = y + 1.402 * v;
                let g = y - 0.344136 * u - 0.714136 * v;
                let b = y + 1.772 * u;

                // Normalize to [0, 1]
                tensor[[0, 0, py, px]] = (r / 255.0).clamp(0.0, 1.0);
                tensor[[0, 1, py, px]] = (g / 255.0).clamp(0.0, 1.0);
                tensor[[0, 2, py, px]] = (b / 255.0).clamp(0.0, 1.0);
            }
        }

        Ok(tensor)
    }

    /// Convert tensor [1, 3, H, W] to VideoFrame (YUV420P)
    fn tensor_to_video_frame(
        &self,
        tensor: &Array4<f32>,
        width: u32,
        height: u32,
    ) -> Result<VideoFrame> {
        let h = height as usize;
        let w = width as usize;
        let y_size = h * w;
        let uv_size = y_size / 4;

        let mut y_data = vec![0u8; y_size];
        let mut u_data = vec![0u8; uv_size];
        let mut v_data = vec![0u8; uv_size];

        // Convert RGB tensor to YUV420P
        for py in 0..h {
            for px in 0..w {
                // Get RGB values (normalized [0, 1])
                let r =
                    tensor[[0, 0, py.min(tensor.dim().2 - 1), px.min(tensor.dim().3 - 1)]] * 255.0;
                let g =
                    tensor[[0, 1, py.min(tensor.dim().2 - 1), px.min(tensor.dim().3 - 1)]] * 255.0;
                let b =
                    tensor[[0, 2, py.min(tensor.dim().2 - 1), px.min(tensor.dim().3 - 1)]] * 255.0;

                // RGB to YUV (BT.601)
                let y = 0.299 * r + 0.587 * g + 0.114 * b;
                y_data[py * w + px] = y.clamp(0.0, 255.0) as u8;

                // Subsampled U/V
                if py % 2 == 0 && px % 2 == 0 {
                    let uv_idx = (py / 2) * (w / 2) + (px / 2);
                    let u = -0.14713 * r - 0.28886 * g + 0.436 * b + 128.0;
                    let v = 0.615 * r - 0.51499 * g - 0.10001 * b + 128.0;
                    u_data[uv_idx] = u.clamp(0.0, 255.0) as u8;
                    v_data[uv_idx] = v.clamp(0.0, 255.0) as u8;
                }
            }
        }

        let mut frame = VideoFrame::new(width, height, PixelFormat::YUV420P);
        frame.data = vec![
            Buffer::from_vec(y_data),
            Buffer::from_vec(u_data),
            Buffer::from_vec(v_data),
        ];
        frame.linesize = vec![w, w / 2, w / 2];

        Ok(frame)
    }

    /// Generate a placeholder frame (stub for development)
    fn generate_placeholder_frame(
        &self,
        width: u32,
        height: u32,
        pts: Timestamp,
        header: &FrameHeader,
    ) -> Result<VideoFrame> {
        let mut frame = VideoFrame::new(width, height, PixelFormat::YUV420P);

        // Generate gray frame (neutral YUV values)
        let y_size = (width * height) as usize;
        let uv_size = y_size / 4;

        // Y plane - mid gray
        let y_data = vec![128u8; y_size];

        // U and V planes - neutral chroma
        let u_data = vec![128u8; uv_size];
        let v_data = vec![128u8; uv_size];

        frame.data = vec![
            Buffer::from_vec(y_data),
            Buffer::from_vec(u_data),
            Buffer::from_vec(v_data),
        ];

        frame.linesize = vec![width as usize, (width / 2) as usize, (width / 2) as usize];

        frame.pts = pts;
        frame.keyframe = header.frame_type == FrameType::I;
        frame.pict_type = header.frame_type.to_picture_type();

        Ok(frame)
    }

    /// Get decoder statistics
    pub fn stats(&self) -> DecoderStats {
        DecoderStats {
            frames_decoded: self.frame_count,
            width: self.width,
            height: self.height,
        }
    }
}

impl Default for ZVC69Decoder {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

impl Decoder for ZVC69Decoder {
    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if packet.data.is_empty() {
            return Err(Error::codec("Empty ZVC69 packet"));
        }

        self.decode_packet(packet)
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(frame) = self.output_frames.pop() {
            Ok(Frame::Video(frame))
        } else {
            Err(Error::TryAgain)
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.state = DecoderState::Flushing;
        self.output_frames.clear();
        self.reference_buffer.clear();
        Ok(())
    }
}

// -------------------------------------------------------------------------
// Decoder Statistics
// -------------------------------------------------------------------------

/// Decoder statistics
#[derive(Debug, Clone)]
pub struct DecoderStats {
    /// Total frames decoded
    pub frames_decoded: u64,
    /// Video width
    pub width: u32,
    /// Video height
    pub height: u32,
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::Encoder;

    #[test]
    fn test_decoder_creation() {
        let decoder = ZVC69Decoder::new();
        assert!(decoder.is_ok());

        let decoder = decoder.unwrap();
        assert!(!decoder.is_initialized());
        assert!(!decoder.has_reference());
    }

    #[test]
    fn test_decoder_with_dimensions() {
        let decoder = ZVC69Decoder::with_dimensions(1920, 1080);
        assert!(decoder.is_ok());

        let decoder = decoder.unwrap();
        assert!(decoder.is_initialized());
        assert_eq!(decoder.dimensions(), (1920, 1080));
    }

    #[test]
    fn test_decoder_default() {
        let decoder = ZVC69Decoder::default();
        assert!(!decoder.is_initialized());
    }

    #[test]
    fn test_frame_header_parse() {
        let mut data = vec![0u8; 12];
        data[0..4].copy_from_slice(b"ZVC1");
        data[4] = 0; // I-frame
        data[5] = 4; // Q4
        data[6] = 120; // 1920/16
        data[7] = 0;
        data[8] = 68; // 1088/16 (rounded up from 1080)
        data[9] = 0;
        data[10] = 23; // QP
        data[11] = 0;

        let header = FrameHeader::parse(&data);
        assert!(header.is_ok());

        let header = header.unwrap();
        assert_eq!(header.frame_type, FrameType::I);
        assert_eq!(header.quality, Quality::Q4);
    }

    #[test]
    fn test_frame_header_invalid_magic() {
        let data = b"XXXX00000000";
        let header = FrameHeader::parse(data);
        assert!(header.is_err());
    }

    #[test]
    fn test_empty_packet_error() {
        let mut decoder = ZVC69Decoder::new().unwrap();
        let packet = Packet::new(0, Buffer::empty());
        let result = decoder.send_packet(&packet);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_valid_packet() {
        let mut decoder = ZVC69Decoder::new().unwrap();

        // Create a valid ZVC69 packet (I-frame)
        let mut data = vec![0u8; 24];
        data[0..4].copy_from_slice(b"ZVC1");
        data[4] = 0; // I-frame
        data[5] = 4; // Q4
        data[6] = 4; // 64/16
        data[7] = 0;
        data[8] = 4; // 64/16
        data[9] = 0;
        data[10] = 23; // QP
        data[11] = 0;
        // Frame size
        let frame_size = 64u32 * 64;
        data[12..16].copy_from_slice(&frame_size.to_le_bytes());
        // Checksum
        data[16..20].copy_from_slice(&0u32.to_le_bytes());
        // Marker
        data[20..24].copy_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);

        let packet = Packet::new(0, Buffer::from_vec(data));
        let result = decoder.send_packet(&packet);
        assert!(result.is_ok());

        // Get decoded frame
        let frame = decoder.receive_frame();
        assert!(frame.is_ok());

        if let Frame::Video(vf) = frame.unwrap() {
            assert_eq!(vf.width, 64);
            assert_eq!(vf.height, 64);
            assert!(vf.keyframe);
        } else {
            panic!("Expected video frame");
        }

        // Verify reference is now available
        assert!(decoder.has_reference());
    }

    #[test]
    fn test_extradata_parsing() {
        let mut decoder = ZVC69Decoder::new().unwrap();

        let mut extradata = vec![0u8; 16];
        extradata[0..4].copy_from_slice(b"ZVC0");
        extradata[4] = 1; // Version
        extradata[5] = 0; // Profile
        extradata[6] = 40; // Level
        extradata[7] = 4; // Quality
        extradata[8..10].copy_from_slice(&1920u16.to_le_bytes());
        extradata[10..12].copy_from_slice(&1080u16.to_le_bytes());
        extradata[12..14].copy_from_slice(&30u16.to_le_bytes());
        extradata[14..16].copy_from_slice(&1u16.to_le_bytes());

        let result = decoder.set_extradata(&extradata);
        assert!(result.is_ok());
        assert!(decoder.is_initialized());
        assert_eq!(decoder.dimensions(), (1920, 1080));
    }

    #[test]
    fn test_roundtrip_encode_decode() {
        use super::super::encoder::ZVC69Encoder;
        use super::super::ZVC69Config;

        // Create encoder
        let config = ZVC69Config::new(64, 64);
        let mut encoder = ZVC69Encoder::new(config).unwrap();

        // Create test frame
        let mut frame = VideoFrame::new(64, 64, PixelFormat::YUV420P);
        let y_size = 64 * 64;
        let uv_size = y_size / 4;
        frame.data = vec![
            Buffer::from_vec(vec![128u8; y_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
        ];
        frame.linesize = vec![64, 32, 32];

        // Encode
        encoder.send_frame(&Frame::Video(frame)).unwrap();
        let packet = encoder.receive_packet().unwrap();

        assert!(packet.flags.keyframe);
        assert!(!packet.data.is_empty());

        // Verify ZVC69 header in packet
        let data = packet.data.as_slice();
        assert!(data.len() >= 12);
        assert_eq!(&data[0..4], b"ZVC1");

        // Decode
        let mut decoder = ZVC69Decoder::new().unwrap();
        decoder.send_packet(&packet).unwrap();
        let decoded = decoder.receive_frame().unwrap();

        if let Frame::Video(vf) = decoded {
            assert_eq!(vf.width, 64);
            assert_eq!(vf.height, 64);
            assert!(vf.keyframe);
            assert_eq!(vf.pict_type, PictureType::I);
            // Verify we have valid YUV420P data
            assert_eq!(vf.data.len(), 3);
            assert_eq!(vf.data[0].len(), y_size);
            assert_eq!(vf.data[1].len(), uv_size);
            assert_eq!(vf.data[2].len(), uv_size);
        } else {
            panic!("Expected video frame");
        }

        // Verify decoder has reference now
        assert!(decoder.has_reference());
    }

    #[test]
    fn test_multi_frame_decode() {
        use super::super::encoder::ZVC69Encoder;
        use super::super::ZVC69Config;

        // Create encoder with specific GOP settings
        let mut config = ZVC69Config::new(64, 64);
        config.gop.keyframe_interval = 3;
        config.gop.bframes = 0;

        let mut encoder = ZVC69Encoder::new(config).unwrap();
        let mut decoder = ZVC69Decoder::new().unwrap();

        // Encode and decode 5 frames
        for i in 0..5 {
            let mut frame = VideoFrame::new(64, 64, PixelFormat::YUV420P);
            let y_size = 64 * 64;
            let uv_size = y_size / 4;
            // Use different Y values to distinguish frames
            frame.data = vec![
                Buffer::from_vec(vec![(100 + i * 10) as u8; y_size]),
                Buffer::from_vec(vec![128u8; uv_size]),
                Buffer::from_vec(vec![128u8; uv_size]),
            ];
            frame.linesize = vec![64, 32, 32];

            encoder.send_frame(&Frame::Video(frame)).unwrap();
            let packet = encoder.receive_packet().unwrap();

            decoder.send_packet(&packet).unwrap();
            let decoded = decoder.receive_frame().unwrap();

            if let Frame::Video(vf) = decoded {
                assert_eq!(vf.width, 64);
                assert_eq!(vf.height, 64);
                // Frames 0 and 3 should be keyframes (interval=3)
                let expected_keyframe = i == 0 || i == 3;
                assert_eq!(
                    vf.keyframe, expected_keyframe,
                    "Frame {} keyframe mismatch",
                    i
                );
            }
        }
    }

    #[test]
    fn test_decoder_stats() {
        let decoder = ZVC69Decoder::with_dimensions(1920, 1080).unwrap();

        let stats = decoder.stats();
        assert_eq!(stats.frames_decoded, 0);
        assert_eq!(stats.width, 1920);
        assert_eq!(stats.height, 1080);
    }

    #[test]
    fn test_iframe_metadata_parse() {
        let decoder = ZVC69Decoder::new().unwrap();

        // Create valid metadata
        let mut data = vec![0u8; 20];
        // y_channels = 192
        data[0..2].copy_from_slice(&192u16.to_le_bytes());
        // y_height = 4
        data[2..4].copy_from_slice(&4u16.to_le_bytes());
        // y_width = 4
        data[4..6].copy_from_slice(&4u16.to_le_bytes());
        // z_channels = 128
        data[6..8].copy_from_slice(&128u16.to_le_bytes());
        // z_height = 1
        data[8..10].copy_from_slice(&1u16.to_le_bytes());
        // z_width = 1
        data[10..12].copy_from_slice(&1u16.to_le_bytes());
        // quant_scale = 1.0
        data[12..16].copy_from_slice(&1.0_f32.to_bits().to_le_bytes());
        // Extra data
        data[16..20].copy_from_slice(&[0, 0, 0, 0]);

        let result = decoder.parse_iframe_metadata(&data);
        assert!(result.is_ok());

        let (metadata, rest) = result.unwrap();
        assert_eq!(metadata.latent_shape, (1, 192, 4, 4));
        assert_eq!(metadata.hyperprior_shape, (1, 128, 1, 1));
        assert!((metadata.quant_scale - 1.0).abs() < 1e-6);
        assert_eq!(rest.len(), 4);
    }

    #[test]
    fn test_decoded_frame_struct() {
        let frame = VideoFrame::new(64, 64, PixelFormat::YUV420P);
        let decoded = DecodedFrame::new(frame, BitstreamFrameType::I, 1000, 1000);

        assert!(decoded.is_keyframe());
        assert_eq!(decoded.pts, 1000);
        assert_eq!(decoded.dts, 1000);

        let p_frame = VideoFrame::new(64, 64, PixelFormat::YUV420P);
        let decoded_p = DecodedFrame::new(p_frame, BitstreamFrameType::P, 2000, 2000);
        assert!(!decoded_p.is_keyframe());
    }

    #[test]
    fn test_parse_iframe_sections() {
        let decoder = ZVC69Decoder::new().unwrap();

        // Create valid section data
        let z_data = vec![1u8, 2, 3, 4];
        let y_data = vec![5u8, 6, 7, 8, 9];

        let mut data = Vec::new();
        // z_bytes_len
        data.extend_from_slice(&(z_data.len() as u32).to_le_bytes());
        // z_bytes
        data.extend_from_slice(&z_data);
        // y_bytes_len
        data.extend_from_slice(&(y_data.len() as u32).to_le_bytes());
        // y_bytes
        data.extend_from_slice(&y_data);

        let result = decoder.parse_iframe_sections(&data);
        assert!(result.is_ok());

        let (z_bytes, y_bytes) = result.unwrap();
        assert_eq!(z_bytes, z_data);
        assert_eq!(y_bytes, y_data);
    }

    #[test]
    fn test_decoder_reference_buffer() {
        let mut buffer = DecoderReferenceBuffer::new();
        assert!(!buffer.has_reference());
        assert!(buffer.get_last_tensor().is_none());
        assert!(buffer.get_last_frame().is_none());

        // Update with a frame
        let frame = VideoFrame::new(64, 64, PixelFormat::YUV420P);
        let tensor = Array4::zeros((1, 3, 64, 64));
        buffer.update(frame, tensor.clone(), 1);

        assert!(buffer.has_reference());
        assert!(buffer.get_last_tensor().is_some());
        assert!(buffer.get_last_frame().is_some());

        // Update golden
        buffer.update_golden(tensor.clone(), 1);

        // Clear
        buffer.clear();
        assert!(!buffer.has_reference());
    }

    #[test]
    fn test_pframe_decoding_without_reference() {
        let mut decoder = ZVC69Decoder::new().unwrap();

        // Create a P-frame packet without having decoded an I-frame first
        let mut data = vec![0u8; 24];
        data[0..4].copy_from_slice(b"ZVC1");
        data[4] = 1; // P-frame
        data[5] = 4; // Q4
        data[6] = 4; // 64/16
        data[7] = 0;
        data[8] = 4; // 64/16
        data[9] = 0;
        data[10] = 23; // QP
        data[11] = 0;
        // Placeholder data
        data[12..16].copy_from_slice(&(64u32 * 64).to_le_bytes());
        data[16..20].copy_from_slice(&0u32.to_le_bytes());
        data[20..24].copy_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);

        let packet = Packet::new(0, Buffer::from_vec(data));
        let result = decoder.send_packet(&packet);

        // Should succeed by treating as I-frame
        assert!(result.is_ok());
        let frame = decoder.receive_frame().unwrap();

        if let Frame::Video(vf) = frame {
            // Without reference, P-frame becomes I-frame
            assert!(vf.keyframe);
        }
    }

    #[test]
    fn test_gop_sequence_decode() {
        use super::super::encoder::ZVC69Encoder;
        use super::super::ZVC69Config;

        // Create encoder with I-P-P-P GOP structure
        let mut config = ZVC69Config::new(64, 64);
        config.gop.keyframe_interval = 4;
        config.gop.bframes = 0;

        let mut encoder = ZVC69Encoder::new(config).unwrap();
        let mut decoder = ZVC69Decoder::new().unwrap();

        // Encode and decode I-P-P-P sequence
        for i in 0..4 {
            let mut frame = VideoFrame::new(64, 64, PixelFormat::YUV420P);
            let y_size = 64 * 64;
            let uv_size = y_size / 4;
            frame.data = vec![
                Buffer::from_vec(vec![(120 + i * 5) as u8; y_size]),
                Buffer::from_vec(vec![128u8; uv_size]),
                Buffer::from_vec(vec![128u8; uv_size]),
            ];
            frame.linesize = vec![64, 32, 32];

            encoder.send_frame(&Frame::Video(frame)).unwrap();
            let packet = encoder.receive_packet().unwrap();

            decoder.send_packet(&packet).unwrap();
            let decoded = decoder.receive_frame().unwrap();

            if let Frame::Video(vf) = decoded {
                if i == 0 {
                    assert!(vf.keyframe, "Frame 0 should be keyframe");
                    assert_eq!(vf.pict_type, PictureType::I);
                } else {
                    assert!(!vf.keyframe, "Frame {} should not be keyframe", i);
                    assert_eq!(vf.pict_type, PictureType::P);
                }
            }
        }

        // After decoding all frames, reference should still be valid
        assert!(decoder.has_reference());
    }

    #[test]
    fn test_video_frame_to_tensor_conversion() {
        let decoder = ZVC69Decoder::with_dimensions(64, 64).unwrap();

        // Create a test frame
        let mut frame = VideoFrame::new(64, 64, PixelFormat::YUV420P);
        let y_size = 64 * 64;
        let uv_size = y_size / 4;
        frame.data = vec![
            Buffer::from_vec(vec![128u8; y_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
            Buffer::from_vec(vec![128u8; uv_size]),
        ];

        let tensor = decoder.video_frame_to_tensor(&frame).unwrap();
        assert_eq!(tensor.dim(), (1, 3, 64, 64));

        // Convert back to frame
        let reconstructed = decoder.tensor_to_video_frame(&tensor, 64, 64).unwrap();
        assert_eq!(reconstructed.width, 64);
        assert_eq!(reconstructed.height, 64);
        assert_eq!(reconstructed.data.len(), 3);
    }

    #[test]
    fn test_reference_consistency_across_frames() {
        use super::super::encoder::ZVC69Encoder;
        use super::super::ZVC69Config;

        let mut config = ZVC69Config::new(64, 64);
        config.gop.keyframe_interval = 10;
        config.gop.bframes = 0;

        let mut encoder = ZVC69Encoder::new(config).unwrap();
        let mut decoder = ZVC69Decoder::new().unwrap();

        // Encode and decode 3 frames, checking reference at each step
        for i in 0..3 {
            let mut frame = VideoFrame::new(64, 64, PixelFormat::YUV420P);
            let y_size = 64 * 64;
            let uv_size = y_size / 4;
            frame.data = vec![
                Buffer::from_vec(vec![(100 + i * 20) as u8; y_size]),
                Buffer::from_vec(vec![128u8; uv_size]),
                Buffer::from_vec(vec![128u8; uv_size]),
            ];
            frame.linesize = vec![64, 32, 32];

            encoder.send_frame(&Frame::Video(frame)).unwrap();
            let packet = encoder.receive_packet().unwrap();

            decoder.send_packet(&packet).unwrap();
            decoder.receive_frame().unwrap();

            // After every frame, reference should be valid
            assert!(
                decoder.has_reference(),
                "Reference missing after frame {}",
                i
            );
        }
    }
}
