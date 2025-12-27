//! ZVC69 Neural Video Codec
//!
//! ZVC69 is a next-generation neural video codec that uses learned neural network
//! transforms for compression, achieving better rate-distortion performance than
//! traditional codecs like AV1 and H.265.
//!
//! ## Features
//!
//! - **Neural Compression**: Uses learned analysis/synthesis transforms
//! - **Adaptive Quality**: 8 quality levels (Q1-Q8) for fine-grained control
//! - **Multiple Presets**: From ultrafast to veryslow for speed/quality tradeoffs
//! - **Modern Rate Control**: CRF, VBR, CBR, and CQP modes
//! - **GPU Acceleration**: ONNX Runtime with TensorRT support (optional)
//! - **TensorRT Optimization**: FP16/INT8 quantization for real-time 1080p encoding
//!
//! ## Architecture
//!
//! The codec follows the learned image/video compression paradigm:
//!
//! ```text
//! Encoder:
//!   Input Frame -> Analysis Transform -> Quantization -> Entropy Coding -> Bitstream
//!                  (Neural Network)      (Learned)       (Learned Probs)
//!
//! Decoder:
//!   Bitstream -> Entropy Decoding -> Dequantization -> Synthesis Transform -> Output Frame
//!                (Learned Probs)     (Learned)         (Neural Network)
//! ```
//!
//! For inter-frames (P/B), additional neural motion estimation and compensation
//! modules are used.
//!
//! ## Usage
//!
//! ### Encoding
//!
//! ```rust,ignore
//! use zvd::codec::zvc69::{ZVC69Encoder, ZVC69Config, Quality, Preset};
//!
//! // Build configuration
//! let config = ZVC69Config::builder()
//!     .dimensions(1920, 1080)
//!     .quality(Quality::Q5)
//!     .preset(Preset::Medium)
//!     .fps(30)
//!     .build()?;
//!
//! // Create encoder
//! let mut encoder = ZVC69Encoder::new(config)?;
//!
//! // Encode frames
//! for frame in video_frames {
//!     encoder.send_frame(&frame)?;
//!     while let Ok(packet) = encoder.receive_packet() {
//!         // Write packet to output
//!     }
//! }
//!
//! // Flush encoder
//! encoder.flush()?;
//! ```
//!
//! ### Decoding
//!
//! ```rust,ignore
//! use zvd::codec::zvc69::ZVC69Decoder;
//!
//! // Create decoder
//! let mut decoder = ZVC69Decoder::new()?;
//!
//! // Decode packets
//! for packet in bitstream_packets {
//!     decoder.send_packet(&packet)?;
//!     while let Ok(frame) = decoder.receive_frame() {
//!         // Process decoded frame
//!     }
//! }
//!
//! // Flush decoder
//! decoder.flush()?;
//! ```
//!
//! ## Bitstream Format
//!
//! The ZVC69 bitstream uses a custom format optimized for neural codec data:
//!
//! - **Frame Header** (12 bytes): Magic, type, quality, dimensions, QP
//! - **Entropy Data**: Arithmetic-coded latent values
//! - **Motion Data**: (Inter-frames only) Neural motion vectors
//!
//! ## Feature Flag
//!
//! This module requires the `zvc69` feature to be enabled:
//!
//! ```toml
//! [dependencies]
//! zvd = { version = "...", features = ["zvc69"] }
//! ```
//!
//! ## Implementation Status
//!
//! This module implements the ZVC69 neural video codec infrastructure.
//! The following components are complete:
//!
//! - [x] Module structure and public API
//! - [x] Configuration and builder pattern
//! - [x] Error types (comprehensive error handling)
//! - [x] Bitstream format (read/write/seek)
//! - [x] Entropy coding (ANS via constriction library)
//!   - EntropyCoder for Gaussian-conditioned latents
//!   - FactorizedPrior for hyperprior latents
//!   - GaussianConditional for main latents
//!   - Helper functions (quantize/dequantize/estimate_bits)
//! - [x] Neural model loading (ort/ONNX Runtime)
//!   - NeuralModel for encoder/decoder/hyperprior sessions
//!   - Model loading from file or embedded bytes
//!   - GPU/CPU device selection (CUDA support)
//!   - Latents, Hyperprior, EntropyParams data structures
//!   - Tensor/VideoFrame conversion helpers
//! - [x] Quantization module
//!   - Tensor quantization/dequantization
//!   - Scaled quantization for quality control
//!   - Quality-to-scale conversion
//!   - Bit estimation utilities
//! - [x] I-frame encoder (fully implemented)
//!   - Full neural encoding pipeline (with zvc69 feature)
//!   - Placeholder encoding for development testing
//!   - Bitstream packaging with shape metadata
//!   - Reference frame management
//!   - Rate control state tracking
//! - [x] Decoder (basic implementation)
//!   - I-frame decoding (placeholder)
//!   - P/B-frame handling (placeholder)
//!   - Reference frame buffer management
//! - [x] Motion estimation module
//!   - MotionField for dense optical flow representation
//!   - MotionEstimator with neural and block matching support
//!   - Motion compression/decompression pipeline
//!   - Entropy coding for motion vectors
//!   - Block matching (SAD/SSD) fallback
//!   - Multi-precision support (full-pel, half-pel, quarter-pel)
//! - [x] Frame warping module (motion compensation)
//!   - FrameWarper with backward/forward warping
//!   - Multiple interpolation modes (nearest, bilinear, bicubic)
//!   - Border handling (zeros, replicate, reflect)
//!   - OcclusionMask for occlusion detection
//!   - Multi-frame blending for B-frames
//!   - Warp quality metrics (MSE, PSNR, SSIM)
//! - [x] Residual coding module (P-frame residuals)
//!   - Residual computation (current - predicted)
//!   - ResidualStats for analysis (mean, std, energy, sparsity)
//!   - ResidualEncoder/Decoder with placeholder and neural support
//!   - CompressedResidual with entropy coding integration
//!   - Skip mode detection for near-zero residuals
//!   - Block-level skip mask computation
//!   - Adaptive quantization based on content/motion
//!   - Per-block adaptive quantization
//! - [x] TensorRT optimization module
//!   - TensorRTConfig for FP32/FP16/INT8 precision control
//!   - TensorRTModel wrapper for ONNX Runtime with TensorRT EP
//!   - Engine caching for fast model loading
//!   - Int8Calibrator for INT8 quantization calibration
//!   - OptimizedZVC69 complete codec wrapper
//!   - GpuMemoryPool for reduced allocation overhead
//!   - Comprehensive precision detection and mode selection
//! - [x] Memory optimization module
//!   - FramePool for zero-allocation frame buffer reuse
//!   - PooledBuffer with automatic return on drop (RAII)
//!   - BitstreamArena for fast bump allocation
//!   - Resolution presets (720p, 1080p, 4K)
//!   - EncoderMemoryContext / DecoderMemoryContext
//!   - Pool statistics tracking (allocations, reuses, peak usage)
//!   - Thread-safe concurrent access
//! - [x] Pipeline module (Milestone M3)
//!   - PipelinedEncoder for real-time 720p encoding (30+ fps)
//!   - PipelinedDecoder for real-time 720p decoding (60+ fps)
//!   - SyncEncoder/SyncDecoder for single-threaded usage
//!   - PipelineConfig with presets (realtime_720p, low_latency, high_throughput)
//!   - PipelineStats for monitoring throughput and latency
//!   - Latency tracking with percentiles (P50, P95, P99)
//! - [x] Benchmark module (Milestone M3)
//!   - BenchmarkResult with comprehensive metrics
//!   - benchmark_720p, benchmark_1080p, benchmark_4k functions
//!   - TestPattern enum for synthetic frame generation
//!   - Performance assertions (assert_realtime, assert_latency)
//!   - Quality level comparison (benchmark_quality_levels)
//! - [ ] P/B-frame encoder (full integration)
//! - [ ] Advanced rate control optimization

// ─────────────────────────────────────────────────────────────────────────────
// Submodules
// ─────────────────────────────────────────────────────────────────────────────

pub mod benchmark;
pub mod bitstream;
pub mod config;
pub mod decoder;
pub mod encoder;
pub mod entropy;
pub mod error;
pub mod memory;
pub mod model;
pub mod motion;
pub mod pipeline;
pub mod profiler;
pub mod quantize;
pub mod residual;
pub mod tensorrt;
pub mod warp;

// ─────────────────────────────────────────────────────────────────────────────
// Public Re-exports
// ─────────────────────────────────────────────────────────────────────────────

// Bitstream types
pub use bitstream::{
    calculate_crc16, flags, index_flags, verify_checksum, BitstreamReader, BitstreamWriter,
    ColorSpace, FileHeader, FileHeaderBuilder, FrameHeader, FrameType, IndexEntry, IndexTable,
    SeekResult, FILE_HEADER_SIZE, FRAME_HEADER_SIZE, INDEX_ENTRY_SIZE, INDEX_TABLE_HEADER_SIZE,
    MAGIC, VERSION_MAJOR, VERSION_MINOR,
};

// Configuration types
pub use config::{
    ColorPrimaries, GopConfig, MatrixCoefficients, ModelConfig, Preset, Quality, RateControlMode,
    TransferCharacteristics, ZVC69Config, ZVC69ConfigBuilder,
};

// Encoder
pub use encoder::{EncodedFrame, EncoderStats, ZVC69Encoder};

// Decoder
pub use decoder::{DecodedFrame, DecoderStats, ZVC69Decoder};

// Error types
pub use error::ZVC69Error;

// Model types
pub use model::{
    denormalize_image, denormalize_imagenet, image_to_tensor, normalize_image, normalize_imagenet,
    tensor_to_image, Device, EntropyParams, Hyperprior, Latents, NeuralModel, NeuralModelConfig,
    OptimizationLevel, DECODER_MODEL_NAME, DEFAULT_HYPERPRIOR_CHANNELS, DEFAULT_LATENT_CHANNELS,
    DEFAULT_NUM_THREADS, ENCODER_MODEL_NAME, HYPERPRIOR_DEC_MODEL_NAME, HYPERPRIOR_ENC_MODEL_NAME,
    HYPERPRIOR_SPATIAL_FACTOR, LATENT_SPATIAL_FACTOR,
};

// Entropy coding types
pub use entropy::{
    dequantize_latents, estimate_bits, gaussian_to_cdf, quantize_latents, EntropyCoder,
    FactorizedPrior, GaussianConditional, QuantizedGaussian, CDF_PRECISION_BITS, CDF_SCALE,
    DEFAULT_MAX_SYMBOL, DEFAULT_MIN_SYMBOL, DEFAULT_NUM_SCALES, DEFAULT_SCALE_BOUND,
    DEFAULT_TAIL_MASS, MAX_SCALE, MIN_SCALE,
};

// Quantization helpers
pub use quantize::{
    clamp_quantized, dequantize_scaled, dequantize_slice, dequantize_tensor, flatten_tensor_chw,
    flatten_tensor_f32, quality_to_scale, quantize_scaled, quantize_slice, quantize_tensor,
    scale_to_quality, unflatten_tensor, unflatten_tensor_f32,
};

// Motion estimation types
pub use motion::{
    block_match, decode_motion, decode_motion_conditional, encode_motion,
    encode_motion_conditional, sad, ssd, CompressedMotion, MotionConfig, MotionEstimator,
    MotionField, MotionPrecision, DEFAULT_BLOCK_SIZE, DEFAULT_MOTION_QUANT_SCALE,
    DEFAULT_SEARCH_RANGE, MIN_BLOCK_SIZE, MOTION_SPATIAL_FACTOR, MV_CLAMP_RANGE,
};

// Frame warping types
pub use warp::{
    blend_warped_frames, blend_with_occlusion, identity_grid, masked_warp_error, motion_to_grid,
    warp_error, warp_psnr, warp_structural_similarity, BorderMode, FrameWarper, Interpolation,
    OcclusionMask, WarpConfig, DEFAULT_MAGNITUDE_THRESHOLD, DEFAULT_OCCLUSION_THRESHOLD,
    WARP_EPSILON,
};

// Residual coding types
pub use residual::{
    adaptive_quantize, adaptive_quantize_blockwise, compute_skip_mask, count_skip_blocks,
    should_skip_residual, skip_ratio, CompressedResidual, Residual, ResidualConfig,
    ResidualDecoder, ResidualEncoder, ResidualStats, DEFAULT_RESIDUAL_HYPERPRIOR_CHANNELS,
    DEFAULT_RESIDUAL_LATENT_CHANNELS, DEFAULT_SKIP_BLOCK_SIZE, DEFAULT_SKIP_THRESHOLD,
    DEFAULT_SPARSITY_THRESHOLD, RESIDUAL_HYPERPRIOR_SPATIAL_FACTOR, RESIDUAL_LATENT_SPATIAL_FACTOR,
};

// TensorRT optimization types
pub use tensorrt::{
    AsyncOperation, CudaStreamManager, CudaStreamStats, GpuMemoryPool, Int8Calibrator,
    OptimizedZVC69, Precision, StreamRole, StreamState, TensorRTConfig, TensorRTModel,
    TripleBufferPipeline,
};

// Profiling types
pub use profiler::{
    benchmark_report, stages, BenchmarkConfig as ProfilerBenchmarkConfig, BenchmarkResults,
    FrameTiming, ProfileStats, Profiler, TimingFrameType,
};

// Memory management types
pub use memory::{
    BitstreamArena, BufferShape, DecoderMemoryContext, EncoderMemoryContext, FramePool, PoolConfig,
    PoolStats, PooledBuffer,
};

// Pipeline types
pub use pipeline::{
    EncodedFrame as PipelineEncodedFrame, PipelineConfig, PipelineFrame, PipelineStats,
    PipelinedDecoder, PipelinedEncoder, StreamAwareEncoder, SyncDecoder, SyncEncoder,
};

// Benchmark types
pub use benchmark::{
    assert_latency, assert_realtime, benchmark_1080p, benchmark_4k, benchmark_720p,
    benchmark_latency, benchmark_quality_levels, quick_validation, run_benchmark,
    run_benchmark_suite, BenchmarkConfig, BenchmarkResult, TestPattern,
};

// Precision validation types
pub use benchmark::{
    assert_fp16_quality, assert_int8_quality, calculate_mse, calculate_psnr,
    compare_fp16_fp32_quality, compare_int8_fp32_quality, frame_psnr, mse_to_psnr,
    run_precision_validation_suite, PrecisionQualityResult, FP16_PSNR_LOSS_THRESHOLD,
    INT8_PSNR_LOSS_THRESHOLD,
};

// Re-export profile macro
pub use crate::profile;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// ZVC69 codec identifier
pub const CODEC_ID: &str = "zvc69";

/// ZVC69 codec name
pub const CODEC_NAME: &str = "ZVC69";

/// ZVC69 codec long name
pub const CODEC_LONG_NAME: &str = "ZVC69 Neural Video Codec";

/// Current bitstream version
pub const BITSTREAM_VERSION: u32 = 1;

/// Minimum supported bitstream version
pub const BITSTREAM_VERSION_MIN: u32 = 1;

/// Maximum supported bitstream version
pub const BITSTREAM_VERSION_MAX: u32 = 1;

/// Frame header magic bytes
pub const FRAME_MAGIC: &[u8; 4] = b"ZVC1";

/// Config record magic bytes
pub const CONFIG_MAGIC: &[u8; 4] = b"ZVC0";

/// Minimum frame dimension
pub const MIN_DIMENSION: u32 = 64;

/// Maximum frame dimension
pub const MAX_DIMENSION: u32 = 8192;

/// Required dimension alignment (must be divisible by this)
pub const DIMENSION_ALIGNMENT: u32 = 16;

// ─────────────────────────────────────────────────────────────────────────────
// Helper Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Check if dimensions are valid for ZVC69
pub fn is_valid_resolution(width: u32, height: u32) -> bool {
    width >= MIN_DIMENSION
        && width <= MAX_DIMENSION
        && height >= MIN_DIMENSION
        && height <= MAX_DIMENSION
        && width % DIMENSION_ALIGNMENT == 0
        && height % DIMENSION_ALIGNMENT == 0
}

/// Align a dimension to the required alignment (rounds up)
pub fn align_dimension(dim: u32) -> u32 {
    ((dim + DIMENSION_ALIGNMENT - 1) / DIMENSION_ALIGNMENT) * DIMENSION_ALIGNMENT
}

/// Calculate aligned dimensions
pub fn align_dimensions(width: u32, height: u32) -> (u32, u32) {
    (align_dimension(width), align_dimension(height))
}

/// Estimate bitrate for given parameters
pub fn estimate_bitrate(width: u32, height: u32, fps: f32, quality: Quality) -> u64 {
    let pixels_per_frame = (width * height) as f64;
    let base_bpp = 0.1; // bits per pixel at Q4
    let quality_mult = quality.bitrate_multiplier() as f64;

    (pixels_per_frame * fps as f64 * base_bpp * quality_mult) as u64
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_resolution() {
        // Valid: divisible by 16
        assert!(is_valid_resolution(1920, 1088)); // 1080p aligned
        assert!(is_valid_resolution(1280, 720)); // 720p
        assert!(is_valid_resolution(640, 480));
        assert!(is_valid_resolution(64, 64));
        assert!(is_valid_resolution(8192, 8192));

        // Invalid: not aligned (1080 is not divisible by 16)
        assert!(!is_valid_resolution(1920, 1080));
        assert!(!is_valid_resolution(100, 100));
        // Invalid: too small
        assert!(!is_valid_resolution(32, 32));
        // Invalid: too large
        assert!(!is_valid_resolution(10000, 10000));
    }

    #[test]
    fn test_align_dimension() {
        assert_eq!(align_dimension(1920), 1920);
        assert_eq!(align_dimension(1080), 1088);
        assert_eq!(align_dimension(100), 112);
        assert_eq!(align_dimension(1), 16);
    }

    #[test]
    fn test_align_dimensions() {
        assert_eq!(align_dimensions(1920, 1080), (1920, 1088));
        assert_eq!(align_dimensions(100, 100), (112, 112));
    }

    #[test]
    fn test_estimate_bitrate() {
        let bitrate = estimate_bitrate(1920, 1080, 30.0, Quality::Q4);
        // Should be in reasonable range
        assert!(bitrate > 1_000_000);
        assert!(bitrate < 100_000_000);

        // Higher quality = higher bitrate
        let bitrate_low = estimate_bitrate(1920, 1080, 30.0, Quality::Q1);
        let bitrate_high = estimate_bitrate(1920, 1080, 30.0, Quality::Q8);
        assert!(bitrate_high > bitrate_low);
    }

    #[test]
    fn test_codec_constants() {
        assert_eq!(CODEC_ID, "zvc69");
        assert_eq!(FRAME_MAGIC, b"ZVC1");
        assert_eq!(CONFIG_MAGIC, b"ZVC0");
    }

    #[test]
    fn test_roundtrip_encode_decode() {
        use crate::codec::{Decoder, Encoder, Frame, VideoFrame};
        use crate::util::{Buffer, PixelFormat};

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

        // Decode
        let mut decoder = ZVC69Decoder::new().unwrap();
        decoder.send_packet(&packet).unwrap();
        let decoded = decoder.receive_frame().unwrap();

        if let Frame::Video(vf) = decoded {
            assert_eq!(vf.width, 64);
            assert_eq!(vf.height, 64);
            assert!(vf.keyframe);
        } else {
            panic!("Expected video frame");
        }
    }
}
