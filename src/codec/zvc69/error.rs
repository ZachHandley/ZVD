//! ZVC69 Neural Video Codec error types
//!
//! This module provides structured error types for the ZVC69 neural video codec,
//! enabling precise error handling and actionable diagnostics.

use thiserror::Error;

/// ZVC69-specific error types for detailed diagnostics
///
/// These errors provide actionable error messages for neural codec issues,
/// helping users understand what went wrong and how to fix it.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ZVC69Error {
    // ─────────────────────────────────────────────────────────────────────────
    // Feature / Initialization Errors
    // ─────────────────────────────────────────────────────────────────────────
    /// The 'zvc69' cargo feature is not enabled
    #[error(
        "ZVC69 codec requires the 'zvc69' feature. Enable it in Cargo.toml: features = [\"zvc69\"]"
    )]
    FeatureNotEnabled,

    /// Encoder or decoder not initialized properly
    #[error("ZVC69 {component} not initialized: {reason}")]
    NotInitialized {
        /// Which component failed (encoder/decoder)
        component: String,
        /// Reason for initialization failure
        reason: String,
    },

    // ─────────────────────────────────────────────────────────────────────────
    // Model / Neural Network Errors
    // ─────────────────────────────────────────────────────────────────────────
    /// Failed to load the neural network model
    #[error("Failed to load ZVC69 model '{model_name}': {reason}")]
    ModelLoadFailed {
        /// Name or path of the model
        model_name: String,
        /// Reason for failure
        reason: String,
    },

    /// Model file not found
    #[error("ZVC69 model file not found: {path}. Ensure model files are installed.")]
    ModelNotFound {
        /// Path where model was expected
        path: String,
    },

    /// Model version mismatch
    #[error(
        "ZVC69 model version mismatch: expected {expected}, got {actual}. \
        Update the model files or use a compatible codec version."
    )]
    ModelVersionMismatch {
        /// Expected version
        expected: String,
        /// Actual version found
        actual: String,
    },

    /// Model inference failed
    #[error("ZVC69 model inference failed: {reason}")]
    InferenceFailed {
        /// Reason for failure
        reason: String,
    },

    /// Invalid model architecture
    #[error("Invalid ZVC69 model architecture: {details}")]
    InvalidModelArchitecture {
        /// Details about the architecture issue
        details: String,
    },

    // ─────────────────────────────────────────────────────────────────────────
    // Entropy Coding Errors
    // ─────────────────────────────────────────────────────────────────────────
    /// Entropy encoder initialization failed
    #[error("ZVC69 entropy encoder initialization failed: {reason}")]
    EntropyEncoderInit {
        /// Reason for failure
        reason: String,
    },

    /// Entropy decoder initialization failed
    #[error("ZVC69 entropy decoder initialization failed: {reason}")]
    EntropyDecoderInit {
        /// Reason for failure
        reason: String,
    },

    /// Entropy coding error during encoding
    #[error("ZVC69 entropy encoding error: {reason}")]
    EntropyEncodingFailed {
        /// Reason for failure
        reason: String,
    },

    /// Entropy coding error during decoding
    #[error("ZVC69 entropy decoding error: {reason}")]
    EntropyDecodingFailed {
        /// Reason for failure
        reason: String,
    },

    /// Invalid probability distribution
    #[error("Invalid probability distribution in entropy coder: {details}")]
    InvalidProbability {
        /// Details about the invalid distribution
        details: String,
    },

    // ─────────────────────────────────────────────────────────────────────────
    // Bitstream Errors
    // ─────────────────────────────────────────────────────────────────────────
    /// Invalid bitstream header
    #[error("Invalid ZVC69 bitstream header: {reason}")]
    InvalidHeader {
        /// Reason for invalidity
        reason: String,
    },

    /// Bitstream corrupted or truncated
    #[error(
        "ZVC69 bitstream corrupted at offset {offset}: {reason}. \
        The file may be damaged or incomplete."
    )]
    BitstreamCorrupted {
        /// Byte offset where corruption was detected
        offset: usize,
        /// Reason for corruption detection
        reason: String,
    },

    /// Unsupported bitstream version
    #[error(
        "Unsupported ZVC69 bitstream version {version}. \
        This codec supports versions {min_supported}-{max_supported}."
    )]
    UnsupportedBitstreamVersion {
        /// Version found in bitstream
        version: u32,
        /// Minimum supported version
        min_supported: u32,
        /// Maximum supported version
        max_supported: u32,
    },

    /// Bitstream write error
    #[error("ZVC69 bitstream write error: {reason}")]
    BitstreamWriteFailed {
        /// Reason for failure
        reason: String,
    },

    /// Bitstream read error
    #[error("ZVC69 bitstream read error: {reason}")]
    BitstreamReadFailed {
        /// Reason for failure
        reason: String,
    },

    // ─────────────────────────────────────────────────────────────────────────
    // Quantization Errors
    // ─────────────────────────────────────────────────────────────────────────
    /// Invalid quantization parameter
    #[error(
        "Invalid ZVC69 quantization parameter: {param} = {value}. \
        Valid range is {min}-{max}."
    )]
    InvalidQuantParam {
        /// Parameter name
        param: String,
        /// Invalid value provided
        value: i32,
        /// Minimum valid value
        min: i32,
        /// Maximum valid value
        max: i32,
    },

    /// Quantization table error
    #[error("ZVC69 quantization table error: {reason}")]
    QuantTableError {
        /// Reason for error
        reason: String,
    },

    // ─────────────────────────────────────────────────────────────────────────
    // Motion / Temporal Errors
    // ─────────────────────────────────────────────────────────────────────────
    /// Motion estimation failed
    #[error("ZVC69 motion estimation failed for frame {frame_num}: {reason}")]
    MotionEstimationFailed {
        /// Frame number where failure occurred
        frame_num: u64,
        /// Reason for failure
        reason: String,
    },

    /// Motion compensation failed
    #[error("ZVC69 motion compensation failed: {reason}")]
    MotionCompensationFailed {
        /// Reason for failure
        reason: String,
    },

    /// Reference frame not available
    #[error(
        "ZVC69 reference frame {ref_idx} not available. \
        This may indicate a GOP structure error or missing keyframe."
    )]
    ReferenceFrameUnavailable {
        /// Reference frame index
        ref_idx: usize,
    },

    // ─────────────────────────────────────────────────────────────────────────
    // Residual / Transform Errors
    // ─────────────────────────────────────────────────────────────────────────
    /// Transform encoding failed
    #[error("ZVC69 transform encoding failed: {reason}")]
    TransformEncodingFailed {
        /// Reason for failure
        reason: String,
    },

    /// Transform decoding failed
    #[error("ZVC69 transform decoding failed: {reason}")]
    TransformDecodingFailed {
        /// Reason for failure
        reason: String,
    },

    /// Residual reconstruction error
    #[error("ZVC69 residual reconstruction error: {reason}")]
    ResidualError {
        /// Reason for error
        reason: String,
    },

    // ─────────────────────────────────────────────────────────────────────────
    // Configuration Errors
    // ─────────────────────────────────────────────────────────────────────────
    /// Invalid configuration
    #[error("Invalid ZVC69 configuration: {reason}")]
    InvalidConfig {
        /// Reason for invalidity
        reason: String,
    },

    /// Unsupported resolution
    #[error(
        "Unsupported resolution {width}x{height}. \
        ZVC69 supports resolutions from {min_dim}x{min_dim} to {max_dim}x{max_dim}, \
        with dimensions divisible by {alignment}."
    )]
    UnsupportedResolution {
        /// Requested width
        width: u32,
        /// Requested height
        height: u32,
        /// Minimum dimension
        min_dim: u32,
        /// Maximum dimension
        max_dim: u32,
        /// Required alignment
        alignment: u32,
    },

    /// Unsupported pixel format
    #[error("Unsupported pixel format: {format}. ZVC69 supports: YUV420P, YUV444P, RGB24")]
    UnsupportedPixelFormat {
        /// The unsupported format
        format: String,
    },

    /// Unsupported color space
    #[error("Unsupported color space: {color_space}")]
    UnsupportedColorSpace {
        /// The unsupported color space
        color_space: String,
    },

    // ─────────────────────────────────────────────────────────────────────────
    // Runtime / Resource Errors
    // ─────────────────────────────────────────────────────────────────────────
    /// GPU/CUDA not available
    #[error(
        "ZVC69 requires GPU acceleration but no compatible GPU found. \
        Ensure CUDA drivers are installed and a compatible NVIDIA GPU is available."
    )]
    GpuNotAvailable,

    /// Out of GPU memory
    #[error(
        "ZVC69 ran out of GPU memory. Required: {required_mb} MB, Available: {available_mb} MB. \
        Try reducing resolution, batch size, or closing other GPU applications."
    )]
    GpuOutOfMemory {
        /// Required memory in MB
        required_mb: u64,
        /// Available memory in MB
        available_mb: u64,
    },

    /// Thread pool error
    #[error("ZVC69 thread pool error: {reason}")]
    ThreadPoolError {
        /// Reason for error
        reason: String,
    },

    /// Memory allocation failed
    #[error("ZVC69 memory allocation failed: requested {size} bytes")]
    AllocationFailed {
        /// Size of allocation that failed
        size: usize,
    },

    // ─────────────────────────────────────────────────────────────────────────
    // Frame Processing Errors
    // ─────────────────────────────────────────────────────────────────────────
    /// Frame encoding failed
    #[error("ZVC69 failed to encode frame {frame_num}: {reason}")]
    FrameEncodingFailed {
        /// Frame number
        frame_num: u64,
        /// Reason for failure
        reason: String,
    },

    /// Frame decoding failed
    #[error("ZVC69 failed to decode frame {frame_num}: {reason}")]
    FrameDecodingFailed {
        /// Frame number
        frame_num: u64,
        /// Reason for failure
        reason: String,
    },

    /// Empty frame data
    #[error("ZVC69 received empty frame data")]
    EmptyFrame,

    /// Frame dimension mismatch
    #[error(
        "Frame dimensions {actual_w}x{actual_h} don't match encoder configuration {expected_w}x{expected_h}"
    )]
    DimensionMismatch {
        /// Actual width
        actual_w: u32,
        /// Actual height
        actual_h: u32,
        /// Expected width
        expected_w: u32,
        /// Expected height
        expected_h: u32,
    },

    // ─────────────────────────────────────────────────────────────────────────
    // Rate Control Errors
    // ─────────────────────────────────────────────────────────────────────────
    /// Rate control error
    #[error("ZVC69 rate control error: {reason}")]
    RateControlError {
        /// Reason for error
        reason: String,
    },

    /// Invalid bitrate
    #[error(
        "Invalid target bitrate {bitrate} bps. \
        Valid range for {width}x{height} at {fps} fps is {min_bps}-{max_bps} bps."
    )]
    InvalidBitrate {
        /// Requested bitrate
        bitrate: u64,
        /// Frame width
        width: u32,
        /// Frame height
        height: u32,
        /// Frames per second
        fps: f32,
        /// Minimum recommended bitrate
        min_bps: u64,
        /// Maximum recommended bitrate
        max_bps: u64,
    },

    // ─────────────────────────────────────────────────────────────────────────
    // Generic Errors
    // ─────────────────────────────────────────────────────────────────────────
    /// Internal codec error (catch-all)
    #[error("ZVC69 internal error: {0}")]
    Internal(String),

    /// Operation not supported
    #[error("ZVC69 operation not supported: {0}")]
    NotSupported(String),

    /// IO error
    #[error("ZVC69 IO error: {0}")]
    Io(String),
}

impl ZVC69Error {
    // ─────────────────────────────────────────────────────────────────────────
    // Convenience constructors
    // ─────────────────────────────────────────────────────────────────────────

    /// Create a model load error
    pub fn model_load_failed(model_name: impl Into<String>, reason: impl Into<String>) -> Self {
        ZVC69Error::ModelLoadFailed {
            model_name: model_name.into(),
            reason: reason.into(),
        }
    }

    /// Create a model not found error
    pub fn model_not_found(path: impl Into<String>) -> Self {
        ZVC69Error::ModelNotFound { path: path.into() }
    }

    /// Create an inference failed error
    pub fn inference_failed(reason: impl Into<String>) -> Self {
        ZVC69Error::InferenceFailed {
            reason: reason.into(),
        }
    }

    /// Create an invalid config error
    pub fn invalid_config(reason: impl Into<String>) -> Self {
        ZVC69Error::InvalidConfig {
            reason: reason.into(),
        }
    }

    /// Create a bitstream corrupted error
    pub fn bitstream_corrupted(offset: usize, reason: impl Into<String>) -> Self {
        ZVC69Error::BitstreamCorrupted {
            offset,
            reason: reason.into(),
        }
    }

    /// Create an invalid header error
    pub fn invalid_header(reason: impl Into<String>) -> Self {
        ZVC69Error::InvalidHeader {
            reason: reason.into(),
        }
    }

    /// Create an unsupported resolution error
    pub fn unsupported_resolution(width: u32, height: u32) -> Self {
        ZVC69Error::UnsupportedResolution {
            width,
            height,
            min_dim: 64,
            max_dim: 8192,
            alignment: 16,
        }
    }

    /// Create a frame encoding failed error
    pub fn frame_encoding_failed(frame_num: u64, reason: impl Into<String>) -> Self {
        ZVC69Error::FrameEncodingFailed {
            frame_num,
            reason: reason.into(),
        }
    }

    /// Create a frame decoding failed error
    pub fn frame_decoding_failed(frame_num: u64, reason: impl Into<String>) -> Self {
        ZVC69Error::FrameDecodingFailed {
            frame_num,
            reason: reason.into(),
        }
    }

    /// Create a dimension mismatch error
    pub fn dimension_mismatch(
        actual_w: u32,
        actual_h: u32,
        expected_w: u32,
        expected_h: u32,
    ) -> Self {
        ZVC69Error::DimensionMismatch {
            actual_w,
            actual_h,
            expected_w,
            expected_h,
        }
    }

    /// Create an internal error
    pub fn internal(msg: impl Into<String>) -> Self {
        ZVC69Error::Internal(msg.into())
    }

    /// Create an IO error
    pub fn io(msg: impl Into<String>) -> Self {
        ZVC69Error::Io(msg.into())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Error category checks
    // ─────────────────────────────────────────────────────────────────────────

    /// Check if this is a model-related error
    pub fn is_model_error(&self) -> bool {
        matches!(
            self,
            ZVC69Error::ModelLoadFailed { .. }
                | ZVC69Error::ModelNotFound { .. }
                | ZVC69Error::ModelVersionMismatch { .. }
                | ZVC69Error::InferenceFailed { .. }
                | ZVC69Error::InvalidModelArchitecture { .. }
        )
    }

    /// Check if this is an entropy coding error
    pub fn is_entropy_error(&self) -> bool {
        matches!(
            self,
            ZVC69Error::EntropyEncoderInit { .. }
                | ZVC69Error::EntropyDecoderInit { .. }
                | ZVC69Error::EntropyEncodingFailed { .. }
                | ZVC69Error::EntropyDecodingFailed { .. }
                | ZVC69Error::InvalidProbability { .. }
        )
    }

    /// Check if this is a bitstream error
    pub fn is_bitstream_error(&self) -> bool {
        matches!(
            self,
            ZVC69Error::InvalidHeader { .. }
                | ZVC69Error::BitstreamCorrupted { .. }
                | ZVC69Error::UnsupportedBitstreamVersion { .. }
                | ZVC69Error::BitstreamWriteFailed { .. }
                | ZVC69Error::BitstreamReadFailed { .. }
        )
    }

    /// Check if this is a GPU/resource error
    pub fn is_resource_error(&self) -> bool {
        matches!(
            self,
            ZVC69Error::GpuNotAvailable
                | ZVC69Error::GpuOutOfMemory { .. }
                | ZVC69Error::ThreadPoolError { .. }
                | ZVC69Error::AllocationFailed { .. }
        )
    }

    /// Check if this is a configuration error
    pub fn is_config_error(&self) -> bool {
        matches!(
            self,
            ZVC69Error::InvalidConfig { .. }
                | ZVC69Error::UnsupportedResolution { .. }
                | ZVC69Error::UnsupportedPixelFormat { .. }
                | ZVC69Error::UnsupportedColorSpace { .. }
                | ZVC69Error::InvalidQuantParam { .. }
                | ZVC69Error::InvalidBitrate { .. }
        )
    }

    /// Check if this error is recoverable (can continue encoding/decoding)
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            ZVC69Error::ReferenceFrameUnavailable { .. }
                | ZVC69Error::RateControlError { .. }
                | ZVC69Error::MotionEstimationFailed { .. }
        )
    }
}

impl From<std::io::Error> for ZVC69Error {
    fn from(err: std::io::Error) -> Self {
        ZVC69Error::Io(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ZVC69Error::model_not_found("/path/to/model.onnx");
        assert!(err.to_string().contains("/path/to/model.onnx"));

        let err = ZVC69Error::unsupported_resolution(100, 100);
        assert!(err.to_string().contains("100x100"));
    }

    #[test]
    fn test_error_categories() {
        let model_err = ZVC69Error::ModelNotFound {
            path: "test".to_string(),
        };
        assert!(model_err.is_model_error());
        assert!(!model_err.is_entropy_error());

        let entropy_err = ZVC69Error::EntropyEncodingFailed {
            reason: "test".to_string(),
        };
        assert!(entropy_err.is_entropy_error());
        assert!(!entropy_err.is_model_error());
    }

    #[test]
    fn test_convenience_constructors() {
        let err = ZVC69Error::internal("something went wrong");
        assert!(matches!(err, ZVC69Error::Internal(_)));

        let err = ZVC69Error::dimension_mismatch(640, 480, 1920, 1080);
        assert!(matches!(err, ZVC69Error::DimensionMismatch { .. }));
    }
}
