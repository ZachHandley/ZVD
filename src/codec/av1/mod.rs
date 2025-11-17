//! AV1 codec support
//!
//! This module provides AV1 video encoding and decoding.
//! - Decoding: dav1d (C library bindings)
//! - Encoding: rav1e (pure Rust)
//!
//! # Features
//!
//! - **High-quality encoding**: rav1e encoder with speed presets 0-10
//! - **Fast decoding**: dav1d decoder optimized for performance
//! - **Pixel format support**: YUV420P, YUV422P, YUV444P, GRAY8, 10-bit variants
//! - **Threading**: Multi-threaded encoding and decoding
//! - **Rate control**: Constant quality (CQ) and bitrate modes
//! - **Low latency**: Configurable for real-time applications
//!
//! # Example: Encoding
//!
//! ```no_run
//! use zvd_lib::codec::av1::Av1EncoderBuilder;
//!
//! let encoder = Av1EncoderBuilder::new(1920, 1080)
//!     .speed_preset(6)        // Balanced speed/quality
//!     .quantizer(100)         // Medium quality
//!     .threads(8)             // 8 encoding threads
//!     .max_keyframe_interval(240)  // Keyframe every 240 frames
//!     .build()
//!     .unwrap();
//! ```
//!
//! # Example: Decoding
//!
//! ```no_run
//! use zvd_lib::codec::av1::Av1Decoder;
//!
//! let decoder = Av1Decoder::new().unwrap();
//! // Or with specific thread count:
//! let decoder = Av1Decoder::with_threads(8).unwrap();
//! ```
//!
//! # Supported Pixel Formats
//!
//! ## Encoding (rav1e)
//! - YUV420P (8-bit 4:2:0)
//! - YUV422P (8-bit 4:2:2)
//! - YUV444P (8-bit 4:4:4)
//! - GRAY8 (8-bit grayscale)
//!
//! ## Decoding (dav1d)
//! - YUV420P, YUV420P10LE (8/10-bit 4:2:0)
//! - YUV422P, YUV422P10LE (8/10-bit 4:2:2)
//! - YUV444P, YUV444P10LE (8/10-bit 4:4:4)
//! - GRAY8, GRAY16 (8/16-bit grayscale)
//!
//! # Threading Support
//!
//! Both encoder and decoder support multi-threading:
//! - **Encoder**: Uses worker threads and tile-based parallelism
//! - **Decoder**: Uses frame-level parallelism with configurable threads
//!
//! Set `threads(0)` for auto-detection based on CPU core count.

pub mod decoder;
pub mod encoder;

pub use decoder::Av1Decoder;
pub use encoder::{Av1Encoder, Av1EncoderBuilder, RateControlMode};

/// Create a new AV1 decoder with default settings
///
/// This is a convenience function that creates an AV1 decoder with auto-detected
/// thread count. For more control, use `Av1Decoder::with_threads()`.
pub fn create_decoder() -> crate::error::Result<Av1Decoder> {
    Av1Decoder::new()
}

/// Create a new AV1 encoder with specified dimensions
///
/// This is a convenience function that creates an AV1 encoder with default settings.
/// For more control over encoding parameters, use `Av1EncoderBuilder`.
///
/// # Arguments
///
/// * `width` - Video width in pixels (must be multiple of 8)
/// * `height` - Video height in pixels (must be multiple of 8)
pub fn create_encoder(width: u32, height: u32) -> crate::error::Result<Av1Encoder> {
    Av1Encoder::new(width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factory_functions() {
        // Test decoder factory
        let decoder = create_decoder();
        assert!(decoder.is_ok(), "Decoder factory should work");

        // Test encoder factory
        let encoder = create_encoder(640, 480);
        assert!(encoder.is_ok(), "Encoder factory should work");
    }
}
