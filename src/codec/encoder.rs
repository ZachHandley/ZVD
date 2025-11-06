//! Encoder implementations

use super::Frame;
use crate::error::{Error, Result};
use crate::format::Packet;

/// Encoder trait for encoding frames
pub trait Encoder {
    /// Send a frame to the encoder
    fn send_frame(&mut self, frame: &Frame) -> Result<()>;

    /// Receive an encoded packet
    fn receive_packet(&mut self) -> Result<Packet>;

    /// Flush the encoder
    fn flush(&mut self) -> Result<()>;
}

/// Encoder context with configuration
pub struct EncoderContext {
    codec_id: String,
    bitrate: Option<u64>,
    quality: Option<f32>,
}

impl EncoderContext {
    /// Create a new encoder context
    pub fn new(codec_id: String) -> Self {
        EncoderContext {
            codec_id,
            bitrate: None,
            quality: None,
        }
    }

    /// Set target bitrate
    pub fn set_bitrate(&mut self, bitrate: u64) {
        self.bitrate = Some(bitrate);
    }

    /// Set quality (0.0 - 1.0, higher is better)
    pub fn set_quality(&mut self, quality: f32) {
        self.quality = Some(quality);
    }

    /// Get the codec ID
    pub fn codec_id(&self) -> &str {
        &self.codec_id
    }

    /// Get bitrate
    pub fn bitrate(&self) -> Option<u64> {
        self.bitrate
    }

    /// Get quality
    pub fn quality(&self) -> Option<f32> {
        self.quality
    }
}

/// Create an encoder for the given codec with dimensions
pub fn create_encoder(codec_id: &str, width: u32, height: u32) -> Result<Box<dyn Encoder>> {
    match codec_id {
        "av1" => {
            use crate::codec::Av1Encoder;
            Ok(Box::new(Av1Encoder::new(width, height)?))
        }
        #[cfg(feature = "h264")]
        "h264" => {
            use crate::codec::H264Encoder;
            Ok(Box::new(H264Encoder::new(width, height)?))
        }
        _ => Err(Error::unsupported(format!(
            "No encoder available for codec: {}",
            codec_id
        ))),
    }
}
