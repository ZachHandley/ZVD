//! Decoder implementations

use super::Frame;
use crate::error::{Error, Result};
use crate::format::Packet;

/// Decoder trait for decoding compressed data
pub trait Decoder {
    /// Send a packet to the decoder
    fn send_packet(&mut self, packet: &Packet) -> Result<()>;

    /// Receive a decoded frame
    fn receive_frame(&mut self) -> Result<Frame>;

    /// Flush the decoder
    fn flush(&mut self) -> Result<()>;
}

/// Decoder context
pub struct DecoderContext {
    codec_id: String,
    extradata: Option<Vec<u8>>,
}

impl DecoderContext {
    /// Create a new decoder context
    pub fn new(codec_id: String) -> Self {
        DecoderContext {
            codec_id,
            extradata: None,
        }
    }

    /// Set extradata (codec-specific configuration)
    pub fn set_extradata(&mut self, data: Vec<u8>) {
        self.extradata = Some(data);
    }

    /// Get the codec ID
    pub fn codec_id(&self) -> &str {
        &self.codec_id
    }

    /// Get extradata
    pub fn extradata(&self) -> Option<&[u8]> {
        self.extradata.as_deref()
    }
}

/// Create a decoder for the given codec
pub fn create_decoder(codec_id: &str) -> Result<Box<dyn Decoder>> {
    match codec_id {
        "av1" => {
            use crate::codec::Av1Decoder;
            Ok(Box::new(Av1Decoder::new()?))
        }
        #[cfg(feature = "h264")]
        "h264" => {
            use crate::codec::H264Decoder;
            Ok(Box::new(H264Decoder::new()?))
        }
        #[cfg(feature = "aac")]
        "aac" => {
            use crate::codec::AacDecoder;
            // Default to stereo 44.1kHz, should be configured based on stream info
            Ok(Box::new(AacDecoder::new(44100, 2)?))
        }
        _ => Err(Error::unsupported(format!(
            "No decoder available for codec: {}",
            codec_id
        ))),
    }
}
