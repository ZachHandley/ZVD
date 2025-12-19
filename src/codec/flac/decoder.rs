//! FLAC audio decoder using Symphonia

use crate::codec::{Decoder, Frame};
use crate::error::{Error, Result};
use crate::format::Packet;

/// FLAC audio decoder using Symphonia
///
/// Note: This is a placeholder that provides the interface.
/// Actual decoding is handled through Symphonia's format demuxer.
/// For direct FLAC decoding, use Symphonia Adapter.
pub struct FlacDecoder {
    sample_rate: u32,
    channels: u16,
}

impl FlacDecoder {
    /// Create a new FLAC decoder
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        Ok(FlacDecoder {
            sample_rate,
            channels,
        })
    }
}

impl Decoder for FlacDecoder {
    fn send_packet(&mut self, _packet: &Packet) -> Result<()> {
        // FLAC decoding through Symphonia is done at the container level
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        Err(Error::unsupported(
            "FLAC decoding should use SymphoniaAdapter for container-level decoding",
        ))
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flac_decoder_creation() {
        let decoder = FlacDecoder::new(44100, 2);
        assert!(decoder.is_ok());
    }
}
