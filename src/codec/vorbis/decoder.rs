//! Vorbis audio decoder using Symphonia

use crate::codec::{Decoder, Frame};
use crate::error::{Error, Result};
use crate::format::Packet;

/// Vorbis audio decoder using Symphonia
///
/// Note: This is a placeholder that provides the interface.
/// Actual decoding is handled through Symphonia's format demuxer.
/// For direct Vorbis packet decoding, use SymphoniaAdapter.
pub struct VorbisDecoder {
    sample_rate: u32,
    channels: u16,
}

impl VorbisDecoder {
    /// Create a new Vorbis decoder
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        Ok(VorbisDecoder {
            sample_rate,
            channels,
        })
    }
}

impl Decoder for VorbisDecoder {
    fn send_packet(&mut self, _packet: &Packet) -> Result<()> {
        // Vorbis decoding through Symphonia is done at the container level
        // This is a placeholder for the decoder interface
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        Err(Error::unsupported(
            "Vorbis decoding should use SymphoniaAdapter for container-level decoding",
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
    fn test_vorbis_decoder_creation() {
        let decoder = VorbisDecoder::new(44100, 2);
        assert!(decoder.is_ok());
    }
}
