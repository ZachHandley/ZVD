//! MP3 audio decoder using Symphonia

use crate::codec::{Decoder, Frame};
use crate::error::{Error, Result};
use crate::format::Packet;

/// MP3 audio decoder using Symphonia
///
/// Note: This is a placeholder that provides the interface.
/// Actual decoding is handled through Symphonia's format demuxer.
/// For direct MP3 packet decoding, use SymphoniaAdapter.
pub struct Mp3Decoder {
    sample_rate: u32,
    channels: u16,
}

impl Mp3Decoder {
    /// Create a new MP3 decoder
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        Ok(Mp3Decoder {
            sample_rate,
            channels,
        })
    }
}

impl Decoder for Mp3Decoder {
    fn send_packet(&mut self, _packet: &Packet) -> Result<()> {
        // MP3 decoding through Symphonia is done at the container level
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        Err(Error::unsupported(
            "MP3 decoding should use SymphoniaAdapter for container-level decoding",
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
    fn test_mp3_decoder_creation() {
        let decoder = Mp3Decoder::new(44100, 2);
        assert!(decoder.is_ok());
    }
}
