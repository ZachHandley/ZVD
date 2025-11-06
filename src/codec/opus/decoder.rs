//! Opus audio decoder

use crate::codec::{AudioFrame, Decoder, Frame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, SampleFormat};
use opus::{Channels, Decoder as OpusDecoderLib};

/// Opus audio decoder
pub struct OpusDecoder {
    decoder: OpusDecoderLib,
    sample_rate: u32,
    channels: u16,
}

impl OpusDecoder {
    /// Create a new Opus decoder
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        let opus_channels = if channels == 1 {
            Channels::Mono
        } else {
            Channels::Stereo
        };

        let decoder = OpusDecoderLib::new(sample_rate, opus_channels)
            .map_err(|e| Error::codec(format!("Failed to create Opus decoder: {:?}", e)))?;

        Ok(OpusDecoder {
            decoder,
            sample_rate,
            channels,
        })
    }

    /// Decode Opus packet to audio frame
    fn decode_opus(&mut self, data: &[u8]) -> Result<AudioFrame> {
        // Maximum frame size for Opus is 120ms at 48kHz
        let max_frame_size = (self.sample_rate as usize * 120) / 1000;
        let mut output = vec![0i16; max_frame_size * self.channels as usize];

        // Decode
        let decoded_samples = self
            .decoder
            .decode(data, &mut output, false)
            .map_err(|e| Error::codec(format!("Opus decoding failed: {:?}", e)))?;

        // Convert i16 samples to bytes
        let mut pcm_data = Vec::with_capacity(decoded_samples * self.channels as usize * 2);
        for sample in output.iter().take(decoded_samples * self.channels as usize) {
            pcm_data.extend_from_slice(&sample.to_le_bytes());
        }

        // Create audio frame
        let mut frame = AudioFrame::new(
            decoded_samples,
            self.sample_rate,
            self.channels,
            SampleFormat::I16,
        );
        frame.data.push(Buffer::from_vec(pcm_data));

        Ok(frame)
    }
}

impl Decoder for OpusDecoder {
    fn send_packet(&mut self, _packet: &Packet) -> Result<()> {
        // Opus decoding is stateless for our purposes
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        // Simplified implementation - would need packet buffering
        Err(Error::TryAgain)
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

impl OpusDecoder {
    /// Convenience method to decode a packet directly to a frame
    pub fn decode_packet(&mut self, packet: &Packet) -> Result<Frame> {
        let mut audio_frame = self.decode_opus(packet.data.as_slice())?;
        audio_frame.pts = packet.pts;
        audio_frame.duration = packet.duration;
        Ok(Frame::Audio(audio_frame))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opus_decoder_creation() {
        let decoder = OpusDecoder::new(48000, 2);
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_opus_decoder_mono() {
        let decoder = OpusDecoder::new(48000, 1);
        assert!(decoder.is_ok());
    }
}
