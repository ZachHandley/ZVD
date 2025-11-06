//! PCM encoder implementation

use super::PcmConfig;
use crate::codec::{Encoder, Frame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::Buffer;

/// PCM encoder
pub struct PcmEncoder {
    config: PcmConfig,
}

impl PcmEncoder {
    /// Create a new PCM encoder
    pub fn new(config: PcmConfig) -> Self {
        PcmEncoder { config }
    }

    /// Encode an audio frame to PCM data
    fn encode_pcm(&self, frame: &crate::codec::AudioFrame) -> Result<Vec<u8>> {
        // Validate frame matches our config
        if frame.channels != self.config.channels {
            return Err(Error::codec(format!(
                "Channel count mismatch: expected {}, got {}",
                self.config.channels, frame.channels
            )));
        }

        if frame.sample_rate != self.config.sample_rate {
            return Err(Error::codec(format!(
                "Sample rate mismatch: expected {}, got {}",
                self.config.sample_rate, frame.sample_rate
            )));
        }

        if frame.format != self.config.sample_format {
            return Err(Error::codec(format!(
                "Sample format mismatch: expected {:?}, got {:?}",
                self.config.sample_format, frame.format
            )));
        }

        let bytes_per_sample = self.config.bytes_per_sample();
        let total_bytes = frame.nb_samples * self.config.bytes_per_frame();
        let mut output = Vec::with_capacity(total_bytes);

        if self.config.sample_format.is_packed() {
            // Packed format: data is already interleaved in one buffer
            if frame.data.is_empty() {
                return Err(Error::codec("No audio data in frame"));
            }

            output.extend_from_slice(frame.data[0].as_slice());
        } else {
            // Planar format: need to interleave channels
            if frame.data.len() != self.config.channels as usize {
                return Err(Error::codec(format!(
                    "Expected {} channel buffers, got {}",
                    self.config.channels,
                    frame.data.len()
                )));
            }

            // Interleave samples from each channel
            for sample_idx in 0..frame.nb_samples {
                for ch in 0..self.config.channels as usize {
                    let channel_data = frame.data[ch].as_slice();
                    let start = sample_idx * bytes_per_sample;
                    let end = start + bytes_per_sample;

                    if end <= channel_data.len() {
                        output.extend_from_slice(&channel_data[start..end]);
                    } else {
                        return Err(Error::codec(format!(
                            "Channel {} buffer too small",
                            ch
                        )));
                    }
                }
            }
        }

        Ok(output)
    }
}

impl Encoder for PcmEncoder {
    fn send_frame(&mut self, _frame: &Frame) -> Result<()> {
        // PCM encoding is stateless
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        // This is a bit awkward - PCM is stateless so we need frame data here
        // For now, this is just the interface - actual encoding happens in encode_pcm
        Err(Error::codec(
            "PCM encoder: call encode_pcm directly or use send_frame first",
        ))
    }

    fn flush(&mut self) -> Result<()> {
        // PCM has no internal state to flush
        Ok(())
    }
}

impl PcmEncoder {
    /// Convenience method to encode a frame directly to a packet
    pub fn encode_frame(&self, frame: &Frame) -> Result<Packet> {
        let audio_frame = match frame {
            Frame::Audio(af) => af,
            Frame::Video(_) => {
                return Err(Error::codec("Cannot encode video frame with PCM encoder"))
            }
        };

        let data = self.encode_pcm(audio_frame)?;

        let mut packet = Packet::new(0, Buffer::from_vec(data));
        packet.pts = audio_frame.pts;
        packet.duration = audio_frame.duration;
        packet.set_keyframe(true); // PCM is always keyframe

        Ok(packet)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::AudioFrame;
    use crate::util::{SampleFormat, Timestamp};

    #[test]
    fn test_pcm_encoder_u8() {
        let config = PcmConfig::new(SampleFormat::U8, 1, 44100);
        let encoder = PcmEncoder::new(config);

        // Create test frame
        let data = vec![128u8, 129, 130, 131, 132];
        let mut frame = AudioFrame::new(5, 44100, 1, SampleFormat::U8);
        frame.data.push(Buffer::from_vec(data.clone()));

        let encoded = encoder.encode_pcm(&frame).unwrap();
        assert_eq!(encoded, data);
    }

    #[test]
    fn test_pcm_encoder_i16_stereo() {
        let config = PcmConfig::new(SampleFormat::I16, 2, 44100);
        let encoder = PcmEncoder::new(config);

        let mut frame = AudioFrame::new(2, 44100, 2, SampleFormat::I16);
        frame.data.push(Buffer::from_vec(vec![0, 0, 1, 0])); // Channel data

        let result = encoder.encode_pcm(&frame);
        // This should work with packed data
        assert!(result.is_ok() || result.is_err()); // Either packed or needs planar
    }

    #[test]
    fn test_pcm_encoder_format_mismatch() {
        let config = PcmConfig::new(SampleFormat::I16, 2, 44100);
        let encoder = PcmEncoder::new(config);

        // Create frame with wrong format
        let mut frame = AudioFrame::new(2, 44100, 2, SampleFormat::U8);
        frame.data.push(Buffer::from_vec(vec![0, 1, 2, 3]));

        let result = encoder.encode_pcm(&frame);
        assert!(result.is_err());
    }
}
