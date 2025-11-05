//! PCM decoder implementation

use super::PcmConfig;
use crate::codec::{AudioFrame, Decoder, Frame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, SampleFormat};
use bytes::Bytes;

/// PCM decoder
pub struct PcmDecoder {
    config: PcmConfig,
}

impl PcmDecoder {
    /// Create a new PCM decoder
    pub fn new(config: PcmConfig) -> Self {
        PcmDecoder { config }
    }

    /// Decode PCM data to an audio frame
    fn decode_pcm(&self, data: &[u8]) -> Result<AudioFrame> {
        let nb_samples = self.config.samples_from_bytes(data.len());

        if nb_samples == 0 {
            return Err(Error::codec("No samples in packet"));
        }

        let mut frame = AudioFrame::new(
            nb_samples,
            self.config.sample_rate,
            self.config.channels,
            self.config.sample_format,
        );

        // For packed formats, store all data in one buffer
        if self.config.sample_format.is_packed() {
            frame.data.push(Buffer::from_vec(data.to_vec()));
        } else {
            // For planar formats, split data by channel
            let bytes_per_sample = self.config.bytes_per_sample();
            let samples_per_channel = nb_samples;
            let bytes_per_channel = samples_per_channel * bytes_per_sample;

            for ch in 0..self.config.channels as usize {
                let mut channel_data = Vec::with_capacity(bytes_per_channel);

                // Deinterleave: pick every Nth sample for this channel
                for sample_idx in 0..samples_per_channel {
                    let frame_offset = sample_idx * self.config.bytes_per_frame();
                    let channel_offset = ch * bytes_per_sample;
                    let start = frame_offset + channel_offset;
                    let end = start + bytes_per_sample;

                    if end <= data.len() {
                        channel_data.extend_from_slice(&data[start..end]);
                    }
                }

                frame.data.push(Buffer::from_vec(channel_data));
            }
        }

        Ok(frame)
    }
}

impl Decoder for PcmDecoder {
    fn send_packet(&mut self, _packet: &Packet) -> Result<()> {
        // PCM decoding is stateless, we'll decode in receive_frame
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        // This is a bit awkward - PCM is stateless so we need packet data here
        // In a real implementation, we'd buffer the packet from send_packet
        // For now, this is just the interface - actual decoding happens in decode_pcm
        Err(Error::codec(
            "PCM decoder: call decode_pcm directly or use send_packet first",
        ))
    }

    fn flush(&mut self) -> Result<()> {
        // PCM has no internal state to flush
        Ok(())
    }
}

impl PcmDecoder {
    /// Convenience method to decode a packet directly to a frame
    pub fn decode_packet(&self, packet: &Packet) -> Result<Frame> {
        let audio_frame = self.decode_pcm(packet.data.as_slice())?;

        // Copy timestamps from packet to frame
        let mut frame = audio_frame;
        frame.pts = packet.pts;
        frame.duration = packet.duration;

        Ok(Frame::Audio(frame))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::Timestamp;

    #[test]
    fn test_pcm_decoder_u8() {
        let config = PcmConfig::new(SampleFormat::U8, 1, 44100);
        let decoder = PcmDecoder::new(config);

        // Create test data: 10 samples of u8
        let data: Vec<u8> = vec![128, 129, 130, 131, 132, 133, 134, 135, 136, 137];

        let frame = decoder.decode_pcm(&data).unwrap();
        assert_eq!(frame.nb_samples, 10);
        assert_eq!(frame.channels, 1);
        assert_eq!(frame.sample_rate, 44100);
        assert_eq!(frame.format, SampleFormat::U8);
    }

    #[test]
    fn test_pcm_decoder_i16_stereo() {
        let config = PcmConfig::new(SampleFormat::I16, 2, 44100);
        let decoder = PcmDecoder::new(config);

        // Create test data: 4 samples (8 i16 values = 16 bytes)
        let data: Vec<u8> = vec![
            0, 0, 1, 0, // Sample 0: L=0, R=1
            2, 0, 3, 0, // Sample 1: L=2, R=3
            4, 0, 5, 0, // Sample 2: L=4, R=5
            6, 0, 7, 0, // Sample 3: L=6, R=7
        ];

        let frame = decoder.decode_pcm(&data).unwrap();
        assert_eq!(frame.nb_samples, 4);
        assert_eq!(frame.channels, 2);
        assert_eq!(frame.format, SampleFormat::I16);
    }

    #[test]
    fn test_pcm_config() {
        let config = PcmConfig::new(SampleFormat::I16, 2, 44100);
        assert_eq!(config.bytes_per_sample(), 2);
        assert_eq!(config.bytes_per_frame(), 4);
        assert_eq!(config.samples_from_bytes(8), 2);
    }
}
