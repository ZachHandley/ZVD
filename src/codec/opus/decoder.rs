//! Opus audio decoder
//!
//! Opus is a versatile audio codec designed for interactive speech and music
//! transmission over the Internet. This decoder supports:
//! - Sample rates: 8000, 12000, 16000, 24000, 48000 Hz
//! - Channels: Mono (1) or Stereo (2)
//! - Packet loss concealment (PLC)
//! - Forward error correction (FEC)

use crate::codec::{AudioFrame, Decoder, Frame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, SampleFormat, Timestamp};
use opus::{Channels, Decoder as OpusDecoderLib};
use std::collections::VecDeque;

/// Valid Opus sample rates
const VALID_SAMPLE_RATES: [u32; 5] = [8000, 12000, 16000, 24000, 48000];

/// Maximum Opus frame size (120ms at 48kHz)
const MAX_FRAME_SIZE: usize = 5760;

/// Opus decoder configuration
#[derive(Debug, Clone)]
pub struct OpusDecoderConfig {
    /// Output sample rate (must be 8000, 12000, 16000, 24000, or 48000)
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo)
    pub channels: u16,
    /// Enable packet loss concealment
    pub enable_plc: bool,
    /// Enable forward error correction decoding
    pub enable_fec: bool,
    /// Output sample format
    pub sample_format: SampleFormat,
}

impl Default for OpusDecoderConfig {
    fn default() -> Self {
        OpusDecoderConfig {
            sample_rate: 48000,
            channels: 2,
            enable_plc: true,
            enable_fec: true,
            sample_format: SampleFormat::I16,
        }
    }
}

impl OpusDecoderConfig {
    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if !VALID_SAMPLE_RATES.contains(&self.sample_rate) {
            return Err(Error::codec(format!(
                "Invalid Opus sample rate: {}. Valid rates: {:?}",
                self.sample_rate, VALID_SAMPLE_RATES
            )));
        }

        if self.channels != 1 && self.channels != 2 {
            return Err(Error::codec(format!(
                "Invalid channel count: {}. Must be 1 (mono) or 2 (stereo)",
                self.channels
            )));
        }

        Ok(())
    }
}

/// Opus audio decoder
pub struct OpusDecoder {
    /// The underlying Opus decoder
    decoder: OpusDecoderLib,
    /// Decoder configuration
    config: OpusDecoderConfig,
    /// Input packet queue
    packet_queue: VecDeque<Packet>,
    /// Output frame queue
    output_queue: VecDeque<AudioFrame>,
    /// Previous packet data for FEC decoding
    prev_packet: Option<Vec<u8>>,
    /// Track if previous packet was lost (for PLC)
    prev_packet_lost: bool,
    /// Current presentation timestamp
    current_pts: i64,
    /// Samples decoded counter
    samples_decoded: u64,
    /// Number of channels
    num_channels: u16,
    /// Decoder gain in dB (0 = unity)
    gain: i32,
}

impl OpusDecoder {
    /// Create a new Opus decoder with default configuration
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        let config = OpusDecoderConfig {
            sample_rate,
            channels,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create a new Opus decoder with custom configuration
    pub fn with_config(config: OpusDecoderConfig) -> Result<Self> {
        config.validate()?;

        let opus_channels = if config.channels == 1 {
            Channels::Mono
        } else {
            Channels::Stereo
        };

        let decoder = OpusDecoderLib::new(config.sample_rate, opus_channels)
            .map_err(|e| Error::codec(format!("Failed to create Opus decoder: {:?}", e)))?;

        Ok(OpusDecoder {
            decoder,
            config,
            packet_queue: VecDeque::new(),
            output_queue: VecDeque::new(),
            prev_packet: None,
            prev_packet_lost: false,
            current_pts: 0,
            samples_decoded: 0,
            num_channels: if opus_channels == Channels::Mono {
                1
            } else {
                2
            },
            gain: 0,
        })
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Get number of channels
    pub fn channels(&self) -> u16 {
        self.num_channels
    }

    /// Get output sample format
    pub fn sample_format(&self) -> SampleFormat {
        self.config.sample_format
    }

    /// Set decoder gain in dB
    pub fn set_gain(&mut self, gain_db: i32) -> Result<()> {
        self.decoder
            .set_gain(gain_db)
            .map_err(|e| Error::codec(format!("Failed to set gain: {:?}", e)))?;
        self.gain = gain_db;
        Ok(())
    }

    /// Get decoder gain in dB
    pub fn gain(&self) -> i32 {
        self.gain
    }

    /// Get last packet duration (for PLC)
    pub fn last_packet_duration(&mut self) -> Result<usize> {
        self.decoder
            .get_last_packet_duration()
            .map(|d| d as usize)
            .map_err(|e| Error::codec(format!("Failed to get last packet duration: {:?}", e)))
    }

    /// Decode Opus packet to PCM samples
    fn decode_packet_internal(&mut self, data: Option<&[u8]>, fec: bool) -> Result<Vec<i16>> {
        // Calculate output buffer size
        let max_samples = MAX_FRAME_SIZE * self.num_channels as usize;
        let mut output = vec![0i16; max_samples];

        let decoded_samples = match data {
            Some(packet_data) => {
                // Normal decode or FEC decode
                self.decoder
                    .decode(packet_data, &mut output, fec)
                    .map_err(|e| Error::codec(format!("Opus decoding failed: {:?}", e)))?
            }
            None => {
                // Packet loss concealment - decode with no data
                // Try to get the duration of the last successfully decoded packet
                let plc_samples = self
                    .decoder
                    .get_last_packet_duration()
                    .unwrap_or(960) as usize;
                self.decoder
                    .decode(&[], &mut output, false)
                    .unwrap_or(plc_samples)
            }
        };

        // Truncate to actual decoded samples
        output.truncate(decoded_samples * self.num_channels as usize);
        Ok(output)
    }

    /// Convert i16 samples to AudioFrame with appropriate format
    fn samples_to_frame(&self, samples: Vec<i16>, pts: Timestamp, duration: i64) -> AudioFrame {
        let nb_samples = samples.len() / self.num_channels as usize;

        // Convert based on output format
        let data = match self.config.sample_format {
            SampleFormat::I16 => {
                // Direct i16 output
                let bytes: Vec<u8> = samples.iter().flat_map(|&s| s.to_le_bytes()).collect();
                Buffer::from_vec(bytes)
            }
            SampleFormat::F32 => {
                // Convert to f32
                let floats: Vec<u8> = samples
                    .iter()
                    .flat_map(|&s| {
                        let f = s as f32 / 32768.0;
                        f.to_le_bytes()
                    })
                    .collect();
                Buffer::from_vec(floats)
            }
            SampleFormat::I32 => {
                // Convert to i32
                let ints: Vec<u8> = samples
                    .iter()
                    .flat_map(|&s| ((s as i32) << 16).to_le_bytes())
                    .collect();
                Buffer::from_vec(ints)
            }
            _ => {
                // Default to i16
                let bytes: Vec<u8> = samples.iter().flat_map(|&s| s.to_le_bytes()).collect();
                Buffer::from_vec(bytes)
            }
        };

        let mut frame = AudioFrame::new(
            nb_samples,
            self.config.sample_rate,
            self.num_channels,
            self.config.sample_format,
        );
        frame.data.push(data);
        frame.pts = pts;
        frame.duration = duration;

        frame
    }

    /// Process queued packets
    fn process_queue(&mut self) -> Result<()> {
        while let Some(packet) = self.packet_queue.pop_front() {
            let packet_data = packet.data.as_slice();

            // Try FEC first if we have previous packet data and last packet was marked as lost
            if self.config.enable_fec && self.prev_packet_lost && packet_data.len() > 0 {
                // Decode with FEC using current packet to recover previous frame
                if let Ok(fec_samples) = self.decode_packet_internal(Some(packet_data), true) {
                    if !fec_samples.is_empty() {
                        let fec_frame = self.samples_to_frame(
                            fec_samples.clone(),
                            Timestamp::new(self.current_pts),
                            (fec_samples.len() / self.num_channels as usize) as i64,
                        );
                        self.output_queue.push_back(fec_frame);
                        self.current_pts +=
                            (fec_samples.len() / self.num_channels as usize) as i64;
                        self.samples_decoded +=
                            (fec_samples.len() / self.num_channels as usize) as u64;
                    }
                }
            }

            // Normal decode
            let samples = self.decode_packet_internal(Some(packet_data), false)?;

            // Store packet for potential FEC in future
            self.prev_packet = Some(packet_data.to_vec());
            self.prev_packet_lost = false;

            // Create frame
            let nb_samples = samples.len() / self.num_channels as usize;
            let pts = if packet.pts.is_valid() {
                packet.pts
            } else {
                Timestamp::new(self.current_pts)
            };

            let frame = self.samples_to_frame(samples, pts, nb_samples as i64);

            // Update tracking
            self.current_pts = pts.value + nb_samples as i64;
            self.samples_decoded += nb_samples as u64;

            self.output_queue.push_back(frame);
        }

        Ok(())
    }

    /// Signal packet loss for PLC
    pub fn signal_packet_loss(&mut self) -> Result<AudioFrame> {
        self.prev_packet_lost = true;

        if self.config.enable_plc {
            // Perform packet loss concealment
            let samples = self.decode_packet_internal(None, false)?;
            let nb_samples = samples.len() / self.num_channels as usize;

            let frame = self.samples_to_frame(
                samples,
                Timestamp::new(self.current_pts),
                nb_samples as i64,
            );

            self.current_pts += nb_samples as i64;
            self.samples_decoded += nb_samples as u64;

            Ok(frame)
        } else {
            Err(Error::codec("PLC disabled and packet lost"))
        }
    }

    /// Decode a packet directly to a frame (convenience method)
    pub fn decode_packet(&mut self, packet: &Packet) -> Result<Frame> {
        let packet_data = packet.data.as_slice();

        // Decode
        let samples = self.decode_packet_internal(Some(packet_data), false)?;

        // Store for FEC
        self.prev_packet = Some(packet_data.to_vec());
        self.prev_packet_lost = false;

        // Create frame
        let nb_samples = samples.len() / self.num_channels as usize;
        let pts = if packet.pts.is_valid() {
            packet.pts
        } else {
            Timestamp::new(self.current_pts)
        };

        let frame = self.samples_to_frame(samples, pts, packet.duration);

        // Update tracking
        self.current_pts = pts.value + nb_samples as i64;
        self.samples_decoded += nb_samples as u64;

        Ok(Frame::Audio(frame))
    }

    /// Decode with FEC recovery from previous packet loss
    pub fn decode_with_fec(&mut self, packet: &Packet) -> Result<(Option<AudioFrame>, AudioFrame)> {
        let packet_data = packet.data.as_slice();

        // First, try to recover previous lost packet using FEC
        let recovered_frame = if self.prev_packet_lost && self.config.enable_fec {
            match self.decode_packet_internal(Some(packet_data), true) {
                Ok(samples) if !samples.is_empty() => {
                    let nb_samples = samples.len() / self.num_channels as usize;
                    Some(self.samples_to_frame(
                        samples,
                        Timestamp::new(self.current_pts),
                        nb_samples as i64,
                    ))
                }
                _ => None,
            }
        } else {
            None
        };

        // Update PTS if we recovered a frame
        if let Some(ref frame) = recovered_frame {
            self.current_pts += frame.nb_samples as i64;
            self.samples_decoded += frame.nb_samples as u64;
        }

        // Now decode the current packet normally
        let samples = self.decode_packet_internal(Some(packet_data), false)?;
        let nb_samples = samples.len() / self.num_channels as usize;

        // Store for future FEC
        self.prev_packet = Some(packet_data.to_vec());
        self.prev_packet_lost = false;

        let pts = if packet.pts.is_valid() {
            packet.pts
        } else {
            Timestamp::new(self.current_pts)
        };

        let current_frame = self.samples_to_frame(samples, pts, nb_samples as i64);

        self.current_pts = pts.value + nb_samples as i64;
        self.samples_decoded += nb_samples as u64;

        Ok((recovered_frame, current_frame))
    }

    /// Reset decoder state
    pub fn reset(&mut self) -> Result<()> {
        // Recreate decoder to reset state
        let opus_channels = if self.config.channels == 1 {
            Channels::Mono
        } else {
            Channels::Stereo
        };

        self.decoder = OpusDecoderLib::new(self.config.sample_rate, opus_channels)
            .map_err(|e| Error::codec(format!("Failed to reset Opus decoder: {:?}", e)))?;

        // Apply gain if set
        if self.gain != 0 {
            self.decoder
                .set_gain(self.gain)
                .map_err(|e| Error::codec(format!("Failed to restore gain: {:?}", e)))?;
        }

        // Clear state
        self.packet_queue.clear();
        self.output_queue.clear();
        self.prev_packet = None;
        self.prev_packet_lost = false;
        self.current_pts = 0;
        self.samples_decoded = 0;

        Ok(())
    }

    /// Get number of samples decoded
    pub fn samples_decoded(&self) -> u64 {
        self.samples_decoded
    }
}

impl Decoder for OpusDecoder {
    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        // Set initial PTS from first packet
        if self.samples_decoded == 0 && packet.pts.is_valid() {
            self.current_pts = packet.pts.value;
        }

        // Queue the packet for processing
        self.packet_queue.push_back(packet.clone());

        // Process immediately
        self.process_queue()?;

        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(frame) = self.output_queue.pop_front() {
            Ok(Frame::Audio(frame))
        } else {
            Err(Error::TryAgain)
        }
    }

    fn flush(&mut self) -> Result<()> {
        // Process any remaining packets
        self.process_queue()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::opus::encoder::OpusEncoder as OpusEncoderImpl;
    use crate::codec::Encoder;

    #[test]
    fn test_opus_decoder_creation() {
        let decoder = OpusDecoder::new(48000, 2);
        assert!(decoder.is_ok());
        let dec = decoder.unwrap();
        assert_eq!(dec.sample_rate(), 48000);
        assert_eq!(dec.channels(), 2);
    }

    #[test]
    fn test_opus_decoder_mono() {
        let decoder = OpusDecoder::new(48000, 1);
        assert!(decoder.is_ok());
        let dec = decoder.unwrap();
        assert_eq!(dec.channels(), 1);
    }

    #[test]
    fn test_opus_decoder_invalid_sample_rate() {
        let config = OpusDecoderConfig {
            sample_rate: 44100, // Invalid for Opus
            channels: 2,
            ..Default::default()
        };
        let result = OpusDecoder::with_config(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_opus_decoder_all_valid_sample_rates() {
        for rate in VALID_SAMPLE_RATES {
            let decoder = OpusDecoder::new(rate, 2);
            assert!(decoder.is_ok(), "Failed for sample rate {}", rate);
        }
    }

    #[test]
    fn test_opus_encode_decode_roundtrip() {
        // Create encoder
        let mut encoder = OpusEncoderImpl::new(48000, 2).unwrap();
        let frame_size = encoder.frame_size();

        // Create decoder
        let mut decoder = OpusDecoder::new(48000, 2).unwrap();

        // Generate a test signal (stereo sine wave)
        let samples: Vec<i16> = (0..(frame_size * 2))
            .map(|i| {
                let t = (i / 2) as f32 / 48000.0;
                let ch = i % 2;
                let freq = if ch == 0 { 440.0 } else { 880.0 };
                (f32::sin(2.0 * std::f32::consts::PI * freq * t) * 16000.0) as i16
            })
            .collect();

        let sample_bytes: Vec<u8> = samples.iter().flat_map(|&s| s.to_le_bytes()).collect();

        // Create input frame
        let mut input_frame = AudioFrame::new(frame_size, 48000, 2, SampleFormat::I16);
        input_frame.data.push(Buffer::from_vec(sample_bytes));
        input_frame.pts = Timestamp::new(0);
        input_frame.duration = frame_size as i64;

        // Encode
        let packet = encoder.encode_frame(&input_frame).unwrap();
        assert!(!packet.data.is_empty());

        // Decode
        let decoded_frame = decoder.decode_packet(&packet).unwrap();

        // Verify
        match decoded_frame {
            Frame::Audio(audio) => {
                assert_eq!(audio.sample_rate, 48000);
                assert_eq!(audio.channels, 2);
                assert!(!audio.data.is_empty());
                // Opus is lossy, so we can't expect exact samples back
                // But we should get the same number of samples
                let decoded_samples =
                    audio.data[0].len() / (2 * audio.channels as usize); // 2 bytes per i16
                assert_eq!(decoded_samples, frame_size);
            }
            Frame::Video(_) => panic!("Expected audio frame"),
        }
    }

    #[test]
    fn test_opus_decode_silence() {
        // Create encoder and decoder
        let mut encoder = OpusEncoderImpl::new(48000, 1).unwrap();
        let mut decoder = OpusDecoder::new(48000, 1).unwrap();
        let frame_size = encoder.frame_size();

        // Create silent frame
        let silence = vec![0i16; frame_size];
        let sample_bytes: Vec<u8> = silence.iter().flat_map(|&s| s.to_le_bytes()).collect();

        let mut frame = AudioFrame::new(frame_size, 48000, 1, SampleFormat::I16);
        frame.data.push(Buffer::from_vec(sample_bytes));
        frame.pts = Timestamp::new(0);

        // Encode and decode
        let packet = encoder.encode_frame(&frame).unwrap();
        let decoded = decoder.decode_packet(&packet).unwrap();

        match decoded {
            Frame::Audio(audio) => {
                // Decoded silence should be mostly zeros or very low values
                let bytes = audio.data[0].as_slice();
                let samples: Vec<i16> = bytes
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]))
                    .collect();

                // Check that samples are near zero (allowing for codec artifacts)
                let max_sample = samples.iter().map(|&s| s.abs()).max().unwrap_or(0);
                assert!(
                    max_sample < 1000,
                    "Silent decode produced too loud output: {}",
                    max_sample
                );
            }
            Frame::Video(_) => panic!("Expected audio frame"),
        }
    }

    #[test]
    fn test_opus_streaming_decode() {
        let mut encoder = OpusEncoderImpl::new(48000, 2).unwrap();
        let mut decoder = OpusDecoder::new(48000, 2).unwrap();
        let frame_size = encoder.frame_size();

        // Encode and decode multiple frames
        for i in 0..10 {
            // Generate test frame
            let samples: Vec<i16> = (0..(frame_size * 2))
                .map(|j| ((i * frame_size * 2 + j) % 1000) as i16)
                .collect();
            let sample_bytes: Vec<u8> = samples.iter().flat_map(|&s| s.to_le_bytes()).collect();

            let mut frame = AudioFrame::new(frame_size, 48000, 2, SampleFormat::I16);
            frame.data.push(Buffer::from_vec(sample_bytes));
            frame.pts = Timestamp::new((i * frame_size) as i64);

            // Send to encoder
            encoder.send_frame(&Frame::Audio(frame)).unwrap();

            // Receive packet
            let packet = encoder.receive_packet().unwrap();

            // Send to decoder
            decoder.send_packet(&packet).unwrap();

            // Receive decoded frame
            let decoded = decoder.receive_frame().unwrap();
            match decoded {
                Frame::Audio(audio) => {
                    assert_eq!(audio.channels, 2);
                }
                Frame::Video(_) => panic!("Expected audio frame"),
            }
        }
    }

    #[test]
    fn test_opus_packet_loss_concealment() {
        let mut decoder = OpusDecoder::new(48000, 2).unwrap();

        // First, decode a valid packet to establish state
        let mut encoder = OpusEncoderImpl::new(48000, 2).unwrap();
        let frame_size = encoder.frame_size();

        let samples: Vec<i16> = (0..(frame_size * 2))
            .map(|i| {
                let t = (i / 2) as f32 / 48000.0;
                (f32::sin(2.0 * std::f32::consts::PI * 440.0 * t) * 16000.0) as i16
            })
            .collect();
        let sample_bytes: Vec<u8> = samples.iter().flat_map(|&s| s.to_le_bytes()).collect();

        let mut frame = AudioFrame::new(frame_size, 48000, 2, SampleFormat::I16);
        frame.data.push(Buffer::from_vec(sample_bytes));
        frame.pts = Timestamp::new(0);

        let packet = encoder.encode_frame(&frame).unwrap();
        let _ = decoder.decode_packet(&packet).unwrap();

        // Now simulate packet loss
        let plc_frame = decoder.signal_packet_loss();
        assert!(plc_frame.is_ok(), "PLC should produce a valid frame");

        let plc_audio = plc_frame.unwrap();
        assert!(!plc_audio.data.is_empty());
    }

    #[test]
    fn test_opus_decoder_reset() {
        let mut decoder = OpusDecoder::new(48000, 2).unwrap();

        // Decode something first
        let mut encoder = OpusEncoderImpl::new(48000, 2).unwrap();
        let frame_size = encoder.frame_size();

        let samples = vec![0i16; frame_size * 2];
        let sample_bytes: Vec<u8> = samples.iter().flat_map(|&s| s.to_le_bytes()).collect();

        let mut frame = AudioFrame::new(frame_size, 48000, 2, SampleFormat::I16);
        frame.data.push(Buffer::from_vec(sample_bytes));

        let packet = encoder.encode_frame(&frame).unwrap();
        let _ = decoder.decode_packet(&packet).unwrap();

        assert!(decoder.samples_decoded() > 0);

        // Reset
        decoder.reset().unwrap();

        assert_eq!(decoder.samples_decoded(), 0);
    }

    #[test]
    fn test_opus_decoder_f32_output() {
        let config = OpusDecoderConfig {
            sample_rate: 48000,
            channels: 2,
            sample_format: SampleFormat::F32,
            ..Default::default()
        };
        let mut decoder = OpusDecoder::with_config(config).unwrap();

        // Create encoder and encode a test frame
        let mut encoder = OpusEncoderImpl::new(48000, 2).unwrap();
        let frame_size = encoder.frame_size();

        let samples: Vec<i16> = (0..(frame_size * 2))
            .map(|i| ((i * 100) % 32767) as i16)
            .collect();
        let sample_bytes: Vec<u8> = samples.iter().flat_map(|&s| s.to_le_bytes()).collect();

        let mut frame = AudioFrame::new(frame_size, 48000, 2, SampleFormat::I16);
        frame.data.push(Buffer::from_vec(sample_bytes));

        let packet = encoder.encode_frame(&frame).unwrap();
        let decoded = decoder.decode_packet(&packet).unwrap();

        match decoded {
            Frame::Audio(audio) => {
                assert_eq!(audio.format, SampleFormat::F32);
                // F32 output should be 4 bytes per sample
                let expected_size = frame_size * 2 * 4; // samples * channels * 4 bytes
                assert_eq!(audio.data[0].len(), expected_size);
            }
            Frame::Video(_) => panic!("Expected audio frame"),
        }
    }

    #[test]
    fn test_opus_decoder_gain() {
        let mut decoder = OpusDecoder::new(48000, 2).unwrap();

        // Set gain
        assert!(decoder.set_gain(6).is_ok()); // +6 dB
        assert_eq!(decoder.gain(), 6);

        assert!(decoder.set_gain(-12).is_ok()); // -12 dB
        assert_eq!(decoder.gain(), -12);
    }
}
