//! Opus audio encoder
//!
//! Opus is a versatile audio codec designed for interactive speech and music
//! transmission over the Internet. It supports:
//! - Sample rates: 8000, 12000, 16000, 24000, 48000 Hz
//! - Channels: Mono (1) or Stereo (2)
//! - Bitrates: 6 kbps to 510 kbps
//! - Frame sizes: 2.5, 5, 10, 20, 40, 60 ms

use crate::codec::{AudioFrame, Encoder, Frame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, SampleFormat, Timestamp};
use opus::{Application, Bitrate, Channels, Encoder as OpusEncoderLib};
use std::collections::VecDeque;

/// Valid Opus sample rates
pub const VALID_SAMPLE_RATES: [u32; 5] = [8000, 12000, 16000, 24000, 48000];

/// Frame duration in milliseconds (Opus supports 2.5, 5, 10, 20, 40, 60 ms)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusFrameDuration {
    /// 2.5 ms frame
    Ms2_5,
    /// 5 ms frame
    Ms5,
    /// 10 ms frame
    Ms10,
    /// 20 ms frame (default, good balance)
    Ms20,
    /// 40 ms frame
    Ms40,
    /// 60 ms frame (maximum latency, best compression)
    Ms60,
}

impl OpusFrameDuration {
    /// Get duration in milliseconds
    pub fn as_ms(&self) -> f32 {
        match self {
            OpusFrameDuration::Ms2_5 => 2.5,
            OpusFrameDuration::Ms5 => 5.0,
            OpusFrameDuration::Ms10 => 10.0,
            OpusFrameDuration::Ms20 => 20.0,
            OpusFrameDuration::Ms40 => 40.0,
            OpusFrameDuration::Ms60 => 60.0,
        }
    }

    /// Calculate frame size in samples for given sample rate
    pub fn frame_size(&self, sample_rate: u32) -> usize {
        let ms = self.as_ms();
        ((sample_rate as f32 * ms) / 1000.0) as usize
    }
}

impl Default for OpusFrameDuration {
    fn default() -> Self {
        OpusFrameDuration::Ms20
    }
}

/// Opus encoder configuration
///
/// Note: Some libopus features like signal type hint, complexity, and DTX
/// are not exposed by the `opus` crate version 0.3.0. These can be enabled
/// in future versions if the crate adds support.
#[derive(Debug, Clone)]
pub struct OpusEncoderConfig {
    /// Sample rate (must be 8000, 12000, 16000, 24000, or 48000)
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo)
    pub channels: Channels,
    /// Application type (audio, voip, or low-delay)
    pub application: Application,
    /// Target bitrate in bits per second (None = auto)
    pub bitrate: Option<i32>,
    /// Frame duration
    pub frame_duration: OpusFrameDuration,
    /// Variable bitrate mode (default: true)
    pub vbr: bool,
    /// Constrained VBR mode (default: false)
    pub cvbr: bool,
    /// Enable in-band FEC for packet loss resilience
    pub inband_fec: bool,
    /// Expected packet loss percentage (0-100) for FEC tuning
    pub packet_loss_pct: i32,
}

impl Default for OpusEncoderConfig {
    fn default() -> Self {
        OpusEncoderConfig {
            sample_rate: 48000,
            channels: Channels::Stereo,
            application: Application::Audio,
            bitrate: None,
            frame_duration: OpusFrameDuration::Ms20,
            vbr: true,
            cvbr: false,
            inband_fec: false,
            packet_loss_pct: 0,
        }
    }
}

impl OpusEncoderConfig {
    /// Create configuration for voice/speech encoding
    pub fn voice(sample_rate: u32, channels: u16) -> Self {
        OpusEncoderConfig {
            sample_rate,
            channels: if channels == 1 {
                Channels::Mono
            } else {
                Channels::Stereo
            },
            application: Application::Voip,
            bitrate: Some(32000), // 32 kbps is good for voice
            frame_duration: OpusFrameDuration::Ms20,
            vbr: true,
            cvbr: false,
            inband_fec: true, // Enable FEC for voice
            packet_loss_pct: 10, // Assume 10% loss for robustness
        }
    }

    /// Create configuration for music encoding
    pub fn music(sample_rate: u32, channels: u16) -> Self {
        OpusEncoderConfig {
            sample_rate,
            channels: if channels == 1 {
                Channels::Mono
            } else {
                Channels::Stereo
            },
            application: Application::Audio,
            bitrate: Some(128000), // 128 kbps for music
            frame_duration: OpusFrameDuration::Ms20,
            vbr: true,
            cvbr: false,
            inband_fec: false,
            packet_loss_pct: 0,
        }
    }

    /// Create configuration for low-latency encoding
    pub fn low_latency(sample_rate: u32, channels: u16) -> Self {
        OpusEncoderConfig {
            sample_rate,
            channels: if channels == 1 {
                Channels::Mono
            } else {
                Channels::Stereo
            },
            application: Application::LowDelay,
            bitrate: Some(64000),
            frame_duration: OpusFrameDuration::Ms10, // Shorter frames
            vbr: true,
            cvbr: false,
            inband_fec: false,
            packet_loss_pct: 0,
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if !VALID_SAMPLE_RATES.contains(&self.sample_rate) {
            return Err(Error::codec(format!(
                "Invalid Opus sample rate: {}. Valid rates: {:?}",
                self.sample_rate, VALID_SAMPLE_RATES
            )));
        }

        if let Some(bitrate) = self.bitrate {
            if bitrate < 6000 || bitrate > 510000 {
                return Err(Error::codec(format!(
                    "Invalid Opus bitrate: {}. Valid range: 6000-510000 bps",
                    bitrate
                )));
            }
        }

        if self.packet_loss_pct < 0 || self.packet_loss_pct > 100 {
            return Err(Error::codec(format!(
                "Invalid packet loss percentage: {}. Valid range: 0-100",
                self.packet_loss_pct
            )));
        }

        Ok(())
    }
}

/// Opus audio encoder
pub struct OpusEncoder {
    /// The underlying Opus encoder
    encoder: OpusEncoderLib,
    /// Encoder configuration
    config: OpusEncoderConfig,
    /// Frame size in samples per channel
    frame_size: usize,
    /// Input sample buffer (interleaved i16 samples)
    input_buffer: Vec<i16>,
    /// Output packet queue
    output_queue: VecDeque<Packet>,
    /// Current presentation timestamp
    current_pts: i64,
    /// Samples processed counter (for PTS calculation)
    samples_processed: u64,
    /// Stream index for output packets
    stream_index: usize,
    /// Number of channels
    num_channels: u16,
}

impl OpusEncoder {
    /// Create a new Opus encoder with default configuration
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        let config = OpusEncoderConfig {
            sample_rate,
            channels: if channels == 1 {
                Channels::Mono
            } else {
                Channels::Stereo
            },
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create a new Opus encoder with custom configuration
    pub fn with_config(config: OpusEncoderConfig) -> Result<Self> {
        config.validate()?;

        let mut encoder =
            OpusEncoderLib::new(config.sample_rate, config.channels, config.application)
                .map_err(|e| Error::codec(format!("Failed to create Opus encoder: {:?}", e)))?;

        // Configure bitrate
        if let Some(bitrate) = config.bitrate {
            encoder
                .set_bitrate(Bitrate::Bits(bitrate))
                .map_err(|e| Error::codec(format!("Failed to set bitrate: {:?}", e)))?;
        }

        // Configure VBR
        encoder
            .set_vbr(config.vbr)
            .map_err(|e| Error::codec(format!("Failed to set VBR: {:?}", e)))?;

        // Configure constrained VBR
        encoder
            .set_vbr_constraint(config.cvbr)
            .map_err(|e| Error::codec(format!("Failed to set CVBR: {:?}", e)))?;

        // Configure in-band FEC
        encoder
            .set_inband_fec(config.inband_fec)
            .map_err(|e| Error::codec(format!("Failed to set in-band FEC: {:?}", e)))?;

        // Configure packet loss percentage
        encoder
            .set_packet_loss_perc(config.packet_loss_pct)
            .map_err(|e| Error::codec(format!("Failed to set packet loss: {:?}", e)))?;

        let frame_size = config.frame_duration.frame_size(config.sample_rate);
        let num_channels = match config.channels {
            Channels::Mono => 1,
            Channels::Stereo => 2,
        };

        Ok(OpusEncoder {
            encoder,
            config,
            frame_size,
            input_buffer: Vec::with_capacity(frame_size * num_channels as usize * 2),
            output_queue: VecDeque::new(),
            current_pts: 0,
            samples_processed: 0,
            stream_index: 0,
            num_channels,
        })
    }

    /// Set stream index for output packets
    pub fn set_stream_index(&mut self, index: usize) {
        self.stream_index = index;
    }

    /// Set target bitrate
    pub fn set_bitrate(&mut self, bitrate: i32) -> Result<()> {
        if bitrate < 6000 || bitrate > 510000 {
            return Err(Error::codec(format!(
                "Invalid bitrate: {}. Valid range: 6000-510000",
                bitrate
            )));
        }
        self.encoder
            .set_bitrate(Bitrate::Bits(bitrate))
            .map_err(|e| Error::codec(format!("Failed to set Opus bitrate: {:?}", e)))?;
        Ok(())
    }

    /// Enable/disable in-band FEC
    pub fn set_fec(&mut self, enabled: bool) -> Result<()> {
        self.encoder
            .set_inband_fec(enabled)
            .map_err(|e| Error::codec(format!("Failed to set FEC: {:?}", e)))?;
        Ok(())
    }

    /// Set expected packet loss percentage (for FEC tuning)
    pub fn set_packet_loss(&mut self, percent: i32) -> Result<()> {
        if percent < 0 || percent > 100 {
            return Err(Error::codec(format!(
                "Invalid packet loss: {}. Valid range: 0-100",
                percent
            )));
        }
        self.encoder
            .set_packet_loss_perc(percent)
            .map_err(|e| Error::codec(format!("Failed to set packet loss: {:?}", e)))?;
        Ok(())
    }

    /// Get current bitrate
    pub fn bitrate(&mut self) -> i32 {
        match self.encoder.get_bitrate() {
            Ok(Bitrate::Bits(b)) => b,
            Ok(Bitrate::Auto) => -1,
            Ok(Bitrate::Max) => 510000,
            Err(_) => -1,
        }
    }

    /// Get frame size in samples
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Get number of channels
    pub fn channels(&self) -> u16 {
        self.num_channels
    }

    /// Convert audio frame to i16 samples
    fn frame_to_samples(&self, frame: &AudioFrame) -> Result<Vec<i16>> {
        if frame.data.is_empty() {
            return Err(Error::codec("Empty audio frame"));
        }

        let data = frame.data[0].as_slice();

        // Convert based on sample format
        let samples: Vec<i16> = match frame.format {
            SampleFormat::I16 => {
                // Already i16, just copy
                data.chunks_exact(2)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect()
            }
            SampleFormat::F32 => {
                // Convert f32 to i16
                data.chunks_exact(4)
                    .map(|chunk| {
                        let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        (f.clamp(-1.0, 1.0) * 32767.0) as i16
                    })
                    .collect()
            }
            SampleFormat::I32 => {
                // Convert i32 to i16
                data.chunks_exact(4)
                    .map(|chunk| {
                        let i = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        (i >> 16) as i16
                    })
                    .collect()
            }
            SampleFormat::U8 => {
                // Convert u8 to i16
                data.iter().map(|&b| ((b as i16) - 128) * 256).collect()
            }
            _ => {
                return Err(Error::codec(format!(
                    "Unsupported sample format: {:?}",
                    frame.format
                )));
            }
        };

        Ok(samples)
    }

    /// Process buffered samples and generate packets
    fn process_buffer(&mut self) -> Result<()> {
        let samples_per_frame = self.frame_size * self.num_channels as usize;

        while self.input_buffer.len() >= samples_per_frame {
            // Extract one frame worth of samples
            let frame_samples: Vec<i16> = self.input_buffer.drain(..samples_per_frame).collect();

            // Encode
            let mut output = vec![0u8; 4000]; // Max Opus packet size
            let encoded_size = self
                .encoder
                .encode(&frame_samples, &mut output)
                .map_err(|e| Error::codec(format!("Opus encoding failed: {:?}", e)))?;

            output.truncate(encoded_size);

            // Calculate PTS and duration
            let pts = Timestamp::new(self.current_pts);
            let duration = self.frame_size as i64;

            // Update tracking
            self.current_pts += self.frame_size as i64;
            self.samples_processed += self.frame_size as u64;

            // Create packet
            let mut packet = Packet::new_audio(self.stream_index, Buffer::from_vec(output));
            packet.pts = pts;
            packet.dts = pts;
            packet.duration = duration;
            packet.flags.keyframe = true; // All Opus packets are independent

            self.output_queue.push_back(packet);
        }

        Ok(())
    }

    /// Encode an audio frame directly to a packet (convenience method)
    pub fn encode_frame(&mut self, frame: &AudioFrame) -> Result<Packet> {
        // Validate sample rate
        if frame.sample_rate != self.config.sample_rate {
            return Err(Error::codec(format!(
                "Sample rate mismatch: encoder expects {}, got {}",
                self.config.sample_rate, frame.sample_rate
            )));
        }

        // Validate channels
        if frame.channels != self.num_channels {
            return Err(Error::codec(format!(
                "Channel count mismatch: encoder expects {}, got {}",
                self.num_channels, frame.channels
            )));
        }

        // Convert frame to samples
        let samples = self.frame_to_samples(frame)?;

        // Ensure we have exactly one frame's worth
        let expected_samples = self.frame_size * self.num_channels as usize;
        if samples.len() < expected_samples {
            // Pad with silence
            let mut padded = samples;
            padded.resize(expected_samples, 0);

            let mut output = vec![0u8; 4000];
            let encoded_size = self
                .encoder
                .encode(&padded, &mut output)
                .map_err(|e| Error::codec(format!("Opus encoding failed: {:?}", e)))?;
            output.truncate(encoded_size);

            let mut packet = Packet::new_audio(self.stream_index, Buffer::from_vec(output));
            packet.pts = frame.pts;
            packet.duration = frame.duration;
            return Ok(packet);
        }

        // Encode
        let mut output = vec![0u8; 4000];
        let encoded_size = self
            .encoder
            .encode(&samples[..expected_samples], &mut output)
            .map_err(|e| Error::codec(format!("Opus encoding failed: {:?}", e)))?;
        output.truncate(encoded_size);

        let mut packet = Packet::new_audio(self.stream_index, Buffer::from_vec(output));
        packet.pts = frame.pts;
        packet.duration = frame.duration;
        Ok(packet)
    }

    /// Get encoder lookahead in samples
    pub fn lookahead(&mut self) -> u32 {
        self.encoder.get_lookahead().unwrap_or(0) as u32
    }
}

impl Encoder for OpusEncoder {
    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        match frame {
            Frame::Audio(audio_frame) => {
                // Validate sample rate
                if audio_frame.sample_rate != self.config.sample_rate {
                    return Err(Error::codec(format!(
                        "Sample rate mismatch: encoder expects {}, got {}",
                        self.config.sample_rate, audio_frame.sample_rate
                    )));
                }

                // Set initial PTS if this is the first frame
                if self.samples_processed == 0 && audio_frame.pts.is_valid() {
                    self.current_pts = audio_frame.pts.value;
                }

                // Convert frame to samples and add to buffer
                let samples = self.frame_to_samples(audio_frame)?;
                self.input_buffer.extend(samples);

                // Process any complete frames
                self.process_buffer()?;

                Ok(())
            }
            Frame::Video(_) => Err(Error::codec("Opus encoder only accepts audio frames")),
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if let Some(packet) = self.output_queue.pop_front() {
            Ok(packet)
        } else {
            Err(Error::TryAgain)
        }
    }

    fn flush(&mut self) -> Result<()> {
        // Pad remaining samples with silence if needed
        if !self.input_buffer.is_empty() {
            let samples_per_frame = self.frame_size * self.num_channels as usize;
            let padding_needed = samples_per_frame - (self.input_buffer.len() % samples_per_frame);
            if padding_needed < samples_per_frame {
                self.input_buffer.extend(vec![0i16; padding_needed]);
            }
            self.process_buffer()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opus_encoder_creation() {
        let encoder = OpusEncoder::new(48000, 2);
        assert!(encoder.is_ok());
        let enc = encoder.unwrap();
        assert_eq!(enc.sample_rate(), 48000);
        assert_eq!(enc.channels(), 2);
    }

    #[test]
    fn test_opus_encoder_mono() {
        let encoder = OpusEncoder::new(48000, 1);
        assert!(encoder.is_ok());
        let enc = encoder.unwrap();
        assert_eq!(enc.channels(), 1);
    }

    #[test]
    fn test_opus_encoder_with_config() {
        let config = OpusEncoderConfig {
            sample_rate: 48000,
            channels: Channels::Mono,
            application: Application::Voip,
            bitrate: Some(64000),
            ..Default::default()
        };
        let encoder = OpusEncoder::with_config(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_opus_encoder_voice_config() {
        let config = OpusEncoderConfig::voice(16000, 1);
        let encoder = OpusEncoder::with_config(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_opus_encoder_music_config() {
        let config = OpusEncoderConfig::music(48000, 2);
        let encoder = OpusEncoder::with_config(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_opus_encoder_low_latency_config() {
        let config = OpusEncoderConfig::low_latency(48000, 2);
        let encoder = OpusEncoder::with_config(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_opus_encoder_invalid_sample_rate() {
        let config = OpusEncoderConfig {
            sample_rate: 44100, // Invalid for Opus
            ..Default::default()
        };
        let result = OpusEncoder::with_config(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_opus_encoder_all_valid_sample_rates() {
        for rate in VALID_SAMPLE_RATES {
            let config = OpusEncoderConfig {
                sample_rate: rate,
                channels: Channels::Stereo,
                ..Default::default()
            };
            let encoder = OpusEncoder::with_config(config);
            assert!(encoder.is_ok(), "Failed for sample rate {}", rate);
        }
    }

    #[test]
    fn test_opus_encode_silence() {
        let mut encoder = OpusEncoder::new(48000, 2).unwrap();
        let frame_size = encoder.frame_size();

        // Create a silent frame
        let silence = vec![0i16; frame_size * 2]; // stereo
        let silence_bytes: Vec<u8> = silence
            .iter()
            .flat_map(|&s| s.to_le_bytes())
            .collect();

        let mut frame = AudioFrame::new(frame_size, 48000, 2, SampleFormat::I16);
        frame.data.push(Buffer::from_vec(silence_bytes));
        frame.pts = Timestamp::new(0);
        frame.duration = frame_size as i64;

        let result = encoder.encode_frame(&frame);
        assert!(result.is_ok());

        let packet = result.unwrap();
        assert!(!packet.data.is_empty());
        assert!(packet.data.len() < 4000); // Should be much smaller than max
    }

    #[test]
    fn test_opus_encode_sine_wave() {
        let mut encoder = OpusEncoder::new(48000, 1).unwrap();
        let frame_size = encoder.frame_size();

        // Generate a 440 Hz sine wave
        let samples: Vec<i16> = (0..frame_size)
            .map(|i| {
                let t = i as f32 / 48000.0;
                (f32::sin(2.0 * std::f32::consts::PI * 440.0 * t) * 16000.0) as i16
            })
            .collect();

        let sample_bytes: Vec<u8> = samples
            .iter()
            .flat_map(|&s| s.to_le_bytes())
            .collect();

        let mut frame = AudioFrame::new(frame_size, 48000, 1, SampleFormat::I16);
        frame.data.push(Buffer::from_vec(sample_bytes));
        frame.pts = Timestamp::new(0);
        frame.duration = frame_size as i64;

        let result = encoder.encode_frame(&frame);
        assert!(result.is_ok());
    }

    #[test]
    fn test_opus_encoder_streaming() {
        let mut encoder = OpusEncoder::new(48000, 2).unwrap();
        let frame_size = encoder.frame_size();

        // Send multiple frames
        for i in 0..5 {
            let samples = vec![0i16; frame_size * 2];
            let sample_bytes: Vec<u8> = samples
                .iter()
                .flat_map(|&s| s.to_le_bytes())
                .collect();

            let mut frame = AudioFrame::new(frame_size, 48000, 2, SampleFormat::I16);
            frame.data.push(Buffer::from_vec(sample_bytes));
            frame.pts = Timestamp::new((i * frame_size) as i64);

            let result = encoder.send_frame(&Frame::Audio(frame));
            assert!(result.is_ok());

            // Should be able to receive a packet
            let packet_result = encoder.receive_packet();
            assert!(packet_result.is_ok());
        }
    }

    #[test]
    fn test_opus_set_bitrate() {
        let mut encoder = OpusEncoder::new(48000, 2).unwrap();

        assert!(encoder.set_bitrate(64000).is_ok());
        assert!(encoder.set_bitrate(128000).is_ok());
        assert!(encoder.set_bitrate(5000).is_err()); // Too low
        assert!(encoder.set_bitrate(600000).is_err()); // Too high
    }

    #[test]
    fn test_opus_set_fec() {
        let mut encoder = OpusEncoder::new(48000, 2).unwrap();

        assert!(encoder.set_fec(true).is_ok());
        assert!(encoder.set_fec(false).is_ok());
    }

    #[test]
    fn test_opus_set_packet_loss() {
        let mut encoder = OpusEncoder::new(48000, 2).unwrap();

        assert!(encoder.set_packet_loss(0).is_ok());
        assert!(encoder.set_packet_loss(50).is_ok());
        assert!(encoder.set_packet_loss(100).is_ok());
        assert!(encoder.set_packet_loss(-1).is_err());
        assert!(encoder.set_packet_loss(101).is_err());
    }

    #[test]
    fn test_frame_duration_calculations() {
        assert_eq!(OpusFrameDuration::Ms20.frame_size(48000), 960);
        assert_eq!(OpusFrameDuration::Ms10.frame_size(48000), 480);
        assert_eq!(OpusFrameDuration::Ms20.frame_size(16000), 320);
        assert_eq!(OpusFrameDuration::Ms60.frame_size(48000), 2880);
    }
}
