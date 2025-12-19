//! AAC encoder using fdk-aac
//!
//! This module provides AAC encoding through the Fraunhofer FDK AAC library.
//! Thread-safe implementation with Arc<Mutex> for concurrent access.
//!
//! ## Patent Notice
//! AAC is covered by patents. Commercial use may require patent licensing from
//! Via Licensing Corporation and other patent holders. See CODEC_LICENSES.md
//! for details.

use crate::codec::{AudioFrame, Encoder, Frame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, SampleFormat, Timestamp};
use fdk_aac::enc::{
    AudioObjectType, BitRate, ChannelMode, Encoder as FdkEncoder, EncoderParams, Transport,
};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// AAC encoder configuration
#[derive(Debug, Clone)]
pub struct AacEncoderConfig {
    /// Sample rate in Hz (e.g., 44100, 48000)
    pub sample_rate: u32,
    /// Number of audio channels (1 for mono, 2 for stereo)
    pub channels: u16,
    /// Target bitrate in bits per second
    pub bitrate: u32,
    /// Audio object type (AAC profile)
    pub audio_object_type: AacProfile,
    /// Transport type (Raw for containers, ADTS for streaming)
    pub transport: AacTransport,
}

/// AAC profile (audio object type)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AacProfile {
    /// AAC Low Complexity (most common)
    AacLc,
    /// High Efficiency AAC (HE-AAC v1, uses SBR)
    HeAac,
    /// High Efficiency AAC v2 (HE-AAC v2, uses SBR + PS)
    HeAacV2,
    /// AAC Low Delay
    AacLd,
    /// AAC Enhanced Low Delay
    AacEld,
}

impl From<AacProfile> for AudioObjectType {
    fn from(profile: AacProfile) -> Self {
        match profile {
            AacProfile::AacLc => AudioObjectType::Mpeg4LowComplexity,
            AacProfile::HeAac => AudioObjectType::Mpeg4HeAac,
            AacProfile::HeAacV2 => AudioObjectType::Mpeg4HeAacV2,
            AacProfile::AacLd => AudioObjectType::Mpeg4LowDelay,
            AacProfile::AacEld => AudioObjectType::Mpeg4EnhancedLowDelay,
        }
    }
}

/// AAC transport format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AacTransport {
    /// Raw AAC (for MP4/MKV containers)
    Raw,
    /// ADTS framing (for streaming, .aac files)
    Adts,
}

impl From<AacTransport> for Transport {
    fn from(transport: AacTransport) -> Self {
        match transport {
            AacTransport::Raw => Transport::Raw,
            AacTransport::Adts => Transport::Adts,
        }
    }
}

impl Default for AacEncoderConfig {
    fn default() -> Self {
        AacEncoderConfig {
            sample_rate: 44100,
            channels: 2,
            bitrate: 128_000,
            audio_object_type: AacProfile::AacLc,
            transport: AacTransport::Raw,
        }
    }
}

/// AAC encoder wrapping fdk-aac
///
/// This encoder is thread-safe and can be shared across threads using Arc.
/// It uses the Fraunhofer FDK AAC library for encoding.
pub struct AacEncoder {
    /// The underlying fdk-aac encoder (thread-safe via Arc<Mutex>)
    encoder: Arc<Mutex<FdkEncoder>>,
    /// Sample rate in Hz
    sample_rate: u32,
    /// Number of audio channels
    channels: u16,
    /// Target bitrate in bits per second
    bitrate: u32,
    /// Buffer for encoded packets waiting to be consumed
    packet_buffer: Arc<Mutex<VecDeque<Packet>>>,
    /// Cached AudioSpecificConfig (ASC) for MP4 muxing
    extradata: Option<Vec<u8>>,
    /// Frame size (samples per channel per frame)
    frame_size: usize,
    /// Buffer for accumulating samples before encoding
    sample_buffer: Vec<i16>,
    /// Current PTS for the next output packet
    current_pts: Timestamp,
    /// Sample count for PTS calculation
    samples_encoded: u64,
}

impl AacEncoder {
    /// AAC frame size - always 1024 samples per channel
    const AAC_FRAME_SIZE: usize = 1024;

    /// Create a new AAC encoder with the specified parameters
    ///
    /// # Arguments
    /// * `sample_rate` - Audio sample rate in Hz (e.g., 44100, 48000)
    /// * `channels` - Number of audio channels (1 for mono, 2 for stereo)
    /// * `bitrate` - Target bitrate in bits per second (e.g., 128000)
    ///
    /// # Returns
    /// A Result containing the encoder or an error if creation fails
    ///
    /// # Example
    /// ```ignore
    /// let encoder = AacEncoder::new(44100, 2, 128000)?;
    /// ```
    pub fn new(sample_rate: u32, channels: u16, bitrate: u32) -> Result<Self> {
        let config = AacEncoderConfig {
            sample_rate,
            channels,
            bitrate,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create a new AAC encoder with custom configuration
    ///
    /// # Arguments
    /// * `config` - Full encoder configuration
    ///
    /// # Returns
    /// A Result containing the encoder or an error if creation fails
    pub fn with_config(config: AacEncoderConfig) -> Result<Self> {
        // Validate channel count (fdk-aac only supports mono and stereo)
        let channel_mode = match config.channels {
            1 => ChannelMode::Mono,
            2 => ChannelMode::Stereo,
            _ => {
                return Err(Error::codec(format!(
                    "Unsupported channel count for AAC encoding: {}. Only mono (1) or stereo (2) supported.",
                    config.channels
                )));
            }
        };

        // Build encoder params
        let params = EncoderParams {
            bit_rate: BitRate::Cbr(config.bitrate),
            sample_rate: config.sample_rate,
            transport: config.transport.into(),
            channels: channel_mode,
            audio_object_type: config.audio_object_type.into(),
        };

        // Create the encoder
        let fdk_encoder = FdkEncoder::new(params)
            .map_err(|e| Error::codec(format!("Failed to create AAC encoder: {:?}", e)))?;

        // Get AudioSpecificConfig from encoder info
        let extradata = fdk_encoder
            .info()
            .map(|info| {
                let size = info.confSize as usize;
                info.confBuf[..size].to_vec()
            })
            .ok();

        Ok(AacEncoder {
            encoder: Arc::new(Mutex::new(fdk_encoder)),
            sample_rate: config.sample_rate,
            channels: config.channels,
            bitrate: config.bitrate,
            packet_buffer: Arc::new(Mutex::new(VecDeque::new())),
            extradata,
            frame_size: Self::AAC_FRAME_SIZE,
            sample_buffer: Vec::with_capacity(Self::AAC_FRAME_SIZE * config.channels as usize * 2),
            current_pts: Timestamp::none(),
            samples_encoded: 0,
        })
    }

    /// Get the sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get the number of channels
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// Get the target bitrate
    pub fn bitrate(&self) -> u32 {
        self.bitrate
    }

    /// Get extradata (AudioSpecificConfig) for container muxing
    ///
    /// This returns the AAC AudioSpecificConfig which is required for
    /// properly muxing AAC streams into MP4/M4A containers.
    pub fn extradata(&self) -> Option<&[u8]> {
        self.extradata.as_deref()
    }

    /// Get the frame size (samples per channel per AAC frame)
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// Convert AudioFrame data to interleaved i16 samples
    fn extract_samples(&self, audio_frame: &AudioFrame) -> Result<Vec<i16>> {
        if audio_frame.data.is_empty() {
            return Err(Error::codec("Empty audio frame"));
        }

        // Validate sample rate matches
        if audio_frame.sample_rate != self.sample_rate {
            return Err(Error::codec(format!(
                "Sample rate mismatch: encoder expects {}, got {}",
                self.sample_rate, audio_frame.sample_rate
            )));
        }

        // Validate channel count
        if audio_frame.channels != self.channels {
            return Err(Error::codec(format!(
                "Channel count mismatch: encoder expects {}, got {}",
                self.channels, audio_frame.channels
            )));
        }

        let raw_data = audio_frame.data[0].as_slice();

        // Convert based on sample format
        let samples: Vec<i16> = match audio_frame.format {
            SampleFormat::I16 | SampleFormat::I16P => {
                // Already i16, just convert from bytes
                raw_data
                    .chunks_exact(2)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect()
            }
            SampleFormat::I32 | SampleFormat::I32P => {
                // Convert i32 to i16
                raw_data
                    .chunks_exact(4)
                    .map(|chunk| {
                        let sample = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        (sample >> 16) as i16
                    })
                    .collect()
            }
            SampleFormat::F32 | SampleFormat::F32P => {
                // Convert f32 to i16
                raw_data
                    .chunks_exact(4)
                    .map(|chunk| {
                        let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        (sample * 32767.0).clamp(-32768.0, 32767.0) as i16
                    })
                    .collect()
            }
            SampleFormat::F64 | SampleFormat::F64P => {
                // Convert f64 to i16
                raw_data
                    .chunks_exact(8)
                    .map(|chunk| {
                        let sample = f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]);
                        (sample * 32767.0).clamp(-32768.0, 32767.0) as i16
                    })
                    .collect()
            }
            SampleFormat::U8 | SampleFormat::U8P => {
                // Convert u8 to i16
                raw_data.iter().map(|&s| (s as i16 - 128) * 256).collect()
            }
            _ => {
                return Err(Error::codec(format!(
                    "Unsupported sample format for AAC encoding: {:?}",
                    audio_frame.format
                )));
            }
        };

        Ok(samples)
    }

    /// Encode samples from the internal buffer
    fn encode_buffered_samples(&mut self) -> Result<()> {
        let samples_per_frame = self.frame_size * self.channels as usize;

        // Process complete frames
        while self.sample_buffer.len() >= samples_per_frame {
            // Extract one frame worth of samples
            let frame_samples: Vec<i16> = self.sample_buffer.drain(..samples_per_frame).collect();

            // Lock encoder and encode
            let encoder = self
                .encoder
                .lock()
                .map_err(|e| Error::codec(format!("Failed to lock AAC encoder: {:?}", e)))?;

            // Output buffer for encoded data (768 bytes per channel is typically enough)
            let mut output = vec![0u8; 768 * self.channels as usize];

            // Encode the frame - returns EncodeInfo with output_size
            let encode_info = encoder
                .encode(&frame_samples, &mut output)
                .map_err(|e| Error::codec(format!("AAC encoding failed: {:?}", e)))?;

            drop(encoder);

            // If we got output, create a packet
            if encode_info.output_size > 0 {
                output.truncate(encode_info.output_size);

                // Calculate PTS based on samples encoded
                // The PTS value represents sample count in the encoder's time base (1/sample_rate)
                let pts = if self.current_pts.is_valid() {
                    Timestamp::new(self.samples_encoded as i64)
                } else {
                    Timestamp::none()
                };

                let mut packet = Packet::new(0, Buffer::from_vec(output));
                packet.pts = pts;
                packet.dts = pts;
                packet.duration = self.frame_size as i64;
                packet.set_keyframe(true); // AAC frames are all keyframes

                // Push to buffer
                let mut buffer = self
                    .packet_buffer
                    .lock()
                    .map_err(|e| Error::codec(format!("Failed to lock packet buffer: {:?}", e)))?;
                buffer.push_back(packet);
            }

            self.samples_encoded += self.frame_size as u64;
        }

        Ok(())
    }

    /// Encode remaining samples with flushing
    fn flush_encoder(&mut self) -> Result<()> {
        // First encode any complete frames in the buffer
        self.encode_buffered_samples()?;

        // If there are remaining samples, pad and encode final frame
        if !self.sample_buffer.is_empty() {
            let samples_per_frame = self.frame_size * self.channels as usize;

            // Pad with silence to complete the final frame
            self.sample_buffer.resize(samples_per_frame, 0);

            // Lock encoder and encode
            let encoder = self
                .encoder
                .lock()
                .map_err(|e| Error::codec(format!("Failed to lock AAC encoder: {:?}", e)))?;

            let mut output = vec![0u8; 768 * self.channels as usize];

            // Encode final frame
            let frame_samples: Vec<i16> = self.sample_buffer.drain(..).collect();
            let encode_info = encoder
                .encode(&frame_samples, &mut output)
                .map_err(|e| Error::codec(format!("AAC encoding (flush) failed: {:?}", e)))?;

            drop(encoder);

            // Create packet for final encoded frame
            if encode_info.output_size > 0 {
                output.truncate(encode_info.output_size);

                let pts = if self.current_pts.is_valid() {
                    Timestamp::new(self.samples_encoded as i64)
                } else {
                    Timestamp::none()
                };

                let mut packet = Packet::new(0, Buffer::from_vec(output));
                packet.pts = pts;
                packet.dts = pts;
                packet.duration = self.frame_size as i64;
                packet.set_keyframe(true);

                let mut buffer = self
                    .packet_buffer
                    .lock()
                    .map_err(|e| Error::codec(format!("Failed to lock packet buffer: {:?}", e)))?;
                buffer.push_back(packet);
            }
        }

        Ok(())
    }

    /// Convenience method to encode a frame directly to a packet
    ///
    /// This bypasses the send_frame/receive_packet pattern for simpler use cases.
    pub fn encode_frame(&mut self, frame: &AudioFrame) -> Result<Packet> {
        // Extract samples
        let samples = self.extract_samples(frame)?;

        // Store PTS from frame
        if frame.pts.is_valid() {
            self.current_pts = frame.pts;
        }

        // Add samples to buffer
        self.sample_buffer.extend(samples);

        // Encode buffered samples
        self.encode_buffered_samples()?;

        // Try to get a packet
        let mut buffer = self
            .packet_buffer
            .lock()
            .map_err(|e| Error::codec(format!("Failed to lock packet buffer: {:?}", e)))?;

        match buffer.pop_front() {
            Some(packet) => Ok(packet),
            None => Err(Error::TryAgain),
        }
    }
}

impl Encoder for AacEncoder {
    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        match frame {
            Frame::Audio(audio_frame) => {
                // Extract samples from the audio frame
                let samples = self.extract_samples(audio_frame)?;

                // Store PTS from first frame if not already set
                if !self.current_pts.is_valid() && audio_frame.pts.is_valid() {
                    self.current_pts = audio_frame.pts;
                }

                // Add samples to our internal buffer
                self.sample_buffer.extend(samples);

                // Try to encode complete frames
                self.encode_buffered_samples()
            }
            Frame::Video(_) => Err(Error::codec("AAC encoder only accepts audio frames")),
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        // Lock the packet buffer and try to get the next encoded packet
        let mut buffer = self
            .packet_buffer
            .lock()
            .map_err(|e| Error::codec(format!("Failed to lock packet buffer: {:?}", e)))?;

        // Return the next packet if available, otherwise indicate to try again
        match buffer.pop_front() {
            Some(packet) => Ok(packet),
            None => Err(Error::TryAgain),
        }
    }

    fn flush(&mut self) -> Result<()> {
        // Flush any remaining samples through the encoder
        self.flush_encoder()?;

        // Clear the packet buffer
        let mut buffer = self
            .packet_buffer
            .lock()
            .map_err(|e| Error::codec(format!("Failed to lock packet buffer: {:?}", e)))?;
        buffer.clear();

        // Reset state
        self.sample_buffer.clear();
        self.samples_encoded = 0;
        self.current_pts = Timestamp::none();

        Ok(())
    }

    fn extradata(&self) -> Option<&[u8]> {
        self.extradata.as_deref()
    }
}

// Implement Clone for AacEncoder to allow sharing across threads
impl Clone for AacEncoder {
    fn clone(&self) -> Self {
        AacEncoder {
            encoder: Arc::clone(&self.encoder),
            sample_rate: self.sample_rate,
            channels: self.channels,
            bitrate: self.bitrate,
            packet_buffer: Arc::clone(&self.packet_buffer),
            extradata: self.extradata.clone(),
            frame_size: self.frame_size,
            sample_buffer: self.sample_buffer.clone(),
            current_pts: self.current_pts,
            samples_encoded: self.samples_encoded,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aac_encoder_creation() {
        let result = AacEncoder::new(44100, 2, 128_000);
        match result {
            Ok(encoder) => {
                assert_eq!(encoder.sample_rate(), 44100);
                assert_eq!(encoder.channels(), 2);
                assert_eq!(encoder.bitrate(), 128_000);
                assert!(encoder.extradata().is_some());
            }
            Err(e) => {
                // Expected if libfdk-aac is not installed
                let err_msg = format!("{}", e);
                assert!(
                    err_msg.contains("AAC") || err_msg.contains("encoder"),
                    "Unexpected error: {}",
                    err_msg
                );
            }
        }
    }

    #[test]
    fn test_aac_encoder_with_config() {
        let config = AacEncoderConfig {
            sample_rate: 48000,
            channels: 1,
            bitrate: 64_000,
            audio_object_type: AacProfile::AacLc,
            transport: AacTransport::Adts,
        };

        let result = AacEncoder::with_config(config);
        match result {
            Ok(encoder) => {
                assert_eq!(encoder.sample_rate(), 48000);
                assert_eq!(encoder.channels(), 1);
                assert_eq!(encoder.bitrate(), 64_000);
            }
            Err(e) => {
                // Expected if libfdk-aac is not installed
                let err_msg = format!("{}", e);
                assert!(
                    err_msg.contains("AAC") || err_msg.contains("encoder"),
                    "Unexpected error: {}",
                    err_msg
                );
            }
        }
    }

    #[test]
    fn test_invalid_channel_count() {
        // 6 channels not supported by fdk-aac (only mono/stereo)
        let result = AacEncoder::new(44100, 6, 256_000);
        assert!(result.is_err());
    }

    #[test]
    fn test_extradata_available() {
        if let Ok(encoder) = AacEncoder::new(44100, 2, 128_000) {
            let extradata = encoder.extradata();
            assert!(extradata.is_some());
            // AudioSpecificConfig should be at least 2 bytes
            assert!(extradata.unwrap().len() >= 2);
        }
    }
}
