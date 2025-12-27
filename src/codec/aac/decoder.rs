//! AAC decoder using Symphonia
//!
//! This module provides AAC-LC (Low Complexity) decoding via Symphonia's AAC codec.
//! HE-AAC (High Efficiency AAC with SBR/PS) is not currently supported by Symphonia.

use crate::codec::{AudioFrame, Decoder, Frame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, SampleFormat, Timestamp};

use std::collections::VecDeque;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CodecParameters, DecoderOptions, CODEC_TYPE_AAC};
use symphonia::core::formats::Packet as SymphoniaPacket;
use symphonia::default::get_codecs;

/// AAC Audio Object Types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AacProfile {
    /// AAC Main profile
    Main = 1,
    /// AAC Low Complexity (LC) - most common
    LowComplexity = 2,
    /// AAC Scalable Sample Rate
    Ssr = 3,
    /// AAC Long Term Prediction
    Ltp = 4,
    /// High Efficiency AAC (HE-AAC v1, uses SBR)
    HeAacV1 = 5,
    /// High Efficiency AAC v2 (HE-AAC v2, uses SBR + PS)
    HeAacV2 = 29,
    /// Unknown or unsupported profile
    Unknown = 0,
}

impl AacProfile {
    /// Parse AAC profile from AudioSpecificConfig
    fn from_object_type(object_type: u8) -> Self {
        match object_type {
            1 => AacProfile::Main,
            2 => AacProfile::LowComplexity,
            3 => AacProfile::Ssr,
            4 => AacProfile::Ltp,
            5 => AacProfile::HeAacV1,
            29 => AacProfile::HeAacV2,
            _ => AacProfile::Unknown,
        }
    }

    /// Check if this profile is supported by Symphonia
    ///
    /// Only AAC-LC, AAC Main, and AAC-LTP are supported.
    /// HE-AAC (SBR) and HE-AAC v2 (PS) are NOT supported.
    pub fn is_supported(&self) -> bool {
        matches!(
            self,
            AacProfile::Main | AacProfile::LowComplexity | AacProfile::Ltp
        )
    }
}

/// Sample rate index to frequency mapping per ISO/IEC 14496-3
const SAMPLE_RATE_TABLE: [u32; 16] = [
    96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350, 0, 0,
    0, // 13-15 are reserved/escape
];

/// Parsed AudioSpecificConfig
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields kept for future HE-AAC support
struct AudioSpecificConfig {
    object_type: AacProfile,
    sample_rate: u32,
    channels: u16,
    /// Whether SBR (Spectral Band Replication) is signaled
    sbr_present: bool,
    /// Whether PS (Parametric Stereo) is signaled
    ps_present: bool,
}

impl AudioSpecificConfig {
    /// Parse AudioSpecificConfig from raw bytes
    /// Format per ISO/IEC 14496-3:
    ///   5 bits: audioObjectType
    ///   4 bits: samplingFrequencyIndex
    ///   4 bits: channelConfiguration
    ///   (optional SBR/PS extensions for HE-AAC)
    fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 2 {
            return Err(Error::codec(
                "AudioSpecificConfig too short (need at least 2 bytes)",
            ));
        }

        // Read first 2 bytes
        let b0 = data[0];
        let b1 = data[1];

        // audioObjectType: 5 bits
        let mut object_type = (b0 >> 3) as u8;

        // samplingFrequencyIndex: 4 bits (3 bits from b0, 1 bit from b1)
        let sample_rate_index = ((b0 & 0x07) << 1) | (b1 >> 7);

        // channelConfiguration: 4 bits
        let channel_config = (b1 >> 3) & 0x0F;

        // Get actual sample rate
        let sample_rate = if sample_rate_index < 13 {
            SAMPLE_RATE_TABLE[sample_rate_index as usize]
        } else if sample_rate_index == 15 {
            // Escape value: 24-bit sample rate follows
            if data.len() < 5 {
                return Err(Error::codec(
                    "AudioSpecificConfig missing explicit sample rate",
                ));
            }
            let sr = ((data[1] as u32 & 0x7F) << 17)
                | ((data[2] as u32) << 9)
                | ((data[3] as u32) << 1)
                | ((data[4] as u32) >> 7);
            sr
        } else {
            return Err(Error::codec(format!(
                "Reserved sample rate index: {}",
                sample_rate_index
            )));
        };

        // Map channel configuration to channel count
        let channels = match channel_config {
            0 => 0, // Defined in program_config_element
            1 => 1, // center front
            2 => 2, // left, right front
            3 => 3, // center, left, right front
            4 => 4, // center, left, right front, rear center
            5 => 5, // center, left, right front, left, right rear
            6 => 6, // center, left, right front, left, right rear, LFE
            7 => 8, // 7.1 surround
            _ => {
                return Err(Error::codec(format!(
                    "Invalid channel configuration: {}",
                    channel_config
                )));
            }
        };

        // Check for SBR/PS explicit signaling (HE-AAC)
        // This is a simplified check - full parsing would require more bytes
        let sbr_present = object_type == 5 || object_type == 29;
        let ps_present = object_type == 29;

        // Handle extended audioObjectType (if 31)
        if object_type == 31 && data.len() >= 3 {
            object_type = 32 + ((b1 & 0x07) << 3 | (data[2] >> 5));
        }

        let profile = AacProfile::from_object_type(object_type);

        Ok(AudioSpecificConfig {
            object_type: profile,
            sample_rate,
            channels,
            sbr_present,
            ps_present,
        })
    }
}

/// AAC decoder wrapping Symphonia's AAC codec
pub struct AacDecoder {
    /// Symphonia AAC decoder
    decoder: Box<dyn symphonia::core::codecs::Decoder>,
    /// Sample buffer for decoded audio
    sample_buffer: Option<SampleBuffer<f32>>,
    /// Queue of decoded frames ready to be returned
    pending_frames: VecDeque<AudioFrame>,
    /// Packets waiting to be decoded
    pending_packets: VecDeque<PendingPacket>,
    /// Sample rate
    sample_rate: u32,
    /// Number of channels
    channels: u16,
    /// Track ID for Symphonia packets
    track_id: u32,
    /// Current presentation timestamp
    current_pts: i64,
    /// Samples decoded so far (for timestamp calculation)
    samples_decoded: u64,
    /// AAC profile detected
    profile: Option<AacProfile>,
}

impl std::fmt::Debug for AacDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AacDecoder")
            .field("sample_rate", &self.sample_rate)
            .field("channels", &self.channels)
            .field("track_id", &self.track_id)
            .field("current_pts", &self.current_pts)
            .field("samples_decoded", &self.samples_decoded)
            .field("profile", &self.profile)
            .field("pending_packets_count", &self.pending_packets.len())
            .field("pending_frames_count", &self.pending_frames.len())
            .finish_non_exhaustive()
    }
}

/// A packet waiting to be decoded
struct PendingPacket {
    data: Vec<u8>,
    pts: Timestamp,
    duration: i64,
}

impl AacDecoder {
    /// Create a new AAC decoder
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        Self::create_decoder(sample_rate, channels, None)
    }

    /// Create decoder with extradata (codec-specific configuration)
    ///
    /// The extradata typically contains AudioSpecificConfig for AAC,
    /// which provides profile, sample rate, and channel configuration.
    pub fn with_extradata(sample_rate: u32, channels: u16, extradata: &[u8]) -> Result<Self> {
        // Parse AudioSpecificConfig if provided
        if !extradata.is_empty() {
            let asc = AudioSpecificConfig::parse(extradata)?;

            // Check for unsupported HE-AAC profiles
            if !asc.object_type.is_supported() {
                return Err(Error::unsupported(format!(
                    "AAC profile {:?} is not supported. Only AAC-LC, AAC Main, and AAC-LTP are supported. \
                     HE-AAC v1 (SBR) and HE-AAC v2 (SBR+PS) require additional tools not available in Symphonia.",
                    asc.object_type
                )));
            }

            // Use parsed values, preferring extradata over provided parameters
            let actual_sample_rate = if asc.sample_rate > 0 {
                asc.sample_rate
            } else {
                sample_rate
            };
            let actual_channels = if asc.channels > 0 {
                asc.channels
            } else {
                channels
            };

            return Self::create_decoder(actual_sample_rate, actual_channels, Some(extradata));
        }

        Self::create_decoder(sample_rate, channels, None)
    }

    /// Internal decoder creation
    fn create_decoder(
        sample_rate: u32,
        channels: u16,
        extradata: Option<&[u8]>,
    ) -> Result<Self> {
        // Build codec parameters for AAC
        let mut codec_params = CodecParameters::new();
        codec_params
            .for_codec(CODEC_TYPE_AAC)
            .with_sample_rate(sample_rate);

        // Set channel layout
        use symphonia::core::audio::Channels;
        let channel_layout = match channels {
            1 => Channels::FRONT_CENTRE,
            2 => Channels::FRONT_LEFT | Channels::FRONT_RIGHT,
            3 => Channels::FRONT_CENTRE | Channels::FRONT_LEFT | Channels::FRONT_RIGHT,
            4 => {
                Channels::FRONT_CENTRE
                    | Channels::FRONT_LEFT
                    | Channels::FRONT_RIGHT
                    | Channels::REAR_CENTRE
            }
            5 => {
                Channels::FRONT_CENTRE
                    | Channels::FRONT_LEFT
                    | Channels::FRONT_RIGHT
                    | Channels::REAR_LEFT
                    | Channels::REAR_RIGHT
            }
            6 => {
                Channels::FRONT_CENTRE
                    | Channels::FRONT_LEFT
                    | Channels::FRONT_RIGHT
                    | Channels::REAR_LEFT
                    | Channels::REAR_RIGHT
                    | Channels::LFE1
            }
            8 => {
                Channels::FRONT_CENTRE
                    | Channels::FRONT_LEFT
                    | Channels::FRONT_RIGHT
                    | Channels::SIDE_LEFT
                    | Channels::SIDE_RIGHT
                    | Channels::REAR_LEFT
                    | Channels::REAR_RIGHT
                    | Channels::LFE1
            }
            _ => {
                // For non-standard channel counts, use stereo layout
                Channels::FRONT_LEFT | Channels::FRONT_RIGHT
            }
        };
        codec_params.with_channels(channel_layout);

        // Set extradata if provided
        if let Some(extra) = extradata {
            codec_params.with_extra_data(extra.to_vec().into_boxed_slice());
        }

        // Create the Symphonia AAC decoder using the default registry
        // The default registry includes AAC when the 'aac' feature is enabled on symphonia
        let decoder = get_codecs()
            .make(&codec_params, &DecoderOptions::default())
            .map_err(|e| {
                Error::codec(format!(
                    "Failed to create AAC decoder: {}. Make sure the 'aac' feature is enabled.",
                    e
                ))
            })?;

        // Parse profile from extradata if available
        let profile = extradata
            .and_then(|e| AudioSpecificConfig::parse(e).ok())
            .map(|asc| asc.object_type);

        Ok(AacDecoder {
            decoder,
            sample_buffer: None,
            pending_frames: VecDeque::new(),
            pending_packets: VecDeque::new(),
            sample_rate,
            channels,
            track_id: 0,
            current_pts: 0,
            samples_decoded: 0,
            profile,
        })
    }

    /// Get the detected AAC profile
    pub fn profile(&self) -> Option<AacProfile> {
        self.profile
    }

    /// Get the sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get the number of channels
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// Decode a single packet and return an AudioFrame
    fn decode_packet_internal(&mut self, pending: &PendingPacket) -> Result<AudioFrame> {
        // Create a Symphonia packet from the raw data
        let symphonia_packet = SymphoniaPacket::new_from_slice(
            self.track_id,
            pending.pts.value as u64,
            pending.duration as u64,
            &pending.data,
        );

        // Decode the packet
        let decoded = self.decoder.decode(&symphonia_packet).map_err(|e| {
            Error::codec(format!("AAC decode error: {}", e))
        })?;

        // Get audio specification from decoded buffer
        let spec = *decoded.spec();
        let num_frames = decoded.frames();

        // Initialize or reuse sample buffer
        if self.sample_buffer.is_none()
            || self.sample_buffer.as_ref().unwrap().capacity() < num_frames
        {
            self.sample_buffer = Some(SampleBuffer::<f32>::new(num_frames as u64, spec));
        }

        let sample_buffer = self.sample_buffer.as_mut().unwrap();

        // Copy decoded audio to sample buffer (converts to f32 interleaved)
        sample_buffer.copy_interleaved_ref(decoded);

        // Get the samples
        let samples = sample_buffer.samples();
        let actual_channels = spec.channels.count() as u16;
        let samples_per_channel = samples.len() / actual_channels as usize;

        // Update decoder state if needed
        if self.channels != actual_channels {
            self.channels = actual_channels;
        }
        if let Some(sr) = spec.rate.checked_mul(1) {
            if self.sample_rate != sr {
                self.sample_rate = sr;
            }
        }

        // Convert f32 samples to bytes
        // We output as F32 (32-bit float) interleaved format
        let byte_data: Vec<u8> = samples
            .iter()
            .flat_map(|&sample| sample.to_le_bytes())
            .collect();

        // Create AudioFrame
        let mut audio_frame = AudioFrame::new(
            samples_per_channel,
            self.sample_rate,
            self.channels,
            SampleFormat::F32,
        );

        audio_frame.pts = pending.pts;
        audio_frame.duration = pending.duration;
        audio_frame.data = vec![Buffer::from_vec(byte_data)];

        // Update samples decoded counter
        self.samples_decoded += samples_per_channel as u64;

        Ok(audio_frame)
    }
}

impl Decoder for AacDecoder {
    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        // Validate packet
        if packet.data.is_empty() {
            return Err(Error::codec("Empty AAC packet"));
        }

        // Queue the packet for decoding
        self.pending_packets.push_back(PendingPacket {
            data: packet.data.as_slice().to_vec(),
            pts: packet.pts,
            duration: packet.duration,
        });

        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        // First check if we have any pending decoded frames
        if let Some(frame) = self.pending_frames.pop_front() {
            return Ok(Frame::Audio(frame));
        }

        // Try to decode a pending packet
        if let Some(pending) = self.pending_packets.pop_front() {
            match self.decode_packet_internal(&pending) {
                Ok(audio_frame) => {
                    return Ok(Frame::Audio(audio_frame));
                }
                Err(e) => {
                    // Log error but don't fail - try next packet
                    // This handles corrupt frames gracefully
                    if !self.pending_packets.is_empty() {
                        // Try the next packet
                        return self.receive_frame();
                    }
                    return Err(e);
                }
            }
        }

        // No frames available
        Err(Error::TryAgain)
    }

    fn flush(&mut self) -> Result<()> {
        // Reset decoder state
        self.decoder.reset();

        // Clear all pending data
        self.pending_frames.clear();
        self.pending_packets.clear();
        self.sample_buffer = None;
        self.current_pts = 0;
        self.samples_decoded = 0;

        Ok(())
    }
}

impl AacDecoder {
    /// Convenience method to decode a packet directly to a frame
    ///
    /// This combines send_packet and receive_frame for simple use cases.
    pub fn decode_packet(&mut self, packet: &Packet) -> Result<Frame> {
        self.send_packet(packet)?;
        self.receive_frame()
    }

    /// Check if a profile is supported
    pub fn is_profile_supported(profile: AacProfile) -> bool {
        profile.is_supported()
    }

    /// Get supported profiles as a list
    pub fn supported_profiles() -> Vec<AacProfile> {
        vec![
            AacProfile::Main,
            AacProfile::LowComplexity,
            AacProfile::Ltp,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aac_decoder_creation() {
        let decoder = AacDecoder::new(44100, 2);
        assert!(decoder.is_ok());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.sample_rate(), 44100);
        assert_eq!(decoder.channels(), 2);
    }

    #[test]
    fn test_aac_decoder_48khz_stereo() {
        let decoder = AacDecoder::new(48000, 2);
        assert!(decoder.is_ok());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.sample_rate(), 48000);
    }

    #[test]
    fn test_aac_decoder_mono() {
        let decoder = AacDecoder::new(44100, 1);
        assert!(decoder.is_ok());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.channels(), 1);
    }

    #[test]
    fn test_audio_specific_config_parsing() {
        // AAC-LC, 44100 Hz, stereo
        // Object type: 2 (LC), Sample rate index: 4 (44100), Channels: 2
        // Binary: 00010 0100 0010 000 = 0x12 0x10
        let asc_data = [0x12, 0x10];
        let asc = AudioSpecificConfig::parse(&asc_data);
        assert!(asc.is_ok());
        let asc = asc.unwrap();
        assert_eq!(asc.object_type, AacProfile::LowComplexity);
        assert_eq!(asc.sample_rate, 44100);
        assert_eq!(asc.channels, 2);
        assert!(!asc.sbr_present);
        assert!(!asc.ps_present);
    }

    #[test]
    fn test_audio_specific_config_48khz() {
        // AAC-LC, 48000 Hz, stereo
        // Object type: 2 (LC), Sample rate index: 3 (48000), Channels: 2
        // Binary: 00010 0011 0010 000 = 0x11 0x90
        let asc_data = [0x11, 0x90];
        let asc = AudioSpecificConfig::parse(&asc_data);
        assert!(asc.is_ok());
        let asc = asc.unwrap();
        assert_eq!(asc.object_type, AacProfile::LowComplexity);
        assert_eq!(asc.sample_rate, 48000);
        assert_eq!(asc.channels, 2);
    }

    #[test]
    fn test_aac_profile_supported() {
        assert!(AacProfile::LowComplexity.is_supported());
        assert!(AacProfile::Main.is_supported());
        assert!(AacProfile::Ltp.is_supported());
        assert!(!AacProfile::HeAacV1.is_supported());
        assert!(!AacProfile::HeAacV2.is_supported());
        assert!(!AacProfile::Ssr.is_supported());
    }

    #[test]
    fn test_aac_decoder_with_extradata() {
        // AAC-LC, 44100 Hz, stereo
        let asc_data = [0x12, 0x10];
        let decoder = AacDecoder::with_extradata(44100, 2, &asc_data);
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_he_aac_rejected() {
        // HE-AAC v1 (SBR), 44100 Hz, stereo
        // Object type: 5 (HE-AAC), Sample rate index: 4 (44100), Channels: 2
        // Binary: 00101 0100 0010 000 = 0x2A 0x10
        let asc_data = [0x2A, 0x10];
        let decoder = AacDecoder::with_extradata(44100, 2, &asc_data);
        assert!(decoder.is_err());

        let err = decoder.unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }

    #[test]
    fn test_empty_packet_rejected() {
        let mut decoder = AacDecoder::new(44100, 2).unwrap();
        let packet = Packet::new_audio(0, Buffer::empty());
        let result = decoder.send_packet(&packet);
        assert!(result.is_err());
    }

    #[test]
    fn test_flush() {
        let mut decoder = AacDecoder::new(44100, 2).unwrap();
        let result = decoder.flush();
        assert!(result.is_ok());
    }

    #[test]
    fn test_receive_without_send() {
        let mut decoder = AacDecoder::new(44100, 2).unwrap();
        let result = decoder.receive_frame();
        // Should return TryAgain since no packets were sent
        assert!(matches!(result, Err(Error::TryAgain)));
    }

    #[test]
    fn test_sample_rate_table() {
        // Verify the sample rate table values
        assert_eq!(SAMPLE_RATE_TABLE[0], 96000);
        assert_eq!(SAMPLE_RATE_TABLE[3], 48000);
        assert_eq!(SAMPLE_RATE_TABLE[4], 44100);
        assert_eq!(SAMPLE_RATE_TABLE[11], 8000);
    }

    #[test]
    fn test_supported_profiles_list() {
        let profiles = AacDecoder::supported_profiles();
        assert_eq!(profiles.len(), 3);
        assert!(profiles.contains(&AacProfile::Main));
        assert!(profiles.contains(&AacProfile::LowComplexity));
        assert!(profiles.contains(&AacProfile::Ltp));
    }
}
