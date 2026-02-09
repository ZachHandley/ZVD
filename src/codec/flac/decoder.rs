//! FLAC audio decoder using Symphonia
//!
//! This module provides standalone FLAC decoding that can decode individual
//! packets without requiring a container/demuxer. This is essential for
//! network streaming scenarios where encoded packets arrive independently.

use crate::codec::{AudioFrame, Decoder, Frame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, SampleFormat, Timestamp};

use std::collections::VecDeque;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CodecParameters, DecoderOptions, CODEC_TYPE_FLAC};
use symphonia::core::formats::Packet as SymphoniaPacket;
use symphonia::default::get_codecs;

/// FLAC stream info parsed from extradata
#[derive(Debug, Clone)]
pub struct FlacStreamInfo {
    /// Minimum block size (samples) used in the stream
    pub min_block_size: u16,
    /// Maximum block size (samples) used in the stream
    pub max_block_size: u16,
    /// Minimum frame size (bytes)
    pub min_frame_size: u32,
    /// Maximum frame size (bytes)
    pub max_frame_size: u32,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels (1-8)
    pub channels: u8,
    /// Bits per sample (4-32)
    pub bits_per_sample: u8,
    /// Total number of samples (can be 0 if unknown)
    pub total_samples: u64,
    /// MD5 signature of unencoded audio data
    pub md5_signature: [u8; 16],
}

impl FlacStreamInfo {
    /// Parse FLAC STREAMINFO metadata block from raw bytes
    ///
    /// The STREAMINFO block is always 34 bytes and contains:
    /// - 16 bits: minimum block size
    /// - 16 bits: maximum block size
    /// - 24 bits: minimum frame size
    /// - 24 bits: maximum frame size
    /// - 20 bits: sample rate
    /// - 3 bits: channels - 1
    /// - 5 bits: bits per sample - 1
    /// - 36 bits: total samples
    /// - 128 bits: MD5 signature
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 34 {
            return Err(Error::codec(format!(
                "FLAC STREAMINFO too short: {} bytes (need 34)",
                data.len()
            )));
        }

        // Parse minimum block size (16 bits)
        let min_block_size = u16::from_be_bytes([data[0], data[1]]);

        // Parse maximum block size (16 bits)
        let max_block_size = u16::from_be_bytes([data[2], data[3]]);

        // Parse minimum frame size (24 bits)
        let min_frame_size = u32::from_be_bytes([0, data[4], data[5], data[6]]);

        // Parse maximum frame size (24 bits)
        let max_frame_size = u32::from_be_bytes([0, data[7], data[8], data[9]]);

        // Parse sample rate (20 bits), channels (3 bits), bits per sample (5 bits)
        // Bytes 10-13 contain: SSSSSSSS SSSSSSSS SSSSCCCC CBBBBBTT
        // where S = sample rate, C = channels - 1, B = bits per sample - 1, T = total samples MSB
        let sample_rate = ((data[10] as u32) << 12) | ((data[11] as u32) << 4) | ((data[12] as u32) >> 4);

        let channels = ((data[12] >> 1) & 0x07) + 1;

        let bits_per_sample = (((data[12] & 0x01) << 4) | ((data[13] >> 4) & 0x0F)) + 1;

        // Parse total samples (36 bits) - 4 bits from byte 13, 32 bits from bytes 14-17
        let total_samples = (((data[13] & 0x0F) as u64) << 32)
            | ((data[14] as u64) << 24)
            | ((data[15] as u64) << 16)
            | ((data[16] as u64) << 8)
            | (data[17] as u64);

        // Parse MD5 signature (128 bits = 16 bytes)
        let mut md5_signature = [0u8; 16];
        md5_signature.copy_from_slice(&data[18..34]);

        Ok(FlacStreamInfo {
            min_block_size,
            max_block_size,
            min_frame_size,
            max_frame_size,
            sample_rate,
            channels,
            bits_per_sample,
            total_samples,
            md5_signature,
        })
    }

    /// Validate the stream info parameters
    pub fn validate(&self) -> Result<()> {
        if self.sample_rate == 0 || self.sample_rate > 655350 {
            return Err(Error::codec(format!(
                "Invalid FLAC sample rate: {}",
                self.sample_rate
            )));
        }

        if self.channels == 0 || self.channels > 8 {
            return Err(Error::codec(format!(
                "Invalid FLAC channel count: {}",
                self.channels
            )));
        }

        if self.bits_per_sample < 4 || self.bits_per_sample > 32 {
            return Err(Error::codec(format!(
                "Invalid FLAC bits per sample: {}",
                self.bits_per_sample
            )));
        }

        if self.min_block_size < 16 {
            return Err(Error::codec(format!(
                "Invalid FLAC minimum block size: {}",
                self.min_block_size
            )));
        }

        Ok(())
    }
}

/// A pending packet waiting to be decoded
struct PendingPacket {
    /// Raw packet data
    data: Vec<u8>,
    /// Presentation timestamp
    pts: Timestamp,
    /// Duration in samples
    duration: i64,
}

/// FLAC audio decoder using Symphonia
///
/// This decoder wraps Symphonia's FLAC codec to provide standalone packet
/// decoding without requiring a container format. It follows the ZVD Decoder
/// trait interface with send_packet/receive_frame semantics.
///
/// ## Example
///
/// ```rust,ignore
/// use zvd_lib::codec::{Decoder, FlacDecoder};
/// use zvd_lib::format::Packet;
///
/// let mut decoder = FlacDecoder::new(44100, 2)?;
/// decoder.send_packet(&packet)?;
/// let frame = decoder.receive_frame()?;
/// ```
pub struct FlacDecoder {
    /// Symphonia FLAC decoder instance
    decoder: Box<dyn symphonia::core::codecs::Decoder>,
    /// Sample buffer for converting decoded audio
    sample_buffer: Option<SampleBuffer<f32>>,
    /// Queue of packets waiting to be decoded
    pending_packets: VecDeque<PendingPacket>,
    /// Sample rate in Hz
    sample_rate: u32,
    /// Number of audio channels
    channels: u16,
    /// Bits per sample (16, 24, or 32)
    bits_per_sample: u8,
    /// Track ID for Symphonia packets
    track_id: u32,
    /// Total samples decoded (for timestamp calculation)
    samples_decoded: u64,
    /// Parsed stream info (if extradata was provided)
    stream_info: Option<FlacStreamInfo>,
}

impl std::fmt::Debug for FlacDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlacDecoder")
            .field("sample_rate", &self.sample_rate)
            .field("channels", &self.channels)
            .field("bits_per_sample", &self.bits_per_sample)
            .field("track_id", &self.track_id)
            .field("samples_decoded", &self.samples_decoded)
            .field("pending_packets_count", &self.pending_packets.len())
            .field("stream_info", &self.stream_info)
            .finish_non_exhaustive()
    }
}

impl FlacDecoder {
    /// Create a new FLAC decoder with specified parameters
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz (e.g., 44100, 48000, 96000)
    /// * `channels` - Number of audio channels (1-8)
    ///
    /// # Returns
    /// A configured FLAC decoder ready to decode packets
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        Self::create_decoder(sample_rate, channels, 16, None)
    }

    /// Create a new FLAC decoder with full configuration
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of audio channels
    /// * `bits_per_sample` - Bit depth (16, 24, or 32)
    pub fn with_bits_per_sample(
        sample_rate: u32,
        channels: u16,
        bits_per_sample: u8,
    ) -> Result<Self> {
        Self::create_decoder(sample_rate, channels, bits_per_sample, None)
    }

    /// Create a decoder with extradata (FLAC STREAMINFO metadata block)
    ///
    /// The extradata should contain the 34-byte STREAMINFO metadata block
    /// which provides precise stream parameters.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz (can be overridden by extradata)
    /// * `channels` - Number of channels (can be overridden by extradata)
    /// * `extradata` - FLAC STREAMINFO metadata block (34 bytes)
    pub fn with_extradata(sample_rate: u32, channels: u16, extradata: &[u8]) -> Result<Self> {
        // Parse STREAMINFO from extradata if it's valid
        if extradata.len() >= 34 {
            let stream_info = FlacStreamInfo::parse(extradata)?;
            stream_info.validate()?;

            // Use values from STREAMINFO
            let actual_sample_rate = stream_info.sample_rate;
            let actual_channels = stream_info.channels as u16;
            let bits_per_sample = stream_info.bits_per_sample;

            return Self::create_decoder(
                actual_sample_rate,
                actual_channels,
                bits_per_sample,
                Some(extradata),
            );
        }

        // Fall back to provided parameters
        Self::create_decoder(sample_rate, channels, 16, Some(extradata))
    }

    /// Internal decoder creation
    fn create_decoder(
        sample_rate: u32,
        channels: u16,
        bits_per_sample: u8,
        extradata: Option<&[u8]>,
    ) -> Result<Self> {
        // Build codec parameters for FLAC
        let mut codec_params = CodecParameters::new();
        codec_params
            .for_codec(CODEC_TYPE_FLAC)
            .with_sample_rate(sample_rate)
            .with_bits_per_coded_sample(bits_per_sample as u32);

        // Set channel layout based on channel count
        use symphonia::core::audio::Channels;
        let channel_layout = match channels {
            1 => Channels::FRONT_CENTRE,
            2 => Channels::FRONT_LEFT | Channels::FRONT_RIGHT,
            3 => Channels::FRONT_LEFT | Channels::FRONT_RIGHT | Channels::FRONT_CENTRE,
            4 => {
                Channels::FRONT_LEFT
                    | Channels::FRONT_RIGHT
                    | Channels::REAR_LEFT
                    | Channels::REAR_RIGHT
            }
            5 => {
                Channels::FRONT_LEFT
                    | Channels::FRONT_RIGHT
                    | Channels::FRONT_CENTRE
                    | Channels::REAR_LEFT
                    | Channels::REAR_RIGHT
            }
            6 => {
                Channels::FRONT_LEFT
                    | Channels::FRONT_RIGHT
                    | Channels::FRONT_CENTRE
                    | Channels::LFE1
                    | Channels::REAR_LEFT
                    | Channels::REAR_RIGHT
            }
            7 => {
                Channels::FRONT_LEFT
                    | Channels::FRONT_RIGHT
                    | Channels::FRONT_CENTRE
                    | Channels::LFE1
                    | Channels::REAR_CENTRE
                    | Channels::SIDE_LEFT
                    | Channels::SIDE_RIGHT
            }
            8 => {
                Channels::FRONT_LEFT
                    | Channels::FRONT_RIGHT
                    | Channels::FRONT_CENTRE
                    | Channels::LFE1
                    | Channels::REAR_LEFT
                    | Channels::REAR_RIGHT
                    | Channels::SIDE_LEFT
                    | Channels::SIDE_RIGHT
            }
            _ => Channels::FRONT_LEFT | Channels::FRONT_RIGHT,
        };
        codec_params.with_channels(channel_layout);

        // Set extradata if provided
        if let Some(extra) = extradata {
            codec_params.with_extra_data(extra.to_vec().into_boxed_slice());
        }

        // Create the Symphonia FLAC decoder
        let decoder = get_codecs()
            .make(&codec_params, &DecoderOptions::default())
            .map_err(|e| {
                Error::codec(format!(
                    "Failed to create FLAC decoder: {}. Ensure symphonia 'flac' feature is enabled.",
                    e
                ))
            })?;

        // Parse stream info from extradata if available
        let stream_info = extradata
            .filter(|e| e.len() >= 34)
            .and_then(|e| FlacStreamInfo::parse(e).ok());

        Ok(FlacDecoder {
            decoder,
            sample_buffer: None,
            pending_packets: VecDeque::new(),
            sample_rate,
            channels,
            bits_per_sample,
            track_id: 0,
            samples_decoded: 0,
            stream_info,
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

    /// Get bits per sample
    pub fn bits_per_sample(&self) -> u8 {
        self.bits_per_sample
    }

    /// Get parsed stream info (if extradata was provided)
    pub fn stream_info(&self) -> Option<&FlacStreamInfo> {
        self.stream_info.as_ref()
    }

    /// Get total samples decoded so far
    pub fn samples_decoded(&self) -> u64 {
        self.samples_decoded
    }

    /// Decode a single packet internally
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
            Error::codec(format!("FLAC decode error: {}", e))
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

        // Update decoder state if format changed
        if self.channels != actual_channels {
            self.channels = actual_channels;
        }
        if let Some(sr) = spec.rate.checked_mul(1) {
            if self.sample_rate != sr {
                self.sample_rate = sr;
            }
        }

        // Convert f32 samples to bytes (output as F32 interleaved)
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

    /// Convenience method to decode a packet directly
    ///
    /// Combines send_packet and receive_frame for simple use cases.
    pub fn decode_packet(&mut self, packet: &Packet) -> Result<Frame> {
        self.send_packet(packet)?;
        self.receive_frame()
    }
}

impl Decoder for FlacDecoder {
    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        // Validate packet
        if packet.data.is_empty() {
            return Err(Error::codec("Empty FLAC packet"));
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
        // Try to decode a pending packet
        if let Some(pending) = self.pending_packets.pop_front() {
            match self.decode_packet_internal(&pending) {
                Ok(audio_frame) => {
                    return Ok(Frame::Audio(audio_frame));
                }
                Err(e) => {
                    // For corrupted frames, try next packet if available
                    if !self.pending_packets.is_empty() {
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
        self.pending_packets.clear();
        self.sample_buffer = None;
        self.samples_decoded = 0;

        Ok(())
    }
}

/// Create a FLAC decoder with default parameters
///
/// This is a convenience factory function.
pub fn create_decoder(sample_rate: u32, channels: u16) -> Result<FlacDecoder> {
    FlacDecoder::new(sample_rate, channels)
}

/// Create a FLAC decoder with extradata
///
/// The extradata should contain the FLAC STREAMINFO metadata block.
pub fn create_decoder_with_extradata(
    sample_rate: u32,
    channels: u16,
    extradata: &[u8],
) -> Result<FlacDecoder> {
    FlacDecoder::with_extradata(sample_rate, channels, extradata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flac_decoder_creation() {
        let decoder = FlacDecoder::new(44100, 2);
        assert!(decoder.is_ok());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.sample_rate(), 44100);
        assert_eq!(decoder.channels(), 2);
    }

    #[test]
    fn test_flac_decoder_48khz_stereo() {
        let decoder = FlacDecoder::new(48000, 2);
        assert!(decoder.is_ok());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.sample_rate(), 48000);
    }

    #[test]
    fn test_flac_decoder_mono() {
        let decoder = FlacDecoder::new(44100, 1);
        assert!(decoder.is_ok());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.channels(), 1);
    }

    #[test]
    fn test_flac_decoder_96khz_24bit() {
        let decoder = FlacDecoder::with_bits_per_sample(96000, 2, 24);
        assert!(decoder.is_ok());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.sample_rate(), 96000);
        assert_eq!(decoder.bits_per_sample(), 24);
    }

    #[test]
    fn test_flac_decoder_multichannel() {
        // Test 5.1 surround
        let decoder = FlacDecoder::new(48000, 6);
        assert!(decoder.is_ok());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.channels(), 6);
    }

    #[test]
    fn test_flac_decoder_8_channels() {
        // Test 7.1 surround
        let decoder = FlacDecoder::new(48000, 8);
        assert!(decoder.is_ok());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.channels(), 8);
    }

    #[test]
    fn test_empty_packet_rejected() {
        let mut decoder = FlacDecoder::new(44100, 2).unwrap();
        let packet = Packet::new_audio(0, Buffer::empty());
        let result = decoder.send_packet(&packet);
        assert!(result.is_err());
    }

    #[test]
    fn test_flush() {
        let mut decoder = FlacDecoder::new(44100, 2).unwrap();
        let result = decoder.flush();
        assert!(result.is_ok());
    }

    #[test]
    fn test_receive_without_send() {
        let mut decoder = FlacDecoder::new(44100, 2).unwrap();
        let result = decoder.receive_frame();
        assert!(matches!(result, Err(Error::TryAgain)));
    }

    #[test]
    fn test_stream_info_parsing() {
        // Valid FLAC STREAMINFO for 44100Hz, stereo, 16-bit
        // min_block=4096, max_block=4096, sample_rate=44100, channels=2, bps=16
        let mut streaminfo = [0u8; 34];

        // min_block_size = 4096 (0x1000)
        streaminfo[0] = 0x10;
        streaminfo[1] = 0x00;

        // max_block_size = 4096 (0x1000)
        streaminfo[2] = 0x10;
        streaminfo[3] = 0x00;

        // min_frame_size = 0
        streaminfo[4] = 0x00;
        streaminfo[5] = 0x00;
        streaminfo[6] = 0x00;

        // max_frame_size = 0
        streaminfo[7] = 0x00;
        streaminfo[8] = 0x00;
        streaminfo[9] = 0x00;

        // sample_rate (20 bits) = 44100 = 0xAC44
        // channels - 1 (3 bits) = 1 (stereo)
        // bits_per_sample - 1 (5 bits) = 15 (16-bit)
        // Encoded as: SSSSSSSS SSSSSSSS SSSSCCCC CBBBBBTT
        // 44100 = 0xAC44 in 20 bits: 0000 1010 1100 0100 0100
        // channels-1 = 1 = 001
        // bps-1 = 15 = 01111
        // Byte 10: 0000_1010 = 0x0A
        // Byte 11: 1100_0100 = 0xC4
        // Byte 12: 0100_001_0 = 0100 0010 = 0x42 (last 4 bits of sample_rate, 3 bits channels, 1 bit of bps)
        // Byte 13: 1111_0000 = 0xF0 (remaining 4 bits of bps, 4 bits of total_samples)
        streaminfo[10] = 0x0A;
        streaminfo[11] = 0xC4;
        streaminfo[12] = 0x42;
        streaminfo[13] = 0xF0;

        // total_samples = 0 (remaining 32 bits)
        streaminfo[14] = 0x00;
        streaminfo[15] = 0x00;
        streaminfo[16] = 0x00;
        streaminfo[17] = 0x00;

        // MD5 = zeros
        // streaminfo[18..34] already zeros

        let info = FlacStreamInfo::parse(&streaminfo);
        assert!(info.is_ok());
        let info = info.unwrap();
        assert_eq!(info.min_block_size, 4096);
        assert_eq!(info.max_block_size, 4096);
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
        assert_eq!(info.bits_per_sample, 16);
    }

    #[test]
    fn test_stream_info_validation() {
        // Test invalid sample rate
        let mut streaminfo = [0u8; 34];
        let info = FlacStreamInfo::parse(&streaminfo);
        assert!(info.is_ok());
        let info = info.unwrap();
        // sample_rate is 0, which is invalid
        assert!(info.validate().is_err());
    }

    #[test]
    fn test_factory_function() {
        let decoder = create_decoder(44100, 2);
        assert!(decoder.is_ok());
    }
}
