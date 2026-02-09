//! MP3 audio decoder using Symphonia
//!
//! This module provides standalone MP3 decoding that can decode individual
//! packets without requiring a container/demuxer. This is essential for
//! network streaming scenarios where encoded packets arrive independently.
//!
//! ## Supported Formats
//!
//! - MPEG-1 Audio Layer III (MP3)
//! - MPEG-2 Audio Layer III
//! - MPEG-2.5 Audio Layer III
//! - Bitrates: 8-320 kbps
//! - VBR (Variable Bit Rate) support
//! - Sample rates: 8000-48000 Hz

use crate::codec::{AudioFrame, Decoder, Frame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, SampleFormat, Timestamp};

use std::collections::VecDeque;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CodecParameters, DecoderOptions, CODEC_TYPE_MP3};
use symphonia::core::formats::Packet as SymphoniaPacket;
use symphonia::default::get_codecs;

/// MPEG Audio version enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MpegVersion {
    /// MPEG Version 1
    Mpeg1,
    /// MPEG Version 2
    Mpeg2,
    /// MPEG Version 2.5 (unofficial extension)
    Mpeg25,
}

impl MpegVersion {
    /// Parse MPEG version from frame header
    fn from_header(version_bits: u8) -> Option<Self> {
        match version_bits {
            0b11 => Some(MpegVersion::Mpeg1),
            0b10 => Some(MpegVersion::Mpeg2),
            0b00 => Some(MpegVersion::Mpeg25),
            _ => None, // 0b01 is reserved
        }
    }
}

/// MP3 frame header information
#[derive(Debug, Clone)]
pub struct Mp3FrameHeader {
    /// MPEG version
    pub version: MpegVersion,
    /// Layer (always 3 for MP3)
    pub layer: u8,
    /// CRC protection flag
    pub crc_protected: bool,
    /// Bitrate in kbps
    pub bitrate: u16,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Padding flag
    pub padding: bool,
    /// Channel mode (0=stereo, 1=joint stereo, 2=dual channel, 3=mono)
    pub channel_mode: u8,
    /// Number of channels
    pub channels: u8,
    /// Frame size in bytes
    pub frame_size: usize,
    /// Samples per frame
    pub samples_per_frame: usize,
}

/// MPEG-1 Layer III bitrate table (kbps)
const MPEG1_LAYER3_BITRATES: [u16; 15] = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320];

/// MPEG-2/2.5 Layer III bitrate table (kbps)
const MPEG2_LAYER3_BITRATES: [u16; 15] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160];

/// MPEG-1 sample rates
const MPEG1_SAMPLE_RATES: [u32; 3] = [44100, 48000, 32000];

/// MPEG-2 sample rates
const MPEG2_SAMPLE_RATES: [u32; 3] = [22050, 24000, 16000];

/// MPEG-2.5 sample rates
const MPEG25_SAMPLE_RATES: [u32; 3] = [11025, 12000, 8000];

impl Mp3FrameHeader {
    /// Parse MP3 frame header from raw bytes
    ///
    /// The header is 4 bytes:
    /// - Bytes 0-1: Frame sync (0xFFE or 0xFFF)
    /// - Byte 1 bits 4-3: Version
    /// - Byte 1 bits 2-1: Layer
    /// - Byte 1 bit 0: CRC protection
    /// - Byte 2 bits 7-4: Bitrate index
    /// - Byte 2 bits 3-2: Sample rate index
    /// - Byte 2 bit 1: Padding
    /// - Byte 2 bit 0: Private bit
    /// - Byte 3 bits 7-6: Channel mode
    /// - Byte 3 bits 5-4: Mode extension
    /// - Byte 3 bit 3: Copyright
    /// - Byte 3 bit 2: Original
    /// - Byte 3 bits 1-0: Emphasis
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 4 {
            return Err(Error::codec("MP3 frame header too short (need 4 bytes)"));
        }

        // Check frame sync (11 bits of 1s)
        if data[0] != 0xFF || (data[1] & 0xE0) != 0xE0 {
            return Err(Error::codec("Invalid MP3 frame sync"));
        }

        // Parse version (2 bits)
        let version_bits = (data[1] >> 3) & 0x03;
        let version = MpegVersion::from_header(version_bits)
            .ok_or_else(|| Error::codec("Reserved MPEG version"))?;

        // Parse layer (2 bits)
        let layer_bits = (data[1] >> 1) & 0x03;
        let layer = match layer_bits {
            0b01 => 3, // Layer III
            0b10 => 2, // Layer II
            0b11 => 1, // Layer I
            _ => return Err(Error::codec("Reserved layer value")),
        };

        // CRC protection (inverted)
        let crc_protected = (data[1] & 0x01) == 0;

        // Bitrate index (4 bits)
        let bitrate_index = (data[2] >> 4) as usize;
        if bitrate_index == 0 || bitrate_index == 15 {
            // 0 = free format, 15 = bad
            if bitrate_index == 15 {
                return Err(Error::codec("Invalid bitrate index"));
            }
        }

        let bitrate = match version {
            MpegVersion::Mpeg1 => {
                if bitrate_index < MPEG1_LAYER3_BITRATES.len() {
                    MPEG1_LAYER3_BITRATES[bitrate_index]
                } else {
                    0
                }
            }
            MpegVersion::Mpeg2 | MpegVersion::Mpeg25 => {
                if bitrate_index < MPEG2_LAYER3_BITRATES.len() {
                    MPEG2_LAYER3_BITRATES[bitrate_index]
                } else {
                    0
                }
            }
        };

        // Sample rate index (2 bits)
        let sample_rate_index = ((data[2] >> 2) & 0x03) as usize;
        if sample_rate_index >= 3 {
            return Err(Error::codec("Invalid sample rate index"));
        }

        let sample_rate = match version {
            MpegVersion::Mpeg1 => MPEG1_SAMPLE_RATES[sample_rate_index],
            MpegVersion::Mpeg2 => MPEG2_SAMPLE_RATES[sample_rate_index],
            MpegVersion::Mpeg25 => MPEG25_SAMPLE_RATES[sample_rate_index],
        };

        // Padding flag
        let padding = (data[2] & 0x02) != 0;

        // Channel mode (2 bits)
        let channel_mode = (data[3] >> 6) & 0x03;
        let channels = if channel_mode == 3 { 1 } else { 2 };

        // Calculate samples per frame
        let samples_per_frame = match version {
            MpegVersion::Mpeg1 => 1152,
            MpegVersion::Mpeg2 | MpegVersion::Mpeg25 => 576,
        };

        // Calculate frame size
        // For Layer III: FrameSize = 144 * BitRate / SampleRate + Padding
        let frame_size = if bitrate > 0 {
            let padding_bytes = if padding { 1 } else { 0 };
            (144 * bitrate as usize * 1000 / sample_rate as usize) + padding_bytes
        } else {
            0
        };

        Ok(Mp3FrameHeader {
            version,
            layer,
            crc_protected,
            bitrate,
            sample_rate,
            padding,
            channel_mode,
            channels,
            frame_size,
            samples_per_frame,
        })
    }

    /// Check if this is a valid Layer III frame
    pub fn is_layer3(&self) -> bool {
        self.layer == 3
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

/// MP3 audio decoder using Symphonia
///
/// This decoder wraps Symphonia's MP3 codec to provide standalone packet
/// decoding without requiring a container format. It follows the ZVD Decoder
/// trait interface with send_packet/receive_frame semantics.
///
/// ## Supported Features
///
/// - MPEG-1/2/2.5 Audio Layer III
/// - Sample rates: 8000-48000 Hz
/// - Bitrates: 8-320 kbps (CBR and VBR)
/// - Mono and stereo modes
///
/// ## Example
///
/// ```rust,ignore
/// use zvd_lib::codec::{Decoder, Mp3Decoder};
/// use zvd_lib::format::Packet;
///
/// let mut decoder = Mp3Decoder::new(44100, 2)?;
/// decoder.send_packet(&packet)?;
/// let frame = decoder.receive_frame()?;
/// ```
pub struct Mp3Decoder {
    /// Symphonia MP3 decoder instance
    decoder: Box<dyn symphonia::core::codecs::Decoder>,
    /// Sample buffer for converting decoded audio
    sample_buffer: Option<SampleBuffer<f32>>,
    /// Queue of packets waiting to be decoded
    pending_packets: VecDeque<PendingPacket>,
    /// Sample rate in Hz
    sample_rate: u32,
    /// Number of audio channels
    channels: u16,
    /// Track ID for Symphonia packets
    track_id: u32,
    /// Total samples decoded (for timestamp calculation)
    samples_decoded: u64,
    /// Last detected frame header (for format detection)
    last_header: Option<Mp3FrameHeader>,
}

impl std::fmt::Debug for Mp3Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mp3Decoder")
            .field("sample_rate", &self.sample_rate)
            .field("channels", &self.channels)
            .field("track_id", &self.track_id)
            .field("samples_decoded", &self.samples_decoded)
            .field("pending_packets_count", &self.pending_packets.len())
            .field("last_header", &self.last_header)
            .finish_non_exhaustive()
    }
}

impl Mp3Decoder {
    /// Create a new MP3 decoder with specified parameters
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz (8000-48000)
    /// * `channels` - Number of audio channels (1 or 2)
    ///
    /// # Returns
    /// A configured MP3 decoder ready to decode packets
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        Self::create_decoder(sample_rate, channels, None)
    }

    /// Create a decoder with extradata
    ///
    /// For MP3, extradata is typically not required as the format is
    /// self-describing (each frame contains its own header). However,
    /// if provided, it can contain initial frame header information.
    pub fn with_extradata(sample_rate: u32, channels: u16, extradata: &[u8]) -> Result<Self> {
        // Try to parse frame header from extradata
        if extradata.len() >= 4 {
            if let Ok(header) = Mp3FrameHeader::parse(extradata) {
                return Self::create_decoder(header.sample_rate, header.channels as u16, Some(extradata));
            }
        }
        Self::create_decoder(sample_rate, channels, Some(extradata))
    }

    /// Internal decoder creation
    fn create_decoder(
        sample_rate: u32,
        channels: u16,
        extradata: Option<&[u8]>,
    ) -> Result<Self> {
        // Build codec parameters for MP3
        let mut codec_params = CodecParameters::new();
        codec_params
            .for_codec(CODEC_TYPE_MP3)
            .with_sample_rate(sample_rate);

        // Set channel layout based on channel count
        use symphonia::core::audio::Channels;
        let channel_layout = match channels {
            1 => Channels::FRONT_CENTRE,
            2 => Channels::FRONT_LEFT | Channels::FRONT_RIGHT,
            _ => Channels::FRONT_LEFT | Channels::FRONT_RIGHT, // Default to stereo
        };
        codec_params.with_channels(channel_layout);

        // Set extradata if provided
        if let Some(extra) = extradata {
            if !extra.is_empty() {
                codec_params.with_extra_data(extra.to_vec().into_boxed_slice());
            }
        }

        // Create the Symphonia MP3 decoder
        let decoder = get_codecs()
            .make(&codec_params, &DecoderOptions::default())
            .map_err(|e| {
                Error::codec(format!(
                    "Failed to create MP3 decoder: {}. Ensure symphonia 'mp3' feature is enabled.",
                    e
                ))
            })?;

        Ok(Mp3Decoder {
            decoder,
            sample_buffer: None,
            pending_packets: VecDeque::new(),
            sample_rate,
            channels,
            track_id: 0,
            samples_decoded: 0,
            last_header: None,
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

    /// Get total samples decoded so far
    pub fn samples_decoded(&self) -> u64 {
        self.samples_decoded
    }

    /// Get the last detected frame header
    pub fn last_header(&self) -> Option<&Mp3FrameHeader> {
        self.last_header.as_ref()
    }

    /// Parse frame header from packet data (if present)
    fn try_parse_header(&mut self, data: &[u8]) {
        if data.len() >= 4 {
            if let Ok(header) = Mp3FrameHeader::parse(data) {
                // Update decoder state based on header
                if header.sample_rate != self.sample_rate {
                    self.sample_rate = header.sample_rate;
                }
                if header.channels as u16 != self.channels {
                    self.channels = header.channels as u16;
                }
                self.last_header = Some(header);
            }
        }
    }

    /// Decode a single packet internally
    fn decode_packet_internal(&mut self, pending: &PendingPacket) -> Result<AudioFrame> {
        // Try to parse header for format detection
        self.try_parse_header(&pending.data);

        // Create a Symphonia packet from the raw data
        let symphonia_packet = SymphoniaPacket::new_from_slice(
            self.track_id,
            pending.pts.value as u64,
            pending.duration as u64,
            &pending.data,
        );

        // Decode the packet
        let decoded = self.decoder.decode(&symphonia_packet).map_err(|e| {
            Error::codec(format!("MP3 decode error: {}", e))
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

impl Decoder for Mp3Decoder {
    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        // Validate packet
        if packet.data.is_empty() {
            return Err(Error::codec("Empty MP3 packet"));
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
        self.last_header = None;

        Ok(())
    }
}

/// Create an MP3 decoder with default parameters
///
/// This is a convenience factory function.
pub fn create_decoder(sample_rate: u32, channels: u16) -> Result<Mp3Decoder> {
    Mp3Decoder::new(sample_rate, channels)
}

/// Create an MP3 decoder with extradata
pub fn create_decoder_with_extradata(
    sample_rate: u32,
    channels: u16,
    extradata: &[u8],
) -> Result<Mp3Decoder> {
    Mp3Decoder::with_extradata(sample_rate, channels, extradata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mp3_decoder_creation() {
        let decoder = Mp3Decoder::new(44100, 2);
        assert!(decoder.is_ok());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.sample_rate(), 44100);
        assert_eq!(decoder.channels(), 2);
    }

    #[test]
    fn test_mp3_decoder_48khz_stereo() {
        let decoder = Mp3Decoder::new(48000, 2);
        assert!(decoder.is_ok());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.sample_rate(), 48000);
    }

    #[test]
    fn test_mp3_decoder_mono() {
        let decoder = Mp3Decoder::new(44100, 1);
        assert!(decoder.is_ok());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.channels(), 1);
    }

    #[test]
    fn test_mp3_decoder_various_sample_rates() {
        // Test various MP3 sample rates
        for sample_rate in [8000u32, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000] {
            let decoder = Mp3Decoder::new(sample_rate, 2);
            assert!(decoder.is_ok(), "Failed for sample rate {}", sample_rate);
            assert_eq!(decoder.unwrap().sample_rate(), sample_rate);
        }
    }

    #[test]
    fn test_empty_packet_rejected() {
        let mut decoder = Mp3Decoder::new(44100, 2).unwrap();
        let packet = Packet::new_audio(0, Buffer::empty());
        let result = decoder.send_packet(&packet);
        assert!(result.is_err());
    }

    #[test]
    fn test_flush() {
        let mut decoder = Mp3Decoder::new(44100, 2).unwrap();
        let result = decoder.flush();
        assert!(result.is_ok());
    }

    #[test]
    fn test_receive_without_send() {
        let mut decoder = Mp3Decoder::new(44100, 2).unwrap();
        let result = decoder.receive_frame();
        assert!(matches!(result, Err(Error::TryAgain)));
    }

    #[test]
    fn test_frame_header_parsing() {
        // Valid MP3 frame header for MPEG-1 Layer III, 128kbps, 44100Hz, stereo
        // Sync: 0xFF 0xFB (MPEG-1, Layer III, no CRC)
        // 0x90 = bitrate index 9 (128kbps), sample rate index 0 (44100Hz), no padding
        // 0x00 = stereo, no mode extension, no copyright, not original, no emphasis
        let header_data = [0xFF, 0xFB, 0x90, 0x00];
        let header = Mp3FrameHeader::parse(&header_data);
        assert!(header.is_ok());
        let header = header.unwrap();
        assert_eq!(header.version, MpegVersion::Mpeg1);
        assert_eq!(header.layer, 3);
        assert_eq!(header.bitrate, 128);
        assert_eq!(header.sample_rate, 44100);
        assert_eq!(header.channels, 2);
    }

    #[test]
    fn test_frame_header_mono() {
        // MP3 frame header for mono
        // 0xC0 in byte 3 = channel mode 3 (mono)
        let header_data = [0xFF, 0xFB, 0x90, 0xC0];
        let header = Mp3FrameHeader::parse(&header_data);
        assert!(header.is_ok());
        let header = header.unwrap();
        assert_eq!(header.channels, 1);
    }

    #[test]
    fn test_frame_header_invalid_sync() {
        let header_data = [0x00, 0x00, 0x00, 0x00];
        let header = Mp3FrameHeader::parse(&header_data);
        assert!(header.is_err());
    }

    #[test]
    fn test_mpeg_version_from_header() {
        assert_eq!(MpegVersion::from_header(0b11), Some(MpegVersion::Mpeg1));
        assert_eq!(MpegVersion::from_header(0b10), Some(MpegVersion::Mpeg2));
        assert_eq!(MpegVersion::from_header(0b00), Some(MpegVersion::Mpeg25));
        assert_eq!(MpegVersion::from_header(0b01), None); // Reserved
    }

    #[test]
    fn test_factory_function() {
        let decoder = create_decoder(44100, 2);
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_samples_per_frame() {
        // MPEG-1 Layer III should have 1152 samples per frame
        let header_data = [0xFF, 0xFB, 0x90, 0x00];
        let header = Mp3FrameHeader::parse(&header_data).unwrap();
        assert_eq!(header.samples_per_frame, 1152);
    }

    #[test]
    fn test_mpeg2_frame_header() {
        // MPEG-2 Layer III header
        // 0xF3 = MPEG-2, Layer III, no CRC
        let header_data = [0xFF, 0xF3, 0x90, 0x00];
        let header = Mp3FrameHeader::parse(&header_data);
        assert!(header.is_ok());
        let header = header.unwrap();
        assert_eq!(header.version, MpegVersion::Mpeg2);
        assert_eq!(header.samples_per_frame, 576);
    }
}
