//! Vorbis audio decoder using Symphonia
//!
//! This module provides standalone Vorbis decoding that can decode individual
//! packets without requiring a container/demuxer. This is essential for
//! network streaming scenarios where encoded packets arrive independently.
//!
//! ## Vorbis Header Format
//!
//! Vorbis streams require three header packets before audio can be decoded:
//! 1. Identification header - basic stream parameters
//! 2. Comment header - metadata (tags)
//! 3. Setup header - codebooks and decode setup
//!
//! These headers should be provided as extradata when creating the decoder,
//! or sent as the first packets before audio data.

use crate::codec::{AudioFrame, Decoder, Frame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, SampleFormat, Timestamp};

use std::collections::VecDeque;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CodecParameters, DecoderOptions, CODEC_TYPE_VORBIS};
use symphonia::core::formats::Packet as SymphoniaPacket;
use symphonia::default::get_codecs;

/// Vorbis identification header information
///
/// Parsed from the first Vorbis header packet (type 0x01)
#[derive(Debug, Clone)]
pub struct VorbisIdHeader {
    /// Vorbis version (always 0)
    pub vorbis_version: u32,
    /// Number of audio channels
    pub audio_channels: u8,
    /// Sample rate in Hz
    pub audio_sample_rate: u32,
    /// Maximum bitrate (0 if unspecified)
    pub bitrate_maximum: i32,
    /// Nominal bitrate (0 if unspecified)
    pub bitrate_nominal: i32,
    /// Minimum bitrate (0 if unspecified)
    pub bitrate_minimum: i32,
    /// Log2 of minimum block size (64-8192)
    pub blocksize_0: u8,
    /// Log2 of maximum block size (64-8192)
    pub blocksize_1: u8,
}

impl VorbisIdHeader {
    /// Parse Vorbis identification header from raw bytes
    ///
    /// The identification header is structured as:
    /// - 1 byte: packet type (0x01)
    /// - 6 bytes: "vorbis" magic
    /// - 4 bytes: vorbis_version (little-endian)
    /// - 1 byte: audio_channels
    /// - 4 bytes: audio_sample_rate (little-endian)
    /// - 4 bytes: bitrate_maximum (little-endian, signed)
    /// - 4 bytes: bitrate_nominal (little-endian, signed)
    /// - 4 bytes: bitrate_minimum (little-endian, signed)
    /// - 1 byte: blocksize_0 (4 bits) | blocksize_1 (4 bits)
    /// - 1 byte: framing bit
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 30 {
            return Err(Error::codec(format!(
                "Vorbis identification header too short: {} bytes (need 30)",
                data.len()
            )));
        }

        // Check packet type
        if data[0] != 0x01 {
            return Err(Error::codec(format!(
                "Invalid Vorbis packet type: expected 0x01, got 0x{:02X}",
                data[0]
            )));
        }

        // Check "vorbis" magic
        if &data[1..7] != b"vorbis" {
            return Err(Error::codec("Invalid Vorbis magic string"));
        }

        // Parse fields
        let vorbis_version = u32::from_le_bytes([data[7], data[8], data[9], data[10]]);
        if vorbis_version != 0 {
            return Err(Error::codec(format!(
                "Unsupported Vorbis version: {}",
                vorbis_version
            )));
        }

        let audio_channels = data[11];
        if audio_channels == 0 {
            return Err(Error::codec("Invalid Vorbis channel count: 0"));
        }

        let audio_sample_rate =
            u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
        if audio_sample_rate == 0 {
            return Err(Error::codec("Invalid Vorbis sample rate: 0"));
        }

        let bitrate_maximum =
            i32::from_le_bytes([data[16], data[17], data[18], data[19]]);
        let bitrate_nominal =
            i32::from_le_bytes([data[20], data[21], data[22], data[23]]);
        let bitrate_minimum =
            i32::from_le_bytes([data[24], data[25], data[26], data[27]]);

        let blocksize_byte = data[28];
        let blocksize_0 = blocksize_byte & 0x0F;
        let blocksize_1 = (blocksize_byte >> 4) & 0x0F;

        // Validate blocksizes (must be 6-13 for sizes 64-8192)
        if blocksize_0 < 6 || blocksize_0 > 13 || blocksize_1 < 6 || blocksize_1 > 13 {
            return Err(Error::codec(format!(
                "Invalid Vorbis blocksizes: {} / {}",
                blocksize_0, blocksize_1
            )));
        }

        if blocksize_0 > blocksize_1 {
            return Err(Error::codec(
                "Vorbis blocksize_0 must be <= blocksize_1",
            ));
        }

        // Check framing bit
        if (data[29] & 0x01) != 1 {
            return Err(Error::codec("Invalid Vorbis framing bit"));
        }

        Ok(VorbisIdHeader {
            vorbis_version,
            audio_channels,
            audio_sample_rate,
            bitrate_maximum,
            bitrate_nominal,
            bitrate_minimum,
            blocksize_0,
            blocksize_1,
        })
    }

    /// Get the actual block size for blocksize_0
    pub fn block_size_0(&self) -> usize {
        1 << self.blocksize_0
    }

    /// Get the actual block size for blocksize_1
    pub fn block_size_1(&self) -> usize {
        1 << self.blocksize_1
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

/// Vorbis audio decoder using Symphonia
///
/// This decoder wraps Symphonia's Vorbis codec to provide standalone packet
/// decoding without requiring a container format. It follows the ZVD Decoder
/// trait interface with send_packet/receive_frame semantics.
///
/// ## Header Requirements
///
/// Vorbis requires three header packets before audio decoding can begin:
/// 1. Identification header (packet type 0x01)
/// 2. Comment header (packet type 0x03)
/// 3. Setup header (packet type 0x05)
///
/// These should be provided via `with_extradata()` or sent as the first
/// packets using `send_packet()`.
///
/// ## Example
///
/// ```rust,ignore
/// use zvd_lib::codec::{Decoder, VorbisDecoder};
/// use zvd_lib::format::Packet;
///
/// // Create a decoder with extradata containing Vorbis headers
/// let mut decoder = VorbisDecoder::with_extradata(44100, 2, &header_data)?;
///
/// // Or create without headers (must send header packets first)
/// let mut decoder = VorbisDecoder::new(44100, 2)?;
/// decoder.send_packet(&id_header_packet)?;
/// decoder.send_packet(&comment_header_packet)?;
/// decoder.send_packet(&setup_header_packet)?;
///
/// // Decode audio packets
/// decoder.send_packet(&audio_packet)?;
/// let frame = decoder.receive_frame()?;
/// ```
pub struct VorbisDecoder {
    /// Symphonia Vorbis decoder instance
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
    /// Parsed identification header (if available)
    id_header: Option<VorbisIdHeader>,
    /// Whether headers have been processed
    headers_received: bool,
}

impl std::fmt::Debug for VorbisDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VorbisDecoder")
            .field("sample_rate", &self.sample_rate)
            .field("channels", &self.channels)
            .field("track_id", &self.track_id)
            .field("samples_decoded", &self.samples_decoded)
            .field("pending_packets_count", &self.pending_packets.len())
            .field("id_header", &self.id_header)
            .field("headers_received", &self.headers_received)
            .finish_non_exhaustive()
    }
}

impl VorbisDecoder {
    /// Create a new Vorbis decoder with specified parameters
    ///
    /// Note: Vorbis decoding requires header packets to be sent first
    /// unless extradata is provided.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of audio channels (1-255)
    ///
    /// # Returns
    /// A configured Vorbis decoder ready to receive header packets
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        Self::create_decoder(sample_rate, channels, None)
    }

    /// Create a decoder with extradata containing Vorbis headers
    ///
    /// The extradata should contain the three Vorbis header packets
    /// concatenated together (identification, comment, setup).
    /// Common container formats like Ogg and WebM provide this data.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of channels
    /// * `extradata` - Concatenated Vorbis header packets
    pub fn with_extradata(sample_rate: u32, channels: u16, extradata: &[u8]) -> Result<Self> {
        // Try to parse identification header from extradata
        if extradata.len() >= 30 {
            if let Ok(id_header) = VorbisIdHeader::parse(extradata) {
                return Self::create_decoder(
                    id_header.audio_sample_rate,
                    id_header.audio_channels as u16,
                    Some(extradata),
                );
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
        // Build codec parameters for Vorbis
        let mut codec_params = CodecParameters::new();
        codec_params
            .for_codec(CODEC_TYPE_VORBIS)
            .with_sample_rate(sample_rate);

        // Set channel layout based on channel count
        // Vorbis supports up to 255 channels, but we handle common configurations
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
            _ => Channels::FRONT_LEFT | Channels::FRONT_RIGHT, // Default to stereo
        };
        codec_params.with_channels(channel_layout);

        // Set extradata if provided (contains Vorbis headers)
        if let Some(extra) = extradata {
            if !extra.is_empty() {
                codec_params.with_extra_data(extra.to_vec().into_boxed_slice());
            }
        }

        // Create the Symphonia Vorbis decoder
        let decoder = get_codecs()
            .make(&codec_params, &DecoderOptions::default())
            .map_err(|e| {
                Error::codec(format!(
                    "Failed to create Vorbis decoder: {}. Ensure symphonia 'vorbis' feature is enabled.",
                    e
                ))
            })?;

        // Parse identification header from extradata if available
        let id_header = extradata
            .filter(|e| e.len() >= 30)
            .and_then(|e| VorbisIdHeader::parse(e).ok());

        let headers_received = extradata.is_some() && !extradata.unwrap_or_default().is_empty();

        Ok(VorbisDecoder {
            decoder,
            sample_buffer: None,
            pending_packets: VecDeque::new(),
            sample_rate,
            channels,
            track_id: 0,
            samples_decoded: 0,
            id_header,
            headers_received,
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

    /// Get the parsed identification header (if available)
    pub fn id_header(&self) -> Option<&VorbisIdHeader> {
        self.id_header.as_ref()
    }

    /// Check if a packet is a Vorbis header packet
    fn is_header_packet(data: &[u8]) -> bool {
        if data.len() < 7 {
            return false;
        }
        // Header packets have type 0x01, 0x03, or 0x05 and start with "vorbis"
        let packet_type = data[0];
        (packet_type == 0x01 || packet_type == 0x03 || packet_type == 0x05)
            && &data[1..7] == b"vorbis"
    }

    /// Process a header packet
    fn process_header(&mut self, data: &[u8]) {
        if data.len() < 7 {
            return;
        }

        match data[0] {
            0x01 => {
                // Identification header
                if let Ok(header) = VorbisIdHeader::parse(data) {
                    self.sample_rate = header.audio_sample_rate;
                    self.channels = header.audio_channels as u16;
                    self.id_header = Some(header);
                }
            }
            0x03 => {
                // Comment header - we don't need to parse this for decoding
            }
            0x05 => {
                // Setup header - indicates headers are complete
                self.headers_received = true;
            }
            _ => {}
        }
    }

    /// Decode a single packet internally
    fn decode_packet_internal(&mut self, pending: &PendingPacket) -> Result<AudioFrame> {
        // Check if this is a header packet
        if Self::is_header_packet(&pending.data) {
            self.process_header(&pending.data);
            // Return TryAgain to indicate we need more packets
            return Err(Error::TryAgain);
        }

        // Create a Symphonia packet from the raw data
        let symphonia_packet = SymphoniaPacket::new_from_slice(
            self.track_id,
            pending.pts.value as u64,
            pending.duration as u64,
            &pending.data,
        );

        // Decode the packet
        let decoded = self.decoder.decode(&symphonia_packet).map_err(|e| {
            Error::codec(format!("Vorbis decode error: {}", e))
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

impl Decoder for VorbisDecoder {
    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        // Validate packet
        if packet.data.is_empty() {
            return Err(Error::codec("Empty Vorbis packet"));
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
        // Try to decode pending packets
        while let Some(pending) = self.pending_packets.pop_front() {
            match self.decode_packet_internal(&pending) {
                Ok(audio_frame) => {
                    return Ok(Frame::Audio(audio_frame));
                }
                Err(Error::TryAgain) => {
                    // This was a header packet, continue to next
                    continue;
                }
                Err(e) => {
                    // For corrupted frames, try next packet if available
                    if !self.pending_packets.is_empty() {
                        continue;
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
        // Note: Don't clear id_header or headers_received as we may resume decoding

        Ok(())
    }
}

/// Create a Vorbis decoder with default parameters
///
/// This is a convenience factory function.
pub fn create_decoder(sample_rate: u32, channels: u16) -> Result<VorbisDecoder> {
    VorbisDecoder::new(sample_rate, channels)
}

/// Create a Vorbis decoder with extradata
///
/// The extradata should contain the concatenated Vorbis header packets.
pub fn create_decoder_with_extradata(
    sample_rate: u32,
    channels: u16,
    extradata: &[u8],
) -> Result<VorbisDecoder> {
    VorbisDecoder::with_extradata(sample_rate, channels, extradata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vorbis_decoder_creation() {
        let decoder = VorbisDecoder::new(44100, 2);
        assert!(decoder.is_ok());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.sample_rate(), 44100);
        assert_eq!(decoder.channels(), 2);
    }

    #[test]
    fn test_vorbis_decoder_48khz_stereo() {
        let decoder = VorbisDecoder::new(48000, 2);
        assert!(decoder.is_ok());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.sample_rate(), 48000);
    }

    #[test]
    fn test_vorbis_decoder_mono() {
        let decoder = VorbisDecoder::new(44100, 1);
        assert!(decoder.is_ok());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.channels(), 1);
    }

    #[test]
    fn test_vorbis_decoder_multichannel() {
        // Test 5.1 surround
        let decoder = VorbisDecoder::new(48000, 6);
        assert!(decoder.is_ok());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.channels(), 6);
    }

    #[test]
    fn test_vorbis_decoder_8_channels() {
        // Test 7.1 surround
        let decoder = VorbisDecoder::new(48000, 8);
        assert!(decoder.is_ok());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.channels(), 8);
    }

    #[test]
    fn test_empty_packet_rejected() {
        let mut decoder = VorbisDecoder::new(44100, 2).unwrap();
        let packet = Packet::new_audio(0, Buffer::empty());
        let result = decoder.send_packet(&packet);
        assert!(result.is_err());
    }

    #[test]
    fn test_flush() {
        let mut decoder = VorbisDecoder::new(44100, 2).unwrap();
        let result = decoder.flush();
        assert!(result.is_ok());
    }

    #[test]
    fn test_receive_without_send() {
        let mut decoder = VorbisDecoder::new(44100, 2).unwrap();
        let result = decoder.receive_frame();
        assert!(matches!(result, Err(Error::TryAgain)));
    }

    #[test]
    fn test_id_header_parsing() {
        // Valid Vorbis identification header
        // Type: 0x01, Magic: "vorbis", Version: 0, Channels: 2, SampleRate: 44100
        let mut header = vec![0x01]; // Packet type
        header.extend_from_slice(b"vorbis"); // Magic
        header.extend_from_slice(&0u32.to_le_bytes()); // Version
        header.push(2); // Channels
        header.extend_from_slice(&44100u32.to_le_bytes()); // Sample rate
        header.extend_from_slice(&0i32.to_le_bytes()); // Max bitrate
        header.extend_from_slice(&128000i32.to_le_bytes()); // Nominal bitrate
        header.extend_from_slice(&0i32.to_le_bytes()); // Min bitrate
        header.push(0x88); // blocksizes: 8 | (8 << 4)
        header.push(0x01); // Framing bit

        let id_header = VorbisIdHeader::parse(&header);
        assert!(id_header.is_ok());
        let id_header = id_header.unwrap();
        assert_eq!(id_header.vorbis_version, 0);
        assert_eq!(id_header.audio_channels, 2);
        assert_eq!(id_header.audio_sample_rate, 44100);
        assert_eq!(id_header.bitrate_nominal, 128000);
        assert_eq!(id_header.blocksize_0, 8);
        assert_eq!(id_header.blocksize_1, 8);
    }

    #[test]
    fn test_id_header_invalid_magic() {
        let mut header = vec![0x01]; // Packet type
        header.extend_from_slice(b"vorbiX"); // Invalid magic
        header.extend_from_slice(&[0u8; 23]); // Padding

        let id_header = VorbisIdHeader::parse(&header);
        assert!(id_header.is_err());
    }

    #[test]
    fn test_id_header_invalid_type() {
        let mut header = vec![0x02]; // Invalid packet type
        header.extend_from_slice(b"vorbis");
        header.extend_from_slice(&[0u8; 23]);

        let id_header = VorbisIdHeader::parse(&header);
        assert!(id_header.is_err());
    }

    #[test]
    fn test_is_header_packet() {
        // Identification header
        let mut id_header = vec![0x01];
        id_header.extend_from_slice(b"vorbis");
        assert!(VorbisDecoder::is_header_packet(&id_header));

        // Comment header
        let mut comment_header = vec![0x03];
        comment_header.extend_from_slice(b"vorbis");
        assert!(VorbisDecoder::is_header_packet(&comment_header));

        // Setup header
        let mut setup_header = vec![0x05];
        setup_header.extend_from_slice(b"vorbis");
        assert!(VorbisDecoder::is_header_packet(&setup_header));

        // Audio packet (not a header)
        let audio_packet = vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        assert!(!VorbisDecoder::is_header_packet(&audio_packet));
    }

    #[test]
    fn test_factory_function() {
        let decoder = create_decoder(44100, 2);
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_block_sizes() {
        // Test block size calculations
        let mut header = vec![0x01];
        header.extend_from_slice(b"vorbis");
        header.extend_from_slice(&0u32.to_le_bytes());
        header.push(2);
        header.extend_from_slice(&44100u32.to_le_bytes());
        header.extend_from_slice(&0i32.to_le_bytes());
        header.extend_from_slice(&0i32.to_le_bytes());
        header.extend_from_slice(&0i32.to_le_bytes());
        header.push(0x8B); // blocksizes: 11 (2048) | (8 << 4) (256)
        header.push(0x01);

        let id_header = VorbisIdHeader::parse(&header).unwrap();
        assert_eq!(id_header.block_size_0(), 2048); // 2^11
        assert_eq!(id_header.block_size_1(), 256); // 2^8
    }
}
