//! H.264 encoder using OpenH264
//!
//! This module provides H.264 encoding using the OpenH264 library. The encoder
//! converts YUV420P video frames to H.264 NAL units in Annex B format.
//!
//! ## Features
//! - Configurable bitrate and quality settings
//! - SPS/PPS parameter set extraction for MP4 muxing
//! - Keyframe detection and forced keyframe insertion
//! - Proper timestamp handling
//!
//! ## Security Notes
//! This encoder requires openh264 >= 0.9.1 which includes fixes for CVE-2025-27091.

use crate::codec::frame::VideoFrame;
use crate::codec::{Encoder, Frame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, PixelFormat};
use openh264::encoder::{
    BitRate, Encoder as OpenH264Encoder, EncoderConfig,
    FrameRate, FrameType, IntraFramePeriod,
};
use openh264::formats::YUVBuffer;
use openh264::OpenH264API;
use std::collections::VecDeque;

use super::nal::{build_avcc, contains_keyframe, extract_sps_pps};

/// Rate control mode for H.264 encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateControlMode {
    /// Quality-based rate control (constant quality)
    Quality,
    /// Bitrate-based rate control (constant bitrate)
    Bitrate,
    /// Variable bitrate with quality constraints
    VariableBitrate,
    /// Timestamp-based rate control
    Timestamp,
    /// No rate control (constant QP)
    Off,
}

/// H.264 encoder configuration
#[derive(Debug, Clone)]
pub struct H264EncoderConfig {
    /// Target bitrate in bits per second
    pub bitrate: u32,
    /// Maximum frame rate
    pub max_frame_rate: f32,
    /// Rate control mode
    pub rate_control_mode: RateControlMode,
    /// Enable scene change detection
    pub scene_change_detect: bool,
    /// Keyframe interval (0 = automatic)
    pub keyframe_interval: u32,
    /// Number of encoding threads (0 = auto)
    pub threads: u16,
}

impl Default for H264EncoderConfig {
    fn default() -> Self {
        H264EncoderConfig {
            bitrate: 1_000_000,           // 1 Mbps default
            max_frame_rate: 30.0,         // 30 fps
            rate_control_mode: RateControlMode::Bitrate,
            scene_change_detect: true,
            keyframe_interval: 60,        // Every 60 frames (2 seconds at 30fps)
            threads: 0,                   // Auto
        }
    }
}

/// H.264 encoder wrapping OpenH264
///
/// The encoder maintains internal state for packet buffering and
/// parameter set extraction.
pub struct H264Encoder {
    /// The underlying OpenH264 encoder
    encoder: OpenH264Encoder,
    /// Target dimensions
    width: usize,
    height: usize,
    /// Encoder configuration
    config: H264EncoderConfig,
    /// Frame counter
    frame_count: u64,
    /// Queue of encoded packets
    packet_queue: VecDeque<Packet>,
    /// Extracted SPS NAL unit (cached for extradata)
    cached_sps: Option<Vec<u8>>,
    /// Extracted PPS NAL unit (cached for extradata)
    cached_pps: Option<Vec<u8>>,
    /// Generated extradata in avcC format
    cached_extradata: Option<Vec<u8>>,
    /// Whether to force the next frame as keyframe
    force_keyframe_flag: bool,
    /// Stream index for output packets
    stream_index: usize,
}

impl H264Encoder {
    /// Create a new H.264 encoder with the given dimensions
    ///
    /// Uses default encoder settings (1 Mbps bitrate, 30fps max).
    pub fn new(width: u32, height: u32) -> Result<Self> {
        Self::with_config(width, height, H264EncoderConfig::default())
    }

    /// Create a new H.264 encoder with specified bitrate
    pub fn with_bitrate(width: u32, height: u32, bitrate: u32) -> Result<Self> {
        let mut config = H264EncoderConfig::default();
        config.bitrate = bitrate;
        Self::with_config(width, height, config)
    }

    /// Create a new H.264 encoder with full configuration
    pub fn with_config(width: u32, height: u32, config: H264EncoderConfig) -> Result<Self> {
        // Build OpenH264 encoder config using the API
        let mut encoder_config = EncoderConfig::new();

        // Apply bitrate using BitRate type
        encoder_config = encoder_config.bitrate(BitRate::from_bps(config.bitrate));

        // Apply frame rate using FrameRate type
        encoder_config = encoder_config.max_frame_rate(FrameRate::from_hz(config.max_frame_rate));

        // Apply rate control mode
        encoder_config = match config.rate_control_mode {
            RateControlMode::Quality => {
                encoder_config.rate_control_mode(openh264::encoder::RateControlMode::Quality)
            }
            RateControlMode::Bitrate => {
                encoder_config.rate_control_mode(openh264::encoder::RateControlMode::Bitrate)
            }
            RateControlMode::VariableBitrate => {
                encoder_config.rate_control_mode(openh264::encoder::RateControlMode::Bitrate)
            }
            RateControlMode::Timestamp => {
                encoder_config.rate_control_mode(openh264::encoder::RateControlMode::Timestamp)
            }
            RateControlMode::Off => {
                encoder_config.rate_control_mode(openh264::encoder::RateControlMode::Off)
            }
        };

        // Apply scene change detection
        encoder_config = encoder_config.scene_change_detect(config.scene_change_detect);

        // Apply keyframe interval using IntraFramePeriod type
        encoder_config = if config.keyframe_interval > 0 {
            encoder_config.intra_frame_period(IntraFramePeriod::from_num_frames(config.keyframe_interval))
        } else {
            encoder_config.intra_frame_period(IntraFramePeriod::auto())
        };

        // Apply threads
        encoder_config = encoder_config.num_threads(config.threads);

        // Create encoder using the correct API with from_source()
        let api = OpenH264API::from_source();
        let encoder = OpenH264Encoder::with_api_config(api, encoder_config)
            .map_err(|e| Error::codec(format!("Failed to create H.264 encoder: {:?}", e)))?;

        Ok(H264Encoder {
            encoder,
            width: width as usize,
            height: height as usize,
            config,
            frame_count: 0,
            packet_queue: VecDeque::with_capacity(4),
            cached_sps: None,
            cached_pps: None,
            cached_extradata: None,
            force_keyframe_flag: false,
            stream_index: 0,
        })
    }

    /// Set the stream index for output packets
    pub fn set_stream_index(&mut self, index: usize) {
        self.stream_index = index;
    }

    /// Force the next encoded frame to be a keyframe
    pub fn force_keyframe(&mut self) {
        self.force_keyframe_flag = true;
    }

    /// Check if parameter sets (SPS/PPS) have been extracted
    pub fn has_parameter_sets(&self) -> bool {
        self.cached_sps.is_some() && self.cached_pps.is_some()
    }

    /// Get the cached SPS NAL unit
    pub fn sps(&self) -> Option<&[u8]> {
        self.cached_sps.as_deref()
    }

    /// Get the cached PPS NAL unit
    pub fn pps(&self) -> Option<&[u8]> {
        self.cached_pps.as_deref()
    }

    /// Get extradata in avcC format (suitable for MP4 muxing)
    ///
    /// Returns None if SPS/PPS have not been extracted yet (encode a frame first).
    pub fn extradata(&self) -> Option<&[u8]> {
        self.cached_extradata.as_deref()
    }

    /// Get the configured bitrate
    pub fn bitrate(&self) -> u32 {
        self.config.bitrate
    }

    /// Get the encoder dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width as u32, self.height as u32)
    }

    /// Convert VideoFrame to YUVBuffer for OpenH264
    fn video_frame_to_yuv(&self, video_frame: &VideoFrame) -> Result<YUVBuffer> {
        // Ensure we have YUV420P format
        if video_frame.format != PixelFormat::YUV420P {
            return Err(Error::codec(format!(
                "Unsupported pixel format for H.264 encoding: {:?}. Only YUV420P is supported.",
                video_frame.format
            )));
        }

        if video_frame.data.len() < 3 {
            return Err(Error::codec("VideoFrame must have 3 planes (Y, U, V)"));
        }

        // Get plane data
        let y_plane = video_frame.data[0].as_slice();
        let u_plane = video_frame.data[1].as_slice();
        let v_plane = video_frame.data[2].as_slice();

        let y_stride = video_frame.linesize[0];
        let u_stride = video_frame.linesize[1];
        let v_stride = video_frame.linesize[2];

        // Create contiguous YUV420P data
        // For YUV420P: Y plane is full size, U and V are 1/4 size each
        let y_size = self.width * self.height;
        let uv_width = (self.width + 1) / 2;
        let uv_height = (self.height + 1) / 2;
        let uv_size = uv_width * uv_height;
        let total_size = y_size + uv_size * 2;

        let mut yuv_data = vec![0u8; total_size];

        // Copy Y plane (respecting stride)
        for y in 0..self.height {
            let src_start = y * y_stride;
            let src_end = src_start + self.width;
            let dst_start = y * self.width;
            let dst_end = dst_start + self.width;

            if src_end <= y_plane.len() && dst_end <= y_size {
                yuv_data[dst_start..dst_end].copy_from_slice(&y_plane[src_start..src_end]);
            }
        }

        // Copy U plane
        let u_offset = y_size;
        for y in 0..uv_height {
            let src_start = y * u_stride;
            let src_end = src_start + uv_width;
            let dst_start = u_offset + y * uv_width;
            let dst_end = dst_start + uv_width;

            if src_end <= u_plane.len() && dst_end <= u_offset + uv_size {
                yuv_data[dst_start..dst_end].copy_from_slice(&u_plane[src_start..src_end]);
            }
        }

        // Copy V plane
        let v_offset = y_size + uv_size;
        for y in 0..uv_height {
            let src_start = y * v_stride;
            let src_end = src_start + uv_width;
            let dst_start = v_offset + y * uv_width;
            let dst_end = dst_start + uv_width;

            if src_end <= v_plane.len() && dst_end <= total_size {
                yuv_data[dst_start..dst_end].copy_from_slice(&v_plane[src_start..src_end]);
            }
        }

        // Create YUVBuffer from contiguous data
        Ok(YUVBuffer::from_vec(yuv_data, self.width, self.height))
    }

    /// Extract and cache SPS/PPS from encoded bitstream
    fn extract_parameter_sets(&mut self, data: &[u8]) {
        if let Some((sps, pps)) = extract_sps_pps(data) {
            self.cached_sps = Some(sps.clone());
            self.cached_pps = Some(pps.clone());

            // Build avcC extradata
            self.cached_extradata = Some(build_avcc(&sps, &pps));
        }
    }

}

impl Encoder for H264Encoder {
    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        match frame {
            Frame::Video(video_frame) => {
                // Convert to YUV buffer
                let yuv = self.video_frame_to_yuv(video_frame)?;

                // Force keyframe if requested
                if self.force_keyframe_flag {
                    self.encoder.force_intra_frame();
                    self.force_keyframe_flag = false;
                }

                // Store the PTS before encoding
                let pts = video_frame.pts;

                // Encode frame and extract data immediately to avoid borrow issues
                // encode() returns EncodedBitStream which borrows from self.encoder
                let encode_result = self.encoder.encode(&yuv);
                let bitstream = encode_result
                    .map_err(|e| Error::codec(format!("H.264 encoding failed: {:?}", e)))?;

                // Extract all data from bitstream before dropping the borrow
                let data = bitstream.to_vec();
                let is_keyframe = matches!(bitstream.frame_type(), FrameType::IDR | FrameType::I);
                // Drop bitstream to release borrow on self.encoder
                drop(bitstream);

                // Now we can safely mutate self
                if !data.is_empty() {
                    // Also check NAL units for IDR
                    let contains_idr = contains_keyframe(&data);
                    let is_key = is_keyframe || contains_idr;

                    // Extract SPS/PPS from keyframes
                    if is_key && !self.has_parameter_sets() {
                        self.extract_parameter_sets(&data);
                    }

                    // Create packet
                    let mut packet = Packet::new_video(self.stream_index, Buffer::from_vec(data));
                    packet.pts = pts;
                    packet.dts = pts; // For simple encoding, DTS == PTS
                    packet.flags.keyframe = is_key;

                    self.packet_queue.push_back(packet);
                }

                self.frame_count += 1;
                Ok(())
            }
            Frame::Audio(_) => Err(Error::codec("H.264 encoder only accepts video frames")),
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if let Some(packet) = self.packet_queue.pop_front() {
            return Ok(packet);
        }
        Err(Error::TryAgain)
    }

    fn flush(&mut self) -> Result<()> {
        // OpenH264 encoder doesn't require explicit flushing
        // All frames are encoded synchronously
        Ok(())
    }
}

impl Default for H264Encoder {
    fn default() -> Self {
        Self::new(1920, 1080).expect("Failed to create default H264Encoder")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::frame::VideoFrame;
    use crate::util::Timestamp;

    fn create_test_yuv_frame(width: u32, height: u32, pts: i64) -> VideoFrame {
        let mut frame = VideoFrame::new(width, height, PixelFormat::YUV420P);
        frame.pts = Timestamp::new(pts);
        frame.keyframe = pts == 0;

        // Y plane
        let y_size = (width * height) as usize;
        frame.data.push(Buffer::from_vec(vec![128u8; y_size]));
        frame.linesize.push(width as usize);

        // U plane
        let uv_size = ((width / 2) * (height / 2)) as usize;
        frame.data.push(Buffer::from_vec(vec![128u8; uv_size]));
        frame.linesize.push((width / 2) as usize);

        // V plane
        frame.data.push(Buffer::from_vec(vec![128u8; uv_size]));
        frame.linesize.push((width / 2) as usize);

        frame
    }

    #[test]
    fn test_h264_encoder_creation() {
        let encoder = H264Encoder::new(640, 480);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_h264_encoder_with_bitrate() {
        let encoder = H264Encoder::with_bitrate(1920, 1080, 5_000_000);
        assert!(encoder.is_ok());

        let encoder = encoder.unwrap();
        assert_eq!(encoder.bitrate(), 5_000_000);
    }

    #[test]
    fn test_h264_encoder_with_config() {
        let config = H264EncoderConfig {
            bitrate: 2_000_000,
            max_frame_rate: 60.0,
            rate_control_mode: RateControlMode::Bitrate,
            scene_change_detect: true,
            keyframe_interval: 30,
            threads: 4,
        };

        let encoder = H264Encoder::with_config(1280, 720, config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_h264_encoder_dimensions() {
        let encoder = H264Encoder::new(1280, 720).unwrap();
        assert_eq!(encoder.dimensions(), (1280, 720));
    }

    #[test]
    fn test_h264_encoder_send_receive() {
        let mut encoder = H264Encoder::new(320, 240).unwrap();

        // Initially no parameter sets
        assert!(!encoder.has_parameter_sets());

        // Send a frame
        let frame = create_test_yuv_frame(320, 240, 0);
        let result = encoder.send_frame(&Frame::Video(frame));
        assert!(result.is_ok());

        // Should have packet available
        let packet_result = encoder.receive_packet();
        assert!(packet_result.is_ok());

        let packet = packet_result.unwrap();
        assert!(!packet.data.is_empty());
        assert!(packet.is_keyframe()); // First frame should be keyframe
    }

    #[test]
    fn test_h264_encoder_sps_pps_extraction() {
        let mut encoder = H264Encoder::new(320, 240).unwrap();

        // Send a frame to trigger SPS/PPS extraction
        let frame = create_test_yuv_frame(320, 240, 0);
        encoder.send_frame(&Frame::Video(frame)).unwrap();

        // Receive the packet
        let _ = encoder.receive_packet();

        // Check parameter sets are extracted
        assert!(encoder.has_parameter_sets());
        assert!(encoder.sps().is_some());
        assert!(encoder.pps().is_some());
        assert!(encoder.extradata().is_some());

        // Verify SPS NAL type
        let sps = encoder.sps().unwrap();
        assert!(!sps.is_empty());
        assert_eq!(sps[0] & 0x1F, super::super::nal::NAL_TYPE_SPS);

        // Verify PPS NAL type
        let pps = encoder.pps().unwrap();
        assert!(!pps.is_empty());
        assert_eq!(pps[0] & 0x1F, super::super::nal::NAL_TYPE_PPS);
    }

    #[test]
    fn test_h264_encoder_rejects_audio() {
        use crate::codec::frame::AudioFrame;
        use crate::util::SampleFormat;

        let mut encoder = H264Encoder::new(640, 480).unwrap();
        let audio_frame = AudioFrame::new(1024, 48000, 2, SampleFormat::I16);

        let result = encoder.send_frame(&Frame::Audio(audio_frame));
        assert!(result.is_err());
    }

    #[test]
    fn test_h264_encoder_rejects_wrong_format() {
        let mut encoder = H264Encoder::new(320, 240).unwrap();

        let mut frame = VideoFrame::new(320, 240, PixelFormat::RGB24);
        frame.data.push(Buffer::from_vec(vec![0u8; 320 * 240 * 3]));
        frame.linesize.push(320 * 3);

        let result = encoder.send_frame(&Frame::Video(frame));
        assert!(result.is_err());
    }

    #[test]
    fn test_h264_encoder_force_keyframe() {
        let mut encoder = H264Encoder::new(320, 240).unwrap();

        // Encode first frame (will be keyframe)
        let frame1 = create_test_yuv_frame(320, 240, 0);
        encoder.send_frame(&Frame::Video(frame1)).unwrap();
        let packet1 = encoder.receive_packet().unwrap();
        assert!(packet1.is_keyframe());

        // Encode second frame (normally P-frame)
        let frame2 = create_test_yuv_frame(320, 240, 1);
        encoder.send_frame(&Frame::Video(frame2)).unwrap();
        let _packet2 = encoder.receive_packet().unwrap();

        // Force keyframe for third frame
        encoder.force_keyframe();
        let frame3 = create_test_yuv_frame(320, 240, 2);
        encoder.send_frame(&Frame::Video(frame3)).unwrap();
        let packet3 = encoder.receive_packet().unwrap();
        assert!(packet3.is_keyframe());
    }

    #[test]
    fn test_h264_encoder_flush() {
        let mut encoder = H264Encoder::new(320, 240).unwrap();
        assert!(encoder.flush().is_ok());
    }
}
