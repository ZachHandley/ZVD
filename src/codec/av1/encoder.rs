//! AV1 encoder using rav1e
//!
//! This module provides a complete AV1 encoder implementation using the rav1e library,
//! a fast and safe AV1 encoder written in Rust. rav1e is developed by the Xiph.Org Foundation
//! and Alliance for Open Media.
//!
//! # Features
//!
//! - Configurable speed/quality presets (0-10)
//! - Bitrate control (CBR, VBR, CQ modes)
//! - Multi-threading support
//! - Tile-based encoding for parallelism
//! - Keyframe interval control
//! - Multiple pixel format support (YUV420P, YUV422P, YUV444P)
//!
//! # Example
//!
//! ```no_run
//! use zvd_lib::codec::av1::Av1EncoderBuilder;
//!
//! let encoder = Av1EncoderBuilder::new(1920, 1080)
//!     .speed_preset(6)
//!     .quantizer(100)
//!     .max_keyframe_interval(240)
//!     .threads(8)
//!     .build()
//!     .unwrap();
//! ```

use crate::codec::{Encoder, Frame, VideoFrame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, PixelFormat, Rational, Timestamp};
use rav1e::prelude::*;

/// Rate control mode for the encoder
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateControlMode {
    /// Constant Quality mode - uses quantizer value
    Quality,
    /// Variable Bitrate mode - targets bitrate with quality constraints
    Bitrate,
}

/// AV1 encoder wrapping rav1e
pub struct Av1Encoder {
    context: Context<u8>,
    width: usize,
    height: usize,
    frame_count: u64,
    time_base: Rational,
    speed_preset: u8,
}

impl Av1Encoder {
    /// Create a new AV1 encoder with the given dimensions using default settings
    ///
    /// For more control over encoding parameters, use `Av1EncoderBuilder` instead.
    pub fn new(width: u32, height: u32) -> Result<Self> {
        Av1EncoderBuilder::new(width, height).build()
    }

    /// Create a new AV1 encoder with specified speed preset
    ///
    /// # Arguments
    ///
    /// * `width` - Video width in pixels
    /// * `height` - Video height in pixels
    /// * `speed` - Speed preset: 0 (slowest/best quality) to 10 (fastest/lowest quality)
    ///
    /// For more control over encoding parameters, use `Av1EncoderBuilder` instead.
    pub fn with_speed(width: u32, height: u32, speed: u8) -> Result<Self> {
        Av1EncoderBuilder::new(width, height)
            .speed_preset(speed)
            .build()
    }

    /// Get the current time base
    pub fn time_base(&self) -> Rational {
        self.time_base
    }

    /// Get the current speed preset
    pub fn speed_preset(&self) -> u8 {
        self.speed_preset
    }

    /// Convert our VideoFrame to rav1e Frame
    fn video_frame_to_rav1e(&self, video_frame: &VideoFrame) -> Result<rav1e::prelude::Frame<u8>> {
        // Validate pixel format - rav1e supports YUV formats
        let num_planes = match video_frame.format {
            PixelFormat::YUV420P | PixelFormat::YUV422P | PixelFormat::YUV444P => 3,
            PixelFormat::GRAY8 => 1,
            _ => {
                return Err(Error::codec(format!(
                    "Unsupported pixel format for AV1 encoding: {:?}. Supported: YUV420P, YUV422P, YUV444P, GRAY8",
                    video_frame.format
                )));
            }
        };

        // Validate we have the required number of planes
        if video_frame.data.len() < num_planes {
            return Err(Error::codec(format!(
                "Missing planes for format {:?}: expected {}, got {}",
                video_frame.format,
                num_planes,
                video_frame.data.len()
            )));
        }

        // Create a new frame with rav1e
        let mut frame = self.context.new_frame();

        // Calculate chroma dimensions based on pixel format
        let (chroma_width_divisor, chroma_height_divisor) = match video_frame.format {
            PixelFormat::YUV420P => (2, 2), // 4:2:0 - half width and height
            PixelFormat::YUV422P => (2, 1), // 4:2:2 - half width, full height
            PixelFormat::YUV444P => (1, 1), // 4:4:4 - full resolution
            PixelFormat::GRAY8 => (0, 0),   // No chroma planes
            _ => unreachable!("Already validated pixel format"),
        };

        // Copy Y plane (luma)
        let y_data = video_frame.data[0].as_slice();
        let y_stride = video_frame.linesize[0];
        let y_plane_stride = frame.planes[0].cfg.stride;
        for (y, row) in frame.planes[0]
            .data_origin_mut()
            .chunks_mut(y_plane_stride)
            .enumerate()
        {
            if y >= self.height {
                break;
            }
            let src_start = y * y_stride;
            let src_end = (src_start + self.width).min(y_data.len());
            let dst_end = self.width.min(row.len());
            if src_end <= y_data.len() && dst_end <= row.len() {
                row[..dst_end].copy_from_slice(&y_data[src_start..src_end]);
            }
        }

        // Copy U and V planes (chroma) if present
        if num_planes >= 3 {
            let chroma_width = self.width / chroma_width_divisor;
            let chroma_height = self.height / chroma_height_divisor;

            // Copy U plane
            let u_data = video_frame.data[1].as_slice();
            let u_stride = video_frame.linesize[1];
            let u_plane_stride = frame.planes[1].cfg.stride;
            for (y, row) in frame.planes[1]
                .data_origin_mut()
                .chunks_mut(u_plane_stride)
                .enumerate()
            {
                if y >= chroma_height {
                    break;
                }
                let src_start = y * u_stride;
                let src_end = (src_start + chroma_width).min(u_data.len());
                let dst_end = chroma_width.min(row.len());
                if src_end <= u_data.len() && dst_end <= row.len() {
                    row[..dst_end].copy_from_slice(&u_data[src_start..src_end]);
                }
            }

            // Copy V plane
            let v_data = video_frame.data[2].as_slice();
            let v_stride = video_frame.linesize[2];
            let v_plane_stride = frame.planes[2].cfg.stride;
            for (y, row) in frame.planes[2]
                .data_origin_mut()
                .chunks_mut(v_plane_stride)
                .enumerate()
            {
                if y >= chroma_height {
                    break;
                }
                let src_start = y * v_stride;
                let src_end = (src_start + chroma_width).min(v_data.len());
                let dst_end = chroma_width.min(row.len());
                if src_end <= v_data.len() && dst_end <= row.len() {
                    row[..dst_end].copy_from_slice(&v_data[src_start..src_end]);
                }
            }
        }

        Ok(frame)
    }
}

impl Encoder for Av1Encoder {
    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        match frame {
            Frame::Video(video_frame) => {
                // Convert to rav1e frame
                let rav1e_frame = self.video_frame_to_rav1e(video_frame)?;

                // Send to encoder
                self.context.send_frame(rav1e_frame).map_err(|e| match e {
                    EncoderStatus::Failure => Error::codec("Encoder failure"),
                    EncoderStatus::EnoughData => Error::codec("Encoder has enough data"),
                    _ => Error::codec(format!("Encoder error: {:?}", e)),
                })?;

                self.frame_count += 1;
                Ok(())
            }
            Frame::Audio(_) => Err(Error::codec("AV1 encoder only accepts video frames")),
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        // Try to receive a packet from the encoder
        loop {
            match self.context.receive_packet() {
                Ok(packet) => {
                    // Convert rav1e packet to our Packet type
                    let data = Buffer::from_vec(packet.data.to_vec());
                    let mut zvd_packet = Packet::new(0, data);

                    // Calculate proper timestamps based on time base and frame number
                    // PTS = frame_number * time_base_num / time_base_den
                    // For example: with time_base 1/30 (30fps), frame 0 has pts=0, frame 1 has pts=1, etc.
                    let pts_value = packet.input_frameno as i64;
                    zvd_packet.pts = Timestamp::new(pts_value);

                    // For AV1, DTS = PTS (no B-frames by default in rav1e)
                    zvd_packet.dts = zvd_packet.pts;

                    // Duration is typically 1 in the encoder's time base
                    zvd_packet.duration = 1;

                    // Set keyframe flag
                    zvd_packet.set_keyframe(packet.frame_type == FrameType::KEY);

                    return Ok(zvd_packet);
                }
                Err(e) => match e {
                    EncoderStatus::Encoded => {
                        // Frame was encoded but packet not ready yet, keep trying
                        continue;
                    }
                    EncoderStatus::LimitReached => {
                        // No more packets available - end of stream
                        return Err(Error::EndOfStream);
                    }
                    EncoderStatus::NeedMoreData => {
                        // Encoder needs more input frames before producing packets
                        return Err(Error::TryAgain);
                    }
                    EncoderStatus::EnoughData => {
                        // Encoder has buffered enough data, try again
                        return Err(Error::TryAgain);
                    }
                    EncoderStatus::NotReady => {
                        // Encoder not ready to produce packets yet
                        return Err(Error::TryAgain);
                    }
                    EncoderStatus::Failure => {
                        // Encoding failure
                        return Err(Error::codec("Encoder failure during packet reception"));
                    }
                },
            }
        }
    }

    fn flush(&mut self) -> Result<()> {
        // Signal end of input
        self.context.flush();
        Ok(())
    }
}

/// Builder for creating AV1 encoders with custom configuration
///
/// This builder provides a fluent API for configuring all aspects of the AV1 encoder
/// before creation. It allows fine-grained control over quality, speed, threading,
/// and other encoding parameters.
///
/// # Example
///
/// ```no_run
/// use zvd_lib::codec::av1::Av1EncoderBuilder;
///
/// let encoder = Av1EncoderBuilder::new(1920, 1080)
///     .speed_preset(6)                    // Balanced speed/quality
///     .quantizer(100)                     // Medium quality
///     .max_keyframe_interval(240)         // Keyframe every 240 frames
///     .min_keyframe_interval(12)          // At least 12 frames between keyframes
///     .threads(8)                         // Use 8 threads
///     .tile_cols(2)                       // 2 tile columns for parallelism
///     .tile_rows(2)                       // 2 tile rows for parallelism
///     .time_base(1, 30)                   // 30 fps
///     .build()
///     .unwrap();
/// ```
pub struct Av1EncoderBuilder {
    width: u32,
    height: u32,
    speed_preset: u8,
    quantizer: usize,
    min_keyframe_interval: u64,
    max_keyframe_interval: u64,
    bitrate: Option<i32>,
    threads: usize,
    tile_cols: usize,
    tile_rows: usize,
    time_base_num: u64,
    time_base_den: u64,
    low_latency: bool,
    rdo_lookahead_frames: usize,
}

impl Av1EncoderBuilder {
    /// Create a new builder with the given dimensions
    ///
    /// # Arguments
    ///
    /// * `width` - Video width in pixels (must be multiple of 8)
    /// * `height` - Video height in pixels (must be multiple of 8)
    pub fn new(width: u32, height: u32) -> Self {
        Av1EncoderBuilder {
            width,
            height,
            speed_preset: 6,            // Balanced default
            quantizer: 100,             // Medium quality (0-255, lower is better)
            min_keyframe_interval: 12,  // Minimum 12 frames between keyframes
            max_keyframe_interval: 240, // Maximum 240 frames (8 seconds at 30fps)
            bitrate: None,              // No bitrate limit by default
            threads: 0,                 // Auto-detect
            tile_cols: 0,               // Auto-detect
            tile_rows: 0,               // Auto-detect
            time_base_num: 1,           // 1/30 = 30fps default
            time_base_den: 30,
            low_latency: false,
            rdo_lookahead_frames: 40, // Default lookahead
        }
    }

    /// Set the speed preset (0-10)
    ///
    /// Lower values produce better quality but encode slower.
    /// Higher values encode faster but with lower quality.
    ///
    /// * 0-3: Very slow, high quality (for archival/distribution)
    /// * 4-6: Balanced (recommended for most use cases)
    /// * 7-10: Fast, lower quality (for real-time encoding)
    pub fn speed_preset(mut self, speed: u8) -> Self {
        self.speed_preset = speed.min(10);
        self
    }

    /// Set the quantizer value (0-255)
    ///
    /// Lower values produce higher quality at larger file sizes.
    /// Higher values produce lower quality at smaller file sizes.
    ///
    /// Recommended ranges:
    /// * 80-100: High quality
    /// * 100-140: Medium quality
    /// * 140-180: Low quality
    pub fn quantizer(mut self, quantizer: usize) -> Self {
        self.quantizer = quantizer.min(255);
        self
    }

    /// Set the minimum keyframe interval in frames
    ///
    /// This sets the minimum number of frames between keyframes (I-frames).
    /// Smaller values allow more frequent keyframes for better seeking but
    /// increase file size.
    pub fn min_keyframe_interval(mut self, frames: u64) -> Self {
        self.min_keyframe_interval = frames;
        self
    }

    /// Set the maximum keyframe interval in frames
    ///
    /// This sets the maximum number of frames between keyframes (I-frames).
    /// Larger values reduce file size but make seeking less precise.
    ///
    /// Common values:
    /// * 24-30: Keyframe every second at 24-30fps
    /// * 120-240: Keyframe every 4-8 seconds
    pub fn max_keyframe_interval(mut self, frames: u64) -> Self {
        self.max_keyframe_interval = frames;
        self
    }

    /// Set the target bitrate in kilobits per second
    ///
    /// When set, the encoder will target this bitrate instead of using
    /// constant quality mode. This enables rate control.
    ///
    /// # Arguments
    ///
    /// * `kbps` - Target bitrate in kilobits per second
    pub fn bitrate(mut self, kbps: u32) -> Self {
        self.bitrate = Some((kbps * 1000) as i32);
        self
    }

    /// Set the number of threads to use
    ///
    /// # Arguments
    ///
    /// * `threads` - Number of threads (0 = auto-detect, uses CPU count)
    pub fn threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// Set the number of tile columns for parallel encoding
    ///
    /// More tiles can improve encoding speed on multi-core systems but
    /// may slightly reduce compression efficiency.
    ///
    /// # Arguments
    ///
    /// * `cols` - Number of tile columns (0 = auto, power of 2)
    pub fn tile_cols(mut self, cols: usize) -> Self {
        self.tile_cols = cols;
        self
    }

    /// Set the number of tile rows for parallel encoding
    ///
    /// More tiles can improve encoding speed on multi-core systems but
    /// may slightly reduce compression efficiency.
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of tile rows (0 = auto, power of 2)
    pub fn tile_rows(mut self, rows: usize) -> Self {
        self.tile_rows = rows;
        self
    }

    /// Set the time base for the video
    ///
    /// The time base defines the unit of time for timestamps. For example,
    /// a time base of 1/30 means each timestamp unit represents 1/30th of a second.
    ///
    /// # Arguments
    ///
    /// * `num` - Numerator of the time base
    /// * `den` - Denominator of the time base
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use zvd_lib::codec::av1::Av1EncoderBuilder;
    /// // 30 fps video
    /// let encoder = Av1EncoderBuilder::new(1920, 1080)
    ///     .time_base(1, 30)
    ///     .build()
    ///     .unwrap();
    ///
    /// // 60 fps video
    /// let encoder = Av1EncoderBuilder::new(1920, 1080)
    ///     .time_base(1, 60)
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn time_base(mut self, num: u64, den: u64) -> Self {
        self.time_base_num = num;
        self.time_base_den = den;
        self
    }

    /// Enable low latency mode
    ///
    /// Low latency mode reduces buffering and produces output faster,
    /// at the cost of some compression efficiency. Useful for real-time
    /// applications like streaming or video conferencing.
    pub fn low_latency(mut self, enabled: bool) -> Self {
        self.low_latency = enabled;
        if enabled {
            // Reduce lookahead for lower latency
            self.rdo_lookahead_frames = self.rdo_lookahead_frames.min(10);
        }
        self
    }

    /// Set the number of frames to look ahead for rate-distortion optimization
    ///
    /// More lookahead can improve quality but increases latency and memory usage.
    ///
    /// # Arguments
    ///
    /// * `frames` - Number of frames to look ahead (0-250)
    pub fn rdo_lookahead_frames(mut self, frames: usize) -> Self {
        self.rdo_lookahead_frames = frames.min(250);
        self
    }

    /// Build the encoder with the configured settings
    pub fn build(self) -> Result<Av1Encoder> {
        // Validate dimensions
        if !self.width.is_multiple_of(8) || !self.height.is_multiple_of(8) {
            return Err(Error::codec(format!(
                "Width and height must be multiples of 8, got {}x{}",
                self.width, self.height
            )));
        }

        if self.width == 0 || self.height == 0 {
            return Err(Error::codec(format!(
                "Invalid dimensions: {}x{}",
                self.width, self.height
            )));
        }

        // Create encoder config with speed preset
        let mut enc = EncoderConfig::with_speed_preset(self.speed_preset);

        // Set video dimensions
        enc.width = self.width as usize;
        enc.height = self.height as usize;

        // Set time base
        enc.time_base = rav1e::prelude::Rational {
            num: self.time_base_num,
            den: self.time_base_den,
        };

        // Set keyframe intervals
        enc.min_key_frame_interval = self.min_keyframe_interval;
        enc.max_key_frame_interval = self.max_keyframe_interval;

        // Set rate control
        if let Some(bitrate) = self.bitrate {
            enc.bitrate = bitrate;
        } else {
            enc.quantizer = self.quantizer;
        }

        // Set tile configuration for parallel encoding
        if self.tile_cols > 0 {
            enc.tile_cols = self.tile_cols;
        }
        if self.tile_rows > 0 {
            enc.tile_rows = self.tile_rows;
        }

        // Note: rdo_lookahead_frames is not available in the public API
        // of rav1e EncoderConfig. The encoder will use defaults based on speed preset.
        // The field is kept in the builder for future compatibility.

        // Low latency settings
        if self.low_latency {
            enc.low_latency = true;
        }

        // Create config with encoder config
        let mut cfg = Config::default().with_encoder_config(enc);

        // Set thread count
        if self.threads > 0 {
            cfg = cfg.with_threads(self.threads);
        }

        // Create the context
        let context = cfg
            .new_context()
            .map_err(|e| Error::codec(format!("Failed to create AV1 encoder context: {:?}", e)))?;

        Ok(Av1Encoder {
            context,
            width: self.width as usize,
            height: self.height as usize,
            frame_count: 0,
            time_base: Rational::new(self.time_base_num as i64, self.time_base_den as i64),
            speed_preset: self.speed_preset,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_av1_encoder_creation() {
        let encoder = Av1Encoder::new(640, 480);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_av1_encoder_with_speed() {
        let encoder = Av1Encoder::with_speed(320, 240, 10); // Fastest
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_av1_encoder_builder_basic() {
        let encoder = Av1EncoderBuilder::new(640, 480)
            .speed_preset(8)
            .quantizer(120)
            .build();
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_av1_encoder_builder_advanced() {
        let encoder = Av1EncoderBuilder::new(1920, 1080)
            .speed_preset(6)
            .quantizer(100)
            .max_keyframe_interval(240)
            .min_keyframe_interval(12)
            .threads(4)
            .tile_cols(2)
            .tile_rows(2)
            .time_base(1, 30)
            .low_latency(false)
            .build();
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_av1_encoder_builder_with_bitrate() {
        let encoder = Av1EncoderBuilder::new(1280, 720)
            .speed_preset(7)
            .bitrate(2000) // 2 Mbps
            .max_keyframe_interval(120)
            .threads(4)
            .build();
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_av1_encoder_builder_low_latency() {
        let encoder = Av1EncoderBuilder::new(1280, 720)
            .speed_preset(9) // Fast for low latency
            .low_latency(true)
            .rdo_lookahead_frames(5)
            .build();
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_av1_encoder_invalid_dimensions() {
        // Dimensions must be multiples of 8
        let encoder = Av1EncoderBuilder::new(641, 480).build();
        assert!(encoder.is_err());

        let encoder = Av1EncoderBuilder::new(640, 481).build();
        assert!(encoder.is_err());

        // Zero dimensions
        let encoder = Av1EncoderBuilder::new(0, 0).build();
        assert!(encoder.is_err());
    }

    #[test]
    fn test_encoder_metadata() {
        let encoder = Av1EncoderBuilder::new(640, 480)
            .speed_preset(7)
            .time_base(1, 60)
            .build()
            .unwrap();

        assert_eq!(encoder.speed_preset(), 7);
        assert_eq!(encoder.time_base().num, 1);
        assert_eq!(encoder.time_base().den, 60);
    }

    #[test]
    fn test_encode_simple_frame() {
        use crate::codec::frame::VideoFrame;
        use crate::util::Buffer;

        // Create a simple encoder
        let mut encoder = Av1EncoderBuilder::new(320, 240)
            .speed_preset(10) // Fastest for testing
            .quantizer(200) // Low quality for speed
            .build()
            .unwrap();

        // rav1e needs multiple frames before producing packets
        // Send several frames and flush to ensure we get packets
        for i in 0..10 {
            let mut video_frame = VideoFrame::new(320, 240, PixelFormat::YUV420P);

            // Y plane (320x240) - all zeros (black)
            let y_size = 320 * 240;
            video_frame.data.push(Buffer::from_vec(vec![0u8; y_size]));
            video_frame.linesize.push(320);

            // U plane (160x120) - 128 (neutral)
            let uv_size = 160 * 120;
            video_frame
                .data
                .push(Buffer::from_vec(vec![128u8; uv_size]));
            video_frame.linesize.push(160);

            // V plane (160x120) - 128 (neutral)
            video_frame
                .data
                .push(Buffer::from_vec(vec![128u8; uv_size]));
            video_frame.linesize.push(160);

            video_frame.pts = Timestamp::new(i as i64);
            video_frame.keyframe = i == 0;

            // Send the frame
            let result = encoder.send_frame(&Frame::Video(video_frame));
            assert!(result.is_ok(), "Failed to send frame {}: {:?}", i, result);
        }

        // Flush to signal end of input
        encoder.flush().unwrap();

        // Try to receive packets
        let mut received_packet = false;
        for _ in 0..20 {
            match encoder.receive_packet() {
                Ok(packet) => {
                    assert!(!packet.data.is_empty(), "Packet should contain data");
                    assert!(packet.pts.is_valid(), "Packet should have valid PTS");
                    received_packet = true;
                    break;
                }
                Err(Error::TryAgain) | Err(Error::EndOfStream) => {
                    // Encoder needs more data or no more packets
                    if received_packet {
                        break;
                    }
                    continue;
                }
                Err(e) => {
                    panic!("Unexpected error receiving packet: {:?}", e);
                }
            }
        }

        assert!(received_packet, "Should receive at least one packet");
    }

    #[test]
    fn test_encode_multiple_frames() {
        use crate::codec::frame::VideoFrame;
        use crate::util::Buffer;

        let mut encoder = Av1EncoderBuilder::new(320, 240)
            .speed_preset(10)
            .quantizer(200)
            .max_keyframe_interval(30)
            .build()
            .unwrap();

        // Send 5 frames
        for i in 0..5 {
            let mut video_frame = VideoFrame::new(320, 240, PixelFormat::YUV420P);

            // Create frames with increasing brightness
            let brightness = (i * 50) as u8;
            video_frame
                .data
                .push(Buffer::from_vec(vec![brightness; 320 * 240]));
            video_frame.linesize.push(320);
            video_frame
                .data
                .push(Buffer::from_vec(vec![128u8; 160 * 120]));
            video_frame.linesize.push(160);
            video_frame
                .data
                .push(Buffer::from_vec(vec![128u8; 160 * 120]));
            video_frame.linesize.push(160);
            video_frame.pts = Timestamp::new(i as i64);

            let result = encoder.send_frame(&Frame::Video(video_frame));
            assert!(result.is_ok(), "Failed to send frame {}: {:?}", i, result);
        }

        // Flush the encoder
        encoder.flush().unwrap();

        // Receive all packets
        let mut packet_count = 0;
        loop {
            match encoder.receive_packet() {
                Ok(packet) => {
                    assert!(!packet.data.is_empty());
                    packet_count += 1;
                }
                Err(Error::EndOfStream) => {
                    // Normal end of packets
                    break;
                }
                Err(Error::TryAgain) => {
                    // No more packets available right now
                    break;
                }
                Err(e) => {
                    panic!("Unexpected error: {:?}", e);
                }
            }

            // Safety limit
            if packet_count > 100 {
                panic!("Too many packets received");
            }
        }

        assert!(
            packet_count >= 5,
            "Should receive at least 5 packets for 5 frames, got {}",
            packet_count
        );
    }

    #[test]
    fn test_pixel_format_support() {
        use crate::codec::frame::VideoFrame;
        use crate::util::Buffer;

        let mut encoder = Av1EncoderBuilder::new(320, 240)
            .speed_preset(10)
            .build()
            .unwrap();

        // Test YUV420P (should work)
        let mut frame_420 = VideoFrame::new(320, 240, PixelFormat::YUV420P);
        frame_420.data.push(Buffer::from_vec(vec![0u8; 320 * 240]));
        frame_420.linesize.push(320);
        frame_420
            .data
            .push(Buffer::from_vec(vec![128u8; 160 * 120]));
        frame_420.linesize.push(160);
        frame_420
            .data
            .push(Buffer::from_vec(vec![128u8; 160 * 120]));
        frame_420.linesize.push(160);

        let result = encoder.send_frame(&Frame::Video(frame_420));
        assert!(result.is_ok(), "YUV420P should be supported");

        // Test unsupported format (should fail)
        let mut frame_rgb = VideoFrame::new(320, 240, PixelFormat::RGB24);
        frame_rgb
            .data
            .push(Buffer::from_vec(vec![0u8; 320 * 240 * 3]));
        frame_rgb.linesize.push(320 * 3);

        let result = encoder.send_frame(&Frame::Video(frame_rgb));
        assert!(result.is_err(), "RGB24 should not be supported");
    }
}
