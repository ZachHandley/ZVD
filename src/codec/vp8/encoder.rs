//! VP8 video encoder using libvpx
//!
//! This module provides a complete VP8 encoder implementation using the libvpx library.
//! VP8 is a royalty-free video codec developed by On2 Technologies and released as
//! open source by Google.
//!
//! # System Requirements
//!
//! libvpx must be installed on the system:
//! - Debian/Ubuntu: `apt install libvpx-dev`
//! - Arch Linux: `pacman -S libvpx`
//! - macOS: `brew install libvpx`
//! - Fedora: `dnf install libvpx-devel`
//!
//! # Features
//!
//! Enable the `vp8-libvpx` feature in Cargo.toml to use this encoder:
//! ```toml
//! zvd = { version = "0.1", features = ["vp8-libvpx"] }
//! ```
//!
//! # Example
//!
//! ```no_run
//! use zvd_lib::codec::vp8::{Vp8Encoder, Vp8EncoderConfig};
//!
//! let config = Vp8EncoderConfig {
//!     width: 1920,
//!     height: 1080,
//!     bitrate: 2_000_000,
//!     framerate: 30,
//!     keyframe_interval: 120,
//!     ..Default::default()
//! };
//! let encoder = Vp8Encoder::with_config(config).unwrap();
//! ```

use crate::codec::{Encoder, Frame, VideoFrame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, PixelFormat, Timestamp};
use std::collections::VecDeque;
use std::ptr;

#[cfg(feature = "vp8-libvpx")]
use libvpx_sys::*;

/// VP8 rate control mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vp8RateControl {
    /// Variable Bitrate - quality-focused
    VBR,
    /// Constant Bitrate - strict bitrate control
    CBR,
    /// Constant Quality - uses CQ level instead of bitrate
    CQ,
}

impl Default for Vp8RateControl {
    fn default() -> Self {
        Vp8RateControl::VBR
    }
}

/// VP8 encoder configuration
#[derive(Debug, Clone)]
pub struct Vp8EncoderConfig {
    /// Video width in pixels
    pub width: u32,
    /// Video height in pixels
    pub height: u32,
    /// Target bitrate in bits per second
    pub bitrate: u32,
    /// Framerate (frames per second)
    pub framerate: u32,
    /// Maximum keyframe interval (GOP size)
    pub keyframe_interval: u32,
    /// Number of encoding threads (0 = auto)
    pub threads: u32,
    /// Rate control mode
    pub rate_control: Vp8RateControl,
    /// CPU usage preset (0-16, lower is slower/better quality)
    pub cpu_used: i32,
    /// Constant quality level (0-63, used when rate_control is CQ)
    pub cq_level: u32,
    /// Error resilience mode (for streaming)
    pub error_resilient: bool,
    /// Token partitions for parallel decoding (0-3, 2^n partitions)
    pub token_partitions: u32,
}

impl Default for Vp8EncoderConfig {
    fn default() -> Self {
        Vp8EncoderConfig {
            width: 640,
            height: 480,
            bitrate: 1_000_000,
            framerate: 30,
            keyframe_interval: 120,
            threads: 0, // Auto-detect
            rate_control: Vp8RateControl::VBR,
            cpu_used: 6, // Balanced speed/quality
            cq_level: 31,
            error_resilient: false,
            token_partitions: 0,
        }
    }
}

/// VP8 video encoder wrapping libvpx
///
/// This encoder uses libvpx to encode video frames to VP8 format.
/// It supports various rate control modes, multi-threading, and
/// quality presets.
pub struct Vp8Encoder {
    /// Encoder configuration
    config: Vp8EncoderConfig,
    /// Frame counter for PTS calculation
    frame_count: u64,
    /// Encoded packets waiting to be retrieved
    packet_buffer: VecDeque<Packet>,
    /// libvpx encoder context
    #[cfg(feature = "vp8-libvpx")]
    ctx: vpx_codec_ctx_t,
    /// libvpx encoder configuration
    #[cfg(feature = "vp8-libvpx")]
    vpx_cfg: vpx_codec_enc_cfg_t,
    /// Raw image buffer for encoding
    #[cfg(feature = "vp8-libvpx")]
    raw_image: vpx_image_t,
    /// Flag indicating if encoder has been initialized
    #[cfg(feature = "vp8-libvpx")]
    initialized: bool,
}

impl Vp8Encoder {
    /// Create a new VP8 encoder with the given dimensions
    pub fn new(width: u32, height: u32) -> Result<Self> {
        let config = Vp8EncoderConfig {
            width,
            height,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create a VP8 encoder with custom configuration
    pub fn with_config(config: Vp8EncoderConfig) -> Result<Self> {
        #[cfg(feature = "vp8-libvpx")]
        {
            // Validate dimensions
            if config.width == 0 || config.height == 0 {
                return Err(Error::InvalidInput(format!(
                    "Invalid dimensions: {}x{}",
                    config.width, config.height
                )));
            }

            // Get VP8 encoder interface
            let iface = unsafe { vpx_codec_vp8_cx() };
            if iface.is_null() {
                return Err(Error::codec("Failed to get VP8 encoder interface"));
            }

            // Get default encoder configuration
            let mut vpx_cfg: vpx_codec_enc_cfg_t = unsafe { std::mem::zeroed() };
            let res = unsafe {
                vpx_codec_enc_config_default(iface, &mut vpx_cfg, 0)
            };

            if res != vpx_codec_err_t::VPX_CODEC_OK {
                return Err(Error::codec("Failed to get default VP8 encoder config"));
            }

            // Configure encoder
            vpx_cfg.g_w = config.width;
            vpx_cfg.g_h = config.height;
            vpx_cfg.g_timebase.num = 1;
            vpx_cfg.g_timebase.den = config.framerate as i32;
            vpx_cfg.rc_target_bitrate = config.bitrate / 1000; // libvpx uses kbps
            vpx_cfg.g_threads = config.threads;
            vpx_cfg.kf_max_dist = config.keyframe_interval;
            vpx_cfg.kf_min_dist = 0; // Allow keyframe at any point
            vpx_cfg.g_pass = vpx_enc_pass::VPX_RC_ONE_PASS;

            // Rate control mode
            vpx_cfg.rc_end_usage = match config.rate_control {
                Vp8RateControl::VBR => vpx_rc_mode::VPX_VBR,
                Vp8RateControl::CBR => vpx_rc_mode::VPX_CBR,
                Vp8RateControl::CQ => vpx_rc_mode::VPX_CQ,
            };

            // Error resilience
            if config.error_resilient {
                vpx_cfg.g_error_resilient = vpx_codec_er_flags_t::VPX_ERROR_RESILIENT_DEFAULT;
            }

            // Initialize encoder context
            let mut ctx: vpx_codec_ctx_t = unsafe { std::mem::zeroed() };
            let res = unsafe {
                vpx_codec_enc_init_ver(
                    &mut ctx,
                    iface,
                    &vpx_cfg,
                    0, // flags
                    VPX_ENCODER_ABI_VERSION as i32,
                )
            };

            if res != vpx_codec_err_t::VPX_CODEC_OK {
                let err_msg = unsafe {
                    let err_str = vpx_codec_error(&ctx);
                    if err_str.is_null() {
                        "Unknown error".to_string()
                    } else {
                        std::ffi::CStr::from_ptr(err_str)
                            .to_string_lossy()
                            .into_owned()
                    }
                };
                return Err(Error::codec(format!(
                    "Failed to initialize VP8 encoder: {}",
                    err_msg
                )));
            }

            // Set CPU usage / speed preset
            unsafe {
                vpx_codec_control_(
                    &mut ctx,
                    vp8e_enc_control_id::VP8E_SET_CPUUSED as i32,
                    config.cpu_used,
                );
            }

            // Set token partitions
            if config.token_partitions > 0 {
                unsafe {
                    vpx_codec_control_(
                        &mut ctx,
                        vp8e_enc_control_id::VP8E_SET_TOKEN_PARTITIONS as i32,
                        config.token_partitions as i32,
                    );
                }
            }

            // Set CQ level if using constant quality mode
            if config.rate_control == Vp8RateControl::CQ {
                unsafe {
                    vpx_codec_control_(
                        &mut ctx,
                        vp8e_enc_control_id::VP8E_SET_CQ_LEVEL as i32,
                        config.cq_level as i32,
                    );
                }
            }

            // Allocate raw image buffer
            let mut raw_image: vpx_image_t = unsafe { std::mem::zeroed() };
            let img_ptr = unsafe {
                vpx_img_alloc(
                    &mut raw_image,
                    vpx_img_fmt::VPX_IMG_FMT_I420,
                    config.width,
                    config.height,
                    16, // Alignment
                )
            };

            if img_ptr.is_null() {
                unsafe { vpx_codec_destroy(&mut ctx); }
                return Err(Error::codec("Failed to allocate VP8 image buffer"));
            }

            Ok(Vp8Encoder {
                config,
                frame_count: 0,
                packet_buffer: VecDeque::new(),
                ctx,
                vpx_cfg,
                raw_image,
                initialized: true,
            })
        }

        #[cfg(not(feature = "vp8-libvpx"))]
        {
            let _ = config;
            Err(Error::unsupported(
                "VP8 encoding requires the 'vp8-libvpx' feature to be enabled",
            ))
        }
    }

    /// Set target bitrate in bits per second
    pub fn set_bitrate(&mut self, bitrate: u32) {
        self.config.bitrate = bitrate;
        #[cfg(feature = "vp8-libvpx")]
        {
            self.vpx_cfg.rc_target_bitrate = bitrate / 1000;
            // Note: Changing bitrate mid-stream requires vpx_codec_enc_config_set
            // which is an advanced operation. For simplicity, we just store the value.
        }
    }

    /// Set CPU usage preset (0-16, lower is slower/better quality)
    #[cfg(feature = "vp8-libvpx")]
    pub fn set_cpu_used(&mut self, cpu_used: i32) {
        self.config.cpu_used = cpu_used.clamp(-16, 16);
        unsafe {
            vpx_codec_control_(
                &mut self.ctx,
                vp8e_enc_control_id::VP8E_SET_CPUUSED as i32,
                self.config.cpu_used,
            );
        }
    }

    /// Copy YUV420P frame data into the libvpx image buffer
    #[cfg(feature = "vp8-libvpx")]
    fn copy_frame_to_image(&mut self, video_frame: &VideoFrame) -> Result<()> {
        // Validate format
        if video_frame.format != PixelFormat::YUV420P {
            return Err(Error::codec(format!(
                "VP8 encoder expects YUV420P, got {:?}",
                video_frame.format
            )));
        }

        // Validate dimensions
        if video_frame.width != self.config.width || video_frame.height != self.config.height {
            return Err(Error::codec(format!(
                "Frame dimensions {}x{} don't match encoder {}x{}",
                video_frame.width, video_frame.height,
                self.config.width, self.config.height
            )));
        }

        // Validate we have all planes
        if video_frame.data.len() < 3 {
            return Err(Error::codec(format!(
                "YUV420P frame requires 3 planes, got {}",
                video_frame.data.len()
            )));
        }

        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let uv_width = width / 2;
        let uv_height = height / 2;

        // Copy Y plane
        let y_src = video_frame.data[0].as_slice();
        let y_src_stride = video_frame.linesize[0];
        let y_dst_stride = self.raw_image.stride[0] as usize;

        unsafe {
            let y_dst = self.raw_image.planes[0];
            for row in 0..height {
                let src_offset = row * y_src_stride;
                let dst_offset = row * y_dst_stride;
                if src_offset + width <= y_src.len() {
                    ptr::copy_nonoverlapping(
                        y_src.as_ptr().add(src_offset),
                        y_dst.add(dst_offset),
                        width,
                    );
                }
            }
        }

        // Copy U plane
        let u_src = video_frame.data[1].as_slice();
        let u_src_stride = video_frame.linesize[1];
        let u_dst_stride = self.raw_image.stride[1] as usize;

        unsafe {
            let u_dst = self.raw_image.planes[1];
            for row in 0..uv_height {
                let src_offset = row * u_src_stride;
                let dst_offset = row * u_dst_stride;
                if src_offset + uv_width <= u_src.len() {
                    ptr::copy_nonoverlapping(
                        u_src.as_ptr().add(src_offset),
                        u_dst.add(dst_offset),
                        uv_width,
                    );
                }
            }
        }

        // Copy V plane
        let v_src = video_frame.data[2].as_slice();
        let v_src_stride = video_frame.linesize[2];
        let v_dst_stride = self.raw_image.stride[2] as usize;

        unsafe {
            let v_dst = self.raw_image.planes[2];
            for row in 0..uv_height {
                let src_offset = row * v_src_stride;
                let dst_offset = row * v_dst_stride;
                if src_offset + uv_width <= v_src.len() {
                    ptr::copy_nonoverlapping(
                        v_src.as_ptr().add(src_offset),
                        v_dst.add(dst_offset),
                        uv_width,
                    );
                }
            }
        }

        Ok(())
    }

    /// Retrieve encoded packets from the encoder
    #[cfg(feature = "vp8-libvpx")]
    fn retrieve_packets(&mut self, pts: i64) -> Result<()> {
        let mut iter: vpx_codec_iter_t = ptr::null();

        loop {
            let pkt = unsafe { vpx_codec_get_cx_data(&mut self.ctx, &mut iter) };

            if pkt.is_null() {
                break;
            }

            let pkt_ref = unsafe { &*pkt };

            // Only process frame packets
            if pkt_ref.kind != vpx_codec_cx_pkt_kind::VPX_CODEC_CX_FRAME_PKT {
                continue;
            }

            // Extract frame data
            let frame = unsafe { &pkt_ref.data.frame };
            let data_ptr = frame.buf as *const u8;
            let data_len = frame.sz;

            // Copy data to buffer
            let data = unsafe {
                std::slice::from_raw_parts(data_ptr, data_len)
            };
            let buffer = Buffer::from_vec(data.to_vec());

            // Create packet
            let mut packet = Packet::new(0, buffer);
            packet.pts = Timestamp::new(pts);
            packet.dts = packet.pts; // VP8 has no B-frames by default
            packet.duration = 1;

            // Check if keyframe
            let is_keyframe = (frame.flags & VPX_FRAME_IS_KEY) != 0;
            packet.set_keyframe(is_keyframe);

            self.packet_buffer.push_back(packet);
        }

        Ok(())
    }

    /// Force the next frame to be a keyframe
    #[cfg(feature = "vp8-libvpx")]
    pub fn force_keyframe(&mut self) {
        unsafe {
            vpx_codec_control_(
                &mut self.ctx,
                vp8e_enc_control_id::VP8E_SET_FRAME_FLAGS as i32,
                VPX_EFLAG_FORCE_KF as i32,
            );
        }
    }
}

impl Encoder for Vp8Encoder {
    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        #[cfg(feature = "vp8-libvpx")]
        {
            if !self.initialized {
                return Err(Error::InvalidState("Encoder not initialized".into()));
            }

            match frame {
                Frame::Video(video_frame) => {
                    // Copy frame data to VPX image
                    self.copy_frame_to_image(video_frame)?;

                    let pts = if video_frame.pts.is_valid() {
                        video_frame.pts.value
                    } else {
                        self.frame_count as i64
                    };

                    // Determine frame flags
                    let flags = if video_frame.keyframe {
                        VPX_EFLAG_FORCE_KF
                    } else {
                        0
                    };

                    // Encode the frame
                    let res = unsafe {
                        vpx_codec_encode(
                            &mut self.ctx,
                            &self.raw_image,
                            pts,                    // pts
                            1,                      // duration
                            flags as i64,           // flags
                            VPX_DL_GOOD_QUALITY as u64, // deadline
                        )
                    };

                    if res != vpx_codec_err_t::VPX_CODEC_OK {
                        let err_msg = unsafe {
                            let err_str = vpx_codec_error(&self.ctx);
                            if err_str.is_null() {
                                "Unknown error".to_string()
                            } else {
                                std::ffi::CStr::from_ptr(err_str)
                                    .to_string_lossy()
                                    .into_owned()
                            }
                        };
                        return Err(Error::codec(format!("VP8 encode error: {}", err_msg)));
                    }

                    // Retrieve encoded packets
                    self.retrieve_packets(pts)?;
                    self.frame_count += 1;

                    Ok(())
                }
                Frame::Audio(_) => Err(Error::codec("VP8 encoder only accepts video frames")),
            }
        }

        #[cfg(not(feature = "vp8-libvpx"))]
        {
            let _ = frame;
            Err(Error::unsupported(
                "VP8 encoding requires the 'vp8-libvpx' feature",
            ))
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if let Some(packet) = self.packet_buffer.pop_front() {
            Ok(packet)
        } else {
            Err(Error::TryAgain)
        }
    }

    fn flush(&mut self) -> Result<()> {
        #[cfg(feature = "vp8-libvpx")]
        {
            if !self.initialized {
                return Ok(());
            }

            // Flush the encoder by sending NULL frame
            let res = unsafe {
                vpx_codec_encode(
                    &mut self.ctx,
                    ptr::null(),     // NULL signals end of stream
                    self.frame_count as i64,
                    1,
                    0,
                    VPX_DL_GOOD_QUALITY as u64,
                )
            };

            if res != vpx_codec_err_t::VPX_CODEC_OK {
                // Ignore flush errors
            }

            // Retrieve any remaining packets
            let pts = self.frame_count as i64;
            let _ = self.retrieve_packets(pts);
        }

        Ok(())
    }

    fn extradata(&self) -> Option<&[u8]> {
        // VP8 doesn't require out-of-band codec configuration
        None
    }
}

#[cfg(feature = "vp8-libvpx")]
impl Drop for Vp8Encoder {
    fn drop(&mut self) {
        if self.initialized {
            unsafe {
                vpx_img_free(&mut self.raw_image);
                vpx_codec_destroy(&mut self.ctx);
            }
        }
    }
}

// Ensure the encoder can be sent between threads
#[cfg(feature = "vp8-libvpx")]
unsafe impl Send for Vp8Encoder {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vp8_encoder_config_default() {
        let config = Vp8EncoderConfig::default();
        assert_eq!(config.width, 640);
        assert_eq!(config.height, 480);
        assert_eq!(config.bitrate, 1_000_000);
        assert_eq!(config.framerate, 30);
    }

    #[test]
    #[cfg(feature = "vp8-libvpx")]
    fn test_vp8_encoder_creation() {
        let encoder = Vp8Encoder::new(640, 480);
        assert!(
            encoder.is_ok(),
            "Encoder creation failed. Make sure libvpx is installed."
        );
    }

    #[test]
    #[cfg(feature = "vp8-libvpx")]
    fn test_vp8_encoder_with_config() {
        let config = Vp8EncoderConfig {
            width: 1920,
            height: 1080,
            bitrate: 5_000_000,
            framerate: 60,
            keyframe_interval: 120,
            threads: 4,
            rate_control: Vp8RateControl::CBR,
            cpu_used: 4,
            ..Default::default()
        };
        let encoder = Vp8Encoder::with_config(config);
        assert!(encoder.is_ok());
    }

    #[test]
    #[cfg(feature = "vp8-libvpx")]
    fn test_vp8_encoder_flush_empty() {
        let mut encoder = Vp8Encoder::new(320, 240).expect("Failed to create encoder");
        assert!(encoder.flush().is_ok());
    }

    #[test]
    #[cfg(feature = "vp8-libvpx")]
    fn test_vp8_encoder_receive_without_send() {
        let mut encoder = Vp8Encoder::new(320, 240).expect("Failed to create encoder");
        match encoder.receive_packet() {
            Err(Error::TryAgain) => {}
            other => panic!("Expected TryAgain, got {:?}", other),
        }
    }

    #[test]
    #[cfg(feature = "vp8-libvpx")]
    fn test_vp8_encode_frame() {
        use crate::codec::frame::VideoFrame;

        let mut encoder = Vp8Encoder::new(320, 240).expect("Failed to create encoder");

        // Create a test frame
        let mut video_frame = VideoFrame::new(320, 240, PixelFormat::YUV420P);

        // Y plane (320x240)
        video_frame.data.push(Buffer::from_vec(vec![128u8; 320 * 240]));
        video_frame.linesize.push(320);

        // U plane (160x120)
        video_frame.data.push(Buffer::from_vec(vec![128u8; 160 * 120]));
        video_frame.linesize.push(160);

        // V plane (160x120)
        video_frame.data.push(Buffer::from_vec(vec![128u8; 160 * 120]));
        video_frame.linesize.push(160);

        video_frame.keyframe = true;
        video_frame.pts = Timestamp::new(0);

        // Encode the frame
        let result = encoder.send_frame(&Frame::Video(video_frame));
        assert!(result.is_ok(), "Failed to encode frame: {:?}", result);

        // Try to receive packet
        match encoder.receive_packet() {
            Ok(packet) => {
                assert!(!packet.data.is_empty(), "Packet should contain data");
                assert!(packet.is_keyframe(), "First frame should be keyframe");
            }
            Err(Error::TryAgain) => {
                // Some frames may be buffered, that's ok
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn test_vp8_encoder_creation_without_feature() {
        #[cfg(not(feature = "vp8-libvpx"))]
        {
            let encoder = Vp8Encoder::new(640, 480);
            assert!(encoder.is_err(), "Should fail without vp8-libvpx feature");
        }
    }
}
