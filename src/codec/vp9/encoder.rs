//! VP9 video encoder using libvpx
//!
//! This module provides a complete VP9 encoder implementation using the libvpx library.
//! The vpx-sys crate provides FFI bindings to libvpx.
//!
//! # System Requirements
//!
//! libvpx must be installed on the system:
//! - Debian/Ubuntu: `apt install libvpx-dev`
//! - Arch Linux: `pacman -S libvpx`
//! - macOS: `brew install libvpx`
//! - Fedora: `dnf install libvpx-devel`

use crate::codec::{Encoder, Frame, VideoFrame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, PixelFormat, Rational, Timestamp};
use std::ptr;

#[cfg(feature = "vp9-codec")]
use vpx_sys::*;

/// Rate control mode for the encoder
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateControlMode {
    /// Variable Bitrate mode
    VBR,
    /// Constant Bitrate mode
    CBR,
    /// Constrained Quality mode
    CQ,
}

/// VP9 encoder configuration
#[derive(Debug, Clone)]
pub struct Vp9EncoderConfig {
    pub width: u32,
    pub height: u32,
    pub bitrate: u32,
    pub framerate: Rational,
    pub keyframe_interval: u32,
    pub threads: u32,
    pub rc_mode: RateControlMode,
    /// CPU speed (0-9, higher is faster but lower quality)
    pub cpu_used: i32,
    /// Target quality (0-63, lower is better)
    pub quality: u32,
    /// Number of tile columns (log2)
    pub tile_columns: u32,
    /// Enable lossless mode
    pub lossless: bool,
}

impl Default for Vp9EncoderConfig {
    fn default() -> Self {
        Vp9EncoderConfig {
            width: 640,
            height: 480,
            bitrate: 1_000_000,
            framerate: Rational::new(30, 1),
            keyframe_interval: 60,
            threads: 0, // auto-detect
            rc_mode: RateControlMode::VBR,
            cpu_used: 5, // Balanced speed/quality
            quality: 20,
            tile_columns: 0, // Auto
            lossless: false,
        }
    }
}

/// VP9 video encoder wrapping libvpx
///
/// This encoder uses libvpx's VP9 encoder interface to encode VP9 video streams.
/// VP9 offers better compression than VP8 and supports advanced features like
/// tile-based encoding, lossless mode, and higher bit depths.
#[cfg(feature = "vp9-codec")]
pub struct Vp9Encoder {
    /// libvpx encoder context
    ctx: vpx_codec_ctx_t,
    /// libvpx image for input frames
    img: vpx_image_t,
    /// Encoder configuration
    config: Vp9EncoderConfig,
    /// Whether the encoder has been initialized
    initialized: bool,
    /// Frame counter
    frame_count: u64,
    /// Buffered packets waiting to be retrieved
    packet_buffer: Vec<Packet>,
}

#[cfg(feature = "vp9-codec")]
impl Vp9Encoder {
    /// Create a new VP9 encoder with default settings
    pub fn new(width: u32, height: u32) -> Result<Self> {
        let config = Vp9EncoderConfig {
            width,
            height,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create a VP9 encoder with custom configuration
    pub fn with_config(config: Vp9EncoderConfig) -> Result<Self> {
        unsafe {
            // Get VP9 encoder interface
            let iface = vpx_codec_vp9_cx();
            if iface.is_null() {
                return Err(Error::codec("Failed to get VP9 encoder interface"));
            }

            // Initialize encoder configuration
            let mut cfg: vpx_codec_enc_cfg_t = std::mem::zeroed();

            // Get default configuration
            let ret = vpx_codec_enc_config_default(iface, &mut cfg, 0);
            if ret != VPX_CODEC_OK {
                return Err(Error::codec(format!(
                    "Failed to get default VP9 encoder config: {}",
                    ret
                )));
            }

            // Set configuration parameters
            cfg.g_w = config.width;
            cfg.g_h = config.height;
            cfg.g_timebase.num = config.framerate.den as i32;
            cfg.g_timebase.den = config.framerate.num as i32;
            cfg.rc_target_bitrate = config.bitrate / 1000; // kbps
            cfg.g_threads = config.threads;
            cfg.g_lag_in_frames = 0; // No frame buffering for low latency

            // Rate control mode
            cfg.rc_end_usage = match config.rc_mode {
                RateControlMode::VBR => vpx_rc_mode::VPX_VBR,
                RateControlMode::CBR => vpx_rc_mode::VPX_CBR,
                RateControlMode::CQ => vpx_rc_mode::VPX_CQ,
            };

            // Quality settings
            cfg.rc_min_quantizer = config.quality.min(63);
            cfg.rc_max_quantizer = 63;

            // Keyframe settings
            cfg.kf_mode = vpx_kf_mode::VPX_KF_AUTO;
            cfg.kf_max_dist = config.keyframe_interval;

            // Initialize encoder context
            let mut ctx: vpx_codec_ctx_t = std::mem::zeroed();
            let ret = vpx_codec_enc_init_ver(
                &mut ctx,
                iface,
                &cfg,
                0, // flags
                VPX_ENCODER_ABI_VERSION as i32,
            );

            if ret != VPX_CODEC_OK {
                let error_str = if !ctx.err_detail.is_null() {
                    std::ffi::CStr::from_ptr(ctx.err_detail)
                        .to_string_lossy()
                        .into_owned()
                } else {
                    format!("Error code: {}", ret)
                };
                return Err(Error::codec(format!(
                    "Failed to initialize VP9 encoder: {}",
                    error_str
                )));
            }

            // Set VP9-specific controls
            // CPU speed (0-9, higher is faster)
            let ret = vpx_codec_control_(
                &mut ctx,
                vp8e_enc_control_id::VP8E_SET_CPUUSED as i32,
                config.cpu_used,
            );
            if ret != VPX_CODEC_OK {
                vpx_codec_destroy(&mut ctx);
                return Err(Error::codec("Failed to set CPU speed"));
            }

            // Tile columns for parallel encoding
            if config.tile_columns > 0 {
                let ret = vpx_codec_control_(
                    &mut ctx,
                    vp9e_enc_control_id::VP9E_SET_TILE_COLUMNS as i32,
                    config.tile_columns as i32,
                );
                if ret != VPX_CODEC_OK {
                    vpx_codec_destroy(&mut ctx);
                    return Err(Error::codec("Failed to set tile columns"));
                }
            }

            // Lossless mode
            if config.lossless {
                let ret = vpx_codec_control_(
                    &mut ctx,
                    vp9e_enc_control_id::VP9E_SET_LOSSLESS as i32,
                    1,
                );
                if ret != VPX_CODEC_OK {
                    vpx_codec_destroy(&mut ctx);
                    return Err(Error::codec("Failed to enable lossless mode"));
                }
            }

            // Create image for input frames (YUV420P)
            let img = vpx_img_alloc(
                ptr::null_mut(),
                vpx_img_fmt::VPX_IMG_FMT_I420,
                config.width,
                config.height,
                1, // alignment
            );

            if img.is_null() {
                vpx_codec_destroy(&mut ctx);
                return Err(Error::codec("Failed to allocate VP9 image"));
            }

            Ok(Vp9Encoder {
                ctx,
                img: *img,
                config,
                initialized: true,
                frame_count: 0,
                packet_buffer: Vec::new(),
            })
        }
    }

    /// Set target bitrate (in bits per second)
    pub fn set_bitrate(&mut self, bitrate: u32) {
        self.config.bitrate = bitrate;
    }

    /// Copy VideoFrame data to vpx image
    fn copy_frame_to_image(&mut self, video_frame: &VideoFrame) -> Result<()> {
        // Validate format - VP9 expects YUV420P for 8-bit
        if video_frame.format != PixelFormat::YUV420P {
            return Err(Error::codec(format!(
                "VP9 encoder expects YUV420P, got {:?}",
                video_frame.format
            )));
        }

        // Validate dimensions
        if video_frame.width != self.config.width || video_frame.height != self.config.height {
            return Err(Error::codec(format!(
                "Frame dimensions {}x{} don't match encoder {}x{}",
                video_frame.width, video_frame.height, self.config.width, self.config.height
            )));
        }

        unsafe {
            // Copy Y plane
            let y_src = video_frame.data[0].as_slice();
            let y_dst = std::slice::from_raw_parts_mut(
                self.img.planes[0],
                (self.img.stride[0] as u32 * self.config.height) as usize,
            );
            let y_src_stride = video_frame.linesize[0];
            let y_dst_stride = self.img.stride[0] as usize;

            for y in 0..self.config.height as usize {
                let src_start = y * y_src_stride;
                let src_end = src_start + self.config.width as usize;
                let dst_start = y * y_dst_stride;
                let dst_end = dst_start + self.config.width as usize;

                if src_end <= y_src.len() && dst_end <= y_dst.len() {
                    y_dst[dst_start..dst_end].copy_from_slice(&y_src[src_start..src_end]);
                }
            }

            // Copy U and V planes (half resolution for 4:2:0)
            let uv_height = (self.config.height / 2) as usize;
            let uv_width = (self.config.width / 2) as usize;

            // U plane
            let u_src = video_frame.data[1].as_slice();
            let u_dst = std::slice::from_raw_parts_mut(
                self.img.planes[1],
                (self.img.stride[1] as u32 * (self.config.height / 2)) as usize,
            );
            let u_src_stride = video_frame.linesize[1];
            let u_dst_stride = self.img.stride[1] as usize;

            for y in 0..uv_height {
                let src_start = y * u_src_stride;
                let src_end = src_start + uv_width;
                let dst_start = y * u_dst_stride;
                let dst_end = dst_start + uv_width;

                if src_end <= u_src.len() && dst_end <= u_dst.len() {
                    u_dst[dst_start..dst_end].copy_from_slice(&u_src[src_start..src_end]);
                }
            }

            // V plane
            let v_src = video_frame.data[2].as_slice();
            let v_dst = std::slice::from_raw_parts_mut(
                self.img.planes[2],
                (self.img.stride[2] as u32 * (self.config.height / 2)) as usize,
            );
            let v_src_stride = video_frame.linesize[2];
            let v_dst_stride = self.img.stride[2] as usize;

            for y in 0..uv_height {
                let src_start = y * v_src_stride;
                let src_end = src_start + uv_width;
                let dst_start = y * v_dst_stride;
                let dst_end = dst_start + uv_width;

                if src_end <= v_src.len() && dst_end <= v_dst.len() {
                    v_dst[dst_start..dst_end].copy_from_slice(&v_src[src_start..src_end]);
                }
            }
        }

        Ok(())
    }

    /// Retrieve all available encoded packets
    fn retrieve_packets(&mut self) -> Result<()> {
        unsafe {
            let mut iter: vpx_codec_iter_t = ptr::null_mut();

            loop {
                let pkt = vpx_codec_get_cx_data(&mut self.ctx, &mut iter);
                if pkt.is_null() {
                    break;
                }

                let pkt_ref = &*pkt;

                // Only process frame packets
                if pkt_ref.kind == vpx_codec_cx_pkt_kind::VPX_CODEC_CX_FRAME_PKT {
                    let frame_pkt = &pkt_ref.data.frame;

                    // Copy packet data
                    let data = std::slice::from_raw_parts(
                        frame_pkt.buf as *const u8,
                        frame_pkt.sz as usize,
                    );

                    let packet = Packet {
                        data: Buffer::from_vec(data.to_vec()),
                        pts: Timestamp::new(frame_pkt.pts),
                        dts: Timestamp::new(frame_pkt.pts), // VP9 has no B-frames in single-pass, PTS == DTS
                        duration: 1, // Duration of one frame
                        is_keyframe: (frame_pkt.flags & VPX_FRAME_IS_KEY) != 0,
                        stream_index: 0,
                    };

                    self.packet_buffer.push(packet);
                }
            }
        }

        Ok(())
    }
}

#[cfg(feature = "vp9-codec")]
impl Drop for Vp9Encoder {
    fn drop(&mut self) {
        if self.initialized {
            unsafe {
                vpx_img_free(&mut self.img);
                vpx_codec_destroy(&mut self.ctx);
            }
            self.initialized = false;
        }
    }
}

#[cfg(feature = "vp9-codec")]
impl Default for Vp9Encoder {
    fn default() -> Self {
        Self::new(640, 480).expect("Failed to create default VP9 encoder")
    }
}

#[cfg(feature = "vp9-codec")]
impl Encoder for Vp9Encoder {
    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        match frame {
            Frame::Video(video_frame) => {
                // Copy frame data to vpx image
                self.copy_frame_to_image(video_frame)?;

                unsafe {
                    // Encode the frame
                    let ret = vpx_codec_encode(
                        &mut self.ctx,
                        &self.img,
                        self.frame_count as i64,
                        1, // duration
                        0, // flags
                        VPX_DL_GOOD_QUALITY as u64, // deadline (best quality for VP9)
                    );

                    if ret != VPX_CODEC_OK {
                        let error_str = if !self.ctx.err_detail.is_null() {
                            std::ffi::CStr::from_ptr(self.ctx.err_detail)
                                .to_string_lossy()
                                .into_owned()
                        } else {
                            format!("Error code: {}", ret)
                        };
                        return Err(Error::codec(format!("VP9 encoding failed: {}", error_str)));
                    }

                    self.frame_count += 1;

                    // Retrieve encoded packets
                    self.retrieve_packets()?;
                }

                Ok(())
            }
            Frame::Audio(_) => Err(Error::codec("VP9 encoder only accepts video frames")),
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if self.packet_buffer.is_empty() {
            return Err(Error::TryAgain);
        }

        Ok(self.packet_buffer.remove(0))
    }

    fn flush(&mut self) -> Result<()> {
        unsafe {
            // Flush the encoder
            let ret = vpx_codec_encode(
                &mut self.ctx,
                ptr::null(),
                -1, // pts
                1,  // duration
                0,  // flags
                VPX_DL_GOOD_QUALITY as u64,
            );

            if ret != VPX_CODEC_OK {
                let error_str = if !self.ctx.err_detail.is_null() {
                    std::ffi::CStr::from_ptr(self.ctx.err_detail)
                        .to_string_lossy()
                        .into_owned()
                } else {
                    format!("Error code: {}", ret)
                };
                return Err(Error::codec(format!("VP9 flush failed: {}", error_str)));
            }

            // Retrieve any remaining packets
            self.retrieve_packets()?;
        }

        Ok(())
    }
}

// Stub implementation when vp9-codec feature is not enabled
#[cfg(not(feature = "vp9-codec"))]
pub struct Vp9Encoder {
    _private: (),
}

#[cfg(not(feature = "vp9-codec"))]
#[allow(dead_code)]
pub struct Vp9EncoderConfig {
    _private: (),
}

#[cfg(not(feature = "vp9-codec"))]
impl Vp9Encoder {
    pub fn new(_width: u32, _height: u32) -> Result<Self> {
        Err(Error::unsupported(
            "VP9 codec support not enabled. Enable the 'vp9-codec' feature and ensure libvpx is installed."
        ))
    }

    #[allow(dead_code)]
    pub fn with_config(_config: Vp9EncoderConfig) -> Result<Self> {
        Err(Error::unsupported("VP9 codec not enabled"))
    }
}

#[cfg(not(feature = "vp9-codec"))]
impl Encoder for Vp9Encoder {
    fn send_frame(&mut self, _frame: &Frame) -> Result<()> {
        Err(Error::unsupported("VP9 codec not enabled"))
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        Err(Error::unsupported("VP9 codec not enabled"))
    }

    fn flush(&mut self) -> Result<()> {
        Err(Error::unsupported("VP9 codec not enabled"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "vp9-codec")]
    fn test_vp9_encoder_creation() {
        let encoder = Vp9Encoder::new(640, 480);
        assert!(
            encoder.is_ok(),
            "Encoder creation failed. Make sure libvpx is installed."
        );
    }

    #[test]
    #[cfg(feature = "vp9-codec")]
    fn test_vp9_encoder_with_config() {
        let config = Vp9EncoderConfig {
            width: 1920,
            height: 1080,
            bitrate: 5_000_000,
            framerate: Rational::new(60, 1),
            keyframe_interval: 120,
            threads: 4,
            rc_mode: RateControlMode::VBR,
            cpu_used: 4,
            quality: 15,
            tile_columns: 2,
            lossless: false,
        };
        let encoder = Vp9Encoder::with_config(config);
        assert!(
            encoder.is_ok(),
            "Encoder creation failed. Make sure libvpx is installed."
        );
    }

    #[test]
    #[cfg(feature = "vp9-codec")]
    fn test_flush() {
        let mut encoder = Vp9Encoder::new(640, 480).expect("Failed to create encoder");
        // Flush should not error
        assert!(encoder.flush().is_ok());
    }

    #[test]
    #[cfg(not(feature = "vp9-codec"))]
    fn test_vp9_disabled() {
        let encoder = Vp9Encoder::new(640, 480);
        assert!(encoder.is_err());
    }
}
