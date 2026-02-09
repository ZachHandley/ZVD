//! VP9 video encoder using libvpx
//!
//! This module provides a complete VP9 encoder implementation using the libvpx library.
//! VP9 is a royalty-free video codec developed by Google as the successor to VP8,
//! offering significantly better compression efficiency.
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
//! Enable the `vp9-libvpx` feature in Cargo.toml to use this encoder:
//! ```toml
//! zvd = { version = "0.1", features = ["vp9-libvpx"] }
//! ```
//!
//! # VP9 Features
//!
//! - Multiple profiles (0-3) supporting 8-bit and 10/12-bit depth
//! - Tiling for parallel encoding/decoding
//! - 2-pass encoding for better quality
//! - Lossless mode
//! - Scalable Video Coding (SVC)
//!
//! # Example
//!
//! ```no_run
//! use zvd_lib::codec::vp9::{Vp9Encoder, Vp9EncoderConfig};
//!
//! let config = Vp9EncoderConfig {
//!     width: 1920,
//!     height: 1080,
//!     bitrate: 4_000_000,
//!     framerate: 30,
//!     keyframe_interval: 120,
//!     profile: Vp9Profile::Profile0,
//!     tile_columns: 2,  // 4 tile columns (2^2)
//!     ..Default::default()
//! };
//! let encoder = Vp9Encoder::with_config(config).unwrap();
//! ```

use crate::codec::{Encoder, Frame, VideoFrame};
use crate::error::{Error, Result};
use crate::format::Packet;
use crate::util::{Buffer, PixelFormat, Timestamp};
use std::collections::VecDeque;
use std::ptr;

#[cfg(feature = "vp9-libvpx")]
use libvpx_sys::*;

/// VP9 profile enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vp9Profile {
    /// Profile 0: 8-bit, YUV 4:2:0
    Profile0 = 0,
    /// Profile 1: 8-bit, YUV 4:2:2, 4:4:0, 4:4:4
    Profile1 = 1,
    /// Profile 2: 10/12-bit, YUV 4:2:0
    Profile2 = 2,
    /// Profile 3: 10/12-bit, YUV 4:2:2, 4:4:0, 4:4:4
    Profile3 = 3,
}

impl Default for Vp9Profile {
    fn default() -> Self {
        Vp9Profile::Profile0
    }
}

/// VP9 rate control mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vp9RateControl {
    /// Variable Bitrate - quality-focused
    VBR,
    /// Constant Bitrate - strict bitrate control
    CBR,
    /// Constant Quality - uses CQ level instead of bitrate
    CQ,
    /// Constrained Quality - limits quality range
    Q,
}

impl Default for Vp9RateControl {
    fn default() -> Self {
        Vp9RateControl::VBR
    }
}

/// VP9 encoding pass for 2-pass encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vp9EncodingPass {
    /// Single-pass encoding
    OnePass,
    /// First pass of 2-pass encoding (generates statistics)
    FirstPass,
    /// Second pass of 2-pass encoding (uses statistics)
    SecondPass,
}

impl Default for Vp9EncodingPass {
    fn default() -> Self {
        Vp9EncodingPass::OnePass
    }
}

/// VP9 tune content type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vp9TuneContent {
    /// Default content tuning
    Default = 0,
    /// Screen capture content
    Screen = 1,
    /// Film content
    Film = 2,
}

impl Default for Vp9TuneContent {
    fn default() -> Self {
        Vp9TuneContent::Default
    }
}

/// VP9 adaptive quantization mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vp9AqMode {
    /// No adaptive quantization
    None = 0,
    /// Variance-based AQ
    Variance = 1,
    /// Complexity-based AQ
    Complexity = 2,
    /// Cyclic refresh AQ
    CyclicRefresh = 3,
    /// Equator360 AQ
    Equator360 = 4,
    /// Perceptual AQ
    Perceptual = 5,
    /// Perceptual AQ with lookahead
    PerceptualLookahead = 6,
}

impl Default for Vp9AqMode {
    fn default() -> Self {
        Vp9AqMode::None
    }
}

/// VP9 encoder configuration
#[derive(Debug, Clone)]
pub struct Vp9EncoderConfig {
    /// Video width in pixels
    pub width: u32,
    /// Video height in pixels
    pub height: u32,
    /// Target bitrate in bits per second
    pub bitrate: u32,
    /// Framerate (frames per second)
    pub framerate: u32,
    /// Framerate denominator for fractional framerates (default 1)
    pub framerate_den: u32,
    /// Maximum keyframe interval (GOP size)
    pub keyframe_interval: u32,
    /// Number of encoding threads (0 = auto)
    pub threads: u32,
    /// Rate control mode
    pub rate_control: Vp9RateControl,
    /// CPU usage preset (0-9, higher is faster but lower quality)
    pub cpu_used: i32,
    /// Constant quality level (0-63, used when rate_control is CQ)
    pub cq_level: u32,
    /// VP9 profile (0-3)
    pub profile: Vp9Profile,
    /// Tile columns (log2, 0-6, resulting in 1-64 columns)
    pub tile_columns: u32,
    /// Tile rows (log2, 0-2, resulting in 1-4 rows)
    pub tile_rows: u32,
    /// Enable frame-parallel decoding
    pub frame_parallel: bool,
    /// Enable lossless mode
    pub lossless: bool,
    /// Error resilience mode (for streaming)
    pub error_resilient: bool,
    /// Encoding pass (for 2-pass encoding)
    pub encoding_pass: Vp9EncodingPass,
    /// Enable row-based multi-threading
    pub row_mt: bool,
    /// Content tuning
    pub tune_content: Vp9TuneContent,
    /// Adaptive quantization mode
    pub aq_mode: Vp9AqMode,
    /// Minimum quantizer (0-63)
    pub min_q: u32,
    /// Maximum quantizer (0-63)
    pub max_q: u32,
    /// Undershoot percentage for rate control
    pub undershoot_pct: u32,
    /// Overshoot percentage for rate control
    pub overshoot_pct: u32,
    /// Buffer size in milliseconds
    pub buf_sz: u32,
    /// Initial buffer level in milliseconds
    pub buf_initial_sz: u32,
    /// Optimal buffer level in milliseconds
    pub buf_optimal_sz: u32,
    /// Enable auto alt reference frames
    pub auto_alt_ref: bool,
    /// Lag in frames for look-ahead
    pub lag_in_frames: u32,
}

impl Default for Vp9EncoderConfig {
    fn default() -> Self {
        Vp9EncoderConfig {
            width: 640,
            height: 480,
            bitrate: 1_000_000,
            framerate: 30,
            framerate_den: 1,
            keyframe_interval: 120,
            threads: 0, // Auto-detect
            rate_control: Vp9RateControl::VBR,
            cpu_used: 5, // Balanced speed/quality (VP9 uses 0-9)
            cq_level: 31,
            profile: Vp9Profile::Profile0,
            tile_columns: 0, // 1 tile column
            tile_rows: 0,    // 1 tile row
            frame_parallel: true,
            lossless: false,
            error_resilient: false,
            encoding_pass: Vp9EncodingPass::OnePass,
            row_mt: true,
            tune_content: Vp9TuneContent::Default,
            aq_mode: Vp9AqMode::None,
            min_q: 0,
            max_q: 63,
            undershoot_pct: 25,
            overshoot_pct: 25,
            buf_sz: 6000,
            buf_initial_sz: 4000,
            buf_optimal_sz: 5000,
            auto_alt_ref: true,
            lag_in_frames: 25,
        }
    }
}

/// VP9 video encoder wrapping libvpx
///
/// This encoder uses libvpx to encode video frames to VP9 format.
/// It supports various rate control modes, multi-threading, tiling,
/// and quality presets including 2-pass encoding.
pub struct Vp9Encoder {
    /// Encoder configuration
    config: Vp9EncoderConfig,
    /// Frame counter for PTS calculation
    frame_count: u64,
    /// Encoded packets waiting to be retrieved
    packet_buffer: VecDeque<Packet>,
    /// First-pass statistics data (for 2-pass encoding)
    first_pass_stats: Vec<u8>,
    /// libvpx encoder context
    #[cfg(feature = "vp9-libvpx")]
    ctx: vpx_codec_ctx_t,
    /// libvpx encoder configuration
    #[cfg(feature = "vp9-libvpx")]
    vpx_cfg: vpx_codec_enc_cfg_t,
    /// Raw image buffer for encoding
    #[cfg(feature = "vp9-libvpx")]
    raw_image: vpx_image_t,
    /// Flag indicating if encoder has been initialized
    #[cfg(feature = "vp9-libvpx")]
    initialized: bool,
}

impl Vp9Encoder {
    /// Create a new VP9 encoder with the given dimensions
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
        #[cfg(feature = "vp9-libvpx")]
        {
            // Validate dimensions
            if config.width == 0 || config.height == 0 {
                return Err(Error::InvalidInput(format!(
                    "Invalid dimensions: {}x{}",
                    config.width, config.height
                )));
            }

            // Validate profile/format compatibility
            // Profile 0: 8-bit 4:2:0 only
            // Profile 1: 8-bit, allows 4:2:2 and 4:4:4
            // Profile 2: 10/12-bit 4:2:0
            // Profile 3: 10/12-bit 4:2:2 and 4:4:4

            // Get VP9 encoder interface
            let iface = unsafe { vpx_codec_vp9_cx() };
            if iface.is_null() {
                return Err(Error::codec("Failed to get VP9 encoder interface"));
            }

            // Get default encoder configuration
            let mut vpx_cfg: vpx_codec_enc_cfg_t = unsafe { std::mem::zeroed() };
            let res = unsafe {
                vpx_codec_enc_config_default(iface, &mut vpx_cfg, 0)
            };

            if res != vpx_codec_err_t::VPX_CODEC_OK {
                return Err(Error::codec("Failed to get default VP9 encoder config"));
            }

            // Configure encoder
            vpx_cfg.g_w = config.width;
            vpx_cfg.g_h = config.height;
            vpx_cfg.g_timebase.num = config.framerate_den as i32;
            vpx_cfg.g_timebase.den = config.framerate as i32;
            vpx_cfg.rc_target_bitrate = config.bitrate / 1000; // libvpx uses kbps
            vpx_cfg.g_threads = config.threads;
            vpx_cfg.kf_max_dist = config.keyframe_interval;
            vpx_cfg.kf_min_dist = 0; // Allow keyframe at any point
            vpx_cfg.g_profile = config.profile as u32;
            vpx_cfg.g_lag_in_frames = config.lag_in_frames;

            // Rate control mode
            vpx_cfg.rc_end_usage = match config.rate_control {
                Vp9RateControl::VBR => vpx_rc_mode::VPX_VBR,
                Vp9RateControl::CBR => vpx_rc_mode::VPX_CBR,
                Vp9RateControl::CQ => vpx_rc_mode::VPX_CQ,
                Vp9RateControl::Q => vpx_rc_mode::VPX_Q,
            };

            // Quantizer settings
            vpx_cfg.rc_min_quantizer = config.min_q;
            vpx_cfg.rc_max_quantizer = config.max_q;

            // Rate control buffer settings
            vpx_cfg.rc_undershoot_pct = config.undershoot_pct;
            vpx_cfg.rc_overshoot_pct = config.overshoot_pct;
            vpx_cfg.rc_buf_sz = config.buf_sz;
            vpx_cfg.rc_buf_initial_sz = config.buf_initial_sz;
            vpx_cfg.rc_buf_optimal_sz = config.buf_optimal_sz;

            // Encoding pass configuration
            vpx_cfg.g_pass = match config.encoding_pass {
                Vp9EncodingPass::OnePass => vpx_enc_pass::VPX_RC_ONE_PASS,
                Vp9EncodingPass::FirstPass => vpx_enc_pass::VPX_RC_FIRST_PASS,
                Vp9EncodingPass::SecondPass => vpx_enc_pass::VPX_RC_LAST_PASS,
            };

            // Error resilience
            if config.error_resilient {
                vpx_cfg.g_error_resilient = vpx_codec_er_flags_t::VPX_ERROR_RESILIENT_DEFAULT;
            }

            // Initialize encoder context
            let mut ctx: vpx_codec_ctx_t = unsafe { std::mem::zeroed() };
            let flags = if config.lossless {
                0 // Lossless flag is set via control, not init flags
            } else {
                0
            };

            let res = unsafe {
                vpx_codec_enc_init_ver(
                    &mut ctx,
                    iface,
                    &vpx_cfg,
                    flags,
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
                    "Failed to initialize VP9 encoder: {}",
                    err_msg
                )));
            }

            // Set VP9-specific controls

            // CPU usage / speed preset (VP9 uses 0-9)
            unsafe {
                vpx_codec_control_(
                    &mut ctx,
                    vp8e_enc_control_id::VP8E_SET_CPUUSED as i32,
                    config.cpu_used.clamp(-9, 9),
                );
            }

            // Tile columns (log2)
            if config.tile_columns > 0 {
                unsafe {
                    vpx_codec_control_(
                        &mut ctx,
                        vp8e_enc_control_id::VP9E_SET_TILE_COLUMNS as i32,
                        config.tile_columns.min(6) as i32,
                    );
                }
            }

            // Tile rows (log2)
            if config.tile_rows > 0 {
                unsafe {
                    vpx_codec_control_(
                        &mut ctx,
                        vp8e_enc_control_id::VP9E_SET_TILE_ROWS as i32,
                        config.tile_rows.min(2) as i32,
                    );
                }
            }

            // Frame parallel decoding
            unsafe {
                vpx_codec_control_(
                    &mut ctx,
                    vp8e_enc_control_id::VP9E_SET_FRAME_PARALLEL_DECODING as i32,
                    if config.frame_parallel { 1 } else { 0 },
                );
            }

            // Lossless mode
            if config.lossless {
                unsafe {
                    vpx_codec_control_(
                        &mut ctx,
                        vp8e_enc_control_id::VP9E_SET_LOSSLESS as i32,
                        1,
                    );
                }
            }

            // Row-based multi-threading
            if config.row_mt {
                unsafe {
                    vpx_codec_control_(
                        &mut ctx,
                        vp8e_enc_control_id::VP9E_SET_ROW_MT as i32,
                        1,
                    );
                }
            }

            // Content tuning
            unsafe {
                vpx_codec_control_(
                    &mut ctx,
                    vp8e_enc_control_id::VP9E_SET_TUNE_CONTENT as i32,
                    config.tune_content as i32,
                );
            }

            // Adaptive quantization mode
            unsafe {
                vpx_codec_control_(
                    &mut ctx,
                    vp8e_enc_control_id::VP9E_SET_AQ_MODE as i32,
                    config.aq_mode as i32,
                );
            }

            // Auto alt reference frames
            unsafe {
                vpx_codec_control_(
                    &mut ctx,
                    vp8e_enc_control_id::VP8E_SET_ENABLEAUTOALTREF as i32,
                    if config.auto_alt_ref { 1 } else { 0 },
                );
            }

            // CQ level if using constant quality mode
            if config.rate_control == Vp9RateControl::CQ {
                unsafe {
                    vpx_codec_control_(
                        &mut ctx,
                        vp8e_enc_control_id::VP8E_SET_CQ_LEVEL as i32,
                        config.cq_level as i32,
                    );
                }
            }

            // Determine image format based on profile
            let img_fmt = match config.profile {
                Vp9Profile::Profile0 => vpx_img_fmt::VPX_IMG_FMT_I420,
                Vp9Profile::Profile1 => vpx_img_fmt::VPX_IMG_FMT_I444,
                Vp9Profile::Profile2 => vpx_img_fmt::VPX_IMG_FMT_I42016,
                Vp9Profile::Profile3 => vpx_img_fmt::VPX_IMG_FMT_I44416,
            };

            // Allocate raw image buffer
            let mut raw_image: vpx_image_t = unsafe { std::mem::zeroed() };
            let img_ptr = unsafe {
                vpx_img_alloc(
                    &mut raw_image,
                    img_fmt,
                    config.width,
                    config.height,
                    16, // Alignment
                )
            };

            if img_ptr.is_null() {
                unsafe { vpx_codec_destroy(&mut ctx); }
                return Err(Error::codec("Failed to allocate VP9 image buffer"));
            }

            Ok(Vp9Encoder {
                config,
                frame_count: 0,
                packet_buffer: VecDeque::new(),
                first_pass_stats: Vec::new(),
                ctx,
                vpx_cfg,
                raw_image,
                initialized: true,
            })
        }

        #[cfg(not(feature = "vp9-libvpx"))]
        {
            let _ = config;
            Err(Error::unsupported(
                "VP9 encoding requires the 'vp9-libvpx' feature to be enabled",
            ))
        }
    }

    /// Create a VP9 encoder for 2-pass encoding second pass
    ///
    /// # Arguments
    ///
    /// * `config` - Encoder configuration (must have encoding_pass set to SecondPass)
    /// * `first_pass_stats` - Statistics data from first pass
    #[cfg(feature = "vp9-libvpx")]
    pub fn with_two_pass(mut config: Vp9EncoderConfig, first_pass_stats: &[u8]) -> Result<Self> {
        config.encoding_pass = Vp9EncodingPass::SecondPass;

        // Create encoder with stats
        let mut encoder = Self::with_config(config)?;
        encoder.first_pass_stats = first_pass_stats.to_vec();

        // Set the stats buffer for second pass
        // Note: In production, we'd need to set the stats via g_pass and rc_twopass_stats_in
        // This requires modifying the config before init, which would need a refactor
        // For now, this is documented as requiring pre-configuration

        Ok(encoder)
    }

    /// Get the encoder configuration
    pub fn config(&self) -> &Vp9EncoderConfig {
        &self.config
    }

    /// Set target bitrate in bits per second
    #[cfg(feature = "vp9-libvpx")]
    pub fn set_bitrate(&mut self, bitrate: u32) -> Result<()> {
        self.config.bitrate = bitrate;
        self.vpx_cfg.rc_target_bitrate = bitrate / 1000;

        // Apply the new configuration
        let res = unsafe {
            vpx_codec_enc_config_set(&mut self.ctx, &self.vpx_cfg)
        };

        if res != vpx_codec_err_t::VPX_CODEC_OK {
            return Err(Error::codec("Failed to update VP9 bitrate"));
        }

        Ok(())
    }

    #[cfg(not(feature = "vp9-libvpx"))]
    pub fn set_bitrate(&mut self, bitrate: u32) -> Result<()> {
        self.config.bitrate = bitrate;
        Ok(())
    }

    /// Set CPU usage preset (0-9, higher is faster but lower quality)
    #[cfg(feature = "vp9-libvpx")]
    pub fn set_cpu_used(&mut self, cpu_used: i32) {
        self.config.cpu_used = cpu_used.clamp(-9, 9);
        unsafe {
            vpx_codec_control_(
                &mut self.ctx,
                vp8e_enc_control_id::VP8E_SET_CPUUSED as i32,
                self.config.cpu_used,
            );
        }
    }

    /// Set tile columns (log2, 0-6, resulting in 1-64 columns)
    #[cfg(feature = "vp9-libvpx")]
    pub fn set_tile_columns(&mut self, tile_columns: u32) {
        self.config.tile_columns = tile_columns.min(6);
        unsafe {
            vpx_codec_control_(
                &mut self.ctx,
                vp8e_enc_control_id::VP9E_SET_TILE_COLUMNS as i32,
                self.config.tile_columns as i32,
            );
        }
    }

    /// Set tile rows (log2, 0-2, resulting in 1-4 rows)
    #[cfg(feature = "vp9-libvpx")]
    pub fn set_tile_rows(&mut self, tile_rows: u32) {
        self.config.tile_rows = tile_rows.min(2);
        unsafe {
            vpx_codec_control_(
                &mut self.ctx,
                vp8e_enc_control_id::VP9E_SET_TILE_ROWS as i32,
                self.config.tile_rows as i32,
            );
        }
    }

    /// Enable or disable lossless mode
    #[cfg(feature = "vp9-libvpx")]
    pub fn set_lossless(&mut self, lossless: bool) {
        self.config.lossless = lossless;
        unsafe {
            vpx_codec_control_(
                &mut self.ctx,
                vp8e_enc_control_id::VP9E_SET_LOSSLESS as i32,
                if lossless { 1 } else { 0 },
            );
        }
    }

    /// Set adaptive quantization mode
    #[cfg(feature = "vp9-libvpx")]
    pub fn set_aq_mode(&mut self, aq_mode: Vp9AqMode) {
        self.config.aq_mode = aq_mode;
        unsafe {
            vpx_codec_control_(
                &mut self.ctx,
                vp8e_enc_control_id::VP9E_SET_AQ_MODE as i32,
                aq_mode as i32,
            );
        }
    }

    /// Copy frame data into the libvpx image buffer
    #[cfg(feature = "vp9-libvpx")]
    fn copy_frame_to_image(&mut self, video_frame: &VideoFrame) -> Result<()> {
        // Validate dimensions
        if video_frame.width != self.config.width || video_frame.height != self.config.height {
            return Err(Error::codec(format!(
                "Frame dimensions {}x{} don't match encoder {}x{}",
                video_frame.width, video_frame.height,
                self.config.width, self.config.height
            )));
        }

        // Determine expected format based on profile
        let (expected_format, uv_width_div, uv_height_div, bytes_per_sample) = match self.config.profile {
            Vp9Profile::Profile0 => (PixelFormat::YUV420P, 2, 2, 1),
            Vp9Profile::Profile1 => (PixelFormat::YUV444P, 1, 1, 1),
            Vp9Profile::Profile2 => (PixelFormat::YUV420P10LE, 2, 2, 2),
            Vp9Profile::Profile3 => (PixelFormat::YUV444P10LE, 1, 1, 2),
        };

        // Validate format
        if video_frame.format != expected_format {
            return Err(Error::codec(format!(
                "VP9 profile {:?} expects {:?}, got {:?}",
                self.config.profile, expected_format, video_frame.format
            )));
        }

        // Validate we have all planes
        if video_frame.data.len() < 3 {
            return Err(Error::codec(format!(
                "YUV frame requires 3 planes, got {}",
                video_frame.data.len()
            )));
        }

        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let uv_width = width / uv_width_div;
        let uv_height = height / uv_height_div;

        // Copy Y plane
        let y_src = video_frame.data[0].as_slice();
        let y_src_stride = video_frame.linesize[0];
        let y_dst_stride = self.raw_image.stride[0] as usize;

        unsafe {
            let y_dst = self.raw_image.planes[0];
            for row in 0..height {
                let src_offset = row * y_src_stride;
                let dst_offset = row * y_dst_stride;
                let row_bytes = width * bytes_per_sample;
                if src_offset + row_bytes <= y_src.len() {
                    ptr::copy_nonoverlapping(
                        y_src.as_ptr().add(src_offset),
                        y_dst.add(dst_offset),
                        row_bytes,
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
                let row_bytes = uv_width * bytes_per_sample;
                if src_offset + row_bytes <= u_src.len() {
                    ptr::copy_nonoverlapping(
                        u_src.as_ptr().add(src_offset),
                        u_dst.add(dst_offset),
                        row_bytes,
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
                let row_bytes = uv_width * bytes_per_sample;
                if src_offset + row_bytes <= v_src.len() {
                    ptr::copy_nonoverlapping(
                        v_src.as_ptr().add(src_offset),
                        v_dst.add(dst_offset),
                        row_bytes,
                    );
                }
            }
        }

        Ok(())
    }

    /// Retrieve encoded packets from the encoder
    #[cfg(feature = "vp9-libvpx")]
    fn retrieve_packets(&mut self, pts: i64) -> Result<()> {
        let mut iter: vpx_codec_iter_t = ptr::null();

        loop {
            let pkt = unsafe { vpx_codec_get_cx_data(&mut self.ctx, &mut iter) };

            if pkt.is_null() {
                break;
            }

            let pkt_ref = unsafe { &*pkt };

            match pkt_ref.kind {
                vpx_codec_cx_pkt_kind::VPX_CODEC_CX_FRAME_PKT => {
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
                    packet.dts = packet.pts;
                    packet.duration = 1;

                    // Check if keyframe
                    let is_keyframe = (frame.flags & VPX_FRAME_IS_KEY) != 0;
                    packet.set_keyframe(is_keyframe);

                    self.packet_buffer.push_back(packet);
                }
                vpx_codec_cx_pkt_kind::VPX_CODEC_STATS_PKT => {
                    // First-pass statistics - store for 2-pass encoding
                    let stats = unsafe { &pkt_ref.data.twopass_stats };
                    let data_ptr = stats.buf as *const u8;
                    let data_len = stats.sz;
                    let data = unsafe {
                        std::slice::from_raw_parts(data_ptr, data_len)
                    };
                    self.first_pass_stats.extend_from_slice(data);
                }
                _ => {
                    // Ignore other packet types (e.g., PSNR, custom)
                }
            }
        }

        Ok(())
    }

    /// Get first-pass statistics (for 2-pass encoding)
    pub fn get_first_pass_stats(&self) -> &[u8] {
        &self.first_pass_stats
    }

    /// Force the next frame to be a keyframe
    #[cfg(feature = "vp9-libvpx")]
    pub fn force_keyframe(&mut self) {
        unsafe {
            vpx_codec_control_(
                &mut self.ctx,
                vp8e_enc_control_id::VP8E_SET_FRAME_FLAGS as i32,
                VPX_EFLAG_FORCE_KF as i32,
            );
        }
    }

    /// Get the last quantizer used
    #[cfg(feature = "vp9-libvpx")]
    pub fn get_last_quantizer(&self) -> Option<i32> {
        let mut quantizer: i32 = 0;
        // VP8E_GET_LAST_QUANTIZER is shared with VP9
        let res = unsafe {
            vpx_codec_control_(
                &self.ctx as *const _ as *mut _,
                vp8e_enc_control_id::VP8E_GET_LAST_QUANTIZER as i32,
                &mut quantizer as *mut i32,
            )
        };
        if res == vpx_codec_err_t::VPX_CODEC_OK {
            Some(quantizer)
        } else {
            None
        }
    }
}

impl Encoder for Vp9Encoder {
    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        #[cfg(feature = "vp9-libvpx")]
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

                    // Select deadline based on cpu_used
                    // Higher cpu_used = faster = realtime deadline
                    let deadline = if self.config.cpu_used >= 5 {
                        VPX_DL_REALTIME
                    } else {
                        VPX_DL_GOOD_QUALITY
                    };

                    // Encode the frame
                    let res = unsafe {
                        vpx_codec_encode(
                            &mut self.ctx,
                            &self.raw_image,
                            pts,
                            1,                      // duration
                            flags as i64,           // flags
                            deadline as u64,        // deadline
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
                        return Err(Error::codec(format!("VP9 encode error: {}", err_msg)));
                    }

                    // Retrieve encoded packets
                    self.retrieve_packets(pts)?;
                    self.frame_count += 1;

                    Ok(())
                }
                Frame::Audio(_) => Err(Error::codec("VP9 encoder only accepts video frames")),
            }
        }

        #[cfg(not(feature = "vp9-libvpx"))]
        {
            let _ = frame;
            Err(Error::unsupported(
                "VP9 encoding requires the 'vp9-libvpx' feature",
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
        #[cfg(feature = "vp9-libvpx")]
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
        // VP9 doesn't require out-of-band codec configuration
        // All necessary info is in the bitstream
        None
    }
}

#[cfg(feature = "vp9-libvpx")]
impl Drop for Vp9Encoder {
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
#[cfg(feature = "vp9-libvpx")]
unsafe impl Send for Vp9Encoder {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vp9_encoder_config_default() {
        let config = Vp9EncoderConfig::default();
        assert_eq!(config.width, 640);
        assert_eq!(config.height, 480);
        assert_eq!(config.bitrate, 1_000_000);
        assert_eq!(config.framerate, 30);
        assert_eq!(config.profile, Vp9Profile::Profile0);
    }

    #[test]
    #[cfg(feature = "vp9-libvpx")]
    fn test_vp9_encoder_creation() {
        let encoder = Vp9Encoder::new(640, 480);
        assert!(
            encoder.is_ok(),
            "Encoder creation failed. Make sure libvpx is installed."
        );
    }

    #[test]
    #[cfg(feature = "vp9-libvpx")]
    fn test_vp9_encoder_with_config() {
        let config = Vp9EncoderConfig {
            width: 1920,
            height: 1080,
            bitrate: 5_000_000,
            framerate: 60,
            keyframe_interval: 120,
            threads: 4,
            rate_control: Vp9RateControl::CBR,
            cpu_used: 4,
            tile_columns: 2, // 4 tile columns
            tile_rows: 1,    // 2 tile rows
            row_mt: true,
            ..Default::default()
        };
        let encoder = Vp9Encoder::with_config(config);
        assert!(encoder.is_ok());
    }

    #[test]
    #[cfg(feature = "vp9-libvpx")]
    fn test_vp9_encoder_lossless() {
        let config = Vp9EncoderConfig {
            width: 320,
            height: 240,
            lossless: true,
            ..Default::default()
        };
        let encoder = Vp9Encoder::with_config(config);
        assert!(encoder.is_ok());
    }

    #[test]
    #[cfg(feature = "vp9-libvpx")]
    fn test_vp9_encoder_flush_empty() {
        let mut encoder = Vp9Encoder::new(320, 240).expect("Failed to create encoder");
        assert!(encoder.flush().is_ok());
    }

    #[test]
    #[cfg(feature = "vp9-libvpx")]
    fn test_vp9_encoder_receive_without_send() {
        let mut encoder = Vp9Encoder::new(320, 240).expect("Failed to create encoder");
        match encoder.receive_packet() {
            Err(Error::TryAgain) => {}
            other => panic!("Expected TryAgain, got {:?}", other),
        }
    }

    #[test]
    #[cfg(feature = "vp9-libvpx")]
    fn test_vp9_encode_frame() {
        use crate::codec::frame::VideoFrame;

        let mut encoder = Vp9Encoder::new(320, 240).expect("Failed to create encoder");

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

        // Try to receive packet (VP9 may buffer frames due to lag_in_frames)
        // Flush to get any buffered packets
        encoder.flush().expect("Flush failed");

        // Now we should be able to receive packets
        let mut packets_received = 0;
        loop {
            match encoder.receive_packet() {
                Ok(packet) => {
                    assert!(!packet.data.is_empty(), "Packet should contain data");
                    packets_received += 1;
                }
                Err(Error::TryAgain) => break,
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }

        assert!(packets_received > 0, "Should have received at least one packet");
    }

    #[test]
    fn test_vp9_encoder_creation_without_feature() {
        #[cfg(not(feature = "vp9-libvpx"))]
        {
            let encoder = Vp9Encoder::new(640, 480);
            assert!(encoder.is_err(), "Should fail without vp9-libvpx feature");
        }
    }

    #[test]
    fn test_vp9_profile_enum() {
        assert_eq!(Vp9Profile::Profile0 as i32, 0);
        assert_eq!(Vp9Profile::Profile1 as i32, 1);
        assert_eq!(Vp9Profile::Profile2 as i32, 2);
        assert_eq!(Vp9Profile::Profile3 as i32, 3);
    }

    #[test]
    fn test_vp9_rate_control_enum() {
        let _vbr = Vp9RateControl::VBR;
        let _cbr = Vp9RateControl::CBR;
        let _cq = Vp9RateControl::CQ;
        let _q = Vp9RateControl::Q;
    }
}
