//! Common hardware acceleration utilities and types
//!
//! This module provides shared types and utilities for all hardware acceleration backends:
//! - VAAPI (Linux - Intel/AMD)
//! - NVENC/NVDEC (NVIDIA)
//! - QSV (Intel Quick Sync)
//! - VideoToolbox (macOS/iOS)

use crate::codec::VideoFrame;
use crate::error::{Error, Result};
use crate::util::PixelFormat;
use std::fmt;
use std::sync::Arc;

/// Hardware pixel formats supported by GPU encoders/decoders
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HwPixelFormat {
    /// NV12 - 8-bit YUV 4:2:0 with interleaved UV plane (most common HW format)
    NV12,
    /// P010 - 10-bit YUV 4:2:0 with interleaved UV plane (HDR content)
    P010,
    /// P016 - 16-bit YUV 4:2:0 with interleaved UV plane
    P016,
    /// YUV420P - Planar 8-bit YUV 4:2:0 (3 separate planes)
    YUV420P,
    /// YUV422P - Planar 8-bit YUV 4:2:2
    YUV422P,
    /// YUV444P - Planar 8-bit YUV 4:4:4
    YUV444P,
    /// BGRA - 8-bit BGRA packed format
    BGRA,
    /// RGBA - 8-bit RGBA packed format
    RGBA,
    /// ARGB - 8-bit ARGB packed format (VideoToolbox preference)
    ARGB,
    /// UYVY - Packed YUV 4:2:2
    UYVY,
    /// YUYV - Packed YUV 4:2:2 (also known as YUY2)
    YUYV,
    /// Y210 - 10-bit packed YUV 4:2:2
    Y210,
    /// Y410 - 10-bit packed YUV 4:4:4
    Y410,
}

impl HwPixelFormat {
    /// Get the number of bytes per pixel (average for subsampled formats)
    pub fn bytes_per_pixel(&self) -> f32 {
        match self {
            HwPixelFormat::NV12 | HwPixelFormat::YUV420P => 1.5,
            HwPixelFormat::P010 | HwPixelFormat::P016 => 3.0,
            HwPixelFormat::YUV422P | HwPixelFormat::UYVY | HwPixelFormat::YUYV => 2.0,
            HwPixelFormat::YUV444P => 3.0,
            HwPixelFormat::BGRA | HwPixelFormat::RGBA | HwPixelFormat::ARGB => 4.0,
            HwPixelFormat::Y210 => 4.0,
            HwPixelFormat::Y410 => 4.0,
        }
    }

    /// Get the number of planes for this format
    pub fn num_planes(&self) -> usize {
        match self {
            HwPixelFormat::NV12 | HwPixelFormat::P010 | HwPixelFormat::P016 => 2,
            HwPixelFormat::YUV420P | HwPixelFormat::YUV422P | HwPixelFormat::YUV444P => 3,
            HwPixelFormat::BGRA
            | HwPixelFormat::RGBA
            | HwPixelFormat::ARGB
            | HwPixelFormat::UYVY
            | HwPixelFormat::YUYV
            | HwPixelFormat::Y210
            | HwPixelFormat::Y410 => 1,
        }
    }

    /// Check if this is an HDR-capable format (10+ bits)
    pub fn is_hdr_capable(&self) -> bool {
        matches!(
            self,
            HwPixelFormat::P010 | HwPixelFormat::P016 | HwPixelFormat::Y210 | HwPixelFormat::Y410
        )
    }

    /// Convert from PixelFormat to most appropriate HwPixelFormat
    pub fn from_pixel_format(pf: PixelFormat) -> Option<Self> {
        match pf {
            PixelFormat::YUV420P => Some(HwPixelFormat::NV12),
            PixelFormat::YUV422P => Some(HwPixelFormat::YUV422P),
            PixelFormat::YUV444P => Some(HwPixelFormat::YUV444P),
            PixelFormat::YUV420P10LE => Some(HwPixelFormat::P010),
            PixelFormat::RGBA => Some(HwPixelFormat::RGBA),
            PixelFormat::BGRA => Some(HwPixelFormat::BGRA),
            PixelFormat::RGB24 => Some(HwPixelFormat::RGBA), // Will need conversion
            PixelFormat::BGR24 => Some(HwPixelFormat::BGRA), // Will need conversion
            _ => None,
        }
    }

    /// Convert to PixelFormat for software fallback
    pub fn to_pixel_format(&self) -> PixelFormat {
        match self {
            HwPixelFormat::NV12 | HwPixelFormat::YUV420P => PixelFormat::YUV420P,
            HwPixelFormat::P010 | HwPixelFormat::P016 => PixelFormat::YUV420P10LE,
            HwPixelFormat::YUV422P | HwPixelFormat::UYVY | HwPixelFormat::YUYV => {
                PixelFormat::YUV422P
            }
            HwPixelFormat::YUV444P | HwPixelFormat::Y410 => PixelFormat::YUV444P,
            HwPixelFormat::BGRA => PixelFormat::BGRA,
            HwPixelFormat::RGBA | HwPixelFormat::ARGB => PixelFormat::RGBA,
            HwPixelFormat::Y210 => PixelFormat::YUV422P10LE,
        }
    }
}

impl fmt::Display for HwPixelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            HwPixelFormat::NV12 => "NV12",
            HwPixelFormat::P010 => "P010",
            HwPixelFormat::P016 => "P016",
            HwPixelFormat::YUV420P => "YUV420P",
            HwPixelFormat::YUV422P => "YUV422P",
            HwPixelFormat::YUV444P => "YUV444P",
            HwPixelFormat::BGRA => "BGRA",
            HwPixelFormat::RGBA => "RGBA",
            HwPixelFormat::ARGB => "ARGB",
            HwPixelFormat::UYVY => "UYVY",
            HwPixelFormat::YUYV => "YUYV",
            HwPixelFormat::Y210 => "Y210",
            HwPixelFormat::Y410 => "Y410",
        };
        write!(f, "{}", name)
    }
}

/// Video codec types supported by hardware encoders/decoders
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HwCodecType {
    /// H.264 / AVC
    H264,
    /// H.265 / HEVC
    H265,
    /// VP8
    VP8,
    /// VP9
    VP9,
    /// AV1
    AV1,
    /// MPEG-2
    MPEG2,
    /// MPEG-4 Part 2
    MPEG4,
    /// VC-1
    VC1,
    /// JPEG
    JPEG,
}

impl HwCodecType {
    /// Get the codec name as used in ffmpeg/gstreamer
    pub fn codec_name(&self) -> &'static str {
        match self {
            HwCodecType::H264 => "h264",
            HwCodecType::H265 => "hevc",
            HwCodecType::VP8 => "vp8",
            HwCodecType::VP9 => "vp9",
            HwCodecType::AV1 => "av1",
            HwCodecType::MPEG2 => "mpeg2video",
            HwCodecType::MPEG4 => "mpeg4",
            HwCodecType::VC1 => "vc1",
            HwCodecType::JPEG => "mjpeg",
        }
    }

    /// Parse codec type from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "h264" | "avc" | "h.264" => Some(HwCodecType::H264),
            "h265" | "hevc" | "h.265" => Some(HwCodecType::H265),
            "vp8" => Some(HwCodecType::VP8),
            "vp9" => Some(HwCodecType::VP9),
            "av1" => Some(HwCodecType::AV1),
            "mpeg2" | "mpeg2video" => Some(HwCodecType::MPEG2),
            "mpeg4" => Some(HwCodecType::MPEG4),
            "vc1" => Some(HwCodecType::VC1),
            "jpeg" | "mjpeg" => Some(HwCodecType::JPEG),
            _ => None,
        }
    }
}

impl fmt::Display for HwCodecType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.codec_name())
    }
}

/// H.264/H.265 profile for hardware encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwProfile {
    // H.264 profiles
    H264Baseline,
    H264Main,
    H264High,
    H264High10,
    H264High422,
    H264High444,

    // H.265/HEVC profiles
    HevcMain,
    HevcMain10,
    HevcMain12,
    HevcMain422_10,
    HevcMain422_12,
    HevcMain444,
    HevcMain444_10,
    HevcMain444_12,

    // VP9 profiles
    Vp9Profile0,
    Vp9Profile1,
    Vp9Profile2,
    Vp9Profile3,

    // AV1 profiles
    Av1Main,
    Av1High,
    Av1Professional,

    /// Auto-select based on input
    Auto,
}

impl Default for HwProfile {
    fn default() -> Self {
        HwProfile::Auto
    }
}

/// Rate control mode for hardware encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwRateControlMode {
    /// Constant QP - fixed quantization parameter (no rate control)
    ConstantQP,
    /// Variable Bitrate - target average bitrate with variation
    VBR,
    /// Constant Bitrate - strict bitrate target
    CBR,
    /// Variable Bitrate with High Quality preset
    VBR_HQ,
    /// Constant Bitrate with High Quality preset
    CBR_HQ,
    /// CRF-like quality mode (NVENC: ConstQual, VAAPI: ICQ)
    Quality,
}

impl Default for HwRateControlMode {
    fn default() -> Self {
        HwRateControlMode::VBR
    }
}

/// Hardware encoder configuration
#[derive(Debug, Clone)]
pub struct HwEncoderConfig {
    /// Target codec
    pub codec: HwCodecType,
    /// Encoding profile
    pub profile: HwProfile,
    /// Video width
    pub width: u32,
    /// Video height
    pub height: u32,
    /// Frame rate numerator
    pub framerate_num: u32,
    /// Frame rate denominator
    pub framerate_den: u32,
    /// Target bitrate in bits per second (0 for auto)
    pub bitrate: u64,
    /// Maximum bitrate for VBR modes
    pub max_bitrate: u64,
    /// Rate control mode
    pub rc_mode: HwRateControlMode,
    /// Quality level (0-51 for CQP, higher = worse quality)
    pub qp: u8,
    /// GOP size (keyframe interval in frames, 0 for auto)
    pub gop_size: u32,
    /// Number of B-frames between I/P frames (0 to disable)
    pub b_frames: u32,
    /// Number of reference frames
    pub ref_frames: u32,
    /// Input pixel format
    pub input_format: HwPixelFormat,
    /// Enable low-latency mode
    pub low_latency: bool,
    /// Encoder preset (0 = fastest/lowest quality, 7 = slowest/highest quality)
    pub preset: u8,
    /// Lookahead depth for better rate control (0 = disabled)
    pub lookahead: u32,
    /// Enable temporal AQ (adaptive quantization)
    pub temporal_aq: bool,
    /// Enable spatial AQ
    pub spatial_aq: bool,
    /// AQ strength (1-15, only if AQ enabled)
    pub aq_strength: u8,
}

impl Default for HwEncoderConfig {
    fn default() -> Self {
        HwEncoderConfig {
            codec: HwCodecType::H264,
            profile: HwProfile::Auto,
            width: 1920,
            height: 1080,
            framerate_num: 30,
            framerate_den: 1,
            bitrate: 5_000_000, // 5 Mbps
            max_bitrate: 10_000_000,
            rc_mode: HwRateControlMode::VBR,
            qp: 23,
            gop_size: 250, // ~8 seconds at 30fps
            b_frames: 2,
            ref_frames: 4,
            input_format: HwPixelFormat::NV12,
            low_latency: false,
            preset: 4, // Medium/balanced
            lookahead: 0,
            temporal_aq: false,
            spatial_aq: false,
            aq_strength: 8,
        }
    }
}

impl HwEncoderConfig {
    /// Create config for streaming (low latency, CBR)
    pub fn streaming(width: u32, height: u32, bitrate: u64) -> Self {
        HwEncoderConfig {
            width,
            height,
            bitrate,
            max_bitrate: bitrate,
            rc_mode: HwRateControlMode::CBR,
            gop_size: 60, // 2 seconds at 30fps
            b_frames: 0,
            low_latency: true,
            lookahead: 0,
            ..Default::default()
        }
    }

    /// Create config for high quality archival
    pub fn archival(width: u32, height: u32) -> Self {
        HwEncoderConfig {
            width,
            height,
            rc_mode: HwRateControlMode::Quality,
            qp: 18, // High quality
            gop_size: 250,
            b_frames: 3,
            ref_frames: 5,
            preset: 7, // Slowest/highest quality
            lookahead: 32,
            temporal_aq: true,
            spatial_aq: true,
            ..Default::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 {
            return Err(Error::invalid_input("Width and height must be non-zero"));
        }
        if self.width % 2 != 0 || self.height % 2 != 0 {
            return Err(Error::invalid_input(
                "Width and height must be even for YUV formats",
            ));
        }
        if self.framerate_num == 0 || self.framerate_den == 0 {
            return Err(Error::invalid_input("Invalid framerate"));
        }
        if self.qp > 51 {
            return Err(Error::invalid_input("QP must be 0-51"));
        }
        if self.preset > 7 {
            return Err(Error::invalid_input("Preset must be 0-7"));
        }
        Ok(())
    }
}

/// Hardware decoder configuration
#[derive(Debug, Clone)]
pub struct HwDecoderConfig {
    /// Codec to decode
    pub codec: HwCodecType,
    /// Maximum supported width (0 for auto)
    pub max_width: u32,
    /// Maximum supported height (0 for auto)
    pub max_height: u32,
    /// Output pixel format preference
    pub output_format: HwPixelFormat,
    /// Number of output surfaces to allocate
    pub num_surfaces: u32,
    /// Enable deinterlacing
    pub deinterlace: bool,
    /// Enable film grain synthesis (AV1)
    pub film_grain: bool,
}

impl Default for HwDecoderConfig {
    fn default() -> Self {
        HwDecoderConfig {
            codec: HwCodecType::H264,
            max_width: 4096,
            max_height: 2160,
            output_format: HwPixelFormat::NV12,
            num_surfaces: 8,
            deinterlace: false,
            film_grain: true,
        }
    }
}

/// Hardware surface representing GPU memory
#[derive(Debug, Clone)]
pub struct HwSurface {
    /// Surface width
    pub width: u32,
    /// Surface height
    pub height: u32,
    /// Pixel format
    pub format: HwPixelFormat,
    /// Platform-specific surface handle (opaque)
    pub handle: HwSurfaceHandle,
    /// Stride (bytes per row) for each plane
    pub strides: Vec<usize>,
    /// Offset to each plane in memory
    pub offsets: Vec<usize>,
    /// Total allocated size in bytes
    pub size: usize,
    /// CPU-accessible data (only valid after map operation)
    pub data: Option<Vec<u8>>,
}

/// Platform-specific surface handle
#[derive(Debug, Clone)]
pub enum HwSurfaceHandle {
    /// VA-API surface ID
    VaSurface(u32),
    /// CUDA device pointer
    CudaPtr(u64),
    /// Direct3D 11 texture (Windows)
    D3D11Texture(u64),
    /// CVPixelBuffer handle (macOS)
    CVPixelBuffer(u64),
    /// Intel QSV surface
    MfxSurface(u64),
    /// Software fallback - data stored in HwSurface::data
    Software,
}

impl HwSurface {
    /// Create a new hardware surface
    pub fn new(width: u32, height: u32, format: HwPixelFormat) -> Self {
        let (strides, offsets, size) = Self::calculate_layout(width, height, format);

        HwSurface {
            width,
            height,
            format,
            handle: HwSurfaceHandle::Software,
            strides,
            offsets,
            size,
            data: None,
        }
    }

    /// Calculate memory layout for a given format
    fn calculate_layout(
        width: u32,
        height: u32,
        format: HwPixelFormat,
    ) -> (Vec<usize>, Vec<usize>, usize) {
        let w = width as usize;
        let h = height as usize;

        match format {
            HwPixelFormat::NV12 => {
                let y_stride = w;
                let y_size = y_stride * h;
                let uv_stride = w;
                let uv_size = uv_stride * (h / 2);
                (
                    vec![y_stride, uv_stride],
                    vec![0, y_size],
                    y_size + uv_size,
                )
            }
            HwPixelFormat::P010 | HwPixelFormat::P016 => {
                let y_stride = w * 2;
                let y_size = y_stride * h;
                let uv_stride = w * 2;
                let uv_size = uv_stride * (h / 2);
                (
                    vec![y_stride, uv_stride],
                    vec![0, y_size],
                    y_size + uv_size,
                )
            }
            HwPixelFormat::YUV420P => {
                let y_stride = w;
                let y_size = y_stride * h;
                let u_stride = w / 2;
                let u_size = u_stride * (h / 2);
                let v_stride = w / 2;
                let v_size = v_stride * (h / 2);
                (
                    vec![y_stride, u_stride, v_stride],
                    vec![0, y_size, y_size + u_size],
                    y_size + u_size + v_size,
                )
            }
            HwPixelFormat::YUV422P => {
                let y_stride = w;
                let y_size = y_stride * h;
                let u_stride = w / 2;
                let u_size = u_stride * h;
                let v_stride = w / 2;
                let v_size = v_stride * h;
                (
                    vec![y_stride, u_stride, v_stride],
                    vec![0, y_size, y_size + u_size],
                    y_size + u_size + v_size,
                )
            }
            HwPixelFormat::YUV444P => {
                let stride = w;
                let plane_size = stride * h;
                (
                    vec![stride, stride, stride],
                    vec![0, plane_size, plane_size * 2],
                    plane_size * 3,
                )
            }
            HwPixelFormat::BGRA | HwPixelFormat::RGBA | HwPixelFormat::ARGB => {
                let stride = w * 4;
                let size = stride * h;
                (vec![stride], vec![0], size)
            }
            HwPixelFormat::UYVY | HwPixelFormat::YUYV => {
                let stride = w * 2;
                let size = stride * h;
                (vec![stride], vec![0], size)
            }
            HwPixelFormat::Y210 => {
                let stride = w * 4;
                let size = stride * h;
                (vec![stride], vec![0], size)
            }
            HwPixelFormat::Y410 => {
                let stride = w * 4;
                let size = stride * h;
                (vec![stride], vec![0], size)
            }
        }
    }

    /// Allocate CPU-side buffer for this surface
    pub fn allocate_data(&mut self) {
        if self.data.is_none() {
            self.data = Some(vec![0u8; self.size]);
        }
    }

    /// Get a reference to the CPU data
    pub fn get_data(&self) -> Option<&[u8]> {
        self.data.as_deref()
    }

    /// Get a mutable reference to the CPU data
    pub fn get_data_mut(&mut self) -> Option<&mut [u8]> {
        self.data.as_deref_mut()
    }

    /// Convert from VideoFrame to HwSurface
    pub fn from_video_frame(frame: &VideoFrame) -> Result<Self> {
        let format = HwPixelFormat::from_pixel_format(frame.format)
            .ok_or_else(|| Error::unsupported(format!("Unsupported pixel format: {}", frame.format)))?;

        let mut surface = HwSurface::new(frame.width, frame.height, format);
        surface.allocate_data();

        // Copy plane data
        if let Some(data) = &mut surface.data {
            let mut dst_offset = 0;
            for (i, plane) in frame.data.iter().enumerate() {
                let plane_data = plane.as_slice();
                let dst_end = dst_offset + plane_data.len();
                if dst_end <= data.len() {
                    data[dst_offset..dst_end].copy_from_slice(plane_data);
                }
                dst_offset = dst_end;
            }
        }

        Ok(surface)
    }

    /// Convert YUV420P to NV12 in-place
    pub fn convert_yuv420p_to_nv12(&mut self) -> Result<()> {
        if self.format != HwPixelFormat::YUV420P {
            return Err(Error::invalid_input("Source must be YUV420P"));
        }

        let data = self
            .data
            .take()
            .ok_or_else(|| Error::invalid_state("No data allocated"))?;

        let w = self.width as usize;
        let h = self.height as usize;

        // Calculate sizes
        let y_size = w * h;
        let uv_size = (w / 2) * (h / 2);

        if data.len() < y_size + uv_size * 2 {
            return Err(Error::invalid_input("Input data too small"));
        }

        // Create NV12 output
        let nv12_size = y_size + w * (h / 2);
        let mut nv12_data = vec![0u8; nv12_size];

        // Copy Y plane directly
        nv12_data[..y_size].copy_from_slice(&data[..y_size]);

        // Interleave U and V planes
        let u_plane = &data[y_size..y_size + uv_size];
        let v_plane = &data[y_size + uv_size..];

        let uv_dst = &mut nv12_data[y_size..];
        for i in 0..uv_size {
            uv_dst[i * 2] = u_plane[i];
            uv_dst[i * 2 + 1] = v_plane[i];
        }

        // Update surface
        self.format = HwPixelFormat::NV12;
        let (strides, offsets, size) =
            Self::calculate_layout(self.width, self.height, HwPixelFormat::NV12);
        self.strides = strides;
        self.offsets = offsets;
        self.size = size;
        self.data = Some(nv12_data);

        Ok(())
    }

    /// Convert NV12 to YUV420P for software processing
    pub fn convert_nv12_to_yuv420p(&mut self) -> Result<()> {
        if self.format != HwPixelFormat::NV12 {
            return Err(Error::invalid_input("Source must be NV12"));
        }

        let data = self
            .data
            .take()
            .ok_or_else(|| Error::invalid_state("No data allocated"))?;

        let w = self.width as usize;
        let h = self.height as usize;

        let y_size = w * h;
        let uv_size = (w / 2) * (h / 2);

        if data.len() < y_size + w * (h / 2) {
            return Err(Error::invalid_input("Input data too small"));
        }

        // Create YUV420P output
        let yuv420_size = y_size + uv_size * 2;
        let mut yuv420_data = vec![0u8; yuv420_size];

        // Copy Y plane
        yuv420_data[..y_size].copy_from_slice(&data[..y_size]);

        // De-interleave UV to separate U and V planes
        let uv_src = &data[y_size..];
        // Use split_at_mut to get two non-overlapping mutable slices
        let (_, uv_region) = yuv420_data.split_at_mut(y_size);
        let (u_dst, v_dst) = uv_region.split_at_mut(uv_size);

        for i in 0..uv_size {
            u_dst[i] = uv_src[i * 2];
            v_dst[i] = uv_src[i * 2 + 1];
        }

        // Update surface
        self.format = HwPixelFormat::YUV420P;
        let (strides, offsets, size) =
            Self::calculate_layout(self.width, self.height, HwPixelFormat::YUV420P);
        self.strides = strides;
        self.offsets = offsets;
        self.size = size;
        self.data = Some(yuv420_data);

        Ok(())
    }
}

/// Encoded bitstream output from hardware encoder
#[derive(Debug, Clone)]
pub struct HwEncodedPacket {
    /// Encoded data
    pub data: Vec<u8>,
    /// Presentation timestamp
    pub pts: i64,
    /// Decode timestamp
    pub dts: i64,
    /// Duration
    pub duration: i64,
    /// Is this a keyframe?
    pub keyframe: bool,
    /// Picture type (I, P, B)
    pub pict_type: char,
    /// Frame index
    pub frame_num: u64,
}

impl HwEncodedPacket {
    /// Create a new encoded packet
    pub fn new(data: Vec<u8>, pts: i64, keyframe: bool) -> Self {
        HwEncodedPacket {
            data,
            pts,
            dts: pts,
            duration: 0,
            keyframe,
            pict_type: if keyframe { 'I' } else { 'P' },
            frame_num: 0,
        }
    }
}

/// Hardware encoder capability information
#[derive(Debug, Clone)]
pub struct HwEncoderCaps {
    /// Supported codec
    pub codec: HwCodecType,
    /// Supported profiles
    pub profiles: Vec<HwProfile>,
    /// Maximum width
    pub max_width: u32,
    /// Maximum height
    pub max_height: u32,
    /// Minimum width
    pub min_width: u32,
    /// Minimum height
    pub min_height: u32,
    /// Supported input formats
    pub input_formats: Vec<HwPixelFormat>,
    /// Supports B-frames
    pub b_frames: bool,
    /// Maximum B-frame count
    pub max_b_frames: u32,
    /// Supports lookahead
    pub lookahead: bool,
    /// Maximum lookahead depth
    pub max_lookahead: u32,
    /// Supports temporal AQ
    pub temporal_aq: bool,
    /// Supports spatial AQ
    pub spatial_aq: bool,
    /// Maximum concurrent encoding sessions
    pub max_sessions: u32,
}

impl Default for HwEncoderCaps {
    fn default() -> Self {
        HwEncoderCaps {
            codec: HwCodecType::H264,
            profiles: vec![HwProfile::H264Main, HwProfile::H264High],
            max_width: 4096,
            max_height: 4096,
            min_width: 64,
            min_height: 64,
            input_formats: vec![HwPixelFormat::NV12],
            b_frames: true,
            max_b_frames: 4,
            lookahead: true,
            max_lookahead: 32,
            temporal_aq: true,
            spatial_aq: true,
            max_sessions: 4,
        }
    }
}

/// Hardware decoder capability information
#[derive(Debug, Clone)]
pub struct HwDecoderCaps {
    /// Supported codec
    pub codec: HwCodecType,
    /// Supported profiles
    pub profiles: Vec<HwProfile>,
    /// Maximum width
    pub max_width: u32,
    /// Maximum height
    pub max_height: u32,
    /// Supported output formats
    pub output_formats: Vec<HwPixelFormat>,
    /// Supports deinterlacing
    pub deinterlace: bool,
    /// Maximum bit depth
    pub max_bit_depth: u8,
}

impl Default for HwDecoderCaps {
    fn default() -> Self {
        HwDecoderCaps {
            codec: HwCodecType::H264,
            profiles: vec![HwProfile::H264Main, HwProfile::H264High],
            max_width: 4096,
            max_height: 4096,
            output_formats: vec![HwPixelFormat::NV12],
            deinterlace: true,
            max_bit_depth: 8,
        }
    }
}

/// Statistics from hardware encoding
#[derive(Debug, Clone, Default)]
pub struct HwEncoderStats {
    /// Total frames encoded
    pub frames_encoded: u64,
    /// Total bytes output
    pub bytes_output: u64,
    /// Total encoding time in microseconds
    pub encode_time_us: u64,
    /// Average frame encode time in microseconds
    pub avg_frame_time_us: u64,
    /// Frames per second achieved
    pub fps: f64,
    /// Average bitrate in bits per second
    pub avg_bitrate: u64,
    /// I-frames encoded
    pub i_frames: u64,
    /// P-frames encoded
    pub p_frames: u64,
    /// B-frames encoded
    pub b_frames: u64,
}

/// GPU memory pool for efficient surface reuse
pub struct HwSurfacePool {
    /// Available surfaces
    available: Vec<HwSurface>,
    /// Surface dimensions
    width: u32,
    height: u32,
    /// Surface format
    format: HwPixelFormat,
    /// Maximum pool size
    max_size: usize,
}

impl HwSurfacePool {
    /// Create a new surface pool
    pub fn new(width: u32, height: u32, format: HwPixelFormat, max_size: usize) -> Self {
        HwSurfacePool {
            available: Vec::with_capacity(max_size),
            width,
            height,
            format,
            max_size,
        }
    }

    /// Acquire a surface from the pool (or create new one)
    pub fn acquire(&mut self) -> HwSurface {
        if let Some(surface) = self.available.pop() {
            surface
        } else {
            let mut surface = HwSurface::new(self.width, self.height, self.format);
            surface.allocate_data();
            surface
        }
    }

    /// Release a surface back to the pool
    pub fn release(&mut self, surface: HwSurface) {
        if self.available.len() < self.max_size {
            self.available.push(surface);
        }
        // Otherwise drop the surface
    }

    /// Get current pool size
    pub fn available_count(&self) -> usize {
        self.available.len()
    }

    /// Pre-allocate surfaces
    pub fn preallocate(&mut self, count: usize) {
        let to_allocate = count.min(self.max_size).saturating_sub(self.available.len());
        for _ in 0..to_allocate {
            let mut surface = HwSurface::new(self.width, self.height, self.format);
            surface.allocate_data();
            self.available.push(surface);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hw_pixel_format_bytes_per_pixel() {
        assert_eq!(HwPixelFormat::NV12.bytes_per_pixel(), 1.5);
        assert_eq!(HwPixelFormat::RGBA.bytes_per_pixel(), 4.0);
        assert_eq!(HwPixelFormat::P010.bytes_per_pixel(), 3.0);
    }

    #[test]
    fn test_hw_surface_layout_nv12() {
        let surface = HwSurface::new(1920, 1080, HwPixelFormat::NV12);
        assert_eq!(surface.strides, vec![1920, 1920]);
        assert_eq!(surface.offsets, vec![0, 1920 * 1080]);
        assert_eq!(surface.size, 1920 * 1080 * 3 / 2);
    }

    #[test]
    fn test_hw_surface_layout_yuv420p() {
        let surface = HwSurface::new(1920, 1080, HwPixelFormat::YUV420P);
        assert_eq!(surface.strides, vec![1920, 960, 960]);
        let y_size = 1920 * 1080;
        let uv_size = 960 * 540;
        assert_eq!(surface.offsets, vec![0, y_size, y_size + uv_size]);
        assert_eq!(surface.size, y_size + uv_size * 2);
    }

    #[test]
    fn test_encoder_config_validation() {
        let mut config = HwEncoderConfig::default();
        assert!(config.validate().is_ok());

        config.width = 0;
        assert!(config.validate().is_err());

        config.width = 1920;
        config.qp = 60;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_surface_pool() {
        let mut pool = HwSurfacePool::new(1920, 1080, HwPixelFormat::NV12, 4);
        pool.preallocate(2);
        assert_eq!(pool.available_count(), 2);

        let s1 = pool.acquire();
        assert_eq!(pool.available_count(), 1);

        pool.release(s1);
        assert_eq!(pool.available_count(), 2);
    }

    #[test]
    fn test_yuv420p_to_nv12_conversion() {
        let mut surface = HwSurface::new(4, 4, HwPixelFormat::YUV420P);
        surface.allocate_data();

        // Fill with test pattern
        if let Some(data) = &mut surface.data {
            // Y plane: 16 bytes
            for i in 0..16 {
                data[i] = i as u8;
            }
            // U plane: 4 bytes
            for i in 0..4 {
                data[16 + i] = 100 + i as u8;
            }
            // V plane: 4 bytes
            for i in 0..4 {
                data[20 + i] = 200 + i as u8;
            }
        }

        surface.convert_yuv420p_to_nv12().unwrap();

        assert_eq!(surface.format, HwPixelFormat::NV12);
        if let Some(data) = &surface.data {
            // Y plane should be unchanged
            for i in 0..16 {
                assert_eq!(data[i], i as u8);
            }
            // UV should be interleaved
            assert_eq!(data[16], 100); // U[0]
            assert_eq!(data[17], 200); // V[0]
            assert_eq!(data[18], 101); // U[1]
            assert_eq!(data[19], 201); // V[1]
        }
    }
}
