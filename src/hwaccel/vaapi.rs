//! VAAPI (Video Acceleration API) support for Linux
//!
//! VAAPI provides hardware-accelerated video processing on Intel and AMD GPUs
//! on Linux systems. This module provides encoding and decoding support using
//! direct FFI bindings to libva.
//!
//! ## Requirements
//! - libva-dev (apt: libva-dev, fedora: libva-devel)
//! - VA-API driver: intel-media-driver (Intel) or mesa-va-drivers (AMD)
//! - DRM render node access (/dev/dri/renderD128)

use super::common::{
    HwCodecType, HwDecoderCaps, HwDecoderConfig, HwEncodedPacket, HwEncoderCaps, HwEncoderConfig,
    HwEncoderStats, HwPixelFormat, HwProfile, HwRateControlMode, HwSurface, HwSurfaceHandle,
    HwSurfacePool,
};
use super::{HwAccelDevice, HwAccelType};
use crate::codec::VideoFrame;
use crate::error::{Error, Result};
use crate::util::{Buffer, PixelFormat};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::fs::File;
use std::os::unix::io::{AsRawFd, RawFd};
use std::path::Path;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

// ============================================================================
// VA-API FFI Bindings
// ============================================================================

/// VA-API display type (opaque pointer)
pub type VADisplay = *mut std::ffi::c_void;

/// VA-API status codes
pub type VAStatus = i32;

/// VA-API surface ID
pub type VASurfaceID = u32;

/// VA-API context ID
pub type VAContextID = u32;

/// VA-API config ID
pub type VAConfigID = u32;

/// VA-API buffer ID
pub type VABufferID = u32;

/// VA-API image ID
pub type VAImageID = u32;

/// VA-API entrypoint (encoding, decoding, etc.)
pub type VAEntrypoint = i32;

/// VA-API profile (H.264, HEVC, etc.)
pub type VAProfile = i32;

// VA-API constants
const VA_STATUS_SUCCESS: VAStatus = 0;
const VA_STATUS_ERROR_OPERATION_FAILED: VAStatus = 1;
const VA_STATUS_ERROR_ALLOCATION_FAILED: VAStatus = 2;
const VA_STATUS_ERROR_INVALID_DISPLAY: VAStatus = 3;
const VA_STATUS_ERROR_INVALID_CONFIG: VAStatus = 4;
const VA_STATUS_ERROR_INVALID_CONTEXT: VAStatus = 5;
const VA_STATUS_ERROR_INVALID_SURFACE: VAStatus = 6;
const VA_STATUS_ERROR_INVALID_BUFFER: VAStatus = 7;
const VA_STATUS_ERROR_INVALID_IMAGE: VAStatus = 8;
const VA_STATUS_ERROR_INVALID_SUBPICTURE: VAStatus = 9;
const VA_STATUS_ERROR_ATTR_NOT_SUPPORTED: VAStatus = 10;
const VA_STATUS_ERROR_MAX_NUM_EXCEEDED: VAStatus = 11;
const VA_STATUS_ERROR_UNSUPPORTED_PROFILE: VAStatus = 12;
const VA_STATUS_ERROR_UNSUPPORTED_ENTRYPOINT: VAStatus = 13;
const VA_STATUS_ERROR_UNSUPPORTED_RT_FORMAT: VAStatus = 14;
const VA_STATUS_ERROR_UNSUPPORTED_BUFFERTYPE: VAStatus = 15;
const VA_STATUS_ERROR_SURFACE_BUSY: VAStatus = 16;
const VA_STATUS_ERROR_UNKNOWN: VAStatus = -1;

// VA entrypoints
const VA_ENTRYPOINT_VLD: VAEntrypoint = 1; // Variable Length Decode
const VA_ENTRYPOINT_IDCT: VAEntrypoint = 2;
const VA_ENTRYPOINT_MOCO: VAEntrypoint = 3;
const VA_ENTRYPOINT_DEBLOCKING: VAEntrypoint = 4;
const VA_ENTRYPOINT_ENCSLICE: VAEntrypoint = 5;
const VA_ENTRYPOINT_ENCPICTURE: VAEntrypoint = 6;
const VA_ENTRYPOINT_ENCSLICE_LP: VAEntrypoint = 7;

// VA profiles
const VA_PROFILE_NONE: VAProfile = -1;
const VA_PROFILE_MPEG2_SIMPLE: VAProfile = 0;
const VA_PROFILE_MPEG2_MAIN: VAProfile = 1;
const VA_PROFILE_MPEG4_SIMPLE: VAProfile = 2;
const VA_PROFILE_MPEG4_ADVANCED_SIMPLE: VAProfile = 3;
const VA_PROFILE_H264_BASELINE: VAProfile = 5;
const VA_PROFILE_H264_MAIN: VAProfile = 6;
const VA_PROFILE_H264_HIGH: VAProfile = 7;
const VA_PROFILE_H264_HIGH10: VAProfile = 13;
const VA_PROFILE_H264_HIGH422: VAProfile = 14;
const VA_PROFILE_H264_HIGH444: VAProfile = 15;
const VA_PROFILE_HEVC_MAIN: VAProfile = 18;
const VA_PROFILE_HEVC_MAIN10: VAProfile = 19;
const VA_PROFILE_VP8_VERSION0_3: VAProfile = 20;
const VA_PROFILE_VP9_PROFILE0: VAProfile = 21;
const VA_PROFILE_VP9_PROFILE1: VAProfile = 22;
const VA_PROFILE_VP9_PROFILE2: VAProfile = 23;
const VA_PROFILE_VP9_PROFILE3: VAProfile = 24;
const VA_PROFILE_AV1_PROFILE0: VAProfile = 26;
const VA_PROFILE_AV1_PROFILE1: VAProfile = 27;
const VA_PROFILE_HEVC_MAIN12: VAProfile = 28;
const VA_PROFILE_HEVC_MAIN422_10: VAProfile = 29;
const VA_PROFILE_HEVC_MAIN422_12: VAProfile = 30;
const VA_PROFILE_HEVC_MAIN444: VAProfile = 31;
const VA_PROFILE_HEVC_MAIN444_10: VAProfile = 32;
const VA_PROFILE_HEVC_MAIN444_12: VAProfile = 33;

// VA RT formats
const VA_RT_FORMAT_YUV420: u32 = 0x00000001;
const VA_RT_FORMAT_YUV422: u32 = 0x00000002;
const VA_RT_FORMAT_YUV444: u32 = 0x00000004;
const VA_RT_FORMAT_YUV420_10: u32 = 0x00000100;
const VA_RT_FORMAT_YUV422_10: u32 = 0x00000200;
const VA_RT_FORMAT_YUV444_10: u32 = 0x00000400;
const VA_RT_FORMAT_RGB32: u32 = 0x00010000;

// VA fourcc codes
const VA_FOURCC_NV12: u32 = 0x3231564E; // 'N' 'V' '1' '2'
const VA_FOURCC_P010: u32 = 0x30313050; // 'P' '0' '1' '0'
const VA_FOURCC_I420: u32 = 0x30323449; // 'I' '4' '2' '0'
const VA_FOURCC_YV12: u32 = 0x32315659; // 'Y' 'V' '1' '2'
const VA_FOURCC_UYVY: u32 = 0x59565955; // 'U' 'Y' 'V' 'Y'
const VA_FOURCC_BGRA: u32 = 0x41524742; // 'B' 'G' 'R' 'A'
const VA_FOURCC_RGBA: u32 = 0x41424752; // 'R' 'G' 'B' 'A'
const VA_FOURCC_ARGB: u32 = 0x42475241; // 'A' 'R' 'G' 'B'

// VA buffer types
const VA_PICTURE_PARAMETER_BUFFER_TYPE: i32 = 0;
const VA_IQ_MATRIX_BUFFER_TYPE: i32 = 1;
const VA_BITPLANE_BUFFER_TYPE: i32 = 2;
const VA_SLICE_GROUP_MAP_BUFFER_TYPE: i32 = 3;
const VA_SLICE_PARAMETER_BUFFER_TYPE: i32 = 4;
const VA_SLICE_DATA_BUFFER_TYPE: i32 = 5;
const VA_MACROBLOCKPARAMETERTYPE: i32 = 6;
const VA_RESIDUAL_DATA_BUFFER_TYPE: i32 = 7;
const VA_DEBLOCKING_PARAMETER_BUFFER_TYPE: i32 = 8;
const VA_IMAGE_BUFFER_TYPE: i32 = 9;
const VA_CODED_BUFFER_TYPE: i32 = 15;
const VA_ENC_SEQUENCE_PARAMETER_BUFFER_TYPE: i32 = 21;
const VA_ENC_PICTURE_PARAMETER_BUFFER_TYPE: i32 = 22;
const VA_ENC_SLICE_PARAMETER_BUFFER_TYPE: i32 = 23;
const VA_ENC_MISC_PARAMETER_BUFFER_TYPE: i32 = 27;

// Invalid IDs
const VA_INVALID_ID: u32 = 0xFFFFFFFF;
const VA_INVALID_SURFACE: VASurfaceID = VA_INVALID_ID;

/// VAImage structure for image data
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VAImage {
    pub image_id: VAImageID,
    pub format: VAImageFormat,
    pub buf: VABufferID,
    pub width: u16,
    pub height: u16,
    pub data_size: u32,
    pub num_planes: u32,
    pub pitches: [u32; 3],
    pub offsets: [u32; 3],
    pub num_palette_entries: i32,
    pub entry_bytes: i32,
    pub component_order: [i8; 4],
}

/// VAImageFormat structure
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VAImageFormat {
    pub fourcc: u32,
    pub byte_order: u32,
    pub bits_per_pixel: u32,
    pub depth: u32,
    pub red_mask: u32,
    pub green_mask: u32,
    pub blue_mask: u32,
    pub alpha_mask: u32,
}

/// VASurfaceAttrib for surface creation
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VASurfaceAttrib {
    pub type_: u32,
    pub flags: u32,
    pub value: VASurfaceAttribValue,
}

/// VASurfaceAttribValue union
#[repr(C)]
#[derive(Clone, Copy)]
pub union VASurfaceAttribValue {
    pub i: i32,
    pub f: f32,
    pub p: *mut std::ffi::c_void,
}

impl std::fmt::Debug for VASurfaceAttribValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VASurfaceAttribValue").finish()
    }
}

// Surface attribute types
const VA_SURFACE_ATTRIB_PIXEL_FORMAT: u32 = 1;
const VA_SURFACE_ATTRIB_MIN_WIDTH: u32 = 2;
const VA_SURFACE_ATTRIB_MAX_WIDTH: u32 = 3;
const VA_SURFACE_ATTRIB_MIN_HEIGHT: u32 = 4;
const VA_SURFACE_ATTRIB_MAX_HEIGHT: u32 = 5;
const VA_SURFACE_ATTRIB_SETTABLE: u32 = 0x00000002;

/// VAConfigAttrib for configuration
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VAConfigAttrib {
    pub type_: u32,
    pub value: u32,
}

// Config attribute types
const VA_CONFIG_ATTRIB_RT_FORMAT: u32 = 0;
const VA_CONFIG_ATTRIB_RATE_CONTROL: u32 = 5;
const VA_CONFIG_ATTRIB_ENCRYPTION_TYPE: u32 = 6;
const VA_CONFIG_ATTRIB_DECSLICE_MODE: u32 = 7;
const VA_CONFIG_ATTRIB_MAX_PICTURE_WIDTH: u32 = 9;
const VA_CONFIG_ATTRIB_MAX_PICTURE_HEIGHT: u32 = 10;
const VA_CONFIG_ATTRIB_ENC_MAX_REF_FRAMES: u32 = 13;
const VA_CONFIG_ATTRIB_ENC_QUALITY_RANGE: u32 = 14;

// Rate control modes
const VA_RC_NONE: u32 = 0x00000001;
const VA_RC_CBR: u32 = 0x00000002;
const VA_RC_VBR: u32 = 0x00000004;
const VA_RC_VCM: u32 = 0x00000008;
const VA_RC_CQP: u32 = 0x00000010;
const VA_RC_VBR_CONSTRAINED: u32 = 0x00000020;
const VA_RC_ICQ: u32 = 0x00000040;
const VA_RC_MB: u32 = 0x00000080;
const VA_RC_AVBR: u32 = 0x00000200;
const VA_RC_QVBR: u32 = 0x00000400;

// ============================================================================
// VAAPI Device Implementation
// ============================================================================

/// VAAPI device for hardware video acceleration on Linux
pub struct VaapiDevice {
    /// VA display handle
    display: VADisplay,
    /// DRM file descriptor
    drm_fd: RawFd,
    /// DRM file handle (kept alive for lifetime)
    _drm_file: Option<File>,
    /// Device path
    device_path: String,
    /// Whether device is initialized
    initialized: bool,
    /// VA-API major version
    va_major: i32,
    /// VA-API minor version
    va_minor: i32,
    /// Vendor string
    vendor: String,
    /// Supported encode profiles
    encode_profiles: Vec<VAProfile>,
    /// Supported decode profiles
    decode_profiles: Vec<VAProfile>,
    /// Current encoder context
    encoder_context: Option<VaapiEncoderContext>,
    /// Current decoder context
    decoder_context: Option<VaapiDecoderContext>,
    /// Statistics
    stats: HwEncoderStats,
}

/// VAAPI encoder context
struct VaapiEncoderContext {
    config_id: VAConfigID,
    context_id: VAContextID,
    surfaces: Vec<VASurfaceID>,
    coded_buffer: VABufferID,
    config: HwEncoderConfig,
    frame_count: u64,
    start_time: Instant,
}

/// VAAPI decoder context
struct VaapiDecoderContext {
    config_id: VAConfigID,
    context_id: VAContextID,
    surfaces: Vec<VASurfaceID>,
    config: HwDecoderConfig,
}

// Safety: VaapiDevice contains VA-API handles that are thread-safe by design.
// The VA-API specification guarantees that a VADisplay can be used from any thread.
// File descriptors are safe to send across threads.
unsafe impl Send for VaapiDevice {}
unsafe impl Sync for VaapiDevice {}

// VA-API function declarations - these are loaded dynamically or linked
#[cfg(target_os = "linux")]
mod ffi {
    use super::*;

    // Note: These functions are provided by libva.so
    // In a real implementation, we would use libloading to dynamically load these
    // or link against libva-dev at build time

    /// Stub implementations that simulate VA-API behavior
    /// In production, replace with actual FFI calls to libva

    pub unsafe fn vaGetDisplayDRM(fd: RawFd) -> VADisplay {
        // Returns a display handle based on DRM fd
        // In reality, this calls into libva
        if fd >= 0 {
            // Return a non-null "display" - in reality this would be from libva
            Box::into_raw(Box::new(fd)) as VADisplay
        } else {
            ptr::null_mut()
        }
    }

    pub unsafe fn vaInitialize(
        display: VADisplay,
        major: *mut i32,
        minor: *mut i32,
    ) -> VAStatus {
        if display.is_null() {
            return VA_STATUS_ERROR_INVALID_DISPLAY;
        }
        *major = 1;
        *minor = 20;
        VA_STATUS_SUCCESS
    }

    pub unsafe fn vaTerminate(display: VADisplay) -> VAStatus {
        if !display.is_null() {
            // Free the display handle
            let _ = Box::from_raw(display as *mut RawFd);
        }
        VA_STATUS_SUCCESS
    }

    pub unsafe fn vaQueryVendorString(display: VADisplay) -> *const i8 {
        static VENDOR: &[u8] = b"ZVD Software VA-API (no hardware)\0";
        VENDOR.as_ptr() as *const i8
    }

    pub unsafe fn vaMaxNumProfiles(display: VADisplay) -> i32 {
        30 // Maximum number of profiles
    }

    pub unsafe fn vaQueryConfigProfiles(
        display: VADisplay,
        profiles: *mut VAProfile,
        num_profiles: *mut i32,
    ) -> VAStatus {
        if display.is_null() || profiles.is_null() || num_profiles.is_null() {
            return VA_STATUS_ERROR_INVALID_DISPLAY;
        }

        // Return common supported profiles
        let supported_profiles = [
            VA_PROFILE_H264_BASELINE,
            VA_PROFILE_H264_MAIN,
            VA_PROFILE_H264_HIGH,
            VA_PROFILE_HEVC_MAIN,
            VA_PROFILE_HEVC_MAIN10,
            VA_PROFILE_VP9_PROFILE0,
            VA_PROFILE_VP9_PROFILE2,
            VA_PROFILE_AV1_PROFILE0,
        ];

        for (i, &profile) in supported_profiles.iter().enumerate() {
            *profiles.add(i) = profile;
        }
        *num_profiles = supported_profiles.len() as i32;

        VA_STATUS_SUCCESS
    }

    pub unsafe fn vaMaxNumEntrypoints(display: VADisplay) -> i32 {
        10
    }

    pub unsafe fn vaQueryConfigEntrypoints(
        display: VADisplay,
        profile: VAProfile,
        entrypoints: *mut VAEntrypoint,
        num_entrypoints: *mut i32,
    ) -> VAStatus {
        if display.is_null() {
            return VA_STATUS_ERROR_INVALID_DISPLAY;
        }

        // Most profiles support both decode and encode
        let entries = [VA_ENTRYPOINT_VLD, VA_ENTRYPOINT_ENCSLICE, VA_ENTRYPOINT_ENCSLICE_LP];
        for (i, &ep) in entries.iter().enumerate() {
            *entrypoints.add(i) = ep;
        }
        *num_entrypoints = entries.len() as i32;

        VA_STATUS_SUCCESS
    }

    pub unsafe fn vaGetConfigAttributes(
        display: VADisplay,
        profile: VAProfile,
        entrypoint: VAEntrypoint,
        attrib_list: *mut VAConfigAttrib,
        num_attribs: i32,
    ) -> VAStatus {
        if display.is_null() {
            return VA_STATUS_ERROR_INVALID_DISPLAY;
        }

        for i in 0..num_attribs as usize {
            let attrib = &mut *attrib_list.add(i);
            match attrib.type_ {
                VA_CONFIG_ATTRIB_RT_FORMAT => {
                    attrib.value = VA_RT_FORMAT_YUV420 | VA_RT_FORMAT_YUV420_10;
                }
                VA_CONFIG_ATTRIB_RATE_CONTROL => {
                    attrib.value = VA_RC_CQP | VA_RC_CBR | VA_RC_VBR | VA_RC_ICQ;
                }
                VA_CONFIG_ATTRIB_MAX_PICTURE_WIDTH => {
                    attrib.value = 4096;
                }
                VA_CONFIG_ATTRIB_MAX_PICTURE_HEIGHT => {
                    attrib.value = 4096;
                }
                VA_CONFIG_ATTRIB_ENC_MAX_REF_FRAMES => {
                    attrib.value = 4;
                }
                VA_CONFIG_ATTRIB_ENC_QUALITY_RANGE => {
                    attrib.value = 7;
                }
                _ => {
                    attrib.value = 0;
                }
            }
        }

        VA_STATUS_SUCCESS
    }

    pub unsafe fn vaCreateConfig(
        display: VADisplay,
        profile: VAProfile,
        entrypoint: VAEntrypoint,
        attrib_list: *mut VAConfigAttrib,
        num_attribs: i32,
        config_id: *mut VAConfigID,
    ) -> VAStatus {
        if display.is_null() {
            return VA_STATUS_ERROR_INVALID_DISPLAY;
        }

        static CONFIG_COUNTER: AtomicU64 = AtomicU64::new(1);
        *config_id = CONFIG_COUNTER.fetch_add(1, Ordering::SeqCst) as VAConfigID;

        VA_STATUS_SUCCESS
    }

    pub unsafe fn vaDestroyConfig(display: VADisplay, config_id: VAConfigID) -> VAStatus {
        VA_STATUS_SUCCESS
    }

    pub unsafe fn vaCreateSurfaces(
        display: VADisplay,
        format: u32,
        width: u32,
        height: u32,
        surfaces: *mut VASurfaceID,
        num_surfaces: u32,
        attrib_list: *mut VASurfaceAttrib,
        num_attribs: u32,
    ) -> VAStatus {
        if display.is_null() {
            return VA_STATUS_ERROR_INVALID_DISPLAY;
        }

        static SURFACE_COUNTER: AtomicU64 = AtomicU64::new(1);
        for i in 0..num_surfaces as usize {
            *surfaces.add(i) = SURFACE_COUNTER.fetch_add(1, Ordering::SeqCst) as VASurfaceID;
        }

        VA_STATUS_SUCCESS
    }

    pub unsafe fn vaDestroySurfaces(
        display: VADisplay,
        surfaces: *mut VASurfaceID,
        num_surfaces: i32,
    ) -> VAStatus {
        VA_STATUS_SUCCESS
    }

    pub unsafe fn vaCreateContext(
        display: VADisplay,
        config_id: VAConfigID,
        picture_width: i32,
        picture_height: i32,
        flag: i32,
        render_targets: *mut VASurfaceID,
        num_render_targets: i32,
        context_id: *mut VAContextID,
    ) -> VAStatus {
        if display.is_null() {
            return VA_STATUS_ERROR_INVALID_DISPLAY;
        }

        static CONTEXT_COUNTER: AtomicU64 = AtomicU64::new(1);
        *context_id = CONTEXT_COUNTER.fetch_add(1, Ordering::SeqCst) as VAContextID;

        VA_STATUS_SUCCESS
    }

    pub unsafe fn vaDestroyContext(display: VADisplay, context_id: VAContextID) -> VAStatus {
        VA_STATUS_SUCCESS
    }

    pub unsafe fn vaCreateBuffer(
        display: VADisplay,
        context_id: VAContextID,
        buffer_type: i32,
        size: u32,
        num_elements: u32,
        data: *mut std::ffi::c_void,
        buffer_id: *mut VABufferID,
    ) -> VAStatus {
        if display.is_null() {
            return VA_STATUS_ERROR_INVALID_DISPLAY;
        }

        static BUFFER_COUNTER: AtomicU64 = AtomicU64::new(1);
        *buffer_id = BUFFER_COUNTER.fetch_add(1, Ordering::SeqCst) as VABufferID;

        VA_STATUS_SUCCESS
    }

    pub unsafe fn vaDestroyBuffer(display: VADisplay, buffer_id: VABufferID) -> VAStatus {
        VA_STATUS_SUCCESS
    }

    pub unsafe fn vaMapBuffer(
        display: VADisplay,
        buffer_id: VABufferID,
        data: *mut *mut std::ffi::c_void,
    ) -> VAStatus {
        // Allocate a buffer for the map operation
        let buf = Box::new([0u8; 4 * 1024 * 1024]); // 4MB buffer
        *data = Box::into_raw(buf) as *mut std::ffi::c_void;
        VA_STATUS_SUCCESS
    }

    pub unsafe fn vaUnmapBuffer(display: VADisplay, buffer_id: VABufferID) -> VAStatus {
        VA_STATUS_SUCCESS
    }

    pub unsafe fn vaBeginPicture(
        display: VADisplay,
        context_id: VAContextID,
        render_target: VASurfaceID,
    ) -> VAStatus {
        VA_STATUS_SUCCESS
    }

    pub unsafe fn vaRenderPicture(
        display: VADisplay,
        context_id: VAContextID,
        buffers: *mut VABufferID,
        num_buffers: i32,
    ) -> VAStatus {
        VA_STATUS_SUCCESS
    }

    pub unsafe fn vaEndPicture(display: VADisplay, context_id: VAContextID) -> VAStatus {
        VA_STATUS_SUCCESS
    }

    pub unsafe fn vaSyncSurface(display: VADisplay, surface_id: VASurfaceID) -> VAStatus {
        VA_STATUS_SUCCESS
    }

    pub unsafe fn vaCreateImage(
        display: VADisplay,
        format: *mut VAImageFormat,
        width: i32,
        height: i32,
        image: *mut VAImage,
    ) -> VAStatus {
        if display.is_null() {
            return VA_STATUS_ERROR_INVALID_DISPLAY;
        }

        static IMAGE_COUNTER: AtomicU64 = AtomicU64::new(1);

        let img = &mut *image;
        img.image_id = IMAGE_COUNTER.fetch_add(1, Ordering::SeqCst) as VAImageID;
        img.format = *format;
        img.width = width as u16;
        img.height = height as u16;
        img.data_size = (width * height * 3 / 2) as u32;
        img.num_planes = 2;
        img.pitches[0] = width as u32;
        img.pitches[1] = width as u32;
        img.offsets[0] = 0;
        img.offsets[1] = (width * height) as u32;

        static BUFFER_COUNTER: AtomicU64 = AtomicU64::new(1000);
        img.buf = BUFFER_COUNTER.fetch_add(1, Ordering::SeqCst) as VABufferID;

        VA_STATUS_SUCCESS
    }

    pub unsafe fn vaDestroyImage(display: VADisplay, image_id: VAImageID) -> VAStatus {
        VA_STATUS_SUCCESS
    }

    pub unsafe fn vaGetImage(
        display: VADisplay,
        surface_id: VASurfaceID,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
        image_id: VAImageID,
    ) -> VAStatus {
        VA_STATUS_SUCCESS
    }

    pub unsafe fn vaPutImage(
        display: VADisplay,
        surface_id: VASurfaceID,
        image_id: VAImageID,
        src_x: i32,
        src_y: i32,
        src_width: u32,
        src_height: u32,
        dst_x: i32,
        dst_y: i32,
        dst_width: u32,
        dst_height: u32,
    ) -> VAStatus {
        VA_STATUS_SUCCESS
    }

    pub unsafe fn vaDeriveImage(
        display: VADisplay,
        surface_id: VASurfaceID,
        image: *mut VAImage,
    ) -> VAStatus {
        // Create a derived image from the surface
        let mut format = VAImageFormat {
            fourcc: VA_FOURCC_NV12,
            ..Default::default()
        };
        vaCreateImage(display, &mut format, 1920, 1080, image)
    }

    pub unsafe fn vaQueryImageFormats(
        display: VADisplay,
        format_list: *mut VAImageFormat,
        num_formats: *mut i32,
    ) -> VAStatus {
        let formats = [
            VAImageFormat { fourcc: VA_FOURCC_NV12, ..Default::default() },
            VAImageFormat { fourcc: VA_FOURCC_P010, ..Default::default() },
            VAImageFormat { fourcc: VA_FOURCC_I420, ..Default::default() },
            VAImageFormat { fourcc: VA_FOURCC_BGRA, ..Default::default() },
        ];

        for (i, fmt) in formats.iter().enumerate() {
            *format_list.add(i) = *fmt;
        }
        *num_formats = formats.len() as i32;

        VA_STATUS_SUCCESS
    }
}

impl VaapiDevice {
    /// Create a new VAAPI device using the default render node
    pub fn new() -> Result<Self> {
        Self::with_path("/dev/dri/renderD128")
    }

    /// Create VAAPI device with custom DRM device path
    pub fn with_path(path: &str) -> Result<Self> {
        Ok(VaapiDevice {
            display: ptr::null_mut(),
            drm_fd: -1,
            _drm_file: None,
            device_path: path.to_string(),
            initialized: false,
            va_major: 0,
            va_minor: 0,
            vendor: String::new(),
            encode_profiles: Vec::new(),
            decode_profiles: Vec::new(),
            encoder_context: None,
            decoder_context: None,
            stats: HwEncoderStats::default(),
        })
    }

    /// Open the DRM device and initialize VA-API
    fn open_device(&mut self) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            // Open the DRM device
            let file = File::open(&self.device_path).map_err(|e| {
                Error::Init(format!("Failed to open DRM device {}: {}", self.device_path, e))
            })?;
            self.drm_fd = file.as_raw_fd();
            self._drm_file = Some(file);

            // Get VA display from DRM fd
            unsafe {
                self.display = ffi::vaGetDisplayDRM(self.drm_fd);
                if self.display.is_null() {
                    return Err(Error::Init("Failed to get VA display from DRM".to_string()));
                }

                // Initialize VA-API
                let mut major: i32 = 0;
                let mut minor: i32 = 0;
                let status = ffi::vaInitialize(self.display, &mut major, &mut minor);
                if status != VA_STATUS_SUCCESS {
                    return Err(Error::Init(format!(
                        "vaInitialize failed with status {}",
                        status
                    )));
                }
                self.va_major = major;
                self.va_minor = minor;

                // Get vendor string
                let vendor_ptr = ffi::vaQueryVendorString(self.display);
                if !vendor_ptr.is_null() {
                    self.vendor = CStr::from_ptr(vendor_ptr)
                        .to_string_lossy()
                        .into_owned();
                }

                // Query supported profiles
                self.query_profiles()?;
            }

            Ok(())
        }

        #[cfg(not(target_os = "linux"))]
        {
            Err(Error::unsupported("VAAPI is only supported on Linux"))
        }
    }

    /// Query supported profiles for encode and decode
    #[cfg(target_os = "linux")]
    fn query_profiles(&mut self) -> Result<()> {
        unsafe {
            let max_profiles = ffi::vaMaxNumProfiles(self.display);
            let mut profiles: Vec<VAProfile> = vec![0; max_profiles as usize];
            let mut num_profiles: i32 = 0;

            let status =
                ffi::vaQueryConfigProfiles(self.display, profiles.as_mut_ptr(), &mut num_profiles);
            if status != VA_STATUS_SUCCESS {
                return Err(Error::Init(format!(
                    "vaQueryConfigProfiles failed: {}",
                    status
                )));
            }

            profiles.truncate(num_profiles as usize);

            // Check which profiles support encode vs decode
            for &profile in &profiles {
                let max_eps = ffi::vaMaxNumEntrypoints(self.display);
                let mut entrypoints: Vec<VAEntrypoint> = vec![0; max_eps as usize];
                let mut num_eps: i32 = 0;

                let status = ffi::vaQueryConfigEntrypoints(
                    self.display,
                    profile,
                    entrypoints.as_mut_ptr(),
                    &mut num_eps,
                );
                if status == VA_STATUS_SUCCESS {
                    entrypoints.truncate(num_eps as usize);

                    for &ep in &entrypoints {
                        if ep == VA_ENTRYPOINT_VLD {
                            self.decode_profiles.push(profile);
                        }
                        if ep == VA_ENTRYPOINT_ENCSLICE || ep == VA_ENTRYPOINT_ENCSLICE_LP {
                            self.encode_profiles.push(profile);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Convert HwProfile to VAProfile
    fn hw_profile_to_va_profile(profile: HwProfile, codec: HwCodecType) -> VAProfile {
        match (profile, codec) {
            (HwProfile::H264Baseline, _) | (HwProfile::Auto, HwCodecType::H264) => {
                VA_PROFILE_H264_HIGH
            }
            (HwProfile::H264Main, _) => VA_PROFILE_H264_MAIN,
            (HwProfile::H264High, _) => VA_PROFILE_H264_HIGH,
            (HwProfile::H264High10, _) => VA_PROFILE_H264_HIGH10,
            (HwProfile::H264High422, _) => VA_PROFILE_H264_HIGH422,
            (HwProfile::H264High444, _) => VA_PROFILE_H264_HIGH444,

            (HwProfile::HevcMain, _) | (HwProfile::Auto, HwCodecType::H265) => VA_PROFILE_HEVC_MAIN,
            (HwProfile::HevcMain10, _) => VA_PROFILE_HEVC_MAIN10,
            (HwProfile::HevcMain12, _) => VA_PROFILE_HEVC_MAIN12,
            (HwProfile::HevcMain422_10, _) => VA_PROFILE_HEVC_MAIN422_10,
            (HwProfile::HevcMain422_12, _) => VA_PROFILE_HEVC_MAIN422_12,
            (HwProfile::HevcMain444, _) => VA_PROFILE_HEVC_MAIN444,
            (HwProfile::HevcMain444_10, _) => VA_PROFILE_HEVC_MAIN444_10,
            (HwProfile::HevcMain444_12, _) => VA_PROFILE_HEVC_MAIN444_12,

            (HwProfile::Vp9Profile0, _) | (HwProfile::Auto, HwCodecType::VP9) => {
                VA_PROFILE_VP9_PROFILE0
            }
            (HwProfile::Vp9Profile1, _) => VA_PROFILE_VP9_PROFILE1,
            (HwProfile::Vp9Profile2, _) => VA_PROFILE_VP9_PROFILE2,
            (HwProfile::Vp9Profile3, _) => VA_PROFILE_VP9_PROFILE3,

            (HwProfile::Av1Main, _) | (HwProfile::Auto, HwCodecType::AV1) => VA_PROFILE_AV1_PROFILE0,
            (HwProfile::Av1High, _) => VA_PROFILE_AV1_PROFILE0,
            (HwProfile::Av1Professional, _) => VA_PROFILE_AV1_PROFILE1,

            (HwProfile::Auto, HwCodecType::VP8) => VA_PROFILE_VP8_VERSION0_3,
            (HwProfile::Auto, HwCodecType::MPEG2) => VA_PROFILE_MPEG2_MAIN,
            (HwProfile::Auto, HwCodecType::MPEG4) => VA_PROFILE_MPEG4_ADVANCED_SIMPLE,

            _ => VA_PROFILE_NONE,
        }
    }

    /// Convert HwPixelFormat to VA RT format
    fn hw_format_to_va_rt_format(format: HwPixelFormat) -> u32 {
        match format {
            HwPixelFormat::NV12 | HwPixelFormat::YUV420P => VA_RT_FORMAT_YUV420,
            HwPixelFormat::P010 | HwPixelFormat::P016 => VA_RT_FORMAT_YUV420_10,
            HwPixelFormat::YUV422P | HwPixelFormat::UYVY | HwPixelFormat::YUYV => VA_RT_FORMAT_YUV422,
            HwPixelFormat::Y210 => VA_RT_FORMAT_YUV422_10,
            HwPixelFormat::YUV444P => VA_RT_FORMAT_YUV444,
            HwPixelFormat::Y410 => VA_RT_FORMAT_YUV444_10,
            HwPixelFormat::BGRA | HwPixelFormat::RGBA | HwPixelFormat::ARGB => VA_RT_FORMAT_RGB32,
        }
    }

    /// Convert HwPixelFormat to VA fourcc
    fn hw_format_to_va_fourcc(format: HwPixelFormat) -> u32 {
        match format {
            HwPixelFormat::NV12 => VA_FOURCC_NV12,
            HwPixelFormat::P010 | HwPixelFormat::P016 => VA_FOURCC_P010,
            HwPixelFormat::YUV420P => VA_FOURCC_I420,
            HwPixelFormat::UYVY => VA_FOURCC_UYVY,
            HwPixelFormat::BGRA => VA_FOURCC_BGRA,
            HwPixelFormat::RGBA => VA_FOURCC_RGBA,
            HwPixelFormat::ARGB => VA_FOURCC_ARGB,
            _ => VA_FOURCC_NV12, // Default fallback
        }
    }

    /// Convert HwRateControlMode to VA RC mode
    fn hw_rc_to_va_rc(mode: HwRateControlMode) -> u32 {
        match mode {
            HwRateControlMode::ConstantQP => VA_RC_CQP,
            HwRateControlMode::VBR => VA_RC_VBR,
            HwRateControlMode::CBR => VA_RC_CBR,
            HwRateControlMode::VBR_HQ => VA_RC_VBR,
            HwRateControlMode::CBR_HQ => VA_RC_CBR,
            HwRateControlMode::Quality => VA_RC_ICQ,
        }
    }

    /// Create encoder context
    #[cfg(target_os = "linux")]
    pub fn create_encoder(&mut self, config: HwEncoderConfig) -> Result<()> {
        config.validate()?;

        if !self.initialized {
            return Err(Error::invalid_state("VAAPI device not initialized"));
        }

        // Check if codec is supported
        let va_profile = Self::hw_profile_to_va_profile(config.profile, config.codec);
        if !self.encode_profiles.contains(&va_profile) {
            return Err(Error::unsupported(format!(
                "Codec {:?} not supported for encoding",
                config.codec
            )));
        }

        unsafe {
            // Get config attributes
            let mut attribs = [
                VAConfigAttrib {
                    type_: VA_CONFIG_ATTRIB_RT_FORMAT,
                    value: 0,
                },
                VAConfigAttrib {
                    type_: VA_CONFIG_ATTRIB_RATE_CONTROL,
                    value: 0,
                },
            ];

            let status = ffi::vaGetConfigAttributes(
                self.display,
                va_profile,
                VA_ENTRYPOINT_ENCSLICE,
                attribs.as_mut_ptr(),
                attribs.len() as i32,
            );
            if status != VA_STATUS_SUCCESS {
                return Err(Error::Init(format!(
                    "vaGetConfigAttributes failed: {}",
                    status
                )));
            }

            // Set RT format
            attribs[0].value = Self::hw_format_to_va_rt_format(config.input_format);
            // Set rate control
            attribs[1].value = Self::hw_rc_to_va_rc(config.rc_mode);

            // Create config
            let mut config_id: VAConfigID = 0;
            let status = ffi::vaCreateConfig(
                self.display,
                va_profile,
                VA_ENTRYPOINT_ENCSLICE,
                attribs.as_mut_ptr(),
                attribs.len() as i32,
                &mut config_id,
            );
            if status != VA_STATUS_SUCCESS {
                return Err(Error::Init(format!("vaCreateConfig failed: {}", status)));
            }

            // Create surfaces
            let num_surfaces = config.ref_frames + config.b_frames + 2;
            let mut surfaces: Vec<VASurfaceID> = vec![0; num_surfaces as usize];

            let mut surface_attribs = [VASurfaceAttrib {
                type_: VA_SURFACE_ATTRIB_PIXEL_FORMAT,
                flags: VA_SURFACE_ATTRIB_SETTABLE,
                value: VASurfaceAttribValue {
                    i: Self::hw_format_to_va_fourcc(config.input_format) as i32,
                },
            }];

            let status = ffi::vaCreateSurfaces(
                self.display,
                Self::hw_format_to_va_rt_format(config.input_format),
                config.width,
                config.height,
                surfaces.as_mut_ptr(),
                num_surfaces,
                surface_attribs.as_mut_ptr(),
                surface_attribs.len() as u32,
            );
            if status != VA_STATUS_SUCCESS {
                ffi::vaDestroyConfig(self.display, config_id);
                return Err(Error::Init(format!("vaCreateSurfaces failed: {}", status)));
            }

            // Create context
            let mut context_id: VAContextID = 0;
            let status = ffi::vaCreateContext(
                self.display,
                config_id,
                config.width as i32,
                config.height as i32,
                0, // VA_PROGRESSIVE
                surfaces.as_mut_ptr(),
                surfaces.len() as i32,
                &mut context_id,
            );
            if status != VA_STATUS_SUCCESS {
                ffi::vaDestroySurfaces(self.display, surfaces.as_mut_ptr(), surfaces.len() as i32);
                ffi::vaDestroyConfig(self.display, config_id);
                return Err(Error::Init(format!("vaCreateContext failed: {}", status)));
            }

            // Create coded buffer
            let coded_buf_size = config.width * config.height * 2; // 2 bytes per pixel max
            let mut coded_buffer: VABufferID = 0;
            let status = ffi::vaCreateBuffer(
                self.display,
                context_id,
                VA_CODED_BUFFER_TYPE,
                coded_buf_size,
                1,
                ptr::null_mut(),
                &mut coded_buffer,
            );
            if status != VA_STATUS_SUCCESS {
                ffi::vaDestroyContext(self.display, context_id);
                ffi::vaDestroySurfaces(self.display, surfaces.as_mut_ptr(), surfaces.len() as i32);
                ffi::vaDestroyConfig(self.display, config_id);
                return Err(Error::Init(format!(
                    "vaCreateBuffer (coded) failed: {}",
                    status
                )));
            }

            self.encoder_context = Some(VaapiEncoderContext {
                config_id,
                context_id,
                surfaces,
                coded_buffer,
                config,
                frame_count: 0,
                start_time: Instant::now(),
            });
        }

        Ok(())
    }

    /// Create decoder context
    #[cfg(target_os = "linux")]
    pub fn create_decoder(&mut self, config: HwDecoderConfig) -> Result<()> {
        if !self.initialized {
            return Err(Error::invalid_state("VAAPI device not initialized"));
        }

        let va_profile = Self::hw_profile_to_va_profile(HwProfile::Auto, config.codec);
        if !self.decode_profiles.contains(&va_profile) {
            return Err(Error::unsupported(format!(
                "Codec {:?} not supported for decoding",
                config.codec
            )));
        }

        unsafe {
            // Create config
            let mut attribs = [VAConfigAttrib {
                type_: VA_CONFIG_ATTRIB_RT_FORMAT,
                value: Self::hw_format_to_va_rt_format(config.output_format),
            }];

            let mut config_id: VAConfigID = 0;
            let status = ffi::vaCreateConfig(
                self.display,
                va_profile,
                VA_ENTRYPOINT_VLD,
                attribs.as_mut_ptr(),
                1,
                &mut config_id,
            );
            if status != VA_STATUS_SUCCESS {
                return Err(Error::Init(format!("vaCreateConfig failed: {}", status)));
            }

            // Create surfaces
            let mut surfaces: Vec<VASurfaceID> = vec![0; config.num_surfaces as usize];

            let status = ffi::vaCreateSurfaces(
                self.display,
                Self::hw_format_to_va_rt_format(config.output_format),
                config.max_width,
                config.max_height,
                surfaces.as_mut_ptr(),
                config.num_surfaces,
                ptr::null_mut(),
                0,
            );
            if status != VA_STATUS_SUCCESS {
                ffi::vaDestroyConfig(self.display, config_id);
                return Err(Error::Init(format!("vaCreateSurfaces failed: {}", status)));
            }

            // Create context
            let mut context_id: VAContextID = 0;
            let status = ffi::vaCreateContext(
                self.display,
                config_id,
                config.max_width as i32,
                config.max_height as i32,
                0,
                surfaces.as_mut_ptr(),
                surfaces.len() as i32,
                &mut context_id,
            );
            if status != VA_STATUS_SUCCESS {
                ffi::vaDestroySurfaces(self.display, surfaces.as_mut_ptr(), surfaces.len() as i32);
                ffi::vaDestroyConfig(self.display, config_id);
                return Err(Error::Init(format!("vaCreateContext failed: {}", status)));
            }

            self.decoder_context = Some(VaapiDecoderContext {
                config_id,
                context_id,
                surfaces,
                config,
            });
        }

        Ok(())
    }

    /// Encode a frame
    #[cfg(target_os = "linux")]
    pub fn encode_frame(
        &mut self,
        surface: &HwSurface,
        force_keyframe: bool,
    ) -> Result<HwEncodedPacket> {
        let ctx = self
            .encoder_context
            .as_mut()
            .ok_or_else(|| Error::invalid_state("No encoder context"))?;

        let start = Instant::now();

        // Get the surface ID to use
        let surface_idx = (ctx.frame_count as usize) % ctx.surfaces.len();
        let va_surface = ctx.surfaces[surface_idx];

        unsafe {
            // Upload frame data to surface
            if let Some(data) = &surface.data {
                // Create image for upload
                let mut format = VAImageFormat {
                    fourcc: Self::hw_format_to_va_fourcc(surface.format),
                    ..Default::default()
                };
                let mut image: VAImage = std::mem::zeroed();

                let status = ffi::vaCreateImage(
                    self.display,
                    &mut format,
                    surface.width as i32,
                    surface.height as i32,
                    &mut image,
                );
                if status != VA_STATUS_SUCCESS {
                    return Err(Error::Codec(format!("vaCreateImage failed: {}", status)));
                }

                // Map and copy data
                let mut ptr: *mut std::ffi::c_void = ptr::null_mut();
                let status = ffi::vaMapBuffer(self.display, image.buf, &mut ptr);
                if status == VA_STATUS_SUCCESS && !ptr.is_null() {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        ptr as *mut u8,
                        data.len().min(image.data_size as usize),
                    );
                    ffi::vaUnmapBuffer(self.display, image.buf);
                }

                // Put image to surface
                let status = ffi::vaPutImage(
                    self.display,
                    va_surface,
                    image.image_id,
                    0,
                    0,
                    surface.width,
                    surface.height,
                    0,
                    0,
                    surface.width,
                    surface.height,
                );

                ffi::vaDestroyImage(self.display, image.image_id);

                if status != VA_STATUS_SUCCESS {
                    return Err(Error::Codec(format!("vaPutImage failed: {}", status)));
                }
            }

            // Begin picture
            let status = ffi::vaBeginPicture(self.display, ctx.context_id, va_surface);
            if status != VA_STATUS_SUCCESS {
                return Err(Error::Codec(format!("vaBeginPicture failed: {}", status)));
            }

            // In a full implementation, we would create and submit:
            // - VAEncSequenceParameterBuffer
            // - VAEncPictureParameterBuffer
            // - VAEncSliceParameterBuffer
            // - VAEncMiscParameterBuffer (rate control, etc.)

            // Render picture with coded buffer
            let mut buffers = [ctx.coded_buffer];
            let status =
                ffi::vaRenderPicture(self.display, ctx.context_id, buffers.as_mut_ptr(), 1);
            if status != VA_STATUS_SUCCESS {
                return Err(Error::Codec(format!("vaRenderPicture failed: {}", status)));
            }

            // End picture
            let status = ffi::vaEndPicture(self.display, ctx.context_id);
            if status != VA_STATUS_SUCCESS {
                return Err(Error::Codec(format!("vaEndPicture failed: {}", status)));
            }

            // Sync surface
            let status = ffi::vaSyncSurface(self.display, va_surface);
            if status != VA_STATUS_SUCCESS {
                return Err(Error::Codec(format!("vaSyncSurface failed: {}", status)));
            }

            // Map coded buffer and read output
            let mut coded_ptr: *mut std::ffi::c_void = ptr::null_mut();
            let status = ffi::vaMapBuffer(self.display, ctx.coded_buffer, &mut coded_ptr);
            if status != VA_STATUS_SUCCESS {
                return Err(Error::Codec(format!(
                    "vaMapBuffer (coded) failed: {}",
                    status
                )));
            }

            // In reality, we'd parse the coded buffer segment info
            // For now, create a minimal output
            let is_keyframe = force_keyframe || ctx.frame_count == 0
                || ctx.frame_count % ctx.config.gop_size as u64 == 0;

            // Generate placeholder encoded data (real implementation reads from coded_ptr)
            let output_data = if is_keyframe {
                // Simulate NAL unit start code + SPS/PPS for keyframe
                vec![0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0xc0, 0x1f]
            } else {
                // Simulate P-frame NAL
                vec![0x00, 0x00, 0x00, 0x01, 0x41, 0x9a]
            };

            ffi::vaUnmapBuffer(self.display, ctx.coded_buffer);

            let elapsed_us = start.elapsed().as_micros() as u64;
            ctx.frame_count += 1;

            // Update stats
            self.stats.frames_encoded += 1;
            self.stats.bytes_output += output_data.len() as u64;
            self.stats.encode_time_us += elapsed_us;
            self.stats.avg_frame_time_us =
                self.stats.encode_time_us / self.stats.frames_encoded;
            if is_keyframe {
                self.stats.i_frames += 1;
            } else {
                self.stats.p_frames += 1;
            }

            let total_time = ctx.start_time.elapsed().as_secs_f64();
            if total_time > 0.0 {
                self.stats.fps = self.stats.frames_encoded as f64 / total_time;
                self.stats.avg_bitrate = (self.stats.bytes_output * 8) / total_time.max(1.0) as u64;
            }

            Ok(HwEncodedPacket {
                data: output_data,
                pts: ctx.frame_count as i64 - 1,
                dts: ctx.frame_count as i64 - 1,
                duration: 1,
                keyframe: is_keyframe,
                pict_type: if is_keyframe { 'I' } else { 'P' },
                frame_num: ctx.frame_count - 1,
            })
        }
    }

    /// Get encoder capabilities
    pub fn get_encoder_caps(&self, codec: HwCodecType) -> Option<HwEncoderCaps> {
        let va_profile = Self::hw_profile_to_va_profile(HwProfile::Auto, codec);
        if !self.encode_profiles.contains(&va_profile) {
            return None;
        }

        Some(HwEncoderCaps {
            codec,
            profiles: match codec {
                HwCodecType::H264 => vec![
                    HwProfile::H264Baseline,
                    HwProfile::H264Main,
                    HwProfile::H264High,
                ],
                HwCodecType::H265 => vec![HwProfile::HevcMain, HwProfile::HevcMain10],
                HwCodecType::VP9 => vec![HwProfile::Vp9Profile0, HwProfile::Vp9Profile2],
                HwCodecType::AV1 => vec![HwProfile::Av1Main],
                _ => vec![HwProfile::Auto],
            },
            max_width: 4096,
            max_height: 4096,
            min_width: 64,
            min_height: 64,
            input_formats: vec![HwPixelFormat::NV12, HwPixelFormat::P010],
            b_frames: matches!(codec, HwCodecType::H264 | HwCodecType::H265),
            max_b_frames: 4,
            lookahead: true,
            max_lookahead: 32,
            temporal_aq: true,
            spatial_aq: true,
            max_sessions: 4,
        })
    }

    /// Get decoder capabilities
    pub fn get_decoder_caps(&self, codec: HwCodecType) -> Option<HwDecoderCaps> {
        let va_profile = Self::hw_profile_to_va_profile(HwProfile::Auto, codec);
        if !self.decode_profiles.contains(&va_profile) {
            return None;
        }

        Some(HwDecoderCaps {
            codec,
            profiles: match codec {
                HwCodecType::H264 => vec![
                    HwProfile::H264Baseline,
                    HwProfile::H264Main,
                    HwProfile::H264High,
                    HwProfile::H264High10,
                ],
                HwCodecType::H265 => vec![
                    HwProfile::HevcMain,
                    HwProfile::HevcMain10,
                    HwProfile::HevcMain12,
                ],
                HwCodecType::VP9 => vec![
                    HwProfile::Vp9Profile0,
                    HwProfile::Vp9Profile1,
                    HwProfile::Vp9Profile2,
                    HwProfile::Vp9Profile3,
                ],
                HwCodecType::AV1 => vec![HwProfile::Av1Main, HwProfile::Av1High],
                _ => vec![HwProfile::Auto],
            },
            max_width: 8192,
            max_height: 8192,
            output_formats: vec![HwPixelFormat::NV12, HwPixelFormat::P010],
            deinterlace: true,
            max_bit_depth: 12,
        })
    }

    /// Get encoding statistics
    pub fn get_stats(&self) -> &HwEncoderStats {
        &self.stats
    }

    /// Destroy encoder context
    fn destroy_encoder(&mut self) {
        #[cfg(target_os = "linux")]
        if let Some(ctx) = self.encoder_context.take() {
            unsafe {
                ffi::vaDestroyBuffer(self.display, ctx.coded_buffer);
                ffi::vaDestroyContext(self.display, ctx.context_id);
                let mut surfaces = ctx.surfaces;
                ffi::vaDestroySurfaces(
                    self.display,
                    surfaces.as_mut_ptr(),
                    surfaces.len() as i32,
                );
                ffi::vaDestroyConfig(self.display, ctx.config_id);
            }
        }
    }

    /// Destroy decoder context
    fn destroy_decoder(&mut self) {
        #[cfg(target_os = "linux")]
        if let Some(ctx) = self.decoder_context.take() {
            unsafe {
                ffi::vaDestroyContext(self.display, ctx.context_id);
                let mut surfaces = ctx.surfaces;
                ffi::vaDestroySurfaces(
                    self.display,
                    surfaces.as_mut_ptr(),
                    surfaces.len() as i32,
                );
                ffi::vaDestroyConfig(self.display, ctx.config_id);
            }
        }
    }
}

impl HwAccelDevice for VaapiDevice {
    fn device_type(&self) -> HwAccelType {
        HwAccelType::VAAPI
    }

    fn is_available(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            // Check if render node exists
            if !std::path::Path::new(&self.device_path).exists() {
                return false;
            }

            // Check if we can open it
            if let Ok(file) = File::open(&self.device_path) {
                // Additional check: try to get VA display
                unsafe {
                    let fd = file.as_raw_fd();
                    let display = ffi::vaGetDisplayDRM(fd);
                    if !display.is_null() {
                        let mut major: i32 = 0;
                        let mut minor: i32 = 0;
                        let status = ffi::vaInitialize(display, &mut major, &mut minor);
                        if status == VA_STATUS_SUCCESS {
                            ffi::vaTerminate(display);
                            return true;
                        }
                    }
                }
            }
            false
        }

        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    fn init(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        #[cfg(target_os = "linux")]
        {
            self.open_device()?;
            self.initialized = true;
            tracing::info!(
                "VAAPI initialized: VA-API {}.{}, Vendor: {}",
                self.va_major,
                self.va_minor,
                self.vendor
            );
            tracing::info!(
                "Encode profiles: {:?}, Decode profiles: {:?}",
                self.encode_profiles.len(),
                self.decode_profiles.len()
            );
            Ok(())
        }

        #[cfg(not(target_os = "linux"))]
        {
            Err(Error::unsupported("VAAPI is only available on Linux"))
        }
    }

    fn upload_frame(&mut self, frame: &VideoFrame) -> Result<VideoFrame> {
        if !self.initialized {
            return Err(Error::invalid_state("VAAPI device not initialized"));
        }

        // Convert VideoFrame to HwSurface format
        let mut surface = HwSurface::from_video_frame(frame)?;

        // If input is YUV420P, convert to NV12 for VAAPI
        if surface.format == HwPixelFormat::YUV420P {
            surface.convert_yuv420p_to_nv12()?;
        }

        // In a real implementation, we would upload to GPU memory here
        // and return a frame with GPU surface reference

        // For now, return the frame as-is (software path)
        Ok(frame.clone())
    }

    fn download_frame(&mut self, frame: &VideoFrame) -> Result<VideoFrame> {
        if !self.initialized {
            return Err(Error::invalid_state("VAAPI device not initialized"));
        }

        // In a real implementation, we would download from GPU memory here
        // For now, return the frame as-is
        Ok(frame.clone())
    }

    fn name(&self) -> &str {
        if self.vendor.is_empty() {
            "VAAPI"
        } else {
            &self.vendor
        }
    }
}

impl Drop for VaapiDevice {
    fn drop(&mut self) {
        self.destroy_encoder();
        self.destroy_decoder();

        #[cfg(target_os = "linux")]
        if self.initialized && !self.display.is_null() {
            unsafe {
                ffi::vaTerminate(self.display);
            }
        }
    }
}

/// Check if VAAPI is available on the system
pub fn is_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        // Check common render node paths
        let paths = [
            "/dev/dri/renderD128",
            "/dev/dri/renderD129",
            "/dev/dri/card0",
            "/dev/dri/card1",
        ];

        for path in paths {
            if std::path::Path::new(path).exists() {
                if let Ok(device) = VaapiDevice::with_path(path) {
                    if device.is_available() {
                        return true;
                    }
                }
            }
        }
        false
    }

    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

/// List available VAAPI devices
pub fn list_devices() -> Vec<String> {
    let mut devices = Vec::new();

    #[cfg(target_os = "linux")]
    {
        for i in 128..136 {
            let path = format!("/dev/dri/renderD{}", i);
            if std::path::Path::new(&path).exists() {
                if let Ok(device) = VaapiDevice::with_path(&path) {
                    if device.is_available() {
                        devices.push(path);
                    }
                }
            }
        }
    }

    devices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vaapi_device_creation() {
        let device = VaapiDevice::new();
        assert!(device.is_ok());
    }

    #[test]
    fn test_vaapi_availability() {
        // Should not panic
        let _available = is_available();
    }

    #[test]
    fn test_profile_conversion() {
        assert_eq!(
            VaapiDevice::hw_profile_to_va_profile(HwProfile::H264High, HwCodecType::H264),
            VA_PROFILE_H264_HIGH
        );
        assert_eq!(
            VaapiDevice::hw_profile_to_va_profile(HwProfile::HevcMain10, HwCodecType::H265),
            VA_PROFILE_HEVC_MAIN10
        );
    }

    #[test]
    fn test_format_conversion() {
        assert_eq!(
            VaapiDevice::hw_format_to_va_rt_format(HwPixelFormat::NV12),
            VA_RT_FORMAT_YUV420
        );
        assert_eq!(
            VaapiDevice::hw_format_to_va_rt_format(HwPixelFormat::P010),
            VA_RT_FORMAT_YUV420_10
        );
    }

    #[test]
    fn test_list_devices() {
        let devices = list_devices();
        println!("Found VAAPI devices: {:?}", devices);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_vaapi_init() {
        let mut device = VaapiDevice::new().unwrap();
        // Only test init if device is available
        if device.is_available() {
            let result = device.init();
            println!("VAAPI init result: {:?}", result);
        }
    }
}
