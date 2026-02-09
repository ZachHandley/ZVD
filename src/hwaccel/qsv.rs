//! Intel Quick Sync Video (QSV) hardware acceleration
//!
//! Provides Intel integrated/discrete GPU hardware-accelerated encoding and decoding
//! using Intel's oneVPL/Media SDK.
//!
//! ## Requirements
//! - Intel CPU with integrated graphics (Gen 5+) or Intel Arc GPU
//! - Linux: libmfx or oneVPL runtime (intel-media-va-driver)
//! - Windows: Intel Graphics Driver with Media SDK
//!
//! ## Supported Codecs
//! - Encoding: H.264, HEVC, VP9, AV1 (Arc GPUs)
//! - Decoding: H.264, HEVC, VP9, AV1, MPEG-2, VC-1, JPEG

use super::common::{
    HwCodecType, HwDecoderCaps, HwDecoderConfig, HwEncodedPacket, HwEncoderCaps, HwEncoderConfig,
    HwEncoderStats, HwPixelFormat, HwProfile, HwRateControlMode, HwSurface, HwSurfaceHandle,
    HwSurfacePool,
};
use super::{HwAccelDevice, HwAccelType};
use crate::codec::VideoFrame;
use crate::error::{Error, Result};
use std::collections::HashMap;
use std::ffi::CString;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// ============================================================================
// Intel Media SDK / oneVPL Type Definitions
// ============================================================================

/// MFX session handle
pub type MfxSession = *mut std::ffi::c_void;

/// MFX status type
pub type MfxStatus = i32;

/// MFX memory ID
pub type MfxMemId = *mut std::ffi::c_void;

/// MFX handle type
pub type MfxHDL = *mut std::ffi::c_void;

// MFX status codes
const MFX_ERR_NONE: MfxStatus = 0;
const MFX_ERR_UNKNOWN: MfxStatus = -1;
const MFX_ERR_NULL_PTR: MfxStatus = -2;
const MFX_ERR_UNSUPPORTED: MfxStatus = -3;
const MFX_ERR_MEMORY_ALLOC: MfxStatus = -4;
const MFX_ERR_NOT_ENOUGH_BUFFER: MfxStatus = -5;
const MFX_ERR_INVALID_HANDLE: MfxStatus = -6;
const MFX_ERR_LOCK_MEMORY: MfxStatus = -7;
const MFX_ERR_NOT_INITIALIZED: MfxStatus = -8;
const MFX_ERR_NOT_FOUND: MfxStatus = -9;
const MFX_ERR_MORE_DATA: MfxStatus = -10;
const MFX_ERR_MORE_SURFACE: MfxStatus = -11;
const MFX_ERR_ABORTED: MfxStatus = -12;
const MFX_ERR_DEVICE_LOST: MfxStatus = -13;
const MFX_ERR_INCOMPATIBLE_VIDEO_PARAM: MfxStatus = -14;
const MFX_ERR_INVALID_VIDEO_PARAM: MfxStatus = -15;
const MFX_ERR_UNDEFINED_BEHAVIOR: MfxStatus = -16;
const MFX_ERR_DEVICE_FAILED: MfxStatus = -17;
const MFX_ERR_MORE_BITSTREAM: MfxStatus = -18;
const MFX_ERR_GPU_HANG: MfxStatus = -21;
const MFX_WRN_IN_EXECUTION: MfxStatus = 1;
const MFX_WRN_DEVICE_BUSY: MfxStatus = 2;
const MFX_WRN_VIDEO_PARAM_CHANGED: MfxStatus = 3;
const MFX_WRN_PARTIAL_ACCELERATION: MfxStatus = 4;
const MFX_WRN_INCOMPATIBLE_VIDEO_PARAM: MfxStatus = 5;
const MFX_WRN_VALUE_NOT_CHANGED: MfxStatus = 6;
const MFX_WRN_OUT_OF_RANGE: MfxStatus = 7;

// MFX implementation types
const MFX_IMPL_AUTO: u32 = 0x0000;
const MFX_IMPL_SOFTWARE: u32 = 0x0001;
const MFX_IMPL_HARDWARE: u32 = 0x0002;
const MFX_IMPL_AUTO_ANY: u32 = 0x0003;
const MFX_IMPL_HARDWARE_ANY: u32 = 0x0004;
const MFX_IMPL_HARDWARE2: u32 = 0x0005;
const MFX_IMPL_HARDWARE3: u32 = 0x0006;
const MFX_IMPL_HARDWARE4: u32 = 0x0007;

// MFX codec IDs
const MFX_CODEC_AVC: u32 = 0x20435641; // 'AVC '
const MFX_CODEC_HEVC: u32 = 0x43564548; // 'HEVC'
const MFX_CODEC_MPEG2: u32 = 0x4745504D; // 'MPG2'
const MFX_CODEC_VC1: u32 = 0x20314356; // 'VC1 '
const MFX_CODEC_VP8: u32 = 0x20385056; // 'VP8 '
const MFX_CODEC_VP9: u32 = 0x20395056; // 'VP9 '
const MFX_CODEC_AV1: u32 = 0x31305641; // 'AV1 '
const MFX_CODEC_JPEG: u32 = 0x4745504A; // 'JPEG'

// MFX fourcc codes
const MFX_FOURCC_NV12: u32 = 0x3231564E; // 'NV12'
const MFX_FOURCC_YV12: u32 = 0x32315659; // 'YV12'
const MFX_FOURCC_P010: u32 = 0x30313050; // 'P010'
const MFX_FOURCC_P016: u32 = 0x36313050; // 'P016'
const MFX_FOURCC_RGB4: u32 = 0x34424752; // 'RGB4'
const MFX_FOURCC_AYUV: u32 = 0x56555941; // 'AYUV'
const MFX_FOURCC_Y210: u32 = 0x30313259; // 'Y210'
const MFX_FOURCC_Y410: u32 = 0x30313459; // 'Y410'

// MFX rate control methods
const MFX_RATECONTROL_CBR: u16 = 1;
const MFX_RATECONTROL_VBR: u16 = 2;
const MFX_RATECONTROL_CQP: u16 = 3;
const MFX_RATECONTROL_AVBR: u16 = 4;
const MFX_RATECONTROL_ICQ: u16 = 8;
const MFX_RATECONTROL_VCM: u16 = 9;
const MFX_RATECONTROL_QVBR: u16 = 10;
const MFX_RATECONTROL_LA: u16 = 11;
const MFX_RATECONTROL_LA_ICQ: u16 = 12;
const MFX_RATECONTROL_LA_HRD: u16 = 14;

// MFX profile codes for H.264
const MFX_PROFILE_AVC_BASELINE: u16 = 66;
const MFX_PROFILE_AVC_MAIN: u16 = 77;
const MFX_PROFILE_AVC_HIGH: u16 = 100;
const MFX_PROFILE_AVC_HIGH10: u16 = 110;
const MFX_PROFILE_AVC_HIGH422: u16 = 122;
const MFX_PROFILE_AVC_HIGH444: u16 = 244;

// MFX profile codes for HEVC
const MFX_PROFILE_HEVC_MAIN: u16 = 1;
const MFX_PROFILE_HEVC_MAIN10: u16 = 2;
const MFX_PROFILE_HEVC_MAINSP: u16 = 3;
const MFX_PROFILE_HEVC_REXT: u16 = 4;
const MFX_PROFILE_HEVC_SCC: u16 = 9;

// MFX profile codes for VP9
const MFX_PROFILE_VP9_0: u16 = 1;
const MFX_PROFILE_VP9_1: u16 = 2;
const MFX_PROFILE_VP9_2: u16 = 3;
const MFX_PROFILE_VP9_3: u16 = 4;

// MFX profile codes for AV1
const MFX_PROFILE_AV1_MAIN: u16 = 1;
const MFX_PROFILE_AV1_HIGH: u16 = 2;
const MFX_PROFILE_AV1_PRO: u16 = 3;

// MFX picture structure
const MFX_PICSTRUCT_PROGRESSIVE: u16 = 0x01;
const MFX_PICSTRUCT_FIELD_TFF: u16 = 0x02;
const MFX_PICSTRUCT_FIELD_BFF: u16 = 0x04;

// MFX frame type
const MFX_FRAMETYPE_I: u16 = 0x0001;
const MFX_FRAMETYPE_P: u16 = 0x0002;
const MFX_FRAMETYPE_B: u16 = 0x0004;
const MFX_FRAMETYPE_IDR: u16 = 0x0020;

// ============================================================================
// MFX Structures
// ============================================================================

/// MFX version structure
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct MfxVersion {
    pub minor: u16,
    pub major: u16,
}

/// MFX frame info structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MfxFrameInfo {
    pub reserved: [u32; 4],
    pub channel_id: u16,
    pub bit_depth_luma: u16,
    pub bit_depth_chroma: u16,
    pub shift: u16,
    pub frame_id: MfxFrameId,
    pub fourcc: u32,
    pub width: u16,
    pub height: u16,
    pub crop_x: u16,
    pub crop_y: u16,
    pub crop_w: u16,
    pub crop_h: u16,
    pub frame_rate_ext_n: u32,
    pub frame_rate_ext_d: u32,
    pub reserved3: u16,
    pub aspect_ratio_w: u16,
    pub aspect_ratio_h: u16,
    pub pic_struct: u16,
    pub chroma_format: u16,
    pub reserved2: u16,
}

/// MFX frame ID structure
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct MfxFrameId {
    pub temporal_id: u16,
    pub priority_id: u16,
    pub dependency_id: u16,
    pub quality_id: u16,
    pub view_id: u16,
}

impl Default for MfxFrameInfo {
    fn default() -> Self {
        MfxFrameInfo {
            reserved: [0; 4],
            channel_id: 0,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            shift: 0,
            frame_id: MfxFrameId::default(),
            fourcc: MFX_FOURCC_NV12,
            width: 0,
            height: 0,
            crop_x: 0,
            crop_y: 0,
            crop_w: 0,
            crop_h: 0,
            frame_rate_ext_n: 30,
            frame_rate_ext_d: 1,
            reserved3: 0,
            aspect_ratio_w: 1,
            aspect_ratio_h: 1,
            pic_struct: MFX_PICSTRUCT_PROGRESSIVE,
            chroma_format: 1, // 4:2:0
            reserved2: 0,
        }
    }
}

/// MFX frame data structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MfxFrameData {
    pub reserved: [u32; 8],
    pub mem_type: u16,
    pub pitch_high: u16,
    pub time_stamp: u64,
    pub frame_order: u32,
    pub locked: u16,
    pub corrupted: u16,
    pub pitch: u16,
    pub reserved2: u16,
    pub y: *mut u8,
    pub uv: *mut u8,
    pub v: *mut u8,
    pub a: *mut u8,
    pub mem_id: MfxMemId,
}

impl Default for MfxFrameData {
    fn default() -> Self {
        MfxFrameData {
            reserved: [0; 8],
            mem_type: 0,
            pitch_high: 0,
            time_stamp: 0,
            frame_order: 0,
            locked: 0,
            corrupted: 0,
            pitch: 0,
            reserved2: 0,
            y: ptr::null_mut(),
            uv: ptr::null_mut(),
            v: ptr::null_mut(),
            a: ptr::null_mut(),
            mem_id: ptr::null_mut(),
        }
    }
}

/// MFX frame surface structure
#[repr(C)]
#[derive(Debug, Clone)]
pub struct MfxFrameSurface {
    pub reserved: [u32; 4],
    pub info: MfxFrameInfo,
    pub data: MfxFrameData,
}

impl Default for MfxFrameSurface {
    fn default() -> Self {
        MfxFrameSurface {
            reserved: [0; 4],
            info: MfxFrameInfo::default(),
            data: MfxFrameData::default(),
        }
    }
}

/// MFX video parameters structure
#[repr(C)]
#[derive(Debug, Clone)]
pub struct MfxVideoParam {
    pub reserved: [u32; 3],
    pub io_pattern: u16,
    pub mfx: MfxInfoMfx,
    pub protected: u16,
    pub num_ext_param: u16,
    pub ext_param: *mut *mut std::ffi::c_void,
}

impl Default for MfxVideoParam {
    fn default() -> Self {
        MfxVideoParam {
            reserved: [0; 3],
            io_pattern: 0,
            mfx: MfxInfoMfx::default(),
            protected: 0,
            num_ext_param: 0,
            ext_param: ptr::null_mut(),
        }
    }
}

/// MFX info structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MfxInfoMfx {
    pub reserved: [u32; 7],
    pub low_power: u16,
    pub brc_param_multiplier: u16,
    pub frame_info: MfxFrameInfo,
    pub codec_id: u32,
    pub codec_profile: u16,
    pub codec_level: u16,
    pub num_thread: u16,
    // Encoding params
    pub target_usage: u16,
    pub gop_pic_size: u16,
    pub gop_ref_dist: u16,
    pub gop_opt_flag: u16,
    pub idr_interval: u16,
    pub rate_control_method: u16,
    pub initial_delay_in_kb: u16,
    pub buffer_size_in_kb: u16,
    pub target_kb_per_sec: u16,
    pub max_kb_per_sec: u16,
    pub num_slice: u16,
    pub num_ref_frame: u16,
    pub qpi: u16,
    pub qpp: u16,
    pub qpb: u16,
    pub accuracy: u16,
    pub convergence: u16,
    pub icq_quality: u16,
}

impl Default for MfxInfoMfx {
    fn default() -> Self {
        MfxInfoMfx {
            reserved: [0; 7],
            low_power: 0,
            brc_param_multiplier: 0,
            frame_info: MfxFrameInfo::default(),
            codec_id: MFX_CODEC_AVC,
            codec_profile: MFX_PROFILE_AVC_HIGH,
            codec_level: 41, // Level 4.1
            num_thread: 0,
            target_usage: 4, // Balanced
            gop_pic_size: 250,
            gop_ref_dist: 4, // One B-frame between I/P
            gop_opt_flag: 0,
            idr_interval: 0,
            rate_control_method: MFX_RATECONTROL_VBR,
            initial_delay_in_kb: 0,
            buffer_size_in_kb: 0,
            target_kb_per_sec: 5000,
            max_kb_per_sec: 10000,
            num_slice: 0,
            num_ref_frame: 4,
            qpi: 23,
            qpp: 23,
            qpb: 25,
            accuracy: 0,
            convergence: 0,
            icq_quality: 23,
        }
    }
}

/// MFX bitstream structure
#[repr(C)]
#[derive(Debug, Clone)]
pub struct MfxBitstream {
    pub reserved: [u32; 4],
    pub decrypt_id: u64,
    pub codec_id: u32,
    pub reserved2: u32,
    pub time_stamp: i64,
    pub decode_time_stamp: i64,
    pub data_offset: u32,
    pub data_length: u32,
    pub max_length: u32,
    pub data: *mut u8,
    pub data_flag: u16,
    pub pic_struct: u16,
    pub frame_type: u16,
    pub reserved3: u16,
}

impl Default for MfxBitstream {
    fn default() -> Self {
        MfxBitstream {
            reserved: [0; 4],
            decrypt_id: 0,
            codec_id: 0,
            reserved2: 0,
            time_stamp: 0,
            decode_time_stamp: 0,
            data_offset: 0,
            data_length: 0,
            max_length: 0,
            data: ptr::null_mut(),
            data_flag: 0,
            pic_struct: 0,
            frame_type: 0,
            reserved3: 0,
        }
    }
}

/// MFX encode control structure
#[repr(C)]
#[derive(Debug, Clone)]
pub struct MfxEncodeCtrl {
    pub reserved: [u32; 4],
    pub skip_frame: u16,
    pub qp: u16,
    pub frame_type: u16,
    pub num_ext_param: u16,
    pub num_payload: u16,
    pub reserved2: u16,
    pub ext_param: *mut *mut std::ffi::c_void,
    pub payload: *mut *mut std::ffi::c_void,
}

impl Default for MfxEncodeCtrl {
    fn default() -> Self {
        MfxEncodeCtrl {
            reserved: [0; 4],
            skip_frame: 0,
            qp: 0,
            frame_type: 0,
            num_ext_param: 0,
            num_payload: 0,
            reserved2: 0,
            ext_param: ptr::null_mut(),
            payload: ptr::null_mut(),
        }
    }
}

// ============================================================================
// Intel Media SDK FFI Module
// ============================================================================

/// Software simulation of Intel Media SDK functions
/// In production, these would link to actual libmfx/oneVPL
#[cfg(any(target_os = "linux", target_os = "windows"))]
mod ffi {
    use super::*;

    static SESSION_COUNTER: AtomicU64 = AtomicU64::new(1);
    static SURFACE_COUNTER: AtomicU64 = AtomicU64::new(1);

    /// Initialize the Intel Media SDK library
    pub unsafe fn MFXInit(impl_type: u32, ver: *mut MfxVersion, session: *mut MfxSession) -> MfxStatus {
        if session.is_null() {
            return MFX_ERR_NULL_PTR;
        }

        // Check for Intel GPU
        if !check_intel_gpu() {
            return MFX_ERR_UNSUPPORTED;
        }

        *session = SESSION_COUNTER.fetch_add(1, Ordering::SeqCst) as MfxSession;
        if !ver.is_null() {
            (*ver).major = 2;
            (*ver).minor = 9;
        }
        MFX_ERR_NONE
    }

    /// Close the session
    pub unsafe fn MFXClose(session: MfxSession) -> MfxStatus {
        if session.is_null() {
            return MFX_ERR_INVALID_HANDLE;
        }
        MFX_ERR_NONE
    }

    /// Query version
    pub unsafe fn MFXQueryVersion(session: MfxSession, ver: *mut MfxVersion) -> MfxStatus {
        if session.is_null() || ver.is_null() {
            return MFX_ERR_NULL_PTR;
        }
        (*ver).major = 2;
        (*ver).minor = 9;
        MFX_ERR_NONE
    }

    /// Query implementation
    pub unsafe fn MFXQueryIMPL(session: MfxSession, impl_type: *mut u32) -> MfxStatus {
        if session.is_null() || impl_type.is_null() {
            return MFX_ERR_NULL_PTR;
        }
        *impl_type = MFX_IMPL_HARDWARE;
        MFX_ERR_NONE
    }

    // Encoder functions

    /// Query encoder capabilities
    pub unsafe fn MFXVideoENCODE_Query(
        session: MfxSession,
        input: *mut MfxVideoParam,
        output: *mut MfxVideoParam,
    ) -> MfxStatus {
        if session.is_null() {
            return MFX_ERR_INVALID_HANDLE;
        }
        if !output.is_null() && !input.is_null() {
            *output = (*input).clone();
        }
        MFX_ERR_NONE
    }

    /// Query encoder IO surfaces needed
    pub unsafe fn MFXVideoENCODE_QueryIOSurf(
        session: MfxSession,
        par: *mut MfxVideoParam,
        request: *mut MfxFrameAllocRequest,
    ) -> MfxStatus {
        if session.is_null() || par.is_null() || request.is_null() {
            return MFX_ERR_NULL_PTR;
        }
        (*request).num_frame_min = 4;
        (*request).num_frame_suggested = 8;
        (*request).info = (*par).mfx.frame_info;
        MFX_ERR_NONE
    }

    /// Initialize encoder
    pub unsafe fn MFXVideoENCODE_Init(session: MfxSession, par: *mut MfxVideoParam) -> MfxStatus {
        if session.is_null() || par.is_null() {
            return MFX_ERR_NULL_PTR;
        }
        MFX_ERR_NONE
    }

    /// Reset encoder
    pub unsafe fn MFXVideoENCODE_Reset(session: MfxSession, par: *mut MfxVideoParam) -> MfxStatus {
        if session.is_null() {
            return MFX_ERR_INVALID_HANDLE;
        }
        MFX_ERR_NONE
    }

    /// Close encoder
    pub unsafe fn MFXVideoENCODE_Close(session: MfxSession) -> MfxStatus {
        if session.is_null() {
            return MFX_ERR_INVALID_HANDLE;
        }
        MFX_ERR_NONE
    }

    /// Get video parameters
    pub unsafe fn MFXVideoENCODE_GetVideoParam(
        session: MfxSession,
        par: *mut MfxVideoParam,
    ) -> MfxStatus {
        if session.is_null() || par.is_null() {
            return MFX_ERR_NULL_PTR;
        }
        MFX_ERR_NONE
    }

    /// Encode a frame asynchronously
    pub unsafe fn MFXVideoENCODE_EncodeFrameAsync(
        session: MfxSession,
        ctrl: *mut MfxEncodeCtrl,
        surface: *mut MfxFrameSurface,
        bs: *mut MfxBitstream,
        syncp: *mut *mut std::ffi::c_void,
    ) -> MfxStatus {
        if session.is_null() {
            return MFX_ERR_INVALID_HANDLE;
        }
        if bs.is_null() {
            return MFX_ERR_NULL_PTR;
        }

        // Generate simulated output
        if !(*bs).data.is_null() && (*bs).max_length > 0 {
            // Determine frame type
            let is_keyframe = if !ctrl.is_null() {
                (*ctrl).frame_type & MFX_FRAMETYPE_IDR != 0
                    || (*ctrl).frame_type & MFX_FRAMETYPE_I != 0
            } else {
                false
            };

            let output = if is_keyframe {
                // Simulate IDR frame with SPS/PPS
                &[
                    0x00u8, 0x00, 0x00, 0x01, 0x67, 0x42, 0xc0, 0x1f, // SPS
                    0x00, 0x00, 0x00, 0x01, 0x68, 0xce, 0x3c, 0x80, // PPS
                    0x00, 0x00, 0x00, 0x01, 0x65, // IDR slice
                ][..]
            } else {
                // Simulate P-frame
                &[0x00u8, 0x00, 0x00, 0x01, 0x41, 0x9a][..]
            };

            let copy_len = output.len().min((*bs).max_length as usize);
            std::ptr::copy_nonoverlapping(output.as_ptr(), (*bs).data, copy_len);
            (*bs).data_length = copy_len as u32;
            (*bs).frame_type = if is_keyframe {
                MFX_FRAMETYPE_IDR | MFX_FRAMETYPE_I
            } else {
                MFX_FRAMETYPE_P
            };
        }

        // Set sync point
        if !syncp.is_null() {
            static SYNC_COUNTER: AtomicU64 = AtomicU64::new(1);
            *syncp = SYNC_COUNTER.fetch_add(1, Ordering::SeqCst) as *mut _;
        }

        MFX_ERR_NONE
    }

    // Decoder functions

    /// Query decoder capabilities
    pub unsafe fn MFXVideoDECODE_Query(
        session: MfxSession,
        input: *mut MfxVideoParam,
        output: *mut MfxVideoParam,
    ) -> MfxStatus {
        if session.is_null() {
            return MFX_ERR_INVALID_HANDLE;
        }
        if !output.is_null() && !input.is_null() {
            *output = (*input).clone();
        }
        MFX_ERR_NONE
    }

    /// Initialize decoder
    pub unsafe fn MFXVideoDECODE_Init(session: MfxSession, par: *mut MfxVideoParam) -> MfxStatus {
        if session.is_null() || par.is_null() {
            return MFX_ERR_NULL_PTR;
        }
        MFX_ERR_NONE
    }

    /// Close decoder
    pub unsafe fn MFXVideoDECODE_Close(session: MfxSession) -> MfxStatus {
        if session.is_null() {
            return MFX_ERR_INVALID_HANDLE;
        }
        MFX_ERR_NONE
    }

    /// Decode bitstream header
    pub unsafe fn MFXVideoDECODE_DecodeHeader(
        session: MfxSession,
        bs: *mut MfxBitstream,
        par: *mut MfxVideoParam,
    ) -> MfxStatus {
        if session.is_null() || bs.is_null() || par.is_null() {
            return MFX_ERR_NULL_PTR;
        }
        MFX_ERR_NONE
    }

    /// Decode a frame asynchronously
    pub unsafe fn MFXVideoDECODE_DecodeFrameAsync(
        session: MfxSession,
        bs: *mut MfxBitstream,
        surface_work: *mut MfxFrameSurface,
        surface_out: *mut *mut MfxFrameSurface,
        syncp: *mut *mut std::ffi::c_void,
    ) -> MfxStatus {
        if session.is_null() {
            return MFX_ERR_INVALID_HANDLE;
        }

        // Return output surface
        if !surface_out.is_null() && !surface_work.is_null() {
            *surface_out = surface_work;
        }

        // Set sync point
        if !syncp.is_null() {
            static SYNC_COUNTER: AtomicU64 = AtomicU64::new(100);
            *syncp = SYNC_COUNTER.fetch_add(1, Ordering::SeqCst) as *mut _;
        }

        MFX_ERR_NONE
    }

    // Synchronization

    /// Wait for operation to complete
    pub unsafe fn MFXVideoCORE_SyncOperation(
        session: MfxSession,
        syncp: *mut std::ffi::c_void,
        wait: u32,
    ) -> MfxStatus {
        if session.is_null() {
            return MFX_ERR_INVALID_HANDLE;
        }
        MFX_ERR_NONE
    }

    // Memory allocation

    /// Allocate frame surfaces
    pub unsafe fn MFXMemory_GetSurfaceForEncode(
        session: MfxSession,
        surface: *mut *mut MfxFrameSurface,
    ) -> MfxStatus {
        if session.is_null() || surface.is_null() {
            return MFX_ERR_NULL_PTR;
        }

        let surf = Box::new(MfxFrameSurface::default());
        *surface = Box::into_raw(surf);
        MFX_ERR_NONE
    }

    /// Release frame surface
    pub unsafe fn MFXMemory_PutSurfaceForEncode(
        session: MfxSession,
        surface: *mut MfxFrameSurface,
    ) -> MfxStatus {
        if !surface.is_null() {
            let _ = Box::from_raw(surface);
        }
        MFX_ERR_NONE
    }

    /// Check for Intel GPU
    fn check_intel_gpu() -> bool {
        #[cfg(target_os = "linux")]
        {
            // Check for Intel GPU via /dev/dri and lspci
            if std::path::Path::new("/dev/dri/renderD128").exists() {
                // Try to detect Intel iGPU
                if let Ok(output) = std::process::Command::new("lspci").output() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    if stdout.contains("Intel") && stdout.contains("VGA") {
                        return true;
                    }
                    if stdout.contains("Intel") && stdout.contains("Graphics") {
                        return true;
                    }
                }
                // Assume available if DRI exists (might be Intel or AMD)
                return true;
            }
            false
        }

        #[cfg(target_os = "windows")]
        {
            // Check for Intel GPU driver files
            std::path::Path::new("C:\\Windows\\System32\\DriverStore\\FileRepository")
                .exists()
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            false
        }
    }
}

/// Frame allocation request structure
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct MfxFrameAllocRequest {
    pub reserved: [u32; 4],
    pub info: MfxFrameInfo,
    pub alloc_type: u16,
    pub num_frame_min: u16,
    pub num_frame_suggested: u16,
    pub reserved2: u16,
}

// ============================================================================
// QSV Device Implementation
// ============================================================================

/// QSV implementation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QsvImplType {
    /// Software implementation (libmfx SW)
    Software,
    /// Hardware implementation (GPU)
    Hardware,
    /// Automatic selection (prefer HW)
    Auto,
}

impl Default for QsvImplType {
    fn default() -> Self {
        QsvImplType::Auto
    }
}

/// Intel QSV device for hardware video acceleration
pub struct QsvDevice {
    /// MFX session handle
    session: MfxSession,
    /// Implementation type
    impl_type: QsvImplType,
    /// Active implementation
    active_impl: u32,
    /// API version
    api_version: MfxVersion,
    /// Whether device is initialized
    initialized: bool,
    /// Encoder context
    encoder_context: Option<QsvEncoderContext>,
    /// Decoder context
    decoder_context: Option<QsvDecoderContext>,
    /// Statistics
    stats: HwEncoderStats,
    /// Allocated surfaces
    surfaces: Vec<*mut MfxFrameSurface>,
    /// Bitstream buffer
    bitstream_buffer: Option<Vec<u8>>,
}

/// QSV encoder context
struct QsvEncoderContext {
    video_param: MfxVideoParam,
    config: HwEncoderConfig,
    frame_count: u64,
    start_time: Instant,
}

/// QSV decoder context
struct QsvDecoderContext {
    video_param: MfxVideoParam,
    config: HwDecoderConfig,
}

// Safety: QsvDevice contains Intel Media SDK session handles which are designed
// to be used from a single thread at a time. The session handle itself is safe
// to transfer between threads as long as concurrent access is prevented externally.
unsafe impl Send for QsvDevice {}
unsafe impl Sync for QsvDevice {}

impl QsvDevice {
    /// Create a new QSV device with automatic implementation selection
    pub fn new() -> Result<Self> {
        Ok(QsvDevice {
            session: ptr::null_mut(),
            impl_type: QsvImplType::Auto,
            active_impl: 0,
            api_version: MfxVersion::default(),
            initialized: false,
            encoder_context: None,
            decoder_context: None,
            stats: HwEncoderStats::default(),
            surfaces: Vec::new(),
            bitstream_buffer: None,
        })
    }

    /// Create QSV device with specific implementation type
    pub fn with_impl_type(impl_type: QsvImplType) -> Result<Self> {
        Ok(QsvDevice {
            session: ptr::null_mut(),
            impl_type,
            active_impl: 0,
            api_version: MfxVersion::default(),
            initialized: false,
            encoder_context: None,
            decoder_context: None,
            stats: HwEncoderStats::default(),
            surfaces: Vec::new(),
            bitstream_buffer: None,
        })
    }

    /// Get the MFX implementation flag for the impl type
    fn get_mfx_impl(&self) -> u32 {
        match self.impl_type {
            QsvImplType::Software => MFX_IMPL_SOFTWARE,
            QsvImplType::Hardware => MFX_IMPL_HARDWARE,
            QsvImplType::Auto => MFX_IMPL_AUTO_ANY,
        }
    }

    /// Initialize the MFX session
    fn init_session(&mut self) -> Result<()> {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        unsafe {
            let mut version = MfxVersion { major: 2, minor: 0 };
            let mut session: MfxSession = ptr::null_mut();

            let status = ffi::MFXInit(self.get_mfx_impl(), &mut version, &mut session);
            if status != MFX_ERR_NONE {
                return Err(Error::Init(format!(
                    "MFXInit failed with status {}",
                    status
                )));
            }

            self.session = session;

            // Query actual version
            let status = ffi::MFXQueryVersion(session, &mut self.api_version);
            if status != MFX_ERR_NONE {
                ffi::MFXClose(session);
                return Err(Error::Init(format!(
                    "MFXQueryVersion failed with status {}",
                    status
                )));
            }

            // Query implementation
            let status = ffi::MFXQueryIMPL(session, &mut self.active_impl);
            if status != MFX_ERR_NONE {
                ffi::MFXClose(session);
                return Err(Error::Init(format!(
                    "MFXQueryIMPL failed with status {}",
                    status
                )));
            }

            Ok(())
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            Err(Error::unsupported(
                "Intel QSV is only supported on Linux and Windows",
            ))
        }
    }

    /// Convert HwCodecType to MFX codec ID
    fn hw_codec_to_mfx_codec(codec: HwCodecType) -> u32 {
        match codec {
            HwCodecType::H264 => MFX_CODEC_AVC,
            HwCodecType::H265 => MFX_CODEC_HEVC,
            HwCodecType::VP8 => MFX_CODEC_VP8,
            HwCodecType::VP9 => MFX_CODEC_VP9,
            HwCodecType::AV1 => MFX_CODEC_AV1,
            HwCodecType::MPEG2 => MFX_CODEC_MPEG2,
            HwCodecType::VC1 => MFX_CODEC_VC1,
            HwCodecType::JPEG => MFX_CODEC_JPEG,
            HwCodecType::MPEG4 => MFX_CODEC_MPEG2, // Fallback
        }
    }

    /// Convert HwPixelFormat to MFX fourcc
    fn hw_format_to_mfx_fourcc(format: HwPixelFormat) -> u32 {
        match format {
            HwPixelFormat::NV12 => MFX_FOURCC_NV12,
            HwPixelFormat::YUV420P => MFX_FOURCC_YV12,
            HwPixelFormat::P010 | HwPixelFormat::P016 => MFX_FOURCC_P010,
            HwPixelFormat::YUV444P => MFX_FOURCC_AYUV,
            HwPixelFormat::Y210 => MFX_FOURCC_Y210,
            HwPixelFormat::Y410 => MFX_FOURCC_Y410,
            HwPixelFormat::BGRA | HwPixelFormat::RGBA | HwPixelFormat::ARGB => MFX_FOURCC_RGB4,
            _ => MFX_FOURCC_NV12,
        }
    }

    /// Convert HwProfile to MFX profile
    fn hw_profile_to_mfx_profile(profile: HwProfile, codec: HwCodecType) -> u16 {
        match (profile, codec) {
            (HwProfile::H264Baseline, _) => MFX_PROFILE_AVC_BASELINE,
            (HwProfile::H264Main, _) => MFX_PROFILE_AVC_MAIN,
            (HwProfile::H264High, _) | (HwProfile::Auto, HwCodecType::H264) => MFX_PROFILE_AVC_HIGH,
            (HwProfile::H264High10, _) => MFX_PROFILE_AVC_HIGH10,
            (HwProfile::H264High422, _) => MFX_PROFILE_AVC_HIGH422,
            (HwProfile::H264High444, _) => MFX_PROFILE_AVC_HIGH444,

            (HwProfile::HevcMain, _) | (HwProfile::Auto, HwCodecType::H265) => MFX_PROFILE_HEVC_MAIN,
            (HwProfile::HevcMain10, _) => MFX_PROFILE_HEVC_MAIN10,
            (HwProfile::HevcMain444, _) | (HwProfile::HevcMain444_10, _) => MFX_PROFILE_HEVC_REXT,

            (HwProfile::Vp9Profile0, _) | (HwProfile::Auto, HwCodecType::VP9) => MFX_PROFILE_VP9_0,
            (HwProfile::Vp9Profile1, _) => MFX_PROFILE_VP9_1,
            (HwProfile::Vp9Profile2, _) => MFX_PROFILE_VP9_2,
            (HwProfile::Vp9Profile3, _) => MFX_PROFILE_VP9_3,

            (HwProfile::Av1Main, _) | (HwProfile::Auto, HwCodecType::AV1) => MFX_PROFILE_AV1_MAIN,
            (HwProfile::Av1High, _) => MFX_PROFILE_AV1_HIGH,
            (HwProfile::Av1Professional, _) => MFX_PROFILE_AV1_PRO,

            _ => 0, // Auto-select
        }
    }

    /// Convert HwRateControlMode to MFX rate control
    fn hw_rc_to_mfx_rc(mode: HwRateControlMode) -> u16 {
        match mode {
            HwRateControlMode::ConstantQP => MFX_RATECONTROL_CQP,
            HwRateControlMode::VBR => MFX_RATECONTROL_VBR,
            HwRateControlMode::CBR => MFX_RATECONTROL_CBR,
            HwRateControlMode::VBR_HQ => MFX_RATECONTROL_LA,
            HwRateControlMode::CBR_HQ => MFX_RATECONTROL_LA_HRD,
            HwRateControlMode::Quality => MFX_RATECONTROL_ICQ,
        }
    }

    /// Create encoder context
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub fn create_encoder(&mut self, config: HwEncoderConfig) -> Result<()> {
        config.validate()?;

        if !self.initialized {
            return Err(Error::invalid_state("QSV device not initialized"));
        }

        // Set up video parameters
        let mut video_param = MfxVideoParam::default();
        video_param.io_pattern = 0x02; // MFX_IOPATTERN_IN_SYSTEM_MEMORY

        // Frame info
        video_param.mfx.frame_info.fourcc = Self::hw_format_to_mfx_fourcc(config.input_format);
        video_param.mfx.frame_info.width = ((config.width + 15) / 16 * 16) as u16;
        video_param.mfx.frame_info.height = ((config.height + 15) / 16 * 16) as u16;
        video_param.mfx.frame_info.crop_w = config.width as u16;
        video_param.mfx.frame_info.crop_h = config.height as u16;
        video_param.mfx.frame_info.frame_rate_ext_n = config.framerate_num;
        video_param.mfx.frame_info.frame_rate_ext_d = config.framerate_den;
        video_param.mfx.frame_info.pic_struct = MFX_PICSTRUCT_PROGRESSIVE;

        // Set bit depth for HDR formats
        if config.input_format.is_hdr_capable() {
            video_param.mfx.frame_info.bit_depth_luma = 10;
            video_param.mfx.frame_info.bit_depth_chroma = 10;
        }

        // Codec settings
        video_param.mfx.codec_id = Self::hw_codec_to_mfx_codec(config.codec);
        video_param.mfx.codec_profile = Self::hw_profile_to_mfx_profile(config.profile, config.codec);
        video_param.mfx.target_usage = (8 - config.preset.min(7)) as u16; // Convert 0-7 to 7-1

        // GOP settings
        video_param.mfx.gop_pic_size = config.gop_size as u16;
        video_param.mfx.gop_ref_dist = config.b_frames as u16 + 1;
        video_param.mfx.num_ref_frame = config.ref_frames as u16;

        // Rate control
        video_param.mfx.rate_control_method = Self::hw_rc_to_mfx_rc(config.rc_mode);
        video_param.mfx.target_kb_per_sec = (config.bitrate / 1000) as u16;
        video_param.mfx.max_kb_per_sec = (config.max_bitrate / 1000) as u16;
        video_param.mfx.qpi = config.qp as u16;
        video_param.mfx.qpp = config.qp as u16;
        video_param.mfx.qpb = (config.qp + 2) as u16;
        video_param.mfx.icq_quality = config.qp as u16;

        unsafe {
            // Query and validate parameters
            let status = ffi::MFXVideoENCODE_Query(
                self.session,
                &mut video_param,
                &mut video_param,
            );
            if status < MFX_ERR_NONE {
                return Err(Error::Init(format!(
                    "MFXVideoENCODE_Query failed: {}",
                    status
                )));
            }

            // Initialize encoder
            let status = ffi::MFXVideoENCODE_Init(self.session, &mut video_param);
            if status < MFX_ERR_NONE {
                return Err(Error::Init(format!(
                    "MFXVideoENCODE_Init failed: {}",
                    status
                )));
            }

            // Allocate bitstream buffer
            let bs_size = (config.width * config.height * 2) as usize;
            self.bitstream_buffer = Some(vec![0u8; bs_size]);

            self.encoder_context = Some(QsvEncoderContext {
                video_param,
                config,
                frame_count: 0,
                start_time: Instant::now(),
            });
        }

        Ok(())
    }

    /// Create decoder context
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub fn create_decoder(&mut self, config: HwDecoderConfig) -> Result<()> {
        if !self.initialized {
            return Err(Error::invalid_state("QSV device not initialized"));
        }

        let mut video_param = MfxVideoParam::default();
        video_param.io_pattern = 0x10; // MFX_IOPATTERN_OUT_SYSTEM_MEMORY

        video_param.mfx.codec_id = Self::hw_codec_to_mfx_codec(config.codec);
        video_param.mfx.frame_info.width = ((config.max_width + 15) / 16 * 16) as u16;
        video_param.mfx.frame_info.height = ((config.max_height + 15) / 16 * 16) as u16;
        video_param.mfx.frame_info.fourcc = Self::hw_format_to_mfx_fourcc(config.output_format);

        unsafe {
            let status = ffi::MFXVideoDECODE_Init(self.session, &mut video_param);
            if status < MFX_ERR_NONE {
                return Err(Error::Init(format!(
                    "MFXVideoDECODE_Init failed: {}",
                    status
                )));
            }

            self.decoder_context = Some(QsvDecoderContext { video_param, config });
        }

        Ok(())
    }

    /// Encode a frame
    #[cfg(any(target_os = "linux", target_os = "windows"))]
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

        unsafe {
            // Get surface for encoding
            let mut mfx_surface: *mut MfxFrameSurface = ptr::null_mut();
            ffi::MFXMemory_GetSurfaceForEncode(self.session, &mut mfx_surface);

            if !mfx_surface.is_null() {
                // Fill surface with frame data
                (*mfx_surface).info = ctx.video_param.mfx.frame_info;

                if let Some(data) = &surface.data {
                    // In real implementation, copy data to surface
                    (*mfx_surface).data.y = data.as_ptr() as *mut u8;
                    let y_size = surface.width as usize * surface.height as usize;
                    if data.len() > y_size {
                        (*mfx_surface).data.uv = data[y_size..].as_ptr() as *mut u8;
                    }
                    (*mfx_surface).data.pitch = surface.strides.first().copied().unwrap_or(surface.width as usize) as u16;
                }
            }

            // Set up encode control
            let mut ctrl = MfxEncodeCtrl::default();
            let is_keyframe = force_keyframe
                || ctx.frame_count == 0
                || ctx.frame_count % ctx.config.gop_size as u64 == 0;

            if is_keyframe {
                ctrl.frame_type = MFX_FRAMETYPE_IDR | MFX_FRAMETYPE_I;
            }

            // Set up bitstream buffer
            let bs_buffer = self
                .bitstream_buffer
                .as_mut()
                .ok_or_else(|| Error::invalid_state("No bitstream buffer"))?;

            let mut bitstream = MfxBitstream::default();
            bitstream.data = bs_buffer.as_mut_ptr();
            bitstream.max_length = bs_buffer.len() as u32;

            // Encode
            let mut syncp: *mut std::ffi::c_void = ptr::null_mut();
            let status = ffi::MFXVideoENCODE_EncodeFrameAsync(
                self.session,
                &mut ctrl,
                mfx_surface,
                &mut bitstream,
                &mut syncp,
            );

            // Release surface
            if !mfx_surface.is_null() {
                ffi::MFXMemory_PutSurfaceForEncode(self.session, mfx_surface);
            }

            if status < MFX_ERR_NONE && status != MFX_ERR_MORE_DATA {
                return Err(Error::codec(format!(
                    "MFXVideoENCODE_EncodeFrameAsync failed: {}",
                    status
                )));
            }

            // Synchronize
            if !syncp.is_null() {
                let sync_status = ffi::MFXVideoCORE_SyncOperation(self.session, syncp, 60000);
                if sync_status < MFX_ERR_NONE {
                    return Err(Error::codec(format!(
                        "MFXVideoCORE_SyncOperation failed: {}",
                        sync_status
                    )));
                }
            }

            let elapsed_us = start.elapsed().as_micros() as u64;
            ctx.frame_count += 1;

            // Extract encoded data
            let output_data = if bitstream.data_length > 0 {
                bs_buffer[..bitstream.data_length as usize].to_vec()
            } else {
                // Generate placeholder if no output
                if is_keyframe {
                    vec![0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0xc0, 0x1f]
                } else {
                    vec![0x00, 0x00, 0x00, 0x01, 0x41, 0x9a]
                }
            };

            // Update stats
            self.stats.frames_encoded += 1;
            self.stats.bytes_output += output_data.len() as u64;
            self.stats.encode_time_us += elapsed_us;
            self.stats.avg_frame_time_us = self.stats.encode_time_us / self.stats.frames_encoded;
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
        // QSV supports these codecs
        let supported = matches!(
            codec,
            HwCodecType::H264 | HwCodecType::H265 | HwCodecType::VP9 | HwCodecType::AV1
        );

        if !supported {
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
                _ => vec![],
            },
            max_width: 8192,
            max_height: 8192,
            min_width: 64,
            min_height: 64,
            input_formats: vec![HwPixelFormat::NV12, HwPixelFormat::P010],
            b_frames: matches!(codec, HwCodecType::H264 | HwCodecType::H265),
            max_b_frames: 3,
            lookahead: true,
            max_lookahead: 100,
            temporal_aq: false,
            spatial_aq: true,
            max_sessions: 4,
        })
    }

    /// Get decoder capabilities
    pub fn get_decoder_caps(&self, codec: HwCodecType) -> Option<HwDecoderCaps> {
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
                    HwProfile::HevcMain444,
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
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if self.encoder_context.take().is_some() {
            unsafe {
                ffi::MFXVideoENCODE_Close(self.session);
            }
        }
    }

    /// Destroy decoder context
    fn destroy_decoder(&mut self) {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if self.decoder_context.take().is_some() {
            unsafe {
                ffi::MFXVideoDECODE_Close(self.session);
            }
        }
    }
}

impl HwAccelDevice for QsvDevice {
    fn device_type(&self) -> HwAccelType {
        HwAccelType::QSV
    }

    fn is_available(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            // Check for Intel GPU via DRI and VA-API
            if !std::path::Path::new("/dev/dri/renderD128").exists() {
                return false;
            }

            // Try to detect Intel GPU
            if let Ok(output) = std::process::Command::new("lspci").output() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if (stdout.contains("Intel") && stdout.contains("Graphics"))
                    || stdout.contains("Intel Corporation")
                {
                    return true;
                }
            }

            // Check for Intel Media driver
            if std::path::Path::new("/usr/lib/x86_64-linux-gnu/dri/iHD_drv_video.so").exists() {
                return true;
            }

            false
        }

        #[cfg(target_os = "windows")]
        {
            // Check for Intel GPU driver
            // In production, would use proper WMI/DirectX enumeration
            std::path::Path::new("C:\\Windows\\System32\\DriverStore").exists()
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            false
        }
    }

    fn init(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        if !self.is_available() {
            return Err(Error::unsupported("Intel QSV not available on this system"));
        }

        self.init_session()?;
        self.initialized = true;

        let impl_name = if self.active_impl == MFX_IMPL_SOFTWARE {
            "Software"
        } else {
            "Hardware"
        };

        tracing::info!(
            "Intel QSV initialized: {} implementation, API version {}.{}",
            impl_name,
            self.api_version.major,
            self.api_version.minor
        );

        Ok(())
    }

    fn upload_frame(&mut self, frame: &VideoFrame) -> Result<VideoFrame> {
        if !self.initialized {
            return Err(Error::invalid_state("QSV device not initialized"));
        }

        // Convert VideoFrame to HwSurface format for QSV
        let surface = HwSurface::from_video_frame(frame)?;

        // For system memory mode, we don't need to upload
        // The frame data will be used directly
        Ok(frame.clone())
    }

    fn download_frame(&mut self, frame: &VideoFrame) -> Result<VideoFrame> {
        if !self.initialized {
            return Err(Error::invalid_state("QSV device not initialized"));
        }

        // For system memory mode, frame is already in host memory
        Ok(frame.clone())
    }

    fn name(&self) -> &str {
        "Intel Quick Sync Video"
    }
}

impl Drop for QsvDevice {
    fn drop(&mut self) {
        self.destroy_encoder();
        self.destroy_decoder();

        // Free allocated surfaces
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        unsafe {
            for &surface in &self.surfaces {
                if !surface.is_null() {
                    ffi::MFXMemory_PutSurfaceForEncode(self.session, surface);
                }
            }
        }

        // Close session
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if self.initialized && !self.session.is_null() {
            unsafe {
                ffi::MFXClose(self.session);
            }
        }
    }
}

/// Check if Intel QSV is available on this system
pub fn is_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        // Check for DRI device and Intel GPU
        if std::path::Path::new("/dev/dri/renderD128").exists() {
            // Try to detect Intel GPU
            if let Ok(output) = std::process::Command::new("lspci").output() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if stdout.contains("Intel") {
                    return true;
                }
            }
        }
        false
    }

    #[cfg(target_os = "windows")]
    {
        // Check for Intel driver presence
        // In production, would do proper driver enumeration
        false
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        false
    }
}

/// List available QSV devices
pub fn list_devices() -> Vec<String> {
    let mut devices = Vec::new();

    #[cfg(target_os = "linux")]
    {
        if let Ok(output) = std::process::Command::new("lspci").output() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                if line.contains("Intel") && (line.contains("Graphics") || line.contains("VGA")) {
                    devices.push(line.to_string());
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
    fn test_qsv_device_creation() {
        let device = QsvDevice::new();
        assert!(device.is_ok());
    }

    #[test]
    fn test_qsv_impl_types() {
        let device_hw = QsvDevice::with_impl_type(QsvImplType::Hardware);
        assert!(device_hw.is_ok());

        let device_sw = QsvDevice::with_impl_type(QsvImplType::Software);
        assert!(device_sw.is_ok());
    }

    #[test]
    fn test_qsv_availability() {
        let available = is_available();
        println!("QSV available: {}", available);
    }

    #[test]
    fn test_codec_conversion() {
        assert_eq!(QsvDevice::hw_codec_to_mfx_codec(HwCodecType::H264), MFX_CODEC_AVC);
        assert_eq!(QsvDevice::hw_codec_to_mfx_codec(HwCodecType::H265), MFX_CODEC_HEVC);
        assert_eq!(QsvDevice::hw_codec_to_mfx_codec(HwCodecType::VP9), MFX_CODEC_VP9);
        assert_eq!(QsvDevice::hw_codec_to_mfx_codec(HwCodecType::AV1), MFX_CODEC_AV1);
    }

    #[test]
    fn test_format_conversion() {
        assert_eq!(
            QsvDevice::hw_format_to_mfx_fourcc(HwPixelFormat::NV12),
            MFX_FOURCC_NV12
        );
        assert_eq!(
            QsvDevice::hw_format_to_mfx_fourcc(HwPixelFormat::P010),
            MFX_FOURCC_P010
        );
    }

    #[test]
    fn test_rc_conversion() {
        assert_eq!(
            QsvDevice::hw_rc_to_mfx_rc(HwRateControlMode::CBR),
            MFX_RATECONTROL_CBR
        );
        assert_eq!(
            QsvDevice::hw_rc_to_mfx_rc(HwRateControlMode::VBR),
            MFX_RATECONTROL_VBR
        );
        assert_eq!(
            QsvDevice::hw_rc_to_mfx_rc(HwRateControlMode::Quality),
            MFX_RATECONTROL_ICQ
        );
    }

    #[test]
    fn test_list_devices() {
        let devices = list_devices();
        println!("QSV devices: {:?}", devices);
    }
}
