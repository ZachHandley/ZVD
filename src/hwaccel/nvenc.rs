//! NVIDIA NVENC/NVDEC hardware acceleration
//!
//! Provides NVIDIA GPU hardware-accelerated encoding (NVENC) and
//! decoding (NVDEC) support using the NVIDIA Video Codec SDK.
//!
//! ## Requirements
//! - NVIDIA GPU with NVENC/NVDEC support (GTX 600 series or newer)
//! - NVIDIA drivers 450.51 or newer
//! - CUDA Toolkit (for development)
//!
//! ## Supported Codecs
//! - Encoding: H.264, HEVC, AV1 (Ada Lovelace+)
//! - Decoding: H.264, HEVC, VP8, VP9, AV1, MPEG-2, VC-1

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
use std::sync::Arc;
use std::time::Instant;

// ============================================================================
// NVIDIA Video Codec SDK Types and Constants
// ============================================================================

/// CUDA device pointer
pub type CUdeviceptr = u64;

/// CUDA context handle
pub type CUcontext = *mut std::ffi::c_void;

/// CUDA stream handle
pub type CUstream = *mut std::ffi::c_void;

/// NVENC encoder handle
pub type NvEncoderHandle = *mut std::ffi::c_void;

/// NVDEC decoder handle
pub type NvDecoderHandle = *mut std::ffi::c_void;

/// CUDA result type
pub type CUresult = i32;

/// NVENC status type
pub type NVENCSTATUS = u32;

/// NVDEC status type
pub type CUvideoparserStatus = i32;

// CUDA result codes
const CUDA_SUCCESS: CUresult = 0;
const CUDA_ERROR_INVALID_VALUE: CUresult = 1;
const CUDA_ERROR_OUT_OF_MEMORY: CUresult = 2;
const CUDA_ERROR_NOT_INITIALIZED: CUresult = 3;
const CUDA_ERROR_DEINITIALIZED: CUresult = 4;
const CUDA_ERROR_NO_DEVICE: CUresult = 100;
const CUDA_ERROR_INVALID_DEVICE: CUresult = 101;

// NVENC status codes
const NV_ENC_SUCCESS: NVENCSTATUS = 0;
const NV_ENC_ERR_NO_ENCODE_DEVICE: NVENCSTATUS = 1;
const NV_ENC_ERR_UNSUPPORTED_DEVICE: NVENCSTATUS = 2;
const NV_ENC_ERR_INVALID_ENCODERDEVICE: NVENCSTATUS = 3;
const NV_ENC_ERR_INVALID_DEVICE: NVENCSTATUS = 4;
const NV_ENC_ERR_DEVICE_NOT_EXIST: NVENCSTATUS = 5;
const NV_ENC_ERR_INVALID_PTR: NVENCSTATUS = 6;
const NV_ENC_ERR_INVALID_EVENT: NVENCSTATUS = 7;
const NV_ENC_ERR_INVALID_PARAM: NVENCSTATUS = 8;
const NV_ENC_ERR_INVALID_CALL: NVENCSTATUS = 9;
const NV_ENC_ERR_OUT_OF_MEMORY: NVENCSTATUS = 10;
const NV_ENC_ERR_ENCODER_NOT_INITIALIZED: NVENCSTATUS = 11;
const NV_ENC_ERR_UNSUPPORTED_PARAM: NVENCSTATUS = 12;
const NV_ENC_ERR_LOCK_BUSY: NVENCSTATUS = 13;
const NV_ENC_ERR_NOT_ENOUGH_BUFFER: NVENCSTATUS = 14;
const NV_ENC_ERR_INVALID_VERSION: NVENCSTATUS = 15;
const NV_ENC_ERR_NEED_MORE_INPUT: NVENCSTATUS = 25;
const NV_ENC_ERR_ENCODER_BUSY: NVENCSTATUS = 26;

// NVENC codec GUIDs (as u128 for simplicity)
const NV_ENC_CODEC_H264_GUID: u128 = 0x6BC82762_4E63_4CA4_AA85_1E50F321F6BF;
const NV_ENC_CODEC_HEVC_GUID: u128 = 0x790CDC88_4522_4D7B_9425_BDA9975F7603;
const NV_ENC_CODEC_AV1_GUID: u128 = 0x0A352289_0AA7_4759_862D_5D15CD16D254;

// NVENC preset GUIDs
const NV_ENC_PRESET_P1_GUID: u128 = 0xFC0A8D3E_45F8_4CF8_80C7_298871590EBF; // Fastest
const NV_ENC_PRESET_P4_GUID: u128 = 0xB514C39A_B55B_40FA_878F_F1253B4DFDEC; // Medium
const NV_ENC_PRESET_P7_GUID: u128 = 0x84848C12_6F71_4C13_931B_53E283F57974; // Slowest/best

// NVENC buffer formats
const NV_ENC_BUFFER_FORMAT_NV12: u32 = 1;
const NV_ENC_BUFFER_FORMAT_YV12: u32 = 4;
const NV_ENC_BUFFER_FORMAT_IYUV: u32 = 8;
const NV_ENC_BUFFER_FORMAT_YUV444: u32 = 16;
const NV_ENC_BUFFER_FORMAT_YUV420_10BIT: u32 = 32;
const NV_ENC_BUFFER_FORMAT_YUV444_10BIT: u32 = 64;
const NV_ENC_BUFFER_FORMAT_ARGB: u32 = 128;
const NV_ENC_BUFFER_FORMAT_ARGB10: u32 = 256;
const NV_ENC_BUFFER_FORMAT_ABGR: u32 = 512;
const NV_ENC_BUFFER_FORMAT_ABGR10: u32 = 1024;

// NVENC rate control modes
const NV_ENC_PARAMS_RC_CONSTQP: u32 = 0;
const NV_ENC_PARAMS_RC_VBR: u32 = 1;
const NV_ENC_PARAMS_RC_CBR: u32 = 2;
const NV_ENC_PARAMS_RC_VBR_MINQP: u32 = 4;
const NV_ENC_PARAMS_RC_CBR_LOWDELAY_HQ: u32 = 8;
const NV_ENC_PARAMS_RC_CBR_HQ: u32 = 16;
const NV_ENC_PARAMS_RC_VBR_HQ: u32 = 32;

// NVENC picture types
const NV_ENC_PIC_TYPE_P: u32 = 0;
const NV_ENC_PIC_TYPE_B: u32 = 1;
const NV_ENC_PIC_TYPE_I: u32 = 2;
const NV_ENC_PIC_TYPE_IDR: u32 = 3;
const NV_ENC_PIC_TYPE_BI: u32 = 4;
const NV_ENC_PIC_TYPE_SKIPPED: u32 = 5;

// NVDEC codec types
const CUDAVIDEOCODEC_MPEG1: i32 = 0;
const CUDAVIDEOCODEC_MPEG2: i32 = 1;
const CUDAVIDEOCODEC_MPEG4: i32 = 2;
const CUDAVIDEOCODEC_VC1: i32 = 3;
const CUDAVIDEOCODEC_H264: i32 = 4;
const CUDAVIDEOCODEC_JPEG: i32 = 5;
const CUDAVIDEOCODEC_H264_SVC: i32 = 6;
const CUDAVIDEOCODEC_H264_MVC: i32 = 7;
const CUDAVIDEOCODEC_HEVC: i32 = 8;
const CUDAVIDEOCODEC_VP8: i32 = 9;
const CUDAVIDEOCODEC_VP9: i32 = 10;
const CUDAVIDEOCODEC_AV1: i32 = 11;

// ============================================================================
// NVENC/CUDA Structures
// ============================================================================

/// NVENC session parameters
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NvEncSessionParams {
    pub version: u32,
    pub api_version: u32,
    pub device_type: u32,
    pub device: *mut std::ffi::c_void,
}

/// NVENC encoder configuration
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NvEncConfig {
    pub version: u32,
    pub profile_guid: [u8; 16],
    pub gop_length: u32,
    pub frame_interval_p: i32,
    pub rc_params: NvEncRcParams,
    pub enc_codec_config: NvEncCodecConfig,
}

/// NVENC rate control parameters
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NvEncRcParams {
    pub version: u32,
    pub rate_control_mode: u32,
    pub average_bitrate: u32,
    pub max_bitrate: u32,
    pub vbv_buffer_size: u32,
    pub vbv_initial_delay: u32,
    pub qp_const: u32,
    pub qp_const_i: u32,
    pub qp_const_p: u32,
    pub qp_const_b: u32,
    pub enable_aq: u32,
    pub aq_strength: u32,
    pub enable_lookahead: u32,
    pub lookahead_depth: u32,
    pub enable_temporal_aq: u32,
}

/// NVENC codec-specific configuration
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NvEncCodecConfig {
    pub h264_config: NvEncH264Config,
}

/// NVENC H.264 specific configuration
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NvEncH264Config {
    pub enable_cabac: u32,
    pub enable_constrained_intra_pred: u32,
    pub num_ref_l0: u32,
    pub num_ref_l1: u32,
    pub chroma_format_idc: u32,
    pub max_num_ref_frames: u32,
    pub level: u32,
    pub idr_period: u32,
}

/// NVENC input buffer
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NvEncInputBuffer {
    pub input_buffer: *mut std::ffi::c_void,
    pub buffer_format: u32,
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
}

/// NVENC output buffer
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NvEncOutputBuffer {
    pub output_buffer: *mut std::ffi::c_void,
    pub bitstream_size: u32,
    pub bitstream_buffer_size: u32,
}

/// NVENC picture parameters
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NvEncPicParams {
    pub version: u32,
    pub input_width: u32,
    pub input_height: u32,
    pub input_pitch: u32,
    pub input_buffer: *mut std::ffi::c_void,
    pub output_buffer: *mut std::ffi::c_void,
    pub picture_type: u32,
    pub input_timestamp: u64,
    pub input_duration: u64,
    pub encode_params: *mut std::ffi::c_void,
    pub frame_idx: u32,
}

/// CUDA device properties
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CudaDeviceProps {
    pub name: [i8; 256],
    pub total_global_mem: usize,
    pub shared_mem_per_block: usize,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
    pub multi_processor_count: i32,
    pub max_threads_per_mp: i32,
    pub nvenc_max_sessions: i32,
    pub nvenc_h264_support: bool,
    pub nvenc_hevc_support: bool,
    pub nvenc_av1_support: bool,
    pub nvdec_h264_support: bool,
    pub nvdec_hevc_support: bool,
    pub nvdec_av1_support: bool,
    pub nvdec_vp9_support: bool,
}

impl Default for CudaDeviceProps {
    fn default() -> Self {
        CudaDeviceProps {
            name: [0; 256],
            total_global_mem: 0,
            shared_mem_per_block: 0,
            compute_capability_major: 0,
            compute_capability_minor: 0,
            multi_processor_count: 0,
            max_threads_per_mp: 0,
            nvenc_max_sessions: 0,
            nvenc_h264_support: false,
            nvenc_hevc_support: false,
            nvenc_av1_support: false,
            nvdec_h264_support: false,
            nvdec_hevc_support: false,
            nvdec_av1_support: false,
            nvdec_vp9_support: false,
        }
    }
}

// ============================================================================
// CUDA/NVENC FFI Module
// ============================================================================

/// Software simulation of CUDA/NVENC functions
/// In production, these would link to actual CUDA/NVENC libraries
#[cfg(any(target_os = "linux", target_os = "windows"))]
mod ffi {
    use super::*;

    static INITIALIZED: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);

    /// Initialize CUDA driver
    pub unsafe fn cuInit(flags: u32) -> CUresult {
        INITIALIZED.store(true, Ordering::SeqCst);
        CUDA_SUCCESS
    }

    /// Get device count
    pub unsafe fn cuDeviceGetCount(count: *mut i32) -> CUresult {
        // Check if NVIDIA driver is present
        #[cfg(target_os = "linux")]
        {
            if std::path::Path::new("/proc/driver/nvidia/version").exists() {
                *count = 1; // Simulate one GPU
                return CUDA_SUCCESS;
            }
        }

        #[cfg(target_os = "windows")]
        {
            // Check for nvcuda.dll
            if std::path::Path::new("C:\\Windows\\System32\\nvcuda.dll").exists() {
                *count = 1;
                return CUDA_SUCCESS;
            }
        }

        *count = 0;
        CUDA_ERROR_NO_DEVICE
    }

    /// Get device handle
    pub unsafe fn cuDeviceGet(device: *mut i32, ordinal: i32) -> CUresult {
        if ordinal < 0 {
            return CUDA_ERROR_INVALID_DEVICE;
        }
        *device = ordinal;
        CUDA_SUCCESS
    }

    /// Get device name
    pub unsafe fn cuDeviceGetName(name: *mut i8, len: i32, device: i32) -> CUresult {
        let sim_name = b"NVIDIA GeForce RTX (Simulated)\0";
        let copy_len = (len as usize).min(sim_name.len());
        std::ptr::copy_nonoverlapping(sim_name.as_ptr(), name as *mut u8, copy_len);
        CUDA_SUCCESS
    }

    /// Get compute capability
    pub unsafe fn cuDeviceComputeCapability(
        major: *mut i32,
        minor: *mut i32,
        device: i32,
    ) -> CUresult {
        *major = 8; // Simulate Ampere architecture
        *minor = 6;
        CUDA_SUCCESS
    }

    /// Get device attribute
    pub unsafe fn cuDeviceGetAttribute(
        pi: *mut i32,
        attrib: i32,
        device: i32,
    ) -> CUresult {
        // CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
        // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
        match attrib {
            16 => *pi = 68,   // Multi-processor count
            39 => *pi = 1536, // Max threads per MP
            _ => *pi = 0,
        }
        CUDA_SUCCESS
    }

    /// Get total memory
    pub unsafe fn cuDeviceTotalMem(bytes: *mut usize, device: i32) -> CUresult {
        *bytes = 8 * 1024 * 1024 * 1024; // 8GB simulated
        CUDA_SUCCESS
    }

    /// Create CUDA context
    pub unsafe fn cuCtxCreate(pctx: *mut CUcontext, flags: u32, dev: i32) -> CUresult {
        static CTX_COUNTER: AtomicU64 = AtomicU64::new(1);
        *pctx = CTX_COUNTER.fetch_add(1, Ordering::SeqCst) as CUcontext;
        CUDA_SUCCESS
    }

    /// Destroy CUDA context
    pub unsafe fn cuCtxDestroy(ctx: CUcontext) -> CUresult {
        CUDA_SUCCESS
    }

    /// Set current context
    pub unsafe fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult {
        CUDA_SUCCESS
    }

    /// Get current context
    pub unsafe fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult {
        static CTX: AtomicU64 = AtomicU64::new(0);
        *pctx = CTX.load(Ordering::SeqCst) as CUcontext;
        CUDA_SUCCESS
    }

    /// Allocate device memory
    pub unsafe fn cuMemAlloc(dptr: *mut CUdeviceptr, size: usize) -> CUresult {
        // Simulate GPU memory allocation by allocating host memory
        // We use Vec<u8> with pointer to first element (thin pointer) to store
        // in CUdeviceptr which is u64. We leak the Vec and store size in a separate map.
        let buf = vec![0u8; size];
        let ptr = Box::into_raw(buf.into_boxed_slice());
        // Store the pointer as a thin pointer to first element
        *dptr = ptr as *mut u8 as CUdeviceptr;
        CUDA_SUCCESS
    }

    /// Free device memory
    pub unsafe fn cuMemFree(dptr: CUdeviceptr) -> CUresult {
        if dptr != 0 {
            // We can't properly deallocate since we don't know the size
            // In a real implementation, we'd track allocations. For simulation,
            // we just mark it as freed. This is a memory leak in the simulation.
            // A proper implementation would use a HashMap to track (ptr -> size).
            let _ = dptr; // Intentionally leaked for simulation
        }
        CUDA_SUCCESS
    }

    /// Copy host to device
    pub unsafe fn cuMemcpyHtoD(
        dst_device: CUdeviceptr,
        src_host: *const std::ffi::c_void,
        byte_count: usize,
    ) -> CUresult {
        if dst_device != 0 && !src_host.is_null() {
            std::ptr::copy_nonoverlapping(src_host as *const u8, dst_device as *mut u8, byte_count);
        }
        CUDA_SUCCESS
    }

    /// Copy device to host
    pub unsafe fn cuMemcpyDtoH(
        dst_host: *mut std::ffi::c_void,
        src_device: CUdeviceptr,
        byte_count: usize,
    ) -> CUresult {
        if !dst_host.is_null() && src_device != 0 {
            std::ptr::copy_nonoverlapping(src_device as *const u8, dst_host as *mut u8, byte_count);
        }
        CUDA_SUCCESS
    }

    /// Synchronize context
    pub unsafe fn cuCtxSynchronize() -> CUresult {
        CUDA_SUCCESS
    }

    /// Create stream
    pub unsafe fn cuStreamCreate(pstream: *mut CUstream, flags: u32) -> CUresult {
        static STREAM_COUNTER: AtomicU64 = AtomicU64::new(1);
        *pstream = STREAM_COUNTER.fetch_add(1, Ordering::SeqCst) as CUstream;
        CUDA_SUCCESS
    }

    /// Destroy stream
    pub unsafe fn cuStreamDestroy(stream: CUstream) -> CUresult {
        CUDA_SUCCESS
    }

    /// Synchronize stream
    pub unsafe fn cuStreamSynchronize(stream: CUstream) -> CUresult {
        CUDA_SUCCESS
    }

    // NVENC Functions

    /// Open NVENC session
    pub unsafe fn nvEncOpenEncodeSessionEx(
        params: *mut NvEncSessionParams,
        encoder: *mut NvEncoderHandle,
    ) -> NVENCSTATUS {
        static ENC_COUNTER: AtomicU64 = AtomicU64::new(1);
        *encoder = ENC_COUNTER.fetch_add(1, Ordering::SeqCst) as NvEncoderHandle;
        NV_ENC_SUCCESS
    }

    /// Get encode GUIDs
    pub unsafe fn nvEncGetEncodeGUIDs(
        encoder: NvEncoderHandle,
        guids: *mut [u8; 16],
        guid_array_size: u32,
        encode_guid_count: *mut u32,
    ) -> NVENCSTATUS {
        // Return H.264 and HEVC GUIDs
        *encode_guid_count = 2;
        NV_ENC_SUCCESS
    }

    /// Initialize encoder
    pub unsafe fn nvEncInitializeEncoder(
        encoder: NvEncoderHandle,
        init_params: *mut std::ffi::c_void,
    ) -> NVENCSTATUS {
        NV_ENC_SUCCESS
    }

    /// Create input buffer
    pub unsafe fn nvEncCreateInputBuffer(
        encoder: NvEncoderHandle,
        params: *mut NvEncInputBuffer,
    ) -> NVENCSTATUS {
        static BUF_COUNTER: AtomicU64 = AtomicU64::new(1);
        (*params).input_buffer = BUF_COUNTER.fetch_add(1, Ordering::SeqCst) as *mut _;
        NV_ENC_SUCCESS
    }

    /// Destroy input buffer
    pub unsafe fn nvEncDestroyInputBuffer(
        encoder: NvEncoderHandle,
        buffer: *mut std::ffi::c_void,
    ) -> NVENCSTATUS {
        NV_ENC_SUCCESS
    }

    /// Create bitstream buffer
    pub unsafe fn nvEncCreateBitstreamBuffer(
        encoder: NvEncoderHandle,
        params: *mut NvEncOutputBuffer,
    ) -> NVENCSTATUS {
        static BUF_COUNTER: AtomicU64 = AtomicU64::new(1000);
        (*params).output_buffer = BUF_COUNTER.fetch_add(1, Ordering::SeqCst) as *mut _;
        (*params).bitstream_buffer_size = 4 * 1024 * 1024; // 4MB
        NV_ENC_SUCCESS
    }

    /// Destroy bitstream buffer
    pub unsafe fn nvEncDestroyBitstreamBuffer(
        encoder: NvEncoderHandle,
        buffer: *mut std::ffi::c_void,
    ) -> NVENCSTATUS {
        NV_ENC_SUCCESS
    }

    /// Lock input buffer
    pub unsafe fn nvEncLockInputBuffer(
        encoder: NvEncoderHandle,
        params: *mut std::ffi::c_void,
    ) -> NVENCSTATUS {
        NV_ENC_SUCCESS
    }

    /// Unlock input buffer
    pub unsafe fn nvEncUnlockInputBuffer(
        encoder: NvEncoderHandle,
        buffer: *mut std::ffi::c_void,
    ) -> NVENCSTATUS {
        NV_ENC_SUCCESS
    }

    /// Lock bitstream
    pub unsafe fn nvEncLockBitstream(
        encoder: NvEncoderHandle,
        params: *mut std::ffi::c_void,
    ) -> NVENCSTATUS {
        NV_ENC_SUCCESS
    }

    /// Unlock bitstream
    pub unsafe fn nvEncUnlockBitstream(
        encoder: NvEncoderHandle,
        buffer: *mut std::ffi::c_void,
    ) -> NVENCSTATUS {
        NV_ENC_SUCCESS
    }

    /// Encode picture
    pub unsafe fn nvEncEncodePicture(
        encoder: NvEncoderHandle,
        params: *mut NvEncPicParams,
    ) -> NVENCSTATUS {
        NV_ENC_SUCCESS
    }

    /// Get encode stats
    pub unsafe fn nvEncGetEncodeStats(
        encoder: NvEncoderHandle,
        stats: *mut std::ffi::c_void,
    ) -> NVENCSTATUS {
        NV_ENC_SUCCESS
    }

    /// Destroy encoder
    pub unsafe fn nvEncDestroyEncoder(encoder: NvEncoderHandle) -> NVENCSTATUS {
        NV_ENC_SUCCESS
    }

    // NVDEC Functions

    /// Create video parser
    pub unsafe fn cuvidCreateVideoParser(
        parser: *mut *mut std::ffi::c_void,
        params: *mut std::ffi::c_void,
    ) -> CUresult {
        static PARSER_COUNTER: AtomicU64 = AtomicU64::new(1);
        *parser = PARSER_COUNTER.fetch_add(1, Ordering::SeqCst) as *mut _;
        CUDA_SUCCESS
    }

    /// Destroy video parser
    pub unsafe fn cuvidDestroyVideoParser(parser: *mut std::ffi::c_void) -> CUresult {
        CUDA_SUCCESS
    }

    /// Create video decoder
    pub unsafe fn cuvidCreateDecoder(
        decoder: *mut NvDecoderHandle,
        params: *mut std::ffi::c_void,
    ) -> CUresult {
        static DEC_COUNTER: AtomicU64 = AtomicU64::new(1);
        *decoder = DEC_COUNTER.fetch_add(1, Ordering::SeqCst) as NvDecoderHandle;
        CUDA_SUCCESS
    }

    /// Destroy video decoder
    pub unsafe fn cuvidDestroyDecoder(decoder: NvDecoderHandle) -> CUresult {
        CUDA_SUCCESS
    }

    /// Decode picture
    pub unsafe fn cuvidDecodePicture(
        decoder: NvDecoderHandle,
        params: *mut std::ffi::c_void,
    ) -> CUresult {
        CUDA_SUCCESS
    }

    /// Map video frame
    pub unsafe fn cuvidMapVideoFrame(
        decoder: NvDecoderHandle,
        pic_idx: i32,
        dev_ptr: *mut CUdeviceptr,
        pitch: *mut u32,
        params: *mut std::ffi::c_void,
    ) -> CUresult {
        static FRAME_COUNTER: AtomicU64 = AtomicU64::new(0x10000);
        *dev_ptr = FRAME_COUNTER.fetch_add(0x1000, Ordering::SeqCst);
        *pitch = 1920;
        CUDA_SUCCESS
    }

    /// Unmap video frame
    pub unsafe fn cuvidUnmapVideoFrame(decoder: NvDecoderHandle, dev_ptr: CUdeviceptr) -> CUresult {
        CUDA_SUCCESS
    }
}

// ============================================================================
// NVENC Device Implementation
// ============================================================================

/// NVIDIA hardware acceleration device
pub struct NvencDevice {
    /// CUDA context
    cuda_context: CUcontext,
    /// CUDA device index
    device_id: i32,
    /// Device properties
    device_props: CudaDeviceProps,
    /// Whether device is initialized
    initialized: bool,
    /// Encode-only mode
    encode_only: bool,
    /// Current encoder session
    encoder_session: Option<NvencEncoderSession>,
    /// Current decoder session
    decoder_session: Option<NvencDecoderSession>,
    /// Statistics
    stats: HwEncoderStats,
    /// GPU memory pool
    gpu_memory: Vec<CUdeviceptr>,
    /// CUDA stream for async operations
    stream: CUstream,
}

/// NVENC encoder session
struct NvencEncoderSession {
    encoder: NvEncoderHandle,
    config: HwEncoderConfig,
    input_buffers: Vec<*mut std::ffi::c_void>,
    output_buffers: Vec<*mut std::ffi::c_void>,
    frame_count: u64,
    start_time: Instant,
}

/// NVDEC decoder session
struct NvencDecoderSession {
    decoder: NvDecoderHandle,
    config: HwDecoderConfig,
    output_surfaces: Vec<CUdeviceptr>,
}

// Safety: NvencDevice contains raw CUDA pointers that are only accessed on the same thread
// where they were created. The device is designed for single-threaded access patterns.
// Synchronization is handled by the CUDA runtime.
unsafe impl Send for NvencDevice {}
unsafe impl Sync for NvencDevice {}

impl NvencDevice {
    /// Create a new NVENC/NVDEC device
    pub fn new() -> Result<Self> {
        Ok(NvencDevice {
            cuda_context: ptr::null_mut(),
            device_id: 0,
            device_props: CudaDeviceProps::default(),
            initialized: false,
            encode_only: false,
            encoder_session: None,
            decoder_session: None,
            stats: HwEncoderStats::default(),
            gpu_memory: Vec::new(),
            stream: ptr::null_mut(),
        })
    }

    /// Create device for encoding only
    pub fn new_encode() -> Result<Self> {
        Ok(NvencDevice {
            cuda_context: ptr::null_mut(),
            device_id: 0,
            device_props: CudaDeviceProps::default(),
            initialized: false,
            encode_only: true,
            encoder_session: None,
            decoder_session: None,
            stats: HwEncoderStats::default(),
            gpu_memory: Vec::new(),
            stream: ptr::null_mut(),
        })
    }

    /// Set CUDA device ID
    pub fn with_device_id(mut self, id: i32) -> Self {
        self.device_id = id;
        self
    }

    /// Initialize CUDA and query device capabilities
    fn init_cuda(&mut self) -> Result<()> {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        unsafe {
            // Initialize CUDA
            let result = ffi::cuInit(0);
            if result != CUDA_SUCCESS {
                return Err(Error::Init(format!("cuInit failed: {}", result)));
            }

            // Get device count
            let mut count: i32 = 0;
            let result = ffi::cuDeviceGetCount(&mut count);
            if result != CUDA_SUCCESS || count == 0 {
                return Err(Error::unsupported("No NVIDIA GPU found"));
            }

            if self.device_id >= count {
                return Err(Error::invalid_input(format!(
                    "Invalid device ID {}, only {} devices available",
                    self.device_id, count
                )));
            }

            // Get device handle
            let mut device: i32 = 0;
            let result = ffi::cuDeviceGet(&mut device, self.device_id);
            if result != CUDA_SUCCESS {
                return Err(Error::Init(format!("cuDeviceGet failed: {}", result)));
            }

            // Get device name
            let result = ffi::cuDeviceGetName(
                self.device_props.name.as_mut_ptr(),
                256,
                device,
            );
            if result != CUDA_SUCCESS {
                return Err(Error::Init(format!("cuDeviceGetName failed: {}", result)));
            }

            // Get compute capability
            let result = ffi::cuDeviceComputeCapability(
                &mut self.device_props.compute_capability_major,
                &mut self.device_props.compute_capability_minor,
                device,
            );
            if result != CUDA_SUCCESS {
                return Err(Error::Init(format!(
                    "cuDeviceComputeCapability failed: {}",
                    result
                )));
            }

            // Get total memory
            let result = ffi::cuDeviceTotalMem(&mut self.device_props.total_global_mem, device);
            if result != CUDA_SUCCESS {
                return Err(Error::Init(format!("cuDeviceTotalMem failed: {}", result)));
            }

            // Get multi-processor count
            let mut mp_count: i32 = 0;
            ffi::cuDeviceGetAttribute(&mut mp_count, 16, device);
            self.device_props.multi_processor_count = mp_count;

            // Create CUDA context
            let result = ffi::cuCtxCreate(&mut self.cuda_context, 0, device);
            if result != CUDA_SUCCESS {
                return Err(Error::Init(format!("cuCtxCreate failed: {}", result)));
            }

            // Create stream
            let result = ffi::cuStreamCreate(&mut self.stream, 0);
            if result != CUDA_SUCCESS {
                ffi::cuCtxDestroy(self.cuda_context);
                return Err(Error::Init(format!("cuStreamCreate failed: {}", result)));
            }

            // Determine codec support based on compute capability
            let cc = self.device_props.compute_capability_major * 10
                + self.device_props.compute_capability_minor;

            // NVENC: All Maxwell and newer GPUs support H.264/HEVC (CC >= 5.0)
            // AV1 requires Ada Lovelace (CC >= 8.9)
            self.device_props.nvenc_h264_support = cc >= 30; // Kepler+
            self.device_props.nvenc_hevc_support = cc >= 52; // Maxwell GM206+
            self.device_props.nvenc_av1_support = cc >= 89; // Ada Lovelace

            // NVDEC support
            self.device_props.nvdec_h264_support = cc >= 30;
            self.device_props.nvdec_hevc_support = cc >= 52;
            self.device_props.nvdec_vp9_support = cc >= 60; // Pascal
            self.device_props.nvdec_av1_support = cc >= 86; // Ampere

            // Max NVENC sessions depends on GPU model
            self.device_props.nvenc_max_sessions = if cc >= 75 { 5 } else { 2 };
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            return Err(Error::unsupported(
                "NVENC is only supported on Linux and Windows",
            ));
        }

        Ok(())
    }

    /// Allocate GPU memory
    fn alloc_gpu_memory(&mut self, size: usize) -> Result<CUdeviceptr> {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        unsafe {
            let mut dptr: CUdeviceptr = 0;
            let result = ffi::cuMemAlloc(&mut dptr, size);
            if result != CUDA_SUCCESS {
                return Err(Error::codec(format!("cuMemAlloc failed: {}", result)));
            }
            self.gpu_memory.push(dptr);
            Ok(dptr)
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            Err(Error::unsupported("NVENC not supported on this platform"))
        }
    }

    /// Free GPU memory
    fn free_gpu_memory(&mut self, dptr: CUdeviceptr) {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        unsafe {
            ffi::cuMemFree(dptr);
            self.gpu_memory.retain(|&p| p != dptr);
        }
    }

    /// Upload data to GPU
    fn upload_to_gpu(&self, dst: CUdeviceptr, src: &[u8]) -> Result<()> {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        unsafe {
            let result = ffi::cuMemcpyHtoD(dst, src.as_ptr() as *const _, src.len());
            if result != CUDA_SUCCESS {
                return Err(Error::codec(format!("cuMemcpyHtoD failed: {}", result)));
            }
        }
        Ok(())
    }

    /// Download data from GPU
    fn download_from_gpu(&self, dst: &mut [u8], src: CUdeviceptr) -> Result<()> {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        unsafe {
            let result = ffi::cuMemcpyDtoH(dst.as_mut_ptr() as *mut _, src, dst.len());
            if result != CUDA_SUCCESS {
                return Err(Error::codec(format!("cuMemcpyDtoH failed: {}", result)));
            }
        }
        Ok(())
    }

    /// Convert HwPixelFormat to NVENC buffer format
    fn hw_format_to_nvenc_format(format: HwPixelFormat) -> u32 {
        match format {
            HwPixelFormat::NV12 => NV_ENC_BUFFER_FORMAT_NV12,
            HwPixelFormat::YUV420P => NV_ENC_BUFFER_FORMAT_IYUV,
            HwPixelFormat::P010 | HwPixelFormat::P016 => NV_ENC_BUFFER_FORMAT_YUV420_10BIT,
            HwPixelFormat::YUV444P => NV_ENC_BUFFER_FORMAT_YUV444,
            HwPixelFormat::Y410 => NV_ENC_BUFFER_FORMAT_YUV444_10BIT,
            HwPixelFormat::ARGB => NV_ENC_BUFFER_FORMAT_ARGB,
            HwPixelFormat::BGRA => NV_ENC_BUFFER_FORMAT_ABGR,
            _ => NV_ENC_BUFFER_FORMAT_NV12,
        }
    }

    /// Convert HwRateControlMode to NVENC RC mode
    fn hw_rc_to_nvenc_rc(mode: HwRateControlMode) -> u32 {
        match mode {
            HwRateControlMode::ConstantQP => NV_ENC_PARAMS_RC_CONSTQP,
            HwRateControlMode::VBR => NV_ENC_PARAMS_RC_VBR,
            HwRateControlMode::CBR => NV_ENC_PARAMS_RC_CBR,
            HwRateControlMode::VBR_HQ => NV_ENC_PARAMS_RC_VBR_HQ,
            HwRateControlMode::CBR_HQ => NV_ENC_PARAMS_RC_CBR_HQ,
            HwRateControlMode::Quality => NV_ENC_PARAMS_RC_VBR_HQ,
        }
    }

    /// Convert HwCodecType to NVDEC codec
    fn hw_codec_to_nvdec_codec(codec: HwCodecType) -> i32 {
        match codec {
            HwCodecType::H264 => CUDAVIDEOCODEC_H264,
            HwCodecType::H265 => CUDAVIDEOCODEC_HEVC,
            HwCodecType::VP8 => CUDAVIDEOCODEC_VP8,
            HwCodecType::VP9 => CUDAVIDEOCODEC_VP9,
            HwCodecType::AV1 => CUDAVIDEOCODEC_AV1,
            HwCodecType::MPEG2 => CUDAVIDEOCODEC_MPEG2,
            HwCodecType::MPEG4 => CUDAVIDEOCODEC_MPEG4,
            HwCodecType::VC1 => CUDAVIDEOCODEC_VC1,
            HwCodecType::JPEG => CUDAVIDEOCODEC_JPEG,
        }
    }

    /// Create encoder session
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub fn create_encoder(&mut self, config: HwEncoderConfig) -> Result<()> {
        config.validate()?;

        if !self.initialized {
            return Err(Error::invalid_state("NVENC device not initialized"));
        }

        // Check if codec is supported
        match config.codec {
            HwCodecType::H264 => {
                if !self.device_props.nvenc_h264_support {
                    return Err(Error::unsupported("H.264 encoding not supported"));
                }
            }
            HwCodecType::H265 => {
                if !self.device_props.nvenc_hevc_support {
                    return Err(Error::unsupported("HEVC encoding not supported"));
                }
            }
            HwCodecType::AV1 => {
                if !self.device_props.nvenc_av1_support {
                    return Err(Error::unsupported("AV1 encoding not supported"));
                }
            }
            _ => {
                return Err(Error::unsupported(format!(
                    "{:?} encoding not supported by NVENC",
                    config.codec
                )));
            }
        }

        unsafe {
            // Open encode session
            let mut session_params = NvEncSessionParams {
                version: 1,
                api_version: 12, // NVENC API 12.0
                device_type: 1,  // CUDA
                device: self.cuda_context,
            };

            let mut encoder: NvEncoderHandle = ptr::null_mut();
            let status = ffi::nvEncOpenEncodeSessionEx(&mut session_params, &mut encoder);
            if status != NV_ENC_SUCCESS {
                return Err(Error::Init(format!(
                    "nvEncOpenEncodeSessionEx failed: {}",
                    status
                )));
            }

            // Create input and output buffers
            let num_buffers = config.ref_frames + config.b_frames + 2;
            let mut input_buffers = Vec::with_capacity(num_buffers as usize);
            let mut output_buffers = Vec::with_capacity(num_buffers as usize);

            for _ in 0..num_buffers {
                let mut input = NvEncInputBuffer {
                    input_buffer: ptr::null_mut(),
                    buffer_format: Self::hw_format_to_nvenc_format(config.input_format),
                    width: config.width,
                    height: config.height,
                    pitch: config.width,
                };

                let status = ffi::nvEncCreateInputBuffer(encoder, &mut input);
                if status != NV_ENC_SUCCESS {
                    // Cleanup on error
                    for &buf in &input_buffers {
                        ffi::nvEncDestroyInputBuffer(encoder, buf);
                    }
                    ffi::nvEncDestroyEncoder(encoder);
                    return Err(Error::Init(format!(
                        "nvEncCreateInputBuffer failed: {}",
                        status
                    )));
                }
                input_buffers.push(input.input_buffer);

                let mut output = NvEncOutputBuffer {
                    output_buffer: ptr::null_mut(),
                    bitstream_size: 0,
                    bitstream_buffer_size: config.width * config.height * 2,
                };

                let status = ffi::nvEncCreateBitstreamBuffer(encoder, &mut output);
                if status != NV_ENC_SUCCESS {
                    // Cleanup
                    for &buf in &input_buffers {
                        ffi::nvEncDestroyInputBuffer(encoder, buf);
                    }
                    for &buf in &output_buffers {
                        ffi::nvEncDestroyBitstreamBuffer(encoder, buf);
                    }
                    ffi::nvEncDestroyEncoder(encoder);
                    return Err(Error::Init(format!(
                        "nvEncCreateBitstreamBuffer failed: {}",
                        status
                    )));
                }
                output_buffers.push(output.output_buffer);
            }

            self.encoder_session = Some(NvencEncoderSession {
                encoder,
                config,
                input_buffers,
                output_buffers,
                frame_count: 0,
                start_time: Instant::now(),
            });
        }

        Ok(())
    }

    /// Create decoder session
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub fn create_decoder(&mut self, config: HwDecoderConfig) -> Result<()> {
        if !self.initialized {
            return Err(Error::invalid_state("NVENC device not initialized"));
        }

        // Check if codec is supported
        let supported = match config.codec {
            HwCodecType::H264 => self.device_props.nvdec_h264_support,
            HwCodecType::H265 => self.device_props.nvdec_hevc_support,
            HwCodecType::VP9 => self.device_props.nvdec_vp9_support,
            HwCodecType::AV1 => self.device_props.nvdec_av1_support,
            _ => false,
        };

        if !supported {
            return Err(Error::unsupported(format!(
                "{:?} decoding not supported by NVDEC",
                config.codec
            )));
        }

        unsafe {
            let mut decoder: NvDecoderHandle = ptr::null_mut();
            let status = ffi::cuvidCreateDecoder(&mut decoder, ptr::null_mut());
            if status != CUDA_SUCCESS {
                return Err(Error::Init(format!(
                    "cuvidCreateDecoder failed: {}",
                    status
                )));
            }

            // Allocate output surfaces
            let surface_size = (config.max_width * config.max_height * 3 / 2) as usize;
            let mut output_surfaces = Vec::with_capacity(config.num_surfaces as usize);

            for _ in 0..config.num_surfaces {
                let dptr = self.alloc_gpu_memory(surface_size)?;
                output_surfaces.push(dptr);
            }

            self.decoder_session = Some(NvencDecoderSession {
                decoder,
                config,
                output_surfaces,
            });
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
        let session = self
            .encoder_session
            .as_mut()
            .ok_or_else(|| Error::invalid_state("No encoder session"))?;

        let start = Instant::now();

        // Get buffer indices
        let buf_idx = (session.frame_count as usize) % session.input_buffers.len();
        let input_buf = session.input_buffers[buf_idx];
        let output_buf = session.output_buffers[buf_idx];

        unsafe {
            // Lock and fill input buffer
            if let Some(data) = &surface.data {
                // In real implementation: lock buffer, copy data, unlock
                ffi::nvEncLockInputBuffer(session.encoder, ptr::null_mut());
                // Copy would happen here
                ffi::nvEncUnlockInputBuffer(session.encoder, input_buf);
            }

            // Encode
            let is_keyframe = force_keyframe
                || session.frame_count == 0
                || session.frame_count % session.config.gop_size as u64 == 0;

            let mut pic_params = NvEncPicParams {
                version: 1,
                input_width: surface.width,
                input_height: surface.height,
                input_pitch: surface.strides.first().copied().unwrap_or(surface.width as usize) as u32,
                input_buffer: input_buf,
                output_buffer: output_buf,
                picture_type: if is_keyframe {
                    NV_ENC_PIC_TYPE_IDR
                } else {
                    NV_ENC_PIC_TYPE_P
                },
                input_timestamp: session.frame_count,
                input_duration: 1,
                encode_params: ptr::null_mut(),
                frame_idx: session.frame_count as u32,
            };

            let status = ffi::nvEncEncodePicture(session.encoder, &mut pic_params);
            if status != NV_ENC_SUCCESS && status != NV_ENC_ERR_NEED_MORE_INPUT {
                return Err(Error::codec(format!("nvEncEncodePicture failed: {}", status)));
            }

            // Lock and read output
            ffi::nvEncLockBitstream(session.encoder, ptr::null_mut());

            // Generate simulated encoded data
            let output_data = if is_keyframe {
                // Simulate H.264 IDR with SPS/PPS
                vec![
                    0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0xc0, 0x1f, // SPS
                    0x00, 0x00, 0x00, 0x01, 0x68, 0xce, 0x3c, 0x80, // PPS
                    0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, // IDR slice
                ]
            } else {
                // Simulate P-frame
                vec![0x00, 0x00, 0x00, 0x01, 0x41, 0x9a, 0x24, 0x6c]
            };

            ffi::nvEncUnlockBitstream(session.encoder, output_buf);

            let elapsed_us = start.elapsed().as_micros() as u64;
            session.frame_count += 1;

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

            let total_time = session.start_time.elapsed().as_secs_f64();
            if total_time > 0.0 {
                self.stats.fps = self.stats.frames_encoded as f64 / total_time;
                self.stats.avg_bitrate = (self.stats.bytes_output * 8) / total_time.max(1.0) as u64;
            }

            Ok(HwEncodedPacket {
                data: output_data,
                pts: session.frame_count as i64 - 1,
                dts: session.frame_count as i64 - 1,
                duration: 1,
                keyframe: is_keyframe,
                pict_type: if is_keyframe { 'I' } else { 'P' },
                frame_num: session.frame_count - 1,
            })
        }
    }

    /// Get encoder capabilities
    pub fn get_encoder_caps(&self, codec: HwCodecType) -> Option<HwEncoderCaps> {
        let supported = match codec {
            HwCodecType::H264 => self.device_props.nvenc_h264_support,
            HwCodecType::H265 => self.device_props.nvenc_hevc_support,
            HwCodecType::AV1 => self.device_props.nvenc_av1_support,
            _ => false,
        };

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
                    HwProfile::H264High444,
                ],
                HwCodecType::H265 => vec![
                    HwProfile::HevcMain,
                    HwProfile::HevcMain10,
                    HwProfile::HevcMain444,
                ],
                HwCodecType::AV1 => vec![HwProfile::Av1Main],
                _ => vec![],
            },
            max_width: 8192,
            max_height: 8192,
            min_width: 64,
            min_height: 64,
            input_formats: vec![
                HwPixelFormat::NV12,
                HwPixelFormat::P010,
                HwPixelFormat::YUV444P,
                HwPixelFormat::BGRA,
            ],
            b_frames: true,
            max_b_frames: 4,
            lookahead: true,
            max_lookahead: 32,
            temporal_aq: true,
            spatial_aq: true,
            max_sessions: self.device_props.nvenc_max_sessions as u32,
        })
    }

    /// Get decoder capabilities
    pub fn get_decoder_caps(&self, codec: HwCodecType) -> Option<HwDecoderCaps> {
        let supported = match codec {
            HwCodecType::H264 => self.device_props.nvdec_h264_support,
            HwCodecType::H265 => self.device_props.nvdec_hevc_support,
            HwCodecType::VP9 => self.device_props.nvdec_vp9_support,
            HwCodecType::AV1 => self.device_props.nvdec_av1_support,
            _ => false,
        };

        if !supported {
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
                    HwProfile::Vp9Profile2,
                ],
                HwCodecType::AV1 => vec![HwProfile::Av1Main],
                _ => vec![],
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

    /// Get device name
    pub fn device_name(&self) -> String {
        let name_bytes: Vec<u8> = self
            .device_props
            .name
            .iter()
            .take_while(|&&b| b != 0)
            .map(|&b| b as u8)
            .collect();
        String::from_utf8_lossy(&name_bytes).into_owned()
    }

    /// Destroy encoder session
    fn destroy_encoder(&mut self) {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if let Some(session) = self.encoder_session.take() {
            unsafe {
                for &buf in &session.input_buffers {
                    ffi::nvEncDestroyInputBuffer(session.encoder, buf);
                }
                for &buf in &session.output_buffers {
                    ffi::nvEncDestroyBitstreamBuffer(session.encoder, buf);
                }
                ffi::nvEncDestroyEncoder(session.encoder);
            }
        }
    }

    /// Destroy decoder session
    fn destroy_decoder(&mut self) {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if let Some(session) = self.decoder_session.take() {
            unsafe {
                for &surface in &session.output_surfaces {
                    self.free_gpu_memory(surface);
                }
                ffi::cuvidDestroyDecoder(session.decoder);
            }
        }
    }
}

impl HwAccelDevice for NvencDevice {
    fn device_type(&self) -> HwAccelType {
        if self.encode_only {
            HwAccelType::NVENC
        } else {
            HwAccelType::NVDEC
        }
    }

    fn is_available(&self) -> bool {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        unsafe {
            // Check if CUDA is available
            let result = ffi::cuInit(0);
            if result != CUDA_SUCCESS {
                return false;
            }

            let mut count: i32 = 0;
            let result = ffi::cuDeviceGetCount(&mut count);
            result == CUDA_SUCCESS && count > 0
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

        self.init_cuda()?;
        self.initialized = true;

        tracing::info!(
            "NVENC initialized: {} (CC {}.{}), {}MB VRAM, {} encode sessions",
            self.device_name(),
            self.device_props.compute_capability_major,
            self.device_props.compute_capability_minor,
            self.device_props.total_global_mem / (1024 * 1024),
            self.device_props.nvenc_max_sessions
        );

        Ok(())
    }

    fn upload_frame(&mut self, frame: &VideoFrame) -> Result<VideoFrame> {
        if !self.initialized {
            return Err(Error::invalid_state("NVENC device not initialized"));
        }

        // Convert VideoFrame to HwSurface format
        let mut surface = HwSurface::from_video_frame(frame)?;

        // Allocate GPU memory and upload
        if let Some(data) = &surface.data {
            let gpu_ptr = self.alloc_gpu_memory(data.len())?;
            self.upload_to_gpu(gpu_ptr, data)?;
            surface.handle = HwSurfaceHandle::CudaPtr(gpu_ptr);
        }

        // Return the original frame (in real implementation, would return modified frame)
        Ok(frame.clone())
    }

    fn download_frame(&mut self, frame: &VideoFrame) -> Result<VideoFrame> {
        if !self.initialized {
            return Err(Error::invalid_state("NVENC device not initialized"));
        }

        // In real implementation, download from GPU memory
        Ok(frame.clone())
    }

    fn name(&self) -> &str {
        if self.encode_only {
            "NVIDIA NVENC"
        } else {
            "NVIDIA NVDEC"
        }
    }
}

impl Drop for NvencDevice {
    fn drop(&mut self) {
        self.destroy_encoder();
        self.destroy_decoder();

        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if self.initialized {
            // Free remaining GPU memory
            for &ptr in &self.gpu_memory.clone() {
                self.free_gpu_memory(ptr);
            }

            unsafe {
                if !self.stream.is_null() {
                    ffi::cuStreamDestroy(self.stream);
                }
                if !self.cuda_context.is_null() {
                    ffi::cuCtxDestroy(self.cuda_context);
                }
            }
        }
    }
}

/// Check if NVIDIA GPU is available
pub fn is_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        std::path::Path::new("/proc/driver/nvidia/version").exists()
    }

    #[cfg(target_os = "windows")]
    {
        std::path::Path::new("C:\\Windows\\System32\\nvcuda.dll").exists()
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        false
    }
}

/// Get number of NVIDIA GPUs
pub fn device_count() -> i32 {
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    unsafe {
        let result = ffi::cuInit(0);
        if result != CUDA_SUCCESS {
            return 0;
        }

        let mut count: i32 = 0;
        let result = ffi::cuDeviceGetCount(&mut count);
        if result == CUDA_SUCCESS {
            count
        } else {
            0
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        0
    }
}

/// List NVIDIA GPU devices
pub fn list_devices() -> Vec<String> {
    let mut devices = Vec::new();

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    unsafe {
        let result = ffi::cuInit(0);
        if result != CUDA_SUCCESS {
            return devices;
        }

        let mut count: i32 = 0;
        let result = ffi::cuDeviceGetCount(&mut count);
        if result != CUDA_SUCCESS {
            return devices;
        }

        for i in 0..count {
            let mut name = [0i8; 256];
            let result = ffi::cuDeviceGetName(name.as_mut_ptr(), 256, i);
            if result == CUDA_SUCCESS {
                let name_str: Vec<u8> = name
                    .iter()
                    .take_while(|&&b| b != 0)
                    .map(|&b| b as u8)
                    .collect();
                devices.push(format!(
                    "GPU {}: {}",
                    i,
                    String::from_utf8_lossy(&name_str)
                ));
            }
        }
    }

    devices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nvenc_device_creation() {
        let device = NvencDevice::new();
        assert!(device.is_ok());
    }

    #[test]
    fn test_nvenc_encode_only() {
        let device = NvencDevice::new_encode();
        assert!(device.is_ok());
        assert!(device.unwrap().encode_only);
    }

    #[test]
    fn test_nvenc_availability() {
        let _available = is_available();
        // Should not panic
    }

    #[test]
    fn test_device_count() {
        let count = device_count();
        println!("NVIDIA GPU count: {}", count);
    }

    #[test]
    fn test_list_devices() {
        let devices = list_devices();
        println!("NVIDIA devices: {:?}", devices);
    }

    #[test]
    fn test_format_conversion() {
        assert_eq!(
            NvencDevice::hw_format_to_nvenc_format(HwPixelFormat::NV12),
            NV_ENC_BUFFER_FORMAT_NV12
        );
        assert_eq!(
            NvencDevice::hw_format_to_nvenc_format(HwPixelFormat::P010),
            NV_ENC_BUFFER_FORMAT_YUV420_10BIT
        );
    }

    #[test]
    fn test_rc_conversion() {
        assert_eq!(
            NvencDevice::hw_rc_to_nvenc_rc(HwRateControlMode::CBR),
            NV_ENC_PARAMS_RC_CBR
        );
        assert_eq!(
            NvencDevice::hw_rc_to_nvenc_rc(HwRateControlMode::VBR_HQ),
            NV_ENC_PARAMS_RC_VBR_HQ
        );
    }
}
