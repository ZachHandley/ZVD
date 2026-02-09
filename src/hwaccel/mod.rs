//! Hardware acceleration support
//!
//! This module provides hardware-accelerated encoding and decoding using various
//! platform-specific APIs:
//!
//! - **VAAPI** - Video Acceleration API (Linux, Intel/AMD)
//! - **NVENC/NVDEC** - NVIDIA hardware acceleration
//! - **QSV** - Intel Quick Sync Video
//! - **VideoToolbox** - Apple hardware acceleration (macOS/iOS)
//! - **AMF** - AMD Media Framework
//! - **DXVA2/D3D11VA** - DirectX Video Acceleration (Windows)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use zvd::hwaccel::{detect_hw_devices, HwAccelContext, HwAccelType};
//!
//! // Detect available hardware acceleration
//! let devices = detect_hw_devices();
//! println!("Available HW acceleration: {:?}", devices);
//!
//! // Create a hardware context
//! if devices.contains(&HwAccelType::NVENC) {
//!     let mut ctx = HwAccelContext::new(HwAccelType::NVENC);
//!     ctx.init().expect("Failed to init NVENC");
//! }
//! ```

pub mod common;

// Platform-specific hardware acceleration modules
#[cfg(feature = "vaapi")]
pub mod vaapi;

#[cfg(feature = "nvenc")]
pub mod nvenc;

#[cfg(feature = "qsv")]
pub mod qsv;

#[cfg(feature = "videotoolbox")]
pub mod videotoolbox;

// Re-export common types
pub use common::{
    HwCodecType, HwDecoderCaps, HwDecoderConfig, HwEncodedPacket, HwEncoderCaps, HwEncoderConfig,
    HwEncoderStats, HwPixelFormat, HwProfile, HwRateControlMode, HwSurface, HwSurfaceHandle,
    HwSurfacePool,
};

use crate::codec::VideoFrame;
use crate::error::{Error, Result};

/// Hardware acceleration device types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum HwAccelType {
    /// No hardware acceleration
    None,
    /// Video Acceleration API (Linux)
    VAAPI,
    /// NVIDIA CUDA/NVENC encoder
    NVENC,
    /// NVIDIA NVDEC decoder
    NVDEC,
    /// Intel Quick Sync Video
    QSV,
    /// Apple VideoToolbox
    VideoToolbox,
    /// AMD Media Framework
    AMF,
    /// DirectX Video Acceleration 2
    DXVA2,
    /// Direct3D 11 Video Acceleration
    D3D11VA,
    /// Vulkan Video
    Vulkan,
}

impl HwAccelType {
    /// Get a human-readable name for this acceleration type
    pub fn name(&self) -> &'static str {
        match self {
            HwAccelType::None => "None",
            HwAccelType::VAAPI => "VA-API",
            HwAccelType::NVENC => "NVIDIA NVENC",
            HwAccelType::NVDEC => "NVIDIA NVDEC",
            HwAccelType::QSV => "Intel Quick Sync",
            HwAccelType::VideoToolbox => "Apple VideoToolbox",
            HwAccelType::AMF => "AMD AMF",
            HwAccelType::DXVA2 => "DXVA2",
            HwAccelType::D3D11VA => "D3D11VA",
            HwAccelType::Vulkan => "Vulkan Video",
        }
    }

    /// Check if this type supports encoding
    pub fn supports_encoding(&self) -> bool {
        matches!(
            self,
            HwAccelType::NVENC
                | HwAccelType::VAAPI
                | HwAccelType::QSV
                | HwAccelType::VideoToolbox
                | HwAccelType::AMF
        )
    }

    /// Check if this type supports decoding
    pub fn supports_decoding(&self) -> bool {
        matches!(
            self,
            HwAccelType::NVDEC
                | HwAccelType::VAAPI
                | HwAccelType::QSV
                | HwAccelType::VideoToolbox
                | HwAccelType::DXVA2
                | HwAccelType::D3D11VA
        )
    }
}

impl std::fmt::Display for HwAccelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Hardware acceleration device trait
///
/// This trait defines the interface for hardware acceleration devices.
/// Each platform-specific implementation (VAAPI, NVENC, QSV, VideoToolbox)
/// implements this trait.
pub trait HwAccelDevice: Send + Sync {
    /// Get the device type
    fn device_type(&self) -> HwAccelType;

    /// Check if the device is available and usable
    fn is_available(&self) -> bool;

    /// Initialize the device
    fn init(&mut self) -> Result<()>;

    /// Upload frame from CPU memory to device memory
    fn upload_frame(&mut self, frame: &VideoFrame) -> Result<VideoFrame>;

    /// Download frame from device memory to CPU memory
    fn download_frame(&mut self, frame: &VideoFrame) -> Result<VideoFrame>;

    /// Get device name/description
    fn name(&self) -> &str;
}

/// Hardware acceleration context
///
/// Wraps a hardware device and provides a unified interface for
/// hardware-accelerated encoding and decoding.
pub struct HwAccelContext {
    device_type: HwAccelType,
    device: Option<Box<dyn HwAccelDevice>>,
}

impl HwAccelContext {
    /// Create a new hardware acceleration context
    pub fn new(device_type: HwAccelType) -> Self {
        HwAccelContext {
            device_type,
            device: None,
        }
    }

    /// Initialize hardware acceleration
    #[allow(unused_variables)]
    pub fn init(&mut self) -> Result<()> {
        match self.device_type {
            HwAccelType::VAAPI => {
                #[cfg(all(target_os = "linux", feature = "vaapi"))]
                {
                    let mut device = vaapi::VaapiDevice::new()?;
                    device.init()?;
                    self.device = Some(Box::new(device));
                }
                #[cfg(not(all(target_os = "linux", feature = "vaapi")))]
                {
                    return Err(Error::unsupported("VAAPI not available"));
                }
            }
            HwAccelType::NVENC | HwAccelType::NVDEC => {
                #[cfg(feature = "nvenc")]
                {
                    let mut device = nvenc::NvencDevice::new()?;
                    device.init()?;
                    self.device = Some(Box::new(device));
                }
                #[cfg(not(feature = "nvenc"))]
                {
                    return Err(Error::unsupported("NVENC not available"));
                }
            }
            HwAccelType::QSV => {
                #[cfg(feature = "qsv")]
                {
                    let mut device = qsv::QsvDevice::new()?;
                    device.init()?;
                    self.device = Some(Box::new(device));
                }
                #[cfg(not(feature = "qsv"))]
                {
                    return Err(Error::unsupported("QSV not available"));
                }
            }
            HwAccelType::VideoToolbox => {
                #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "videotoolbox"))]
                {
                    let mut device = videotoolbox::VideoToolboxDevice::new()?;
                    device.init()?;
                    self.device = Some(Box::new(device));
                }
                #[cfg(not(all(any(target_os = "macos", target_os = "ios"), feature = "videotoolbox")))]
                {
                    return Err(Error::unsupported("VideoToolbox not available"));
                }
            }
            HwAccelType::None => {
                // No hardware acceleration
            }
            _ => {
                return Err(Error::unsupported(format!(
                    "{} is not supported on this platform",
                    self.device_type
                )));
            }
        }
        Ok(())
    }

    /// Get the device type
    pub fn device_type(&self) -> HwAccelType {
        self.device_type
    }

    /// Get the device
    pub fn device(&self) -> Option<&dyn HwAccelDevice> {
        self.device.as_ref().map(|d| d.as_ref())
    }

    /// Get mutable device reference
    pub fn device_mut(&mut self) -> Option<&mut (dyn HwAccelDevice + '_)> {
        match &mut self.device {
            Some(device) => Some(device.as_mut()),
            None => None,
        }
    }

    /// Check if hardware acceleration is initialized and available
    pub fn is_available(&self) -> bool {
        self.device.as_ref().map_or(false, |d| d.is_available())
    }

    /// Check if this context is initialized
    pub fn is_initialized(&self) -> bool {
        self.device.is_some()
    }

    /// Get device name
    pub fn device_name(&self) -> Option<&str> {
        self.device.as_ref().map(|d| d.name())
    }

    /// Upload a frame to device memory
    pub fn upload_frame(&mut self, frame: &VideoFrame) -> Result<VideoFrame> {
        if let Some(device) = self.device.as_mut() {
            device.upload_frame(frame)
        } else {
            Err(Error::invalid_state("Hardware device not initialized"))
        }
    }

    /// Download a frame from device memory
    pub fn download_frame(&mut self, frame: &VideoFrame) -> Result<VideoFrame> {
        if let Some(device) = self.device.as_mut() {
            device.download_frame(frame)
        } else {
            Err(Error::invalid_state("Hardware device not initialized"))
        }
    }
}

/// Detailed information about a detected hardware device
#[derive(Debug, Clone)]
pub struct HwDeviceInfo {
    /// Device type
    pub device_type: HwAccelType,
    /// Device name/description
    pub name: String,
    /// Vendor name
    pub vendor: String,
    /// Supported encoding codecs
    pub encode_codecs: Vec<HwCodecType>,
    /// Supported decoding codecs
    pub decode_codecs: Vec<HwCodecType>,
    /// Maximum encoding sessions
    pub max_encode_sessions: u32,
    /// Maximum resolution width
    pub max_width: u32,
    /// Maximum resolution height
    pub max_height: u32,
}

impl HwDeviceInfo {
    /// Check if this device supports encoding a specific codec
    pub fn supports_encode(&self, codec: HwCodecType) -> bool {
        self.encode_codecs.contains(&codec)
    }

    /// Check if this device supports decoding a specific codec
    pub fn supports_decode(&self, codec: HwCodecType) -> bool {
        self.decode_codecs.contains(&codec)
    }
}

/// Detect available hardware acceleration devices
///
/// Returns a list of available hardware acceleration types on this system.
pub fn detect_hw_devices() -> Vec<HwAccelType> {
    let mut devices = Vec::new();

    // Check VAAPI (Linux only)
    #[cfg(all(target_os = "linux", feature = "vaapi"))]
    {
        if vaapi::is_available() {
            devices.push(HwAccelType::VAAPI);
        }
    }

    // Check NVIDIA (Linux and Windows)
    #[cfg(feature = "nvenc")]
    {
        if nvenc::is_available() {
            devices.push(HwAccelType::NVENC);
            devices.push(HwAccelType::NVDEC);
        }
    }

    // Check Intel QSV (Linux and Windows)
    #[cfg(feature = "qsv")]
    {
        if qsv::is_available() {
            devices.push(HwAccelType::QSV);
        }
    }

    // Check VideoToolbox (macOS/iOS only)
    #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "videotoolbox"))]
    {
        if videotoolbox::is_available() {
            devices.push(HwAccelType::VideoToolbox);
        }
    }

    devices
}

/// Get detailed information about all available hardware devices
#[allow(unused_mut)]
pub fn get_hw_device_info() -> Vec<HwDeviceInfo> {
    let mut infos = Vec::new();

    // Check VAAPI
    #[cfg(all(target_os = "linux", feature = "vaapi"))]
    {
        if vaapi::is_available() {
            infos.push(HwDeviceInfo {
                device_type: HwAccelType::VAAPI,
                name: "VA-API".to_string(),
                vendor: "Intel/AMD".to_string(),
                encode_codecs: vec![
                    HwCodecType::H264,
                    HwCodecType::H265,
                    HwCodecType::VP9,
                    HwCodecType::AV1,
                ],
                decode_codecs: vec![
                    HwCodecType::H264,
                    HwCodecType::H265,
                    HwCodecType::VP9,
                    HwCodecType::AV1,
                    HwCodecType::MPEG2,
                ],
                max_encode_sessions: 4,
                max_width: 8192,
                max_height: 8192,
            });
        }
    }

    // Check NVIDIA
    #[cfg(feature = "nvenc")]
    {
        if nvenc::is_available() {
            let devices = nvenc::list_devices();
            let name = devices.first().cloned().unwrap_or_else(|| "NVIDIA GPU".to_string());

            infos.push(HwDeviceInfo {
                device_type: HwAccelType::NVENC,
                name,
                vendor: "NVIDIA".to_string(),
                encode_codecs: vec![HwCodecType::H264, HwCodecType::H265, HwCodecType::AV1],
                decode_codecs: vec![
                    HwCodecType::H264,
                    HwCodecType::H265,
                    HwCodecType::VP9,
                    HwCodecType::AV1,
                ],
                max_encode_sessions: 5,
                max_width: 8192,
                max_height: 8192,
            });
        }
    }

    // Check Intel QSV
    #[cfg(feature = "qsv")]
    {
        if qsv::is_available() {
            let devices = qsv::list_devices();
            let name = devices.first().cloned().unwrap_or_else(|| "Intel GPU".to_string());

            infos.push(HwDeviceInfo {
                device_type: HwAccelType::QSV,
                name,
                vendor: "Intel".to_string(),
                encode_codecs: vec![
                    HwCodecType::H264,
                    HwCodecType::H265,
                    HwCodecType::VP9,
                    HwCodecType::AV1,
                ],
                decode_codecs: vec![
                    HwCodecType::H264,
                    HwCodecType::H265,
                    HwCodecType::VP9,
                    HwCodecType::AV1,
                    HwCodecType::MPEG2,
                    HwCodecType::VC1,
                ],
                max_encode_sessions: 4,
                max_width: 8192,
                max_height: 8192,
            });
        }
    }

    // Check VideoToolbox
    #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "videotoolbox"))]
    {
        if videotoolbox::is_available() {
            let codecs = videotoolbox::list_supported_codecs();

            infos.push(HwDeviceInfo {
                device_type: HwAccelType::VideoToolbox,
                name: videotoolbox::get_version_info(),
                vendor: "Apple".to_string(),
                encode_codecs: vec![HwCodecType::H264, HwCodecType::H265],
                decode_codecs: codecs,
                max_encode_sessions: 8,
                max_width: 8192,
                max_height: 8192,
            });
        }
    }

    infos
}

/// Select the best hardware acceleration type for a given codec
///
/// Returns the best available hardware acceleration type for encoding
/// the specified codec, or None if no hardware support is available.
pub fn select_best_hw_encoder(codec: HwCodecType) -> Option<HwAccelType> {
    let devices = detect_hw_devices();

    // Priority order: NVENC > QSV > VideoToolbox > VAAPI
    for hw_type in &[
        HwAccelType::NVENC,
        HwAccelType::QSV,
        HwAccelType::VideoToolbox,
        HwAccelType::VAAPI,
    ] {
        if !devices.contains(hw_type) {
            continue;
        }

        // Check if this device supports the codec
        let supported = match hw_type {
            HwAccelType::NVENC => matches!(
                codec,
                HwCodecType::H264 | HwCodecType::H265 | HwCodecType::AV1
            ),
            HwAccelType::QSV => matches!(
                codec,
                HwCodecType::H264 | HwCodecType::H265 | HwCodecType::VP9 | HwCodecType::AV1
            ),
            HwAccelType::VideoToolbox => matches!(codec, HwCodecType::H264 | HwCodecType::H265),
            HwAccelType::VAAPI => matches!(
                codec,
                HwCodecType::H264 | HwCodecType::H265 | HwCodecType::VP9 | HwCodecType::AV1
            ),
            _ => false,
        };

        if supported {
            return Some(*hw_type);
        }
    }

    None
}

/// Select the best hardware acceleration type for decoding
pub fn select_best_hw_decoder(codec: HwCodecType) -> Option<HwAccelType> {
    let devices = detect_hw_devices();

    // Priority order: NVDEC > QSV > VideoToolbox > VAAPI
    for hw_type in &[
        HwAccelType::NVDEC,
        HwAccelType::QSV,
        HwAccelType::VideoToolbox,
        HwAccelType::VAAPI,
    ] {
        if !devices.contains(hw_type) {
            continue;
        }

        // Check if this device supports the codec
        let supported = match hw_type {
            HwAccelType::NVDEC => matches!(
                codec,
                HwCodecType::H264
                    | HwCodecType::H265
                    | HwCodecType::VP9
                    | HwCodecType::AV1
                    | HwCodecType::MPEG2
            ),
            HwAccelType::QSV => matches!(
                codec,
                HwCodecType::H264
                    | HwCodecType::H265
                    | HwCodecType::VP9
                    | HwCodecType::AV1
                    | HwCodecType::MPEG2
            ),
            HwAccelType::VideoToolbox => matches!(
                codec,
                HwCodecType::H264 | HwCodecType::H265 | HwCodecType::VP9 | HwCodecType::AV1
            ),
            HwAccelType::VAAPI => matches!(
                codec,
                HwCodecType::H264
                    | HwCodecType::H265
                    | HwCodecType::VP9
                    | HwCodecType::AV1
                    | HwCodecType::MPEG2
            ),
            _ => false,
        };

        if supported {
            return Some(*hw_type);
        }
    }

    None
}

/// Create a hardware encoder for the given configuration
///
/// Automatically selects the best available hardware encoder for the
/// specified codec and creates an initialized context.
pub fn create_hw_encoder(config: &HwEncoderConfig) -> Result<HwAccelContext> {
    let hw_type = select_best_hw_encoder(config.codec).ok_or_else(|| {
        Error::unsupported(format!(
            "No hardware encoder available for {:?}",
            config.codec
        ))
    })?;

    let mut ctx = HwAccelContext::new(hw_type);
    ctx.init()?;

    Ok(ctx)
}

/// Create a hardware decoder for the given configuration
pub fn create_hw_decoder(config: &HwDecoderConfig) -> Result<HwAccelContext> {
    let hw_type = select_best_hw_decoder(config.codec).ok_or_else(|| {
        Error::unsupported(format!(
            "No hardware decoder available for {:?}",
            config.codec
        ))
    })?;

    let mut ctx = HwAccelContext::new(hw_type);
    ctx.init()?;

    Ok(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hw_accel_context_creation() {
        let ctx = HwAccelContext::new(HwAccelType::None);
        assert_eq!(ctx.device_type, HwAccelType::None);
        assert!(!ctx.is_initialized());
    }

    #[test]
    fn test_detect_hw_devices() {
        let devices = detect_hw_devices();
        // Should not panic and return a list (possibly empty)
        println!("Detected HW devices: {:?}", devices);
    }

    #[test]
    fn test_get_hw_device_info() {
        let infos = get_hw_device_info();
        for info in &infos {
            println!(
                "Device: {} ({:?}), Encode: {:?}, Decode: {:?}",
                info.name, info.device_type, info.encode_codecs, info.decode_codecs
            );
        }
    }

    #[test]
    fn test_hw_accel_type_properties() {
        assert!(HwAccelType::NVENC.supports_encoding());
        assert!(!HwAccelType::NVENC.supports_decoding());
        assert!(!HwAccelType::NVDEC.supports_encoding());
        assert!(HwAccelType::NVDEC.supports_decoding());
        assert!(HwAccelType::VAAPI.supports_encoding());
        assert!(HwAccelType::VAAPI.supports_decoding());
    }

    #[test]
    fn test_select_best_hw_encoder() {
        let best = select_best_hw_encoder(HwCodecType::H264);
        println!("Best HW encoder for H.264: {:?}", best);
    }

    #[test]
    fn test_select_best_hw_decoder() {
        let best = select_best_hw_decoder(HwCodecType::H264);
        println!("Best HW decoder for H.264: {:?}", best);
    }

    #[test]
    fn test_hw_accel_type_display() {
        assert_eq!(format!("{}", HwAccelType::NVENC), "NVIDIA NVENC");
        assert_eq!(format!("{}", HwAccelType::VAAPI), "VA-API");
        assert_eq!(format!("{}", HwAccelType::QSV), "Intel Quick Sync");
    }
}
