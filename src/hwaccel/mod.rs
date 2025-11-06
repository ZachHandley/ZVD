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

pub mod vaapi;
pub mod nvenc;
pub mod qsv;
pub mod videotoolbox;
pub mod common;

use crate::error::Result;
use crate::codec::{VideoFrame, Frame};

/// Hardware acceleration device types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwAccelType {
    /// No hardware acceleration
    None,
    /// Video Acceleration API (Linux)
    VAAPI,
    /// NVIDIA CUDA
    NVENC,
    /// NVIDIA decoder
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
    /// Vulkan
    Vulkan,
}

/// Hardware acceleration device
pub trait HwAccelDevice: Send + Sync {
    /// Get the device type
    fn device_type(&self) -> HwAccelType;

    /// Check if the device is available
    fn is_available(&self) -> bool;

    /// Initialize the device
    fn init(&mut self) -> Result<()>;

    /// Upload frame to device memory
    fn upload_frame(&mut self, frame: &VideoFrame) -> Result<VideoFrame>;

    /// Download frame from device memory
    fn download_frame(&mut self, frame: &VideoFrame) -> Result<VideoFrame>;

    /// Get device name/description
    fn name(&self) -> &str;
}

/// Hardware acceleration context
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
    pub fn init(&mut self) -> Result<()> {
        match self.device_type {
            HwAccelType::VAAPI => {
                #[cfg(target_os = "linux")]
                {
                    let mut device = vaapi::VaapiDevice::new()?;
                    device.init()?;
                    self.device = Some(Box::new(device));
                }
            }
            HwAccelType::NVENC | HwAccelType::NVDEC => {
                let mut device = nvenc::NvencDevice::new()?;
                device.init()?;
                self.device = Some(Box::new(device));
            }
            HwAccelType::QSV => {
                let mut device = qsv::QsvDevice::new()?;
                device.init()?;
                self.device = Some(Box::new(device));
            }
            HwAccelType::VideoToolbox => {
                #[cfg(target_os = "macos")]
                {
                    let mut device = videotoolbox::VideoToolboxDevice::new()?;
                    device.init()?;
                    self.device = Some(Box::new(device));
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Get the device
    pub fn device(&self) -> Option<&dyn HwAccelDevice> {
        self.device.as_ref().map(|d| d.as_ref())
    }

    /// Get mutable device
    pub fn device_mut(&mut self) -> Option<&mut (dyn HwAccelDevice + '_)> {
        match &mut self.device {
            Some(device) => Some(&mut **device),
            None => None,
        }
    }

    /// Check if hardware acceleration is available
    pub fn is_available(&self) -> bool {
        self.device.as_ref().map_or(false, |d| d.is_available())
    }
}

/// Detect available hardware acceleration devices
pub fn detect_hw_devices() -> Vec<HwAccelType> {
    let mut devices = Vec::new();

    // Check VAAPI (Linux only)
    #[cfg(target_os = "linux")]
    {
        if vaapi::is_available() {
            devices.push(HwAccelType::VAAPI);
        }
    }

    // Check NVIDIA
    if nvenc::is_available() {
        devices.push(HwAccelType::NVENC);
        devices.push(HwAccelType::NVDEC);
    }

    // Check Intel QSV
    if qsv::is_available() {
        devices.push(HwAccelType::QSV);
    }

    // Check VideoToolbox (macOS/iOS only)
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        if videotoolbox::is_available() {
            devices.push(HwAccelType::VideoToolbox);
        }
    }

    devices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hw_accel_context_creation() {
        let ctx = HwAccelContext::new(HwAccelType::None);
        assert_eq!(ctx.device_type, HwAccelType::None);
    }

    #[test]
    fn test_detect_hw_devices() {
        let devices = detect_hw_devices();
        // Should not panic and return a list (possibly empty)
        println!("Detected HW devices: {:?}", devices);
    }
}
