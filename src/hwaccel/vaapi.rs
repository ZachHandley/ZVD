//! VAAPI (Video Acceleration API) support for Linux
//!
//! VAAPI provides hardware-accelerated video processing on Intel and AMD GPUs
//! on Linux systems. This module provides encoding and decoding support.

use super::{HwAccelDevice, HwAccelType};
use crate::codec::VideoFrame;
use crate::error::{Error, Result};

/// VAAPI device
pub struct VaapiDevice {
    initialized: bool,
    device_path: String,
}

impl VaapiDevice {
    /// Create a new VAAPI device
    pub fn new() -> Result<Self> {
        Ok(VaapiDevice {
            initialized: false,
            device_path: "/dev/dri/renderD128".to_string(),
        })
    }

    /// Create VAAPI device with custom path
    pub fn with_path(path: &str) -> Result<Self> {
        Ok(VaapiDevice {
            initialized: false,
            device_path: path.to_string(),
        })
    }
}

impl HwAccelDevice for VaapiDevice {
    fn device_type(&self) -> HwAccelType {
        HwAccelType::VAAPI
    }

    fn is_available(&self) -> bool {
        // Check if VAAPI device exists
        #[cfg(target_os = "linux")]
        {
            std::path::Path::new(&self.device_path).exists()
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    fn init(&mut self) -> Result<()> {
        if !self.is_available() {
            return Err(Error::unsupported("VAAPI device not found"));
        }

        // Placeholder for actual VAAPI initialization
        // Would use libva to initialize the device
        self.initialized = true;
        Ok(())
    }

    fn upload_frame(&mut self, frame: &VideoFrame) -> Result<VideoFrame> {
        if !self.initialized {
            return Err(Error::invalid_state("VAAPI device not initialized"));
        }

        // Placeholder - would upload frame to GPU memory
        Ok(frame.clone())
    }

    fn download_frame(&mut self, frame: &VideoFrame) -> Result<VideoFrame> {
        if !self.initialized {
            return Err(Error::invalid_state("VAAPI device not initialized"));
        }

        // Placeholder - would download frame from GPU memory
        Ok(frame.clone())
    }

    fn name(&self) -> &str {
        "VAAPI"
    }
}

/// Check if VAAPI is available
pub fn is_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        std::path::Path::new("/dev/dri/renderD128").exists()
            || std::path::Path::new("/dev/dri/card0").exists()
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
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
}
