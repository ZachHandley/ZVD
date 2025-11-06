//! Intel Quick Sync Video (QSV) hardware acceleration
//!
//! Provides Intel integrated GPU hardware-accelerated encoding and decoding.

use super::{HwAccelDevice, HwAccelType};
use crate::codec::VideoFrame;
use crate::error::{Error, Result};

/// Intel QSV device
pub struct QsvDevice {
    initialized: bool,
    impl_type: QsvImplType,
}

/// QSV implementation type
#[derive(Debug, Clone, Copy)]
pub enum QsvImplType {
    /// Software implementation
    Software,
    /// Hardware implementation
    Hardware,
    /// Automatic selection
    Auto,
}

impl QsvDevice {
    /// Create a new QSV device
    pub fn new() -> Result<Self> {
        Ok(QsvDevice {
            initialized: false,
            impl_type: QsvImplType::Auto,
        })
    }

    /// Create QSV device with specific implementation
    pub fn with_impl_type(impl_type: QsvImplType) -> Result<Self> {
        Ok(QsvDevice {
            initialized: false,
            impl_type,
        })
    }
}

impl HwAccelDevice for QsvDevice {
    fn device_type(&self) -> HwAccelType {
        HwAccelType::QSV
    }

    fn is_available(&self) -> bool {
        // Placeholder - would check for Intel GPU and Media SDK
        #[cfg(target_os = "windows")]
        {
            // Check for Intel GPU
            false // Conservative
        }
        #[cfg(target_os = "linux")]
        {
            // Check for /dev/dri/renderD128 and Intel GPU
            std::path::Path::new("/dev/dri/renderD128").exists()
        }
        #[cfg(not(any(target_os = "windows", target_os = "linux")))]
        {
            false
        }
    }

    fn init(&mut self) -> Result<()> {
        if !self.is_available() {
            return Err(Error::unsupported("Intel QSV not available"));
        }

        // Placeholder for Intel Media SDK initialization
        self.initialized = true;
        Ok(())
    }

    fn upload_frame(&mut self, frame: &VideoFrame) -> Result<VideoFrame> {
        if !self.initialized {
            return Err(Error::invalid_state("QSV device not initialized"));
        }

        // Placeholder - would upload to QSV surface
        Ok(frame.clone())
    }

    fn download_frame(&mut self, frame: &VideoFrame) -> Result<VideoFrame> {
        if !self.initialized {
            return Err(Error::invalid_state("QSV device not initialized"));
        }

        // Placeholder - would download from QSV surface
        Ok(frame.clone())
    }

    fn name(&self) -> &str {
        "Intel Quick Sync Video"
    }
}

/// Check if Intel QSV is available
pub fn is_available() -> bool {
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    {
        // Would check for Intel GPU and drivers
        // For now, conservative false
        false
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        false
    }
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
        let device = QsvDevice::with_impl_type(QsvImplType::Hardware);
        assert!(device.is_ok());
    }
}
