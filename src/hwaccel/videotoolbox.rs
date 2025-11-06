//! Apple VideoToolbox hardware acceleration
//!
//! Provides hardware-accelerated encoding and decoding on macOS and iOS
//! using Apple's VideoToolbox framework.

use super::{HwAccelDevice, HwAccelType};
use crate::codec::VideoFrame;
use crate::error::{Error, Result};

/// VideoToolbox device
pub struct VideoToolboxDevice {
    initialized: bool,
}

impl VideoToolboxDevice {
    /// Create a new VideoToolbox device
    pub fn new() -> Result<Self> {
        Ok(VideoToolboxDevice {
            initialized: false,
        })
    }
}

impl HwAccelDevice for VideoToolboxDevice {
    fn device_type(&self) -> HwAccelType {
        HwAccelType::VideoToolbox
    }

    fn is_available(&self) -> bool {
        // VideoToolbox is always available on macOS/iOS
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            true
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        {
            false
        }
    }

    fn init(&mut self) -> Result<()> {
        if !self.is_available() {
            return Err(Error::unsupported("VideoToolbox only available on macOS/iOS"));
        }

        // Placeholder for VideoToolbox initialization
        self.initialized = true;
        Ok(())
    }

    fn upload_frame(&mut self, frame: &VideoFrame) -> Result<VideoFrame> {
        if !self.initialized {
            return Err(Error::invalid_state("VideoToolbox not initialized"));
        }

        // Placeholder - would convert to CVPixelBuffer
        Ok(frame.clone())
    }

    fn download_frame(&mut self, frame: &VideoFrame) -> Result<VideoFrame> {
        if !self.initialized {
            return Err(Error::invalid_state("VideoToolbox not initialized"));
        }

        // Placeholder - would convert from CVPixelBuffer
        Ok(frame.clone())
    }

    fn name(&self) -> &str {
        "Apple VideoToolbox"
    }
}

/// Check if VideoToolbox is available
pub fn is_available() -> bool {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        true
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_videotoolbox_device_creation() {
        let device = VideoToolboxDevice::new();
        assert!(device.is_ok());
    }

    #[test]
    fn test_videotoolbox_availability() {
        let available = is_available();
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        assert!(available);
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        assert!(!available);
    }
}
