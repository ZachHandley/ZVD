//! Common hardware acceleration utilities

use crate::error::Result;

/// Hardware surface/buffer representation
#[derive(Debug, Clone)]
pub struct HwSurface {
    pub width: u32,
    pub height: u32,
    pub format: HwPixelFormat,
    pub data: Vec<u8>,
}

/// Hardware pixel formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwPixelFormat {
    NV12,
    P010,
    YUV420P,
    YUV422P,
    BGRA,
    RGBA,
}

impl HwSurface {
    pub fn new(width: u32, height: u32, format: HwPixelFormat) -> Self {
        let size = match format {
            HwPixelFormat::NV12 | HwPixelFormat::YUV420P => (width * height * 3 / 2) as usize,
            HwPixelFormat::P010 => (width * height * 3) as usize,
            HwPixelFormat::YUV422P => (width * height * 2) as usize,
            HwPixelFormat::BGRA | HwPixelFormat::RGBA => (width * height * 4) as usize,
        };

        HwSurface {
            width,
            height,
            format,
            data: vec![0; size],
        }
    }
}
