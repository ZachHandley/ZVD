//! Pixel format definitions

use std::fmt;

/// Pixel format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// RGB24 - 8 bits per component, packed
    RGB24,
    /// RGBA - RGB with alpha channel
    RGBA,
    /// BGR24 - BGR format
    BGR24,
    /// BGRA - BGR with alpha
    BGRA,
    /// YUV420P - Planar YUV 4:2:0
    YUV420P,
    /// YUV422P - Planar YUV 4:2:2
    YUV422P,
    /// YUV444P - Planar YUV 4:4:4
    YUV444P,
    /// YUV420P10LE - 10-bit YUV 4:2:0
    YUV420P10LE,
    /// YUV422P10LE - 10-bit YUV 4:2:2
    YUV422P10LE,
    /// YUV444P10LE - 10-bit YUV 4:4:4
    YUV444P10LE,
    /// GRAY8 - 8-bit grayscale
    GRAY8,
    /// GRAY16 - 16-bit grayscale
    GRAY16,
    /// Unknown format
    Unknown,
}

impl PixelFormat {
    /// Get the number of components in this pixel format
    pub fn num_components(&self) -> usize {
        match self {
            PixelFormat::RGB24 | PixelFormat::BGR24 => 3,
            PixelFormat::RGBA | PixelFormat::BGRA => 4,
            PixelFormat::YUV420P | PixelFormat::YUV422P | PixelFormat::YUV444P => 3,
            PixelFormat::YUV420P10LE | PixelFormat::YUV422P10LE | PixelFormat::YUV444P10LE => 3,
            PixelFormat::GRAY8 | PixelFormat::GRAY16 => 1,
            PixelFormat::Unknown => 0,
        }
    }

    /// Get the bits per pixel for this format
    pub fn bits_per_pixel(&self) -> usize {
        match self {
            PixelFormat::RGB24 | PixelFormat::BGR24 => 24,
            PixelFormat::RGBA | PixelFormat::BGRA => 32,
            PixelFormat::YUV420P => 12,
            PixelFormat::YUV422P => 16,
            PixelFormat::YUV444P => 24,
            PixelFormat::YUV420P10LE => 15,
            PixelFormat::YUV422P10LE => 20,
            PixelFormat::YUV444P10LE => 30,
            PixelFormat::GRAY8 => 8,
            PixelFormat::GRAY16 => 16,
            PixelFormat::Unknown => 0,
        }
    }

    /// Check if this is a planar format
    pub fn is_planar(&self) -> bool {
        matches!(
            self,
            PixelFormat::YUV420P
                | PixelFormat::YUV422P
                | PixelFormat::YUV444P
                | PixelFormat::YUV420P10LE
                | PixelFormat::YUV422P10LE
                | PixelFormat::YUV444P10LE
        )
    }

    /// Check if this is a YUV format
    pub fn is_yuv(&self) -> bool {
        matches!(
            self,
            PixelFormat::YUV420P
                | PixelFormat::YUV422P
                | PixelFormat::YUV444P
                | PixelFormat::YUV420P10LE
                | PixelFormat::YUV422P10LE
                | PixelFormat::YUV444P10LE
        )
    }

    /// Check if this is an RGB format
    pub fn is_rgb(&self) -> bool {
        matches!(
            self,
            PixelFormat::RGB24 | PixelFormat::RGBA | PixelFormat::BGR24 | PixelFormat::BGRA
        )
    }
}

impl fmt::Display for PixelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            PixelFormat::RGB24 => "rgb24",
            PixelFormat::RGBA => "rgba",
            PixelFormat::BGR24 => "bgr24",
            PixelFormat::BGRA => "bgra",
            PixelFormat::YUV420P => "yuv420p",
            PixelFormat::YUV422P => "yuv422p",
            PixelFormat::YUV444P => "yuv444p",
            PixelFormat::YUV420P10LE => "yuv420p10le",
            PixelFormat::YUV422P10LE => "yuv422p10le",
            PixelFormat::YUV444P10LE => "yuv444p10le",
            PixelFormat::GRAY8 => "gray8",
            PixelFormat::GRAY16 => "gray16",
            PixelFormat::Unknown => "unknown",
        };
        write!(f, "{}", name)
    }
}

impl Default for PixelFormat {
    fn default() -> Self {
        PixelFormat::Unknown
    }
}
