//! Color Space Conversion Utilities
//!
//! Provides conversion between different color spaces commonly used in video:
//!
//! - **RGB** - Red, Green, Blue (computer graphics, displays)
//! - **YUV** - Luminance (Y) + Chrominance (U, V) (video compression)
//! - **YCbCr** - Digital version of YUV (ITU-R BT.601, BT.709, BT.2020)
//! - **HSV** - Hue, Saturation, Value (color manipulation)
//!
//! ## Color Spaces
//!
//! **RGB:**
//! - Linear additive color
//! - Used in computer graphics, displays
//! - Range: [0, 255] per channel (8-bit)
//!
//! **YUV/YCbCr:**
//! - Separates luminance from chrominance
//! - Enables chroma subsampling (4:2:0, 4:2:2)
//! - Used in JPEG, MPEG, broadcast video
//! - Y (luma): 16-235, Cb/Cr (chroma): 16-240 (limited range)
//! - Y: 0-255, Cb/Cr: 0-255 (full range)
//!
//! **Standards:**
//! - BT.601 (SD): Standard definition (SDTV)
//! - BT.709 (HD): High definition (HDTV)
//! - BT.2020 (UHD): Ultra high definition (4K/8K)

use crate::error::Result;

/// YUV/YCbCr conversion standards
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorStandard {
    /// ITU-R BT.601 (Standard Definition)
    Bt601,
    /// ITU-R BT.709 (High Definition)
    Bt709,
    /// ITU-R BT.2020 (Ultra High Definition)
    Bt2020,
}

/// Color range
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorRange {
    /// Limited range (16-235 for Y, 16-240 for Cb/Cr)
    Limited,
    /// Full range (0-255 for all)
    Full,
}

/// RGB color (8-bit per channel)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb {
    /// Create a new RGB color
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Rgb { r, g, b }
    }

    /// Create from slice [r, g, b]
    pub fn from_slice(data: &[u8]) -> Self {
        Rgb {
            r: data[0],
            g: data[1],
            b: data[2],
        }
    }

    /// Convert to slice [r, g, b]
    pub fn to_array(&self) -> [u8; 3] {
        [self.r, self.g, self.b]
    }
}

/// YUV color
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Yuv {
    pub y: u8,
    pub u: u8,
    pub v: u8,
}

impl Yuv {
    /// Create a new YUV color
    pub fn new(y: u8, u: u8, v: u8) -> Self {
        Yuv { y, u, v }
    }

    /// Create from slice [y, u, v]
    pub fn from_slice(data: &[u8]) -> Self {
        Yuv {
            y: data[0],
            u: data[1],
            v: data[2],
        }
    }

    /// Convert to slice [y, u, v]
    pub fn to_array(&self) -> [u8; 3] {
        [self.y, self.u, self.v]
    }
}

/// HSV color (hue, saturation, value)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hsv {
    /// Hue (0-360 degrees)
    pub h: f32,
    /// Saturation (0.0-1.0)
    pub s: f32,
    /// Value/Brightness (0.0-1.0)
    pub v: f32,
}

impl Hsv {
    /// Create a new HSV color
    pub fn new(h: f32, s: f32, v: f32) -> Self {
        Hsv {
            h: h.rem_euclid(360.0),
            s: s.clamp(0.0, 1.0),
            v: v.clamp(0.0, 1.0),
        }
    }
}

/// Color space converter
pub struct ColorConverter {
    standard: ColorStandard,
    range: ColorRange,
}

impl ColorConverter {
    /// Create a new converter
    pub fn new(standard: ColorStandard, range: ColorRange) -> Self {
        ColorConverter { standard, range }
    }

    /// RGB to YUV conversion
    pub fn rgb_to_yuv(&self, rgb: Rgb) -> Yuv {
        let r = rgb.r as f32;
        let g = rgb.g as f32;
        let b = rgb.b as f32;

        // Get conversion matrix coefficients
        let (kr, kg, kb) = self.get_rgb_to_yuv_coefficients();

        // Calculate YUV (normalized 0-1)
        let y_norm = kr * r + kg * g + kb * b;
        let u_norm = (b - y_norm) / (2.0 * (1.0 - kb));
        let v_norm = (r - y_norm) / (2.0 * (1.0 - kr));

        // Scale to appropriate range
        let (y, u, v) = match self.range {
            ColorRange::Limited => {
                // Y: 16-235, U/V: 16-240
                let y = (16.0 + y_norm * 219.0 / 255.0).clamp(16.0, 235.0) as u8;
                let u = (128.0 + u_norm * 224.0 / 255.0).clamp(16.0, 240.0) as u8;
                let v = (128.0 + v_norm * 224.0 / 255.0).clamp(16.0, 240.0) as u8;
                (y, u, v)
            }
            ColorRange::Full => {
                // Y/U/V: 0-255
                let y = y_norm.clamp(0.0, 255.0) as u8;
                let u = (128.0 + u_norm).clamp(0.0, 255.0) as u8;
                let v = (128.0 + v_norm).clamp(0.0, 255.0) as u8;
                (y, u, v)
            }
        };

        Yuv::new(y, u, v)
    }

    /// YUV to RGB conversion
    pub fn yuv_to_rgb(&self, yuv: Yuv) -> Rgb {
        // Normalize to 0-1 range based on color range
        let (y_norm, u_norm, v_norm) = match self.range {
            ColorRange::Limited => {
                // Y: 16-235, U/V: 16-240
                let y = ((yuv.y as f32 - 16.0) * 255.0 / 219.0).clamp(0.0, 255.0);
                let u = ((yuv.u as f32 - 128.0) * 255.0 / 224.0).clamp(-128.0, 127.0);
                let v = ((yuv.v as f32 - 128.0) * 255.0 / 224.0).clamp(-128.0, 127.0);
                (y, u, v)
            }
            ColorRange::Full => {
                // Y/U/V: 0-255
                let y = yuv.y as f32;
                let u = (yuv.u as f32 - 128.0);
                let v = (yuv.v as f32 - 128.0);
                (y, u, v)
            }
        };

        // Get conversion matrix coefficients
        let (kr, kg, kb) = self.get_rgb_to_yuv_coefficients();

        // Calculate RGB
        let r = y_norm + v_norm * 2.0 * (1.0 - kr);
        let g = y_norm - u_norm * 2.0 * kb * (1.0 - kb) / kg - v_norm * 2.0 * kr * (1.0 - kr) / kg;
        let b = y_norm + u_norm * 2.0 * (1.0 - kb);

        Rgb::new(
            r.clamp(0.0, 255.0) as u8,
            g.clamp(0.0, 255.0) as u8,
            b.clamp(0.0, 255.0) as u8,
        )
    }

    /// Get RGB to YUV matrix coefficients
    fn get_rgb_to_yuv_coefficients(&self) -> (f32, f32, f32) {
        match self.standard {
            ColorStandard::Bt601 => {
                // ITU-R BT.601 (Standard Definition)
                let kr = 0.299;
                let kb = 0.114;
                let kg = 1.0 - kr - kb; // 0.587
                (kr, kg, kb)
            }
            ColorStandard::Bt709 => {
                // ITU-R BT.709 (High Definition)
                let kr = 0.2126;
                let kb = 0.0722;
                let kg = 1.0 - kr - kb; // 0.7152
                (kr, kg, kb)
            }
            ColorStandard::Bt2020 => {
                // ITU-R BT.2020 (Ultra High Definition)
                let kr = 0.2627;
                let kb = 0.0593;
                let kg = 1.0 - kr - kb; // 0.6780
                (kr, kg, kb)
            }
        }
    }

    /// Convert RGB buffer to YUV420P planar format
    pub fn rgb_buffer_to_yuv420p(&self, rgb_data: &[u8], width: usize, height: usize) -> Result<Vec<Vec<u8>>> {
        if rgb_data.len() != width * height * 3 {
            return Err(crate::error::Error::InvalidInput(format!(
                "RGB buffer size mismatch: expected {}, got {}",
                width * height * 3,
                rgb_data.len()
            )));
        }

        let mut y_plane = vec![0u8; width * height];
        let mut u_plane = vec![0u8; (width / 2) * (height / 2)];
        let mut v_plane = vec![0u8; (width / 2) * (height / 2)];

        // Convert each pixel
        for y in 0..height {
            for x in 0..width {
                let rgb_idx = (y * width + x) * 3;
                let rgb = Rgb::from_slice(&rgb_data[rgb_idx..rgb_idx + 3]);
                let yuv = self.rgb_to_yuv(rgb);

                // Y plane (full resolution)
                y_plane[y * width + x] = yuv.y;

                // U and V planes (2x2 subsampled for 4:2:0)
                if y % 2 == 0 && x % 2 == 0 {
                    let uv_idx = (y / 2) * (width / 2) + (x / 2);
                    u_plane[uv_idx] = yuv.u;
                    v_plane[uv_idx] = yuv.v;
                }
            }
        }

        Ok(vec![y_plane, u_plane, v_plane])
    }

    /// Convert YUV420P planar format to RGB buffer
    pub fn yuv420p_to_rgb_buffer(&self, y_plane: &[u8], u_plane: &[u8], v_plane: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
        if y_plane.len() != width * height {
            return Err(crate::error::Error::InvalidInput("Y plane size mismatch".to_string()));
        }
        if u_plane.len() != (width / 2) * (height / 2) {
            return Err(crate::error::Error::InvalidInput("U plane size mismatch".to_string()));
        }
        if v_plane.len() != (width / 2) * (height / 2) {
            return Err(crate::error::Error::InvalidInput("V plane size mismatch".to_string()));
        }

        let mut rgb_data = vec![0u8; width * height * 3];

        for y in 0..height {
            for x in 0..width {
                let y_val = y_plane[y * width + x];
                let uv_idx = (y / 2) * (width / 2) + (x / 2);
                let u_val = u_plane[uv_idx];
                let v_val = v_plane[uv_idx];

                let yuv = Yuv::new(y_val, u_val, v_val);
                let rgb = self.yuv_to_rgb(yuv);

                let rgb_idx = (y * width + x) * 3;
                rgb_data[rgb_idx] = rgb.r;
                rgb_data[rgb_idx + 1] = rgb.g;
                rgb_data[rgb_idx + 2] = rgb.b;
            }
        }

        Ok(rgb_data)
    }
}

/// RGB to HSV conversion
pub fn rgb_to_hsv(rgb: Rgb) -> Hsv {
    let r = rgb.r as f32 / 255.0;
    let g = rgb.g as f32 / 255.0;
    let b = rgb.b as f32 / 255.0;

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let v = max;

    let s = if max == 0.0 { 0.0 } else { delta / max };

    let h = if delta == 0.0 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / delta) % 6.0)
    } else if max == g {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };

    Hsv::new(if h < 0.0 { h + 360.0 } else { h }, s, v)
}

/// HSV to RGB conversion
pub fn hsv_to_rgb(hsv: Hsv) -> Rgb {
    let c = hsv.v * hsv.s;
    let x = c * (1.0 - ((hsv.h / 60.0) % 2.0 - 1.0).abs());
    let m = hsv.v - c;

    let (r, g, b) = if hsv.h < 60.0 {
        (c, x, 0.0)
    } else if hsv.h < 120.0 {
        (x, c, 0.0)
    } else if hsv.h < 180.0 {
        (0.0, c, x)
    } else if hsv.h < 240.0 {
        (0.0, x, c)
    } else if hsv.h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    Rgb::new(
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_to_yuv_bt709() {
        let converter = ColorConverter::new(ColorStandard::Bt709, ColorRange::Full);
        let rgb = Rgb::new(255, 0, 0); // Pure red
        let yuv = converter.rgb_to_yuv(rgb);

        // Red should have high Y, low U, high V
        assert!(yuv.y > 0);
        assert!(yuv.u < 128);
        assert!(yuv.v > 128);
    }

    #[test]
    fn test_yuv_to_rgb_bt709() {
        let converter = ColorConverter::new(ColorStandard::Bt709, ColorRange::Full);
        let yuv = Yuv::new(128, 128, 128); // Gray
        let rgb = converter.yuv_to_rgb(yuv);

        // Gray should have equal RGB components
        assert!((rgb.r as i32 - rgb.g as i32).abs() < 2);
        assert!((rgb.g as i32 - rgb.b as i32).abs() < 2);
    }

    #[test]
    fn test_rgb_yuv_roundtrip() {
        let converter = ColorConverter::new(ColorStandard::Bt709, ColorRange::Full);
        let original = Rgb::new(100, 150, 200);
        let yuv = converter.rgb_to_yuv(original);
        let rgb = converter.yuv_to_rgb(yuv);

        // Allow some error due to rounding
        assert!((original.r as i32 - rgb.r as i32).abs() < 3);
        assert!((original.g as i32 - rgb.g as i32).abs() < 3);
        assert!((original.b as i32 - rgb.b as i32).abs() < 3);
    }

    #[test]
    fn test_rgb_to_hsv() {
        let rgb = Rgb::new(255, 0, 0); // Pure red
        let hsv = rgb_to_hsv(rgb);

        assert!((hsv.h - 0.0).abs() < 1.0); // Hue should be ~0
        assert!((hsv.s - 1.0).abs() < 0.01); // Full saturation
        assert!((hsv.v - 1.0).abs() < 0.01); // Full value
    }

    #[test]
    fn test_hsv_to_rgb() {
        let hsv = Hsv::new(120.0, 1.0, 1.0); // Pure green
        let rgb = hsv_to_rgb(hsv);

        assert_eq!(rgb.r, 0);
        assert_eq!(rgb.g, 255);
        assert_eq!(rgb.b, 0);
    }

    #[test]
    fn test_rgb_hsv_roundtrip() {
        let original = Rgb::new(100, 150, 200);
        let hsv = rgb_to_hsv(original);
        let rgb = hsv_to_rgb(hsv);

        assert!((original.r as i32 - rgb.r as i32).abs() < 2);
        assert!((original.g as i32 - rgb.g as i32).abs() < 2);
        assert!((original.b as i32 - rgb.b as i32).abs() < 2);
    }

    #[test]
    fn test_color_standards() {
        let rgb = Rgb::new(128, 128, 128);

        // Test BT.601
        let conv601 = ColorConverter::new(ColorStandard::Bt601, ColorRange::Full);
        let yuv601 = conv601.rgb_to_yuv(rgb);

        // Test BT.709
        let conv709 = ColorConverter::new(ColorStandard::Bt709, ColorRange::Full);
        let yuv709 = conv709.rgb_to_yuv(rgb);

        // Test BT.2020
        let conv2020 = ColorConverter::new(ColorStandard::Bt2020, ColorRange::Full);
        let yuv2020 = conv2020.rgb_to_yuv(rgb);

        // Gray should have similar Y values across standards
        assert!((yuv601.y as i32 - yuv709.y as i32).abs() < 5);
        assert!((yuv709.y as i32 - yuv2020.y as i32).abs() < 5);
    }

    #[test]
    fn test_limited_vs_full_range() {
        let rgb = Rgb::new(0, 0, 0); // Black

        let conv_full = ColorConverter::new(ColorStandard::Bt709, ColorRange::Full);
        let yuv_full = conv_full.rgb_to_yuv(rgb);

        let conv_limited = ColorConverter::new(ColorStandard::Bt709, ColorRange::Limited);
        let yuv_limited = conv_limited.rgb_to_yuv(rgb);

        // Full range black should be Y=0
        // Limited range black should be Y=16
        assert_eq!(yuv_full.y, 0);
        assert_eq!(yuv_limited.y, 16);
    }
}
