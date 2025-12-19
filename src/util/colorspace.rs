//! Color Space Conversion Utilities
//!
//! Provides conversion between different color spaces commonly used in video:
//!
//! - **RGB** - Red, Green, Blue (computer graphics, displays)
//! - **YUV** - Luminance (Y) + Chrominance (U, V) (video compression)
//! - **YCbCr** - Digital version of YUV (ITU-R BT.601, BT.709, BT.2020)
//! - **HSV** - Hue, Saturation, Value (color manipulation)
//! - **XYZ** - CIE XYZ (device-independent color space)
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
//!
//! ## Advanced Features
//!
//! **Gamut Conversion:**
//! - Rec.709 ↔ Rec.2020 (HD ↔ UHD)
//! - RGB primaries transformation
//! - Chromatic adaptation
//!
//! **Transfer Functions:**
//! - Linear ↔ sRGB
//! - Linear ↔ PQ (SMPTE ST 2084)
//! - Linear ↔ HLG (ITU-R BT.2100)
//!
//! **Gamut Mapping:**
//! - Clip (simple)
//! - Compress (preserve hue)
//! - Adaptive (perceptual)

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

/// Transfer function (EOTF/OETF)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferFunction {
    /// Linear (no transfer function)
    Linear,
    /// sRGB (gamma ~2.2)
    Srgb,
    /// ITU-R BT.709 (similar to sRGB)
    Bt709,
    /// Perceptual Quantizer (SMPTE ST 2084) - HDR
    Pq,
    /// Hybrid Log-Gamma (ITU-R BT.2100) - HDR
    Hlg,
}

/// Gamut mapping method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GamutMapping {
    /// Clip out-of-gamut colors
    Clip,
    /// Compress to gamut boundary
    Compress,
    /// Adaptive perceptual mapping
    Adaptive,
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

/// CIE XYZ color (device-independent)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Xyz {
    /// X component (0.0-1.0)
    pub x: f32,
    /// Y component (luminance, 0.0-1.0)
    pub y: f32,
    /// Z component (0.0-1.0)
    pub z: f32,
}

impl Xyz {
    /// Create new XYZ color
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Xyz { x, y, z }
    }
}

/// RGB color with floating point (0.0-1.0 per channel)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RgbFloat {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl RgbFloat {
    /// Create new RGB float color
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        RgbFloat { r, g, b }
    }

    /// Convert from 8-bit RGB
    pub fn from_rgb(rgb: Rgb) -> Self {
        RgbFloat {
            r: rgb.r as f32 / 255.0,
            g: rgb.g as f32 / 255.0,
            b: rgb.b as f32 / 255.0,
        }
    }

    /// Convert to 8-bit RGB
    pub fn to_rgb(&self) -> Rgb {
        Rgb::new(
            (self.r * 255.0).clamp(0.0, 255.0) as u8,
            (self.g * 255.0).clamp(0.0, 255.0) as u8,
            (self.b * 255.0).clamp(0.0, 255.0) as u8,
        )
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
    pub fn rgb_buffer_to_yuv420p(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Vec<Vec<u8>>> {
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
    pub fn yuv420p_to_rgb_buffer(
        &self,
        y_plane: &[u8],
        u_plane: &[u8],
        v_plane: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Vec<u8>> {
        if y_plane.len() != width * height {
            return Err(crate::error::Error::InvalidInput(
                "Y plane size mismatch".to_string(),
            ));
        }
        if u_plane.len() != (width / 2) * (height / 2) {
            return Err(crate::error::Error::InvalidInput(
                "U plane size mismatch".to_string(),
            ));
        }
        if v_plane.len() != (width / 2) * (height / 2) {
            return Err(crate::error::Error::InvalidInput(
                "V plane size mismatch".to_string(),
            ));
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

/// Gamut converter for converting between different RGB primaries
pub struct GamutConverter {
    /// Source color standard
    source: ColorStandard,
    /// Target color standard
    target: ColorStandard,
    /// Gamut mapping method
    mapping: GamutMapping,
}

impl GamutConverter {
    /// Create new gamut converter
    pub fn new(source: ColorStandard, target: ColorStandard, mapping: GamutMapping) -> Self {
        GamutConverter {
            source,
            target,
            mapping,
        }
    }

    /// Convert between gamuts (e.g., Rec.709 → Rec.2020)
    pub fn convert(&self, rgb: RgbFloat) -> RgbFloat {
        // Convert to XYZ (device-independent)
        let xyz = self.rgb_to_xyz(rgb, self.source);

        // Convert back to target RGB
        let mut target_rgb = self.xyz_to_rgb(xyz, self.target);

        // Apply gamut mapping if needed
        if self.is_out_of_gamut(&target_rgb) {
            target_rgb = self.apply_gamut_mapping(target_rgb);
        }

        target_rgb
    }

    /// RGB to XYZ conversion
    fn rgb_to_xyz(&self, rgb: RgbFloat, standard: ColorStandard) -> Xyz {
        // Apply linearization (remove gamma)
        let linear = RgbFloat::new(
            Self::srgb_to_linear(rgb.r),
            Self::srgb_to_linear(rgb.g),
            Self::srgb_to_linear(rgb.b),
        );

        // Get conversion matrix based on primaries
        let matrix = self.get_rgb_to_xyz_matrix(standard);

        Xyz::new(
            matrix[0][0] * linear.r + matrix[0][1] * linear.g + matrix[0][2] * linear.b,
            matrix[1][0] * linear.r + matrix[1][1] * linear.g + matrix[1][2] * linear.b,
            matrix[2][0] * linear.r + matrix[2][1] * linear.g + matrix[2][2] * linear.b,
        )
    }

    /// XYZ to RGB conversion
    fn xyz_to_rgb(&self, xyz: Xyz, standard: ColorStandard) -> RgbFloat {
        // Get inverse matrix
        let matrix = self.get_xyz_to_rgb_matrix(standard);

        let r = matrix[0][0] * xyz.x + matrix[0][1] * xyz.y + matrix[0][2] * xyz.z;
        let g = matrix[1][0] * xyz.x + matrix[1][1] * xyz.y + matrix[1][2] * xyz.z;
        let b = matrix[2][0] * xyz.x + matrix[2][1] * xyz.y + matrix[2][2] * xyz.z;

        // Apply gamma
        RgbFloat::new(
            Self::linear_to_srgb(r),
            Self::linear_to_srgb(g),
            Self::linear_to_srgb(b),
        )
    }

    /// Get RGB to XYZ matrix for standard
    fn get_rgb_to_xyz_matrix(&self, standard: ColorStandard) -> [[f32; 3]; 3] {
        match standard {
            ColorStandard::Bt709 => [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ],
            ColorStandard::Bt2020 => [
                [0.6369580, 0.1446169, 0.1688810],
                [0.2627002, 0.6779981, 0.0593017],
                [0.0000000, 0.0280727, 1.0609851],
            ],
            ColorStandard::Bt601 => [
                [0.4306190, 0.3415419, 0.1783091],
                [0.2220379, 0.7066384, 0.0713236],
                [0.0201853, 0.1295504, 0.9390944],
            ],
        }
    }

    /// Get XYZ to RGB matrix for standard
    fn get_xyz_to_rgb_matrix(&self, standard: ColorStandard) -> [[f32; 3]; 3] {
        match standard {
            ColorStandard::Bt709 => [
                [3.2404542, -1.5371385, -0.4985314],
                [-0.9692660, 1.8760108, 0.0415560],
                [0.0556434, -0.2040259, 1.0572252],
            ],
            ColorStandard::Bt2020 => [
                [1.7166512, -0.3556708, -0.2533663],
                [-0.6666844, 1.6164812, 0.0157685],
                [0.0176399, -0.0427706, 0.9421031],
            ],
            ColorStandard::Bt601 => [
                [3.0628971, -1.3931791, -0.4757517],
                [-0.9692660, 1.8760108, 0.0415560],
                [0.0678775, -0.2288548, 1.0693490],
            ],
        }
    }

    /// Check if color is out of gamut
    fn is_out_of_gamut(&self, rgb: &RgbFloat) -> bool {
        rgb.r < 0.0 || rgb.r > 1.0 || rgb.g < 0.0 || rgb.g > 1.0 || rgb.b < 0.0 || rgb.b > 1.0
    }

    /// Apply gamut mapping
    fn apply_gamut_mapping(&self, rgb: RgbFloat) -> RgbFloat {
        match self.mapping {
            GamutMapping::Clip => RgbFloat::new(
                rgb.r.clamp(0.0, 1.0),
                rgb.g.clamp(0.0, 1.0),
                rgb.b.clamp(0.0, 1.0),
            ),
            GamutMapping::Compress => self.compress_gamut(rgb),
            GamutMapping::Adaptive => self.adaptive_gamut_map(rgb),
        }
    }

    /// Compress out-of-gamut colors (preserve hue)
    pub fn compress_gamut(&self, rgb: RgbFloat) -> RgbFloat {
        let max = rgb.r.max(rgb.g).max(rgb.b);
        let min = rgb.r.min(rgb.g).min(rgb.b);

        // Handle negative values first by clamping to 0
        let r = rgb.r.max(0.0);
        let g = rgb.g.max(0.0);
        let b = rgb.b.max(0.0);

        let max_pos = r.max(g).max(b);

        if max_pos <= 1.0 {
            return RgbFloat::new(r, g, b);
        }

        // Scale down to fit in [0, 1], preserving ratios
        let scale = 1.0 / max_pos;
        RgbFloat::new(r * scale, g * scale, b * scale)
    }

    /// Adaptive perceptual gamut mapping
    fn adaptive_gamut_map(&self, rgb: RgbFloat) -> RgbFloat {
        // Soft clip using sigmoid
        let soft_clip = |x: f32| {
            if x < 0.0 {
                0.0
            } else if x > 1.0 {
                let excess = x - 1.0;
                1.0 - (-excess * 3.0).exp() * 0.1
            } else {
                x
            }
        };

        RgbFloat::new(soft_clip(rgb.r), soft_clip(rgb.g), soft_clip(rgb.b))
    }

    /// sRGB to linear
    fn srgb_to_linear(val: f32) -> f32 {
        if val <= 0.04045 {
            val / 12.92
        } else {
            ((val + 0.055) / 1.055).powf(2.4)
        }
    }

    /// Linear to sRGB
    fn linear_to_srgb(val: f32) -> f32 {
        if val <= 0.0031308 {
            val * 12.92
        } else {
            1.055 * val.powf(1.0 / 2.4) - 0.055
        }
    }
}

/// Transfer function converter
pub struct TransferFunctionConverter;

impl TransferFunctionConverter {
    /// Apply transfer function (EOTF - electrical to optical)
    pub fn eotf(value: f32, function: TransferFunction) -> f32 {
        match function {
            TransferFunction::Linear => value,
            TransferFunction::Srgb | TransferFunction::Bt709 => Self::srgb_eotf(value),
            TransferFunction::Pq => Self::pq_eotf(value),
            TransferFunction::Hlg => Self::hlg_eotf(value),
        }
    }

    /// Apply inverse transfer function (OETF - optical to electrical)
    pub fn oetf(value: f32, function: TransferFunction) -> f32 {
        match function {
            TransferFunction::Linear => value,
            TransferFunction::Srgb | TransferFunction::Bt709 => Self::srgb_oetf(value),
            TransferFunction::Pq => Self::pq_oetf(value),
            TransferFunction::Hlg => Self::hlg_oetf(value),
        }
    }

    /// sRGB EOTF (gamma decode)
    fn srgb_eotf(e: f32) -> f32 {
        if e <= 0.04045 {
            e / 12.92
        } else {
            ((e + 0.055) / 1.055).powf(2.4)
        }
    }

    /// sRGB OETF (gamma encode)
    fn srgb_oetf(l: f32) -> f32 {
        if l <= 0.0031308 {
            l * 12.92
        } else {
            1.055 * l.powf(1.0 / 2.4) - 0.055
        }
    }

    /// PQ (Perceptual Quantizer) EOTF - SMPTE ST 2084
    fn pq_eotf(e: f32) -> f32 {
        let m1 = 0.1593017578125;
        let m2 = 78.84375;
        let c1 = 0.8359375;
        let c2 = 18.8515625;
        let c3 = 18.6875;

        let e_m2 = e.powf(1.0 / m2);
        let num = (e_m2 - c1).max(0.0);
        let den = c2 - c3 * e_m2;

        let y = (num / den).powf(1.0 / m1);
        y * 10000.0 // Scale to nits
    }

    /// PQ (Perceptual Quantizer) OETF - SMPTE ST 2084
    fn pq_oetf(l: f32) -> f32 {
        let m1 = 0.1593017578125;
        let m2 = 78.84375;
        let c1 = 0.8359375;
        let c2 = 18.8515625;
        let c3 = 18.6875;

        let y = (l / 10000.0).max(0.0); // Normalize from nits
        let y_m1 = y.powf(m1);
        let num = c1 + c2 * y_m1;
        let den = 1.0 + c3 * y_m1;

        (num / den).powf(m2)
    }

    /// HLG (Hybrid Log-Gamma) EOTF - ITU-R BT.2100
    fn hlg_eotf(e: f32) -> f32 {
        let a = 0.17883277;
        let b = 0.28466892;
        let c = 0.55991073;

        if e <= 0.5 {
            (e * e) / 3.0
        } else {
            // Correct HLG EOTF: L = (exp((E - c) / a) + b) / 12
            (((e - c) / a).exp() + b) / 12.0
        }
    }

    /// HLG (Hybrid Log-Gamma) OETF - ITU-R BT.2100
    fn hlg_oetf(l: f32) -> f32 {
        let a = 0.17883277;
        let b = 0.28466892;
        let c = 0.55991073;

        if l <= (1.0 / 12.0) {
            (3.0 * l).sqrt()
        } else {
            // Correct HLG OETF: E = a * ln(12*L - b) + c
            a * (12.0 * l - b).ln() + c
        }
    }

    /// Convert RGB with transfer function
    pub fn apply_to_rgb(rgb: RgbFloat, from: TransferFunction, to: TransferFunction) -> RgbFloat {
        // Decode to linear
        let linear = RgbFloat::new(
            Self::eotf(rgb.r, from),
            Self::eotf(rgb.g, from),
            Self::eotf(rgb.b, from),
        );

        // Encode to target
        RgbFloat::new(
            Self::oetf(linear.r, to),
            Self::oetf(linear.g, to),
            Self::oetf(linear.b, to),
        )
    }
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

        // Allow some error due to rounding (8-bit quantization can introduce errors up to 4)
        assert!((original.r as i32 - rgb.r as i32).abs() <= 4);
        assert!((original.g as i32 - rgb.g as i32).abs() <= 4);
        assert!((original.b as i32 - rgb.b as i32).abs() <= 4);
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

    #[test]
    fn test_rgb_float_conversion() {
        let rgb = Rgb::new(128, 64, 192);
        let rgb_float = RgbFloat::from_rgb(rgb);

        assert!((rgb_float.r - 128.0 / 255.0).abs() < 0.01);
        assert!((rgb_float.g - 64.0 / 255.0).abs() < 0.01);
        assert!((rgb_float.b - 192.0 / 255.0).abs() < 0.01);

        let rgb_back = rgb_float.to_rgb();
        assert_eq!(rgb_back.r, 128);
        assert_eq!(rgb_back.g, 64);
        assert_eq!(rgb_back.b, 192);
    }

    #[test]
    fn test_gamut_converter_709_to_2020() {
        let converter = GamutConverter::new(
            ColorStandard::Bt709,
            ColorStandard::Bt2020,
            GamutMapping::Clip,
        );

        let rgb709 = RgbFloat::new(0.5, 0.5, 0.5); // Gray
        let rgb2020 = converter.convert(rgb709);

        // Gray should remain approximately gray
        assert!((rgb2020.r - rgb2020.g).abs() < 0.1);
        assert!((rgb2020.g - rgb2020.b).abs() < 0.1);
    }

    #[test]
    fn test_gamut_converter_2020_to_709() {
        let converter = GamutConverter::new(
            ColorStandard::Bt2020,
            ColorStandard::Bt709,
            GamutMapping::Compress,
        );

        let rgb2020 = RgbFloat::new(1.0, 0.0, 0.0); // Pure red
        let rgb709 = converter.convert(rgb2020);

        // Should be within gamut
        assert!(rgb709.r >= 0.0 && rgb709.r <= 1.0);
        assert!(rgb709.g >= 0.0 && rgb709.g <= 1.0);
        assert!(rgb709.b >= 0.0 && rgb709.b <= 1.0);
    }

    #[test]
    fn test_gamut_mapping_clip() {
        let converter = GamutConverter::new(
            ColorStandard::Bt709,
            ColorStandard::Bt709,
            GamutMapping::Clip,
        );

        let rgb = RgbFloat::new(1.5, -0.2, 0.5);
        let mapped = converter.apply_gamut_mapping(rgb);

        assert_eq!(mapped.r, 1.0);
        assert_eq!(mapped.g, 0.0);
        assert_eq!(mapped.b, 0.5);
    }

    #[test]
    fn test_gamut_mapping_compress() {
        let converter = GamutConverter::new(
            ColorStandard::Bt709,
            ColorStandard::Bt709,
            GamutMapping::Compress,
        );

        let rgb = RgbFloat::new(2.0, 1.0, 0.5);
        let mapped = converter.compress_gamut(rgb);

        // Should scale down to fit
        assert!(mapped.r <= 1.0);
        assert!(mapped.g <= 1.0);
        assert!(mapped.b <= 1.0);

        // Should preserve ratios
        assert!((mapped.r / mapped.g - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_transfer_function_srgb() {
        let linear = 0.5;
        let srgb = TransferFunctionConverter::oetf(linear, TransferFunction::Srgb);
        let linear_back = TransferFunctionConverter::eotf(srgb, TransferFunction::Srgb);

        assert!((linear - linear_back).abs() < 0.001);
    }

    #[test]
    fn test_transfer_function_pq() {
        let linear = 100.0; // 100 nits
        let pq = TransferFunctionConverter::oetf(linear, TransferFunction::Pq);
        let linear_back = TransferFunctionConverter::eotf(pq, TransferFunction::Pq);

        assert!((linear - linear_back).abs() < 1.0);
    }

    #[test]
    fn test_transfer_function_hlg() {
        let linear = 0.5;
        let hlg = TransferFunctionConverter::oetf(linear, TransferFunction::Hlg);
        let linear_back = TransferFunctionConverter::eotf(hlg, TransferFunction::Hlg);

        assert!((linear - linear_back).abs() < 0.01);
    }

    #[test]
    fn test_transfer_function_rgb() {
        let rgb = RgbFloat::new(0.5, 0.5, 0.5);
        let srgb = TransferFunctionConverter::apply_to_rgb(
            rgb,
            TransferFunction::Linear,
            TransferFunction::Srgb,
        );
        let linear_back = TransferFunctionConverter::apply_to_rgb(
            srgb,
            TransferFunction::Srgb,
            TransferFunction::Linear,
        );

        assert!((rgb.r - linear_back.r).abs() < 0.001);
        assert!((rgb.g - linear_back.g).abs() < 0.001);
        assert!((rgb.b - linear_back.b).abs() < 0.001);
    }

    #[test]
    fn test_xyz_creation() {
        let xyz = Xyz::new(0.5, 0.5, 0.5);
        assert_eq!(xyz.x, 0.5);
        assert_eq!(xyz.y, 0.5);
        assert_eq!(xyz.z, 0.5);
    }

    #[test]
    fn test_rgb_to_xyz_to_rgb() {
        let converter = GamutConverter::new(
            ColorStandard::Bt709,
            ColorStandard::Bt709,
            GamutMapping::Clip,
        );

        let rgb = RgbFloat::new(0.5, 0.3, 0.7);
        let xyz = converter.rgb_to_xyz(rgb, ColorStandard::Bt709);
        let rgb_back = converter.xyz_to_rgb(xyz, ColorStandard::Bt709);

        assert!((rgb.r - rgb_back.r).abs() < 0.01);
        assert!((rgb.g - rgb_back.g).abs() < 0.01);
        assert!((rgb.b - rgb_back.b).abs() < 0.01);
    }

    #[test]
    fn test_out_of_gamut_detection() {
        let converter = GamutConverter::new(
            ColorStandard::Bt709,
            ColorStandard::Bt709,
            GamutMapping::Clip,
        );

        let in_gamut = RgbFloat::new(0.5, 0.5, 0.5);
        let out_of_gamut = RgbFloat::new(1.5, 0.5, -0.1);

        assert!(!converter.is_out_of_gamut(&in_gamut));
        assert!(converter.is_out_of_gamut(&out_of_gamut));
    }

    #[test]
    fn test_adaptive_gamut_mapping() {
        let converter = GamutConverter::new(
            ColorStandard::Bt709,
            ColorStandard::Bt709,
            GamutMapping::Adaptive,
        );

        let rgb = RgbFloat::new(1.2, 0.5, -0.1);
        let mapped = converter.adaptive_gamut_map(rgb);

        // Should soft-clip to valid range
        assert!(mapped.r >= 0.0 && mapped.r <= 1.1); // Allows slight overshoot
        assert!(mapped.g >= 0.0 && mapped.g <= 1.0);
        assert!(mapped.b >= 0.0);
    }
}
