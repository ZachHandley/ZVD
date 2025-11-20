//! Aspect Ratio Conversion
//!
//! Convert between different aspect ratios with various fitting modes for
//! video display, broadcasting, and content delivery.
//!
//! ## Common Aspect Ratios
//!
//! - **4:3** (1.33:1): Standard Definition TV, vintage content
//! - **16:9** (1.78:1): HD/4K TV, most modern content
//! - **21:9** (2.35:1): Cinematic widescreen
//! - **1:1**: Square (social media)
//! - **9:16**: Vertical (mobile, stories)
//!
//! ## Conversion Modes
//!
//! - **Letterbox**: Add black bars top/bottom (preserve full width)
//! - **Pillarbox**: Add black bars left/right (preserve full height)
//! - **Crop**: Cut edges to fit (may lose content)
//! - **Stretch**: Distort to fit (changes proportions)
//! - **Fit**: Scale to fit inside target (maintains aspect)
//!
//! ## Usage
//!
//! ```rust
//! use zvd_lib::util::aspect::{AspectRatioConverter, AspectRatio, FitMode};
//!
//! let converter = AspectRatioConverter::new(
//!     AspectRatio::new(16, 9),
//!     AspectRatio::new(4, 3),
//!     FitMode::Letterbox,
//! );
//!
//! let converted = converter.convert(&frame_data, src_width, src_height)?;
//! ```

use crate::error::{Error, Result};

/// Aspect ratio
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AspectRatio {
    /// Numerator (width)
    pub width: u32,
    /// Denominator (height)
    pub height: u32,
}

impl AspectRatio {
    /// Create new aspect ratio
    pub fn new(width: u32, height: u32) -> Self {
        let gcd = Self::gcd(width, height);
        AspectRatio {
            width: width / gcd,
            height: height / gcd,
        }
    }

    /// Common aspect ratios
    pub fn ratio_4_3() -> Self {
        AspectRatio {
            width: 4,
            height: 3,
        }
    }

    pub fn ratio_16_9() -> Self {
        AspectRatio {
            width: 16,
            height: 9,
        }
    }

    pub fn ratio_21_9() -> Self {
        AspectRatio {
            width: 21,
            height: 9,
        }
    }

    pub fn ratio_1_1() -> Self {
        AspectRatio {
            width: 1,
            height: 1,
        }
    }

    pub fn ratio_9_16() -> Self {
        AspectRatio {
            width: 9,
            height: 16,
        }
    }

    /// Calculate decimal ratio
    pub fn as_f64(&self) -> f64 {
        self.width as f64 / self.height as f64
    }

    /// Create from decimal ratio (approximate)
    pub fn from_f64(ratio: f64) -> Self {
        // Common ratios
        if (ratio - 1.333).abs() < 0.01 {
            return Self::ratio_4_3();
        }
        if (ratio - 1.778).abs() < 0.01 {
            return Self::ratio_16_9();
        }
        if (ratio - 2.35).abs() < 0.01 {
            return Self::ratio_21_9();
        }
        if (ratio - 1.0).abs() < 0.01 {
            return Self::ratio_1_1();
        }

        // Approximate with 1000 denominator
        let height = 1000;
        let width = (ratio * height as f64) as u32;
        Self::new(width, height)
    }

    /// GCD for simplification
    fn gcd(mut a: u32, mut b: u32) -> u32 {
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a
    }
}

/// Fit mode for aspect ratio conversion
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FitMode {
    /// Add black bars (letterbox/pillarbox)
    Fit,
    /// Crop to fill
    Fill,
    /// Stretch (distort)
    Stretch,
    /// Letterbox (horizontal bars)
    Letterbox,
    /// Pillarbox (vertical bars)
    Pillarbox,
}

/// Aspect ratio converter
pub struct AspectRatioConverter {
    /// Source aspect ratio
    source: AspectRatio,
    /// Target aspect ratio
    target: AspectRatio,
    /// Fit mode
    mode: FitMode,
    /// Bar color (RGB)
    bar_color: [u8; 3],
}

impl AspectRatioConverter {
    /// Create new aspect ratio converter
    pub fn new(source: AspectRatio, target: AspectRatio, mode: FitMode) -> Self {
        AspectRatioConverter {
            source,
            target,
            mode,
            bar_color: [0, 0, 0], // Black
        }
    }

    /// Set bar color for letterbox/pillarbox
    pub fn set_bar_color(&mut self, color: [u8; 3]) {
        self.bar_color = color;
    }

    /// Convert frame to target aspect ratio
    pub fn convert(
        &self,
        src_rgb: &[u8],
        src_width: usize,
        src_height: usize,
        target_width: usize,
        target_height: usize,
    ) -> Result<Vec<u8>> {
        if src_rgb.len() != src_width * src_height * 3 {
            return Err(Error::InvalidInput(format!(
                "Invalid source dimensions: {}x{} != {} bytes",
                src_width,
                src_height,
                src_rgb.len()
            )));
        }

        match self.mode {
            FitMode::Fit => {
                self.convert_fit(src_rgb, src_width, src_height, target_width, target_height)
            }
            FitMode::Fill => {
                self.convert_fill(src_rgb, src_width, src_height, target_width, target_height)
            }
            FitMode::Stretch => {
                self.convert_stretch(src_rgb, src_width, src_height, target_width, target_height)
            }
            FitMode::Letterbox => {
                self.convert_letterbox(src_rgb, src_width, src_height, target_width, target_height)
            }
            FitMode::Pillarbox => {
                self.convert_pillarbox(src_rgb, src_width, src_height, target_width, target_height)
            }
        }
    }

    /// Fit mode: scale to fit inside target, add bars
    fn convert_fit(
        &self,
        src_rgb: &[u8],
        src_width: usize,
        src_height: usize,
        target_width: usize,
        target_height: usize,
    ) -> Result<Vec<u8>> {
        let src_ratio = src_width as f64 / src_height as f64;
        let target_ratio = target_width as f64 / target_height as f64;

        if src_ratio > target_ratio {
            // Letterbox (horizontal bars)
            self.convert_letterbox(src_rgb, src_width, src_height, target_width, target_height)
        } else {
            // Pillarbox (vertical bars)
            self.convert_pillarbox(src_rgb, src_width, src_height, target_width, target_height)
        }
    }

    /// Letterbox: scale width to fit, add bars top/bottom
    fn convert_letterbox(
        &self,
        src_rgb: &[u8],
        src_width: usize,
        src_height: usize,
        target_width: usize,
        target_height: usize,
    ) -> Result<Vec<u8>> {
        // Scale to target width
        let scale = target_width as f64 / src_width as f64;
        let scaled_height = (src_height as f64 * scale) as usize;

        if scaled_height > target_height {
            // Fallback to pillarbox
            return self.convert_pillarbox(
                src_rgb,
                src_width,
                src_height,
                target_width,
                target_height,
            );
        }

        // Create output buffer with bars
        let mut output = vec![0u8; target_width * target_height * 3];

        // Fill with bar color
        for pixel in output.chunks_exact_mut(3) {
            pixel.copy_from_slice(&self.bar_color);
        }

        // Calculate vertical offset (center)
        let y_offset = (target_height - scaled_height) / 2;

        // Scale and copy image
        let scaled =
            self.bilinear_scale(src_rgb, src_width, src_height, target_width, scaled_height)?;

        // Copy scaled image to center
        for y in 0..scaled_height {
            let dst_y = y + y_offset;
            let src_row_start = y * target_width * 3;
            let dst_row_start = dst_y * target_width * 3;

            output[dst_row_start..dst_row_start + target_width * 3]
                .copy_from_slice(&scaled[src_row_start..src_row_start + target_width * 3]);
        }

        Ok(output)
    }

    /// Pillarbox: scale height to fit, add bars left/right
    fn convert_pillarbox(
        &self,
        src_rgb: &[u8],
        src_width: usize,
        src_height: usize,
        target_width: usize,
        target_height: usize,
    ) -> Result<Vec<u8>> {
        // Scale to target height
        let scale = target_height as f64 / src_height as f64;
        let scaled_width = (src_width as f64 * scale) as usize;

        if scaled_width > target_width {
            // Fallback to letterbox
            return self.convert_letterbox(
                src_rgb,
                src_width,
                src_height,
                target_width,
                target_height,
            );
        }

        // Create output buffer with bars
        let mut output = vec![0u8; target_width * target_height * 3];

        // Fill with bar color
        for pixel in output.chunks_exact_mut(3) {
            pixel.copy_from_slice(&self.bar_color);
        }

        // Calculate horizontal offset (center)
        let x_offset = (target_width - scaled_width) / 2;

        // Scale image
        let scaled =
            self.bilinear_scale(src_rgb, src_width, src_height, scaled_width, target_height)?;

        // Copy scaled image to center
        for y in 0..target_height {
            for x in 0..scaled_width {
                let src_idx = (y * scaled_width + x) * 3;
                let dst_idx = (y * target_width + (x + x_offset)) * 3;

                output[dst_idx..dst_idx + 3].copy_from_slice(&scaled[src_idx..src_idx + 3]);
            }
        }

        Ok(output)
    }

    /// Fill mode: crop to fill target
    fn convert_fill(
        &self,
        src_rgb: &[u8],
        src_width: usize,
        src_height: usize,
        target_width: usize,
        target_height: usize,
    ) -> Result<Vec<u8>> {
        let src_ratio = src_width as f64 / src_height as f64;
        let target_ratio = target_width as f64 / target_height as f64;

        if src_ratio > target_ratio {
            // Crop sides
            let scale = target_height as f64 / src_height as f64;
            let scaled_width = (src_width as f64 * scale) as usize;

            let scaled =
                self.bilinear_scale(src_rgb, src_width, src_height, scaled_width, target_height)?;

            // Crop horizontally
            let x_offset = (scaled_width - target_width) / 2;
            let mut output = vec![0u8; target_width * target_height * 3];

            for y in 0..target_height {
                let src_row_start = (y * scaled_width + x_offset) * 3;
                let dst_row_start = y * target_width * 3;

                output[dst_row_start..dst_row_start + target_width * 3]
                    .copy_from_slice(&scaled[src_row_start..src_row_start + target_width * 3]);
            }

            Ok(output)
        } else {
            // Crop top/bottom
            let scale = target_width as f64 / src_width as f64;
            let scaled_height = (src_height as f64 * scale) as usize;

            let scaled =
                self.bilinear_scale(src_rgb, src_width, src_height, target_width, scaled_height)?;

            // Crop vertically
            let y_offset = (scaled_height - target_height) / 2;
            let mut output = vec![0u8; target_width * target_height * 3];

            for y in 0..target_height {
                let src_row_start = ((y + y_offset) * target_width) * 3;
                let dst_row_start = y * target_width * 3;

                output[dst_row_start..dst_row_start + target_width * 3]
                    .copy_from_slice(&scaled[src_row_start..src_row_start + target_width * 3]);
            }

            Ok(output)
        }
    }

    /// Stretch mode: distort to fit
    fn convert_stretch(
        &self,
        src_rgb: &[u8],
        src_width: usize,
        src_height: usize,
        target_width: usize,
        target_height: usize,
    ) -> Result<Vec<u8>> {
        self.bilinear_scale(src_rgb, src_width, src_height, target_width, target_height)
    }

    /// Bilinear scaling
    fn bilinear_scale(
        &self,
        src_rgb: &[u8],
        src_width: usize,
        src_height: usize,
        dst_width: usize,
        dst_height: usize,
    ) -> Result<Vec<u8>> {
        let mut output = vec![0u8; dst_width * dst_height * 3];

        let x_ratio = src_width as f32 / dst_width as f32;
        let y_ratio = src_height as f32 / dst_height as f32;

        for dst_y in 0..dst_height {
            for dst_x in 0..dst_width {
                let src_x = dst_x as f32 * x_ratio;
                let src_y = dst_y as f32 * y_ratio;

                let x0 = src_x.floor() as usize;
                let y0 = src_y.floor() as usize;
                let x1 = (x0 + 1).min(src_width - 1);
                let y1 = (y0 + 1).min(src_height - 1);

                let fx = src_x - x0 as f32;
                let fy = src_y - y0 as f32;

                for c in 0..3 {
                    let p00 = src_rgb[(y0 * src_width + x0) * 3 + c] as f32;
                    let p10 = src_rgb[(y0 * src_width + x1) * 3 + c] as f32;
                    let p01 = src_rgb[(y1 * src_width + x0) * 3 + c] as f32;
                    let p11 = src_rgb[(y1 * src_width + x1) * 3 + c] as f32;

                    let val = p00 * (1.0 - fx) * (1.0 - fy)
                        + p10 * fx * (1.0 - fy)
                        + p01 * (1.0 - fx) * fy
                        + p11 * fx * fy;

                    output[(dst_y * dst_width + dst_x) * 3 + c] = val.round() as u8;
                }
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aspect_ratio_creation() {
        let ar = AspectRatio::new(16, 9);
        assert_eq!(ar.width, 16);
        assert_eq!(ar.height, 9);

        let ar2 = AspectRatio::new(32, 18);
        assert_eq!(ar2.width, 16); // Simplified
        assert_eq!(ar2.height, 9);
    }

    #[test]
    fn test_common_ratios() {
        let ar_4_3 = AspectRatio::ratio_4_3();
        assert_eq!(ar_4_3.as_f64(), 4.0 / 3.0);

        let ar_16_9 = AspectRatio::ratio_16_9();
        assert_eq!(ar_16_9.as_f64(), 16.0 / 9.0);

        let ar_21_9 = AspectRatio::ratio_21_9();
        assert_eq!(ar_21_9.as_f64(), 21.0 / 9.0);
    }

    #[test]
    fn test_from_f64() {
        let ar = AspectRatio::from_f64(1.778);
        assert_eq!(ar, AspectRatio::ratio_16_9());

        let ar2 = AspectRatio::from_f64(1.333);
        assert_eq!(ar2, AspectRatio::ratio_4_3());
    }

    #[test]
    fn test_converter_creation() {
        let converter = AspectRatioConverter::new(
            AspectRatio::ratio_16_9(),
            AspectRatio::ratio_4_3(),
            FitMode::Letterbox,
        );

        assert_eq!(converter.source, AspectRatio::ratio_16_9());
        assert_eq!(converter.target, AspectRatio::ratio_4_3());
    }

    #[test]
    fn test_letterbox() {
        let converter = AspectRatioConverter::new(
            AspectRatio::ratio_16_9(),
            AspectRatio::ratio_4_3(),
            FitMode::Letterbox,
        );

        // 16:9 source (1920x1080) → 4:3 target (640x480)
        let src = vec![128u8; 1920 * 1080 * 3];
        let result = converter.convert(&src, 1920, 1080, 640, 480);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 640 * 480 * 3);
    }

    #[test]
    fn test_pillarbox() {
        let converter = AspectRatioConverter::new(
            AspectRatio::ratio_4_3(),
            AspectRatio::ratio_16_9(),
            FitMode::Pillarbox,
        );

        // 4:3 source (640x480) → 16:9 target (1920x1080)
        let src = vec![128u8; 640 * 480 * 3];
        let result = converter.convert(&src, 640, 480, 1920, 1080);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 1920 * 1080 * 3);
    }

    #[test]
    fn test_stretch() {
        let converter = AspectRatioConverter::new(
            AspectRatio::ratio_4_3(),
            AspectRatio::ratio_16_9(),
            FitMode::Stretch,
        );

        let src = vec![128u8; 640 * 480 * 3];
        let result = converter.convert(&src, 640, 480, 1920, 1080);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 1920 * 1080 * 3);
    }

    #[test]
    fn test_fill_crop() {
        let converter = AspectRatioConverter::new(
            AspectRatio::ratio_16_9(),
            AspectRatio::ratio_1_1(),
            FitMode::Fill,
        );

        let src = vec![128u8; 1920 * 1080 * 3];
        let result = converter.convert(&src, 1920, 1080, 1080, 1080);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 1080 * 1080 * 3);
    }

    #[test]
    fn test_bar_color() {
        let mut converter = AspectRatioConverter::new(
            AspectRatio::ratio_16_9(),
            AspectRatio::ratio_4_3(),
            FitMode::Letterbox,
        );

        converter.set_bar_color([255, 0, 0]); // Red bars
        assert_eq!(converter.bar_color, [255, 0, 0]);
    }

    #[test]
    fn test_bilinear_scale() {
        let converter = AspectRatioConverter::new(
            AspectRatio::ratio_16_9(),
            AspectRatio::ratio_16_9(),
            FitMode::Stretch,
        );

        let src = vec![128u8; 100 * 100 * 3];
        let result = converter.bilinear_scale(&src, 100, 100, 50, 50);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 50 * 50 * 3);
    }

    #[test]
    fn test_fit_mode() {
        let converter = AspectRatioConverter::new(
            AspectRatio::ratio_16_9(),
            AspectRatio::ratio_4_3(),
            FitMode::Fit,
        );

        let src = vec![128u8; 1920 * 1080 * 3];
        let result = converter.convert(&src, 1920, 1080, 640, 480);

        assert!(result.is_ok());
    }

    #[test]
    fn test_gcd() {
        assert_eq!(AspectRatio::gcd(16, 9), 1);
        assert_eq!(AspectRatio::gcd(32, 18), 2);
        assert_eq!(AspectRatio::gcd(1920, 1080), 120);
    }
}
