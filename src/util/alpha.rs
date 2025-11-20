//! Alpha Channel and Transparency Processing
//!
//! Handle RGBA images, alpha compositing, premultiplied alpha, and
//! transparency operations for professional video compositing workflows.
//!
//! ## Alpha Types
//!
//! - **Straight Alpha**: RGB values independent of alpha
//! - **Premultiplied Alpha**: RGB values pre-multiplied by alpha
//! - **Unassociated Alpha**: Straight alpha (standard)
//! - **Associated Alpha**: Premultiplied alpha (compositing-optimized)
//!
//! ## Compositing Operations
//!
//! - **Over**: Standard alpha blending (src over dst)
//! - **Add**: Additive compositing
//! - **Multiply**: Multiplicative compositing
//! - **Screen**: Inverse multiply
//!
//! ## Usage
//!
//! ```rust
//! use zvd_lib::util::alpha::{AlphaCompositor, CompositeOp, AlphaType};
//!
//! let compositor = AlphaCompositor::new(CompositeOp::Over);
//!
//! // Composite foreground over background
//! compositor.composite(&foreground_rgba, &background_rgba, &mut output_rgba, width, height)?;
//! ```

use crate::error::{Error, Result};

/// Alpha channel type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaType {
    /// Straight alpha (unassociated)
    Straight,
    /// Premultiplied alpha (associated)
    Premultiplied,
}

/// Composite operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeOp {
    /// Porter-Duff Over (src over dst)
    Over,
    /// Additive
    Add,
    /// Multiplicative
    Multiply,
    /// Screen (inverse multiply)
    Screen,
    /// Source (replace)
    Source,
}

/// RGBA frame
#[derive(Debug, Clone)]
pub struct RgbaFrame {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>, // RGBA interleaved
    pub alpha_type: AlphaType,
}

impl RgbaFrame {
    /// Create new RGBA frame
    pub fn new(width: usize, height: usize, alpha_type: AlphaType) -> Self {
        RgbaFrame {
            width,
            height,
            data: vec![0u8; width * height * 4],
            alpha_type,
        }
    }

    /// Convert from RGB (add opaque alpha)
    pub fn from_rgb(rgb_data: &[u8], width: usize, height: usize) -> Result<Self> {
        if rgb_data.len() != width * height * 3 {
            return Err(Error::InvalidInput("Invalid RGB data size".to_string()));
        }

        let mut rgba = RgbaFrame::new(width, height, AlphaType::Straight);

        for (i, rgb) in rgb_data.chunks_exact(3).enumerate() {
            rgba.data[i * 4] = rgb[0];
            rgba.data[i * 4 + 1] = rgb[1];
            rgba.data[i * 4 + 2] = rgb[2];
            rgba.data[i * 4 + 3] = 255; // Opaque
        }

        Ok(rgba)
    }

    /// Convert to RGB (discard alpha)
    pub fn to_rgb(&self) -> Vec<u8> {
        let mut rgb = Vec::with_capacity(self.width * self.height * 3);

        for rgba in self.data.chunks_exact(4) {
            rgb.push(rgba[0]);
            rgb.push(rgba[1]);
            rgb.push(rgba[2]);
        }

        rgb
    }

    /// Convert to premultiplied alpha
    pub fn to_premultiplied(&mut self) {
        if self.alpha_type == AlphaType::Premultiplied {
            return; // Already premultiplied
        }

        for rgba in self.data.chunks_exact_mut(4) {
            let alpha = rgba[3] as f32 / 255.0;
            rgba[0] = (rgba[0] as f32 * alpha) as u8;
            rgba[1] = (rgba[1] as f32 * alpha) as u8;
            rgba[2] = (rgba[2] as f32 * alpha) as u8;
        }

        self.alpha_type = AlphaType::Premultiplied;
    }

    /// Convert to straight alpha
    pub fn to_straight(&mut self) {
        if self.alpha_type == AlphaType::Straight {
            return; // Already straight
        }

        for rgba in self.data.chunks_exact_mut(4) {
            let alpha = rgba[3] as f32 / 255.0;

            if alpha > 0.0 {
                rgba[0] = (rgba[0] as f32 / alpha).min(255.0) as u8;
                rgba[1] = (rgba[1] as f32 / alpha).min(255.0) as u8;
                rgba[2] = (rgba[2] as f32 / alpha).min(255.0) as u8;
            }
        }

        self.alpha_type = AlphaType::Straight;
    }
}

/// Alpha compositor
pub struct AlphaCompositor {
    operation: CompositeOp,
}

impl AlphaCompositor {
    /// Create new compositor
    pub fn new(operation: CompositeOp) -> Self {
        AlphaCompositor { operation }
    }

    /// Composite foreground over background
    pub fn composite(
        &self,
        foreground: &RgbaFrame,
        background: &RgbaFrame,
        output: &mut RgbaFrame,
    ) -> Result<()> {
        if foreground.width != background.width
            || foreground.height != background.height
            || foreground.width != output.width
            || foreground.height != output.height
        {
            return Err(Error::InvalidInput(
                "Frame dimensions must match".to_string(),
            ));
        }

        // Ensure both frames use same alpha type for compositing
        let mut fg = foreground.clone();
        let mut bg = background.clone();

        fg.to_premultiplied();
        bg.to_premultiplied();

        match self.operation {
            CompositeOp::Over => self.composite_over(&fg, &bg, output),
            CompositeOp::Add => self.composite_add(&fg, &bg, output),
            CompositeOp::Multiply => self.composite_multiply(&fg, &bg, output),
            CompositeOp::Screen => self.composite_screen(&fg, &bg, output),
            CompositeOp::Source => self.composite_source(&fg, output),
        }

        output.alpha_type = AlphaType::Premultiplied;
        Ok(())
    }

    /// Porter-Duff Over: fg over bg
    fn composite_over(&self, fg: &RgbaFrame, bg: &RgbaFrame, output: &mut RgbaFrame) {
        for i in 0..fg.data.len() / 4 {
            let idx = i * 4;

            let fa = fg.data[idx + 3] as f32 / 255.0;
            let ba = bg.data[idx + 3] as f32 / 255.0;

            // Alpha compositing
            let out_a = fa + ba * (1.0 - fa);

            if out_a > 0.0 {
                for c in 0..3 {
                    let fg_val = fg.data[idx + c] as f32;
                    let bg_val = bg.data[idx + c] as f32;

                    // Porter-Duff Over formula
                    let out_val = (fg_val + bg_val * (1.0 - fa)) / out_a;

                    output.data[idx + c] = (out_val * out_a) as u8;
                }
            } else {
                output.data[idx] = 0;
                output.data[idx + 1] = 0;
                output.data[idx + 2] = 0;
            }

            output.data[idx + 3] = (out_a * 255.0) as u8;
        }
    }

    /// Additive compositing
    fn composite_add(&self, fg: &RgbaFrame, bg: &RgbaFrame, output: &mut RgbaFrame) {
        for i in 0..fg.data.len() / 4 {
            let idx = i * 4;

            for c in 0..3 {
                output.data[idx + c] = (fg.data[idx + c] as u16 + bg.data[idx + c] as u16)
                    .min(255) as u8;
            }

            output.data[idx + 3] = ((fg.data[idx + 3] as u16 + bg.data[idx + 3] as u16).min(255)) as u8;
        }
    }

    /// Multiplicative compositing
    fn composite_multiply(&self, fg: &RgbaFrame, bg: &RgbaFrame, output: &mut RgbaFrame) {
        for i in 0..fg.data.len() / 4 {
            let idx = i * 4;

            for c in 0..3 {
                let fg_val = fg.data[idx + c] as f32 / 255.0;
                let bg_val = bg.data[idx + c] as f32 / 255.0;
                output.data[idx + c] = (fg_val * bg_val * 255.0) as u8;
            }

            let fa = fg.data[idx + 3] as f32 / 255.0;
            let ba = bg.data[idx + 3] as f32 / 255.0;
            output.data[idx + 3] = (fa * ba * 255.0) as u8;
        }
    }

    /// Screen compositing
    fn composite_screen(&self, fg: &RgbaFrame, bg: &RgbaFrame, output: &mut RgbaFrame) {
        for i in 0..fg.data.len() / 4 {
            let idx = i * 4;

            for c in 0..3 {
                let fg_val = fg.data[idx + c] as f32 / 255.0;
                let bg_val = bg.data[idx + c] as f32 / 255.0;
                let screen = 1.0 - (1.0 - fg_val) * (1.0 - bg_val);
                output.data[idx + c] = (screen * 255.0) as u8;
            }

            output.data[idx + 3] = fg.data[idx + 3].max(bg.data[idx + 3]);
        }
    }

    /// Source (replace)
    fn composite_source(&self, fg: &RgbaFrame, output: &mut RgbaFrame) {
        output.data.copy_from_slice(&fg.data);
    }
}

/// Alpha mask generator
pub struct AlphaMask;

impl AlphaMask {
    /// Create alpha mask from luminance
    pub fn from_luminance(rgb_data: &[u8]) -> Vec<u8> {
        let mut alpha = Vec::new();

        for rgb in rgb_data.chunks_exact(3) {
            // Calculate luma
            let luma = (0.2126 * rgb[0] as f32 + 0.7152 * rgb[1] as f32 + 0.0722 * rgb[2] as f32) as u8;
            alpha.push(luma);
        }

        alpha
    }

    /// Create alpha mask from specific color key (chroma key)
    pub fn from_color_key(rgb_data: &[u8], key_color: [u8; 3], tolerance: u8) -> Vec<u8> {
        let mut alpha = Vec::new();

        for rgb in rgb_data.chunks_exact(3) {
            // Calculate color distance
            let dr = (rgb[0] as i32 - key_color[0] as i32).abs();
            let dg = (rgb[1] as i32 - key_color[1] as i32).abs();
            let db = (rgb[2] as i32 - key_color[2] as i32).abs();

            let distance = (dr + dg + db) as u8;

            if distance <= tolerance {
                alpha.push(0); // Transparent
            } else {
                alpha.push(255); // Opaque
            }
        }

        alpha
    }

    /// Feather alpha mask (soft edges)
    pub fn feather(alpha_data: &mut [u8], width: usize, _height: usize, radius: usize) {
        // Simple box blur for feathering
        let mut blurred = alpha_data.to_vec();

        for y in radius..(_height - radius) {
            for x in radius..(width - radius) {
                let mut sum = 0u32;
                let mut count = 0u32;

                for dy in -(radius as i32)..=(radius as i32) {
                    for dx in -(radius as i32)..=(radius as i32) {
                        let px = (x as i32 + dx) as usize;
                        let py = (y as i32 + dy) as usize;
                        let idx = py * width + px;

                        sum += alpha_data[idx] as u32;
                        count += 1;
                    }
                }

                blurred[y * width + x] = (sum / count) as u8;
            }
        }

        alpha_data.copy_from_slice(&blurred);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgba_frame_creation() {
        let frame = RgbaFrame::new(64, 64, AlphaType::Straight);
        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 64);
        assert_eq!(frame.data.len(), 64 * 64 * 4);
        assert_eq!(frame.alpha_type, AlphaType::Straight);
    }

    #[test]
    fn test_rgba_from_rgb() {
        let rgb = vec![255u8, 0, 0].repeat(10); // 10 red pixels
        let rgba = RgbaFrame::from_rgb(&rgb, 10, 1).unwrap();

        assert_eq!(rgba.width, 10);
        assert_eq!(rgba.data[0], 255); // R
        assert_eq!(rgba.data[1], 0);   // G
        assert_eq!(rgba.data[2], 0);   // B
        assert_eq!(rgba.data[3], 255); // A (opaque)
    }

    #[test]
    fn test_rgba_to_rgb() {
        let mut rgba = RgbaFrame::new(10, 1, AlphaType::Straight);
        rgba.data[0] = 255; // R
        rgba.data[1] = 128; // G
        rgba.data[2] = 64;  // B
        rgba.data[3] = 200; // A

        let rgb = rgba.to_rgb();

        assert_eq!(rgb.len(), 10 * 3);
        assert_eq!(rgb[0], 255);
        assert_eq!(rgb[1], 128);
        assert_eq!(rgb[2], 64);
    }

    #[test]
    fn test_premultiplied_conversion() {
        let mut frame = RgbaFrame::new(1, 1, AlphaType::Straight);
        frame.data[0] = 255; // R
        frame.data[1] = 255; // G
        frame.data[2] = 255; // B
        frame.data[3] = 128; // A (50%)

        frame.to_premultiplied();

        assert_eq!(frame.alpha_type, AlphaType::Premultiplied);
        assert!(frame.data[0] < 255); // Should be multiplied by alpha
        assert!(frame.data[1] < 255);
        assert!(frame.data[2] < 255);
        assert_eq!(frame.data[3], 128); // Alpha unchanged
    }

    #[test]
    fn test_straight_conversion() {
        let mut frame = RgbaFrame::new(1, 1, AlphaType::Premultiplied);
        frame.data[0] = 128; // R (premultiplied)
        frame.data[1] = 128; // G
        frame.data[2] = 128; // B
        frame.data[3] = 128; // A (50%)

        frame.to_straight();

        assert_eq!(frame.alpha_type, AlphaType::Straight);
        assert!(frame.data[0] > 128); // Should be divided by alpha
    }

    #[test]
    fn test_composite_over() {
        let mut fg = RgbaFrame::new(1, 1, AlphaType::Straight);
        fg.data[0] = 255; // Red
        fg.data[3] = 128; // 50% alpha

        let bg = RgbaFrame::new(1, 1, AlphaType::Straight);
        // Black background

        let mut output = RgbaFrame::new(1, 1, AlphaType::Straight);

        let compositor = AlphaCompositor::new(CompositeOp::Over);
        compositor.composite(&fg, &bg, &mut output).unwrap();

        assert!(output.data[3] > 0); // Some alpha
    }

    #[test]
    fn test_composite_add() {
        let mut fg = RgbaFrame::new(1, 1, AlphaType::Straight);
        fg.data[0] = 100;

        let mut bg = RgbaFrame::new(1, 1, AlphaType::Straight);
        bg.data[0] = 100;

        let mut output = RgbaFrame::new(1, 1, AlphaType::Straight);

        let compositor = AlphaCompositor::new(CompositeOp::Add);
        compositor.composite(&fg, &bg, &mut output).unwrap();

        assert_eq!(output.data[0], 200); // 100 + 100
    }

    #[test]
    fn test_alpha_mask_from_luminance() {
        let rgb = vec![255u8, 255, 255, 0, 0, 0]; // White, Black
        let alpha = AlphaMask::from_luminance(&rgb);

        assert_eq!(alpha.len(), 2);
        assert_eq!(alpha[0], 255); // White -> opaque
        assert_eq!(alpha[1], 0);   // Black -> transparent
    }

    #[test]
    fn test_alpha_mask_from_color_key() {
        let rgb = vec![0u8, 255, 0, 255, 0, 0]; // Green, Red
        let alpha = AlphaMask::from_color_key(&rgb, [0, 255, 0], 10);

        assert_eq!(alpha.len(), 2);
        assert_eq!(alpha[0], 0);   // Green -> transparent (keyed out)
        assert_eq!(alpha[1], 255); // Red -> opaque
    }

    #[test]
    fn test_compositor_creation() {
        let compositor = AlphaCompositor::new(CompositeOp::Over);
        assert_eq!(compositor.operation, CompositeOp::Over);
    }
}
