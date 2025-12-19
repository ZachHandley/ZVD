//! Video and Image Watermarking
//!
//! Add visible and invisible watermarks for copyright protection,
//! content tracking, and brand identification.
//!
//! ## Watermark Types
//!
//! - **Visible**: Logo, text overlay (like burn-in)
//! - **Invisible**: LSB steganography for forensic tracking
//! - **Semi-Transparent**: Overlay with alpha blending
//! - **Tiled**: Repeated pattern across frame
//!
//! ## Use Cases
//!
//! - Copyright protection
//! - Content tracking and attribution
//! - Brand identification
//! - Proof of ownership
//! - Leak source identification
//!
//! ## Usage
//!
//! ```rust
//! use zvd_lib::util::watermark::{Watermark, WatermarkPosition, WatermarkStyle};
//!
//! // Visible watermark
//! let watermark = Watermark::new_visible(
//!     "© 2024 MyCompany",
//!     WatermarkPosition::BottomRight,
//!     WatermarkStyle::SemiTransparent { opacity: 0.5 },
//! );
//! watermark.apply(&mut frame)?;
//!
//! // Invisible watermark
//! let invisible = Watermark::new_invisible("tracking-id-12345");
//! invisible.embed(&mut frame)?;
//! ```

use crate::error::{Error, Result};

/// Watermark position on frame
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatermarkPosition {
    TopLeft,
    TopCenter,
    TopRight,
    CenterLeft,
    Center,
    CenterRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
    Tiled, // Repeated across entire frame
    Custom { x: i32, y: i32 },
}

impl WatermarkPosition {
    /// Calculate pixel position
    pub fn calculate_position(
        &self,
        frame_width: usize,
        frame_height: usize,
        watermark_width: usize,
        watermark_height: usize,
    ) -> (i32, i32) {
        let margin = 10;

        match self {
            WatermarkPosition::TopLeft => (margin, margin),
            WatermarkPosition::TopCenter => {
                ((frame_width as i32 - watermark_width as i32) / 2, margin)
            }
            WatermarkPosition::TopRight => {
                (frame_width as i32 - watermark_width as i32 - margin, margin)
            }
            WatermarkPosition::CenterLeft => {
                (margin, (frame_height as i32 - watermark_height as i32) / 2)
            }
            WatermarkPosition::Center => (
                (frame_width as i32 - watermark_width as i32) / 2,
                (frame_height as i32 - watermark_height as i32) / 2,
            ),
            WatermarkPosition::CenterRight => (
                frame_width as i32 - watermark_width as i32 - margin,
                (frame_height as i32 - watermark_height as i32) / 2,
            ),
            WatermarkPosition::BottomLeft => (
                margin,
                frame_height as i32 - watermark_height as i32 - margin,
            ),
            WatermarkPosition::BottomCenter => (
                (frame_width as i32 - watermark_width as i32) / 2,
                frame_height as i32 - watermark_height as i32 - margin,
            ),
            WatermarkPosition::BottomRight => (
                frame_width as i32 - watermark_width as i32 - margin,
                frame_height as i32 - watermark_height as i32 - margin,
            ),
            WatermarkPosition::Tiled | WatermarkPosition::Custom { .. } => (0, 0),
        }
    }
}

/// Watermark style
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WatermarkStyle {
    /// Fully opaque
    Opaque,
    /// Semi-transparent with alpha
    SemiTransparent { opacity: f32 },
    /// Additive blending
    Additive { strength: f32 },
}

/// Watermark type
#[derive(Debug, Clone, PartialEq)]
pub enum WatermarkType {
    /// Visible text watermark
    Text { content: String },
    /// Visible image watermark
    Image {
        data: Vec<u8>,
        width: usize,
        height: usize,
    },
    /// Invisible LSB steganography
    Invisible { payload: Vec<u8> },
}

/// Watermark configuration
pub struct Watermark {
    watermark_type: WatermarkType,
    position: WatermarkPosition,
    style: WatermarkStyle,
}

impl Watermark {
    /// Create visible text watermark
    pub fn new_visible_text(
        text: &str,
        position: WatermarkPosition,
        style: WatermarkStyle,
    ) -> Self {
        Watermark {
            watermark_type: WatermarkType::Text {
                content: text.to_string(),
            },
            position,
            style,
        }
    }

    /// Create visible image watermark
    pub fn new_visible_image(
        image_data: Vec<u8>,
        width: usize,
        height: usize,
        position: WatermarkPosition,
        style: WatermarkStyle,
    ) -> Self {
        Watermark {
            watermark_type: WatermarkType::Image {
                data: image_data,
                width,
                height,
            },
            position,
            style,
        }
    }

    /// Create invisible watermark
    pub fn new_invisible(payload: &str) -> Self {
        Watermark {
            watermark_type: WatermarkType::Invisible {
                payload: payload.as_bytes().to_vec(),
            },
            position: WatermarkPosition::Custom { x: 0, y: 0 },
            style: WatermarkStyle::Opaque,
        }
    }

    /// Apply watermark to RGB frame
    pub fn apply(&self, frame_rgb: &mut [u8], width: usize, height: usize) -> Result<()> {
        match &self.watermark_type {
            WatermarkType::Text { content } => {
                self.apply_text_watermark(frame_rgb, width, height, content)
            }
            WatermarkType::Image {
                data,
                width: wm_width,
                height: wm_height,
            } => self.apply_image_watermark(frame_rgb, width, height, data, *wm_width, *wm_height),
            WatermarkType::Invisible { payload } => {
                self.embed_invisible(frame_rgb, width, height, payload)
            }
        }
    }

    /// Apply text watermark (simplified - uses basic rendering)
    fn apply_text_watermark(
        &self,
        frame_rgb: &mut [u8],
        width: usize,
        height: usize,
        text: &str,
    ) -> Result<()> {
        // Simplified implementation - would use proper font rendering
        // For now, just render a simple pattern
        let text_width = text.len() * 8; // Assume 8px per char
        let text_height = 16;

        let (x, y) = self
            .position
            .calculate_position(width, height, text_width, text_height);

        // Draw simple text representation
        for (i, _ch) in text.chars().enumerate() {
            let char_x = x + (i * 8) as i32;

            if char_x >= 0 && char_x < width as i32 && y >= 0 && y < height as i32 {
                // Draw simple rectangle as placeholder for character
                self.draw_rect(frame_rgb, width, char_x, y, 6, 12, [255, 255, 255]);
            }
        }

        Ok(())
    }

    /// Draw rectangle
    fn draw_rect(
        &self,
        frame: &mut [u8],
        width: usize,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        color: [u8; 3],
    ) {
        for dy in 0..h {
            for dx in 0..w {
                let px = x + dx;
                let py = y + dy;

                if px >= 0 && px < width as i32 && py >= 0 {
                    let idx = (py as usize * width + px as usize) * 3;

                    if idx + 2 < frame.len() {
                        match self.style {
                            WatermarkStyle::Opaque => {
                                frame[idx] = color[0];
                                frame[idx + 1] = color[1];
                                frame[idx + 2] = color[2];
                            }
                            WatermarkStyle::SemiTransparent { opacity } => {
                                frame[idx] = ((1.0 - opacity) * frame[idx] as f32
                                    + opacity * color[0] as f32)
                                    as u8;
                                frame[idx + 1] = ((1.0 - opacity) * frame[idx + 1] as f32
                                    + opacity * color[1] as f32)
                                    as u8;
                                frame[idx + 2] = ((1.0 - opacity) * frame[idx + 2] as f32
                                    + opacity * color[2] as f32)
                                    as u8;
                            }
                            WatermarkStyle::Additive { strength } => {
                                frame[idx] = (frame[idx] as f32 + strength * color[0] as f32)
                                    .min(255.0) as u8;
                                frame[idx + 1] =
                                    (frame[idx + 1] as f32 + strength * color[1] as f32).min(255.0)
                                        as u8;
                                frame[idx + 2] =
                                    (frame[idx + 2] as f32 + strength * color[2] as f32).min(255.0)
                                        as u8;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Apply image watermark
    fn apply_image_watermark(
        &self,
        frame_rgb: &mut [u8],
        width: usize,
        height: usize,
        watermark_data: &[u8],
        wm_width: usize,
        wm_height: usize,
    ) -> Result<()> {
        let (x, y) = self
            .position
            .calculate_position(width, height, wm_width, wm_height);

        if self.position == WatermarkPosition::Tiled {
            // Tile watermark across frame
            for tile_y in (0..height).step_by(wm_height) {
                for tile_x in (0..width).step_by(wm_width) {
                    self.blit_watermark(
                        frame_rgb,
                        width,
                        watermark_data,
                        wm_width,
                        wm_height,
                        tile_x as i32,
                        tile_y as i32,
                    );
                }
            }
        } else {
            self.blit_watermark(frame_rgb, width, watermark_data, wm_width, wm_height, x, y);
        }

        Ok(())
    }

    /// Blit watermark onto frame
    fn blit_watermark(
        &self,
        frame: &mut [u8],
        frame_width: usize,
        watermark: &[u8],
        wm_width: usize,
        wm_height: usize,
        x: i32,
        y: i32,
    ) {
        for dy in 0..wm_height {
            for dx in 0..wm_width {
                let px = x + dx as i32;
                let py = y + dy as i32;

                if px >= 0 && px < frame_width as i32 && py >= 0 {
                    let frame_idx = (py as usize * frame_width + px as usize) * 3;
                    let wm_idx = (dy * wm_width + dx) * 3;

                    if frame_idx + 2 < frame.len() && wm_idx + 2 < watermark.len() {
                        let wm_color = [
                            watermark[wm_idx],
                            watermark[wm_idx + 1],
                            watermark[wm_idx + 2],
                        ];
                        self.blend_pixel(frame, frame_idx, wm_color);
                    }
                }
            }
        }
    }

    /// Blend pixel with watermark color
    fn blend_pixel(&self, frame: &mut [u8], idx: usize, color: [u8; 3]) {
        match self.style {
            WatermarkStyle::Opaque => {
                frame[idx] = color[0];
                frame[idx + 1] = color[1];
                frame[idx + 2] = color[2];
            }
            WatermarkStyle::SemiTransparent { opacity } => {
                frame[idx] =
                    ((1.0 - opacity) * frame[idx] as f32 + opacity * color[0] as f32) as u8;
                frame[idx + 1] =
                    ((1.0 - opacity) * frame[idx + 1] as f32 + opacity * color[1] as f32) as u8;
                frame[idx + 2] =
                    ((1.0 - opacity) * frame[idx + 2] as f32 + opacity * color[2] as f32) as u8;
            }
            WatermarkStyle::Additive { strength } => {
                frame[idx] = (frame[idx] as f32 + strength * color[0] as f32).min(255.0) as u8;
                frame[idx + 1] =
                    (frame[idx + 1] as f32 + strength * color[1] as f32).min(255.0) as u8;
                frame[idx + 2] =
                    (frame[idx + 2] as f32 + strength * color[2] as f32).min(255.0) as u8;
            }
        }
    }

    /// Embed invisible watermark using LSB steganography
    fn embed_invisible(
        &self,
        frame_rgb: &mut [u8],
        width: usize,
        height: usize,
        payload: &[u8],
    ) -> Result<()> {
        // Embed payload in least significant bits
        let max_bytes = (width * height * 3) / 8; // 1 bit per byte

        if payload.len() > max_bytes {
            return Err(Error::InvalidInput(
                "Payload too large for frame".to_string(),
            ));
        }

        let mut bit_idx = 0;

        for &byte in payload.iter() {
            for bit_pos in (0..8).rev() {
                let bit = (byte >> bit_pos) & 1;

                if bit_idx < frame_rgb.len() {
                    // Clear LSB and set to bit value
                    frame_rgb[bit_idx] = (frame_rgb[bit_idx] & 0xFE) | bit;
                    bit_idx += 1;
                }
            }
        }

        Ok(())
    }

    /// Extract invisible watermark
    pub fn extract_invisible(frame_rgb: &[u8], payload_length: usize) -> Vec<u8> {
        let mut payload = Vec::new();
        let mut current_byte = 0u8;
        let mut bit_count = 0;

        for (i, &pixel_value) in frame_rgb.iter().enumerate() {
            if i / 8 >= payload_length {
                break;
            }

            // Extract LSB
            let bit = pixel_value & 1;
            current_byte = (current_byte << 1) | bit;
            bit_count += 1;

            if bit_count == 8 {
                payload.push(current_byte);
                current_byte = 0;
                bit_count = 0;
            }
        }

        payload
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watermark_position_top_left() {
        let pos = WatermarkPosition::TopLeft;
        let (x, y) = pos.calculate_position(1920, 1080, 100, 50);
        assert_eq!(x, 10);
        assert_eq!(y, 10);
    }

    #[test]
    fn test_watermark_position_center() {
        let pos = WatermarkPosition::Center;
        let (x, y) = pos.calculate_position(1920, 1080, 100, 50);
        assert_eq!(x, (1920 - 100) / 2);
        assert_eq!(y, (1080 - 50) / 2);
    }

    #[test]
    fn test_watermark_style_opacity() {
        let style = WatermarkStyle::SemiTransparent { opacity: 0.5 };
        match style {
            WatermarkStyle::SemiTransparent { opacity } => assert_eq!(opacity, 0.5),
            _ => panic!("Wrong style"),
        }
    }

    #[test]
    fn test_visible_text_watermark() {
        let watermark =
            Watermark::new_visible_text("Test", WatermarkPosition::TopLeft, WatermarkStyle::Opaque);

        let mut frame = vec![0u8; 64 * 64 * 3];
        let result = watermark.apply(&mut frame, 64, 64);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invisible_watermark_embed() {
        let watermark = Watermark::new_invisible("secret");
        let mut frame = vec![128u8; 1000 * 3]; // Enough space

        let result = watermark.apply(&mut frame, 100, 10);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invisible_watermark_extract() {
        let payload = b"hello";
        let watermark = Watermark::new_invisible("hello");
        let mut frame = vec![128u8; 1000 * 3];

        watermark.apply(&mut frame, 100, 10).unwrap();
        let extracted = Watermark::extract_invisible(&frame, payload.len());

        assert_eq!(extracted, payload);
    }

    #[test]
    fn test_watermark_too_large() {
        let large_payload = vec![0u8; 10000];
        let mut frame = vec![0u8; 100 * 3]; // Too small

        let watermark = Watermark {
            watermark_type: WatermarkType::Invisible {
                payload: large_payload,
            },
            position: WatermarkPosition::Center,
            style: WatermarkStyle::Opaque,
        };

        let result = watermark.apply(&mut frame, 10, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_image_watermark() {
        let wm_data = vec![255u8; 10 * 10 * 3]; // 10x10 white image
        let watermark = Watermark::new_visible_image(
            wm_data,
            10,
            10,
            WatermarkPosition::TopLeft,
            WatermarkStyle::SemiTransparent { opacity: 0.5 },
        );

        let mut frame = vec![0u8; 64 * 64 * 3];
        let result = watermark.apply(&mut frame, 64, 64);
        assert!(result.is_ok());
    }

    #[test]
    fn test_tiled_watermark() {
        let wm_data = vec![255u8; 10 * 10 * 3];
        let watermark = Watermark::new_visible_image(
            wm_data,
            10,
            10,
            WatermarkPosition::Tiled,
            WatermarkStyle::Opaque,
        );

        let mut frame = vec![0u8; 32 * 32 * 3];
        let result = watermark.apply(&mut frame, 32, 32);
        assert!(result.is_ok());
    }
}
