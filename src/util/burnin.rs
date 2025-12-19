//! Burn-in Timecode and Text Overlay
//!
//! Render visible timecode, metadata, and text directly onto video frames
//! for dailies, review, and archival purposes.
//!
//! ## Timecode Types
//!
//! - **Source Timecode**: Original media timecode
//! - **Record Run**: Continuous timecode across takes
//! - **Free Run**: Real-time clock timecode
//! - **Frame Count**: Simple frame number
//!
//! ## Common Use Cases
//!
//! - Dailies review and approval
//! - VFX reference plates
//! - Color grading reference
//! - Archive and preservation
//! - Multi-camera sync reference
//!
//! ## Usage
//!
//! ```rust
//! use zvd_lib::util::burnin::{BurninGenerator, TimecodeStyle, BurninPosition};
//!
//! let mut burnin = BurninGenerator::new(1920, 1080);
//! burnin.set_position(BurninPosition::TopCenter);
//! burnin.set_style(TimecodeStyle::Large);
//!
//! let timecode = "01:23:45:12";
//! burnin.render_timecode(&mut frame, timecode)?;
//! ```

use crate::error::{Error, Result};

/// Timecode display format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimecodeFormat {
    /// HH:MM:SS:FF (standard)
    Standard,
    /// HH:MM:SS;FF (drop frame)
    DropFrame,
    /// Frame count (e.g., "123456")
    FrameCount,
    /// Feet+Frames (film, e.g., "1234+15")
    FeetFrames,
}

impl TimecodeFormat {
    /// Get separator character
    pub fn separator(&self) -> char {
        match self {
            TimecodeFormat::Standard => ':',
            TimecodeFormat::DropFrame => ';',
            TimecodeFormat::FrameCount => ' ',
            TimecodeFormat::FeetFrames => '+',
        }
    }
}

/// Burn-in position on frame
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BurninPosition {
    TopLeft,
    TopCenter,
    TopRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
    Custom { x: i32, y: i32 },
}

impl BurninPosition {
    /// Calculate pixel position
    pub fn calculate_position(
        &self,
        frame_width: usize,
        frame_height: usize,
        text_width: usize,
        text_height: usize,
    ) -> (i32, i32) {
        let margin = 20; // Pixels from edge

        match self {
            BurninPosition::TopLeft => (margin, margin),
            BurninPosition::TopCenter => ((frame_width as i32 - text_width as i32) / 2, margin),
            BurninPosition::TopRight => (frame_width as i32 - text_width as i32 - margin, margin),
            BurninPosition::BottomLeft => {
                (margin, frame_height as i32 - text_height as i32 - margin)
            }
            BurninPosition::BottomCenter => (
                (frame_width as i32 - text_width as i32) / 2,
                frame_height as i32 - text_height as i32 - margin,
            ),
            BurninPosition::BottomRight => (
                frame_width as i32 - text_width as i32 - margin,
                frame_height as i32 - text_height as i32 - margin,
            ),
            BurninPosition::Custom { x, y } => (*x, *y),
        }
    }
}

/// Timecode style
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimecodeStyle {
    /// Small (12pt equivalent)
    Small,
    /// Medium (18pt equivalent)
    Medium,
    /// Large (24pt equivalent)
    Large,
    /// Extra Large (36pt equivalent)
    ExtraLarge,
}

impl TimecodeStyle {
    /// Get character height in pixels
    pub fn char_height(&self) -> usize {
        match self {
            TimecodeStyle::Small => 16,
            TimecodeStyle::Medium => 24,
            TimecodeStyle::Large => 32,
            TimecodeStyle::ExtraLarge => 48,
        }
    }

    /// Get character width in pixels (monospace)
    pub fn char_width(&self) -> usize {
        match self {
            TimecodeStyle::Small => 10,
            TimecodeStyle::Medium => 14,
            TimecodeStyle::Large => 20,
            TimecodeStyle::ExtraLarge => 30,
        }
    }
}

/// Text color
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8, // Alpha (opacity)
}

impl Color {
    /// White
    pub const WHITE: Color = Color {
        r: 255,
        g: 255,
        b: 255,
        a: 255,
    };

    /// Black
    pub const BLACK: Color = Color {
        r: 0,
        g: 0,
        b: 0,
        a: 255,
    };

    /// Yellow (typical timecode color)
    pub const YELLOW: Color = Color {
        r: 255,
        g: 255,
        b: 0,
        a: 255,
    };

    /// Green
    pub const GREEN: Color = Color {
        r: 0,
        g: 255,
        b: 0,
        a: 255,
    };

    /// Semi-transparent black (for backgrounds)
    pub const BLACK_ALPHA: Color = Color {
        r: 0,
        g: 0,
        b: 0,
        a: 180,
    };
}

/// Burn-in configuration
#[derive(Debug, Clone)]
pub struct BurninConfig {
    /// Position on frame
    pub position: BurninPosition,
    /// Timecode format
    pub format: TimecodeFormat,
    /// Text style
    pub style: TimecodeStyle,
    /// Text color
    pub color: Color,
    /// Background color (None = no background)
    pub background: Option<Color>,
    /// Drop shadow
    pub drop_shadow: bool,
}

impl BurninConfig {
    /// Create new config with defaults
    pub fn new() -> Self {
        BurninConfig {
            position: BurninPosition::TopCenter,
            format: TimecodeFormat::Standard,
            style: TimecodeStyle::Large,
            color: Color::YELLOW,
            background: Some(Color::BLACK_ALPHA),
            drop_shadow: true,
        }
    }
}

impl Default for BurninConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Burn-in generator
pub struct BurninGenerator {
    width: usize,
    height: usize,
    config: BurninConfig,
}

impl BurninGenerator {
    /// Create new burn-in generator
    pub fn new(width: usize, height: usize) -> Self {
        BurninGenerator {
            width,
            height,
            config: BurninConfig::new(),
        }
    }

    /// Set position
    pub fn set_position(&mut self, position: BurninPosition) {
        self.config.position = position;
    }

    /// Set style
    pub fn set_style(&mut self, style: TimecodeStyle) {
        self.config.style = style;
    }

    /// Set color
    pub fn set_color(&mut self, color: Color) {
        self.config.color = color;
    }

    /// Set background
    pub fn set_background(&mut self, background: Option<Color>) {
        self.config.background = background;
    }

    /// Render timecode onto RGB frame
    pub fn render_timecode(&self, frame_rgb: &mut [u8], timecode: &str) -> Result<()> {
        if frame_rgb.len() != self.width * self.height * 3 {
            return Err(Error::InvalidInput("Invalid frame size".to_string()));
        }

        let text_width = timecode.len() * self.config.style.char_width();
        let text_height = self.config.style.char_height();

        let (x, y) = self.config.position.calculate_position(
            self.width,
            self.height,
            text_width,
            text_height,
        );

        // Draw background if enabled
        if let Some(bg_color) = self.config.background {
            self.draw_background(frame_rgb, x, y, text_width, text_height, bg_color);
        }

        // Draw drop shadow if enabled
        if self.config.drop_shadow {
            self.draw_text(frame_rgb, timecode, x + 2, y + 2, Color::BLACK);
        }

        // Draw timecode text
        self.draw_text(frame_rgb, timecode, x, y, self.config.color);

        Ok(())
    }

    /// Draw background rectangle
    fn draw_background(
        &self,
        frame_rgb: &mut [u8],
        x: i32,
        y: i32,
        width: usize,
        height: usize,
        color: Color,
    ) {
        let padding = 4;
        let x = (x - padding as i32).max(0);
        let y = (y - padding as i32).max(0);
        let width = width + 2 * padding;
        let height = height + 2 * padding;

        for dy in 0..height {
            for dx in 0..width {
                let px = x + dx as i32;
                let py = y + dy as i32;

                if px >= 0 && px < self.width as i32 && py >= 0 && py < self.height as i32 {
                    let idx = (py as usize * self.width + px as usize) * 3;

                    // Alpha blend
                    let alpha = color.a as f32 / 255.0;
                    frame_rgb[idx] =
                        ((1.0 - alpha) * frame_rgb[idx] as f32 + alpha * color.r as f32) as u8;
                    frame_rgb[idx + 1] =
                        ((1.0 - alpha) * frame_rgb[idx + 1] as f32 + alpha * color.g as f32) as u8;
                    frame_rgb[idx + 2] =
                        ((1.0 - alpha) * frame_rgb[idx + 2] as f32 + alpha * color.b as f32) as u8;
                }
            }
        }
    }

    /// Draw text (simplified bitmap font rendering)
    fn draw_text(&self, frame_rgb: &mut [u8], text: &str, x: i32, y: i32, color: Color) {
        let char_width = self.config.style.char_width();
        let char_height = self.config.style.char_height();

        for (i, ch) in text.chars().enumerate() {
            let char_x = x + (i * char_width) as i32;
            self.draw_char(frame_rgb, ch, char_x, y, char_width, char_height, color);
        }
    }

    /// Draw single character (simple block rendering)
    fn draw_char(
        &self,
        frame_rgb: &mut [u8],
        ch: char,
        x: i32,
        y: i32,
        width: usize,
        height: usize,
        color: Color,
    ) {
        // Simple 5x7 bitmap font patterns for digits and common chars
        let pattern = self.get_char_pattern(ch);

        let scale_x = width / 5;
        let scale_y = height / 7;

        for (row, &bits) in pattern.iter().enumerate() {
            for col in 0..5 {
                if (bits >> (4 - col)) & 1 == 1 {
                    // Draw scaled pixel
                    for dy in 0..scale_y {
                        for dx in 0..scale_x {
                            let px = x + (col * scale_x + dx) as i32;
                            let py = y + (row * scale_y + dy) as i32;

                            if px >= 0
                                && px < self.width as i32
                                && py >= 0
                                && py < self.height as i32
                            {
                                let idx = (py as usize * self.width + px as usize) * 3;
                                frame_rgb[idx] = color.r;
                                frame_rgb[idx + 1] = color.g;
                                frame_rgb[idx + 2] = color.b;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Get 5x7 bitmap pattern for character
    fn get_char_pattern(&self, ch: char) -> &[u8] {
        match ch {
            '0' => &[
                0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
            ],
            '1' => &[
                0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
            ],
            '2' => &[
                0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
            ],
            '3' => &[
                0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110,
            ],
            '4' => &[
                0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
            ],
            '5' => &[
                0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
            ],
            '6' => &[
                0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
            ],
            '7' => &[
                0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
            ],
            '8' => &[
                0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
            ],
            '9' => &[
                0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
            ],
            ':' => &[
                0b00000, 0b00000, 0b01100, 0b00000, 0b00000, 0b01100, 0b00000,
            ],
            ';' => &[
                0b00000, 0b00000, 0b01100, 0b00000, 0b00000, 0b01100, 0b01000,
            ],
            '+' => &[
                0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000,
            ],
            '-' => &[
                0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000,
            ],
            ' ' => &[
                0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000,
            ],
            _ => &[
                0b11111, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11111,
            ], // Unknown char
        }
    }

    /// Render metadata overlay (multi-line)
    pub fn render_metadata(&self, frame_rgb: &mut [u8], lines: &[String]) -> Result<()> {
        let line_height = self.config.style.char_height() + 4; // 4px spacing

        for (i, line) in lines.iter().enumerate() {
            let y_offset = i as i32 * line_height as i32;

            // Calculate position for this line
            let text_width = line.len() * self.config.style.char_width();
            let (base_x, base_y) = self.config.position.calculate_position(
                self.width,
                self.height,
                text_width,
                self.config.style.char_height(),
            );

            let y = base_y + y_offset;

            // Draw background
            if let Some(bg_color) = self.config.background {
                self.draw_background(
                    frame_rgb,
                    base_x,
                    y,
                    text_width,
                    self.config.style.char_height(),
                    bg_color,
                );
            }

            // Draw text
            if self.config.drop_shadow {
                self.draw_text(frame_rgb, line, base_x + 2, y + 2, Color::BLACK);
            }
            self.draw_text(frame_rgb, line, base_x, y, self.config.color);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timecode_format_separator() {
        assert_eq!(TimecodeFormat::Standard.separator(), ':');
        assert_eq!(TimecodeFormat::DropFrame.separator(), ';');
    }

    #[test]
    fn test_burnin_position_top_left() {
        let pos = BurninPosition::TopLeft;
        let (x, y) = pos.calculate_position(1920, 1080, 100, 20);
        assert_eq!(x, 20); // margin
        assert_eq!(y, 20);
    }

    #[test]
    fn test_burnin_position_center() {
        let pos = BurninPosition::TopCenter;
        let (x, y) = pos.calculate_position(1920, 1080, 200, 20);
        assert_eq!(x, (1920 - 200) / 2);
        assert_eq!(y, 20);
    }

    #[test]
    fn test_burnin_position_custom() {
        let pos = BurninPosition::Custom { x: 100, y: 200 };
        let (x, y) = pos.calculate_position(1920, 1080, 50, 20);
        assert_eq!(x, 100);
        assert_eq!(y, 200);
    }

    #[test]
    fn test_timecode_style_dimensions() {
        assert_eq!(TimecodeStyle::Small.char_height(), 16);
        assert_eq!(TimecodeStyle::Small.char_width(), 10);

        assert_eq!(TimecodeStyle::Large.char_height(), 32);
        assert_eq!(TimecodeStyle::Large.char_width(), 20);
    }

    #[test]
    fn test_color_constants() {
        assert_eq!(Color::WHITE.r, 255);
        assert_eq!(Color::BLACK.r, 0);
        assert_eq!(Color::YELLOW.r, 255);
        assert_eq!(Color::YELLOW.g, 255);
        assert_eq!(Color::YELLOW.b, 0);
    }

    #[test]
    fn test_burnin_config_default() {
        let config = BurninConfig::new();
        assert_eq!(config.position, BurninPosition::TopCenter);
        assert_eq!(config.format, TimecodeFormat::Standard);
        assert_eq!(config.style, TimecodeStyle::Large);
        assert!(config.drop_shadow);
    }

    #[test]
    fn test_burnin_generator_creation() {
        let burnin = BurninGenerator::new(1920, 1080);
        assert_eq!(burnin.width, 1920);
        assert_eq!(burnin.height, 1080);
    }

    #[test]
    fn test_burnin_render_timecode() {
        let burnin = BurninGenerator::new(640, 480);
        let mut frame = vec![0u8; 640 * 480 * 3];

        let result = burnin.render_timecode(&mut frame, "01:23:45:12");
        assert!(result.is_ok());
    }

    #[test]
    fn test_burnin_render_invalid_size() {
        let burnin = BurninGenerator::new(640, 480);
        let mut frame = vec![0u8; 100]; // Wrong size

        let result = burnin.render_timecode(&mut frame, "01:23:45:12");
        assert!(result.is_err());
    }

    #[test]
    fn test_burnin_set_position() {
        let mut burnin = BurninGenerator::new(1920, 1080);
        burnin.set_position(BurninPosition::BottomRight);
        assert_eq!(burnin.config.position, BurninPosition::BottomRight);
    }

    #[test]
    fn test_burnin_set_style() {
        let mut burnin = BurninGenerator::new(1920, 1080);
        burnin.set_style(TimecodeStyle::ExtraLarge);
        assert_eq!(burnin.config.style, TimecodeStyle::ExtraLarge);
    }

    #[test]
    fn test_burnin_metadata_rendering() {
        let burnin = BurninGenerator::new(1920, 1080);
        let mut frame = vec![0u8; 1920 * 1080 * 3];

        let metadata = vec![
            "Scene: 42A".to_string(),
            "Take: 3".to_string(),
            "Camera: A".to_string(),
        ];

        let result = burnin.render_metadata(&mut frame, &metadata);
        assert!(result.is_ok());
    }
}
