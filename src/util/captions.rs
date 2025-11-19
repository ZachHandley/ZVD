//! Closed Captions and Subtitle Rendering
//!
//! Parse, render, and burn-in subtitles and closed captions for
//! accessibility, localization, and broadcast compliance.
//!
//! ## Subtitle Formats
//!
//! - **SRT**: SubRip Text (most common)
//! - **WebVTT**: Web Video Text Tracks (HTML5)
//! - **CEA-608**: Line 21 closed captions (NTSC)
//! - **CEA-708**: Digital closed captions (ATSC)
//!
//! ## Use Cases
//!
//! - Accessibility compliance
//! - Multi-language support
//! - Broadcast requirements (FCC)
//! - Educational content
//! - Social media (silent autoplay)
//!
//! ## Usage
//!
//! ```rust
//! use zvd_lib::util::captions::{SubtitleRenderer, SubtitleFormat};
//!
//! // Parse SRT file
//! let srt_content = std::fs::read_to_string("subtitles.srt")?;
//! let subtitles = SubtitleRenderer::parse_srt(&srt_content)?;
//!
//! // Render at specific time
//! let frame_time = 10.5; // 10.5 seconds
//! renderer.render_at_time(&mut frame, frame_time, width, height)?;
//! ```

use crate::error::{Error, Result};
use std::time::Duration;

/// Subtitle format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitleFormat {
    /// SubRip Text (.srt)
    SRT,
    /// Web Video Text Tracks (.vtt)
    WebVTT,
    /// CEA-608 (Line 21)
    CEA608,
    /// CEA-708 (Digital)
    CEA708,
}

/// Subtitle entry
#[derive(Debug, Clone)]
pub struct Subtitle {
    /// Start time in seconds
    pub start_time: f64,
    /// End time in seconds
    pub end_time: f64,
    /// Text content (may include formatting)
    pub text: String,
    /// Optional position hint
    pub position: Option<CaptionPosition>,
}

/// Caption position on screen
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptionPosition {
    Bottom,
    Top,
    Middle,
}

/// Subtitle style
#[derive(Debug, Clone)]
pub struct SubtitleStyle {
    /// Font size (pixels)
    pub font_size: usize,
    /// Text color (RGB)
    pub color: [u8; 3],
    /// Background color (RGBA)
    pub background: Option<[u8; 4]>,
    /// Outline/shadow
    pub outline: bool,
}

impl Default for SubtitleStyle {
    fn default() -> Self {
        SubtitleStyle {
            font_size: 24,
            color: [255, 255, 255],           // White
            background: Some([0, 0, 0, 180]), // Semi-transparent black
            outline: true,
        }
    }
}

/// Subtitle renderer
pub struct SubtitleRenderer {
    subtitles: Vec<Subtitle>,
    style: SubtitleStyle,
    format: SubtitleFormat,
}

impl SubtitleRenderer {
    /// Create new subtitle renderer
    pub fn new(format: SubtitleFormat) -> Self {
        SubtitleRenderer {
            subtitles: Vec::new(),
            style: SubtitleStyle::default(),
            format,
        }
    }

    /// Set subtitle style
    pub fn set_style(&mut self, style: SubtitleStyle) {
        self.style = style;
    }

    /// Parse SRT format
    ///
    /// Format:
    /// ```text
    /// 1
    /// 00:00:01,000 --> 00:00:04,000
    /// First subtitle text
    ///
    /// 2
    /// 00:00:05,000 --> 00:00:08,000
    /// Second subtitle text
    /// ```
    pub fn parse_srt(content: &str) -> Result<Vec<Subtitle>> {
        let mut subtitles = Vec::new();
        let blocks: Vec<&str> = content.split("\n\n").collect();

        for block in blocks {
            let lines: Vec<&str> = block.trim().lines().collect();
            if lines.len() < 3 {
                continue;
            }

            // Line 0: index (ignore)
            // Line 1: timestamp
            // Line 2+: text

            let timestamp_line = lines[1];
            let text = lines[2..].join("\n");

            if let Some((start, end)) = Self::parse_srt_timestamp(timestamp_line) {
                subtitles.push(Subtitle {
                    start_time: start,
                    end_time: end,
                    text,
                    position: Some(CaptionPosition::Bottom),
                });
            }
        }

        Ok(subtitles)
    }

    /// Parse SRT timestamp line
    fn parse_srt_timestamp(line: &str) -> Option<(f64, f64)> {
        let parts: Vec<&str> = line.split(" --> ").collect();
        if parts.len() != 2 {
            return None;
        }

        let start = Self::parse_srt_time(parts[0])?;
        let end = Self::parse_srt_time(parts[1])?;

        Some((start, end))
    }

    /// Parse SRT time (HH:MM:SS,mmm)
    fn parse_srt_time(time_str: &str) -> Option<f64> {
        let parts: Vec<&str> = time_str.split(',').collect();
        if parts.len() != 2 {
            return None;
        }

        let hms: Vec<&str> = parts[0].split(':').collect();
        if hms.len() != 3 {
            return None;
        }

        let hours: f64 = hms[0].trim().parse().ok()?;
        let minutes: f64 = hms[1].trim().parse().ok()?;
        let seconds: f64 = hms[2].trim().parse().ok()?;
        let millis: f64 = parts[1].trim().parse().ok()?;

        Some(hours * 3600.0 + minutes * 60.0 + seconds + millis / 1000.0)
    }

    /// Parse WebVTT format
    ///
    /// Format:
    /// ```text
    /// WEBVTT
    ///
    /// 00:00:01.000 --> 00:00:04.000
    /// First subtitle text
    ///
    /// 00:00:05.000 --> 00:00:08.000
    /// Second subtitle text
    /// ```
    pub fn parse_webvtt(content: &str) -> Result<Vec<Subtitle>> {
        let mut subtitles = Vec::new();

        // Skip "WEBVTT" header
        let content = content.trim_start_matches("WEBVTT").trim();

        let blocks: Vec<&str> = content.split("\n\n").collect();

        for block in blocks {
            let lines: Vec<&str> = block.trim().lines().collect();
            if lines.is_empty() {
                continue;
            }

            // First line: timestamp
            // Remaining lines: text
            let timestamp_line = lines[0];
            let text = if lines.len() > 1 {
                lines[1..].join("\n")
            } else {
                continue;
            };

            if let Some((start, end)) = Self::parse_webvtt_timestamp(timestamp_line) {
                subtitles.push(Subtitle {
                    start_time: start,
                    end_time: end,
                    text,
                    position: Some(CaptionPosition::Bottom),
                });
            }
        }

        Ok(subtitles)
    }

    /// Parse WebVTT timestamp line
    fn parse_webvtt_timestamp(line: &str) -> Option<(f64, f64)> {
        let parts: Vec<&str> = line.split(" --> ").collect();
        if parts.len() < 2 {
            return None;
        }

        let start = Self::parse_webvtt_time(parts[0])?;
        let end = Self::parse_webvtt_time(parts[1].split_whitespace().next()?)?;

        Some((start, end))
    }

    /// Parse WebVTT time (HH:MM:SS.mmm or MM:SS.mmm)
    fn parse_webvtt_time(time_str: &str) -> Option<f64> {
        let parts: Vec<&str> = time_str.split('.').collect();
        if parts.is_empty() {
            return None;
        }

        let hms: Vec<&str> = parts[0].split(':').collect();
        let millis: f64 = if parts.len() > 1 {
            parts[1].parse().ok()?
        } else {
            0.0
        };

        let time = if hms.len() == 3 {
            // HH:MM:SS
            let hours: f64 = hms[0].parse().ok()?;
            let minutes: f64 = hms[1].parse().ok()?;
            let seconds: f64 = hms[2].parse().ok()?;
            hours * 3600.0 + minutes * 60.0 + seconds
        } else if hms.len() == 2 {
            // MM:SS
            let minutes: f64 = hms[0].parse().ok()?;
            let seconds: f64 = hms[1].parse().ok()?;
            minutes * 60.0 + seconds
        } else {
            return None;
        };

        Some(time + millis / 1000.0)
    }

    /// Load subtitles
    pub fn load_subtitles(&mut self, subtitles: Vec<Subtitle>) {
        self.subtitles = subtitles;
    }

    /// Get active subtitle at time
    pub fn get_subtitle_at_time(&self, time: f64) -> Option<&Subtitle> {
        self.subtitles
            .iter()
            .find(|s| time >= s.start_time && time <= s.end_time)
    }

    /// Render subtitle at specific time onto RGB frame
    pub fn render_at_time(
        &self,
        frame_rgb: &mut [u8],
        time: f64,
        width: usize,
        height: usize,
    ) -> Result<()> {
        if let Some(subtitle) = self.get_subtitle_at_time(time) {
            self.render_subtitle(frame_rgb, subtitle, width, height)?;
        }
        Ok(())
    }

    /// Render subtitle text onto frame
    fn render_subtitle(
        &self,
        frame_rgb: &mut [u8],
        subtitle: &Subtitle,
        width: usize,
        height: usize,
    ) -> Result<()> {
        if frame_rgb.len() != width * height * 3 {
            return Err(Error::InvalidInput("Invalid frame size".to_string()));
        }

        // Calculate position
        let position = subtitle.position.unwrap_or(CaptionPosition::Bottom);
        let (x, y) = self.calculate_position(&subtitle.text, width, height, position);

        // Draw background if enabled
        if let Some(bg) = self.style.background {
            self.draw_background(
                frame_rgb,
                x,
                y,
                subtitle.text.len() * 12,
                self.style.font_size,
                bg,
                width,
            );
        }

        // Draw text (simplified rendering)
        self.draw_text(frame_rgb, &subtitle.text, x, y, width, height);

        Ok(())
    }

    /// Calculate position for text
    fn calculate_position(
        &self,
        text: &str,
        width: usize,
        height: usize,
        position: CaptionPosition,
    ) -> (i32, i32) {
        let text_width = text.len() * 12; // Approximate
        let text_height = self.style.font_size;

        let x = ((width - text_width) / 2) as i32;
        let y = match position {
            CaptionPosition::Top => 20,
            CaptionPosition::Middle => ((height - text_height) / 2) as i32,
            CaptionPosition::Bottom => (height - text_height - 20) as i32,
        };

        (x, y)
    }

    /// Draw background box
    fn draw_background(
        &self,
        frame: &mut [u8],
        x: i32,
        y: i32,
        w: usize,
        h: usize,
        color: [u8; 4],
        frame_width: usize,
    ) {
        let padding = 8;
        for dy in 0..(h + 2 * padding) {
            for dx in 0..(w + 2 * padding) {
                let px = x - padding as i32 + dx as i32;
                let py = y - padding as i32 + dy as i32;

                if px >= 0 && py >= 0 {
                    let idx = (py as usize * frame_width + px as usize) * 3;
                    if idx + 2 < frame.len() {
                        let alpha = color[3] as f32 / 255.0;
                        frame[idx] =
                            ((1.0 - alpha) * frame[idx] as f32 + alpha * color[0] as f32) as u8;
                        frame[idx + 1] =
                            ((1.0 - alpha) * frame[idx + 1] as f32 + alpha * color[1] as f32) as u8;
                        frame[idx + 2] =
                            ((1.0 - alpha) * frame[idx + 2] as f32 + alpha * color[2] as f32) as u8;
                    }
                }
            }
        }
    }

    /// Draw text (simplified)
    fn draw_text(
        &self,
        frame: &mut [u8],
        text: &str,
        x: i32,
        y: i32,
        width: usize,
        _height: usize,
    ) {
        // Simplified text rendering - just draw rectangles for each character
        for (i, _ch) in text.chars().enumerate() {
            let char_x = x + (i * 12) as i32;

            // Draw outline if enabled
            if self.style.outline {
                self.draw_char_rect(frame, char_x + 1, y + 1, width, [0, 0, 0]);
            }

            // Draw character
            self.draw_char_rect(frame, char_x, y, width, self.style.color);
        }
    }

    /// Draw character rectangle
    fn draw_char_rect(&self, frame: &mut [u8], x: i32, y: i32, frame_width: usize, color: [u8; 3]) {
        let char_width = 10;
        let char_height = self.style.font_size;

        for dy in 0..char_height {
            for dx in 0..char_width {
                let px = x + dx as i32;
                let py = y + dy as i32;

                if px >= 0 && py >= 0 {
                    let idx = (py as usize * frame_width + px as usize) * 3;
                    if idx + 2 < frame.len() {
                        frame[idx] = color[0];
                        frame[idx + 1] = color[1];
                        frame[idx + 2] = color[2];
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_srt_time() {
        let time = SubtitleRenderer::parse_srt_time("00:01:23,456");
        assert_eq!(time, Some(83.456)); // 1*60 + 23 + 0.456
    }

    #[test]
    fn test_parse_srt_timestamp() {
        let (start, end) =
            SubtitleRenderer::parse_srt_timestamp("00:00:01,000 --> 00:00:04,000").unwrap();
        assert_eq!(start, 1.0);
        assert_eq!(end, 4.0);
    }

    #[test]
    fn test_parse_srt() {
        let srt = "1\n00:00:01,000 --> 00:00:04,000\nFirst subtitle\n\n2\n00:00:05,000 --> 00:00:08,000\nSecond subtitle";
        let subtitles = SubtitleRenderer::parse_srt(srt).unwrap();

        assert_eq!(subtitles.len(), 2);
        assert_eq!(subtitles[0].text, "First subtitle");
        assert_eq!(subtitles[0].start_time, 1.0);
        assert_eq!(subtitles[0].end_time, 4.0);
    }

    #[test]
    fn test_parse_webvtt_time() {
        let time = SubtitleRenderer::parse_webvtt_time("00:01:23.456");
        assert_eq!(time, Some(83.456));

        let time_short = SubtitleRenderer::parse_webvtt_time("01:23.456");
        assert_eq!(time_short, Some(83.456));
    }

    #[test]
    fn test_parse_webvtt() {
        let vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nFirst subtitle\n\n00:00:05.000 --> 00:00:08.000\nSecond subtitle";
        let subtitles = SubtitleRenderer::parse_webvtt(vtt).unwrap();

        assert_eq!(subtitles.len(), 2);
        assert_eq!(subtitles[0].text, "First subtitle");
    }

    #[test]
    fn test_get_subtitle_at_time() {
        let mut renderer = SubtitleRenderer::new(SubtitleFormat::SRT);
        renderer.load_subtitles(vec![
            Subtitle {
                start_time: 1.0,
                end_time: 4.0,
                text: "First".to_string(),
                position: None,
            },
            Subtitle {
                start_time: 5.0,
                end_time: 8.0,
                text: "Second".to_string(),
                position: None,
            },
        ]);

        let sub = renderer.get_subtitle_at_time(2.5).unwrap();
        assert_eq!(sub.text, "First");

        let sub2 = renderer.get_subtitle_at_time(6.0).unwrap();
        assert_eq!(sub2.text, "Second");

        let none = renderer.get_subtitle_at_time(10.0);
        assert!(none.is_none());
    }

    #[test]
    fn test_subtitle_style_default() {
        let style = SubtitleStyle::default();
        assert_eq!(style.font_size, 24);
        assert_eq!(style.color, [255, 255, 255]);
        assert!(style.outline);
    }

    #[test]
    fn test_caption_position() {
        assert_eq!(CaptionPosition::Bottom, CaptionPosition::Bottom);
        assert_ne!(CaptionPosition::Top, CaptionPosition::Bottom);
    }

    #[test]
    fn test_render_subtitle() {
        let renderer = SubtitleRenderer::new(SubtitleFormat::SRT);
        let mut frame = vec![0u8; 640 * 480 * 3];

        let subtitle = Subtitle {
            start_time: 0.0,
            end_time: 5.0,
            text: "Test".to_string(),
            position: Some(CaptionPosition::Bottom),
        };

        let result = renderer.render_subtitle(&mut frame, &subtitle, 640, 480);
        assert!(result.is_ok());
    }
}
