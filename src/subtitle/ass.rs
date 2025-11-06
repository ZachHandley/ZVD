//! ASS (Advanced SubStation Alpha) / SSA subtitle format parser
//!
//! ASS/SSA is a more advanced subtitle format supporting styling,
//! positioning, and animation effects.

use super::{Subtitle, SubtitleParser, SubtitleFormat, parse_timestamp};
use crate::error::{Error, Result};
use std::collections::HashMap;
use std::time::Duration;

/// ASS event (dialogue line)
pub type AssEvent = Subtitle;

/// ASS style definition
#[derive(Debug, Clone)]
pub struct AssStyle {
    pub name: String,
    pub font_name: String,
    pub font_size: u32,
    pub primary_color: String,
    pub secondary_color: String,
    pub outline_color: String,
    pub back_color: String,
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
    pub strikeout: bool,
    pub scale_x: f32,
    pub scale_y: f32,
    pub spacing: f32,
    pub angle: f32,
    pub border_style: u8,
    pub outline: f32,
    pub shadow: f32,
    pub alignment: u8,
    pub margin_l: u32,
    pub margin_r: u32,
    pub margin_v: u32,
    pub encoding: u8,
}

impl Default for AssStyle {
    fn default() -> Self {
        AssStyle {
            name: "Default".to_string(),
            font_name: "Arial".to_string(),
            font_size: 20,
            primary_color: "&H00FFFFFF".to_string(),
            secondary_color: "&H00FFFF00".to_string(),
            outline_color: "&H00000000".to_string(),
            back_color: "&H00000000".to_string(),
            bold: false,
            italic: false,
            underline: false,
            strikeout: false,
            scale_x: 100.0,
            scale_y: 100.0,
            spacing: 0.0,
            angle: 0.0,
            border_style: 1,
            outline: 2.0,
            shadow: 0.0,
            alignment: 2,
            margin_l: 10,
            margin_r: 10,
            margin_v: 10,
            encoding: 1,
        }
    }
}

/// ASS subtitle parser
#[derive(Debug)]
pub struct AssParser {
    script_info: HashMap<String, String>,
    styles: HashMap<String, AssStyle>,
    is_ssa: bool,
}

impl Default for AssParser {
    fn default() -> Self {
        Self::new()
    }
}

impl AssParser {
    pub fn new() -> Self {
        AssParser {
            script_info: HashMap::new(),
            styles: HashMap::new(),
            is_ssa: false,
        }
    }

    /// Parse ASS timestamp (H:MM:SS.cc)
    fn parse_ass_timestamp(s: &str) -> Option<Duration> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 3 {
            return None;
        }

        let hours: u64 = parts[0].parse().ok()?;
        let minutes: u64 = parts[1].parse().ok()?;

        let sec_parts: Vec<&str> = parts[2].split('.').collect();
        let seconds: u64 = sec_parts[0].parse().ok()?;
        let centiseconds: u64 = if sec_parts.len() > 1 {
            sec_parts[1].parse().ok()?
        } else {
            0
        };

        Some(Duration::from_millis(
            hours * 3600_000 + minutes * 60_000 + seconds * 1000 + centiseconds * 10,
        ))
    }

    /// Format timestamp as ASS format
    fn format_ass_timestamp(duration: Duration) -> String {
        let total_centiseconds = duration.as_millis() / 10;
        let hours = total_centiseconds / 360_000;
        let minutes = (total_centiseconds % 360_000) / 6000;
        let seconds = (total_centiseconds % 6000) / 100;
        let centiseconds = total_centiseconds % 100;

        format!("{}:{:02}:{:02}.{:02}", hours, minutes, seconds, centiseconds)
    }

    /// Parse style line
    fn parse_style(&mut self, line: &str, format_fields: &[&str]) {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.is_empty() {
            return;
        }

        let mut style = AssStyle::default();
        style.name = parts[0].to_string();

        // Parse fields based on format
        for (i, field) in format_fields.iter().enumerate() {
            if i >= parts.len() {
                break;
            }

            match field.trim() {
                "Fontname" => style.font_name = parts[i].to_string(),
                "Fontsize" => style.font_size = parts[i].parse().unwrap_or(20),
                "Bold" => style.bold = parts[i] == "-1" || parts[i] == "1",
                "Italic" => style.italic = parts[i] == "-1" || parts[i] == "1",
                "Alignment" => style.alignment = parts[i].parse().unwrap_or(2),
                _ => {}
            }
        }

        self.styles.insert(style.name.clone(), style);
    }

    /// Parse dialogue line
    fn parse_dialogue(&self, line: &str, format_fields: &[&str]) -> Result<Subtitle> {
        let parts: Vec<&str> = line.splitn(format_fields.len(), ',').collect();

        let mut layer = 0;
        let mut start_time = Duration::ZERO;
        let mut end_time = Duration::ZERO;
        let mut style_name = "Default".to_string();
        let mut text = String::new();

        for (i, field) in format_fields.iter().enumerate() {
            if i >= parts.len() {
                break;
            }

            match field.trim() {
                "Layer" => layer = parts[i].parse().unwrap_or(0),
                "Start" => {
                    start_time = Self::parse_ass_timestamp(parts[i].trim())
                        .ok_or_else(|| Error::invalid_input("Invalid start time"))?;
                }
                "End" => {
                    end_time = Self::parse_ass_timestamp(parts[i].trim())
                        .ok_or_else(|| Error::invalid_input("Invalid end time"))?;
                }
                "Style" => style_name = parts[i].trim().to_string(),
                "Text" => text = parts[i..].join(","),
                _ => {}
            }
        }

        let mut subtitle = Subtitle::new(start_time, end_time, text);
        subtitle.style = Some(style_name);

        Ok(subtitle)
    }
}

impl SubtitleParser for AssParser {
    fn parse(&mut self, content: &str) -> Result<Vec<Subtitle>> {
        let mut subtitles = Vec::new();
        let mut current_section = String::new();
        let mut style_format: Vec<&str> = Vec::new();
        let mut event_format: Vec<&str> = Vec::new();

        for line in content.lines() {
            let trimmed = line.trim();

            // Check for section headers
            if trimmed.starts_with('[') && trimmed.ends_with(']') {
                current_section = trimmed[1..trimmed.len() - 1].to_string();
                continue;
            }

            // Skip empty lines and comments
            if trimmed.is_empty() || trimmed.starts_with(';') {
                continue;
            }

            match current_section.as_str() {
                "Script Info" => {
                    if let Some(colon_pos) = trimmed.find(':') {
                        let key = trimmed[..colon_pos].trim().to_string();
                        let value = trimmed[colon_pos + 1..].trim().to_string();
                        self.script_info.insert(key, value);
                    }
                }
                "V4 Styles" | "V4+ Styles" => {
                    if trimmed.starts_with("Format:") {
                        let format_line = &trimmed[7..];
                        style_format = format_line.split(',').map(|s| s.trim()).collect();
                    } else if trimmed.starts_with("Style:") {
                        let style_line = &trimmed[6..].trim();
                        self.parse_style(style_line, &style_format);
                    }
                }
                "Events" => {
                    if trimmed.starts_with("Format:") {
                        let format_line = &trimmed[7..];
                        event_format = format_line.split(',').map(|s| s.trim()).collect();
                    } else if trimmed.starts_with("Dialogue:") {
                        let dialogue_line = &trimmed[9..].trim();
                        if let Ok(subtitle) = self.parse_dialogue(dialogue_line, &event_format) {
                            subtitles.push(subtitle);
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(subtitles)
    }

    fn format(&self, subtitles: &[Subtitle]) -> Result<String> {
        let mut output = String::from("[Script Info]\n");
        output.push_str("Title: Generated by ZVD\n");
        output.push_str("ScriptType: v4.00+\n\n");

        output.push_str("[V4+ Styles]\n");
        output.push_str("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n");

        // Write default style
        let default_style = self.styles.get("Default").cloned().unwrap_or_default();
        output.push_str(&format!(
            "Style: {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n\n",
            default_style.name,
            default_style.font_name,
            default_style.font_size,
            default_style.primary_color,
            default_style.secondary_color,
            default_style.outline_color,
            default_style.back_color,
            if default_style.bold { -1 } else { 0 },
            if default_style.italic { -1 } else { 0 },
            if default_style.underline { -1 } else { 0 },
            if default_style.strikeout { -1 } else { 0 },
            default_style.scale_x,
            default_style.scale_y,
            default_style.spacing,
            default_style.angle,
            default_style.border_style,
            default_style.outline,
            default_style.shadow,
            default_style.alignment,
            default_style.margin_l,
            default_style.margin_r,
            default_style.margin_v,
            default_style.encoding,
        ));

        output.push_str("[Events]\n");
        output.push_str("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n");

        for subtitle in subtitles {
            let style = subtitle.style.as_deref().unwrap_or("Default");
            output.push_str(&format!(
                "Dialogue: 0,{},{},{},,,0,0,0,,{}\n",
                Self::format_ass_timestamp(subtitle.start_time),
                Self::format_ass_timestamp(subtitle.end_time),
                style,
                subtitle.text
            ));
        }

        Ok(output)
    }

    fn format_type(&self) -> SubtitleFormat {
        if self.is_ssa {
            SubtitleFormat::Ssa
        } else {
            SubtitleFormat::Ass
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ass_timestamp_parse() {
        let duration = AssParser::parse_ass_timestamp("0:01:23.45").unwrap();
        assert_eq!(duration.as_millis(), 83_450);
    }

    #[test]
    fn test_ass_timestamp_format() {
        let duration = Duration::from_millis(83_450);
        let timestamp = AssParser::format_ass_timestamp(duration);
        assert_eq!(timestamp, "0:01:23.45");
    }

    #[test]
    fn test_ass_parser_basic() {
        let content = r#"[Script Info]
Title: Test

[V4+ Styles]
Format: Name, Fontname, Fontsize
Style: Default,Arial,20

[Events]
Format: Layer, Start, End, Style, Text
Dialogue: 0,0:00:01.00,0:00:04.00,Default,First subtitle
"#;

        let mut parser = AssParser::new();
        let subtitles = parser.parse(content).unwrap();

        assert_eq!(subtitles.len(), 1);
        assert_eq!(subtitles[0].text, "First subtitle");
    }
}
