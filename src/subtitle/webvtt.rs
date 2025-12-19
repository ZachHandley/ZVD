//! WebVTT (Web Video Text Tracks) subtitle format parser
//!
//! WebVTT is the W3C standard for displaying timed text on the web.

use super::{parse_timestamp, Subtitle, SubtitleFormat, SubtitleParser};
use crate::error::{Error, Result};
use std::time::Duration;

/// WebVTT cue
pub type WebVttCue = Subtitle;

/// WebVTT parser
#[derive(Debug, Default)]
pub struct WebVttParser {
    header_settings: Vec<String>,
}

impl WebVttParser {
    pub fn new() -> Self {
        WebVttParser {
            header_settings: Vec::new(),
        }
    }

    /// Format duration as WebVTT timestamp
    fn format_timestamp(duration: Duration) -> String {
        let total_millis = duration.as_millis();
        let hours = total_millis / 3600_000;
        let minutes = (total_millis % 3600_000) / 60_000;
        let seconds = (total_millis % 60_000) / 1000;
        let millis = total_millis % 1000;

        if hours > 0 {
            format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
        } else {
            format!("{:02}:{:02}.{:03}", minutes, seconds, millis)
        }
    }
}

impl SubtitleParser for WebVttParser {
    fn parse(&mut self, content: &str) -> Result<Vec<Subtitle>> {
        let mut lines = content.lines().peekable();

        // Check for WEBVTT header
        let first_line = lines
            .next()
            .ok_or_else(|| Error::invalid_input("Empty WebVTT file"))?;

        if !first_line.starts_with("WEBVTT") {
            return Err(Error::invalid_input("Invalid WebVTT header"));
        }

        // Parse optional header settings
        while let Some(line) = lines.peek() {
            if line.trim().is_empty() {
                lines.next();
                break;
            }
            if line.contains("-->") {
                break; // Start of cues
            }
            self.header_settings.push(lines.next().unwrap().to_string());
        }

        let mut subtitles = Vec::new();

        while lines.peek().is_some() {
            // Skip empty lines
            while let Some(line) = lines.peek() {
                if !line.trim().is_empty() {
                    break;
                }
                lines.next();
            }

            if lines.peek().is_none() {
                break;
            }

            // Check for optional cue identifier
            let first_line = lines.peek().unwrap();
            let cue_id = if !first_line.contains("-->") {
                Some(lines.next().unwrap().trim().to_string())
            } else {
                None
            };

            // Read timestamp line
            let timestamp_line = lines
                .next()
                .ok_or_else(|| Error::invalid_input("Missing timestamp line"))?;

            // Parse timestamp and optional settings
            let parts: Vec<&str> = timestamp_line.split_whitespace().collect();
            if parts.len() < 3 || parts[1] != "-->" {
                return Err(Error::invalid_input("Invalid timestamp format"));
            }

            let start_time = parse_timestamp(parts[0])?;
            let end_time = parse_timestamp(parts[2])?;

            // Optional cue settings (position, align, etc.)
            let _cue_settings: Vec<&str> = parts[3..].to_vec();

            // Read cue text (until empty line or NOTE)
            let mut text_lines = Vec::new();
            while let Some(line) = lines.peek() {
                if line.trim().is_empty() || line.starts_with("NOTE") {
                    break;
                }
                text_lines.push(lines.next().unwrap().to_string());
            }

            let text = text_lines.join("\n");

            let subtitle = Subtitle::new(start_time, end_time, text);
            subtitles.push(subtitle);
        }

        Ok(subtitles)
    }

    fn format(&self, subtitles: &[Subtitle]) -> Result<String> {
        let mut output = String::from("WEBVTT\n\n");

        for subtitle in subtitles {
            // Timestamp
            output.push_str(&format!(
                "{} --> {}\n",
                Self::format_timestamp(subtitle.start_time),
                Self::format_timestamp(subtitle.end_time)
            ));

            // Text
            output.push_str(&subtitle.text);
            output.push_str("\n\n");
        }

        Ok(output)
    }

    fn format_type(&self) -> SubtitleFormat {
        SubtitleFormat::WebVtt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webvtt_parser() {
        let content = r#"WEBVTT

00:00:01.000 --> 00:00:04.000
First subtitle

00:00:05.000 --> 00:00:08.000
Second subtitle
"#;

        let mut parser = WebVttParser::new();
        let subtitles = parser.parse(content).unwrap();

        assert_eq!(subtitles.len(), 2);
        assert_eq!(subtitles[0].text, "First subtitle");
        assert_eq!(subtitles[1].text, "Second subtitle");
    }

    #[test]
    fn test_webvtt_with_cue_id() {
        let content = r#"WEBVTT

cue1
00:00:01.000 --> 00:00:04.000
First subtitle
"#;

        let mut parser = WebVttParser::new();
        let subtitles = parser.parse(content).unwrap();

        assert_eq!(subtitles.len(), 1);
        assert_eq!(subtitles[0].text, "First subtitle");
    }

    #[test]
    fn test_webvtt_format() {
        let subtitles = vec![Subtitle {
            index: None,
            start_time: Duration::from_secs(1),
            end_time: Duration::from_secs(4),
            text: "Hello".to_string(),
            style: None,
        }];

        let parser = WebVttParser::new();
        let output = parser.format(&subtitles).unwrap();

        assert!(output.starts_with("WEBVTT\n"));
        assert!(output.contains("00:01.000 --> 00:04.000"));
        assert!(output.contains("Hello"));
    }

    #[test]
    fn test_webvtt_timestamp_format() {
        let duration = Duration::from_millis(83_456);
        let timestamp = WebVttParser::format_timestamp(duration);
        assert_eq!(timestamp, "01:23.456");
    }
}
