//! SRT (SubRip) subtitle format parser
//!
//! SRT is one of the most popular and widely supported subtitle formats.

use super::{Subtitle, SubtitleParser, SubtitleFormat, parse_timestamp};
use crate::error::{Error, Result};
use std::time::Duration;

/// SRT subtitle entry
pub type SrtSubtitle = Subtitle;

/// SRT subtitle parser
#[derive(Debug, Default)]
pub struct SrtParser;

impl SrtParser {
    pub fn new() -> Self {
        SrtParser
    }

    /// Format duration as SRT timestamp
    fn format_timestamp(duration: Duration) -> String {
        let total_millis = duration.as_millis();
        let hours = total_millis / 3600_000;
        let minutes = (total_millis % 3600_000) / 60_000;
        let seconds = (total_millis % 60_000) / 1000;
        let millis = total_millis % 1000;

        format!("{:02}:{:02}:{:02},{:03}", hours, minutes, seconds, millis)
    }
}

impl SubtitleParser for SrtParser {
    fn parse(&mut self, content: &str) -> Result<Vec<Subtitle>> {
        let mut subtitles = Vec::new();
        let mut lines = content.lines().peekable();

        while lines.peek().is_some() {
            // Read index
            let index_line = match lines.next() {
                Some(line) if !line.trim().is_empty() => line.trim(),
                Some(_) => continue, // Skip empty lines
                None => break,
            };

            let index: u32 = index_line.parse()
                .map_err(|_| Error::invalid_input("Invalid subtitle index"))?;

            // Read timestamp line
            let timestamp_line = lines.next()
                .ok_or_else(|| Error::invalid_input("Missing timestamp line"))?;

            let timestamps: Vec<&str> = timestamp_line.split(" --> ").collect();
            if timestamps.len() != 2 {
                return Err(Error::invalid_input("Invalid timestamp format"));
            }

            let start_time = parse_timestamp(timestamps[0].trim())?;
            let end_time = parse_timestamp(timestamps[1].trim())?;

            // Read subtitle text (until empty line)
            let mut text_lines = Vec::new();
            while let Some(line) = lines.peek() {
                if line.trim().is_empty() {
                    lines.next(); // Consume empty line
                    break;
                }
                text_lines.push(lines.next().unwrap().to_string());
            }

            let text = text_lines.join("\n");

            let mut subtitle = Subtitle::new(start_time, end_time, text);
            subtitle.index = Some(index);
            subtitles.push(subtitle);
        }

        Ok(subtitles)
    }

    fn format(&self, subtitles: &[Subtitle]) -> Result<String> {
        let mut output = String::new();

        for (idx, subtitle) in subtitles.iter().enumerate() {
            let index = subtitle.index.unwrap_or(idx as u32 + 1);

            // Index
            output.push_str(&format!("{}\n", index));

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
        SubtitleFormat::Srt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srt_parser() {
        let content = r#"1
00:00:01,000 --> 00:00:04,000
First subtitle

2
00:00:05,000 --> 00:00:08,000
Second subtitle
with multiple lines
"#;

        let mut parser = SrtParser::new();
        let subtitles = parser.parse(content).unwrap();

        assert_eq!(subtitles.len(), 2);
        assert_eq!(subtitles[0].index, Some(1));
        assert_eq!(subtitles[0].text, "First subtitle");
        assert_eq!(subtitles[1].text, "Second subtitle\nwith multiple lines");
    }

    #[test]
    fn test_srt_format() {
        let subtitles = vec![
            Subtitle {
                index: Some(1),
                start_time: Duration::from_secs(1),
                end_time: Duration::from_secs(4),
                text: "Hello".to_string(),
                style: None,
            },
        ];

        let parser = SrtParser::new();
        let output = parser.format(&subtitles).unwrap();

        assert!(output.contains("1\n"));
        assert!(output.contains("00:00:01,000 --> 00:00:04,000"));
        assert!(output.contains("Hello"));
    }

    #[test]
    fn test_srt_timestamp_format() {
        let duration = Duration::from_millis(83_456);
        let timestamp = SrtParser::format_timestamp(duration);
        assert_eq!(timestamp, "00:01:23,456");
    }
}
