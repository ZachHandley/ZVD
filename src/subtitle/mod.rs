//! Subtitle support
//!
//! Support for common subtitle formats including SRT, WebVTT, and ASS/SSA.

pub mod srt;
pub mod webvtt;
pub mod ass;

pub use srt::{SrtParser, SrtSubtitle};
pub use webvtt::{WebVttParser, WebVttCue};
pub use ass::{AssParser, AssEvent, AssStyle};

use crate::error::Result;
use std::time::Duration;

/// Subtitle format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitleFormat {
    Srt,
    WebVtt,
    Ass,
    Ssa,
    SubRip,
    MicroDvd,
    SubViewer,
}

/// Generic subtitle entry
#[derive(Debug, Clone)]
pub struct Subtitle {
    pub index: Option<u32>,
    pub start_time: Duration,
    pub end_time: Duration,
    pub text: String,
    pub style: Option<String>,
}

impl Subtitle {
    pub fn new(start: Duration, end: Duration, text: String) -> Self {
        Subtitle {
            index: None,
            start_time: start,
            end_time: end,
            text,
            style: None,
        }
    }

    /// Get duration of subtitle
    pub fn duration(&self) -> Duration {
        self.end_time.saturating_sub(self.start_time)
    }

    /// Check if subtitle is active at given time
    pub fn is_active_at(&self, time: Duration) -> bool {
        time >= self.start_time && time < self.end_time
    }
}

/// Subtitle parser trait
pub trait SubtitleParser {
    /// Parse subtitles from text
    fn parse(&mut self, content: &str) -> Result<Vec<Subtitle>>;

    /// Format subtitles to text
    fn format(&self, subtitles: &[Subtitle]) -> Result<String>;

    /// Get format
    fn format_type(&self) -> SubtitleFormat;
}

/// Parse timestamp in various formats
pub fn parse_timestamp(s: &str) -> Result<Duration> {
    // Try HH:MM:SS,mmm format (SRT)
    if let Some(duration) = parse_srt_timestamp(s) {
        return Ok(duration);
    }

    // Try HH:MM:SS.mmm format (WebVTT, ASS)
    if let Some(duration) = parse_webvtt_timestamp(s) {
        return Ok(duration);
    }

    Err(crate::error::Error::invalid_input("Invalid timestamp format"))
}

fn parse_srt_timestamp(s: &str) -> Option<Duration> {
    // Format: HH:MM:SS,mmm
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 2 {
        return None;
    }

    let time_parts: Vec<&str> = parts[0].split(':').collect();
    if time_parts.len() != 3 {
        return None;
    }

    let hours: u64 = time_parts[0].parse().ok()?;
    let minutes: u64 = time_parts[1].parse().ok()?;
    let seconds: u64 = time_parts[2].parse().ok()?;
    let millis: u64 = parts[1].parse().ok()?;

    Some(Duration::from_millis(
        hours * 3600_000 + minutes * 60_000 + seconds * 1000 + millis,
    ))
}

fn parse_webvtt_timestamp(s: &str) -> Option<Duration> {
    // Format: HH:MM:SS.mmm or MM:SS.mmm
    let parts: Vec<&str> = s.split('.').collect();
    if parts.is_empty() {
        return None;
    }

    let time_parts: Vec<&str> = parts[0].split(':').collect();
    let millis: u64 = if parts.len() > 1 {
        parts[1].parse().ok()?
    } else {
        0
    };

    let (hours, minutes, seconds): (u64, u64, u64) = match time_parts.len() {
        2 => (0, time_parts[0].parse().ok()?, time_parts[1].parse().ok()?),
        3 => (
            time_parts[0].parse().ok()?,
            time_parts[1].parse().ok()?,
            time_parts[2].parse().ok()?,
        ),
        _ => return None,
    };

    Some(Duration::from_millis(
        hours * 3600_000 + minutes * 60_000 + seconds * 1000 + millis,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subtitle_creation() {
        let sub = Subtitle::new(
            Duration::from_secs(10),
            Duration::from_secs(15),
            "Hello, world!".to_string(),
        );
        assert_eq!(sub.duration(), Duration::from_secs(5));
    }

    #[test]
    fn test_subtitle_is_active() {
        let sub = Subtitle::new(
            Duration::from_secs(10),
            Duration::from_secs(15),
            "Hello".to_string(),
        );
        assert!(!sub.is_active_at(Duration::from_secs(5)));
        assert!(sub.is_active_at(Duration::from_secs(12)));
        assert!(!sub.is_active_at(Duration::from_secs(20)));
    }

    #[test]
    fn test_parse_srt_timestamp() {
        let duration = parse_timestamp("00:01:23,456").unwrap();
        assert_eq!(duration.as_millis(), 83_456);
    }

    #[test]
    fn test_parse_webvtt_timestamp() {
        let duration = parse_timestamp("00:01:23.456").unwrap();
        assert_eq!(duration.as_millis(), 83_456);
    }
}
