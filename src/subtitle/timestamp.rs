//! Timestamp parsing and formatting for subtitle formats
//!
//! Supports SRT format (HH:MM:SS,mmm) and WebVTT format (HH:MM:SS.mmm)

use std::time::Duration;
use crate::error::{Error, Result};

/// Parse SRT timestamp format: HH:MM:SS,mmm
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use zvd_lib::subtitle::timestamp::parse_srt_timestamp;
///
/// let duration = parse_srt_timestamp("00:01:23,456").unwrap();
/// assert_eq!(duration, Duration::from_millis(83456));
/// ```
pub fn parse_srt_timestamp(s: &str) -> Result<Duration> {
    let parts: Vec<&str> = s.split(&[',', ':']).collect();

    if parts.len() != 4 {
        return Err(Error::InvalidData(format!(
            "Invalid SRT timestamp format: '{}' (expected HH:MM:SS,mmm)",
            s
        )));
    }

    let hours: u64 = parts[0].parse().map_err(|_| {
        Error::InvalidData(format!("Invalid hours in timestamp: '{}'", parts[0]))
    })?;

    let minutes: u64 = parts[1].parse().map_err(|_| {
        Error::InvalidData(format!("Invalid minutes in timestamp: '{}'", parts[1]))
    })?;

    let seconds: u64 = parts[2].parse().map_err(|_| {
        Error::InvalidData(format!("Invalid seconds in timestamp: '{}'", parts[2]))
    })?;

    let millis: u64 = parts[3].parse().map_err(|_| {
        Error::InvalidData(format!("Invalid milliseconds in timestamp: '{}'", parts[3]))
    })?;

    // Validate ranges
    if minutes >= 60 {
        return Err(Error::InvalidData(format!(
            "Minutes must be 0-59, got: {}",
            minutes
        )));
    }

    if seconds >= 60 {
        return Err(Error::InvalidData(format!(
            "Seconds must be 0-59, got: {}",
            seconds
        )));
    }

    if millis >= 1000 {
        return Err(Error::InvalidData(format!(
            "Milliseconds must be 0-999, got: {}",
            millis
        )));
    }

    let total_millis = hours * 3600 * 1000 + minutes * 60 * 1000 + seconds * 1000 + millis;
    Ok(Duration::from_millis(total_millis))
}

/// Format Duration as SRT timestamp: HH:MM:SS,mmm
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use zvd_lib::subtitle::timestamp::format_srt_timestamp;
///
/// let timestamp = format_srt_timestamp(Duration::from_millis(83456));
/// assert_eq!(timestamp, "00:01:23,456");
/// ```
pub fn format_srt_timestamp(duration: Duration) -> String {
    let total_millis = duration.as_millis();
    let hours = total_millis / (3600 * 1000);
    let minutes = (total_millis % (3600 * 1000)) / (60 * 1000);
    let seconds = (total_millis % (60 * 1000)) / 1000;
    let millis = total_millis % 1000;

    format!("{:02}:{:02}:{:02},{:03}", hours, minutes, seconds, millis)
}

/// Parse WebVTT timestamp format: HH:MM:SS.mmm or MM:SS.mmm
///
/// WebVTT allows hours to be omitted if less than 1 hour.
pub fn parse_webvtt_timestamp(s: &str) -> Result<Duration> {
    let parts: Vec<&str> = s.split(&['.', ':']).collect();

    if parts.len() < 3 || parts.len() > 4 {
        return Err(Error::InvalidData(format!(
            "Invalid WebVTT timestamp format: '{}' (expected HH:MM:SS.mmm or MM:SS.mmm)",
            s
        )));
    }

    let (hours, minutes, seconds, millis) = if parts.len() == 4 {
        // HH:MM:SS.mmm
        let h: u64 = parts[0].parse().map_err(|_| {
            Error::InvalidData(format!("Invalid hours in timestamp: '{}'", parts[0]))
        })?;
        let m: u64 = parts[1].parse().map_err(|_| {
            Error::InvalidData(format!("Invalid minutes in timestamp: '{}'", parts[1]))
        })?;
        let s: u64 = parts[2].parse().map_err(|_| {
            Error::InvalidData(format!("Invalid seconds in timestamp: '{}'", parts[2]))
        })?;
        let ms: u64 = parts[3].parse().map_err(|_| {
            Error::InvalidData(format!("Invalid milliseconds in timestamp: '{}'", parts[3]))
        })?;
        (h, m, s, ms)
    } else {
        // MM:SS.mmm
        let m: u64 = parts[0].parse().map_err(|_| {
            Error::InvalidData(format!("Invalid minutes in timestamp: '{}'", parts[0]))
        })?;
        let s: u64 = parts[1].parse().map_err(|_| {
            Error::InvalidData(format!("Invalid seconds in timestamp: '{}'", parts[1]))
        })?;
        let ms: u64 = parts[2].parse().map_err(|_| {
            Error::InvalidData(format!("Invalid milliseconds in timestamp: '{}'", parts[2]))
        })?;
        (0, m, s, ms)
    };

    // Validate ranges
    if minutes >= 60 {
        return Err(Error::InvalidData(format!(
            "Minutes must be 0-59, got: {}",
            minutes
        )));
    }

    if seconds >= 60 {
        return Err(Error::InvalidData(format!(
            "Seconds must be 0-59, got: {}",
            seconds
        )));
    }

    if millis >= 1000 {
        return Err(Error::InvalidData(format!(
            "Milliseconds must be 0-999, got: {}",
            millis
        )));
    }

    let total_millis = hours * 3600 * 1000 + minutes * 60 * 1000 + seconds * 1000 + millis;
    Ok(Duration::from_millis(total_millis))
}

/// Format Duration as WebVTT timestamp: HH:MM:SS.mmm
pub fn format_webvtt_timestamp(duration: Duration) -> String {
    let total_millis = duration.as_millis();
    let hours = total_millis / (3600 * 1000);
    let minutes = (total_millis % (3600 * 1000)) / (60 * 1000);
    let seconds = (total_millis % (60 * 1000)) / 1000;
    let millis = total_millis % 1000;

    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_srt_timestamp_basic() {
        let duration = parse_srt_timestamp("00:00:00,000").unwrap();
        assert_eq!(duration, Duration::from_millis(0));

        let duration = parse_srt_timestamp("00:00:01,000").unwrap();
        assert_eq!(duration, Duration::from_millis(1000));

        let duration = parse_srt_timestamp("00:01:00,000").unwrap();
        assert_eq!(duration, Duration::from_millis(60000));

        let duration = parse_srt_timestamp("01:00:00,000").unwrap();
        assert_eq!(duration, Duration::from_millis(3600000));
    }

    #[test]
    fn test_parse_srt_timestamp_complex() {
        let duration = parse_srt_timestamp("01:23:45,678").unwrap();
        let expected = Duration::from_millis(1 * 3600000 + 23 * 60000 + 45 * 1000 + 678);
        assert_eq!(duration, expected);
    }

    #[test]
    fn test_parse_srt_timestamp_invalid() {
        assert!(parse_srt_timestamp("invalid").is_err());
        assert!(parse_srt_timestamp("00:00:60,000").is_err()); // Seconds > 59
        assert!(parse_srt_timestamp("00:60:00,000").is_err()); // Minutes > 59
        assert!(parse_srt_timestamp("00:00:00,1000").is_err()); // Milliseconds >= 1000
    }

    #[test]
    fn test_format_srt_timestamp() {
        assert_eq!(format_srt_timestamp(Duration::from_millis(0)), "00:00:00,000");
        assert_eq!(format_srt_timestamp(Duration::from_millis(1000)), "00:00:01,000");
        assert_eq!(format_srt_timestamp(Duration::from_millis(60000)), "00:01:00,000");
        assert_eq!(format_srt_timestamp(Duration::from_millis(3600000)), "01:00:00,000");

        let expected = Duration::from_millis(1 * 3600000 + 23 * 60000 + 45 * 1000 + 678);
        assert_eq!(format_srt_timestamp(expected), "01:23:45,678");
    }

    #[test]
    fn test_srt_timestamp_roundtrip() {
        let original = "01:23:45,678";
        let parsed = parse_srt_timestamp(original).unwrap();
        let formatted = format_srt_timestamp(parsed);
        assert_eq!(original, formatted);
    }

    #[test]
    fn test_parse_webvtt_timestamp_with_hours() {
        let duration = parse_webvtt_timestamp("01:23:45.678").unwrap();
        let expected = Duration::from_millis(1 * 3600000 + 23 * 60000 + 45 * 1000 + 678);
        assert_eq!(duration, expected);
    }

    #[test]
    fn test_parse_webvtt_timestamp_without_hours() {
        let duration = parse_webvtt_timestamp("23:45.678").unwrap();
        let expected = Duration::from_millis(23 * 60000 + 45 * 1000 + 678);
        assert_eq!(duration, expected);
    }

    #[test]
    fn test_format_webvtt_timestamp() {
        let expected = Duration::from_millis(1 * 3600000 + 23 * 60000 + 45 * 1000 + 678);
        assert_eq!(format_webvtt_timestamp(expected), "01:23:45.678");
    }

    #[test]
    fn test_webvtt_timestamp_roundtrip() {
        let original = "01:23:45.678";
        let parsed = parse_webvtt_timestamp(original).unwrap();
        let formatted = format_webvtt_timestamp(parsed);
        assert_eq!(original, formatted);
    }
}
