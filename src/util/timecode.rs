//! SMPTE Timecode Support
//!
//! SMPTE (Society of Motion Picture and Television Engineers) timecode is the
//! standard for frame-accurate time representation in professional video production.
//!
//! ## Format
//!
//! Timecode is typically displayed as: `HH:MM:SS:FF`
//! - HH: Hours (00-23)
//! - MM: Minutes (00-59)
//! - SS: Seconds (00-59)
//! - FF: Frames (00-{fps-1})
//!
//! ## Drop Frame vs Non-Drop Frame
//!
//! **Non-Drop Frame (NDF):**
//! - Simple sequential frame counting
//! - Used for film (24fps) and integer frame rates (25fps, 30fps)
//! - Easy to calculate but drifts from real time at 29.97fps and 59.94fps
//!
//! **Drop Frame (DF):**
//! - Drops frame numbers (not actual frames!) to stay synchronized with real time
//! - Used for NTSC video (29.97fps, 59.94fps)
//! - Drops frames :00 and :01 every minute except every 10th minute
//! - Example: 00:01:00:00 is followed by 00:01:00:02 (skips :00 and :01)

use crate::error::{Error, Result};
use std::fmt;
use std::time::Duration;

/// SMPTE timecode frame rates
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameRate {
    /// 23.976 fps (film transferred to NTSC)
    Fps23_976,
    /// 24 fps (film)
    Fps24,
    /// 25 fps (PAL)
    Fps25,
    /// 29.97 fps (NTSC drop-frame)
    Fps29_97DF,
    /// 29.97 fps (NTSC non-drop-frame)
    Fps29_97NDF,
    /// 30 fps (non-drop-frame)
    Fps30,
    /// 50 fps (PAL high frame rate)
    Fps50,
    /// 59.94 fps (NTSC high frame rate drop-frame)
    Fps59_94DF,
    /// 59.94 fps (NTSC high frame rate non-drop-frame)
    Fps59_94NDF,
    /// 60 fps (non-drop-frame)
    Fps60,
}

impl FrameRate {
    /// Get the exact frame rate as a floating point number
    pub fn as_f64(self) -> f64 {
        match self {
            FrameRate::Fps23_976 => 24000.0 / 1001.0,
            FrameRate::Fps24 => 24.0,
            FrameRate::Fps25 => 25.0,
            FrameRate::Fps29_97DF | FrameRate::Fps29_97NDF => 30000.0 / 1001.0,
            FrameRate::Fps30 => 30.0,
            FrameRate::Fps50 => 50.0,
            FrameRate::Fps59_94DF | FrameRate::Fps59_94NDF => 60000.0 / 1001.0,
            FrameRate::Fps60 => 60.0,
        }
    }

    /// Get the nominal frame rate (rounded)
    pub fn nominal_fps(self) -> u32 {
        match self {
            FrameRate::Fps23_976 => 24,
            FrameRate::Fps24 => 24,
            FrameRate::Fps25 => 25,
            FrameRate::Fps29_97DF | FrameRate::Fps29_97NDF => 30,
            FrameRate::Fps30 => 30,
            FrameRate::Fps50 => 50,
            FrameRate::Fps59_94DF | FrameRate::Fps59_94NDF => 60,
            FrameRate::Fps60 => 60,
        }
    }

    /// Is this a drop-frame timecode?
    pub fn is_drop_frame(self) -> bool {
        matches!(self, FrameRate::Fps29_97DF | FrameRate::Fps59_94DF)
    }

    /// Get duration of a single frame
    pub fn frame_duration(self) -> Duration {
        Duration::from_secs_f64(1.0 / self.as_f64())
    }
}

impl fmt::Display for FrameRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FrameRate::Fps23_976 => write!(f, "23.976"),
            FrameRate::Fps24 => write!(f, "24"),
            FrameRate::Fps25 => write!(f, "25"),
            FrameRate::Fps29_97DF => write!(f, "29.97 DF"),
            FrameRate::Fps29_97NDF => write!(f, "29.97 NDF"),
            FrameRate::Fps30 => write!(f, "30"),
            FrameRate::Fps50 => write!(f, "50"),
            FrameRate::Fps59_94DF => write!(f, "59.94 DF"),
            FrameRate::Fps59_94NDF => write!(f, "59.94 NDF"),
            FrameRate::Fps60 => write!(f, "60"),
        }
    }
}

/// SMPTE timecode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timecode {
    /// Hours (0-23)
    pub hours: u8,
    /// Minutes (0-59)
    pub minutes: u8,
    /// Seconds (0-59)
    pub seconds: u8,
    /// Frames (0-fps-1)
    pub frames: u8,
    /// Frame rate
    frame_rate: FrameRate,
}

impl Timecode {
    /// Create a new timecode
    pub fn new(
        hours: u8,
        minutes: u8,
        seconds: u8,
        frames: u8,
        frame_rate: FrameRate,
    ) -> Result<Self> {
        // Validate ranges
        if hours > 23 {
            return Err(Error::InvalidInput(format!(
                "Hours must be 0-23, got {}",
                hours
            )));
        }
        if minutes > 59 {
            return Err(Error::InvalidInput(format!(
                "Minutes must be 0-59, got {}",
                minutes
            )));
        }
        if seconds > 59 {
            return Err(Error::InvalidInput(format!(
                "Seconds must be 0-59, got {}",
                seconds
            )));
        }
        if frames >= frame_rate.nominal_fps() as u8 {
            return Err(Error::InvalidInput(format!(
                "Frames must be 0-{}, got {}",
                frame_rate.nominal_fps() - 1,
                frames
            )));
        }

        Ok(Timecode {
            hours,
            minutes,
            seconds,
            frames,
            frame_rate,
        })
    }

    /// Create timecode from total frame count
    pub fn from_frames(total_frames: u64, frame_rate: FrameRate) -> Self {
        let fps = frame_rate.nominal_fps() as u64;

        if frame_rate.is_drop_frame() {
            // Drop-frame calculation
            // Drop 2 frames every minute except every 10th minute
            let frames_per_minute = fps * 60 - 2; // 1798 for 29.97 DF
            let frames_per_10_minutes = (fps * 60 * 10) - (2 * 9); // 17982 for 29.97 DF

            let mut frames_remaining = total_frames;

            // Calculate hours
            let frames_per_hour = frames_per_10_minutes * 6;
            let hours = (frames_remaining / frames_per_hour) as u8;
            frames_remaining %= frames_per_hour;

            // Calculate 10-minute groups
            let ten_minute_groups = frames_remaining / frames_per_10_minutes;
            frames_remaining %= frames_per_10_minutes;

            // Calculate minutes within 10-minute group
            let minutes_in_group = if frames_remaining < fps * 60 {
                0
            } else {
                (frames_remaining - fps * 60) / frames_per_minute + 1
            };

            let total_minutes = (ten_minute_groups * 10 + minutes_in_group) as u8;

            // Calculate remaining frames
            if minutes_in_group == 0 {
                frames_remaining %= (fps * 60);
            } else {
                frames_remaining = (frames_remaining - fps * 60) % frames_per_minute;
                // Add back dropped frames
                frames_remaining += 2;
            }

            let seconds = (frames_remaining / fps) as u8;
            let frames = (frames_remaining % fps) as u8;

            Timecode {
                hours,
                minutes: total_minutes,
                seconds,
                frames,
                frame_rate,
            }
        } else {
            // Non-drop-frame calculation (simple)
            let hours = (total_frames / (fps * 60 * 60)) as u8;
            let minutes = ((total_frames / (fps * 60)) % 60) as u8;
            let seconds = ((total_frames / fps) % 60) as u8;
            let frames = (total_frames % fps) as u8;

            Timecode {
                hours,
                minutes,
                seconds,
                frames,
                frame_rate,
            }
        }
    }

    /// Convert timecode to total frame count
    pub fn to_frames(&self) -> u64 {
        let fps = self.frame_rate.nominal_fps() as u64;

        if self.frame_rate.is_drop_frame() {
            // Drop-frame calculation
            let total_minutes = self.hours as u64 * 60 + self.minutes as u64;
            let frames_from_hours_minutes = total_minutes * (fps * 60 - 2);

            // Add back frames for every 10th minute (no drop)
            let ten_minute_groups = total_minutes / 10;
            let adjustment = ten_minute_groups * 2;

            frames_from_hours_minutes + adjustment + self.seconds as u64 * fps + self.frames as u64
        } else {
            // Non-drop-frame calculation (simple)
            let total_seconds =
                self.hours as u64 * 3600 + self.minutes as u64 * 60 + self.seconds as u64;
            total_seconds * fps + self.frames as u64
        }
    }

    /// Convert timecode to duration
    pub fn to_duration(&self) -> Duration {
        let frame_count = self.to_frames();
        Duration::from_secs_f64(frame_count as f64 / self.frame_rate.as_f64())
    }

    /// Create timecode from duration
    pub fn from_duration(duration: Duration, frame_rate: FrameRate) -> Self {
        let total_frames = (duration.as_secs_f64() * frame_rate.as_f64()) as u64;
        Self::from_frames(total_frames, frame_rate)
    }

    /// Add frames to timecode
    pub fn add_frames(&self, frames: i64) -> Self {
        let current_frames = self.to_frames() as i64;
        let new_frames = (current_frames + frames).max(0) as u64;
        Self::from_frames(new_frames, self.frame_rate)
    }

    /// Add duration to timecode
    pub fn add_duration(&self, duration: Duration) -> Self {
        let current = self.to_duration();
        let new_duration = current + duration;
        Self::from_duration(new_duration, self.frame_rate)
    }

    /// Subtract frames from timecode
    pub fn sub_frames(&self, frames: i64) -> Self {
        self.add_frames(-frames)
    }

    /// Subtract duration from timecode
    pub fn sub_duration(&self, duration: Duration) -> Self {
        let current = self.to_duration();
        let new_duration = current.saturating_sub(duration);
        Self::from_duration(new_duration, self.frame_rate)
    }

    /// Get frame rate
    pub fn frame_rate(&self) -> FrameRate {
        self.frame_rate
    }

    /// Parse timecode from string (HH:MM:SS:FF or HH:MM:SS;FF for drop-frame)
    pub fn parse(s: &str, frame_rate: FrameRate) -> Result<Self> {
        let parts: Vec<&str> = s.split(|c| c == ':' || c == ';').collect();

        if parts.len() != 4 {
            return Err(Error::InvalidInput(format!(
                "Invalid timecode format: {}. Expected HH:MM:SS:FF",
                s
            )));
        }

        let hours = parts[0]
            .parse::<u8>()
            .map_err(|_| Error::InvalidInput(format!("Invalid hours: {}", parts[0])))?;
        let minutes = parts[1]
            .parse::<u8>()
            .map_err(|_| Error::InvalidInput(format!("Invalid minutes: {}", parts[1])))?;
        let seconds = parts[2]
            .parse::<u8>()
            .map_err(|_| Error::InvalidInput(format!("Invalid seconds: {}", parts[2])))?;
        let frames = parts[3]
            .parse::<u8>()
            .map_err(|_| Error::InvalidInput(format!("Invalid frames: {}", parts[3])))?;

        Self::new(hours, minutes, seconds, frames, frame_rate)
    }
}

impl fmt::Display for Timecode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let separator = if self.frame_rate.is_drop_frame() {
            ';' // Semicolon for drop-frame
        } else {
            ':' // Colon for non-drop-frame
        };

        write!(
            f,
            "{:02}:{:02}:{:02}{}{}",
            self.hours,
            self.minutes,
            self.seconds,
            separator,
            format!("{:02}", self.frames)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timecode_creation() {
        let tc = Timecode::new(1, 23, 45, 12, FrameRate::Fps30).unwrap();
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 23);
        assert_eq!(tc.seconds, 45);
        assert_eq!(tc.frames, 12);
    }

    #[test]
    fn test_timecode_validation() {
        assert!(Timecode::new(24, 0, 0, 0, FrameRate::Fps30).is_err());
        assert!(Timecode::new(0, 60, 0, 0, FrameRate::Fps30).is_err());
        assert!(Timecode::new(0, 0, 60, 0, FrameRate::Fps30).is_err());
        assert!(Timecode::new(0, 0, 0, 30, FrameRate::Fps30).is_err());
    }

    #[test]
    fn test_non_drop_frame_conversion() {
        let tc = Timecode::new(0, 1, 0, 0, FrameRate::Fps30).unwrap();
        let frames = tc.to_frames();
        assert_eq!(frames, 1800); // 60 seconds * 30 fps

        let tc2 = Timecode::from_frames(1800, FrameRate::Fps30);
        assert_eq!(tc2.hours, 0);
        assert_eq!(tc2.minutes, 1);
        assert_eq!(tc2.seconds, 0);
        assert_eq!(tc2.frames, 0);
    }

    #[test]
    fn test_timecode_display() {
        let tc = Timecode::new(1, 23, 45, 12, FrameRate::Fps30).unwrap();
        assert_eq!(format!("{}", tc), "01:23:45:12");

        let tc_df = Timecode::new(1, 23, 45, 12, FrameRate::Fps29_97DF).unwrap();
        assert_eq!(format!("{}", tc_df), "01:23:45;12"); // Semicolon for drop-frame
    }

    #[test]
    fn test_timecode_parse() {
        let tc = Timecode::parse("01:23:45:12", FrameRate::Fps30).unwrap();
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 23);
        assert_eq!(tc.seconds, 45);
        assert_eq!(tc.frames, 12);
    }

    #[test]
    fn test_timecode_arithmetic() {
        let tc = Timecode::new(0, 0, 0, 10, FrameRate::Fps30).unwrap();
        let tc2 = tc.add_frames(20);
        assert_eq!(tc2.frames, 0);
        assert_eq!(tc2.seconds, 1);

        let tc3 = tc2.sub_frames(15);
        assert_eq!(tc3.frames, 15);
        assert_eq!(tc3.seconds, 0);
    }

    #[test]
    fn test_duration_conversion() {
        let tc = Timecode::new(0, 1, 0, 0, FrameRate::Fps30).unwrap();
        let duration = tc.to_duration();
        assert_eq!(duration.as_secs(), 60);

        let tc2 = Timecode::from_duration(Duration::from_secs(60), FrameRate::Fps30);
        assert_eq!(tc2.hours, 0);
        assert_eq!(tc2.minutes, 1);
        assert_eq!(tc2.seconds, 0);
        assert_eq!(tc2.frames, 0);
    }

    #[test]
    fn test_frame_rate_properties() {
        assert_eq!(FrameRate::Fps24.nominal_fps(), 24);
        assert_eq!(FrameRate::Fps29_97DF.nominal_fps(), 30);
        assert!(FrameRate::Fps29_97DF.is_drop_frame());
        assert!(!FrameRate::Fps30.is_drop_frame());
    }
}
