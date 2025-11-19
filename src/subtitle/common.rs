//! Common subtitle data structures and types
//!
//! Provides shared types used across all subtitle format implementations.

use std::time::Duration;

/// Subtitle format type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitleFormat {
    /// SubRip (.srt) format
    SRT,
    /// WebVTT (.vtt) format
    WebVTT,
    /// Advanced SubStation Alpha (.ass) format
    ASS,
    /// SubStation Alpha (.ssa) format
    SSA,
    /// CEA-608 closed captions
    CEA608,
    /// CEA-708 digital TV captions
    CEA708,
}

/// Position for subtitle display
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position {
    /// Horizontal position (0.0 = left, 1.0 = right)
    pub x: f32,
    /// Vertical position (0.0 = top, 1.0 = bottom)
    pub y: f32,
}

impl Default for Position {
    fn default() -> Self {
        Self { x: 0.5, y: 0.9 } // Bottom center
    }
}

/// Text alignment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Alignment {
    /// Left-aligned text
    Left,
    /// Center-aligned text
    Center,
    /// Right-aligned text
    Right,
}

impl Default for Alignment {
    fn default() -> Self {
        Self::Center
    }
}

/// Subtitle style information
#[derive(Debug, Clone, PartialEq)]
pub struct Style {
    /// Font name
    pub font_name: Option<String>,
    /// Font size (in points)
    pub font_size: Option<u16>,
    /// Bold text
    pub bold: bool,
    /// Italic text
    pub italic: bool,
    /// Underline text
    pub underline: bool,
    /// Text color (RGB)
    pub color: Option<(u8, u8, u8)>,
    /// Background color (RGB)
    pub background_color: Option<(u8, u8, u8)>,
    /// Text alignment
    pub alignment: Alignment,
}

impl Default for Style {
    fn default() -> Self {
        Self {
            font_name: None,
            font_size: None,
            bold: false,
            italic: false,
            underline: false,
            color: None,
            background_color: None,
            alignment: Alignment::default(),
        }
    }
}

/// A single subtitle cue (timed text entry)
#[derive(Debug, Clone, PartialEq)]
pub struct SubtitleCue {
    /// Sequence number (used in SRT, optional for others)
    pub id: usize,
    /// Start time of the subtitle
    pub start_time: Duration,
    /// End time of the subtitle
    pub end_time: Duration,
    /// Text content (may contain formatting tags)
    pub text: String,
    /// Display position (optional)
    pub position: Option<Position>,
    /// Style information (optional)
    pub style: Option<Style>,
}

impl SubtitleCue {
    /// Create a new subtitle cue
    pub fn new(id: usize, start_time: Duration, end_time: Duration, text: String) -> Self {
        Self {
            id,
            start_time,
            end_time,
            text,
            position: None,
            style: None,
        }
    }

    /// Get duration of this cue
    pub fn duration(&self) -> Duration {
        self.end_time.saturating_sub(self.start_time)
    }

    /// Check if this cue is active at the given time
    pub fn is_active_at(&self, time: Duration) -> bool {
        time >= self.start_time && time < self.end_time
    }
}

/// Container for subtitle tracks
#[derive(Debug, Clone, PartialEq)]
pub struct Subtitle {
    /// All subtitle cues
    pub cues: Vec<SubtitleCue>,
    /// Format of this subtitle
    pub format: SubtitleFormat,
}

impl Subtitle {
    /// Create a new empty subtitle
    pub fn new(format: SubtitleFormat) -> Self {
        Self {
            cues: Vec::new(),
            format,
        }
    }

    /// Add a cue to the subtitle
    pub fn add_cue(&mut self, cue: SubtitleCue) {
        self.cues.push(cue);
    }

    /// Sort cues by start time
    pub fn sort_by_time(&mut self) {
        self.cues.sort_by_key(|cue| cue.start_time);
    }

    /// Get all cues active at a given time
    pub fn get_active_cues(&self, time: Duration) -> Vec<&SubtitleCue> {
        self.cues
            .iter()
            .filter(|cue| cue.is_active_at(time))
            .collect()
    }

    /// Get total duration (end time of last cue)
    pub fn total_duration(&self) -> Duration {
        self.cues
            .iter()
            .map(|cue| cue.end_time)
            .max()
            .unwrap_or(Duration::ZERO)
    }

    /// Renumber all cues sequentially
    pub fn renumber(&mut self) {
        for (i, cue) in self.cues.iter_mut().enumerate() {
            cue.id = i + 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subtitle_cue_new() {
        let cue = SubtitleCue::new(
            1,
            Duration::from_secs(0),
            Duration::from_secs(2),
            "Hello, world!".to_string(),
        );
        assert_eq!(cue.id, 1);
        assert_eq!(cue.text, "Hello, world!");
        assert_eq!(cue.duration(), Duration::from_secs(2));
    }

    #[test]
    fn test_subtitle_cue_is_active() {
        let cue = SubtitleCue::new(
            1,
            Duration::from_secs(1),
            Duration::from_secs(3),
            "Test".to_string(),
        );

        assert!(!cue.is_active_at(Duration::from_millis(500)));
        assert!(cue.is_active_at(Duration::from_secs(1)));
        assert!(cue.is_active_at(Duration::from_secs(2)));
        assert!(!cue.is_active_at(Duration::from_secs(3)));
    }

    #[test]
    fn test_subtitle_new() {
        let mut sub = Subtitle::new(SubtitleFormat::SRT);
        assert_eq!(sub.cues.len(), 0);
        assert_eq!(sub.format, SubtitleFormat::SRT);

        sub.add_cue(SubtitleCue::new(
            1,
            Duration::from_secs(0),
            Duration::from_secs(2),
            "First".to_string(),
        ));

        assert_eq!(sub.cues.len(), 1);
    }

    #[test]
    fn test_subtitle_get_active_cues() {
        let mut sub = Subtitle::new(SubtitleFormat::SRT);
        sub.add_cue(SubtitleCue::new(
            1,
            Duration::from_secs(0),
            Duration::from_secs(2),
            "First".to_string(),
        ));
        sub.add_cue(SubtitleCue::new(
            2,
            Duration::from_secs(1),
            Duration::from_secs(3),
            "Second".to_string(),
        ));

        let active = sub.get_active_cues(Duration::from_millis(1500));
        assert_eq!(active.len(), 2);

        let active = sub.get_active_cues(Duration::from_millis(2500));
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].text, "Second");
    }

    #[test]
    fn test_subtitle_renumber() {
        let mut sub = Subtitle::new(SubtitleFormat::SRT);
        sub.add_cue(SubtitleCue::new(
            5,
            Duration::from_secs(0),
            Duration::from_secs(2),
            "First".to_string(),
        ));
        sub.add_cue(SubtitleCue::new(
            10,
            Duration::from_secs(2),
            Duration::from_secs(4),
            "Second".to_string(),
        ));

        sub.renumber();
        assert_eq!(sub.cues[0].id, 1);
        assert_eq!(sub.cues[1].id, 2);
    }
}
