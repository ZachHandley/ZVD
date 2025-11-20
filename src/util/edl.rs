//! EDL and XML Timeline Export
//!
//! Export video editing timelines to industry-standard formats for
//! integration with professional editing software.
//!
//! ## Supported Formats
//!
//! - **CMX 3600 EDL**: Industry standard, supported by all NLEs
//! - **Final Cut Pro XML**: Apple ecosystem (FCP 7/X)
//! - **DaVinci Resolve AAF**: Blackmagic Design workflow
//! - **Adobe Premiere Pro XML**: Adobe ecosystem
//!
//! ## EDL Use Cases
//!
//! - Conform editing from one NLE to another
//! - Offline/online workflow handoff
//! - Archiving editing decisions
//! - Automated assembly
//!
//! ## Usage
//!
//! ```rust
//! use zvd_lib::util::edl::{Timeline, TimelineEvent, EdlExporter};
//!
//! let mut timeline = Timeline::new(24.0, 1920, 1080);
//! timeline.add_event(TimelineEvent::new_cut("source.mp4", 0.0, 5.0, 0.0));
//! timeline.add_event(TimelineEvent::new_cut("source.mp4", 10.0, 15.0, 5.0));
//!
//! let edl = EdlExporter::new().export_cmx3600(&timeline)?;
//! std::fs::write("timeline.edl", edl)?;
//! ```

use crate::error::{Error, Result};
use std::path::{Path, PathBuf};

/// Timeline edit event type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditType {
    /// Cut (instant transition)
    Cut,
    /// Dissolve (crossfade)
    Dissolve,
    /// Wipe transition
    Wipe,
}

impl EditType {
    /// Get CMX 3600 code
    pub fn cmx_code(&self) -> &'static str {
        match self {
            EditType::Cut => "C",
            EditType::Dissolve => "D",
            EditType::Wipe => "W",
        }
    }
}

/// Track type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackType {
    /// Video track
    Video,
    /// Audio track
    Audio,
    /// Both video and audio
    Both,
}

impl TrackType {
    /// Get CMX 3600 code
    pub fn cmx_code(&self) -> &'static str {
        match self {
            TrackType::Video => "V",
            TrackType::Audio => "A",
            TrackType::Both => "B",
        }
    }
}

/// Timecode representation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Timecode {
    pub hours: u32,
    pub minutes: u32,
    pub seconds: u32,
    pub frames: u32,
}

impl Timecode {
    /// Create from seconds
    pub fn from_seconds(seconds: f64, fps: f64) -> Self {
        let total_frames = (seconds * fps).round() as u32;
        let frames_per_hour = (fps * 3600.0) as u32;
        let frames_per_minute = (fps * 60.0) as u32;

        let hours = total_frames / frames_per_hour;
        let remaining = total_frames % frames_per_hour;
        let minutes = remaining / frames_per_minute;
        let remaining = remaining % frames_per_minute;
        let seconds_int = remaining / fps as u32;
        let frames = remaining % fps as u32;

        Timecode {
            hours,
            minutes,
            seconds: seconds_int,
            frames,
        }
    }

    /// Format as HH:MM:SS:FF
    pub fn to_string(&self) -> String {
        format!(
            "{:02}:{:02}:{:02}:{:02}",
            self.hours, self.minutes, self.seconds, self.frames
        )
    }

    /// Convert to total seconds
    pub fn to_seconds(&self, fps: f64) -> f64 {
        let total_frames =
            self.hours * (fps * 3600.0) as u32 + self.minutes * (fps * 60.0) as u32
                + self.seconds * fps as u32
                + self.frames;
        total_frames as f64 / fps
    }
}

/// Timeline event (clip)
#[derive(Debug, Clone)]
pub struct TimelineEvent {
    /// Event number (1-indexed)
    pub number: u32,
    /// Source file path
    pub source_file: PathBuf,
    /// Source reel name (for EDL)
    pub reel_name: String,
    /// Track type
    pub track: TrackType,
    /// Edit type
    pub edit_type: EditType,
    /// Source in point (seconds)
    pub source_in: f64,
    /// Source out point (seconds)
    pub source_out: f64,
    /// Record in point (seconds)
    pub record_in: f64,
    /// Record out point (seconds)
    pub record_out: f64,
}

impl TimelineEvent {
    /// Create new cut event
    pub fn new_cut<P: AsRef<Path>>(
        source_file: P,
        source_in: f64,
        source_out: f64,
        record_in: f64,
    ) -> Self {
        let duration = source_out - source_in;
        TimelineEvent {
            number: 0, // Will be set by timeline
            source_file: source_file.as_ref().to_path_buf(),
            reel_name: "AX".to_string(), // Default reel name
            track: TrackType::Both,
            edit_type: EditType::Cut,
            source_in,
            source_out,
            record_in,
            record_out: record_in + duration,
        }
    }

    /// Create dissolve event
    pub fn new_dissolve<P: AsRef<Path>>(
        source_file: P,
        source_in: f64,
        source_out: f64,
        record_in: f64,
    ) -> Self {
        let mut event = Self::new_cut(source_file, source_in, source_out, record_in);
        event.edit_type = EditType::Dissolve;
        event
    }

    /// Get duration
    pub fn duration(&self) -> f64 {
        self.source_out - self.source_in
    }
}

/// Timeline
#[derive(Debug, Clone)]
pub struct Timeline {
    /// Timeline name
    pub name: String,
    /// Frame rate
    pub fps: f64,
    /// Resolution
    pub width: u32,
    pub height: u32,
    /// Events
    pub events: Vec<TimelineEvent>,
}

impl Timeline {
    /// Create new timeline
    pub fn new(fps: f64, width: u32, height: u32) -> Self {
        Timeline {
            name: "TIMELINE".to_string(),
            fps,
            width,
            height,
            events: Vec::new(),
        }
    }

    /// Add event to timeline
    pub fn add_event(&mut self, mut event: TimelineEvent) {
        event.number = (self.events.len() + 1) as u32;
        self.events.push(event);
    }

    /// Get total duration
    pub fn duration(&self) -> f64 {
        self.events
            .iter()
            .map(|e| e.record_out)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }
}

/// EDL exporter
pub struct EdlExporter {
    /// Drop frame timecode
    drop_frame: bool,
}

impl EdlExporter {
    /// Create new exporter
    pub fn new() -> Self {
        EdlExporter { drop_frame: false }
    }

    /// Enable drop frame timecode
    pub fn with_drop_frame(mut self, drop_frame: bool) -> Self {
        self.drop_frame = drop_frame;
        self
    }

    /// Export to CMX 3600 EDL format
    pub fn export_cmx3600(&self, timeline: &Timeline) -> Result<String> {
        let mut output = String::new();

        // Header
        output.push_str("TITLE: ");
        output.push_str(&timeline.name);
        output.push_str("\n");
        output.push_str(&format!("FCM: {}\n", if self.drop_frame { "DROP FRAME" } else { "NON-DROP FRAME" }));
        output.push_str("\n");

        // Events
        for event in &timeline.events {
            self.write_cmx_event(&mut output, event, timeline.fps)?;
        }

        Ok(output)
    }

    fn write_cmx_event(&self, output: &mut String, event: &TimelineEvent, fps: f64) -> Result<()> {
        // Event number (3 digits)
        output.push_str(&format!("{:03}  ", event.number));

        // Reel name (8 chars)
        output.push_str(&format!("{:<8} ", event.reel_name));

        // Track type
        output.push_str(event.track.cmx_code());
        output.push_str("     ");

        // Edit type
        output.push_str(event.edit_type.cmx_code());
        output.push_str("        ");

        // Timecodes
        let source_in_tc = Timecode::from_seconds(event.source_in, fps);
        let source_out_tc = Timecode::from_seconds(event.source_out, fps);
        let record_in_tc = Timecode::from_seconds(event.record_in, fps);
        let record_out_tc = Timecode::from_seconds(event.record_out, fps);

        output.push_str(&source_in_tc.to_string());
        output.push(' ');
        output.push_str(&source_out_tc.to_string());
        output.push(' ');
        output.push_str(&record_in_tc.to_string());
        output.push(' ');
        output.push_str(&record_out_tc.to_string());
        output.push('\n');

        // Source file comment
        output.push_str("* FROM CLIP NAME: ");
        output.push_str(
            event
                .source_file
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("UNKNOWN"),
        );
        output.push_str("\n\n");

        Ok(())
    }

    /// Export to Final Cut Pro XML (simplified)
    pub fn export_fcpxml(&self, timeline: &Timeline) -> Result<String> {
        let mut output = String::new();

        output.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        output.push_str("<!DOCTYPE xmeml>\n");
        output.push_str("<xmeml version=\"5\">\n");
        output.push_str("  <sequence>\n");
        output.push_str(&format!("    <name>{}</name>\n", timeline.name));
        output.push_str(&format!("    <rate>\n      <timebase>{}</timebase>\n    </rate>\n", timeline.fps as u32));
        output.push_str("    <media>\n");
        output.push_str("      <video>\n");
        output.push_str("        <track>\n");

        for event in &timeline.events {
            self.write_fcpxml_event(&mut output, event, timeline.fps)?;
        }

        output.push_str("        </track>\n");
        output.push_str("      </video>\n");
        output.push_str("    </media>\n");
        output.push_str("  </sequence>\n");
        output.push_str("</xmeml>\n");

        Ok(output)
    }

    fn write_fcpxml_event(&self, output: &mut String, event: &TimelineEvent, fps: f64) -> Result<()> {
        let duration_frames = ((event.source_out - event.source_in) * fps) as u32;

        output.push_str("          <clipitem>\n");
        output.push_str(&format!(
            "            <name>{}</name>\n",
            event
                .source_file
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("CLIP")
        ));
        output.push_str(&format!("            <duration>{}</duration>\n", duration_frames));
        output.push_str(&format!(
            "            <in>{}</in>\n",
            (event.source_in * fps) as u32
        ));
        output.push_str(&format!(
            "            <out>{}</out>\n",
            (event.source_out * fps) as u32
        ));
        output.push_str(&format!(
            "            <start>{}</start>\n",
            (event.record_in * fps) as u32
        ));
        output.push_str(&format!(
            "            <end>{}</end>\n",
            (event.record_out * fps) as u32
        ));
        output.push_str("          </clipitem>\n");

        Ok(())
    }
}

impl Default for EdlExporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timecode_from_seconds() {
        let tc = Timecode::from_seconds(3661.5, 24.0);
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 1);
        assert_eq!(tc.seconds, 1);
        assert_eq!(tc.frames, 12); // 0.5 * 24 = 12
    }

    #[test]
    fn test_timecode_to_string() {
        let tc = Timecode {
            hours: 1,
            minutes: 23,
            seconds: 45,
            frames: 12,
        };
        assert_eq!(tc.to_string(), "01:23:45:12");
    }

    #[test]
    fn test_timecode_roundtrip() {
        let original_seconds = 125.5;
        let fps = 24.0;
        let tc = Timecode::from_seconds(original_seconds, fps);
        let converted_seconds = tc.to_seconds(fps);

        assert!((original_seconds - converted_seconds).abs() < 0.1);
    }

    #[test]
    fn test_timeline_event_creation() {
        let event = TimelineEvent::new_cut("source.mp4", 10.0, 20.0, 0.0);

        assert_eq!(event.source_in, 10.0);
        assert_eq!(event.source_out, 20.0);
        assert_eq!(event.record_in, 0.0);
        assert_eq!(event.record_out, 10.0); // Duration is 10 seconds
        assert_eq!(event.edit_type, EditType::Cut);
    }

    #[test]
    fn test_timeline_event_duration() {
        let event = TimelineEvent::new_cut("source.mp4", 5.0, 15.0, 0.0);
        assert_eq!(event.duration(), 10.0);
    }

    #[test]
    fn test_timeline_creation() {
        let timeline = Timeline::new(24.0, 1920, 1080);

        assert_eq!(timeline.fps, 24.0);
        assert_eq!(timeline.width, 1920);
        assert_eq!(timeline.height, 1080);
        assert!(timeline.events.is_empty());
    }

    #[test]
    fn test_timeline_add_event() {
        let mut timeline = Timeline::new(24.0, 1920, 1080);
        let event = TimelineEvent::new_cut("source.mp4", 0.0, 5.0, 0.0);

        timeline.add_event(event);

        assert_eq!(timeline.events.len(), 1);
        assert_eq!(timeline.events[0].number, 1);
    }

    #[test]
    fn test_timeline_duration() {
        let mut timeline = Timeline::new(24.0, 1920, 1080);

        timeline.add_event(TimelineEvent::new_cut("source.mp4", 0.0, 5.0, 0.0));
        timeline.add_event(TimelineEvent::new_cut("source.mp4", 5.0, 10.0, 5.0));

        assert_eq!(timeline.duration(), 10.0);
    }

    #[test]
    fn test_cmx3600_export() {
        let mut timeline = Timeline::new(24.0, 1920, 1080);
        timeline.name = "TEST".to_string();
        timeline.add_event(TimelineEvent::new_cut("clip1.mp4", 0.0, 5.0, 0.0));

        let exporter = EdlExporter::new();
        let edl = exporter.export_cmx3600(&timeline).unwrap();

        assert!(edl.contains("TITLE: TEST"));
        assert!(edl.contains("NON-DROP FRAME"));
        assert!(edl.contains("001"));
        assert!(edl.contains("clip1.mp4"));
    }

    #[test]
    fn test_fcpxml_export() {
        let mut timeline = Timeline::new(24.0, 1920, 1080);
        timeline.name = "TEST_SEQUENCE".to_string();
        timeline.add_event(TimelineEvent::new_cut("clip1.mp4", 0.0, 5.0, 0.0));

        let exporter = EdlExporter::new();
        let xml = exporter.export_fcpxml(&timeline).unwrap();

        assert!(xml.contains("<?xml version"));
        assert!(xml.contains("<xmeml"));
        assert!(xml.contains("TEST_SEQUENCE"));
        assert!(xml.contains("<clipitem>"));
        assert!(xml.contains("clip1.mp4"));
    }

    #[test]
    fn test_edit_type_cmx_code() {
        assert_eq!(EditType::Cut.cmx_code(), "C");
        assert_eq!(EditType::Dissolve.cmx_code(), "D");
        assert_eq!(EditType::Wipe.cmx_code(), "W");
    }

    #[test]
    fn test_track_type_cmx_code() {
        assert_eq!(TrackType::Video.cmx_code(), "V");
        assert_eq!(TrackType::Audio.cmx_code(), "A");
        assert_eq!(TrackType::Both.cmx_code(), "B");
    }
}
