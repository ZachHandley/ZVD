//! Container metadata support (chapters, cue points, tags)
//!
//! This module provides structures for rich metadata that can be embedded
//! in various container formats:
//!
//! - **Chapters**: Navigation points with titles and timestamps
//! - **Cue Points**: Markers for seeking or event synchronization
//! - **Tags**: Key-value metadata (title, artist, copyright, etc.)
//!
//! ## Format Support
//!
//! - **MP4/MOV**: Chapter track (text track with chapter markers)
//! - **WebM/MKV**: Chapters element with editions and atoms
//! - **Others**: Best-effort mapping where supported

use crate::error::{Error, Result};
use std::collections::HashMap;
use std::time::Duration;

/// Chapter marker with title and timestamp
#[derive(Debug, Clone)]
pub struct Chapter {
    /// Chapter title/name
    pub title: String,
    /// Start time of the chapter
    pub start_time: Duration,
    /// End time of the chapter (optional, defaults to next chapter or media end)
    pub end_time: Option<Duration>,
    /// Language code (ISO 639-2, e.g., "eng", "fra")
    pub language: Option<String>,
}

impl Chapter {
    /// Create a new chapter
    pub fn new(title: &str, start_time: Duration) -> Self {
        Chapter {
            title: title.to_string(),
            start_time,
            end_time: None,
            language: Some("eng".to_string()),
        }
    }

    /// Create a new chapter with end time
    pub fn with_end_time(title: &str, start_time: Duration, end_time: Duration) -> Self {
        Chapter {
            title: title.to_string(),
            start_time,
            end_time: Some(end_time),
            language: Some("eng".to_string()),
        }
    }

    /// Set the language
    pub fn with_language(mut self, language: &str) -> Self {
        self.language = Some(language.to_string());
        self
    }

    /// Get duration of the chapter
    pub fn duration(&self) -> Option<Duration> {
        self.end_time.map(|end| end - self.start_time)
    }
}

/// Cue point types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CuePointType {
    /// Navigation cue (for seeking)
    Navigation,
    /// Event marker (for synchronization)
    Event,
    /// Ad break marker
    AdBreak,
    /// Custom marker
    Custom,
}

/// Cue point for seeking and event synchronization
#[derive(Debug, Clone)]
pub struct CuePoint {
    /// Cue point type
    pub cue_type: CuePointType,
    /// Timestamp
    pub timestamp: Duration,
    /// Optional name/label
    pub name: Option<String>,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

impl CuePoint {
    /// Create a new navigation cue point
    pub fn navigation(timestamp: Duration) -> Self {
        CuePoint {
            cue_type: CuePointType::Navigation,
            timestamp,
            name: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a new event cue point
    pub fn event(timestamp: Duration, name: &str) -> Self {
        CuePoint {
            cue_type: CuePointType::Event,
            timestamp,
            name: Some(name.to_string()),
            metadata: HashMap::new(),
        }
    }

    /// Create an ad break cue point
    pub fn ad_break(timestamp: Duration) -> Self {
        CuePoint {
            cue_type: CuePointType::AdBreak,
            timestamp,
            name: Some("Ad Break".to_string()),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the cue point
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// Container tags (metadata key-value pairs)
#[derive(Debug, Clone, Default)]
pub struct ContainerTags {
    tags: HashMap<String, String>,
}

impl ContainerTags {
    /// Create a new empty tag collection
    pub fn new() -> Self {
        ContainerTags {
            tags: HashMap::new(),
        }
    }

    /// Set a tag
    pub fn set(&mut self, key: &str, value: &str) {
        self.tags.insert(key.to_string(), value.to_string());
    }

    /// Get a tag
    pub fn get(&self, key: &str) -> Option<&String> {
        self.tags.get(key)
    }

    /// Remove a tag
    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.tags.remove(key)
    }

    /// Get all tags
    pub fn all(&self) -> &HashMap<String, String> {
        &self.tags
    }

    /// Common tag setters
    pub fn set_title(&mut self, title: &str) {
        self.set("title", title);
    }

    pub fn set_artist(&mut self, artist: &str) {
        self.set("artist", artist);
    }

    pub fn set_album(&mut self, album: &str) {
        self.set("album", album);
    }

    pub fn set_year(&mut self, year: u32) {
        self.set("year", &year.to_string());
    }

    pub fn set_comment(&mut self, comment: &str) {
        self.set("comment", comment);
    }

    pub fn set_copyright(&mut self, copyright: &str) {
        self.set("copyright", copyright);
    }

    pub fn set_encoder(&mut self, encoder: &str) {
        self.set("encoder", encoder);
    }

    pub fn set_language(&mut self, language: &str) {
        self.set("language", language);
    }
}

impl From<HashMap<String, String>> for ContainerTags {
    fn from(tags: HashMap<String, String>) -> Self {
        ContainerTags { tags }
    }
}

/// Container metadata collection
#[derive(Debug, Clone, Default)]
pub struct ContainerMetadata {
    /// Chapters
    pub chapters: Vec<Chapter>,
    /// Cue points
    pub cue_points: Vec<CuePoint>,
    /// Tags
    pub tags: ContainerTags,
}

impl ContainerMetadata {
    /// Create a new empty metadata collection
    pub fn new() -> Self {
        ContainerMetadata {
            chapters: Vec::new(),
            cue_points: Vec::new(),
            tags: ContainerTags::new(),
        }
    }

    /// Add a chapter
    pub fn add_chapter(&mut self, chapter: Chapter) {
        self.chapters.push(chapter);
    }

    /// Add multiple chapters
    pub fn add_chapters(&mut self, chapters: Vec<Chapter>) {
        self.chapters.extend(chapters);
    }

    /// Add a cue point
    pub fn add_cue_point(&mut self, cue_point: CuePoint) {
        self.cue_points.push(cue_point);
    }

    /// Add multiple cue points
    pub fn add_cue_points(&mut self, cue_points: Vec<CuePoint>) {
        self.cue_points.extend(cue_points);
    }

    /// Sort chapters and cue points by timestamp
    pub fn sort(&mut self) {
        self.chapters.sort_by_key(|c| c.start_time);
        self.cue_points.sort_by_key(|c| c.timestamp);
    }

    /// Validate chapters (check for overlaps, missing end times)
    pub fn validate(&mut self) -> Result<()> {
        // Sort first
        self.sort();

        // Fill in missing end times
        for i in 0..self.chapters.len() {
            if self.chapters[i].end_time.is_none() {
                if i + 1 < self.chapters.len() {
                    self.chapters[i].end_time = Some(self.chapters[i + 1].start_time);
                }
                // Last chapter's end time remains None (will be set to media duration)
            }
        }

        // Check for overlaps
        for i in 0..self.chapters.len().saturating_sub(1) {
            let current_end = self.chapters[i]
                .end_time
                .ok_or_else(|| Error::InvalidInput("Chapter end time not set".to_string()))?;
            let next_start = self.chapters[i + 1].start_time;

            if current_end > next_start {
                return Err(Error::InvalidInput(format!(
                    "Chapter overlap detected: chapter {} ends at {:?} but chapter {} starts at {:?}",
                    i,
                    current_end,
                    i + 1,
                    next_start
                )));
            }
        }

        Ok(())
    }
}

/// Builder for ContainerMetadata
pub struct MetadataBuilder {
    metadata: ContainerMetadata,
}

impl MetadataBuilder {
    /// Create a new metadata builder
    pub fn new() -> Self {
        MetadataBuilder {
            metadata: ContainerMetadata::new(),
        }
    }

    /// Add a chapter
    pub fn chapter(mut self, chapter: Chapter) -> Self {
        self.metadata.add_chapter(chapter);
        self
    }

    /// Add a cue point
    pub fn cue_point(mut self, cue_point: CuePoint) -> Self {
        self.metadata.add_cue_point(cue_point);
        self
    }

    /// Set a tag
    pub fn tag(mut self, key: &str, value: &str) -> Self {
        self.metadata.tags.set(key, value);
        self
    }

    /// Set title
    pub fn title(mut self, title: &str) -> Self {
        self.metadata.tags.set_title(title);
        self
    }

    /// Set artist
    pub fn artist(mut self, artist: &str) -> Self {
        self.metadata.tags.set_artist(artist);
        self
    }

    /// Build and validate
    pub fn build(mut self) -> Result<ContainerMetadata> {
        self.metadata.validate()?;
        Ok(self.metadata)
    }

    /// Build without validation
    pub fn build_unchecked(self) -> ContainerMetadata {
        self.metadata
    }
}

impl Default for MetadataBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chapter_creation() {
        let chapter = Chapter::new("Introduction", Duration::from_secs(0));
        assert_eq!(chapter.title, "Introduction");
        assert_eq!(chapter.start_time, Duration::from_secs(0));
        assert_eq!(chapter.language, Some("eng".to_string()));
    }

    #[test]
    fn test_chapter_with_end_time() {
        let chapter = Chapter::with_end_time(
            "Chapter 1",
            Duration::from_secs(0),
            Duration::from_secs(300),
        );
        assert_eq!(chapter.duration(), Some(Duration::from_secs(300)));
    }

    #[test]
    fn test_cue_point_creation() {
        let cue = CuePoint::navigation(Duration::from_secs(10));
        assert_eq!(cue.cue_type, CuePointType::Navigation);
        assert_eq!(cue.timestamp, Duration::from_secs(10));
    }

    #[test]
    fn test_container_tags() {
        let mut tags = ContainerTags::new();
        tags.set_title("My Video");
        tags.set_artist("John Doe");
        tags.set_year(2024);

        assert_eq!(tags.get("title"), Some(&"My Video".to_string()));
        assert_eq!(tags.get("artist"), Some(&"John Doe".to_string()));
        assert_eq!(tags.get("year"), Some(&"2024".to_string()));
    }

    #[test]
    fn test_metadata_builder() {
        let metadata = MetadataBuilder::new()
            .title("Test Video")
            .chapter(Chapter::new("Intro", Duration::from_secs(0)))
            .chapter(Chapter::new("Main", Duration::from_secs(60)))
            .cue_point(CuePoint::navigation(Duration::from_secs(30)))
            .build()
            .unwrap();

        assert_eq!(metadata.chapters.len(), 2);
        assert_eq!(metadata.cue_points.len(), 1);
        assert_eq!(
            metadata.tags.get("title"),
            Some(&"Test Video".to_string())
        );
    }

    #[test]
    fn test_metadata_validation() {
        let mut metadata = ContainerMetadata::new();
        metadata.add_chapter(Chapter::new("Chapter 1", Duration::from_secs(0)));
        metadata.add_chapter(Chapter::new("Chapter 2", Duration::from_secs(100)));
        metadata.add_chapter(Chapter::new("Chapter 3", Duration::from_secs(200)));

        assert!(metadata.validate().is_ok());

        // Check that end times were filled in
        assert_eq!(
            metadata.chapters[0].end_time,
            Some(Duration::from_secs(100))
        );
        assert_eq!(
            metadata.chapters[1].end_time,
            Some(Duration::from_secs(200))
        );
    }
}
