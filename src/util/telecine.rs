//! Telecine and Inverse Telecine (3:2 Pulldown)
//!
//! Convert between film frame rates (24fps) and video frame rates (29.97fps NTSC)
//! using 3:2 pulldown patterns, and reverse the process for film restoration.
//!
//! ## Telecine Process
//!
//! Film (24fps) → Video (29.97fps) using 3:2 pulldown:
//! - Frame A: 3 fields (top, bottom, top)
//! - Frame B: 2 fields (bottom, top)
//! - Frame C: 3 fields (top, bottom, top)
//! - Frame D: 2 fields (bottom, top)
//! - Repeats: AABBBCCCDDD pattern
//!
//! ## Inverse Telecine
//!
//! Video (29.97fps) → Film (24fps):
//! - Detect pulldown cadence
//! - Remove duplicate fields
//! - Reconstruct original 24fps frames
//!
//! ## Usage
//!
//! ```rust
//! use zvd_lib::util::telecine::{Telecine, PulldownPattern};
//!
//! // 24fps → 29.97fps
//! let telecine = Telecine::new(PulldownPattern::ThreeTwo);
//! let video_frames = telecine.apply_pulldown(&film_frames)?;
//!
//! // 29.97fps → 24fps
//! let inverse = telecine.inverse_telecine(&video_frames)?;
//! ```

use crate::error::{Error, Result};

/// Pulldown pattern type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PulldownPattern {
    /// 3:2 pulldown (24fps → 29.97fps NTSC)
    ThreeTwo,
    /// 2:2 pulldown (25fps → 50fps PAL)
    TwoTwo,
    /// 2:3:3:2 pulldown (24fps → 60fps)
    TwoThreeThreeTwo,
}

impl PulldownPattern {
    /// Get field repeat pattern
    pub fn pattern(&self) -> &[usize] {
        match self {
            PulldownPattern::ThreeTwo => &[3, 2, 3, 2], // A:3, B:2, C:3, D:2
            PulldownPattern::TwoTwo => &[2, 2], // A:2, B:2
            PulldownPattern::TwoThreeThreeTwo => &[2, 3, 3, 2], // A:2, B:3, C:3, D:2
        }
    }

    /// Get total fields in cycle
    pub fn cycle_fields(&self) -> usize {
        self.pattern().iter().sum()
    }

    /// Get source frames in cycle
    pub fn cycle_frames(&self) -> usize {
        self.pattern().len()
    }
}

/// Field order
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldOrder {
    /// Top field first
    TopFieldFirst,
    /// Bottom field first
    BottomFieldFirst,
}

/// Interlaced field
#[derive(Debug, Clone)]
pub struct Field {
    /// Field data (half vertical resolution)
    pub data: Vec<u8>,
    /// Field order (top or bottom)
    pub order: FieldOrder,
}

/// Frame with field information
#[derive(Debug, Clone)]
pub struct InterlacedFrame {
    pub width: usize,
    pub height: usize,
    pub top_field: Vec<u8>,
    pub bottom_field: Vec<u8>,
}

impl InterlacedFrame {
    /// Create from progressive frame
    pub fn from_progressive(data: &[u8], width: usize, height: usize) -> Result<Self> {
        if data.len() != width * height * 3 {
            return Err(Error::InvalidInput("Invalid frame size".to_string()));
        }

        let field_height = height / 2;
        let mut top_field = Vec::with_capacity(width * field_height * 3);
        let mut bottom_field = Vec::with_capacity(width * field_height * 3);

        // Split into fields
        for y in 0..height {
            let row_start = y * width * 3;
            let row_end = row_start + width * 3;

            if y % 2 == 0 {
                top_field.extend_from_slice(&data[row_start..row_end]);
            } else {
                bottom_field.extend_from_slice(&data[row_start..row_end]);
            }
        }

        Ok(InterlacedFrame {
            width,
            height,
            top_field,
            bottom_field,
        })
    }

    /// Convert to progressive frame
    pub fn to_progressive(&self) -> Vec<u8> {
        let mut progressive = vec![0u8; self.width * self.height * 3];

        let field_height = self.height / 2;

        for y in 0..self.height {
            let row_start = y * self.width * 3;
            let row_end = row_start + self.width * 3;

            let field_row = (y / 2) * self.width * 3;

            if y % 2 == 0 {
                // Top field
                progressive[row_start..row_end].copy_from_slice(
                    &self.top_field[field_row..field_row + self.width * 3],
                );
            } else {
                // Bottom field
                progressive[row_start..row_end].copy_from_slice(
                    &self.bottom_field[field_row..field_row + self.width * 3],
                );
            }
        }

        progressive
    }
}

/// Telecine converter
pub struct Telecine {
    pattern: PulldownPattern,
    field_order: FieldOrder,
}

impl Telecine {
    /// Create new telecine converter
    pub fn new(pattern: PulldownPattern) -> Self {
        Telecine {
            pattern,
            field_order: FieldOrder::TopFieldFirst,
        }
    }

    /// Set field order
    pub fn with_field_order(mut self, field_order: FieldOrder) -> Self {
        self.field_order = field_order;
        self
    }

    /// Apply pulldown to progressive frames (24fps → 29.97fps)
    ///
    /// This is a simplified stub - real implementation would generate interlaced fields
    pub fn apply_pulldown(&self, frames: &[Vec<u8>], width: usize, height: usize) -> Result<Vec<Vec<u8>>> {
        let pattern = self.pattern.pattern();
        let mut output = Vec::new();

        for (i, frame) in frames.iter().enumerate() {
            let pattern_idx = i % pattern.len();
            let repeat_count = pattern[pattern_idx];

            // Repeat frame according to pattern
            for _ in 0..repeat_count {
                output.push(frame.clone());
            }
        }

        Ok(output)
    }

    /// Remove pulldown (inverse telecine) - 29.97fps → 24fps
    pub fn inverse_telecine(&self, frames: &[Vec<u8>], width: usize, height: usize) -> Result<Vec<Vec<u8>>> {
        let cadence = self.detect_cadence(frames, width, height)?;

        let pattern = self.pattern.pattern();
        let cycle_len = pattern.len();

        let mut output = Vec::new();
        let mut i = 0;

        while i < frames.len() {
            let pattern_idx = (i + cadence) % cycle_len;

            // Take first frame of each repeat sequence
            if i < frames.len() {
                output.push(frames[i].clone());
            }

            // Skip the repeated frames
            i += pattern[pattern_idx];
        }

        Ok(output)
    }

    /// Detect pulldown cadence (which frame starts the pattern)
    pub fn detect_cadence(&self, frames: &[Vec<u8>], width: usize, height: usize) -> Result<usize> {
        if frames.len() < 10 {
            return Ok(0); // Default to 0 if not enough frames
        }

        // Detect cadence by finding duplicate frames
        // Simplified implementation - would use field matching in production
        let mut best_cadence = 0;
        let mut best_score = 0;

        let pattern = self.pattern.pattern();

        for cadence in 0..pattern.len() {
            let score = self.score_cadence(frames, cadence, width, height);
            if score > best_score {
                best_score = score;
                best_cadence = cadence;
            }
        }

        Ok(best_cadence)
    }

    /// Score a potential cadence
    fn score_cadence(&self, frames: &[Vec<u8>], cadence: usize, width: usize, height: usize) -> usize {
        let pattern = self.pattern.pattern();
        let mut score = 0;

        for i in 0..frames.len().saturating_sub(1) {
            let pattern_idx = (i + cadence) % pattern.len();

            // Check if this frame should be similar to next
            if pattern[pattern_idx] > 1 && i + 1 < frames.len() {
                let similarity = self.frame_similarity(&frames[i], &frames[i + 1]);
                if similarity > 0.95 {
                    score += 1;
                }
            }
        }

        score
    }

    /// Calculate frame similarity (0.0 to 1.0)
    fn frame_similarity(&self, frame1: &[u8], frame2: &[u8]) -> f64 {
        if frame1.len() != frame2.len() {
            return 0.0;
        }

        let mut diff_sum = 0u64;
        for (a, b) in frame1.iter().zip(frame2.iter()) {
            diff_sum += (*a as i32 - *b as i32).unsigned_abs() as u64;
        }

        let max_diff = frame1.len() as u64 * 255;
        1.0 - (diff_sum as f64 / max_diff as f64)
    }
}

/// Cadence detector (for automatic inverse telecine)
pub struct CadenceDetector {
    /// History of frame differences
    frame_diffs: Vec<f64>,
    /// Detected pattern
    detected_pattern: Option<PulldownPattern>,
    /// Detected cadence offset
    cadence_offset: usize,
}

impl CadenceDetector {
    /// Create new cadence detector
    pub fn new() -> Self {
        CadenceDetector {
            frame_diffs: Vec::new(),
            detected_pattern: None,
            cadence_offset: 0,
        }
    }

    /// Add frame difference measurement
    pub fn add_frame_diff(&mut self, diff: f64) {
        self.frame_diffs.push(diff);

        // Keep last 60 measurements
        if self.frame_diffs.len() > 60 {
            self.frame_diffs.remove(0);
        }

        // Try to detect pattern
        if self.frame_diffs.len() >= 10 {
            self.detect_pattern();
        }
    }

    /// Detect pulldown pattern from differences
    fn detect_pattern(&mut self) {
        // Look for 3:2 pattern (low-low-high-low-high repeating)
        if self.matches_pattern(&[3, 2, 3, 2]) {
            self.detected_pattern = Some(PulldownPattern::ThreeTwo);
        } else if self.matches_pattern(&[2, 2]) {
            self.detected_pattern = Some(PulldownPattern::TwoTwo);
        } else if self.matches_pattern(&[2, 3, 3, 2]) {
            self.detected_pattern = Some(PulldownPattern::TwoThreeThreeTwo);
        }
    }

    /// Check if differences match pattern
    fn matches_pattern(&self, pattern: &[usize]) -> bool {
        // Simplified check - would analyze diff peaks in production
        true // Stub
    }

    /// Get detected pattern
    pub fn get_pattern(&self) -> Option<PulldownPattern> {
        self.detected_pattern
    }

    /// Get cadence offset
    pub fn get_offset(&self) -> usize {
        self.cadence_offset
    }
}

impl Default for CadenceDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(width: usize, height: usize, value: u8) -> Vec<u8> {
        vec![value; width * height * 3]
    }

    #[test]
    fn test_pulldown_pattern_three_two() {
        let pattern = PulldownPattern::ThreeTwo;
        assert_eq!(pattern.pattern(), &[3, 2, 3, 2]);
        assert_eq!(pattern.cycle_fields(), 10); // 3+2+3+2
        assert_eq!(pattern.cycle_frames(), 4);
    }

    #[test]
    fn test_pulldown_pattern_two_two() {
        let pattern = PulldownPattern::TwoTwo;
        assert_eq!(pattern.pattern(), &[2, 2]);
        assert_eq!(pattern.cycle_fields(), 4);
        assert_eq!(pattern.cycle_frames(), 2);
    }

    #[test]
    fn test_interlaced_frame_creation() {
        let data = create_test_frame(64, 64, 128);
        let interlaced = InterlacedFrame::from_progressive(&data, 64, 64).unwrap();

        assert_eq!(interlaced.width, 64);
        assert_eq!(interlaced.height, 64);
        assert_eq!(interlaced.top_field.len(), 64 * 32 * 3); // Half height
        assert_eq!(interlaced.bottom_field.len(), 64 * 32 * 3);
    }

    #[test]
    fn test_interlaced_to_progressive() {
        let data = create_test_frame(64, 64, 128);
        let interlaced = InterlacedFrame::from_progressive(&data, 64, 64).unwrap();
        let progressive = interlaced.to_progressive();

        assert_eq!(progressive.len(), data.len());
    }

    #[test]
    fn test_telecine_creation() {
        let telecine = Telecine::new(PulldownPattern::ThreeTwo);
        assert_eq!(telecine.pattern, PulldownPattern::ThreeTwo);
        assert_eq!(telecine.field_order, FieldOrder::TopFieldFirst);
    }

    #[test]
    fn test_telecine_with_field_order() {
        let telecine = Telecine::new(PulldownPattern::ThreeTwo)
            .with_field_order(FieldOrder::BottomFieldFirst);
        assert_eq!(telecine.field_order, FieldOrder::BottomFieldFirst);
    }

    #[test]
    fn test_apply_pulldown() {
        let telecine = Telecine::new(PulldownPattern::ThreeTwo);
        let frames = vec![
            create_test_frame(64, 64, 0),
            create_test_frame(64, 64, 1),
            create_test_frame(64, 64, 2),
            create_test_frame(64, 64, 3),
        ];

        let pulled = telecine.apply_pulldown(&frames, 64, 64).unwrap();

        // Pattern is [3, 2, 3, 2], so 4 frames → 10 frames
        assert_eq!(pulled.len(), 10);
    }

    #[test]
    fn test_inverse_telecine() {
        let telecine = Telecine::new(PulldownPattern::ThreeTwo);

        // Create pulled-down frames (4 → 10)
        let original_frames = vec![
            create_test_frame(64, 64, 0),
            create_test_frame(64, 64, 1),
            create_test_frame(64, 64, 2),
            create_test_frame(64, 64, 3),
        ];

        let pulled = telecine.apply_pulldown(&original_frames, 64, 64).unwrap();
        let restored = telecine.inverse_telecine(&pulled, 64, 64).unwrap();

        // Should restore to 4 frames
        assert_eq!(restored.len(), 4);
    }

    #[test]
    fn test_frame_similarity_identical() {
        let telecine = Telecine::new(PulldownPattern::ThreeTwo);
        let frame = create_test_frame(64, 64, 128);

        let similarity = telecine.frame_similarity(&frame, &frame);
        assert_eq!(similarity, 1.0);
    }

    #[test]
    fn test_frame_similarity_different() {
        let telecine = Telecine::new(PulldownPattern::ThreeTwo);
        let frame1 = create_test_frame(64, 64, 0);
        let frame2 = create_test_frame(64, 64, 255);

        let similarity = telecine.frame_similarity(&frame1, &frame2);
        assert_eq!(similarity, 0.0); // Completely different
    }

    #[test]
    fn test_cadence_detector() {
        let mut detector = CadenceDetector::new();

        // Add some measurements
        detector.add_frame_diff(0.1);
        detector.add_frame_diff(0.1);
        detector.add_frame_diff(0.9);

        // Pattern may or may not be detected yet
        assert!(detector.frame_diffs.len() == 3);
    }

    #[test]
    fn test_field_order() {
        assert_eq!(FieldOrder::TopFieldFirst, FieldOrder::TopFieldFirst);
        assert_ne!(FieldOrder::TopFieldFirst, FieldOrder::BottomFieldFirst);
    }
}
