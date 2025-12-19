//! Silence and Black Frame Detection
//!
//! Detect silent audio sections, black frames, freeze frames, and other
//! anomalies for automated editing, quality control, and content analysis.
//!
//! ## Detection Types
//!
//! - **Silence**: Audio below threshold for duration
//! - **Black Frames**: Video luminance below threshold
//! - **Freeze Frames**: Identical consecutive frames
//! - **Fade to Black**: Gradual luminance decrease
//!
//! ## Common Use Cases
//!
//! - Automated scene detection
//! - Commercial break identification
//! - Quality control (missing audio/video)
//! - Content trimming
//! - Chapter point detection
//!
//! ## Usage
//!
//! ```rust
//! use zvd_lib::util::detection::{SilenceDetector, BlackFrameDetector};
//!
//! // Detect silence
//! let detector = SilenceDetector::new(-40.0, 2.0); // -40dB, 2 seconds
//! let silent_regions = detector.detect(&audio_samples, sample_rate)?;
//!
//! // Detect black frames
//! let black_detector = BlackFrameDetector::new(0.1, 0.95); // 10% threshold, 95% coverage
//! if black_detector.is_black(&frame_data, width, height) {
//!     println!("Black frame detected");
//! }
//! ```

use crate::error::{Error, Result};

/// Silence detection parameters
#[derive(Debug, Clone)]
pub struct SilenceDetector {
    /// Threshold in dB (e.g., -40.0)
    threshold_db: f64,
    /// Minimum duration in seconds
    min_duration: f64,
}

impl SilenceDetector {
    /// Create new silence detector
    ///
    /// # Arguments
    /// * `threshold_db` - Silence threshold in dB (typically -40 to -60)
    /// * `min_duration` - Minimum silence duration in seconds
    pub fn new(threshold_db: f64, min_duration: f64) -> Self {
        SilenceDetector {
            threshold_db,
            min_duration: min_duration.max(0.0),
        }
    }

    /// Detect silent regions in audio
    pub fn detect(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<SilentRegion>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        let threshold_linear = self.db_to_linear(self.threshold_db);
        let min_samples = (self.min_duration * sample_rate as f64) as usize;

        let mut regions = Vec::new();
        let mut in_silence = false;
        let mut silence_start = 0;

        for (i, &sample) in samples.iter().enumerate() {
            let is_silent = sample.abs() < threshold_linear;

            if is_silent && !in_silence {
                // Start of silence
                silence_start = i;
                in_silence = true;
            } else if !is_silent && in_silence {
                // End of silence
                let duration_samples = i - silence_start;

                if duration_samples >= min_samples {
                    regions.push(SilentRegion {
                        start_sample: silence_start,
                        end_sample: i,
                        duration_seconds: duration_samples as f64 / sample_rate as f64,
                    });
                }

                in_silence = false;
            }
        }

        // Check if still in silence at end
        if in_silence {
            let duration_samples = samples.len() - silence_start;
            if duration_samples >= min_samples {
                regions.push(SilentRegion {
                    start_sample: silence_start,
                    end_sample: samples.len(),
                    duration_seconds: duration_samples as f64 / sample_rate as f64,
                });
            }
        }

        Ok(regions)
    }

    /// Convert dB to linear amplitude
    fn db_to_linear(&self, db: f64) -> f32 {
        10.0_f64.powf(db / 20.0) as f32
    }

    /// Detect silence in RMS windows
    pub fn detect_rms(
        &self,
        samples: &[f32],
        sample_rate: u32,
        window_size: usize,
    ) -> Result<Vec<SilentRegion>> {
        let threshold_linear = self.db_to_linear(self.threshold_db);
        let min_windows = (self.min_duration * sample_rate as f64 / window_size as f64) as usize;

        let mut regions = Vec::new();
        let mut in_silence = false;
        let mut silence_start_sample = 0;

        for (window_idx, window) in samples.chunks(window_size).enumerate() {
            // Calculate RMS
            let rms = self.calculate_rms(window);
            let is_silent = rms < threshold_linear;

            if is_silent && !in_silence {
                silence_start_sample = window_idx * window_size;
                in_silence = true;
            } else if !is_silent && in_silence {
                let end_sample = window_idx * window_size;
                let duration_samples = end_sample - silence_start_sample;

                if duration_samples >= min_windows * window_size {
                    regions.push(SilentRegion {
                        start_sample: silence_start_sample,
                        end_sample,
                        duration_seconds: duration_samples as f64 / sample_rate as f64,
                    });
                }

                in_silence = false;
            }
        }

        Ok(regions)
    }

    /// Calculate RMS level
    fn calculate_rms(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = samples.iter().map(|&s| s * s).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }
}

/// Silent region
#[derive(Debug, Clone)]
pub struct SilentRegion {
    pub start_sample: usize,
    pub end_sample: usize,
    pub duration_seconds: f64,
}

/// Black frame detector
#[derive(Debug, Clone)]
pub struct BlackFrameDetector {
    /// Luminance threshold (0.0 to 1.0)
    luma_threshold: f64,
    /// Percentage of pixels that must be below threshold
    coverage_threshold: f64,
}

impl BlackFrameDetector {
    /// Create new black frame detector
    ///
    /// # Arguments
    /// * `luma_threshold` - Luminance threshold (0.0 to 1.0, typically 0.1)
    /// * `coverage_threshold` - Pixel coverage (0.0 to 1.0, typically 0.95)
    pub fn new(luma_threshold: f64, coverage_threshold: f64) -> Self {
        BlackFrameDetector {
            luma_threshold: luma_threshold.clamp(0.0, 1.0),
            coverage_threshold: coverage_threshold.clamp(0.0, 1.0),
        }
    }

    /// Check if frame is black (RGB format)
    pub fn is_black(&self, frame_rgb: &[u8], width: usize, height: usize) -> bool {
        if frame_rgb.len() != width * height * 3 {
            return false;
        }

        let threshold_u8 = (self.luma_threshold * 255.0) as u8;
        let mut black_pixels = 0;
        let total_pixels = width * height;

        for rgb in frame_rgb.chunks_exact(3) {
            // Calculate luma (ITU-R BT.709)
            let luma =
                (0.2126 * rgb[0] as f32 + 0.7152 * rgb[1] as f32 + 0.0722 * rgb[2] as f32) as u8;

            if luma < threshold_u8 {
                black_pixels += 1;
            }
        }

        let coverage = black_pixels as f64 / total_pixels as f64;
        coverage >= self.coverage_threshold
    }

    /// Get blackness percentage (0.0 to 1.0)
    pub fn blackness_level(&self, frame_rgb: &[u8]) -> f64 {
        if frame_rgb.len() % 3 != 0 {
            return 0.0;
        }

        let threshold_u8 = (self.luma_threshold * 255.0) as u8;
        let mut black_pixels = 0;
        let total_pixels = frame_rgb.len() / 3;

        for rgb in frame_rgb.chunks_exact(3) {
            let luma =
                (0.2126 * rgb[0] as f32 + 0.7152 * rgb[1] as f32 + 0.0722 * rgb[2] as f32) as u8;

            if luma < threshold_u8 {
                black_pixels += 1;
            }
        }

        black_pixels as f64 / total_pixels as f64
    }
}

/// Freeze frame detector
#[derive(Debug, Clone)]
pub struct FreezeFrameDetector {
    /// Similarity threshold (0.0 to 1.0)
    similarity_threshold: f64,
    /// Minimum freeze duration in frames
    min_duration_frames: usize,
}

impl FreezeFrameDetector {
    /// Create new freeze frame detector
    pub fn new(similarity_threshold: f64, min_duration_frames: usize) -> Self {
        FreezeFrameDetector {
            similarity_threshold: similarity_threshold.clamp(0.0, 1.0),
            min_duration_frames,
        }
    }

    /// Detect freeze frames in sequence
    pub fn detect(&self, frames: &[Vec<u8>]) -> Vec<FreezeRegion> {
        if frames.len() < 2 {
            return Vec::new();
        }

        let mut regions = Vec::new();
        let mut in_freeze = false;
        let mut freeze_start = 0;

        for i in 1..frames.len() {
            let similarity = self.frame_similarity(&frames[i - 1], &frames[i]);
            let is_frozen = similarity >= self.similarity_threshold;

            if is_frozen && !in_freeze {
                freeze_start = i - 1;
                in_freeze = true;
            } else if !is_frozen && in_freeze {
                let duration = i - freeze_start;

                if duration >= self.min_duration_frames {
                    regions.push(FreezeRegion {
                        start_frame: freeze_start,
                        end_frame: i,
                        duration_frames: duration,
                    });
                }

                in_freeze = false;
            }
        }

        regions
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

/// Freeze frame region
#[derive(Debug, Clone)]
pub struct FreezeRegion {
    pub start_frame: usize,
    pub end_frame: usize,
    pub duration_frames: usize,
}

/// Fade detector
pub struct FadeDetector {
    /// Minimum fade duration in frames
    min_duration_frames: usize,
}

impl FadeDetector {
    /// Create new fade detector
    pub fn new(min_duration_frames: usize) -> Self {
        FadeDetector {
            min_duration_frames,
        }
    }

    /// Detect fade to black
    pub fn detect_fade_to_black(&self, frames: &[Vec<u8>]) -> Vec<FadeRegion> {
        let mut regions = Vec::new();

        if frames.len() < self.min_duration_frames {
            return regions;
        }

        // Calculate average luminance for each frame
        let lumas: Vec<f64> = frames.iter().map(|f| self.average_luma(f)).collect();

        // Look for decreasing luminance sequences
        for i in 0..lumas.len().saturating_sub(self.min_duration_frames) {
            let start_luma = lumas[i];
            let end_idx = (i + self.min_duration_frames).min(lumas.len());
            let end_luma = lumas[end_idx - 1];

            // Check if luminance decreased significantly
            if start_luma > 0.3 && end_luma < 0.1 {
                regions.push(FadeRegion {
                    start_frame: i,
                    end_frame: end_idx,
                    start_value: start_luma,
                    end_value: end_luma,
                });
            }
        }

        regions
    }

    /// Calculate average luminance
    fn average_luma(&self, frame: &[u8]) -> f64 {
        if frame.len() % 3 != 0 || frame.is_empty() {
            return 0.0;
        }

        let mut sum = 0.0;
        let pixels = frame.len() / 3;

        for rgb in frame.chunks_exact(3) {
            let luma = 0.2126 * rgb[0] as f64 + 0.7152 * rgb[1] as f64 + 0.0722 * rgb[2] as f64;
            sum += luma / 255.0; // Normalize to 0-1
        }

        sum / pixels as f64
    }
}

/// Fade region
#[derive(Debug, Clone)]
pub struct FadeRegion {
    pub start_frame: usize,
    pub end_frame: usize,
    pub start_value: f64,
    pub end_value: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silence_detector_creation() {
        let detector = SilenceDetector::new(-40.0, 2.0);
        assert_eq!(detector.threshold_db, -40.0);
        assert_eq!(detector.min_duration, 2.0);
    }

    #[test]
    fn test_silence_detection() {
        let mut samples = vec![0.001f32; 96000]; // 2 seconds of near-silence @ 48kHz
        samples.extend(vec![0.5f32; 48000]); // 1 second of audio

        let detector = SilenceDetector::new(-40.0, 1.0);
        let regions = detector.detect(&samples, 48000).unwrap();

        assert!(!regions.is_empty());
        assert!(regions[0].duration_seconds >= 1.0);
    }

    #[test]
    fn test_db_to_linear() {
        let detector = SilenceDetector::new(-40.0, 1.0);
        let linear = detector.db_to_linear(-6.0);
        assert!((linear - 0.5).abs() < 0.01); // -6dB ≈ 0.5
    }

    #[test]
    fn test_black_frame_detector() {
        let detector = BlackFrameDetector::new(0.1, 0.95);

        // Black frame
        let black_frame = vec![0u8; 64 * 64 * 3];
        assert!(detector.is_black(&black_frame, 64, 64));

        // White frame
        let white_frame = vec![255u8; 64 * 64 * 3];
        assert!(!detector.is_black(&white_frame, 64, 64));
    }

    #[test]
    fn test_blackness_level() {
        let detector = BlackFrameDetector::new(0.1, 0.95);

        let black_frame = vec![0u8; 100 * 3];
        assert_eq!(detector.blackness_level(&black_frame), 1.0);

        let white_frame = vec![255u8; 100 * 3];
        assert_eq!(detector.blackness_level(&white_frame), 0.0);
    }

    #[test]
    fn test_freeze_frame_detector() {
        let detector = FreezeFrameDetector::new(0.99, 3);

        let frame1 = vec![128u8; 64 * 64 * 3];
        let frame2 = frame1.clone();
        let frame3 = frame1.clone();
        let frame4 = vec![64u8; 64 * 64 * 3]; // Different

        let frames = vec![frame1, frame2, frame3, frame4];
        let regions = detector.detect(&frames);

        assert!(!regions.is_empty());
        assert!(regions[0].duration_frames >= 3);
    }

    #[test]
    fn test_frame_similarity() {
        let detector = FreezeFrameDetector::new(0.99, 2);

        let frame1 = vec![128u8; 100];
        let frame2 = frame1.clone();

        let similarity = detector.frame_similarity(&frame1, &frame2);
        assert_eq!(similarity, 1.0);
    }

    #[test]
    fn test_fade_detector() {
        let detector = FadeDetector::new(5);

        // Create fade: bright → dark (must go from > 0.3 to < 0.1)
        // Start at 200 (luma ~0.78), end at 10 (luma ~0.04)
        let mut frames = Vec::new();
        for i in 0..10 {
            let value = (200u8).saturating_sub(i * 22); // 200, 178, 156, ..., 2
            frames.push(vec![value; 64 * 64 * 3]);
        }

        let fades = detector.detect_fade_to_black(&frames);
        assert!(!fades.is_empty());
    }

    #[test]
    fn test_calculate_rms() {
        let detector = SilenceDetector::new(-40.0, 1.0);

        let samples = vec![0.5f32, -0.5, 0.5, -0.5];
        let rms = detector.calculate_rms(&samples);

        assert!((rms - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_silent_region() {
        let region = SilentRegion {
            start_sample: 0,
            end_sample: 48000,
            duration_seconds: 1.0,
        };

        assert_eq!(region.duration_seconds, 1.0);
    }
}
