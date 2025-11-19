//! Thumbnail Generation and Scene Detection
//!
//! This module provides tools for extracting representative frames from video,
//! creating thumbnail strips, and detecting scene boundaries.
//!
//! ## Features
//!
//! - **Smart Thumbnail Selection**: Avoid black frames, motion blur, low contrast
//! - **Scene Detection**: Histogram-based shot boundary detection
//! - **Temporal Thumbnails**: Grid/strip layouts for video preview
//! - **Quality Scoring**: Automatic frame quality assessment
//!
//! ## Usage
//!
//! ```rust
//! use zvd_lib::util::thumbnail::{ThumbnailGenerator, SceneDetector};
//!
//! // Generate single thumbnail
//! let generator = ThumbnailGenerator::new(1920, 1080);
//! let thumbnail = generator.generate_single(&frames, ThumbnailMethod::Smart)?;
//!
//! // Detect scenes
//! let mut detector = SceneDetector::new(0.3);
//! for frame in frames {
//!     if detector.is_scene_change(&frame) {
//!         println!("Scene change at frame {}", frame.index);
//!     }
//! }
//! ```

use crate::error::{Error, Result};
use image::{ImageBuffer, Rgb, RgbImage};
use std::path::Path;

/// Method for selecting thumbnail frame
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThumbnailMethod {
    /// First frame
    First,
    /// Middle frame
    Middle,
    /// Frame at specific timestamp (seconds)
    Timestamp(u64),
    /// Frame at specific index
    Index(usize),
    /// Smart selection (highest quality frame)
    Smart,
    /// Most representative frame (median characteristics)
    Representative,
}

/// Frame quality metrics
#[derive(Debug, Clone)]
pub struct FrameQuality {
    /// Frame index
    pub index: usize,
    /// Overall quality score (0.0-1.0, higher is better)
    pub score: f64,
    /// Brightness (0.0-1.0)
    pub brightness: f64,
    /// Contrast (0.0-1.0)
    pub contrast: f64,
    /// Sharpness/detail (0.0-1.0)
    pub sharpness: f64,
    /// Color diversity (0.0-1.0)
    pub color_diversity: f64,
    /// Motion blur estimate (0.0-1.0, lower is better)
    pub motion_blur: f64,
}

impl FrameQuality {
    /// Calculate quality score from metrics
    pub fn calculate_score(&mut self) {
        // Weighted combination of quality factors
        // Penalize very dark/bright frames, low contrast, motion blur
        let brightness_score = 1.0 - (self.brightness - 0.5).abs() * 2.0; // Prefer mid brightness
        let contrast_score = self.contrast;
        let sharpness_score = self.sharpness;
        let color_score = self.color_diversity;
        let blur_penalty = 1.0 - self.motion_blur;

        self.score = (brightness_score * 0.15
            + contrast_score * 0.25
            + sharpness_score * 0.3
            + color_score * 0.15
            + blur_penalty * 0.15)
            .clamp(0.0, 1.0);
    }
}

/// Frame data for thumbnail generation
#[derive(Clone)]
pub struct FrameData {
    /// Frame index
    pub index: usize,
    /// RGB pixel data (width * height * 3)
    pub data: Vec<u8>,
    /// Width in pixels
    pub width: usize,
    /// Height in pixels
    pub height: usize,
    /// Timestamp in seconds
    pub timestamp: f64,
}

impl FrameData {
    /// Create new frame data
    pub fn new(index: usize, data: Vec<u8>, width: usize, height: usize, timestamp: f64) -> Self {
        FrameData {
            index,
            data,
            width,
            height,
            timestamp,
        }
    }

    /// Analyze frame quality
    pub fn analyze_quality(&self) -> FrameQuality {
        let mut quality = FrameQuality {
            index: self.index,
            score: 0.0,
            brightness: self.calculate_brightness(),
            contrast: self.calculate_contrast(),
            sharpness: self.calculate_sharpness(),
            color_diversity: self.calculate_color_diversity(),
            motion_blur: 0.0, // Requires previous frame
        };

        quality.calculate_score();
        quality
    }

    /// Calculate average brightness (0.0-1.0)
    fn calculate_brightness(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }

        let sum: u64 = self
            .data
            .chunks_exact(3)
            .map(|rgb| {
                // Luminance: Y = 0.299*R + 0.587*G + 0.114*B
                ((rgb[0] as f64 * 0.299 + rgb[1] as f64 * 0.587 + rgb[2] as f64 * 0.114) as u64)
            })
            .sum();

        sum as f64 / (self.data.len() as f64 / 3.0) / 255.0
    }

    /// Calculate contrast (0.0-1.0)
    fn calculate_contrast(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }

        // Calculate luminance values
        let luminances: Vec<f64> = self
            .data
            .chunks_exact(3)
            .map(|rgb| rgb[0] as f64 * 0.299 + rgb[1] as f64 * 0.587 + rgb[2] as f64 * 0.114)
            .collect();

        let mean = luminances.iter().sum::<f64>() / luminances.len() as f64;

        // Calculate standard deviation (normalized)
        let variance: f64 = luminances
            .iter()
            .map(|&l| {
                let diff = l - mean;
                diff * diff
            })
            .sum::<f64>()
            / luminances.len() as f64;

        let std_dev = variance.sqrt();

        // Normalize to 0-1 (max std dev for 0-255 range is ~73.9)
        (std_dev / 73.9).min(1.0)
    }

    /// Calculate sharpness using Laplacian variance (0.0-1.0)
    fn calculate_sharpness(&self) -> f64 {
        if self.width < 3 || self.height < 3 {
            return 0.0;
        }

        let mut laplacian_values = Vec::new();

        // Laplacian kernel: [-1 -1 -1]
        //                   [-1  8 -1]
        //                   [-1 -1 -1]
        for y in 1..(self.height - 1) {
            for x in 1..(self.width - 1) {
                let center_idx = (y * self.width + x) * 3;

                // Use green channel (highest resolution in Bayer pattern)
                let center = self.data[center_idx + 1] as f64;

                // Get 8 neighbors
                let mut neighbors = [0.0; 8];
                let offsets = [
                    (-1, -1),
                    (0, -1),
                    (1, -1),
                    (-1, 0),
                    (1, 0),
                    (-1, 1),
                    (0, 1),
                    (1, 1),
                ];

                for (i, (dx, dy)) in offsets.iter().enumerate() {
                    let nx = (x as i32 + dx) as usize;
                    let ny = (y as i32 + dy) as usize;
                    let idx = (ny * self.width + nx) * 3;
                    neighbors[i] = self.data[idx + 1] as f64;
                }

                let laplacian = center * 8.0 - neighbors.iter().sum::<f64>();
                laplacian_values.push(laplacian);
            }
        }

        if laplacian_values.is_empty() {
            return 0.0;
        }

        // Calculate variance of Laplacian (higher = sharper)
        let mean = laplacian_values.iter().sum::<f64>() / laplacian_values.len() as f64;
        let variance: f64 = laplacian_values
            .iter()
            .map(|&l| {
                let diff = l - mean;
                diff * diff
            })
            .sum::<f64>()
            / laplacian_values.len() as f64;

        // Normalize (typical sharp image has variance 100-1000)
        (variance / 1000.0).min(1.0)
    }

    /// Calculate color diversity (0.0-1.0)
    fn calculate_color_diversity(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }

        // Use simple RGB variance
        let mut r_values = Vec::new();
        let mut g_values = Vec::new();
        let mut b_values = Vec::new();

        for rgb in self.data.chunks_exact(3) {
            r_values.push(rgb[0] as f64);
            g_values.push(rgb[1] as f64);
            b_values.push(rgb[2] as f64);
        }

        let r_var = Self::calculate_variance(&r_values);
        let g_var = Self::calculate_variance(&g_values);
        let b_var = Self::calculate_variance(&b_values);

        let avg_var = (r_var + g_var + b_var) / 3.0;

        // Normalize (max variance is ~73.9 per channel)
        (avg_var / 73.9).min(1.0)
    }

    /// Calculate variance helper
    fn calculate_variance(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|&v| {
                let diff = v - mean;
                diff * diff
            })
            .sum::<f64>()
            / values.len() as f64;

        variance.sqrt()
    }

    /// Calculate motion blur estimate (requires previous frame)
    pub fn calculate_motion_blur(&self, previous: &FrameData) -> f64 {
        if self.data.len() != previous.data.len() {
            return 0.0;
        }

        // Calculate frame difference
        let diff_sum: u64 = self
            .data
            .iter()
            .zip(previous.data.iter())
            .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs() as u64)
            .sum();

        let avg_diff = diff_sum as f64 / self.data.len() as f64;

        // High difference suggests motion blur or scene change
        // Normalize to 0-1 (values typically 0-50 for motion blur)
        (avg_diff / 50.0).min(1.0)
    }

    /// Convert to RgbImage
    pub fn to_image(&self) -> Result<RgbImage> {
        ImageBuffer::from_vec(self.width as u32, self.height as u32, self.data.clone())
            .ok_or_else(|| Error::InvalidInput("Invalid image dimensions".to_string()))
    }
}

/// Thumbnail generator
pub struct ThumbnailGenerator {
    width: usize,
    height: usize,
}

impl ThumbnailGenerator {
    /// Create a new thumbnail generator
    pub fn new(width: usize, height: usize) -> Self {
        ThumbnailGenerator { width, height }
    }

    /// Generate a single thumbnail from frames
    pub fn generate_single(
        &self,
        frames: &[FrameData],
        method: ThumbnailMethod,
    ) -> Result<RgbImage> {
        if frames.is_empty() {
            return Err(Error::InvalidInput("No frames provided".to_string()));
        }

        let selected_frame = match method {
            ThumbnailMethod::First => &frames[0],
            ThumbnailMethod::Middle => &frames[frames.len() / 2],
            ThumbnailMethod::Timestamp(ts) => {
                // Find closest frame to timestamp
                frames
                    .iter()
                    .min_by_key(|f| ((f.timestamp - ts as f64).abs() * 1000.0) as u64)
                    .unwrap()
            }
            ThumbnailMethod::Index(idx) => {
                if idx >= frames.len() {
                    return Err(Error::InvalidInput(format!(
                        "Frame index {} out of range (max: {})",
                        idx,
                        frames.len() - 1
                    )));
                }
                &frames[idx]
            }
            ThumbnailMethod::Smart => self.select_best_frame(frames),
            ThumbnailMethod::Representative => self.select_representative_frame(frames),
        };

        selected_frame.to_image()
    }

    /// Select best quality frame
    fn select_best_frame<'a>(&self, frames: &'a [FrameData]) -> &'a FrameData {
        frames
            .iter()
            .enumerate()
            .map(|(i, frame)| {
                let mut quality = frame.analyze_quality();

                // Factor in motion blur if not first frame
                if i > 0 {
                    quality.motion_blur = frame.calculate_motion_blur(&frames[i - 1]);
                    quality.calculate_score();
                }

                (frame, quality)
            })
            .max_by(|(_, qa), (_, qb)| qa.score.partial_cmp(&qb.score).unwrap())
            .map(|(frame, _)| frame)
            .unwrap_or(&frames[0])
    }

    /// Select most representative frame (median characteristics)
    fn select_representative_frame<'a>(&self, frames: &'a [FrameData]) -> &'a FrameData {
        let qualities: Vec<FrameQuality> =
            frames.iter().map(|f| f.analyze_quality()).collect();

        // Calculate median brightness and contrast
        let mut brightnesses: Vec<f64> = qualities.iter().map(|q| q.brightness).collect();
        let mut contrasts: Vec<f64> = qualities.iter().map(|q| q.contrast).collect();

        brightnesses.sort_by(|a, b| a.partial_cmp(b).unwrap());
        contrasts.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median_brightness = brightnesses[brightnesses.len() / 2];
        let median_contrast = contrasts[contrasts.len() / 2];

        // Find frame closest to median
        qualities
            .iter()
            .enumerate()
            .min_by_key(|(_, q)| {
                let brightness_diff = (q.brightness - median_brightness).abs();
                let contrast_diff = (q.contrast - median_contrast).abs();
                ((brightness_diff + contrast_diff) * 1000.0) as u64
            })
            .map(|(i, _)| &frames[i])
            .unwrap_or(&frames[0])
    }

    /// Generate thumbnail strip (temporal preview)
    pub fn generate_strip(
        &self,
        frames: &[FrameData],
        count: usize,
        spacing: usize,
    ) -> Result<RgbImage> {
        if frames.is_empty() || count == 0 {
            return Err(Error::InvalidInput(
                "Must provide frames and count > 0".to_string(),
            ));
        }

        let thumb_width = self.width / count;
        let thumb_height = self.height;

        let mut strip =
            ImageBuffer::from_pixel(self.width as u32, thumb_height as u32, Rgb([0, 0, 0]));

        let step = (frames.len() - 1) / count.max(1);

        for i in 0..count {
            let frame_idx = (i * step).min(frames.len() - 1);
            let frame = &frames[frame_idx];

            // Resize frame to thumbnail size
            let thumb_img = frame.to_image()?;
            let resized = image::imageops::resize(
                &thumb_img,
                thumb_width as u32,
                thumb_height as u32,
                image::imageops::FilterType::Lanczos3,
            );

            // Copy to strip
            let x_offset = i * (thumb_width + spacing);
            image::imageops::overlay(&mut strip, &resized, x_offset as i64, 0);
        }

        Ok(strip)
    }

    /// Generate thumbnail grid
    pub fn generate_grid(
        &self,
        frames: &[FrameData],
        rows: usize,
        cols: usize,
    ) -> Result<RgbImage> {
        if frames.is_empty() || rows == 0 || cols == 0 {
            return Err(Error::InvalidInput(
                "Must provide frames and rows/cols > 0".to_string(),
            ));
        }

        let count = rows * cols;
        let thumb_width = self.width / cols;
        let thumb_height = self.height / rows;

        let mut grid = ImageBuffer::from_pixel(
            (thumb_width * cols) as u32,
            (thumb_height * rows) as u32,
            Rgb([0, 0, 0]),
        );

        let step = (frames.len() - 1) / count.max(1);

        for i in 0..count {
            let frame_idx = (i * step).min(frames.len() - 1);
            let frame = &frames[frame_idx];

            let thumb_img = frame.to_image()?;
            let resized = image::imageops::resize(
                &thumb_img,
                thumb_width as u32,
                thumb_height as u32,
                image::imageops::FilterType::Lanczos3,
            );

            let row = i / cols;
            let col = i % cols;
            let x_offset = col * thumb_width;
            let y_offset = row * thumb_height;

            image::imageops::overlay(&mut grid, &resized, x_offset as i64, y_offset as i64);
        }

        Ok(grid)
    }

    /// Save thumbnail to file
    pub fn save_thumbnail(&self, image: &RgbImage, path: &Path) -> Result<()> {
        image
            .save(path)
            .map_err(|e| Error::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
        Ok(())
    }
}

/// Scene detector using histogram-based shot boundary detection
pub struct SceneDetector {
    /// Threshold for scene change detection (0.0-1.0)
    threshold: f64,
    /// Previous frame histogram
    prev_histogram: Option<Vec<u32>>,
    /// Histogram bins per channel
    bins: usize,
}

impl SceneDetector {
    /// Create a new scene detector
    ///
    /// # Arguments
    /// * `threshold` - Scene change threshold (0.2-0.5 recommended, higher = fewer scenes)
    pub fn new(threshold: f64) -> Self {
        SceneDetector {
            threshold: threshold.clamp(0.0, 1.0),
            prev_histogram: None,
            bins: 32, // 32 bins per channel = 32^3 = 32768 total bins
        }
    }

    /// Check if frame represents a scene change
    pub fn is_scene_change(&mut self, frame: &FrameData) -> bool {
        let histogram = self.calculate_histogram(frame);

        let is_change = if let Some(ref prev) = self.prev_histogram {
            let diff = self.histogram_difference(&histogram, prev);
            diff > self.threshold
        } else {
            false // First frame is not a scene change
        };

        self.prev_histogram = Some(histogram);

        is_change
    }

    /// Calculate RGB histogram
    fn calculate_histogram(&self, frame: &FrameData) -> Vec<u32> {
        let mut histogram = vec![0u32; self.bins * self.bins * self.bins];

        for rgb in frame.data.chunks_exact(3) {
            let r_bin = ((rgb[0] as usize * self.bins) / 256).min(self.bins - 1);
            let g_bin = ((rgb[1] as usize * self.bins) / 256).min(self.bins - 1);
            let b_bin = ((rgb[2] as usize * self.bins) / 256).min(self.bins - 1);

            let idx = r_bin * self.bins * self.bins + g_bin * self.bins + b_bin;
            histogram[idx] += 1;
        }

        histogram
    }

    /// Calculate histogram difference (chi-square distance, normalized)
    fn histogram_difference(&self, hist1: &[u32], hist2: &[u32]) -> f64 {
        let total: f64 = hist1.iter().map(|&x| x as f64).sum();

        if total == 0.0 {
            return 0.0;
        }

        let mut diff = 0.0;

        for ((&h1, &h2)) in hist1.iter().zip(hist2.iter()) {
            let h1_norm = h1 as f64 / total;
            let h2_norm = h2 as f64 / total;

            if h1_norm + h2_norm > 0.0 {
                let delta = h1_norm - h2_norm;
                diff += (delta * delta) / (h1_norm + h2_norm);
            }
        }

        (diff / 2.0).min(1.0)
    }

    /// Reset detector state
    pub fn reset(&mut self) {
        self.prev_histogram = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(width: usize, height: usize, brightness: u8) -> FrameData {
        let data = vec![brightness; width * height * 3];
        FrameData::new(0, data, width, height, 0.0)
    }

    #[test]
    fn test_frame_brightness() {
        let frame = create_test_frame(100, 100, 128);
        let brightness = frame.calculate_brightness();
        assert!((brightness - 0.5).abs() < 0.01); // ~0.5 for mid-gray
    }

    #[test]
    fn test_frame_quality_analysis() {
        let frame = create_test_frame(100, 100, 128);
        let quality = frame.analyze_quality();

        assert!(quality.brightness > 0.0 && quality.brightness <= 1.0);
        assert!(quality.score >= 0.0 && quality.score <= 1.0);
    }

    #[test]
    fn test_thumbnail_generator() {
        let generator = ThumbnailGenerator::new(320, 240);
        assert_eq!(generator.width, 320);
        assert_eq!(generator.height, 240);
    }

    #[test]
    fn test_thumbnail_method_selection() {
        let frames = vec![
            create_test_frame(100, 100, 50),
            create_test_frame(100, 100, 100),
            create_test_frame(100, 100, 150),
        ];

        let generator = ThumbnailGenerator::new(100, 100);

        // First frame
        let first = generator
            .generate_single(&frames, ThumbnailMethod::First)
            .unwrap();
        assert_eq!(first.width(), 100);

        // Middle frame
        let middle = generator
            .generate_single(&frames, ThumbnailMethod::Middle)
            .unwrap();
        assert_eq!(middle.width(), 100);
    }

    #[test]
    fn test_scene_detector() {
        let mut detector = SceneDetector::new(0.3);

        let frame1 = create_test_frame(100, 100, 50);
        let frame2 = create_test_frame(100, 100, 55); // Similar
        let frame3 = create_test_frame(100, 100, 200); // Very different

        assert!(!detector.is_scene_change(&frame1)); // First frame
        assert!(!detector.is_scene_change(&frame2)); // Similar to previous
        assert!(detector.is_scene_change(&frame3)); // Scene change
    }

    #[test]
    fn test_histogram_calculation() {
        let detector = SceneDetector::new(0.3);
        let frame = create_test_frame(100, 100, 128);

        let histogram = detector.calculate_histogram(&frame);
        let total: u32 = histogram.iter().sum();

        assert_eq!(total, 100 * 100); // All pixels counted
    }

    #[test]
    fn test_motion_blur_detection() {
        let frame1 = create_test_frame(100, 100, 100);
        let frame2 = create_test_frame(100, 100, 150);

        let blur = frame2.calculate_motion_blur(&frame1);
        assert!(blur > 0.0); // Should detect difference
    }
}
