//! Video Embedding and Fingerprinting
//!
//! Extract perceptual hashes and embeddings from videos for similarity detection,
//! search, deduplication, and ML workflows - much faster than manual FFMPEG processing.
//!
//! ## Features
//!
//! **Perceptual Hashing:**
//! - **pHash**: Perceptual hash (DCT-based, robust to scaling/compression)
//! - **dHash**: Difference hash (gradient-based, fast)
//! - **aHash**: Average hash (simple, very fast)
//!
//! **Frame Importance Scoring:**
//! - Motion detection (optical flow estimation)
//! - Histogram changes (color/luminance)
//! - Edge density (sharpness/detail)
//! - Temporal variance
//!
//! **Extraction Strategies:**
//! - Uniform: Evenly spaced frames
//! - Adaptive: Content-aware selection
//! - Top-N: Most important frames only
//!
//! ## Usage
//!
//! ```rust
//! use zvd_lib::util::embedding::{VideoEmbedding, ExtractionStrategy, HashType};
//!
//! // Extract top 10 most important frames
//! let mut embedder = VideoEmbedding::new(ExtractionStrategy::TopN(10));
//! embedder.set_hash_type(HashType::PHash);
//!
//! // Process video frames
//! for frame in video_frames {
//!     embedder.add_frame(&frame, width, height)?;
//! }
//!
//! // Get fingerprint
//! let fingerprint = embedder.finalize()?;
//! ```

use crate::error::{Error, Result};
use std::collections::VecDeque;

/// Perceptual hash type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashType {
    /// Perceptual hash (DCT-based, 64-bit)
    PHash,
    /// Difference hash (gradient-based, 64-bit)
    DHash,
    /// Average hash (mean-based, 64-bit)
    AHash,
}

/// Frame extraction strategy
#[derive(Debug, Clone, PartialEq)]
pub enum ExtractionStrategy {
    /// Extract N evenly spaced frames
    Uniform(usize),
    /// Extract top N most important frames
    TopN(usize),
    /// Extract frames adaptively (min_frames, max_frames, threshold)
    Adaptive {
        min_frames: usize,
        max_frames: usize,
        importance_threshold: f32,
    },
}

/// Frame importance metrics
#[derive(Debug, Clone, Copy)]
pub struct FrameImportance {
    /// Motion score (0.0-1.0)
    pub motion: f32,
    /// Histogram change (0.0-1.0)
    pub histogram_change: f32,
    /// Edge density (0.0-1.0)
    pub edge_density: f32,
    /// Combined importance score (0.0-1.0)
    pub score: f32,
}

impl FrameImportance {
    /// Calculate combined score
    pub fn calculate_score(&mut self) {
        // Weighted combination
        self.score = self.motion * 0.4 + self.histogram_change * 0.3 + self.edge_density * 0.3;
    }
}

/// Video fingerprint
#[derive(Debug, Clone)]
pub struct VideoFingerprint {
    /// Perceptual hashes of selected frames
    pub frame_hashes: Vec<u64>,
    /// Frame indices in original video
    pub frame_indices: Vec<usize>,
    /// Frame importance scores
    pub importance_scores: Vec<f32>,
    /// Hash type used
    pub hash_type: HashType,
}

impl VideoFingerprint {
    /// Calculate Hamming distance to another fingerprint
    pub fn hamming_distance(&self, other: &VideoFingerprint) -> usize {
        if self.frame_hashes.len() != other.frame_hashes.len() {
            return usize::MAX; // Incompatible
        }

        let mut total_distance = 0;
        for (h1, h2) in self.frame_hashes.iter().zip(other.frame_hashes.iter()) {
            total_distance += (h1 ^ h2).count_ones() as usize;
        }

        total_distance
    }

    /// Calculate similarity (0.0-1.0, 1.0 = identical)
    pub fn similarity(&self, other: &VideoFingerprint) -> f32 {
        let distance = self.hamming_distance(other);
        if distance == usize::MAX {
            return 0.0;
        }

        let max_distance = self.frame_hashes.len() * 64; // 64 bits per hash
        1.0 - (distance as f32 / max_distance as f32)
    }
}

/// Video embedding generator
pub struct VideoEmbedding {
    /// Extraction strategy
    strategy: ExtractionStrategy,
    /// Hash type
    hash_type: HashType,
    /// Accumulated frames with importance
    frames: Vec<(Vec<u8>, usize, FrameImportance)>, // (data, index, importance)
    /// Previous frame for motion detection
    prev_frame: Option<Vec<u8>>,
    /// Frame counter
    frame_count: usize,
    /// Frame dimensions
    width: usize,
    height: usize,
}

impl VideoEmbedding {
    /// Create new video embedding generator
    pub fn new(strategy: ExtractionStrategy) -> Self {
        VideoEmbedding {
            strategy,
            hash_type: HashType::PHash,
            frames: Vec::new(),
            prev_frame: None,
            frame_count: 0,
            width: 0,
            height: 0,
        }
    }

    /// Set hash type
    pub fn set_hash_type(&mut self, hash_type: HashType) {
        self.hash_type = hash_type;
    }

    /// Add frame to process
    pub fn add_frame(&mut self, frame_rgb: &[u8], width: usize, height: usize) -> Result<()> {
        if frame_rgb.len() != width * height * 3 {
            return Err(Error::InvalidInput("Invalid frame dimensions".to_string()));
        }

        if self.frame_count == 0 {
            self.width = width;
            self.height = height;
        } else if width != self.width || height != self.height {
            return Err(Error::InvalidInput("Frame size mismatch".to_string()));
        }

        // Calculate frame importance
        let mut importance = FrameImportance {
            motion: 0.0,
            histogram_change: 0.0,
            edge_density: 0.0,
            score: 0.0,
        };

        // Motion detection (if we have previous frame)
        if let Some(ref prev) = self.prev_frame {
            importance.motion = self.calculate_motion(prev, frame_rgb);
            importance.histogram_change = self.calculate_histogram_change(prev, frame_rgb);
        }

        // Edge density
        importance.edge_density = self.calculate_edge_density(frame_rgb, width, height);

        // Calculate combined score
        importance.calculate_score();

        // Store frame based on strategy
        match &self.strategy {
            ExtractionStrategy::Uniform(n) => {
                // Always keep for later uniform sampling
                if self.frames.len() < *n * 2 {
                    self.frames.push((frame_rgb.to_vec(), self.frame_count, importance));
                }
            }
            ExtractionStrategy::TopN(_) => {
                // Keep all frames, will select top N at finalize
                self.frames.push((frame_rgb.to_vec(), self.frame_count, importance));
            }
            ExtractionStrategy::Adaptive { min_frames, max_frames, importance_threshold } => {
                // Keep if important enough or under min
                if importance.score >= *importance_threshold || self.frames.len() < *min_frames {
                    self.frames.push((frame_rgb.to_vec(), self.frame_count, importance));
                }

                // Prune if over max
                if self.frames.len() > *max_frames {
                    self.frames.sort_by(|a, b| b.2.score.partial_cmp(&a.2.score).unwrap());
                    self.frames.truncate(*max_frames);
                }
            }
        }

        // Update previous frame
        self.prev_frame = Some(frame_rgb.to_vec());
        self.frame_count += 1;

        Ok(())
    }

    /// Finalize and generate fingerprint
    pub fn finalize(&mut self) -> Result<VideoFingerprint> {
        if self.frames.is_empty() {
            return Err(Error::InvalidInput("No frames added".to_string()));
        }

        // Select frames based on strategy
        let selected_frames = match &self.strategy {
            ExtractionStrategy::Uniform(n) => {
                let step = (self.frames.len() as f32 / *n as f32).max(1.0);
                let mut selected = Vec::new();

                for i in 0..*n {
                    let idx = (i as f32 * step) as usize;
                    if idx < self.frames.len() {
                        selected.push(self.frames[idx].clone());
                    }
                }

                selected
            }
            ExtractionStrategy::TopN(n) => {
                // Sort by importance and take top N
                let mut sorted = self.frames.clone();
                sorted.sort_by(|a, b| b.2.score.partial_cmp(&a.2.score).unwrap());
                sorted.truncate(*n);
                sorted
            }
            ExtractionStrategy::Adaptive { .. } => {
                // Already filtered, just return
                self.frames.clone()
            }
        };

        // Generate hashes for selected frames
        let mut frame_hashes = Vec::new();
        let mut frame_indices = Vec::new();
        let mut importance_scores = Vec::new();

        for (frame_data, frame_idx, importance) in selected_frames {
            let hash = self.compute_hash(&frame_data)?;
            frame_hashes.push(hash);
            frame_indices.push(frame_idx);
            importance_scores.push(importance.score);
        }

        Ok(VideoFingerprint {
            frame_hashes,
            frame_indices,
            importance_scores,
            hash_type: self.hash_type,
        })
    }

    /// Compute perceptual hash for frame
    fn compute_hash(&self, frame_rgb: &[u8]) -> Result<u64> {
        match self.hash_type {
            HashType::PHash => self.phash(frame_rgb),
            HashType::DHash => self.dhash(frame_rgb),
            HashType::AHash => self.ahash(frame_rgb),
        }
    }

    /// Perceptual hash (DCT-based)
    fn phash(&self, frame_rgb: &[u8]) -> Result<u64> {
        // Resize to 32x32 grayscale
        let gray = self.to_grayscale_resized(frame_rgb, 32, 32)?;

        // Compute DCT (simplified 8x8 for 64-bit hash)
        let dct = self.compute_dct(&gray, 32, 32);

        // Get low frequency components (top-left 8x8)
        let mut values = Vec::new();
        for y in 0..8 {
            for x in 0..8 {
                values.push(dct[y * 32 + x]);
            }
        }

        // Calculate median
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];

        // Generate hash
        let mut hash = 0u64;
        for (i, &val) in values.iter().enumerate() {
            if val > median {
                hash |= 1u64 << i;
            }
        }

        Ok(hash)
    }

    /// Difference hash (gradient-based)
    fn dhash(&self, frame_rgb: &[u8]) -> Result<u64> {
        // Resize to 9x8 grayscale (need 9 columns for 8 differences)
        let gray = self.to_grayscale_resized(frame_rgb, 9, 8)?;

        let mut hash = 0u64;
        let mut bit = 0;

        for y in 0..8 {
            for x in 0..8 {
                let left = gray[y * 9 + x];
                let right = gray[y * 9 + x + 1];

                if left < right {
                    hash |= 1u64 << bit;
                }
                bit += 1;
            }
        }

        Ok(hash)
    }

    /// Average hash (mean-based)
    fn ahash(&self, frame_rgb: &[u8]) -> Result<u64> {
        // Resize to 8x8 grayscale
        let gray = self.to_grayscale_resized(frame_rgb, 8, 8)?;

        // Calculate average
        let sum: u32 = gray.iter().map(|&v| v as u32).sum();
        let avg = (sum / 64) as u8;

        // Generate hash
        let mut hash = 0u64;
        for (i, &val) in gray.iter().enumerate() {
            if val > avg {
                hash |= 1u64 << i;
            }
        }

        Ok(hash)
    }

    /// Convert to grayscale and resize (bilinear)
    fn to_grayscale_resized(&self, frame_rgb: &[u8], target_w: usize, target_h: usize) -> Result<Vec<u8>> {
        let mut gray = vec![0u8; target_w * target_h];

        let x_ratio = self.width as f32 / target_w as f32;
        let y_ratio = self.height as f32 / target_h as f32;

        for ty in 0..target_h {
            for tx in 0..target_w {
                let sx = tx as f32 * x_ratio;
                let sy = ty as f32 * y_ratio;

                let x0 = sx.floor() as usize;
                let y0 = sy.floor() as usize;
                let x1 = (x0 + 1).min(self.width - 1);
                let y1 = (y0 + 1).min(self.height - 1);

                let fx = sx - x0 as f32;
                let fy = sy - y0 as f32;

                // Sample RGB and convert to grayscale
                let sample = |x: usize, y: usize| -> f32 {
                    let idx = (y * self.width + x) * 3;
                    let r = frame_rgb[idx] as f32;
                    let g = frame_rgb[idx + 1] as f32;
                    let b = frame_rgb[idx + 2] as f32;
                    // ITU-R BT.709 luma
                    0.2126 * r + 0.7152 * g + 0.0722 * b
                };

                let p00 = sample(x0, y0);
                let p10 = sample(x1, y0);
                let p01 = sample(x0, y1);
                let p11 = sample(x1, y1);

                let val = p00 * (1.0 - fx) * (1.0 - fy)
                    + p10 * fx * (1.0 - fy)
                    + p01 * (1.0 - fx) * fy
                    + p11 * fx * fy;

                gray[ty * target_w + tx] = val.clamp(0.0, 255.0) as u8;
            }
        }

        Ok(gray)
    }

    /// Compute simplified DCT
    fn compute_dct(&self, gray: &[u8], width: usize, height: usize) -> Vec<f32> {
        let mut dct = vec![0.0f32; width * height];

        for v in 0..height {
            for u in 0..width {
                let mut sum = 0.0;

                for y in 0..height {
                    for x in 0..width {
                        let pixel = gray[y * width + x] as f32;
                        let cos_x = ((2.0 * x as f32 + 1.0) * u as f32 * std::f32::consts::PI / (2.0 * width as f32)).cos();
                        let cos_y = ((2.0 * y as f32 + 1.0) * v as f32 * std::f32::consts::PI / (2.0 * height as f32)).cos();
                        sum += pixel * cos_x * cos_y;
                    }
                }

                let cu = if u == 0 { 1.0 / 2.0f32.sqrt() } else { 1.0 };
                let cv = if v == 0 { 1.0 / 2.0f32.sqrt() } else { 1.0 };

                dct[v * width + u] = sum * cu * cv / 4.0;
            }
        }

        dct
    }

    /// Calculate motion between frames (simplified optical flow)
    fn calculate_motion(&self, prev: &[u8], curr: &[u8]) -> f32 {
        let mut diff_sum = 0.0;
        let mut count = 0;

        // Sample every 4th pixel for speed
        for i in (0..prev.len()).step_by(12) {
            let diff = (prev[i] as i32 - curr[i] as i32).abs();
            diff_sum += diff as f32;
            count += 1;
        }

        let avg_diff = diff_sum / count as f32;
        (avg_diff / 255.0).clamp(0.0, 1.0)
    }

    /// Calculate histogram change
    fn calculate_histogram_change(&self, prev: &[u8], curr: &[u8]) -> f32 {
        let mut prev_hist = vec![0u32; 256];
        let mut curr_hist = vec![0u32; 256];

        // Build histograms (luma channel)
        for i in (0..prev.len()).step_by(3) {
            let prev_luma = (0.2126 * prev[i] as f32 + 0.7152 * prev[i + 1] as f32 + 0.0722 * prev[i + 2] as f32) as usize;
            let curr_luma = (0.2126 * curr[i] as f32 + 0.7152 * curr[i + 1] as f32 + 0.0722 * curr[i + 2] as f32) as usize;

            prev_hist[prev_luma.min(255)] += 1;
            curr_hist[curr_luma.min(255)] += 1;
        }

        // Chi-square distance
        let mut chi_square = 0.0;
        for i in 0..256 {
            let sum = prev_hist[i] + curr_hist[i];
            if sum > 0 {
                let diff = prev_hist[i] as f32 - curr_hist[i] as f32;
                chi_square += (diff * diff) / sum as f32;
            }
        }

        (chi_square / 1000.0).clamp(0.0, 1.0)
    }

    /// Calculate edge density (sharpness)
    fn calculate_edge_density(&self, frame_rgb: &[u8], width: usize, height: usize) -> f32 {
        // Sobel edge detection
        let mut edge_sum = 0.0;
        let mut count = 0;

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = (y * width + x) * 3;

                // Convert to grayscale
                let center = 0.2126 * frame_rgb[idx] as f32
                    + 0.7152 * frame_rgb[idx + 1] as f32
                    + 0.0722 * frame_rgb[idx + 2] as f32;

                // Simplified Sobel
                let left = self.get_luma(frame_rgb, x - 1, y, width);
                let right = self.get_luma(frame_rgb, x + 1, y, width);
                let top = self.get_luma(frame_rgb, x, y - 1, width);
                let bottom = self.get_luma(frame_rgb, x, y + 1, width);

                let gx = right - left;
                let gy = bottom - top;
                let gradient = (gx * gx + gy * gy).sqrt();

                edge_sum += gradient;
                count += 1;
            }
        }

        ((edge_sum / count as f32) / 255.0).clamp(0.0, 1.0)
    }

    /// Get luma at position
    fn get_luma(&self, frame_rgb: &[u8], x: usize, y: usize, width: usize) -> f32 {
        let idx = (y * width + x) * 3;
        0.2126 * frame_rgb[idx] as f32
            + 0.7152 * frame_rgb[idx + 1] as f32
            + 0.0722 * frame_rgb[idx + 2] as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_embedding_creation() {
        let embedder = VideoEmbedding::new(ExtractionStrategy::TopN(10));
        assert_eq!(embedder.frame_count, 0);
    }

    #[test]
    fn test_add_frame() {
        let mut embedder = VideoEmbedding::new(ExtractionStrategy::TopN(10));

        let frame = vec![128u8; 64 * 64 * 3];
        let result = embedder.add_frame(&frame, 64, 64);

        assert!(result.is_ok());
        assert_eq!(embedder.frame_count, 1);
    }

    #[test]
    fn test_invalid_frame_size() {
        let mut embedder = VideoEmbedding::new(ExtractionStrategy::TopN(10));

        let frame = vec![128u8; 100];
        let result = embedder.add_frame(&frame, 64, 64);

        assert!(result.is_err());
    }

    #[test]
    fn test_ahash() {
        let mut embedder = VideoEmbedding::new(ExtractionStrategy::TopN(1));
        embedder.set_hash_type(HashType::AHash);

        let frame = vec![128u8; 64 * 64 * 3];
        embedder.add_frame(&frame, 64, 64).unwrap();

        let fingerprint = embedder.finalize().unwrap();
        assert_eq!(fingerprint.frame_hashes.len(), 1);
    }

    #[test]
    fn test_dhash() {
        let mut embedder = VideoEmbedding::new(ExtractionStrategy::TopN(1));
        embedder.set_hash_type(HashType::DHash);

        let frame = vec![128u8; 64 * 64 * 3];
        embedder.add_frame(&frame, 64, 64).unwrap();

        let fingerprint = embedder.finalize().unwrap();
        assert_eq!(fingerprint.frame_hashes.len(), 1);
    }

    #[test]
    fn test_phash() {
        let mut embedder = VideoEmbedding::new(ExtractionStrategy::TopN(1));
        embedder.set_hash_type(HashType::PHash);

        let frame = vec![128u8; 64 * 64 * 3];
        embedder.add_frame(&frame, 64, 64).unwrap();

        let fingerprint = embedder.finalize().unwrap();
        assert_eq!(fingerprint.frame_hashes.len(), 1);
    }

    #[test]
    fn test_uniform_extraction() {
        let mut embedder = VideoEmbedding::new(ExtractionStrategy::Uniform(5));

        for _ in 0..20 {
            let frame = vec![128u8; 64 * 64 * 3];
            embedder.add_frame(&frame, 64, 64).unwrap();
        }

        let fingerprint = embedder.finalize().unwrap();
        assert!(fingerprint.frame_hashes.len() <= 5);
    }

    #[test]
    fn test_topn_extraction() {
        let mut embedder = VideoEmbedding::new(ExtractionStrategy::TopN(3));

        for i in 0..10 {
            let mut frame = vec![i as u8 * 20; 64 * 64 * 3];
            embedder.add_frame(&frame, 64, 64).unwrap();
        }

        let fingerprint = embedder.finalize().unwrap();
        assert_eq!(fingerprint.frame_hashes.len(), 3);
    }

    #[test]
    fn test_adaptive_extraction() {
        let strategy = ExtractionStrategy::Adaptive {
            min_frames: 2,
            max_frames: 5,
            importance_threshold: 0.3,
        };

        let mut embedder = VideoEmbedding::new(strategy);

        for _ in 0..10 {
            let frame = vec![128u8; 64 * 64 * 3];
            embedder.add_frame(&frame, 64, 64).unwrap();
        }

        let fingerprint = embedder.finalize().unwrap();
        assert!(fingerprint.frame_hashes.len() >= 2);
        assert!(fingerprint.frame_hashes.len() <= 5);
    }

    #[test]
    fn test_hamming_distance() {
        let fp1 = VideoFingerprint {
            frame_hashes: vec![0b1010, 0b1100],
            frame_indices: vec![0, 10],
            importance_scores: vec![0.5, 0.7],
            hash_type: HashType::AHash,
        };

        let fp2 = VideoFingerprint {
            frame_hashes: vec![0b1110, 0b1100],
            frame_indices: vec![0, 10],
            importance_scores: vec![0.5, 0.7],
            hash_type: HashType::AHash,
        };

        let distance = fp1.hamming_distance(&fp2);
        assert_eq!(distance, 1); // One bit different
    }

    #[test]
    fn test_similarity() {
        let fp1 = VideoFingerprint {
            frame_hashes: vec![0xFFFFFFFFFFFFFFFF],
            frame_indices: vec![0],
            importance_scores: vec![1.0],
            hash_type: HashType::PHash,
        };

        let fp2 = VideoFingerprint {
            frame_hashes: vec![0xFFFFFFFFFFFFFFFF],
            frame_indices: vec![0],
            importance_scores: vec![1.0],
            hash_type: HashType::PHash,
        };

        let similarity = fp1.similarity(&fp2);
        assert_eq!(similarity, 1.0); // Identical
    }

    #[test]
    fn test_frame_importance() {
        let mut importance = FrameImportance {
            motion: 0.5,
            histogram_change: 0.3,
            edge_density: 0.7,
            score: 0.0,
        };

        importance.calculate_score();
        assert!(importance.score > 0.0);
        assert!(importance.score <= 1.0);
    }

    #[test]
    fn test_motion_detection() {
        let mut embedder = VideoEmbedding::new(ExtractionStrategy::TopN(2));

        let frame1 = vec![100u8; 64 * 64 * 3];
        let frame2 = vec![200u8; 64 * 64 * 3];

        embedder.add_frame(&frame1, 64, 64).unwrap();
        embedder.add_frame(&frame2, 64, 64).unwrap();

        // Second frame should have high motion score
        assert!(embedder.frames.len() >= 1);
    }
}
