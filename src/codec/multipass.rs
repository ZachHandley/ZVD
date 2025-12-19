//! Multi-Pass Encoding Support
//!
//! Multi-pass encoding (typically two-pass) analyzes video in a first pass to collect
//! statistics, then uses those statistics in a second pass to optimally allocate bitrate.
//!
//! ## Benefits
//!
//! - **Better Quality**: Optimal bitrate allocation based on frame complexity
//! - **Accurate Bitrate**: Hits target bitrate more precisely than single-pass
//! - **Efficient Compression**: Saves bits on simple scenes, spends on complex scenes
//! - **Consistent Quality**: More uniform perceptual quality across the video
//!
//! ## Workflow
//!
//! **First Pass:**
//! - Encode quickly (low quality settings)
//! - Analyze each frame's complexity
//! - Measure motion vectors and prediction accuracy
//! - Calculate optimal bitrate distribution
//! - Save statistics to file
//!
//! **Second Pass:**
//! - Load statistics file
//! - Encode with full quality settings
//! - Allocate bitrate based on frame complexity
//! - Use lookahead from statistics for better decisions

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Type of encoding pass
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassType {
    /// First pass: analyze video and collect statistics
    FirstPass,
    /// Second pass: encode using statistics
    SecondPass,
    /// N-pass: additional passes for further optimization (experimental)
    NthPass { pass_number: u32 },
}

impl PassType {
    /// Is this a first pass?
    pub fn is_first_pass(&self) -> bool {
        matches!(self, PassType::FirstPass)
    }

    /// Is this a final pass?
    pub fn is_final_pass(&self, total_passes: u32) -> bool {
        match self {
            PassType::FirstPass => total_passes == 1,
            PassType::SecondPass => total_passes == 2,
            PassType::NthPass { pass_number } => *pass_number == total_passes,
        }
    }

    /// Get pass number (1-indexed)
    pub fn pass_number(&self) -> u32 {
        match self {
            PassType::FirstPass => 1,
            PassType::SecondPass => 2,
            PassType::NthPass { pass_number } => *pass_number,
        }
    }
}

/// Rate control mode for multi-pass encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RateControlMode {
    /// Constant Bit Rate (CBR)
    CBR,
    /// Variable Bit Rate (VBR) with target average
    VBR,
    /// Constant Rate Factor (quality-based)
    CRF,
    /// Constrained VBR (VBR with max bitrate cap)
    CVBR,
}

/// Statistics for a single frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameStats {
    /// Frame index
    pub frame_index: u64,
    /// Frame type (I, P, B)
    pub frame_type: FrameType,
    /// Presentation timestamp
    pub pts: i64,
    /// Encoded size in bytes (from first pass)
    pub size: u64,
    /// Complexity metric (0.0-1.0, higher = more complex)
    pub complexity: f64,
    /// Motion activity metric (0.0-1.0, higher = more motion)
    pub motion: f64,
    /// Temporal complexity (difference from previous frame)
    pub temporal_complexity: f64,
    /// Spatial complexity (detail within frame)
    pub spatial_complexity: f64,
    /// Quantizer used in first pass
    pub quantizer: f64,
    /// Bits allocated in first pass
    pub bits: u64,
    /// SSIM score (if calculated)
    pub ssim: Option<f64>,
    /// PSNR score (if calculated)
    pub psnr: Option<f64>,
}

/// Frame type for encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrameType {
    /// Intra frame (keyframe)
    I,
    /// Predicted frame
    P,
    /// Bi-directional predicted frame
    B,
    /// Skip frame (copy previous)
    S,
}

impl std::fmt::Display for FrameType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameType::I => write!(f, "I"),
            FrameType::P => write!(f, "P"),
            FrameType::B => write!(f, "B"),
            FrameType::S => write!(f, "S"),
        }
    }
}

/// Complete statistics from first pass
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassStatistics {
    /// Total number of frames analyzed
    pub frame_count: u64,
    /// Duration of video
    pub duration: Duration,
    /// Target bitrate (bits per second)
    pub target_bitrate: u32,
    /// Rate control mode
    pub rate_control: RateControlMode,
    /// Frame rate
    pub framerate: f64,
    /// Per-frame statistics
    pub frames: Vec<FrameStats>,
    /// GOP (Group of Pictures) size
    pub gop_size: u32,
    /// Total bits used in first pass
    pub total_bits: u64,
    /// Average complexity
    pub avg_complexity: f64,
    /// Peak complexity
    pub peak_complexity: f64,
    /// Average motion
    pub avg_motion: f64,
    /// Complexity histogram (10 buckets)
    pub complexity_histogram: [u32; 10],
}

impl PassStatistics {
    /// Create new statistics
    pub fn new(
        target_bitrate: u32,
        framerate: f64,
        gop_size: u32,
        rate_control: RateControlMode,
    ) -> Self {
        PassStatistics {
            frame_count: 0,
            duration: Duration::ZERO,
            target_bitrate,
            rate_control,
            framerate,
            frames: Vec::new(),
            gop_size,
            total_bits: 0,
            avg_complexity: 0.0,
            peak_complexity: 0.0,
            avg_motion: 0.0,
            complexity_histogram: [0; 10],
        }
    }

    /// Add frame statistics
    pub fn add_frame(&mut self, frame: FrameStats) {
        self.total_bits += frame.bits;
        self.avg_complexity = (self.avg_complexity * self.frame_count as f64 + frame.complexity)
            / (self.frame_count as f64 + 1.0);
        self.peak_complexity = self.peak_complexity.max(frame.complexity);
        self.avg_motion = (self.avg_motion * self.frame_count as f64 + frame.motion)
            / (self.frame_count as f64 + 1.0);

        // Update complexity histogram
        let bucket = (frame.complexity * 9.99).floor() as usize;
        self.complexity_histogram[bucket.min(9)] += 1;

        self.frame_count += 1;
        self.frames.push(frame);
    }

    /// Finalize statistics
    pub fn finalize(&mut self) {
        if self.frame_count > 0 {
            self.duration = Duration::from_secs_f64(self.frame_count as f64 / self.framerate);
        }
    }

    /// Calculate optimal bitrate for a frame based on complexity
    pub fn calculate_frame_bitrate(&self, frame_index: usize, lookahead_window: usize) -> u32 {
        if frame_index >= self.frames.len() {
            return self.target_bitrate;
        }

        let frame = &self.frames[frame_index];

        // Base allocation based on frame complexity
        let complexity_factor = frame.complexity / self.avg_complexity;

        // Consider lookahead window for smoother allocation
        let mut lookahead_complexity = frame.complexity;
        let mut lookahead_count = 1.0;

        let end_idx = (frame_index + lookahead_window).min(self.frames.len());
        for i in (frame_index + 1)..end_idx {
            lookahead_complexity += self.frames[i].complexity;
            lookahead_count += 1.0;
        }
        lookahead_complexity /= lookahead_count;

        // Weight current frame more than lookahead
        let weighted_complexity =
            (frame.complexity * 0.7 + lookahead_complexity * 0.3) / self.avg_complexity;

        // Boost I-frames, normal P-frames, reduce B-frames
        let frame_type_factor = match frame.frame_type {
            FrameType::I => 1.5,
            FrameType::P => 1.0,
            FrameType::B => 0.7,
            FrameType::S => 0.1,
        };

        // Calculate final bitrate
        let base_bitrate = self.target_bitrate as f64 / self.framerate;
        let adjusted_bitrate = base_bitrate * weighted_complexity * frame_type_factor;

        // Clamp to reasonable range (0.5x to 3x target)
        let min_bitrate = base_bitrate * 0.5;
        let max_bitrate = base_bitrate * 3.0;

        adjusted_bitrate.clamp(min_bitrate, max_bitrate) as u32
    }

    /// Get quantizer recommendation for a frame
    pub fn calculate_frame_quantizer(&self, frame_index: usize, base_quantizer: f64) -> f64 {
        if frame_index >= self.frames.len() {
            return base_quantizer;
        }

        let frame = &self.frames[frame_index];

        // Adjust quantizer based on complexity (inverse relationship)
        // High complexity = lower quantizer (more bits)
        // Low complexity = higher quantizer (fewer bits)
        let complexity_factor = self.avg_complexity / frame.complexity.max(0.1);

        // Frame type adjustment
        let frame_type_factor = match frame.frame_type {
            FrameType::I => 0.8, // Lower quantizer for I-frames
            FrameType::P => 1.0,
            FrameType::B => 1.2, // Higher quantizer for B-frames
            FrameType::S => 2.0, // Much higher for skip frames
        };

        let adjusted_qp = base_quantizer * complexity_factor * frame_type_factor;

        // Clamp to valid quantizer range (0-51 for H.264/H.265)
        adjusted_qp.clamp(0.0, 51.0)
    }

    /// Save statistics to file
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| Error::InvalidInput(format!("Failed to serialize statistics: {}", e)))?;

        std::fs::write(path, json).map_err(|e| Error::Io(e))?;

        Ok(())
    }

    /// Load statistics from file
    pub fn load(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path).map_err(|e| Error::Io(e))?;

        let stats: PassStatistics = serde_json::from_str(&json)
            .map_err(|e| Error::InvalidInput(format!("Failed to deserialize statistics: {}", e)))?;

        Ok(stats)
    }

    /// Get summary statistics
    pub fn summary(&self) -> String {
        format!(
            "Multi-Pass Statistics:\n\
             - Frames: {}\n\
             - Duration: {:.2}s\n\
             - Target Bitrate: {} kbps\n\
             - Actual Bitrate: {} kbps\n\
             - Avg Complexity: {:.3}\n\
             - Peak Complexity: {:.3}\n\
             - Avg Motion: {:.3}\n\
             - Rate Control: {:?}",
            self.frame_count,
            self.duration.as_secs_f64(),
            self.target_bitrate / 1000,
            (self.total_bits as f64 / self.duration.as_secs_f64() / 1000.0) as u32,
            self.avg_complexity,
            self.peak_complexity,
            self.avg_motion,
            self.rate_control
        )
    }
}

/// Multi-pass encoder coordinator
pub struct MultiPassEncoder {
    /// Current pass type
    pass_type: PassType,
    /// Total number of passes
    total_passes: u32,
    /// Statistics file path
    stats_file: PathBuf,
    /// Statistics (None for first pass, Some for subsequent passes)
    statistics: Option<PassStatistics>,
    /// Current frame index
    current_frame: u64,
}

impl MultiPassEncoder {
    /// Create a new multi-pass encoder
    pub fn new(pass_type: PassType, total_passes: u32, stats_file: PathBuf) -> Self {
        MultiPassEncoder {
            pass_type,
            total_passes,
            stats_file,
            statistics: None,
            current_frame: 0,
        }
    }

    /// Create a two-pass encoder (most common)
    pub fn two_pass(stats_file: PathBuf, is_first_pass: bool) -> Self {
        let pass_type = if is_first_pass {
            PassType::FirstPass
        } else {
            PassType::SecondPass
        };

        Self::new(pass_type, 2, stats_file)
    }

    /// Initialize the encoder
    pub fn initialize(&mut self, target_bitrate: u32, framerate: f64, gop_size: u32) -> Result<()> {
        match self.pass_type {
            PassType::FirstPass => {
                // Create new statistics
                self.statistics = Some(PassStatistics::new(
                    target_bitrate,
                    framerate,
                    gop_size,
                    RateControlMode::VBR,
                ));
                Ok(())
            }
            PassType::SecondPass | PassType::NthPass { .. } => {
                // Load statistics from first pass
                if !self.stats_file.exists() {
                    return Err(Error::InvalidInput(format!(
                        "Statistics file not found: {:?}. Run first pass first.",
                        self.stats_file
                    )));
                }

                self.statistics = Some(PassStatistics::load(&self.stats_file)?);
                Ok(())
            }
        }
    }

    /// Record frame statistics (first pass)
    pub fn record_frame(&mut self, frame_stats: FrameStats) -> Result<()> {
        if !self.pass_type.is_first_pass() {
            return Err(Error::InvalidInput(
                "Cannot record statistics in non-first pass".to_string(),
            ));
        }

        if let Some(ref mut stats) = self.statistics {
            stats.add_frame(frame_stats);
            self.current_frame += 1;
            Ok(())
        } else {
            Err(Error::InvalidInput(
                "Statistics not initialized".to_string(),
            ))
        }
    }

    /// Get bitrate recommendation for current frame (second pass)
    pub fn get_frame_bitrate(&self, lookahead_window: usize) -> Result<u32> {
        if self.pass_type.is_first_pass() {
            return Err(Error::InvalidInput(
                "Bitrate recommendations only available in second pass".to_string(),
            ));
        }

        if let Some(ref stats) = self.statistics {
            Ok(stats.calculate_frame_bitrate(self.current_frame as usize, lookahead_window))
        } else {
            Err(Error::InvalidInput("Statistics not loaded".to_string()))
        }
    }

    /// Get quantizer recommendation for current frame (second pass)
    pub fn get_frame_quantizer(&self, base_quantizer: f64) -> Result<f64> {
        if self.pass_type.is_first_pass() {
            return Err(Error::InvalidInput(
                "Quantizer recommendations only available in second pass".to_string(),
            ));
        }

        if let Some(ref stats) = self.statistics {
            Ok(stats.calculate_frame_quantizer(self.current_frame as usize, base_quantizer))
        } else {
            Err(Error::InvalidInput("Statistics not loaded".to_string()))
        }
    }

    /// Advance to next frame
    pub fn next_frame(&mut self) {
        self.current_frame += 1;
    }

    /// Finalize encoding
    pub fn finalize(&mut self) -> Result<()> {
        if self.pass_type.is_first_pass() {
            if let Some(ref mut stats) = self.statistics {
                stats.finalize();
                stats.save(&self.stats_file)?;
                println!(
                    "First pass complete. Statistics saved to: {:?}",
                    self.stats_file
                );
                println!("{}", stats.summary());
            }
        }

        Ok(())
    }

    /// Get statistics (if available)
    pub fn statistics(&self) -> Option<&PassStatistics> {
        self.statistics.as_ref()
    }

    /// Get current pass type
    pub fn pass_type(&self) -> PassType {
        self.pass_type
    }

    /// Is this the final pass?
    pub fn is_final_pass(&self) -> bool {
        self.pass_type.is_final_pass(self.total_passes)
    }
}

/// Frame complexity analyzer
pub struct ComplexityAnalyzer;

impl ComplexityAnalyzer {
    /// Calculate frame complexity from pixel data
    pub fn analyze_frame(
        data: &[u8],
        width: usize,
        height: usize,
        prev_data: Option<&[u8]>,
    ) -> (f64, f64, f64) {
        let spatial = Self::spatial_complexity(data, width, height);
        let temporal = if let Some(prev) = prev_data {
            Self::temporal_complexity(data, prev, width, height)
        } else {
            0.0
        };
        let motion = temporal; // Simplified: temporal difference approximates motion

        (spatial, temporal, motion)
    }

    /// Calculate spatial complexity (variance/detail)
    fn spatial_complexity(data: &[u8], width: usize, height: usize) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        // Calculate variance of luminance (Y plane in YUV420P)
        let y_plane_size = width * height;
        let y_data = &data[..y_plane_size.min(data.len())];

        let mean = y_data.iter().map(|&x| x as f64).sum::<f64>() / y_data.len() as f64;
        let variance = y_data
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / y_data.len() as f64;

        // Normalize to 0.0-1.0 (variance can be 0-255^2 = 65025)
        (variance / 65025.0).min(1.0)
    }

    /// Calculate temporal complexity (difference from previous frame)
    fn temporal_complexity(data: &[u8], prev_data: &[u8], width: usize, height: usize) -> f64 {
        let y_plane_size = width * height;
        let len = y_plane_size.min(data.len()).min(prev_data.len());

        if len == 0 {
            return 0.0;
        }

        let sad: u64 = data[..len]
            .iter()
            .zip(prev_data[..len].iter())
            .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs() as u64)
            .sum();

        // Normalize to 0.0-1.0 (max SAD per pixel is 255)
        (sad as f64 / (len as f64 * 255.0)).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pass_type() {
        let first = PassType::FirstPass;
        assert!(first.is_first_pass());
        assert_eq!(first.pass_number(), 1);
        assert!(first.is_final_pass(1));
        assert!(!first.is_final_pass(2));

        let second = PassType::SecondPass;
        assert!(!second.is_first_pass());
        assert_eq!(second.pass_number(), 2);
        assert!(second.is_final_pass(2));
    }

    #[test]
    fn test_frame_stats() {
        let stats = FrameStats {
            frame_index: 0,
            frame_type: FrameType::I,
            pts: 0,
            size: 10000,
            complexity: 0.8,
            motion: 0.3,
            temporal_complexity: 0.2,
            spatial_complexity: 0.9,
            quantizer: 23.0,
            bits: 80000,
            ssim: Some(0.95),
            psnr: Some(42.0),
        };

        assert_eq!(stats.frame_type, FrameType::I);
        assert_eq!(stats.complexity, 0.8);
    }

    #[test]
    fn test_pass_statistics() {
        let mut stats = PassStatistics::new(5_000_000, 30.0, 60, RateControlMode::VBR);

        // Add some frames
        for i in 0..100 {
            let frame = FrameStats {
                frame_index: i,
                frame_type: if i % 60 == 0 {
                    FrameType::I
                } else if i % 3 == 0 {
                    FrameType::B
                } else {
                    FrameType::P
                },
                pts: i as i64,
                size: 8000,
                complexity: 0.5 + (i as f64 % 10.0) / 20.0,
                motion: 0.3,
                temporal_complexity: 0.4,
                spatial_complexity: 0.6,
                quantizer: 25.0,
                bits: 64000,
                ssim: None,
                psnr: None,
            };

            stats.add_frame(frame);
        }

        stats.finalize();

        assert_eq!(stats.frame_count, 100);
        assert!(stats.avg_complexity > 0.0);
        assert!(stats.peak_complexity >= stats.avg_complexity);
    }

    #[test]
    fn test_bitrate_calculation() {
        let mut stats = PassStatistics::new(5_000_000, 30.0, 60, RateControlMode::VBR);

        // Add frames with varying complexity
        for i in 0..60 {
            stats.add_frame(FrameStats {
                frame_index: i,
                frame_type: if i == 0 { FrameType::I } else { FrameType::P },
                pts: i as i64,
                size: 8000,
                complexity: if i < 30 { 0.3 } else { 0.9 }, // Low then high complexity
                motion: 0.5,
                temporal_complexity: 0.5,
                spatial_complexity: 0.5,
                quantizer: 25.0,
                bits: 64000,
                ssim: None,
                psnr: None,
            });
        }

        stats.finalize();

        // High complexity frames should get more bitrate
        let low_complexity_bitrate = stats.calculate_frame_bitrate(10, 5);
        let high_complexity_bitrate = stats.calculate_frame_bitrate(50, 5);

        assert!(high_complexity_bitrate > low_complexity_bitrate);
    }

    #[test]
    fn test_complexity_analyzer() {
        let width = 64;
        let height = 64;
        let size = width * height;

        // Create flat frame
        let flat_frame = vec![128u8; size];
        let (spatial, _, _) = ComplexityAnalyzer::analyze_frame(&flat_frame, width, height, None);
        assert!(spatial < 0.01); // Very low complexity

        // Create noisy frame
        let mut noisy_frame = vec![0u8; size];
        for i in 0..size {
            noisy_frame[i] = ((i * 123) % 256) as u8;
        }
        let (spatial_noisy, _, _) =
            ComplexityAnalyzer::analyze_frame(&noisy_frame, width, height, None);
        assert!(spatial_noisy > spatial); // Higher complexity

        // Test temporal complexity
        let (_, temporal, _) =
            ComplexityAnalyzer::analyze_frame(&noisy_frame, width, height, Some(&flat_frame));
        assert!(temporal > 0.0); // Should detect difference
    }

    #[test]
    fn test_multipass_encoder_lifecycle() {
        let temp_dir = std::env::temp_dir();
        let stats_file = temp_dir.join("test_multipass_stats.json");

        // First pass
        {
            let mut encoder = MultiPassEncoder::two_pass(stats_file.clone(), true);
            encoder.initialize(5_000_000, 30.0, 60).unwrap();

            // Record some frames
            for i in 0..10 {
                encoder
                    .record_frame(FrameStats {
                        frame_index: i,
                        frame_type: if i == 0 { FrameType::I } else { FrameType::P },
                        pts: i as i64,
                        size: 8000,
                        complexity: 0.5,
                        motion: 0.3,
                        temporal_complexity: 0.4,
                        spatial_complexity: 0.6,
                        quantizer: 25.0,
                        bits: 64000,
                        ssim: None,
                        psnr: None,
                    })
                    .unwrap();
            }

            encoder.finalize().unwrap();
        }

        // Second pass
        {
            let mut encoder = MultiPassEncoder::two_pass(stats_file.clone(), false);
            encoder.initialize(5_000_000, 30.0, 60).unwrap();

            // Get bitrate recommendation
            let bitrate = encoder.get_frame_bitrate(5).unwrap();
            assert!(bitrate > 0);

            // Get quantizer recommendation
            let qp = encoder.get_frame_quantizer(25.0).unwrap();
            assert!(qp >= 0.0 && qp <= 51.0);
        }

        // Cleanup
        let _ = std::fs::remove_file(stats_file);
    }
}
