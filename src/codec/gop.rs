//! GOP (Group of Pictures) Management
//!
//! Intelligent GOP structure management with scene-aware keyframe placement,
//! adaptive sizing, and B-frame pyramid support for optimal encoding efficiency.
//!
//! ## GOP Structure
//!
//! A GOP is a sequence of frames between two keyframes (I-frames).
//!
//! **Frame Types:**
//! - **I-frame (Intra)**: Full frame, no dependencies (keyframe/IDR)
//! - **P-frame (Predicted)**: References previous frames
//! - **B-frame (Bi-directional)**: References both previous and future frames
//!
//! **GOP Patterns:**
//! - IBBPBBPBBPBBI (M=3, N=12)
//! - IBPBPBPBPBPBI (M=2, N=12)
//! - IIIIIIIIIIII (All-intra)
//!
//! ## Scene-Aware Placement
//!
//! Automatically inserts keyframes at scene changes to:
//! - Improve seeking accuracy
//! - Reduce temporal artifacts
//! - Optimize compression efficiency
//!
//! ## Usage
//!
//! ```rust
//! use zvd_lib::codec::gop::{GopManager, GopConfig, GopStructure};
//!
//! // Create GOP manager
//! let config = GopConfig::default()
//!     .with_max_size(120)
//!     .with_min_size(30)
//!     .with_b_frames(3)
//!     .with_scene_detection(true);
//!
//! let mut gop = GopManager::new(config);
//!
//! // Process frames
//! for frame in frames {
//!     let frame_type = gop.determine_frame_type(&frame)?;
//!     encoder.encode_with_type(frame, frame_type);
//! }
//! ```

use crate::error::{Error, Result};
use crate::util::thumbnail::FrameData;
use std::collections::VecDeque;

/// Frame type in GOP structure
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GopFrameType {
    /// I-frame (Intra, Keyframe)
    I,
    /// IDR frame (Instantaneous Decoder Refresh)
    IDR,
    /// P-frame (Predicted)
    P,
    /// B-frame (Bi-directional)
    B,
}

impl GopFrameType {
    /// Is this a keyframe (I or IDR)?
    pub fn is_keyframe(&self) -> bool {
        matches!(self, GopFrameType::I | GopFrameType::IDR)
    }

    /// Is this a reference frame (I, IDR, or P)?
    pub fn is_reference(&self) -> bool {
        matches!(self, GopFrameType::I | GopFrameType::IDR | GopFrameType::P)
    }

    /// Get display name
    pub fn name(&self) -> &'static str {
        match self {
            GopFrameType::I => "I-frame",
            GopFrameType::IDR => "IDR-frame",
            GopFrameType::P => "P-frame",
            GopFrameType::B => "B-frame",
        }
    }
}

/// GOP structure type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GopStructure {
    /// Closed GOP (no frames reference across GOP boundaries)
    Closed,
    /// Open GOP (frames can reference across boundaries)
    Open,
}

/// B-frame pyramid structure
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BFramePyramid {
    /// No pyramid (flat B-frames)
    None,
    /// Strict pyramid (hierarchical references)
    Strict,
    /// Normal pyramid (allows some flexibility)
    Normal,
}

/// GOP configuration
#[derive(Debug, Clone)]
pub struct GopConfig {
    /// Maximum GOP size (frames)
    pub max_size: usize,
    /// Minimum GOP size (frames)
    pub min_size: usize,
    /// Number of B-frames between reference frames
    pub b_frames: usize,
    /// GOP structure type
    pub structure: GopStructure,
    /// B-frame pyramid structure
    pub b_pyramid: BFramePyramid,
    /// Enable scene-aware keyframe placement
    pub scene_detection: bool,
    /// Scene change threshold (0.0-1.0)
    pub scene_threshold: f64,
    /// Enable adaptive GOP sizing
    pub adaptive: bool,
    /// Force IDR at GOP start
    pub force_idr: bool,
    /// Keyframe interval (0 = auto)
    pub keyframe_interval: usize,
}

impl Default for GopConfig {
    fn default() -> Self {
        GopConfig {
            max_size: 250, // ~10 seconds at 25fps
            min_size: 25,  // ~1 second at 25fps
            b_frames: 3,   // 3 B-frames between references
            structure: GopStructure::Open,
            b_pyramid: BFramePyramid::Normal,
            scene_detection: true,
            scene_threshold: 0.3,
            adaptive: true,
            force_idr: false,
            keyframe_interval: 0, // Auto
        }
    }
}

impl GopConfig {
    /// Set maximum GOP size
    pub fn with_max_size(mut self, size: usize) -> Self {
        self.max_size = size;
        self
    }

    /// Set minimum GOP size
    pub fn with_min_size(mut self, size: usize) -> Self {
        self.min_size = size;
        self
    }

    /// Set number of B-frames
    pub fn with_b_frames(mut self, count: usize) -> Self {
        self.b_frames = count;
        self
    }

    /// Enable/disable scene detection
    pub fn with_scene_detection(mut self, enabled: bool) -> Self {
        self.scene_detection = enabled;
        self
    }

    /// Set GOP structure
    pub fn with_structure(mut self, structure: GopStructure) -> Self {
        self.structure = structure;
        self
    }

    /// Set B-frame pyramid
    pub fn with_b_pyramid(mut self, pyramid: BFramePyramid) -> Self {
        self.b_pyramid = pyramid;
        self
    }

    /// Set keyframe interval (0 = auto)
    pub fn with_keyframe_interval(mut self, interval: usize) -> Self {
        self.keyframe_interval = interval;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.min_size > self.max_size {
            return Err(Error::InvalidInput(
                "min_size must be <= max_size".to_string(),
            ));
        }

        if self.max_size == 0 {
            return Err(Error::InvalidInput("max_size must be > 0".to_string()));
        }

        if self.scene_threshold < 0.0 || self.scene_threshold > 1.0 {
            return Err(Error::InvalidInput(
                "scene_threshold must be 0.0-1.0".to_string(),
            ));
        }

        Ok(())
    }
}

/// Frame metadata for GOP decision
#[derive(Debug, Clone)]
pub struct FrameMetadata {
    /// Frame index
    pub index: usize,
    /// Scene change detected
    pub scene_change: bool,
    /// Complexity score (0.0-1.0)
    pub complexity: f64,
    /// Temporal activity (motion)
    pub temporal_activity: f64,
}

/// GOP manager
pub struct GopManager {
    config: GopConfig,
    /// Current position in GOP
    gop_position: usize,
    /// Frames since last keyframe
    frames_since_keyframe: usize,
    /// Frame count
    frame_count: usize,
    /// Previous frame for scene detection
    prev_frame_histogram: Option<Vec<u32>>,
    /// Pending B-frames (for reordering)
    pending_b_frames: VecDeque<usize>,
    /// Last reference frame index
    last_reference: usize,
}

impl GopManager {
    /// Create new GOP manager
    pub fn new(config: GopConfig) -> Result<Self> {
        config.validate()?;

        Ok(GopManager {
            config,
            gop_position: 0,
            frames_since_keyframe: 0,
            frame_count: 0,
            prev_frame_histogram: None,
            pending_b_frames: VecDeque::new(),
            last_reference: 0,
        })
    }

    /// Determine frame type for next frame
    pub fn determine_frame_type(&mut self, frame: Option<&FrameData>) -> Result<GopFrameType> {
        let frame_type = self.determine_frame_type_internal(frame)?;

        // Update state
        if frame_type.is_keyframe() {
            self.frames_since_keyframe = 0;
            self.gop_position = 0;
            self.last_reference = self.frame_count;
        } else if frame_type == GopFrameType::P {
            self.last_reference = self.frame_count;
        }

        self.frame_count += 1;
        self.frames_since_keyframe += 1;
        self.gop_position += 1;

        Ok(frame_type)
    }

    fn determine_frame_type_internal(&mut self, frame: Option<&FrameData>) -> Result<GopFrameType> {
        // First frame is always keyframe
        if self.frame_count == 0 {
            return Ok(if self.config.force_idr {
                GopFrameType::IDR
            } else {
                GopFrameType::I
            });
        }

        // Fixed keyframe interval
        if self.config.keyframe_interval > 0
            && self.frames_since_keyframe >= self.config.keyframe_interval
        {
            return Ok(GopFrameType::I);
        }

        // Scene change detection
        if self.config.scene_detection {
            if let Some(frame_data) = frame {
                if self.is_scene_change(frame_data) {
                    // Only force keyframe if we've met minimum GOP size
                    if self.frames_since_keyframe >= self.config.min_size {
                        return Ok(GopFrameType::I);
                    }
                }
            }
        }

        // Maximum GOP size reached
        if self.frames_since_keyframe >= self.config.max_size {
            return Ok(GopFrameType::I);
        }

        // Determine P or B frame based on position
        // Pattern: I [B B ... B] P [B B ... B] P ...
        // With b_frames=2: I B B P B B P ...
        // gop_position after I is 1, then 2, 3, 4, ...
        // P frames occur at positions that are multiples of (b_frames + 1)
        if self.config.b_frames == 0 {
            // No B-frames, all P
            Ok(GopFrameType::P)
        } else if self.gop_position % (self.config.b_frames + 1) == 0 {
            // Reference frame (P) at every (b_frames + 1) position
            Ok(GopFrameType::P)
        } else {
            // B-frame
            Ok(GopFrameType::B)
        }
    }

    /// Detect scene change using histogram difference
    fn is_scene_change(&mut self, frame: &FrameData) -> bool {
        let histogram = self.calculate_histogram(frame);

        let is_change = if let Some(ref prev) = self.prev_frame_histogram {
            let diff = self.histogram_difference(&histogram, prev);
            diff > self.config.scene_threshold
        } else {
            false
        };

        self.prev_frame_histogram = Some(histogram);

        is_change
    }

    /// Calculate RGB histogram (simplified, 32 bins per channel)
    fn calculate_histogram(&self, frame: &FrameData) -> Vec<u32> {
        const BINS: usize = 32;
        let mut histogram = vec![0u32; BINS * BINS * BINS];

        for rgb in frame.data.chunks_exact(3) {
            let r_bin = ((rgb[0] as usize * BINS) / 256).min(BINS - 1);
            let g_bin = ((rgb[1] as usize * BINS) / 256).min(BINS - 1);
            let b_bin = ((rgb[2] as usize * BINS) / 256).min(BINS - 1);

            let idx = r_bin * BINS * BINS + g_bin * BINS + b_bin;
            histogram[idx] += 1;
        }

        histogram
    }

    /// Calculate histogram difference (chi-square)
    fn histogram_difference(&self, hist1: &[u32], hist2: &[u32]) -> f64 {
        let total: f64 = hist1.iter().map(|&x| x as f64).sum();

        if total == 0.0 {
            return 0.0;
        }

        let mut diff = 0.0;

        for (&h1, &h2) in hist1.iter().zip(hist2.iter()) {
            let h1_norm = h1 as f64 / total;
            let h2_norm = h2 as f64 / total;

            if h1_norm + h2_norm > 0.0 {
                let delta = h1_norm - h2_norm;
                diff += (delta * delta) / (h1_norm + h2_norm);
            }
        }

        (diff / 2.0).min(1.0)
    }

    /// Get recommended display order for frame reordering
    pub fn get_display_order(&self, encode_order: usize) -> usize {
        // For B-frames, display order differs from encode order
        // This is a simplified implementation
        encode_order
    }

    /// Get GOP statistics
    pub fn statistics(&self) -> GopStatistics {
        GopStatistics {
            total_frames: self.frame_count,
            gops_completed: self.frame_count / self.config.max_size,
            current_gop_size: self.frames_since_keyframe,
            avg_gop_size: if self.frame_count > 0 {
                self.frame_count / (self.frame_count / self.config.max_size).max(1)
            } else {
                0
            },
        }
    }

    /// Reset GOP manager
    pub fn reset(&mut self) {
        self.gop_position = 0;
        self.frames_since_keyframe = 0;
        self.frame_count = 0;
        self.prev_frame_histogram = None;
        self.pending_b_frames.clear();
        self.last_reference = 0;
    }

    /// Get current GOP position
    pub fn current_position(&self) -> usize {
        self.gop_position
    }

    /// Get frames since last keyframe
    pub fn frames_since_keyframe(&self) -> usize {
        self.frames_since_keyframe
    }
}

/// GOP statistics
#[derive(Debug, Clone)]
pub struct GopStatistics {
    /// Total frames processed
    pub total_frames: usize,
    /// Number of completed GOPs
    pub gops_completed: usize,
    /// Current GOP size
    pub current_gop_size: usize,
    /// Average GOP size
    pub avg_gop_size: usize,
}

impl GopStatistics {
    pub fn summary(&self) -> String {
        format!(
            "GOP Statistics:\n\
             - Total Frames: {}\n\
             - GOPs Completed: {}\n\
             - Current GOP Size: {}\n\
             - Average GOP Size: {}",
            self.total_frames, self.gops_completed, self.current_gop_size, self.avg_gop_size
        )
    }
}

/// Pre-configured GOP patterns
impl GopConfig {
    /// All-intra (every frame is I-frame)
    pub fn all_intra() -> Self {
        GopConfig {
            max_size: 1,
            min_size: 1,
            b_frames: 0,
            structure: GopStructure::Closed,
            b_pyramid: BFramePyramid::None,
            scene_detection: false,
            scene_threshold: 0.0,
            adaptive: false,
            force_idr: true,
            keyframe_interval: 1,
        }
    }

    /// Low latency (no B-frames)
    pub fn low_latency() -> Self {
        GopConfig {
            max_size: 60,
            min_size: 15,
            b_frames: 0,
            structure: GopStructure::Open,
            b_pyramid: BFramePyramid::None,
            scene_detection: true,
            scene_threshold: 0.3,
            adaptive: true,
            force_idr: false,
            keyframe_interval: 0,
        }
    }

    /// High efficiency (many B-frames)
    pub fn high_efficiency() -> Self {
        GopConfig {
            max_size: 250,
            min_size: 60,
            b_frames: 7,
            structure: GopStructure::Open,
            b_pyramid: BFramePyramid::Normal,
            scene_detection: true,
            scene_threshold: 0.3,
            adaptive: true,
            force_idr: false,
            keyframe_interval: 0,
        }
    }

    /// Broadcast (closed GOP for editing)
    pub fn broadcast() -> Self {
        GopConfig {
            max_size: 50,
            min_size: 25,
            b_frames: 2,
            structure: GopStructure::Closed,
            b_pyramid: BFramePyramid::None,
            scene_detection: false,
            scene_threshold: 0.3,
            adaptive: false,
            force_idr: true,
            keyframe_interval: 50,
        }
    }

    /// Streaming (adaptive with scene detection)
    pub fn streaming() -> Self {
        GopConfig {
            max_size: 120,
            min_size: 30,
            b_frames: 3,
            structure: GopStructure::Open,
            b_pyramid: BFramePyramid::Normal,
            scene_detection: true,
            scene_threshold: 0.3,
            adaptive: true,
            force_idr: false,
            keyframe_interval: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gop_config_default() {
        let config = GopConfig::default();
        assert_eq!(config.max_size, 250);
        assert_eq!(config.b_frames, 3);
        assert!(config.scene_detection);
    }

    #[test]
    fn test_gop_config_validation() {
        let mut config = GopConfig::default();
        assert!(config.validate().is_ok());

        config.min_size = 300;
        config.max_size = 100;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_gop_manager_creation() {
        let config = GopConfig::default();
        let manager = GopManager::new(config);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_first_frame_is_keyframe() {
        let config = GopConfig::default();
        let mut manager = GopManager::new(config).unwrap();

        let frame_type = manager.determine_frame_type(None).unwrap();
        assert!(frame_type.is_keyframe());
    }

    #[test]
    fn test_gop_pattern() {
        let config = GopConfig::default().with_b_frames(2).with_max_size(100);
        let mut manager = GopManager::new(config).unwrap();

        // First frame: I
        let ft = manager.determine_frame_type(None).unwrap();
        assert_eq!(ft, GopFrameType::I);

        // Pattern should be: I B B P B B P ...
        let ft = manager.determine_frame_type(None).unwrap();
        assert_eq!(ft, GopFrameType::B);

        let ft = manager.determine_frame_type(None).unwrap();
        assert_eq!(ft, GopFrameType::B);

        let ft = manager.determine_frame_type(None).unwrap();
        assert_eq!(ft, GopFrameType::P);
    }

    #[test]
    fn test_max_gop_size() {
        let config = GopConfig::default().with_max_size(10).with_min_size(5);
        let mut manager = GopManager::new(config).unwrap();

        // Skip first I-frame
        manager.determine_frame_type(None).unwrap();

        // Process 9 more frames
        for _ in 0..9 {
            manager.determine_frame_type(None).unwrap();
        }

        // 11th frame should be keyframe (max_size reached)
        let ft = manager.determine_frame_type(None).unwrap();
        assert!(ft.is_keyframe());
    }

    #[test]
    fn test_frame_type_properties() {
        assert!(GopFrameType::I.is_keyframe());
        assert!(GopFrameType::IDR.is_keyframe());
        assert!(!GopFrameType::P.is_keyframe());
        assert!(!GopFrameType::B.is_keyframe());

        assert!(GopFrameType::I.is_reference());
        assert!(GopFrameType::P.is_reference());
        assert!(!GopFrameType::B.is_reference());
    }

    #[test]
    fn test_preset_configs() {
        let all_intra = GopConfig::all_intra();
        assert_eq!(all_intra.max_size, 1);
        assert_eq!(all_intra.b_frames, 0);

        let low_latency = GopConfig::low_latency();
        assert_eq!(low_latency.b_frames, 0);

        let high_eff = GopConfig::high_efficiency();
        assert_eq!(high_eff.b_frames, 7);

        let broadcast = GopConfig::broadcast();
        assert_eq!(broadcast.structure, GopStructure::Closed);

        let streaming = GopConfig::streaming();
        assert!(streaming.scene_detection);
    }

    #[test]
    fn test_gop_statistics() {
        let config = GopConfig::default().with_max_size(10).with_min_size(5);
        let mut manager = GopManager::new(config).unwrap();

        for _ in 0..25 {
            manager.determine_frame_type(None).unwrap();
        }

        let stats = manager.statistics();
        assert_eq!(stats.total_frames, 25);
        assert!(stats.gops_completed >= 2);
    }

    #[test]
    fn test_reset() {
        let config = GopConfig::default();
        let mut manager = GopManager::new(config).unwrap();

        for _ in 0..10 {
            manager.determine_frame_type(None).unwrap();
        }

        manager.reset();
        assert_eq!(manager.frame_count, 0);
        assert_eq!(manager.gop_position, 0);
    }

    #[test]
    fn test_keyframe_interval() {
        let config = GopConfig::default()
            .with_keyframe_interval(5)
            .with_max_size(100);
        let mut manager = GopManager::new(config).unwrap();

        // First frame
        manager.determine_frame_type(None).unwrap();

        // Process 4 frames
        for _ in 0..4 {
            manager.determine_frame_type(None).unwrap();
        }

        // 6th frame should be keyframe (interval = 5)
        let ft = manager.determine_frame_type(None).unwrap();
        assert!(ft.is_keyframe());
    }
}
