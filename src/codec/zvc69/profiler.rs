//! ZVC69 Performance Profiling Infrastructure
//!
//! This module provides comprehensive performance profiling tools for the ZVC69
//! neural video codec. It enables detailed timing analysis of encoder/decoder
//! pipeline stages, bottleneck identification, and benchmark utilities.
//!
//! ## Features
//!
//! - **Per-Stage Timing**: Measure duration of each pipeline stage
//! - **Statistical Analysis**: Compute average, min, max, and total times
//! - **FPS Calculation**: Convert timings to frames-per-second metrics
//! - **Frame-Level Analysis**: Track individual frame timings with type info
//! - **Benchmark Utilities**: Standard benchmark configurations and runners
//!
//! ## Usage
//!
//! ```rust,ignore
//! use zvd::codec::zvc69::profiler::{Profiler, stages};
//!
//! let mut profiler = Profiler::new();
//! profiler.set_enabled(true);
//!
//! // Time a stage
//! let result = profiler.time(stages::ENCODER_NETWORK, || {
//!     encoder.run_network(&input)
//! });
//!
//! // Get report
//! println!("{}", profiler.report());
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::config::Quality;

// ─────────────────────────────────────────────────────────────────────────────
// Stage Name Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Stage names for encoder profiling
pub mod stages {
    // ── I-Frame Encoder Stages ──

    /// Convert video frame to tensor [1, 3, H, W]
    pub const FRAME_TO_TENSOR: &str = "frame_to_tensor";

    /// Run encoder neural network (analysis transform)
    pub const ENCODER_NETWORK: &str = "encoder_network";

    /// Run hyperprior encoder (latents -> side info)
    pub const HYPERPRIOR_ENCODE: &str = "hyperprior_encode";

    /// Quantize latent representations
    pub const QUANTIZATION: &str = "quantization";

    /// Entropy encode compressed latents (ANS coding)
    pub const ENTROPY_ENCODE: &str = "entropy_encode";

    /// Total I-frame encoding time
    pub const IFRAME_ENCODE: &str = "iframe_encode";

    // ── P-Frame Encoder Stages ──

    /// Motion estimation (optical flow)
    pub const MOTION_ESTIMATION: &str = "motion_estimation";

    /// Warp reference frame using motion vectors
    pub const FRAME_WARP: &str = "frame_warp";

    /// Encode residual after motion compensation
    pub const RESIDUAL_ENCODE: &str = "residual_encode";

    /// Motion vector entropy encoding
    pub const MOTION_ENCODE: &str = "motion_encode";

    /// Total P-frame encoding time
    pub const PFRAME_ENCODE: &str = "pframe_encode";

    // ── Decoder Stages ──

    /// Entropy decode compressed data
    pub const ENTROPY_DECODE: &str = "entropy_decode";

    /// Run hyperprior decoder (side info -> entropy params)
    pub const HYPERPRIOR_DECODE: &str = "hyperprior_decode";

    /// Run decoder neural network (synthesis transform)
    pub const DECODER_NETWORK: &str = "decoder_network";

    /// Convert tensor back to video frame
    pub const TENSOR_TO_FRAME: &str = "tensor_to_frame";

    /// Decode motion vectors
    pub const MOTION_DECODE: &str = "motion_decode";

    /// Decode residual
    pub const RESIDUAL_DECODE: &str = "residual_decode";

    /// Total I-frame decoding time
    pub const IFRAME_DECODE: &str = "iframe_decode";

    /// Total P-frame decoding time
    pub const PFRAME_DECODE: &str = "pframe_decode";

    // ── General Stages ──

    /// Write data to bitstream
    pub const BITSTREAM_WRITE: &str = "bitstream_write";

    /// Read data from bitstream
    pub const BITSTREAM_READ: &str = "bitstream_read";

    /// Total encode time (all frame types)
    pub const TOTAL_ENCODE: &str = "total_encode";

    /// Total decode time (all frame types)
    pub const TOTAL_DECODE: &str = "total_decode";

    /// Memory transfer (CPU <-> GPU)
    pub const MEMORY_TRANSFER: &str = "memory_transfer";

    /// GPU synchronization
    pub const GPU_SYNC: &str = "gpu_sync";
}

// ─────────────────────────────────────────────────────────────────────────────
// Performance Statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Performance statistics for a single profiling stage
#[derive(Debug, Clone)]
pub struct ProfileStats {
    /// Stage name
    pub stage: String,

    /// Number of samples
    pub count: usize,

    /// Total time in milliseconds
    pub total_ms: f64,

    /// Average time in milliseconds
    pub avg_ms: f64,

    /// Minimum time in milliseconds
    pub min_ms: f64,

    /// Maximum time in milliseconds
    pub max_ms: f64,

    /// Standard deviation in milliseconds
    pub std_dev_ms: f64,

    /// Equivalent FPS (1000.0 / avg_ms)
    pub fps_equivalent: f64,

    /// Percentage of total time (if available)
    pub percent_of_total: f64,
}

impl ProfileStats {
    /// Create stats from a list of durations
    pub fn from_durations(stage: &str, durations: &[Duration]) -> Self {
        if durations.is_empty() {
            return ProfileStats {
                stage: stage.to_string(),
                count: 0,
                total_ms: 0.0,
                avg_ms: 0.0,
                min_ms: 0.0,
                max_ms: 0.0,
                std_dev_ms: 0.0,
                fps_equivalent: 0.0,
                percent_of_total: 0.0,
            };
        }

        let count = durations.len();
        let total_ms: f64 = durations.iter().map(|d| d.as_secs_f64() * 1000.0).sum();
        let avg_ms = total_ms / count as f64;

        let min_ms = durations
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .fold(f64::INFINITY, f64::min);

        let max_ms = durations
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .fold(0.0, f64::max);

        // Calculate standard deviation
        let variance: f64 = durations
            .iter()
            .map(|d| {
                let ms = d.as_secs_f64() * 1000.0;
                (ms - avg_ms).powi(2)
            })
            .sum::<f64>()
            / count as f64;
        let std_dev_ms = variance.sqrt();

        let fps_equivalent = if avg_ms > 0.0 { 1000.0 / avg_ms } else { 0.0 };

        ProfileStats {
            stage: stage.to_string(),
            count,
            total_ms,
            avg_ms,
            min_ms,
            max_ms,
            std_dev_ms,
            fps_equivalent,
            percent_of_total: 0.0,
        }
    }

    /// Set the percentage of total time
    pub fn with_percent(mut self, total_time_ms: f64) -> Self {
        if total_time_ms > 0.0 {
            self.percent_of_total = (self.total_ms / total_time_ms) * 100.0;
        }
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Frame-Level Timing
// ─────────────────────────────────────────────────────────────────────────────

/// Frame type for timing purposes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimingFrameType {
    /// Intra frame (keyframe)
    I = 0,
    /// Predicted frame
    P = 1,
    /// Bidirectional frame
    B = 2,
}

impl From<u8> for TimingFrameType {
    fn from(value: u8) -> Self {
        match value {
            0 => TimingFrameType::I,
            1 => TimingFrameType::P,
            2 => TimingFrameType::B,
            _ => TimingFrameType::I,
        }
    }
}

/// Frame-level timing for detailed analysis
#[derive(Debug, Clone)]
pub struct FrameTiming {
    /// Frame number in sequence
    pub frame_number: u64,

    /// Frame type (I, P, or B)
    pub frame_type: TimingFrameType,

    /// Total time in milliseconds
    pub total_ms: f64,

    /// Per-stage timings in milliseconds
    pub stage_timings: HashMap<String, f64>,

    /// Frame size in bytes (after encoding)
    pub frame_size_bytes: usize,

    /// Bits per pixel
    pub bpp: f64,
}

impl FrameTiming {
    /// Create a new frame timing record
    pub fn new(frame_number: u64, frame_type: TimingFrameType) -> Self {
        FrameTiming {
            frame_number,
            frame_type,
            total_ms: 0.0,
            stage_timings: HashMap::new(),
            frame_size_bytes: 0,
            bpp: 0.0,
        }
    }

    /// Record a stage timing
    pub fn record_stage(&mut self, stage: &str, duration_ms: f64) {
        self.stage_timings.insert(stage.to_string(), duration_ms);
    }

    /// Finalize the frame timing
    pub fn finalize(&mut self, total_ms: f64, frame_size_bytes: usize, width: u32, height: u32) {
        self.total_ms = total_ms;
        self.frame_size_bytes = frame_size_bytes;
        let pixels = (width * height) as f64;
        self.bpp = if pixels > 0.0 {
            (frame_size_bytes * 8) as f64 / pixels
        } else {
            0.0
        };
    }

    /// Get encoding FPS for this frame
    pub fn fps(&self) -> f64 {
        if self.total_ms > 0.0 {
            1000.0 / self.total_ms
        } else {
            0.0
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Profiler
// ─────────────────────────────────────────────────────────────────────────────

/// Performance profiler for ZVC69 codec operations
///
/// The profiler collects timing data for various stages of the encoding
/// and decoding pipelines, enabling bottleneck identification and
/// performance optimization.
#[derive(Debug)]
pub struct Profiler {
    /// Collected timings per stage
    timings: HashMap<String, Vec<Duration>>,

    /// Whether profiling is enabled
    enabled: bool,

    /// Frame-level timings
    frame_timings: Vec<FrameTiming>,

    /// Current frame being profiled
    current_frame: Option<FrameTiming>,

    /// Start time for current frame
    current_frame_start: Option<Instant>,
}

impl Profiler {
    /// Create a new profiler (disabled by default)
    pub fn new() -> Self {
        Profiler {
            timings: HashMap::new(),
            enabled: false,
            frame_timings: Vec::new(),
            current_frame: None,
            current_frame_start: None,
        }
    }

    /// Create a new enabled profiler
    pub fn enabled() -> Self {
        let mut profiler = Self::new();
        profiler.enabled = true;
        profiler
    }

    /// Enable or disable profiling
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if profiling is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Time a closure and record the duration
    ///
    /// Returns the result of the closure. If profiling is disabled,
    /// the closure is still executed but no timing is recorded.
    pub fn time<T, F: FnOnce() -> T>(&mut self, name: &str, f: F) -> T {
        if !self.enabled {
            return f();
        }

        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();

        self.record(name, duration);

        result
    }

    /// Time a closure that returns a Result, recording duration only on success
    pub fn time_result<T, E, F: FnOnce() -> Result<T, E>>(
        &mut self,
        name: &str,
        f: F,
    ) -> Result<T, E> {
        if !self.enabled {
            return f();
        }

        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();

        if result.is_ok() {
            self.record(name, duration);
        }

        result
    }

    /// Record a pre-measured duration
    pub fn record(&mut self, name: &str, duration: Duration) {
        if !self.enabled {
            return;
        }

        self.timings
            .entry(name.to_string())
            .or_default()
            .push(duration);

        // Also record to current frame if active
        if let Some(ref mut frame) = self.current_frame {
            frame.record_stage(name, duration.as_secs_f64() * 1000.0);
        }
    }

    /// Begin timing a new frame
    pub fn begin_frame(&mut self, frame_number: u64, frame_type: TimingFrameType) {
        if !self.enabled {
            return;
        }

        self.current_frame = Some(FrameTiming::new(frame_number, frame_type));
        self.current_frame_start = Some(Instant::now());
    }

    /// End timing the current frame
    pub fn end_frame(&mut self, frame_size_bytes: usize, width: u32, height: u32) {
        if !self.enabled {
            return;
        }

        if let (Some(mut frame), Some(start)) =
            (self.current_frame.take(), self.current_frame_start.take())
        {
            let total_ms = start.elapsed().as_secs_f64() * 1000.0;
            frame.finalize(total_ms, frame_size_bytes, width, height);
            self.frame_timings.push(frame);
        }
    }

    /// Get average duration for a stage
    pub fn average(&self, name: &str) -> Option<Duration> {
        self.timings.get(name).map(|durations| {
            if durations.is_empty() {
                Duration::ZERO
            } else {
                let total: Duration = durations.iter().sum();
                total / durations.len() as u32
            }
        })
    }

    /// Get average duration in milliseconds
    pub fn average_ms(&self, name: &str) -> Option<f64> {
        self.average(name).map(|d| d.as_secs_f64() * 1000.0)
    }

    /// Get total time for a stage
    pub fn total(&self, name: &str) -> Option<Duration> {
        self.timings
            .get(name)
            .map(|durations| durations.iter().sum())
    }

    /// Get total time in milliseconds
    pub fn total_ms(&self, name: &str) -> Option<f64> {
        self.total(name).map(|d| d.as_secs_f64() * 1000.0)
    }

    /// Get the number of samples for a stage
    pub fn count(&self, name: &str) -> usize {
        self.timings.get(name).map_or(0, |d| d.len())
    }

    /// Get statistics for a specific stage
    pub fn stats(&self, name: &str) -> Option<ProfileStats> {
        self.timings
            .get(name)
            .map(|durations| ProfileStats::from_durations(name, durations))
    }

    /// Get all stage names
    pub fn stage_names(&self) -> Vec<&str> {
        self.timings.keys().map(|s| s.as_str()).collect()
    }

    /// Get all statistics sorted by total time (descending)
    pub fn all_stats(&self) -> Vec<ProfileStats> {
        // Calculate total time for percentage calculation
        let total_time_ms: f64 = self
            .timings
            .values()
            .flat_map(|durations| durations.iter())
            .map(|d| d.as_secs_f64() * 1000.0)
            .sum();

        let mut stats: Vec<ProfileStats> = self
            .timings
            .iter()
            .map(|(name, durations)| {
                ProfileStats::from_durations(name, durations).with_percent(total_time_ms)
            })
            .collect();

        // Sort by total time descending
        stats.sort_by(|a, b| {
            b.total_ms
                .partial_cmp(&a.total_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        stats
    }

    /// Get frame-level timings
    pub fn frame_timings(&self) -> &[FrameTiming] {
        &self.frame_timings
    }

    /// Get timing report as formatted string
    pub fn report(&self) -> String {
        let stats = self.all_stats();

        if stats.is_empty() {
            return "No profiling data collected.".to_string();
        }

        let mut report = String::new();
        report.push_str("=== ZVC69 Performance Report ===\n\n");

        // Header
        report.push_str(&format!(
            "{:<25} {:>8} {:>10} {:>10} {:>10} {:>10} {:>8}\n",
            "Stage", "Count", "Total(ms)", "Avg(ms)", "Min(ms)", "Max(ms)", "FPS"
        ));
        report.push_str(&"-".repeat(91));
        report.push('\n');

        // Stage rows
        for stat in &stats {
            report.push_str(&format!(
                "{:<25} {:>8} {:>10.2} {:>10.3} {:>10.3} {:>10.3} {:>8.1}\n",
                stat.stage,
                stat.count,
                stat.total_ms,
                stat.avg_ms,
                stat.min_ms,
                stat.max_ms,
                stat.fps_equivalent
            ));
        }

        // Summary
        if !self.frame_timings.is_empty() {
            report.push('\n');
            report.push_str("=== Frame Statistics ===\n\n");

            let i_frames: Vec<&FrameTiming> = self
                .frame_timings
                .iter()
                .filter(|f| f.frame_type == TimingFrameType::I)
                .collect();
            let p_frames: Vec<&FrameTiming> = self
                .frame_timings
                .iter()
                .filter(|f| f.frame_type == TimingFrameType::P)
                .collect();

            if !i_frames.is_empty() {
                let avg_i =
                    i_frames.iter().map(|f| f.total_ms).sum::<f64>() / i_frames.len() as f64;
                let fps_i = if avg_i > 0.0 { 1000.0 / avg_i } else { 0.0 };
                report.push_str(&format!(
                    "I-frames: {} frames, avg {:.2}ms ({:.1} fps)\n",
                    i_frames.len(),
                    avg_i,
                    fps_i
                ));
            }

            if !p_frames.is_empty() {
                let avg_p =
                    p_frames.iter().map(|f| f.total_ms).sum::<f64>() / p_frames.len() as f64;
                let fps_p = if avg_p > 0.0 { 1000.0 / avg_p } else { 0.0 };
                report.push_str(&format!(
                    "P-frames: {} frames, avg {:.2}ms ({:.1} fps)\n",
                    p_frames.len(),
                    avg_p,
                    fps_p
                ));
            }

            // Overall average
            let total_frames = self.frame_timings.len();
            let total_time: f64 = self.frame_timings.iter().map(|f| f.total_ms).sum();
            let avg_all = total_time / total_frames as f64;
            let fps_all = if avg_all > 0.0 { 1000.0 / avg_all } else { 0.0 };
            report.push_str(&format!(
                "Overall:  {} frames, avg {:.2}ms ({:.1} fps)\n",
                total_frames, avg_all, fps_all
            ));
        }

        report
    }

    /// Reset all timings
    pub fn reset(&mut self) {
        self.timings.clear();
        self.frame_timings.clear();
        self.current_frame = None;
        self.current_frame_start = None;
    }

    /// Get top N bottlenecks by average time
    pub fn top_bottlenecks(&self, n: usize) -> Vec<ProfileStats> {
        let mut stats = self.all_stats();
        stats.sort_by(|a, b| {
            b.avg_ms
                .partial_cmp(&a.avg_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        stats.truncate(n);
        stats
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Profiler {
    fn clone(&self) -> Self {
        Profiler {
            timings: self.timings.clone(),
            enabled: self.enabled,
            frame_timings: self.frame_timings.clone(),
            current_frame: None, // Don't clone in-progress frame
            current_frame_start: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Profiling Macro
// ─────────────────────────────────────────────────────────────────────────────

/// Macro for easy profiling with optional profiler
///
/// Usage:
/// ```rust,ignore
/// use zvd::codec::zvc69::profile;
///
/// let result = profile!(profiler, "stage_name", {
///     expensive_operation()
/// });
/// ```
#[macro_export]
macro_rules! profile {
    ($profiler:expr, $name:expr, $code:block) => {
        if let Some(ref mut p) = $profiler {
            p.time($name, || $code)
        } else {
            $code
        }
    };
}

// Re-export the macro at module level
pub use crate::profile;

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Frame width
    pub width: u32,

    /// Frame height
    pub height: u32,

    /// Number of frames to encode
    pub num_frames: usize,

    /// Number of warmup frames (not counted in results)
    pub warmup_frames: usize,

    /// Quality level
    pub quality: Quality,

    /// Include I-frames in timing
    pub include_iframes: bool,

    /// Include P-frames in timing
    pub include_pframes: bool,

    /// GOP size (keyframe interval)
    pub gop_size: u32,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        BenchmarkConfig {
            width: 1920,
            height: 1088, // Aligned to 16
            num_frames: 100,
            warmup_frames: 10,
            quality: Quality::Q4,
            include_iframes: true,
            include_pframes: true,
            gop_size: 30,
        }
    }
}

impl BenchmarkConfig {
    /// Create a 720p benchmark preset
    pub fn preset_720p() -> Self {
        BenchmarkConfig {
            width: 1280,
            height: 720,
            num_frames: 150,
            warmup_frames: 10,
            quality: Quality::Q4,
            include_iframes: true,
            include_pframes: true,
            gop_size: 30,
        }
    }

    /// Create a 1080p benchmark preset
    pub fn preset_1080p() -> Self {
        BenchmarkConfig {
            width: 1920,
            height: 1088,
            num_frames: 100,
            warmup_frames: 10,
            quality: Quality::Q4,
            include_iframes: true,
            include_pframes: true,
            gop_size: 30,
        }
    }

    /// Create a 4K benchmark preset
    pub fn preset_4k() -> Self {
        BenchmarkConfig {
            width: 3840,
            height: 2160,
            num_frames: 50,
            warmup_frames: 5,
            quality: Quality::Q4,
            include_iframes: true,
            include_pframes: true,
            gop_size: 30,
        }
    }

    /// Set the quality level
    pub fn with_quality(mut self, quality: Quality) -> Self {
        self.quality = quality;
        self
    }

    /// Set the number of frames
    pub fn with_num_frames(mut self, num_frames: usize) -> Self {
        self.num_frames = num_frames;
        self
    }

    /// Set warmup frames
    pub fn with_warmup(mut self, warmup_frames: usize) -> Self {
        self.warmup_frames = warmup_frames;
        self
    }

    /// Set GOP size
    pub fn with_gop_size(mut self, gop_size: u32) -> Self {
        self.gop_size = gop_size;
        self
    }

    /// Calculate total pixels per frame
    pub fn pixels_per_frame(&self) -> u64 {
        (self.width * self.height) as u64
    }

    /// Calculate expected I-frame count
    pub fn expected_iframes(&self) -> usize {
        if self.gop_size == 0 {
            self.num_frames
        } else {
            (self.num_frames + self.gop_size as usize - 1) / self.gop_size as usize
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark Results
// ─────────────────────────────────────────────────────────────────────────────

/// Results from a benchmark run
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Configuration used
    pub config: BenchmarkConfig,

    /// Frame timings (excluding warmup)
    pub frame_timings: Vec<FrameTiming>,

    /// Total benchmark duration
    pub total_duration: Duration,

    /// Average encode time per frame
    pub avg_encode_ms: f64,

    /// Average decode time per frame
    pub avg_decode_ms: f64,

    /// Encode FPS
    pub encode_fps: f64,

    /// Decode FPS
    pub decode_fps: f64,

    /// I-frame average encode time
    pub iframe_avg_ms: f64,

    /// P-frame average encode time
    pub pframe_avg_ms: f64,

    /// Average bits per pixel
    pub avg_bpp: f64,

    /// Total bytes encoded
    pub total_bytes: usize,
}

impl BenchmarkResults {
    /// Create results from frame timings
    pub fn from_timings(
        config: BenchmarkConfig,
        frame_timings: Vec<FrameTiming>,
        total_duration: Duration,
    ) -> Self {
        let encode_times: Vec<f64> = frame_timings.iter().map(|f| f.total_ms).collect();
        let avg_encode_ms = if encode_times.is_empty() {
            0.0
        } else {
            encode_times.iter().sum::<f64>() / encode_times.len() as f64
        };

        let i_times: Vec<f64> = frame_timings
            .iter()
            .filter(|f| f.frame_type == TimingFrameType::I)
            .map(|f| f.total_ms)
            .collect();
        let iframe_avg_ms = if i_times.is_empty() {
            0.0
        } else {
            i_times.iter().sum::<f64>() / i_times.len() as f64
        };

        let p_times: Vec<f64> = frame_timings
            .iter()
            .filter(|f| f.frame_type == TimingFrameType::P)
            .map(|f| f.total_ms)
            .collect();
        let pframe_avg_ms = if p_times.is_empty() {
            0.0
        } else {
            p_times.iter().sum::<f64>() / p_times.len() as f64
        };

        let total_bytes: usize = frame_timings.iter().map(|f| f.frame_size_bytes).sum();
        let avg_bpp = if frame_timings.is_empty() {
            0.0
        } else {
            frame_timings.iter().map(|f| f.bpp).sum::<f64>() / frame_timings.len() as f64
        };

        let encode_fps = if avg_encode_ms > 0.0 {
            1000.0 / avg_encode_ms
        } else {
            0.0
        };

        BenchmarkResults {
            config,
            frame_timings,
            total_duration,
            avg_encode_ms,
            avg_decode_ms: 0.0, // Filled by decode benchmark
            encode_fps,
            decode_fps: 0.0, // Filled by decode benchmark
            iframe_avg_ms,
            pframe_avg_ms,
            avg_bpp,
            total_bytes,
        }
    }

    /// Generate a summary report
    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== ZVC69 Benchmark Results ===\n\n");

        report.push_str(&format!(
            "Resolution: {}x{}\n",
            self.config.width, self.config.height
        ));
        report.push_str(&format!("Quality: {}\n", self.config.quality));
        report.push_str(&format!("Frames: {}\n", self.frame_timings.len()));
        report.push_str(&format!(
            "Total Duration: {:.2}s\n\n",
            self.total_duration.as_secs_f64()
        ));

        report.push_str("Encode Performance:\n");
        report.push_str(&format!("  Average: {:.2}ms/frame\n", self.avg_encode_ms));
        report.push_str(&format!("  FPS: {:.1}\n", self.encode_fps));
        report.push_str(&format!("  I-frame avg: {:.2}ms\n", self.iframe_avg_ms));
        report.push_str(&format!("  P-frame avg: {:.2}ms\n", self.pframe_avg_ms));

        if self.avg_decode_ms > 0.0 {
            report.push_str("\nDecode Performance:\n");
            report.push_str(&format!("  Average: {:.2}ms/frame\n", self.avg_decode_ms));
            report.push_str(&format!("  FPS: {:.1}\n", self.decode_fps));
        }

        report.push_str("\nBitrate:\n");
        report.push_str(&format!("  Average BPP: {:.4}\n", self.avg_bpp));
        report.push_str(&format!("  Total Bytes: {}\n", self.total_bytes));

        let bitrate = if self.frame_timings.is_empty() {
            0.0
        } else {
            (self.total_bytes as f64 * 8.0) / self.total_duration.as_secs_f64()
        };
        report.push_str(&format!(
            "  Bitrate: {:.0} bps ({:.2} Mbps)\n",
            bitrate,
            bitrate / 1_000_000.0
        ));

        report
    }
}

/// Generate a summary report comparing encode and decode benchmarks
pub fn benchmark_report(encode_timings: &[FrameTiming], decode_timings: &[FrameTiming]) -> String {
    let mut report = String::new();
    report.push_str("=== ZVC69 Encode/Decode Benchmark ===\n\n");

    // Encode stats
    if !encode_timings.is_empty() {
        let avg_encode: f64 =
            encode_timings.iter().map(|f| f.total_ms).sum::<f64>() / encode_timings.len() as f64;
        let fps_encode = if avg_encode > 0.0 {
            1000.0 / avg_encode
        } else {
            0.0
        };

        report.push_str("Encode:\n");
        report.push_str(&format!("  Frames: {}\n", encode_timings.len()));
        report.push_str(&format!("  Average: {:.2}ms\n", avg_encode));
        report.push_str(&format!("  FPS: {:.1}\n\n", fps_encode));
    }

    // Decode stats
    if !decode_timings.is_empty() {
        let avg_decode: f64 =
            decode_timings.iter().map(|f| f.total_ms).sum::<f64>() / decode_timings.len() as f64;
        let fps_decode = if avg_decode > 0.0 {
            1000.0 / avg_decode
        } else {
            0.0
        };

        report.push_str("Decode:\n");
        report.push_str(&format!("  Frames: {}\n", decode_timings.len()));
        report.push_str(&format!("  Average: {:.2}ms\n", avg_decode));
        report.push_str(&format!("  FPS: {:.1}\n", fps_decode));
    }

    report
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_profiler_creation() {
        let profiler = Profiler::new();
        assert!(!profiler.is_enabled());

        let profiler = Profiler::enabled();
        assert!(profiler.is_enabled());
    }

    #[test]
    fn test_profiler_enable_disable() {
        let mut profiler = Profiler::new();
        assert!(!profiler.is_enabled());

        profiler.set_enabled(true);
        assert!(profiler.is_enabled());

        profiler.set_enabled(false);
        assert!(!profiler.is_enabled());
    }

    #[test]
    fn test_profiler_timing_accuracy() {
        let mut profiler = Profiler::enabled();

        // Time a 10ms sleep
        profiler.time("test_sleep", || {
            thread::sleep(Duration::from_millis(10));
        });

        let avg = profiler.average_ms("test_sleep").unwrap();
        // Allow some tolerance (10-50ms)
        assert!(avg >= 9.0, "avg {} too low", avg);
        assert!(avg < 50.0, "avg {} too high", avg);
    }

    #[test]
    fn test_profiler_disabled_still_runs() {
        let mut profiler = Profiler::new(); // Disabled by default

        let result = profiler.time("test", || 42);

        assert_eq!(result, 42);
        assert_eq!(profiler.count("test"), 0); // No timing recorded
    }

    #[test]
    fn test_profiler_record_duration() {
        let mut profiler = Profiler::enabled();

        profiler.record("manual", Duration::from_millis(50));
        profiler.record("manual", Duration::from_millis(100));

        assert_eq!(profiler.count("manual"), 2);
        let avg = profiler.average_ms("manual").unwrap();
        assert!((avg - 75.0).abs() < 0.1);
    }

    #[test]
    fn test_profiler_stats() {
        let mut profiler = Profiler::enabled();

        profiler.record("stage1", Duration::from_millis(10));
        profiler.record("stage1", Duration::from_millis(20));
        profiler.record("stage1", Duration::from_millis(30));

        let stats = profiler.stats("stage1").unwrap();
        assert_eq!(stats.count, 3);
        assert!((stats.total_ms - 60.0).abs() < 0.1);
        assert!((stats.avg_ms - 20.0).abs() < 0.1);
        assert!((stats.min_ms - 10.0).abs() < 0.1);
        assert!((stats.max_ms - 30.0).abs() < 0.1);
    }

    #[test]
    fn test_profiler_fps_calculation() {
        let mut profiler = Profiler::enabled();

        // 16.67ms per frame = 60 FPS
        profiler.record("encode", Duration::from_micros(16667));

        let stats = profiler.stats("encode").unwrap();
        assert!((stats.fps_equivalent - 60.0).abs() < 1.0);
    }

    #[test]
    fn test_profiler_reset() {
        let mut profiler = Profiler::enabled();

        profiler.record("test", Duration::from_millis(10));
        assert_eq!(profiler.count("test"), 1);

        profiler.reset();
        assert_eq!(profiler.count("test"), 0);
        assert!(profiler.stats("test").is_none());
    }

    #[test]
    fn test_profiler_report_generation() {
        let mut profiler = Profiler::enabled();

        profiler.record(stages::ENCODER_NETWORK, Duration::from_millis(10));
        profiler.record(stages::QUANTIZATION, Duration::from_millis(2));
        profiler.record(stages::ENTROPY_ENCODE, Duration::from_millis(5));

        let report = profiler.report();
        assert!(report.contains("encoder_network"));
        assert!(report.contains("quantization"));
        assert!(report.contains("entropy_encode"));
    }

    #[test]
    fn test_profiler_all_stats_sorted() {
        let mut profiler = Profiler::enabled();

        profiler.record("slow", Duration::from_millis(100));
        profiler.record("fast", Duration::from_millis(10));
        profiler.record("medium", Duration::from_millis(50));

        let stats = profiler.all_stats();
        assert_eq!(stats.len(), 3);
        // Should be sorted by total time descending
        assert_eq!(stats[0].stage, "slow");
        assert_eq!(stats[1].stage, "medium");
        assert_eq!(stats[2].stage, "fast");
    }

    #[test]
    fn test_profiler_top_bottlenecks() {
        let mut profiler = Profiler::enabled();

        profiler.record("a", Duration::from_millis(10));
        profiler.record("b", Duration::from_millis(50));
        profiler.record("c", Duration::from_millis(30));
        profiler.record("d", Duration::from_millis(20));

        let top2 = profiler.top_bottlenecks(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].stage, "b");
        assert_eq!(top2[1].stage, "c");
    }

    #[test]
    fn test_frame_timing() {
        let mut timing = FrameTiming::new(0, TimingFrameType::I);
        timing.record_stage(stages::ENCODER_NETWORK, 10.0);
        timing.record_stage(stages::QUANTIZATION, 2.0);
        timing.finalize(15.0, 50000, 1920, 1080);

        assert_eq!(timing.frame_number, 0);
        assert_eq!(timing.frame_type, TimingFrameType::I);
        assert_eq!(timing.total_ms, 15.0);
        assert_eq!(timing.frame_size_bytes, 50000);
        assert!(timing.bpp > 0.0);
        assert!(timing.fps() > 0.0);
    }

    #[test]
    fn test_profiler_frame_timing_integration() {
        let mut profiler = Profiler::enabled();

        profiler.begin_frame(0, TimingFrameType::I);
        profiler.record(stages::ENCODER_NETWORK, Duration::from_millis(10));
        profiler.end_frame(50000, 1920, 1080);

        let timings = profiler.frame_timings();
        assert_eq!(timings.len(), 1);
        assert_eq!(timings[0].frame_number, 0);
        assert!(timings[0]
            .stage_timings
            .contains_key(stages::ENCODER_NETWORK));
    }

    #[test]
    fn test_benchmark_config_presets() {
        let config_720p = BenchmarkConfig::preset_720p();
        assert_eq!(config_720p.width, 1280);
        assert_eq!(config_720p.height, 720);

        let config_1080p = BenchmarkConfig::preset_1080p();
        assert_eq!(config_1080p.width, 1920);
        assert_eq!(config_1080p.height, 1088);

        let config_4k = BenchmarkConfig::preset_4k();
        assert_eq!(config_4k.width, 3840);
        assert_eq!(config_4k.height, 2160);
    }

    #[test]
    fn test_benchmark_config_builder() {
        let config = BenchmarkConfig::default()
            .with_quality(Quality::Q6)
            .with_num_frames(50)
            .with_warmup(5)
            .with_gop_size(10);

        assert_eq!(config.quality, Quality::Q6);
        assert_eq!(config.num_frames, 50);
        assert_eq!(config.warmup_frames, 5);
        assert_eq!(config.gop_size, 10);
    }

    #[test]
    fn test_benchmark_results() {
        let config = BenchmarkConfig::preset_720p();

        let frame_timings = vec![
            {
                let mut t = FrameTiming::new(0, TimingFrameType::I);
                t.finalize(20.0, 100000, 1280, 720);
                t
            },
            {
                let mut t = FrameTiming::new(1, TimingFrameType::P);
                t.finalize(10.0, 30000, 1280, 720);
                t
            },
        ];

        let results =
            BenchmarkResults::from_timings(config, frame_timings, Duration::from_millis(30));

        assert_eq!(results.frame_timings.len(), 2);
        assert!((results.avg_encode_ms - 15.0).abs() < 0.1);
        assert!((results.iframe_avg_ms - 20.0).abs() < 0.1);
        assert!((results.pframe_avg_ms - 10.0).abs() < 0.1);
        assert_eq!(results.total_bytes, 130000);
    }

    #[test]
    fn test_benchmark_report_generation() {
        let encode_timings = vec![{
            let mut t = FrameTiming::new(0, TimingFrameType::I);
            t.finalize(20.0, 100000, 1920, 1080);
            t
        }];
        let decode_timings = vec![{
            let mut t = FrameTiming::new(0, TimingFrameType::I);
            t.finalize(10.0, 0, 1920, 1080);
            t
        }];

        let report = benchmark_report(&encode_timings, &decode_timings);
        assert!(report.contains("Encode:"));
        assert!(report.contains("Decode:"));
        assert!(report.contains("FPS:"));
    }

    #[test]
    fn test_stage_constants() {
        // Verify stage names are defined
        assert_eq!(stages::FRAME_TO_TENSOR, "frame_to_tensor");
        assert_eq!(stages::ENCODER_NETWORK, "encoder_network");
        assert_eq!(stages::MOTION_ESTIMATION, "motion_estimation");
        assert_eq!(stages::TOTAL_ENCODE, "total_encode");
        assert_eq!(stages::TOTAL_DECODE, "total_decode");
    }

    #[test]
    fn test_profile_stats_empty() {
        let stats = ProfileStats::from_durations("empty", &[]);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.avg_ms, 0.0);
        assert_eq!(stats.fps_equivalent, 0.0);
    }

    #[test]
    fn test_profile_stats_with_percent() {
        let durations = vec![Duration::from_millis(25)];
        let stats = ProfileStats::from_durations("test", &durations).with_percent(100.0);

        assert!((stats.percent_of_total - 25.0).abs() < 0.1);
    }

    #[test]
    fn test_timing_frame_type_from_u8() {
        assert_eq!(TimingFrameType::from(0), TimingFrameType::I);
        assert_eq!(TimingFrameType::from(1), TimingFrameType::P);
        assert_eq!(TimingFrameType::from(2), TimingFrameType::B);
        assert_eq!(TimingFrameType::from(99), TimingFrameType::I); // Default
    }

    #[test]
    fn test_profiler_clone() {
        let mut profiler = Profiler::enabled();
        profiler.record("test", Duration::from_millis(10));

        let cloned = profiler.clone();
        assert!(cloned.is_enabled());
        assert_eq!(cloned.count("test"), 1);
    }

    #[test]
    fn test_time_result_success() {
        let mut profiler = Profiler::enabled();

        let result: Result<i32, &str> = profiler.time_result("success", || Ok(42));

        assert_eq!(result.unwrap(), 42);
        assert_eq!(profiler.count("success"), 1);
    }

    #[test]
    fn test_time_result_error() {
        let mut profiler = Profiler::enabled();

        let result: Result<i32, &str> = profiler.time_result("error", || Err("fail"));

        assert!(result.is_err());
        assert_eq!(profiler.count("error"), 0); // Not recorded on error
    }

    #[test]
    fn test_benchmark_pixels_per_frame() {
        let config = BenchmarkConfig::preset_1080p();
        assert_eq!(config.pixels_per_frame(), 1920 * 1088);
    }

    #[test]
    fn test_benchmark_expected_iframes() {
        let config = BenchmarkConfig::default()
            .with_num_frames(100)
            .with_gop_size(30);

        assert_eq!(config.expected_iframes(), 4); // Frames 0, 30, 60, 90
    }

    #[test]
    fn test_std_dev_calculation() {
        let mut profiler = Profiler::enabled();

        // Record identical durations - std dev should be 0
        profiler.record("same", Duration::from_millis(10));
        profiler.record("same", Duration::from_millis(10));
        profiler.record("same", Duration::from_millis(10));

        let stats = profiler.stats("same").unwrap();
        assert!(stats.std_dev_ms.abs() < 0.001);
    }
}
