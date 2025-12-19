//! ZVC69 Benchmark Suite for Performance Validation
//!
//! This module provides comprehensive benchmarking tools to validate that
//! ZVC69 meets real-time performance requirements:
//!
//! - **720p Encode**: 30+ fps (target: <33ms per frame)
//! - **720p Decode**: 60+ fps (target: <16ms per frame)
//! - **1080p Encode**: 24+ fps (target: <42ms per frame)
//! - **1080p Decode**: 30+ fps (target: <33ms per frame)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use zvd::codec::zvc69::benchmark::{benchmark_720p, run_benchmark_suite};
//!
//! // Run 720p benchmark
//! let result = benchmark_720p(100);
//! println!("{}", result.to_table());
//!
//! // Run full benchmark suite
//! let results = run_benchmark_suite();
//! for result in results {
//!     println!("{}", result.to_table());
//! }
//! ```
//!
//! ## Performance Targets
//!
//! | Resolution | Encode Target | Decode Target | Latency |
//! |------------|---------------|---------------|---------|
//! | 720p       | 30 fps        | 60 fps        | <33ms   |
//! | 1080p      | 24 fps        | 30 fps        | <42ms   |
//! | 4K         | 8 fps         | 15 fps        | <125ms  |

use std::time::Instant;

use super::config::{Quality, ZVC69Config};
use super::pipeline::{SyncDecoder, SyncEncoder};
use crate::codec::VideoFrame;
use crate::util::{Buffer, PixelFormat, Timestamp};

// ============================================================================
// Benchmark Result
// ============================================================================

/// Results from a benchmark run
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Resolution (width, height)
    pub resolution: (u32, u32),

    /// Number of frames benchmarked
    pub num_frames: usize,

    /// Encode frames per second
    pub encode_fps: f64,

    /// Decode frames per second
    pub decode_fps: f64,

    /// Average encode time per frame (milliseconds)
    pub avg_encode_ms: f64,

    /// Average decode time per frame (milliseconds)
    pub avg_decode_ms: f64,

    /// P50 (median) encode time (milliseconds)
    pub p50_encode_ms: f64,

    /// P50 (median) decode time (milliseconds)
    pub p50_decode_ms: f64,

    /// P95 encode time (milliseconds)
    pub p95_encode_ms: f64,

    /// P95 decode time (milliseconds)
    pub p95_decode_ms: f64,

    /// P99 encode time (milliseconds)
    pub p99_encode_ms: f64,

    /// P99 decode time (milliseconds)
    pub p99_decode_ms: f64,

    /// Minimum encode time (milliseconds)
    pub min_encode_ms: f64,

    /// Maximum encode time (milliseconds)
    pub max_encode_ms: f64,

    /// Minimum decode time (milliseconds)
    pub min_decode_ms: f64,

    /// Maximum decode time (milliseconds)
    pub max_decode_ms: f64,

    /// Average bitrate in kilobits per second
    pub bitrate_kbps: f64,

    /// Average bits per pixel
    pub bpp: f64,

    /// Peak memory usage in megabytes
    pub memory_mb: f64,

    /// Total bytes produced
    pub total_bytes: usize,

    /// Quality level used
    pub quality: Quality,

    /// Whether warmup frames were used
    pub warmed_up: bool,
}

impl BenchmarkResult {
    /// Create a new empty result for the given resolution
    pub fn new(width: u32, height: u32) -> Self {
        BenchmarkResult {
            resolution: (width, height),
            num_frames: 0,
            encode_fps: 0.0,
            decode_fps: 0.0,
            avg_encode_ms: 0.0,
            avg_decode_ms: 0.0,
            p50_encode_ms: 0.0,
            p50_decode_ms: 0.0,
            p95_encode_ms: 0.0,
            p95_decode_ms: 0.0,
            p99_encode_ms: 0.0,
            p99_decode_ms: 0.0,
            min_encode_ms: f64::INFINITY,
            max_encode_ms: 0.0,
            min_decode_ms: f64::INFINITY,
            max_decode_ms: 0.0,
            bitrate_kbps: 0.0,
            bpp: 0.0,
            memory_mb: 0.0,
            total_bytes: 0,
            quality: Quality::Q4,
            warmed_up: false,
        }
    }

    /// Check if the benchmark meets real-time requirements
    ///
    /// # Arguments
    ///
    /// * `target_fps` - Target frames per second
    ///
    /// # Returns
    ///
    /// `true` if both encode and decode exceed the target fps
    pub fn is_realtime(&self, target_fps: f64) -> bool {
        self.encode_fps >= target_fps && self.decode_fps >= target_fps
    }

    /// Check if encode meets target
    pub fn encode_meets_target(&self, target_fps: f64) -> bool {
        self.encode_fps >= target_fps
    }

    /// Check if decode meets target
    pub fn decode_meets_target(&self, target_fps: f64) -> bool {
        self.decode_fps >= target_fps
    }

    /// Check if latency meets target
    pub fn latency_meets_target(&self, target_encode_ms: f64, target_decode_ms: f64) -> bool {
        self.avg_encode_ms <= target_encode_ms && self.avg_decode_ms <= target_decode_ms
    }

    /// Format as a human-readable table
    pub fn to_table(&self) -> String {
        let (w, h) = self.resolution;
        let resolution_name = match (w, h) {
            (1280, 720) => "720p",
            (1920, 1080) | (1920, 1088) => "1080p",
            (3840, 2160) => "4K",
            _ => "Custom",
        };

        format!(
            r#"
+==================================================+
| ZVC69 Benchmark Results - {} ({} x {})
+==================================================+
| Frames:        {:>8} frames
| Quality:       {:>8}
| Warmup:        {:>8}
+--------------------------------------------------+
|                  ENCODE           DECODE
+--------------------------------------------------+
| FPS:           {:>8.1} fps     {:>8.1} fps
| Avg Latency:   {:>8.2} ms      {:>8.2} ms
| P50 Latency:   {:>8.2} ms      {:>8.2} ms
| P95 Latency:   {:>8.2} ms      {:>8.2} ms
| P99 Latency:   {:>8.2} ms      {:>8.2} ms
| Min:           {:>8.2} ms      {:>8.2} ms
| Max:           {:>8.2} ms      {:>8.2} ms
+--------------------------------------------------+
| Bitrate:       {:>8.2} kbps
| BPP:           {:>8.4}
| Total Size:    {:>8.2} MB
| Memory:        {:>8.2} MB
+==================================================+"#,
            resolution_name,
            w,
            h,
            self.num_frames,
            self.quality,
            if self.warmed_up { "Yes" } else { "No" },
            self.encode_fps,
            self.decode_fps,
            self.avg_encode_ms,
            self.avg_decode_ms,
            self.p50_encode_ms,
            self.p50_decode_ms,
            self.p95_encode_ms,
            self.p95_decode_ms,
            self.p99_encode_ms,
            self.p99_decode_ms,
            self.min_encode_ms,
            self.min_decode_ms,
            self.max_encode_ms,
            self.max_decode_ms,
            self.bitrate_kbps,
            self.bpp,
            self.total_bytes as f64 / 1_000_000.0,
            self.memory_mb
        )
    }

    /// Format as CSV row
    pub fn to_csv(&self) -> String {
        format!(
            "{},{},{},{},{:.1},{:.1},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
            self.resolution.0,
            self.resolution.1,
            self.num_frames,
            self.quality,
            self.encode_fps,
            self.decode_fps,
            self.avg_encode_ms,
            self.avg_decode_ms,
            self.p99_encode_ms,
            self.p99_decode_ms,
            self.bitrate_kbps,
            self.bpp,
            self.memory_mb
        )
    }

    /// Get CSV header
    pub fn csv_header() -> &'static str {
        "width,height,frames,quality,encode_fps,decode_fps,avg_encode_ms,avg_decode_ms,p99_encode_ms,p99_decode_ms,bitrate_kbps,bpp,memory_mb"
    }
}

// ============================================================================
// Benchmark Configuration
// ============================================================================

/// Configuration for benchmark runs
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of frames to benchmark
    pub num_frames: usize,

    /// Number of warmup frames (not counted)
    pub warmup_frames: usize,

    /// Frame width
    pub width: u32,

    /// Frame height
    pub height: u32,

    /// Quality level
    pub quality: Quality,

    /// Target framerate (for bitrate calculation)
    pub framerate: f64,

    /// Whether to benchmark decoding
    pub benchmark_decode: bool,

    /// Whether to use synthetic test patterns
    pub use_synthetic: bool,

    /// Pattern type for synthetic frames
    pub pattern: TestPattern,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        BenchmarkConfig {
            num_frames: 100,
            warmup_frames: 10,
            width: 1280,
            height: 720,
            quality: Quality::Q4,
            framerate: 30.0,
            benchmark_decode: true,
            use_synthetic: true,
            pattern: TestPattern::Gradient,
        }
    }
}

impl BenchmarkConfig {
    /// Create a 720p benchmark configuration
    pub fn preset_720p() -> Self {
        BenchmarkConfig {
            width: 1280,
            height: 720,
            num_frames: 150,
            warmup_frames: 10,
            ..Default::default()
        }
    }

    /// Create a 1080p benchmark configuration
    pub fn preset_1080p() -> Self {
        BenchmarkConfig {
            width: 1920,
            height: 1088, // Aligned to 16
            num_frames: 100,
            warmup_frames: 10,
            ..Default::default()
        }
    }

    /// Create a 4K benchmark configuration
    pub fn preset_4k() -> Self {
        BenchmarkConfig {
            width: 3840,
            height: 2160,
            num_frames: 50,
            warmup_frames: 5,
            ..Default::default()
        }
    }

    /// Set the number of frames
    pub fn with_frames(mut self, frames: usize) -> Self {
        self.num_frames = frames;
        self
    }

    /// Set the quality level
    pub fn with_quality(mut self, quality: Quality) -> Self {
        self.quality = quality;
        self
    }

    /// Set whether to benchmark decoding
    pub fn with_decode(mut self, decode: bool) -> Self {
        self.benchmark_decode = decode;
        self
    }

    /// Set test pattern
    pub fn with_pattern(mut self, pattern: TestPattern) -> Self {
        self.pattern = pattern;
        self
    }
}

/// Test pattern types for synthetic frames
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestPattern {
    /// Solid color (gray)
    Solid,
    /// Horizontal/vertical gradient
    Gradient,
    /// Checkerboard pattern
    Checkerboard,
    /// Random noise
    Noise,
    /// Moving pattern (for temporal variation)
    Moving,
}

impl Default for TestPattern {
    fn default() -> Self {
        TestPattern::Gradient
    }
}

// ============================================================================
// Frame Generation
// ============================================================================

/// Generate a synthetic test frame
fn generate_test_frame(
    width: u32,
    height: u32,
    frame_num: usize,
    pattern: TestPattern,
) -> VideoFrame {
    let y_size = (width * height) as usize;
    let uv_size = y_size / 4;

    let mut y_data = vec![0u8; y_size];
    let u_data = vec![128u8; uv_size];
    let v_data = vec![128u8; uv_size];

    match pattern {
        TestPattern::Solid => {
            y_data.fill(128);
        }
        TestPattern::Gradient => {
            for row in 0..height as usize {
                for col in 0..width as usize {
                    let y_val =
                        ((row as f32 / height as f32 + col as f32 / width as f32) / 2.0 * 255.0)
                            as u8;
                    y_data[row * width as usize + col] = y_val;
                }
            }
        }
        TestPattern::Checkerboard => {
            let block_size = 16;
            for row in 0..height as usize {
                for col in 0..width as usize {
                    let block_x = col / block_size;
                    let block_y = row / block_size;
                    let is_white = (block_x + block_y) % 2 == 0;
                    y_data[row * width as usize + col] = if is_white { 235 } else { 16 };
                }
            }
        }
        TestPattern::Noise => {
            // Simple pseudo-random based on position and frame
            for i in 0..y_size {
                let seed = (i as u64)
                    .wrapping_mul(1103515245)
                    .wrapping_add(12345 + frame_num as u64);
                y_data[i] = ((seed >> 16) & 0xFF) as u8;
            }
        }
        TestPattern::Moving => {
            // Moving gradient to simulate motion
            let offset = (frame_num * 4) % width as usize;
            for row in 0..height as usize {
                for col in 0..width as usize {
                    let shifted_col = (col + offset) % width as usize;
                    let y_val = ((shifted_col as f32 / width as f32) * 200.0 + 28.0) as u8;
                    y_data[row * width as usize + col] = y_val;
                }
            }
        }
    }

    let mut frame = VideoFrame::new(width, height, PixelFormat::YUV420P);
    frame.data = vec![
        Buffer::from_vec(y_data),
        Buffer::from_vec(u_data),
        Buffer::from_vec(v_data),
    ];
    frame.linesize = vec![width as usize, (width / 2) as usize, (width / 2) as usize];
    frame.pts = Timestamp::new(frame_num as i64 * 1000 / 30); // 30fps

    frame
}

// ============================================================================
// Percentile Calculation
// ============================================================================

/// Calculate percentile from a sorted slice
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ============================================================================
// Benchmark Functions
// ============================================================================

/// Run a 720p benchmark
///
/// # Arguments
///
/// * `num_frames` - Number of frames to benchmark
///
/// # Returns
///
/// Benchmark results for 720p encoding/decoding
pub fn benchmark_720p(num_frames: usize) -> BenchmarkResult {
    let config = BenchmarkConfig::preset_720p().with_frames(num_frames);
    run_benchmark(config)
}

/// Run a 1080p benchmark
///
/// # Arguments
///
/// * `num_frames` - Number of frames to benchmark
///
/// # Returns
///
/// Benchmark results for 1080p encoding/decoding
pub fn benchmark_1080p(num_frames: usize) -> BenchmarkResult {
    let config = BenchmarkConfig::preset_1080p().with_frames(num_frames);
    run_benchmark(config)
}

/// Run a 4K benchmark
///
/// # Arguments
///
/// * `num_frames` - Number of frames to benchmark
///
/// # Returns
///
/// Benchmark results for 4K encoding/decoding
pub fn benchmark_4k(num_frames: usize) -> BenchmarkResult {
    let config = BenchmarkConfig::preset_4k().with_frames(num_frames);
    run_benchmark(config)
}

/// Run a benchmark with custom configuration
///
/// # Arguments
///
/// * `config` - Benchmark configuration
///
/// # Returns
///
/// Benchmark results
pub fn run_benchmark(config: BenchmarkConfig) -> BenchmarkResult {
    let mut result = BenchmarkResult::new(config.width, config.height);
    result.quality = config.quality;
    result.num_frames = config.num_frames;

    // Create encoder
    let codec_config = ZVC69Config::builder()
        .dimensions(config.width, config.height)
        .quality(config.quality)
        .build()
        .unwrap();

    let mut encoder = match SyncEncoder::new(codec_config) {
        Ok(e) => e,
        Err(_) => return result,
    };

    // Warmup phase
    for i in 0..config.warmup_frames {
        let frame = generate_test_frame(config.width, config.height, i, config.pattern);
        let _ = encoder.encode(frame);
    }
    result.warmed_up = config.warmup_frames > 0;

    // Collect encoded frames for decode benchmark
    let mut encoded_frames = Vec::with_capacity(config.num_frames);
    let mut encode_times = Vec::with_capacity(config.num_frames);
    let mut total_bytes = 0usize;

    // Encode benchmark
    let encode_start = Instant::now();
    for i in 0..config.num_frames {
        let frame = generate_test_frame(
            config.width,
            config.height,
            i + config.warmup_frames,
            config.pattern,
        );

        let frame_start = Instant::now();
        if let Ok(encoded) = encoder.encode(frame) {
            let elapsed = frame_start.elapsed().as_secs_f64() * 1000.0;
            encode_times.push(elapsed);
            total_bytes += encoded.data.len();
            encoded_frames.push(encoded);
        }
    }
    let total_encode_time = encode_start.elapsed();

    // Calculate encode statistics
    if !encode_times.is_empty() {
        encode_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = encode_times.len();

        result.avg_encode_ms = encode_times.iter().sum::<f64>() / n as f64;
        result.p50_encode_ms = percentile(&encode_times, 50.0);
        result.p95_encode_ms = percentile(&encode_times, 95.0);
        result.p99_encode_ms = percentile(&encode_times, 99.0);
        result.min_encode_ms = encode_times[0];
        result.max_encode_ms = encode_times[n - 1];
        result.encode_fps = config.num_frames as f64 / total_encode_time.as_secs_f64();
    }

    result.total_bytes = total_bytes;
    let pixels = config.width as f64 * config.height as f64 * config.num_frames as f64;
    result.bpp = if pixels > 0.0 {
        (total_bytes * 8) as f64 / pixels
    } else {
        0.0
    };
    result.bitrate_kbps = if total_encode_time.as_secs_f64() > 0.0 {
        (total_bytes * 8) as f64 / total_encode_time.as_secs_f64() / 1000.0
    } else {
        0.0
    };

    // Decode benchmark
    if config.benchmark_decode && !encoded_frames.is_empty() {
        let mut decoder = match SyncDecoder::new() {
            Ok(d) => d,
            Err(_) => return result,
        };

        let mut decode_times = Vec::with_capacity(encoded_frames.len());

        let decode_start = Instant::now();
        for encoded in encoded_frames {
            let frame_start = Instant::now();
            if decoder.decode(encoded).is_ok() {
                let elapsed = frame_start.elapsed().as_secs_f64() * 1000.0;
                decode_times.push(elapsed);
            }
        }
        let total_decode_time = decode_start.elapsed();

        // Calculate decode statistics
        if !decode_times.is_empty() {
            decode_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = decode_times.len();

            result.avg_decode_ms = decode_times.iter().sum::<f64>() / n as f64;
            result.p50_decode_ms = percentile(&decode_times, 50.0);
            result.p95_decode_ms = percentile(&decode_times, 95.0);
            result.p99_decode_ms = percentile(&decode_times, 99.0);
            result.min_decode_ms = decode_times[0];
            result.max_decode_ms = decode_times[n - 1];
            result.decode_fps = config.num_frames as f64 / total_decode_time.as_secs_f64();
        }
    }

    // Estimate memory usage
    let frame_bytes = (config.width * config.height * 3 / 2) as f64; // YUV420
    let buffer_count = 8.0; // Approximate buffer pool size
    result.memory_mb = (frame_bytes * buffer_count) / 1_000_000.0;

    result
}

/// Run the full benchmark suite
///
/// Tests 720p, 1080p, and 4K at various quality levels.
///
/// # Returns
///
/// Vector of benchmark results for all configurations
pub fn run_benchmark_suite() -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    // 720p benchmarks
    results.push(benchmark_720p(100));

    // 1080p benchmarks
    results.push(benchmark_1080p(50));

    results
}

/// Run quick validation benchmark
///
/// Fast benchmark to verify real-time capability.
///
/// # Returns
///
/// `true` if 720p meets 30 fps target
pub fn quick_validation() -> bool {
    let result = benchmark_720p(30);
    result.encode_meets_target(30.0)
}

/// Run latency-focused benchmark
///
/// Benchmarks with focus on P99 latency.
pub fn benchmark_latency(width: u32, height: u32, num_frames: usize) -> BenchmarkResult {
    let config = BenchmarkConfig {
        width,
        height,
        num_frames,
        warmup_frames: 20, // More warmup for stable latency
        pattern: TestPattern::Moving, // Motion pattern for realistic P-frames
        ..Default::default()
    };
    run_benchmark(config)
}

/// Compare different quality levels
///
/// Benchmarks across all quality levels for the given resolution.
pub fn benchmark_quality_levels(width: u32, height: u32) -> Vec<BenchmarkResult> {
    let qualities = [
        Quality::Q1,
        Quality::Q2,
        Quality::Q3,
        Quality::Q4,
        Quality::Q5,
        Quality::Q6,
        Quality::Q7,
        Quality::Q8,
    ];

    qualities
        .iter()
        .map(|&quality| {
            let config = BenchmarkConfig {
                width,
                height,
                num_frames: 50,
                warmup_frames: 5,
                quality,
                ..Default::default()
            };
            run_benchmark(config)
        })
        .collect()
}

// ============================================================================
// Performance Assertions
// ============================================================================

/// Assert that the benchmark meets real-time requirements
///
/// # Panics
///
/// Panics if the benchmark does not meet the specified fps target.
pub fn assert_realtime(result: &BenchmarkResult, target_fps: f64) {
    assert!(
        result.encode_fps >= target_fps,
        "Encode FPS {:.1} does not meet target {:.1}",
        result.encode_fps,
        target_fps
    );
    assert!(
        result.decode_fps >= target_fps,
        "Decode FPS {:.1} does not meet target {:.1}",
        result.decode_fps,
        target_fps
    );
}

/// Assert that latency meets target
///
/// # Panics
///
/// Panics if P99 latency exceeds the target.
pub fn assert_latency(result: &BenchmarkResult, target_encode_ms: f64, target_decode_ms: f64) {
    assert!(
        result.p99_encode_ms <= target_encode_ms,
        "Encode P99 latency {:.2}ms exceeds target {:.2}ms",
        result.p99_encode_ms,
        target_encode_ms
    );
    assert!(
        result.p99_decode_ms <= target_decode_ms,
        "Decode P99 latency {:.2}ms exceeds target {:.2}ms",
        result.p99_decode_ms,
        target_decode_ms
    );
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── BenchmarkResult Tests ──

    #[test]
    fn test_benchmark_result_new() {
        let result = BenchmarkResult::new(1280, 720);
        assert_eq!(result.resolution, (1280, 720));
        assert_eq!(result.num_frames, 0);
        assert_eq!(result.encode_fps, 0.0);
    }

    #[test]
    fn test_benchmark_result_is_realtime() {
        let mut result = BenchmarkResult::new(1280, 720);
        result.encode_fps = 35.0;
        result.decode_fps = 65.0;

        assert!(result.is_realtime(30.0));
        assert!(!result.is_realtime(40.0));
    }

    #[test]
    fn test_benchmark_result_encode_meets_target() {
        let mut result = BenchmarkResult::new(1280, 720);
        result.encode_fps = 35.0;

        assert!(result.encode_meets_target(30.0));
        assert!(!result.encode_meets_target(40.0));
    }

    #[test]
    fn test_benchmark_result_decode_meets_target() {
        let mut result = BenchmarkResult::new(1280, 720);
        result.decode_fps = 65.0;

        assert!(result.decode_meets_target(60.0));
        assert!(!result.decode_meets_target(70.0));
    }

    #[test]
    fn test_benchmark_result_latency_meets_target() {
        let mut result = BenchmarkResult::new(1280, 720);
        result.avg_encode_ms = 25.0;
        result.avg_decode_ms = 12.0;

        assert!(result.latency_meets_target(33.0, 16.0));
        assert!(!result.latency_meets_target(20.0, 16.0));
    }

    #[test]
    fn test_benchmark_result_to_table() {
        let mut result = BenchmarkResult::new(1280, 720);
        result.num_frames = 100;
        result.encode_fps = 35.0;
        result.decode_fps = 65.0;
        result.avg_encode_ms = 28.5;
        result.avg_decode_ms = 15.4;

        let table = result.to_table();
        assert!(table.contains("720p"));
        assert!(table.contains("100"));
        assert!(table.contains("35.0"));
    }

    #[test]
    fn test_benchmark_result_to_csv() {
        let mut result = BenchmarkResult::new(1280, 720);
        result.num_frames = 100;
        result.encode_fps = 35.0;

        let csv = result.to_csv();
        assert!(csv.contains("1280,720"));
        assert!(csv.contains("100"));
    }

    #[test]
    fn test_csv_header() {
        let header = BenchmarkResult::csv_header();
        assert!(header.contains("width"));
        assert!(header.contains("encode_fps"));
    }

    // ── BenchmarkConfig Tests ──

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
        assert_eq!(config.num_frames, 100);
    }

    #[test]
    fn test_benchmark_config_preset_720p() {
        let config = BenchmarkConfig::preset_720p();
        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
        assert_eq!(config.num_frames, 150);
    }

    #[test]
    fn test_benchmark_config_preset_1080p() {
        let config = BenchmarkConfig::preset_1080p();
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1088);
    }

    #[test]
    fn test_benchmark_config_preset_4k() {
        let config = BenchmarkConfig::preset_4k();
        assert_eq!(config.width, 3840);
        assert_eq!(config.height, 2160);
    }

    #[test]
    fn test_benchmark_config_builder() {
        let config = BenchmarkConfig::default()
            .with_frames(200)
            .with_quality(Quality::Q6)
            .with_decode(false)
            .with_pattern(TestPattern::Noise);

        assert_eq!(config.num_frames, 200);
        assert_eq!(config.quality, Quality::Q6);
        assert!(!config.benchmark_decode);
        assert_eq!(config.pattern, TestPattern::Noise);
    }

    // ── Test Pattern Tests ──

    #[test]
    fn test_test_pattern_default() {
        let pattern = TestPattern::default();
        assert_eq!(pattern, TestPattern::Gradient);
    }

    // ── Frame Generation Tests ──

    #[test]
    fn test_generate_test_frame_solid() {
        let frame = generate_test_frame(64, 64, 0, TestPattern::Solid);
        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 64);
        assert_eq!(frame.format, PixelFormat::YUV420P);
        assert_eq!(frame.data.len(), 3);

        // Check Y plane is solid gray
        let y_data = frame.data[0].as_slice();
        assert!(y_data.iter().all(|&v| v == 128));
    }

    #[test]
    fn test_generate_test_frame_gradient() {
        let frame = generate_test_frame(64, 64, 0, TestPattern::Gradient);
        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 64);

        // Check Y plane has variation (gradient)
        let y_data = frame.data[0].as_slice();
        assert!(y_data[0] != y_data[y_data.len() - 1]);
    }

    #[test]
    fn test_generate_test_frame_checkerboard() {
        let frame = generate_test_frame(64, 64, 0, TestPattern::Checkerboard);
        let y_data = frame.data[0].as_slice();

        // Check for alternating blocks
        assert!(y_data[0] == 235 || y_data[0] == 16);
        assert!(y_data[16] == 235 || y_data[16] == 16);
        assert_ne!(y_data[0], y_data[16]);
    }

    #[test]
    fn test_generate_test_frame_noise() {
        let frame = generate_test_frame(64, 64, 0, TestPattern::Noise);
        let y_data = frame.data[0].as_slice();

        // Check for variation (pseudo-random)
        let unique_values: std::collections::HashSet<_> = y_data.iter().collect();
        assert!(unique_values.len() > 10); // Should have many unique values
    }

    #[test]
    fn test_generate_test_frame_moving() {
        let frame0 = generate_test_frame(64, 64, 0, TestPattern::Moving);
        let frame1 = generate_test_frame(64, 64, 1, TestPattern::Moving);

        // Frames should be different due to motion
        let y0 = frame0.data[0].as_slice();
        let y1 = frame1.data[0].as_slice();
        assert_ne!(y0, y1);
    }

    // ── Percentile Tests ──

    #[test]
    fn test_percentile_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&data, 50.0) - 3.0).abs() < 0.1);
        assert!((percentile(&data, 0.0) - 1.0).abs() < 0.1);
        assert!((percentile(&data, 100.0) - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_percentile_empty() {
        let data: Vec<f64> = vec![];
        assert_eq!(percentile(&data, 50.0), 0.0);
    }

    // ── Benchmark Function Tests ──

    #[test]
    fn test_benchmark_720p_small() {
        let result = benchmark_720p(5);
        assert_eq!(result.resolution, (1280, 720));
        assert_eq!(result.num_frames, 5);
        assert!(result.encode_fps > 0.0);
    }

    #[test]
    fn test_run_benchmark_custom() {
        let config = BenchmarkConfig {
            width: 64,
            height: 64,
            num_frames: 5,
            warmup_frames: 1,
            ..Default::default()
        };
        let result = run_benchmark(config);

        assert_eq!(result.resolution, (64, 64));
        assert_eq!(result.num_frames, 5);
        assert!(result.encode_fps > 0.0);
        assert!(result.avg_encode_ms > 0.0);
    }

    #[test]
    fn test_run_benchmark_with_decode() {
        let config = BenchmarkConfig {
            width: 64,
            height: 64,
            num_frames: 3,
            warmup_frames: 1,
            benchmark_decode: true,
            ..Default::default()
        };
        let result = run_benchmark(config);

        assert!(result.encode_fps > 0.0);
        assert!(result.decode_fps > 0.0);
    }

    #[test]
    fn test_run_benchmark_without_decode() {
        let config = BenchmarkConfig {
            width: 64,
            height: 64,
            num_frames: 3,
            warmup_frames: 0,
            benchmark_decode: false,
            ..Default::default()
        };
        let result = run_benchmark(config);

        assert!(result.encode_fps > 0.0);
        assert_eq!(result.decode_fps, 0.0);
    }

    #[test]
    fn test_run_benchmark_suite() {
        // This is a quick test - full suite would take too long
        let results = run_benchmark_suite();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_quick_validation() {
        // Just verify it runs without crashing
        let _ = quick_validation();
    }

    #[test]
    fn test_benchmark_latency() {
        let result = benchmark_latency(64, 64, 5);
        assert_eq!(result.resolution, (64, 64));
        assert!(result.warmed_up); // Should have extra warmup
    }

    // ── Performance Assertion Tests ──

    #[test]
    fn test_assert_realtime_pass() {
        let mut result = BenchmarkResult::new(1280, 720);
        result.encode_fps = 35.0;
        result.decode_fps = 65.0;

        // Should not panic
        assert_realtime(&result, 30.0);
    }

    #[test]
    #[should_panic(expected = "Encode FPS")]
    fn test_assert_realtime_fail_encode() {
        let mut result = BenchmarkResult::new(1280, 720);
        result.encode_fps = 25.0;
        result.decode_fps = 65.0;

        assert_realtime(&result, 30.0);
    }

    #[test]
    #[should_panic(expected = "Decode FPS")]
    fn test_assert_realtime_fail_decode() {
        let mut result = BenchmarkResult::new(1280, 720);
        result.encode_fps = 35.0;
        result.decode_fps = 25.0;

        assert_realtime(&result, 30.0);
    }

    #[test]
    fn test_assert_latency_pass() {
        let mut result = BenchmarkResult::new(1280, 720);
        result.p99_encode_ms = 30.0;
        result.p99_decode_ms = 14.0;

        // Should not panic
        assert_latency(&result, 33.0, 16.0);
    }

    #[test]
    #[should_panic(expected = "Encode P99 latency")]
    fn test_assert_latency_fail_encode() {
        let mut result = BenchmarkResult::new(1280, 720);
        result.p99_encode_ms = 40.0;
        result.p99_decode_ms = 14.0;

        assert_latency(&result, 33.0, 16.0);
    }

    // ── Memory and Bitrate Tests ──

    #[test]
    fn test_benchmark_calculates_memory() {
        let config = BenchmarkConfig {
            width: 1280,
            height: 720,
            num_frames: 5,
            warmup_frames: 0,
            ..Default::default()
        };
        let result = run_benchmark(config);

        assert!(result.memory_mb > 0.0);
    }

    #[test]
    fn test_benchmark_calculates_bitrate() {
        let config = BenchmarkConfig {
            width: 64,
            height: 64,
            num_frames: 10,
            warmup_frames: 0,
            ..Default::default()
        };
        let result = run_benchmark(config);

        assert!(result.bitrate_kbps > 0.0);
        assert!(result.bpp > 0.0);
    }

    // ── Quality Level Tests ──

    #[test]
    fn test_benchmark_quality_levels() {
        let results = benchmark_quality_levels(64, 64);

        assert_eq!(results.len(), 8);

        // Higher quality should generally produce larger files
        // (though with synthetic patterns this may vary)
        for result in &results {
            assert!(result.encode_fps > 0.0);
        }
    }
}
