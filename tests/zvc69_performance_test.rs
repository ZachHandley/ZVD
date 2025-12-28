//! ZVC69 Performance Regression Tests
//!
//! This module contains performance regression tests for the ZVC69 neural codec.
//! These tests verify that the codec meets real-time performance targets on Apple M3 hardware.
//!
//! ## Performance Targets (M3 at 720p)
//!
//! | Operation | FPS Target | Latency Target (P99) |
//! |-----------|------------|----------------------|
//! | Encode    | 30 fps     | < 33ms               |
//! | Decode    | 60 fps     | < 16ms               |
//!
//! ## Running the Tests
//!
//! These tests are marked with `#[ignore]` because they require GPU hardware to run.
//! To execute them:
//!
//! ```bash
//! # Run all performance tests
//! cargo test --test zvc69_performance_test --features zvc69 -- --ignored
//!
//! # Run a specific test
//! cargo test --test zvc69_performance_test --features zvc69 test_m3_720p_encode_fps -- --ignored
//!
//! # Run with output
//! cargo test --test zvc69_performance_test --features zvc69 -- --ignored --nocapture
//! ```
//!
//! ## Environment Requirements
//!
//! - Apple M3 (or compatible) hardware with GPU
//! - `zvc69` feature enabled
//! - ONNX Runtime with CoreML/Metal backend available

#[cfg(feature = "zvc69")]
mod performance_tests {
    use zvd_lib::codec::zvc69::benchmark::{
        assert_latency, assert_m4_1080p_all_targets, assert_m4_1080p_decode_fps,
        assert_m4_1080p_decode_latency, assert_m4_1080p_encode_fps, assert_m4_1080p_encode_latency,
        assert_realtime, benchmark_1080p_m4, benchmark_720p, benchmark_latency,
        check_m4_1080p_targets, BenchmarkConfig, BenchmarkResult, TestPattern,
        M4_1080P_DECODE_FPS_TARGET, M4_1080P_DECODE_P99_LATENCY_MS, M4_1080P_ENCODE_FPS_TARGET,
        M4_1080P_ENCODE_P99_LATENCY_MS,
    };

    // ============================================================================
    // Performance Target Constants
    // ============================================================================

    /// M3 target encode FPS at 720p (30 fps for real-time video)
    const M3_720P_ENCODE_FPS_TARGET: f64 = 30.0;

    /// M3 target decode FPS at 720p (60 fps for smooth playback)
    const M3_720P_DECODE_FPS_TARGET: f64 = 60.0;

    /// M3 target P99 encode latency at 720p (must complete within frame budget)
    /// 33ms = 1/30 fps
    const M3_720P_ENCODE_P99_LATENCY_MS: f64 = 33.0;

    /// M3 target P99 decode latency at 720p (must complete within frame budget)
    /// 16ms = ~1/60 fps
    const M3_720P_DECODE_P99_LATENCY_MS: f64 = 16.0;

    /// Number of frames to benchmark for FPS tests
    const FPS_BENCHMARK_FRAMES: usize = 150;

    /// Number of frames to benchmark for latency tests (more frames for stable P99)
    const LATENCY_BENCHMARK_FRAMES: usize = 200;

    /// Warmup frames to exclude from measurements
    const WARMUP_FRAMES: usize = 20;

    /// Number of frames for M4 1080p benchmarks
    const M4_1080P_BENCHMARK_FRAMES: usize = 150;

    // ============================================================================
    // Helper Functions
    // ============================================================================

    /// Print benchmark results in a readable format for CI logs
    fn print_benchmark_summary(result: &BenchmarkResult, test_name: &str) {
        println!("\n========================================");
        println!("Performance Test: {}", test_name);
        println!("========================================");
        println!(
            "Resolution: {}x{}",
            result.resolution.0, result.resolution.1
        );
        println!("Frames: {}", result.num_frames);
        println!("Quality: {}", result.quality);
        println!("----------------------------------------");
        println!("ENCODE:");
        println!("  FPS:        {:.1} fps", result.encode_fps);
        println!("  Avg:        {:.2} ms", result.avg_encode_ms);
        println!("  P50:        {:.2} ms", result.p50_encode_ms);
        println!("  P95:        {:.2} ms", result.p95_encode_ms);
        println!("  P99:        {:.2} ms", result.p99_encode_ms);
        println!(
            "  Min/Max:    {:.2}/{:.2} ms",
            result.min_encode_ms, result.max_encode_ms
        );
        println!("----------------------------------------");
        println!("DECODE:");
        println!("  FPS:        {:.1} fps", result.decode_fps);
        println!("  Avg:        {:.2} ms", result.avg_decode_ms);
        println!("  P50:        {:.2} ms", result.p50_decode_ms);
        println!("  P95:        {:.2} ms", result.p95_decode_ms);
        println!("  P99:        {:.2} ms", result.p99_decode_ms);
        println!(
            "  Min/Max:    {:.2}/{:.2} ms",
            result.min_decode_ms, result.max_decode_ms
        );
        println!("----------------------------------------");
        println!("Bitrate:      {:.2} kbps", result.bitrate_kbps);
        println!("BPP:          {:.4}", result.bpp);
        println!(
            "Total Size:   {:.2} MB",
            result.total_bytes as f64 / 1_000_000.0
        );
        println!("========================================\n");
    }

    /// Create a benchmark config optimized for latency testing
    fn create_latency_benchmark_config() -> BenchmarkConfig {
        BenchmarkConfig {
            width: 1280,
            height: 720,
            num_frames: LATENCY_BENCHMARK_FRAMES,
            warmup_frames: WARMUP_FRAMES,
            pattern: TestPattern::Moving, // Motion pattern for realistic P-frame behavior
            benchmark_decode: true,
            ..Default::default()
        }
    }

    // ============================================================================
    // FPS Performance Tests
    // ============================================================================

    /// Test that 720p encoding meets M3 target of 30 fps
    ///
    /// This test verifies that the ZVC69 encoder can encode 720p video frames
    /// at a sustained rate of at least 30 frames per second on M3 hardware.
    #[test]
    #[ignore]
    fn test_m3_720p_encode_fps() {
        let result = benchmark_720p(FPS_BENCHMARK_FRAMES);
        print_benchmark_summary(&result, "M3 720p Encode FPS");

        assert!(
            result.encode_fps >= M3_720P_ENCODE_FPS_TARGET,
            "720p encode FPS {:.1} does not meet M3 target of {:.1} fps",
            result.encode_fps,
            M3_720P_ENCODE_FPS_TARGET
        );

        println!(
            "PASS: Encode FPS {:.1} >= target {:.1}",
            result.encode_fps, M3_720P_ENCODE_FPS_TARGET
        );
    }

    /// Test that 720p decoding meets M3 target of 60 fps
    ///
    /// This test verifies that the ZVC69 decoder can decode 720p video frames
    /// at a sustained rate of at least 60 frames per second on M3 hardware.
    #[test]
    #[ignore]
    fn test_m3_720p_decode_fps() {
        let result = benchmark_720p(FPS_BENCHMARK_FRAMES);
        print_benchmark_summary(&result, "M3 720p Decode FPS");

        assert!(
            result.decode_fps >= M3_720P_DECODE_FPS_TARGET,
            "720p decode FPS {:.1} does not meet M3 target of {:.1} fps",
            result.decode_fps,
            M3_720P_DECODE_FPS_TARGET
        );

        println!(
            "PASS: Decode FPS {:.1} >= target {:.1}",
            result.decode_fps, M3_720P_DECODE_FPS_TARGET
        );
    }

    // ============================================================================
    // Latency Performance Tests
    // ============================================================================

    /// Test that 720p encode P99 latency meets M3 target of < 33ms
    ///
    /// This test verifies that 99% of encode operations complete within 33ms,
    /// ensuring consistent real-time encoding performance.
    #[test]
    #[ignore]
    fn test_m3_720p_encode_latency() {
        let result = benchmark_latency(1280, 720, LATENCY_BENCHMARK_FRAMES);
        print_benchmark_summary(&result, "M3 720p Encode Latency");

        assert!(
            result.p99_encode_ms <= M3_720P_ENCODE_P99_LATENCY_MS,
            "720p encode P99 latency {:.2}ms exceeds M3 target of {:.2}ms",
            result.p99_encode_ms,
            M3_720P_ENCODE_P99_LATENCY_MS
        );

        println!(
            "PASS: Encode P99 {:.2}ms <= target {:.2}ms",
            result.p99_encode_ms, M3_720P_ENCODE_P99_LATENCY_MS
        );
    }

    /// Test that 720p decode P99 latency meets M3 target of < 16ms
    ///
    /// This test verifies that 99% of decode operations complete within 16ms,
    /// ensuring consistent smooth playback at 60fps.
    #[test]
    #[ignore]
    fn test_m3_720p_decode_latency() {
        let result = benchmark_latency(1280, 720, LATENCY_BENCHMARK_FRAMES);
        print_benchmark_summary(&result, "M3 720p Decode Latency");

        assert!(
            result.p99_decode_ms <= M3_720P_DECODE_P99_LATENCY_MS,
            "720p decode P99 latency {:.2}ms exceeds M3 target of {:.2}ms",
            result.p99_decode_ms,
            M3_720P_DECODE_P99_LATENCY_MS
        );

        println!(
            "PASS: Decode P99 {:.2}ms <= target {:.2}ms",
            result.p99_decode_ms, M3_720P_DECODE_P99_LATENCY_MS
        );
    }

    // ============================================================================
    // Combined Real-time Tests
    // ============================================================================

    /// Test that both encode and decode meet real-time requirements simultaneously
    ///
    /// This is a comprehensive test that verifies the codec can sustain real-time
    /// performance for both encoding and decoding operations.
    #[test]
    #[ignore]
    fn test_m3_720p_realtime() {
        let result = benchmark_720p(FPS_BENCHMARK_FRAMES);
        print_benchmark_summary(&result, "M3 720p Real-time");

        // Use the benchmark module's assert function
        assert_realtime(&result, M3_720P_ENCODE_FPS_TARGET);

        println!(
            "PASS: Real-time encode {:.1} fps, decode {:.1} fps",
            result.encode_fps, result.decode_fps
        );
    }

    /// Test that both encode and decode latencies meet targets simultaneously
    ///
    /// This is a comprehensive test that verifies P99 latencies for both
    /// encoding and decoding meet their respective targets.
    #[test]
    #[ignore]
    fn test_m3_720p_combined_latency() {
        let result = benchmark_latency(1280, 720, LATENCY_BENCHMARK_FRAMES);
        print_benchmark_summary(&result, "M3 720p Combined Latency");

        // Use the benchmark module's assert function
        assert_latency(
            &result,
            M3_720P_ENCODE_P99_LATENCY_MS,
            M3_720P_DECODE_P99_LATENCY_MS,
        );

        println!(
            "PASS: Encode P99 {:.2}ms, Decode P99 {:.2}ms",
            result.p99_encode_ms, result.p99_decode_ms
        );
    }

    // ============================================================================
    // Regression Detection Tests
    // ============================================================================

    /// Test for performance regression with a safety margin
    ///
    /// This test uses a 10% safety margin below targets to catch regressions
    /// before they become critical failures.
    #[test]
    #[ignore]
    fn test_m3_720p_regression_margin() {
        let result = benchmark_720p(FPS_BENCHMARK_FRAMES);
        print_benchmark_summary(&result, "M3 720p Regression Margin");

        // Apply 10% safety margin
        let encode_fps_with_margin = M3_720P_ENCODE_FPS_TARGET * 1.1;
        let decode_fps_with_margin = M3_720P_DECODE_FPS_TARGET * 1.1;
        let encode_latency_with_margin = M3_720P_ENCODE_P99_LATENCY_MS * 0.9;
        let decode_latency_with_margin = M3_720P_DECODE_P99_LATENCY_MS * 0.9;

        // Check FPS with margin (warning, not failure)
        if result.encode_fps < encode_fps_with_margin {
            println!(
                "WARNING: Encode FPS {:.1} below 10% margin threshold {:.1}",
                result.encode_fps, encode_fps_with_margin
            );
        }
        if result.decode_fps < decode_fps_with_margin {
            println!(
                "WARNING: Decode FPS {:.1} below 10% margin threshold {:.1}",
                result.decode_fps, decode_fps_with_margin
            );
        }

        // Check latency with margin (warning, not failure)
        if result.p99_encode_ms > encode_latency_with_margin {
            println!(
                "WARNING: Encode P99 {:.2}ms above 10% margin threshold {:.2}ms",
                result.p99_encode_ms, encode_latency_with_margin
            );
        }
        if result.p99_decode_ms > decode_latency_with_margin {
            println!(
                "WARNING: Decode P99 {:.2}ms above 10% margin threshold {:.2}ms",
                result.p99_decode_ms, decode_latency_with_margin
            );
        }

        // Hard assertion on actual targets
        assert!(
            result.encode_fps >= M3_720P_ENCODE_FPS_TARGET,
            "Encode FPS regression: {:.1} < {:.1}",
            result.encode_fps,
            M3_720P_ENCODE_FPS_TARGET
        );
        assert!(
            result.decode_fps >= M3_720P_DECODE_FPS_TARGET,
            "Decode FPS regression: {:.1} < {:.1}",
            result.decode_fps,
            M3_720P_DECODE_FPS_TARGET
        );
    }

    // ============================================================================
    // Stress Tests
    // ============================================================================

    /// Test sustained performance over a longer duration
    ///
    /// This test runs for 500 frames to verify performance remains stable
    /// and doesn't degrade over time (thermal throttling, memory pressure, etc.)
    #[test]
    #[ignore]
    fn test_m3_720p_sustained_performance() {
        const SUSTAINED_FRAMES: usize = 500;

        let config = BenchmarkConfig {
            width: 1280,
            height: 720,
            num_frames: SUSTAINED_FRAMES,
            warmup_frames: WARMUP_FRAMES,
            pattern: TestPattern::Moving,
            benchmark_decode: true,
            ..Default::default()
        };

        let result = zvd_lib::codec::zvc69::benchmark::run_benchmark(config);
        print_benchmark_summary(&result, "M3 720p Sustained Performance (500 frames)");

        // Sustained performance should still meet targets
        assert!(
            result.encode_fps >= M3_720P_ENCODE_FPS_TARGET,
            "Sustained encode FPS {:.1} does not meet target {:.1}",
            result.encode_fps,
            M3_720P_ENCODE_FPS_TARGET
        );
        assert!(
            result.decode_fps >= M3_720P_DECODE_FPS_TARGET,
            "Sustained decode FPS {:.1} does not meet target {:.1}",
            result.decode_fps,
            M3_720P_DECODE_FPS_TARGET
        );

        // P99 latency should remain stable
        assert!(
            result.p99_encode_ms <= M3_720P_ENCODE_P99_LATENCY_MS,
            "Sustained encode P99 {:.2}ms exceeds target {:.2}ms",
            result.p99_encode_ms,
            M3_720P_ENCODE_P99_LATENCY_MS
        );

        println!(
            "PASS: Sustained {} frames - encode {:.1} fps, decode {:.1} fps",
            SUSTAINED_FRAMES, result.encode_fps, result.decode_fps
        );
    }

    // ============================================================================
    // M4 1080p Performance Tests (RTX 3080 Targets)
    // ============================================================================

    /// Test that 1080p encoding meets M4 target of 30 fps
    ///
    /// This test verifies that the ZVC69 encoder can encode 1080p video frames
    /// at a sustained rate of at least 30 frames per second on RTX 3080 hardware.
    #[test]
    #[ignore]
    fn test_m4_1080p_encode_fps() {
        let result = benchmark_1080p_m4(M4_1080P_BENCHMARK_FRAMES);
        print_benchmark_summary(&result, "M4 1080p Encode FPS");

        assert_m4_1080p_encode_fps(&result);

        println!(
            "PASS: M4 1080p encode FPS {:.1} >= target {:.1}",
            result.encode_fps, M4_1080P_ENCODE_FPS_TARGET
        );
    }

    /// Test that 1080p decoding meets M4 target of 60 fps
    ///
    /// This test verifies that the ZVC69 decoder can decode 1080p video frames
    /// at a sustained rate of at least 60 frames per second on RTX 3080 hardware.
    #[test]
    #[ignore]
    fn test_m4_1080p_decode_fps() {
        let result = benchmark_1080p_m4(M4_1080P_BENCHMARK_FRAMES);
        print_benchmark_summary(&result, "M4 1080p Decode FPS");

        assert_m4_1080p_decode_fps(&result);

        println!(
            "PASS: M4 1080p decode FPS {:.1} >= target {:.1}",
            result.decode_fps, M4_1080P_DECODE_FPS_TARGET
        );
    }

    /// Test that 1080p encode P99 latency meets M4 target of < 33ms
    ///
    /// This test verifies that 99% of encode operations complete within 33ms,
    /// ensuring consistent real-time encoding performance at 1080p.
    #[test]
    #[ignore]
    fn test_m4_1080p_encode_latency() {
        let result = benchmark_1080p_m4(LATENCY_BENCHMARK_FRAMES);
        print_benchmark_summary(&result, "M4 1080p Encode Latency");

        assert_m4_1080p_encode_latency(&result);

        println!(
            "PASS: M4 1080p encode P99 {:.2}ms <= target {:.2}ms",
            result.p99_encode_ms, M4_1080P_ENCODE_P99_LATENCY_MS
        );
    }

    /// Test that 1080p decode P99 latency meets M4 target of < 16ms
    ///
    /// This test verifies that 99% of decode operations complete within 16ms,
    /// ensuring consistent smooth playback at 60fps for 1080p content.
    #[test]
    #[ignore]
    fn test_m4_1080p_decode_latency() {
        let result = benchmark_1080p_m4(LATENCY_BENCHMARK_FRAMES);
        print_benchmark_summary(&result, "M4 1080p Decode Latency");

        assert_m4_1080p_decode_latency(&result);

        println!(
            "PASS: M4 1080p decode P99 {:.2}ms <= target {:.2}ms",
            result.p99_decode_ms, M4_1080P_DECODE_P99_LATENCY_MS
        );
    }

    /// Test that all M4 1080p targets are met simultaneously
    ///
    /// Comprehensive test that verifies all four M4 performance requirements:
    /// - 30+ fps encode
    /// - 60+ fps decode
    /// - P99 encode latency < 33ms
    /// - P99 decode latency < 16ms
    #[test]
    #[ignore]
    fn test_m4_1080p_all_targets() {
        let result = benchmark_1080p_m4(M4_1080P_BENCHMARK_FRAMES);
        print_benchmark_summary(&result, "M4 1080p All Targets");

        // Use the comprehensive assertion
        assert_m4_1080p_all_targets(&result);

        println!(
            "PASS: M4 1080p meets all targets - encode {:.1} fps, decode {:.1} fps, P99 encode {:.2}ms, P99 decode {:.2}ms",
            result.encode_fps, result.decode_fps, result.p99_encode_ms, result.p99_decode_ms
        );
    }

    /// Test M4 1080p with safety margin for regression detection
    ///
    /// Uses 10% margin below targets to catch performance regressions early.
    #[test]
    #[ignore]
    fn test_m4_1080p_regression_margin() {
        let result = benchmark_1080p_m4(M4_1080P_BENCHMARK_FRAMES);
        print_benchmark_summary(&result, "M4 1080p Regression Margin");

        // Apply 10% safety margin
        let encode_fps_with_margin = M4_1080P_ENCODE_FPS_TARGET * 1.1;
        let decode_fps_with_margin = M4_1080P_DECODE_FPS_TARGET * 1.1;
        let encode_latency_with_margin = M4_1080P_ENCODE_P99_LATENCY_MS * 0.9;
        let decode_latency_with_margin = M4_1080P_DECODE_P99_LATENCY_MS * 0.9;

        // Check with margin (warning, not failure)
        if result.encode_fps < encode_fps_with_margin {
            println!(
                "WARNING: M4 1080p encode FPS {:.1} below 10% margin threshold {:.1}",
                result.encode_fps, encode_fps_with_margin
            );
        }
        if result.decode_fps < decode_fps_with_margin {
            println!(
                "WARNING: M4 1080p decode FPS {:.1} below 10% margin threshold {:.1}",
                result.decode_fps, decode_fps_with_margin
            );
        }
        if result.p99_encode_ms > encode_latency_with_margin {
            println!(
                "WARNING: M4 1080p encode P99 {:.2}ms above 10% margin threshold {:.2}ms",
                result.p99_encode_ms, encode_latency_with_margin
            );
        }
        if result.p99_decode_ms > decode_latency_with_margin {
            println!(
                "WARNING: M4 1080p decode P99 {:.2}ms above 10% margin threshold {:.2}ms",
                result.p99_decode_ms, decode_latency_with_margin
            );
        }

        // Hard assertion on actual targets
        let (meets_all, _, _, _, _) = check_m4_1080p_targets(&result);
        assert!(
            meets_all,
            "M4 1080p regression: one or more targets not met"
        );
    }

    /// Test sustained M4 1080p performance over 500 frames
    ///
    /// Verifies performance remains stable under prolonged operation
    /// without thermal throttling or memory pressure degradation.
    #[test]
    #[ignore]
    fn test_m4_1080p_sustained_performance() {
        const SUSTAINED_FRAMES: usize = 500;

        let config = BenchmarkConfig::m4_1080p().with_frames(SUSTAINED_FRAMES);

        let result = zvd_lib::codec::zvc69::benchmark::run_benchmark(config);
        print_benchmark_summary(&result, "M4 1080p Sustained Performance (500 frames)");

        // Sustained performance should still meet all M4 targets
        assert_m4_1080p_all_targets(&result);

        println!(
            "PASS: M4 1080p sustained {} frames - encode {:.1} fps, decode {:.1} fps",
            SUSTAINED_FRAMES, result.encode_fps, result.decode_fps
        );
    }
}

// ============================================================================
// Non-feature-gated Tests (Always Run)
// ============================================================================

/// Placeholder test that runs without the zvc69 feature
///
/// This ensures the test file compiles even without the zvc69 feature enabled.
#[test]
fn test_performance_test_file_compiles() {
    // This test exists to verify the file compiles without features
    assert!(true, "Performance test file compiled successfully");
}
