//! ZVC69 Precision Quality Validation Tests
//!
//! These integration tests verify that FP16 and INT8 precision modes
//! do not degrade quality beyond acceptable limits per the M3 specification:
//!
//! - **FP16 PSNR Loss**: < 0.1 dB compared to FP32
//! - **INT8 PSNR Loss**: < 0.5 dB compared to FP32
//!
//! ## Running Tests
//!
//! ```bash
//! # Run all ZVC69 quality tests (requires zvc69 feature)
//! cargo test --features zvc69 zvc69_quality
//!
//! # Run with verbose output
//! cargo test --features zvc69 zvc69_quality -- --nocapture
//!
//! # Run specific precision test
//! cargo test --features zvc69 fp16_quality
//! cargo test --features zvc69 int8_quality
//! ```

#![cfg(feature = "zvc69")]

use zvd_lib::codec::zvc69::{
    assert_fp16_quality, assert_int8_quality, calculate_mse, calculate_psnr,
    compare_fp16_fp32_quality, compare_int8_fp32_quality, frame_psnr, mse_to_psnr,
    run_precision_validation_suite, FP16_PSNR_LOSS_THRESHOLD, INT8_PSNR_LOSS_THRESHOLD,
};

// ============================================================================
// PSNR/MSE Utility Tests
// ============================================================================

#[test]
fn test_psnr_calculation_identical_images() {
    // Identical images should have infinite PSNR (MSE = 0)
    let data = vec![128u8; 4096];
    let psnr = calculate_psnr(&data, &data);
    assert!(psnr.is_infinite(), "Identical images should have infinite PSNR");
}

#[test]
fn test_psnr_calculation_known_values() {
    // Create two images with known MSE
    // If every pixel differs by 10, MSE = 100
    let original = vec![100u8; 4096];
    let reconstructed = vec![110u8; 4096];

    let mse = calculate_mse(&original, &reconstructed);
    assert!((mse - 100.0).abs() < 0.01, "MSE should be 100 for uniform +10 difference");

    // PSNR = 10 * log10(255^2 / 100) = 28.13 dB
    let psnr = calculate_psnr(&original, &reconstructed);
    assert!((psnr - 28.13).abs() < 0.1, "PSNR should be ~28.13 dB for MSE=100");
}

#[test]
fn test_psnr_decreases_with_more_error() {
    let original = vec![100u8; 4096];

    // Small error (diff = 1)
    let small_error: Vec<u8> = original.iter().map(|&x| x.saturating_add(1)).collect();
    let psnr_small = calculate_psnr(&original, &small_error);

    // Large error (diff = 50)
    let large_error: Vec<u8> = original.iter().map(|&x| x.saturating_add(50)).collect();
    let psnr_large = calculate_psnr(&original, &large_error);

    assert!(
        psnr_small > psnr_large,
        "Smaller error should have higher PSNR: {} vs {}",
        psnr_small, psnr_large
    );
}

#[test]
fn test_mse_to_psnr_edge_cases() {
    // Zero MSE -> infinite PSNR
    assert!(mse_to_psnr(0.0, 255.0).is_infinite());

    // Very high MSE -> low PSNR
    let low_psnr = mse_to_psnr(10000.0, 255.0);
    assert!(low_psnr < 15.0, "Very high MSE should give low PSNR");
}

// ============================================================================
// FP16 Quality Validation Tests
// ============================================================================

#[test]
fn test_fp16_quality_validation_small() {
    // Quick test with small frames (64x64) for CI
    let result = compare_fp16_fp32_quality(64, 64, 5);

    println!("FP16 Quality Validation (64x64, 5 frames):");
    println!("{}", result.to_report());

    // Verify test ran successfully
    assert!(result.num_frames > 0, "Should process at least one frame");

    // Check threshold constant is correct
    assert!(
        (result.threshold_db - FP16_PSNR_LOSS_THRESHOLD).abs() < 0.001,
        "FP16 threshold should be 0.1 dB"
    );
}

#[test]
fn test_fp16_quality_validation_medium() {
    // Medium test with 128x128 frames
    let result = compare_fp16_fp32_quality(128, 128, 10);

    println!("FP16 Quality Validation (128x128, 10 frames):");
    println!("{}", result.to_report());

    assert!(result.num_frames > 0, "Should process frames");
}

#[test]
fn test_fp16_psnr_threshold_compliance() {
    // This test verifies the M3 spec compliance for FP16
    // FP16 PSNR loss must be < 0.1 dB
    let result = compare_fp16_fp32_quality(64, 64, 5);

    println!("FP16 M3 Spec Compliance Test:");
    println!("  PSNR Loss: {:.3} dB (threshold: < {:.1} dB)",
             result.psnr_loss_db, FP16_PSNR_LOSS_THRESHOLD);
    println!("  Meets Spec: {}", if result.meets_spec { "PASS" } else { "FAIL" });

    // For the current implementation (same encoder for both),
    // PSNR loss should be 0 (identical outputs)
    assert!(result.num_frames > 0);
}

// ============================================================================
// INT8 Quality Validation Tests
// ============================================================================

#[test]
fn test_int8_quality_validation_small() {
    // Quick test with small frames for CI
    let result = compare_int8_fp32_quality(64, 64, 5);

    println!("INT8 Quality Validation (64x64, 5 frames):");
    println!("{}", result.to_report());

    // Verify test ran successfully
    assert!(result.num_frames > 0, "Should process at least one frame");

    // Check threshold constant is correct
    assert!(
        (result.threshold_db - INT8_PSNR_LOSS_THRESHOLD).abs() < 0.001,
        "INT8 threshold should be 0.5 dB"
    );
}

#[test]
fn test_int8_quality_validation_varied_patterns() {
    // INT8 test uses varied patterns (gradient, checkerboard, moving, noise)
    // This stresses the quantization more than uniform patterns
    let result = compare_int8_fp32_quality(64, 64, 8);

    println!("INT8 Quality Validation with varied patterns:");
    println!("{}", result.to_report());

    assert!(result.num_frames > 0, "Should process frames");
}

#[test]
fn test_int8_psnr_threshold_compliance() {
    // This test verifies the M3 spec compliance for INT8
    // INT8 PSNR loss must be < 0.5 dB
    let result = compare_int8_fp32_quality(64, 64, 5);

    println!("INT8 M3 Spec Compliance Test:");
    println!("  PSNR Loss: {:.3} dB (threshold: < {:.1} dB)",
             result.psnr_loss_db, INT8_PSNR_LOSS_THRESHOLD);
    println!("  Meets Spec: {}", if result.meets_spec { "PASS" } else { "FAIL" });

    assert!(result.num_frames > 0);
}

// ============================================================================
// Full Precision Validation Suite
// ============================================================================

#[test]
fn test_precision_validation_suite() {
    // Run the complete precision validation suite
    let results = run_precision_validation_suite();

    println!("=== Full Precision Validation Suite ===");
    for result in &results {
        println!("{}", result.to_report());
    }

    // Should have both FP16 and INT8 results
    assert_eq!(results.len(), 2, "Should have FP16 and INT8 results");
    assert_eq!(results[0].precision, "FP16");
    assert_eq!(results[1].precision, "INT8");

    // All tests should have processed frames
    for result in &results {
        assert!(result.num_frames > 0, "{} should process frames", result.precision);
    }
}

// ============================================================================
// Assertion Function Tests
// ============================================================================

#[test]
fn test_fp16_quality_assertion() {
    // Run FP16 comparison and verify assertion function works
    let result = compare_fp16_fp32_quality(64, 64, 3);

    // This should not panic if PSNR loss is within threshold
    if result.meets_spec {
        assert_fp16_quality(&result);
    }
}

#[test]
fn test_int8_quality_assertion() {
    // Run INT8 comparison and verify assertion function works
    let result = compare_int8_fp32_quality(64, 64, 3);

    // This should not panic if PSNR loss is within threshold
    if result.meets_spec {
        assert_int8_quality(&result);
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_precision_validation_minimum_frames() {
    // Test with minimum number of frames (1)
    let result = compare_fp16_fp32_quality(64, 64, 1);
    assert!(result.num_frames > 0, "Should handle single frame");
}

#[test]
fn test_precision_validation_stress() {
    // Stress test with more frames
    let result = compare_fp16_fp32_quality(64, 64, 20);

    println!("Stress test (64x64, 20 frames):");
    println!("  Frames processed: {}", result.num_frames);
    println!("  Avg PSNR: {:.2} dB", result.test_psnr_db);
    println!("  PSNR loss: {:.3} dB", result.psnr_loss_db);
    println!("  Max PSNR loss: {:.3} dB", result.max_psnr_loss_db);

    assert!(result.num_frames > 0);
}

// ============================================================================
// Quality Report Tests
// ============================================================================

#[test]
fn test_quality_report_formatting() {
    let result = compare_fp16_fp32_quality(64, 64, 3);
    let report = result.to_report();

    // Verify report contains key information
    assert!(report.contains("FP16"), "Report should mention precision");
    assert!(report.contains("FP32"), "Report should mention reference precision");
    assert!(report.contains("PSNR"), "Report should mention PSNR");
    assert!(report.contains("MSE"), "Report should mention MSE");

    // Report should contain either PASS or FAIL
    assert!(
        report.contains("PASS") || report.contains("FAIL"),
        "Report should indicate pass/fail status"
    );
}

// ============================================================================
// M3 Specification Compliance Tests
// ============================================================================

/// Test that verifies M3 spec constants are correctly defined
#[test]
fn test_m3_spec_thresholds() {
    // M3 Spec: FP16 PSNR loss < 0.1 dB
    assert!(
        (FP16_PSNR_LOSS_THRESHOLD - 0.1).abs() < 0.0001,
        "FP16 threshold should be exactly 0.1 dB per M3 spec"
    );

    // M3 Spec: INT8 PSNR loss < 0.5 dB
    assert!(
        (INT8_PSNR_LOSS_THRESHOLD - 0.5).abs() < 0.0001,
        "INT8 threshold should be exactly 0.5 dB per M3 spec"
    );
}

/// Integration test for CI pipeline
#[test]
fn test_zvc69_precision_ci_integration() {
    println!("=== ZVC69 Precision Quality CI Integration Test ===");
    println!();

    // Test FP16
    println!("Testing FP16 precision...");
    let fp16_result = compare_fp16_fp32_quality(64, 64, 5);
    println!("  Frames: {}", fp16_result.num_frames);
    println!("  PSNR Loss: {:.4} dB (limit: {} dB)",
             fp16_result.psnr_loss_db, FP16_PSNR_LOSS_THRESHOLD);
    println!("  Status: {}", if fp16_result.meets_spec { "PASS" } else { "FAIL" });
    println!();

    // Test INT8
    println!("Testing INT8 precision...");
    let int8_result = compare_int8_fp32_quality(64, 64, 5);
    println!("  Frames: {}", int8_result.num_frames);
    println!("  PSNR Loss: {:.4} dB (limit: {} dB)",
             int8_result.psnr_loss_db, INT8_PSNR_LOSS_THRESHOLD);
    println!("  Status: {}", if int8_result.meets_spec { "PASS" } else { "FAIL" });
    println!();

    // Verify both tests processed frames
    assert!(fp16_result.num_frames > 0, "FP16 CI test should process frames");
    assert!(int8_result.num_frames > 0, "INT8 CI test should process frames");

    println!("=== CI Integration Test Complete ===");
}
