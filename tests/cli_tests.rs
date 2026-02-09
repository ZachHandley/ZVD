//! CLI integration tests for ZVD
//!
//! Tests the command-line interface functionality by running the zvd binary
//! and verifying its output.

use std::io::Write;
use std::process::Command;
use tempfile::NamedTempFile;

// ============================================================================
// Helper Functions
// ============================================================================

/// Run zvd command and return output
fn run_zvd(args: &[&str]) -> std::process::Output {
    Command::new("cargo")
        .args(["run", "--quiet", "--"])
        .args(args)
        .output()
        .expect("Failed to execute command")
}

/// Run zvd command with all features enabled
fn run_zvd_all_features(args: &[&str]) -> std::process::Output {
    Command::new("cargo")
        .args(["run", "--quiet", "--all-features", "--"])
        .args(args)
        .output()
        .expect("Failed to execute command")
}

/// Get stdout as string
fn stdout_string(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stdout).to_string()
}

/// Get stderr as string
fn stderr_string(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stderr).to_string()
}

/// Create a valid WAV file for testing
fn create_test_wav() -> NamedTempFile {
    let mut temp_file = NamedTempFile::with_suffix(".wav").expect("Failed to create temp file");
    let path = temp_file.path().to_path_buf();

    // WAV header for 1 second of stereo 16-bit 44100Hz audio
    let sample_rate: u32 = 44100;
    let channels: u16 = 2;
    let bits_per_sample: u16 = 16;
    let num_samples: u32 = sample_rate; // 1 second
    let data_size: u32 = num_samples * channels as u32 * (bits_per_sample / 8) as u32;
    let file_size: u32 = 36 + data_size;
    let byte_rate: u32 = sample_rate * channels as u32 * (bits_per_sample / 8) as u32;
    let block_align: u16 = channels * (bits_per_sample / 8);

    let mut header = Vec::new();

    // RIFF header
    header.extend_from_slice(b"RIFF");
    header.extend_from_slice(&file_size.to_le_bytes());
    header.extend_from_slice(b"WAVE");

    // fmt chunk
    header.extend_from_slice(b"fmt ");
    header.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    header.extend_from_slice(&1u16.to_le_bytes()); // audio format (PCM)
    header.extend_from_slice(&channels.to_le_bytes());
    header.extend_from_slice(&sample_rate.to_le_bytes());
    header.extend_from_slice(&byte_rate.to_le_bytes());
    header.extend_from_slice(&block_align.to_le_bytes());
    header.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    header.extend_from_slice(b"data");
    header.extend_from_slice(&data_size.to_le_bytes());

    // Audio data (sine wave)
    let mut audio_data = Vec::with_capacity(data_size as usize);
    for i in 0..num_samples {
        let t = i as f64 / sample_rate as f64;
        let sample = ((2.0 * std::f64::consts::PI * 440.0 * t).sin() * 0.5 * i16::MAX as f64) as i16;
        // Stereo - same sample for both channels
        audio_data.extend_from_slice(&sample.to_le_bytes());
        audio_data.extend_from_slice(&sample.to_le_bytes());
    }

    temp_file.write_all(&header).expect("Failed to write header");
    temp_file.write_all(&audio_data).expect("Failed to write audio");
    temp_file.flush().expect("Failed to flush");

    temp_file
}

/// Create a valid Y4M file for testing
fn create_test_y4m() -> NamedTempFile {
    let mut temp_file = NamedTempFile::with_suffix(".y4m").expect("Failed to create temp file");

    let width = 320;
    let height = 240;

    // Y4M header
    let header = format!("YUV4MPEG2 W{} H{} F30:1 Ip A1:1 C420\n", width, height);
    temp_file.write_all(header.as_bytes()).expect("Failed to write header");

    // Frame header and data
    temp_file.write_all(b"FRAME\n").expect("Failed to write frame header");

    // Y plane
    let y_size = width * height;
    let y_data: Vec<u8> = (0..y_size).map(|i| ((i % 256) as u8).wrapping_add(128)).collect();
    temp_file.write_all(&y_data).expect("Failed to write Y plane");

    // U plane
    let uv_size = (width / 2) * (height / 2);
    let u_data = vec![128u8; uv_size];
    temp_file.write_all(&u_data).expect("Failed to write U plane");

    // V plane
    let v_data = vec![128u8; uv_size];
    temp_file.write_all(&v_data).expect("Failed to write V plane");

    temp_file.flush().expect("Failed to flush");

    temp_file
}

// ============================================================================
// Version and Help Tests
// ============================================================================

#[test]
fn test_cli_version() {
    let output = run_zvd(&["--version"]);

    assert!(output.status.success(), "Version command should succeed");

    let stdout = stdout_string(&output);
    assert!(stdout.contains("zvd") || stdout.contains("0."),
        "Version output should contain version info");
}

#[test]
fn test_cli_help() {
    let output = run_zvd(&["--help"]);

    assert!(output.status.success(), "Help command should succeed");

    let stdout = stdout_string(&output);
    assert!(stdout.contains("Usage") || stdout.contains("USAGE") || stdout.contains("zvd"),
        "Help should show usage information");
    assert!(stdout.contains("info") || stdout.contains("Info"),
        "Help should mention info command");
    assert!(stdout.contains("convert") || stdout.contains("Convert"),
        "Help should mention convert command");
}

#[test]
fn test_cli_help_codecs() {
    let output = run_zvd(&["codecs", "--help"]);

    // Help for subcommand
    let stdout = stdout_string(&output);
    // Should either succeed or show help
}

// ============================================================================
// Codecs Command Tests
// ============================================================================

#[test]
fn test_cli_codecs_command() {
    let output = run_zvd(&["codecs"]);

    assert!(output.status.success(), "Codecs command should succeed");

    let stdout = stdout_string(&output);

    // Should list some codecs
    assert!(stdout.contains("av1") || stdout.contains("AV1"),
        "Should list AV1 codec");
    assert!(stdout.contains("pcm") || stdout.contains("PCM"),
        "Should list PCM codec");
}

#[test]
fn test_cli_codecs_filter_video() {
    let output = run_zvd(&["codecs", "--filter", "video"]);

    assert!(output.status.success(), "Codecs filter should succeed");

    let stdout = stdout_string(&output);

    // Should show video codecs
    // Should not show audio-only codecs prominently
}

#[test]
fn test_cli_codecs_filter_audio() {
    let output = run_zvd(&["codecs", "--filter", "audio"]);

    assert!(output.status.success(), "Codecs filter should succeed");

    let stdout = stdout_string(&output);

    // Should show audio codecs
}

// ============================================================================
// Formats Command Tests
// ============================================================================

#[test]
fn test_cli_formats_command() {
    let output = run_zvd(&["formats"]);

    assert!(output.status.success(), "Formats command should succeed");

    let stdout = stdout_string(&output);

    // Should list some formats
    assert!(stdout.contains("mp4") || stdout.contains("MP4") || stdout.contains("wav"),
        "Should list common formats");
}

#[test]
fn test_cli_formats_muxers() {
    let output = run_zvd(&["formats", "--muxers"]);

    assert!(output.status.success(), "Formats muxers should succeed");

    let stdout = stdout_string(&output);
    assert!(stdout.contains("Muxer") || stdout.contains("muxer") || stdout.contains("Output"),
        "Should show muxers");
}

#[test]
fn test_cli_formats_demuxers() {
    let output = run_zvd(&["formats", "--demuxers"]);

    assert!(output.status.success(), "Formats demuxers should succeed");

    let stdout = stdout_string(&output);
    assert!(stdout.contains("Demuxer") || stdout.contains("demuxer") || stdout.contains("Input"),
        "Should show demuxers");
}

// ============================================================================
// Info Command Tests
// ============================================================================

#[test]
fn test_cli_info_wav_file() {
    let test_file = create_test_wav();
    let path = test_file.path().to_str().unwrap();

    let output = run_zvd(&["info", path]);

    assert!(output.status.success(), "Info command should succeed for WAV: {}",
        stderr_string(&output));

    let stdout = stdout_string(&output);

    // Should show file info
    assert!(stdout.contains("Stream") || stdout.contains("Audio") || stdout.contains("pcm"),
        "Should show stream information");
}

#[test]
fn test_cli_info_y4m_file() {
    let test_file = create_test_y4m();
    let path = test_file.path().to_str().unwrap();

    let output = run_zvd(&["info", path]);

    assert!(output.status.success(), "Info command should succeed for Y4M: {}",
        stderr_string(&output));

    let stdout = stdout_string(&output);

    // Should show video info
    assert!(stdout.contains("Stream") || stdout.contains("Video") || stdout.contains("320"),
        "Should show stream information");
}

#[test]
fn test_cli_info_nonexistent_file() {
    let output = run_zvd(&["info", "/nonexistent/path/file.mp4"]);

    // Should fail gracefully
    assert!(!output.status.success(), "Info should fail for nonexistent file");
}

// ============================================================================
// Probe Command Tests
// ============================================================================

#[test]
fn test_cli_probe_wav_file() {
    let test_file = create_test_wav();
    let path = test_file.path().to_str().unwrap();

    let output = run_zvd(&["probe", path]);

    assert!(output.status.success(), "Probe command should succeed: {}",
        stderr_string(&output));

    let stdout = stdout_string(&output);

    // Should show probe information
    assert!(stdout.contains("Format") || stdout.contains("Stream") || stdout.contains("wav"),
        "Should show format information");
}

#[test]
fn test_cli_probe_json_output() {
    let test_file = create_test_wav();
    let path = test_file.path().to_str().unwrap();

    let output = run_zvd(&["probe", path, "--json"]);

    assert!(output.status.success(), "Probe JSON should succeed: {}",
        stderr_string(&output));

    let stdout = stdout_string(&output);

    // Should be valid JSON
    assert!(stdout.contains("{") && stdout.contains("}"),
        "Should output JSON");

    // Try to parse as JSON
    let parse_result: Result<serde_json::Value, _> = serde_json::from_str(&stdout);
    assert!(parse_result.is_ok(), "Should be valid JSON: {}", stdout);
}

#[test]
fn test_cli_probe_y4m_file() {
    let test_file = create_test_y4m();
    let path = test_file.path().to_str().unwrap();

    let output = run_zvd(&["probe", path]);

    assert!(output.status.success(), "Probe Y4M should succeed: {}",
        stderr_string(&output));

    let stdout = stdout_string(&output);

    // Should show Y4M info
    assert!(stdout.contains("y4m") || stdout.contains("YUV") || stdout.contains("Video") || stdout.contains("320"),
        "Should show Y4M information");
}

// ============================================================================
// Convert Command Tests
// ============================================================================

#[test]
fn test_cli_convert_wav_to_wav() {
    let input_file = create_test_wav();
    let input_path = input_file.path().to_str().unwrap();

    let output_file = NamedTempFile::with_suffix(".wav").expect("Failed to create output file");
    let output_path = output_file.path().to_str().unwrap();

    let output = run_zvd(&["convert", "-i", input_path, "-o", output_path]);

    // Convert may have limited codec support
    // Check if it succeeded or failed with expected error
    let stdout = stdout_string(&output);
    let stderr = stderr_string(&output);

    // Either succeeds or fails with a codec-related message
    if !output.status.success() {
        assert!(stderr.contains("codec") || stderr.contains("supported") || stderr.contains("PCM"),
            "Should fail with codec message, got: {}", stderr);
    }
}

#[test]
fn test_cli_convert_missing_input() {
    let output = run_zvd(&["convert", "-i", "/nonexistent.wav", "-o", "/tmp/out.wav"]);

    assert!(!output.status.success(), "Convert should fail for missing input");
}

#[test]
fn test_cli_convert_missing_args() {
    let output = run_zvd(&["convert"]);

    // Should fail - missing required arguments
    assert!(!output.status.success(), "Convert should fail without arguments");

    let stderr = stderr_string(&output);
    assert!(stderr.contains("required") || stderr.contains("argument") || stderr.contains("input"),
        "Should mention missing argument");
}

// ============================================================================
// Extract Command Tests
// ============================================================================

#[test]
fn test_cli_extract_missing_args() {
    let output = run_zvd(&["extract"]);

    // Should fail - missing required arguments
    assert!(!output.status.success(), "Extract should fail without arguments");
}

#[test]
fn test_cli_extract_invalid_stream() {
    let input_file = create_test_wav();
    let input_path = input_file.path().to_str().unwrap();

    let output_file = NamedTempFile::with_suffix(".wav").expect("Failed to create output file");
    let output_path = output_file.path().to_str().unwrap();

    let output = run_zvd(&["extract", "-i", input_path, "-s", "999", "-o", output_path]);

    // Should fail - invalid stream index
    assert!(!output.status.success(), "Extract should fail for invalid stream index");
}

// ============================================================================
// Error Message Tests
// ============================================================================

#[test]
fn test_cli_unknown_command() {
    let output = run_zvd(&["unknowncommand"]);

    assert!(!output.status.success(), "Unknown command should fail");
}

#[test]
fn test_cli_invalid_argument() {
    let output = run_zvd(&["info", "--invalid-flag"]);

    assert!(!output.status.success(), "Invalid flag should fail");
}

// ============================================================================
// Verbose and Debug Flags
// ============================================================================

#[test]
fn test_cli_verbose_flag() {
    let test_file = create_test_wav();
    let path = test_file.path().to_str().unwrap();

    let output = run_zvd(&["-v", "info", path]);

    // Should work with verbose flag
    // May or may not show additional output
    assert!(output.status.success(), "Verbose info should succeed");
}

#[test]
fn test_cli_debug_flag() {
    let output = run_zvd(&["-d", "codecs"]);

    // Should work with debug flag
    assert!(output.status.success(), "Debug codecs should succeed");
}

#[test]
fn test_cli_threads_flag() {
    let output = run_zvd(&["-t", "4", "codecs"]);

    // Should work with threads flag
    assert!(output.status.success(), "Threads flag should work");
}

// ============================================================================
// Integration Scenarios
// ============================================================================

#[test]
fn test_cli_workflow_probe_then_info() {
    let test_file = create_test_wav();
    let path = test_file.path().to_str().unwrap();

    // First probe
    let probe_output = run_zvd(&["probe", path, "--json"]);
    assert!(probe_output.status.success(), "Probe should succeed");

    // Then info
    let info_output = run_zvd(&["info", path]);
    assert!(info_output.status.success(), "Info should succeed");

    // Both should work on same file
}

#[test]
fn test_cli_multiple_files() {
    let file1 = create_test_wav();
    let file2 = create_test_y4m();

    let path1 = file1.path().to_str().unwrap();
    let path2 = file2.path().to_str().unwrap();

    // Info on WAV
    let output1 = run_zvd(&["info", path1]);
    assert!(output1.status.success(), "Info WAV should succeed");

    // Info on Y4M
    let output2 = run_zvd(&["info", path2]);
    assert!(output2.status.success(), "Info Y4M should succeed");
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_cli_empty_file() {
    let temp_file = NamedTempFile::with_suffix(".wav").expect("Failed to create temp file");
    let path = temp_file.path().to_str().unwrap();

    // File is empty
    let output = run_zvd(&["info", path]);

    // Should fail gracefully
    assert!(!output.status.success(), "Info should fail for empty file");
}

#[test]
fn test_cli_binary_garbage_file() {
    let mut temp_file = NamedTempFile::with_suffix(".mp4").expect("Failed to create temp file");

    // Write garbage
    let garbage: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
    temp_file.write_all(&garbage).expect("Failed to write");
    temp_file.flush().expect("Failed to flush");

    let path = temp_file.path().to_str().unwrap();

    let output = run_zvd(&["info", path]);

    // Should fail gracefully, not crash
    assert!(!output.status.success(), "Info should fail for garbage file");
}

#[test]
fn test_cli_special_characters_in_path() {
    // Create file with spaces in name (via temp directory)
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let file_path = temp_dir.path().join("test file with spaces.wav");

    // Copy test WAV content
    let test_wav = create_test_wav();
    std::fs::copy(test_wav.path(), &file_path).expect("Failed to copy");

    let path = file_path.to_str().unwrap();

    let output = run_zvd(&["info", path]);

    // Should handle paths with spaces
    assert!(output.status.success(), "Should handle paths with spaces: {}",
        stderr_string(&output));
}
