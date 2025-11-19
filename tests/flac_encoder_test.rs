//! Integration tests for FLAC encoder
//!
//! These tests verify the FLAC encoder implementation.

use zvd_lib::codec::flac::{FlacEncoder, FlacEncoderConfig};
use zvd_lib::codec::{AudioFrame, Encoder, Frame};
use zvd_lib::util::{Buffer, SampleFormat, Timestamp};

/// Test FLAC encoder creation
#[test]
fn test_flac_encoder_creation() {
    let encoder = FlacEncoder::new(44100, 2);
    assert!(encoder.is_ok(), "Should create FLAC encoder");

    let encoder = encoder.unwrap();
    assert_eq!(encoder.config().sample_rate, 44100);
    assert_eq!(encoder.config().channels, 2);
}

/// Test FLAC encoder with various sample rates
#[test]
fn test_flac_encoder_sample_rates() {
    let rates = [8000, 16000, 22050, 44100, 48000, 96000, 192000];

    for &rate in &rates {
        let encoder = FlacEncoder::new(rate, 2);
        assert!(
            encoder.is_ok(),
            "Should support sample rate {} Hz",
            rate
        );
    }
}

/// Test FLAC encoder with various channel counts
#[test]
fn test_flac_encoder_channels() {
    let channels = [1, 2, 6, 8];

    for &ch in &channels {
        let encoder = FlacEncoder::new(44100, ch);
        assert!(
            encoder.is_ok(),
            "Should support {} channels",
            ch
        );
    }
}

/// Test FLAC encoder configuration validation
#[test]
fn test_flac_encoder_config_validation() {
    // Valid config
    let config = FlacEncoderConfig::default();
    assert!(config.validate().is_ok());

    // Invalid sample rate (too high)
    let mut config = FlacEncoderConfig::default();
    config.sample_rate = 1_000_000;
    assert!(config.validate().is_err());

    // Invalid channels
    let mut config = FlacEncoderConfig::default();
    config.channels = 0;
    assert!(config.validate().is_err());

    let mut config = FlacEncoderConfig::default();
    config.channels = 9;
    assert!(config.validate().is_err());

    // Invalid compression level
    let mut config = FlacEncoderConfig::default();
    config.compression_level = 9;
    assert!(config.validate().is_err());
}

/// Test FLAC encoder compression levels
#[test]
fn test_flac_encoder_compression_levels() {
    let mut encoder = FlacEncoder::new(44100, 2).unwrap();

    // Valid compression levels
    for level in 0..=8 {
        assert!(encoder.set_compression_level(level).is_ok());
    }

    // Invalid compression level
    assert!(encoder.set_compression_level(9).is_err());
}

/// Test FLAC encoder block sizes
#[test]
fn test_flac_encoder_block_sizes() {
    let mut encoder = FlacEncoder::new(44100, 2).unwrap();

    // Valid block sizes
    assert!(encoder.set_block_size(4096).is_ok());
    assert!(encoder.set_block_size(1024).is_ok());
    assert!(encoder.set_block_size(8192).is_ok());

    // Invalid block sizes
    assert!(encoder.set_block_size(15).is_err());
    assert!(encoder.set_block_size(65536).is_err());
}

/// Test FLAC encoder with I16 samples
#[test]
fn test_flac_encoder_i16_samples() {
    let mut encoder = FlacEncoder::new(44100, 2).unwrap();

    // Create frame with I16 samples
    let mut frame = AudioFrame::new(4096, 2, SampleFormat::I16);

    // Generate sine wave test data
    let mut samples = Vec::new();
    for i in 0..4096 * 2 {
        let sample = ((i as f32 * 0.01).sin() * 16384.0) as i16;
        samples.extend_from_slice(&sample.to_le_bytes());
    }

    frame.data.push(Buffer::from_vec(samples));
    frame.pts = Timestamp::new(0);

    // Encode frame
    let result = encoder.send_frame(&Frame::Audio(frame));
    assert!(result.is_ok(), "Should encode I16 samples");

    // Flush encoder
    assert!(encoder.flush().is_ok());

    // Should have received at least header packet
    assert!(encoder.receive_packet().is_ok(), "Should have stream header");
}

/// Test FLAC encoder with F32 samples
#[test]
fn test_flac_encoder_f32_samples() {
    let mut encoder = FlacEncoder::new(48000, 2).unwrap();

    // Create frame with F32 samples
    let mut frame = AudioFrame::new(4096, 2, SampleFormat::F32);

    // Generate sine wave test data
    let mut samples = Vec::new();
    for i in 0..4096 * 2 {
        let sample = (i as f32 * 0.01).sin() * 0.5;
        samples.extend_from_slice(&sample.to_le_bytes());
    }

    frame.data.push(Buffer::from_vec(samples));
    frame.pts = Timestamp::new(0);

    // Encode frame
    let result = encoder.send_frame(&Frame::Audio(frame));
    assert!(result.is_ok(), "Should encode F32 samples");

    // Flush encoder
    assert!(encoder.flush().is_ok());

    // Should have packets
    assert!(encoder.receive_packet().is_ok());
}

/// Test FLAC encoder with multiple frames
#[test]
fn test_flac_encoder_multiple_frames() {
    let mut encoder = FlacEncoder::new(44100, 2).unwrap();

    // Encode 10 frames
    for i in 0..10 {
        let mut frame = AudioFrame::new(4096, 2, SampleFormat::I16);

        // Generate test data
        let mut samples = Vec::new();
        for j in 0..4096 * 2 {
            let sample = ((j as f32 * 0.01 + i as f32).sin() * 16384.0) as i16;
            samples.extend_from_slice(&sample.to_le_bytes());
        }

        frame.data.push(Buffer::from_vec(samples));
        frame.pts = Timestamp::new(i * 4096);

        encoder.send_frame(&Frame::Audio(frame)).expect("Failed to encode");
    }

    encoder.flush().expect("Failed to flush");

    // Should have multiple packets (header + audio frames)
    let mut packet_count = 0;
    for _ in 0..20 {
        match encoder.receive_packet() {
            Ok(_) => packet_count += 1,
            Err(_) => break,
        }
    }

    assert!(packet_count > 1, "Should have multiple packets");
}

/// Test FLAC encoder wrong channel count
#[test]
fn test_flac_encoder_wrong_channels() {
    let mut encoder = FlacEncoder::new(44100, 2).unwrap();

    // Create mono frame for stereo encoder
    let mut frame = AudioFrame::new(1024, 1, SampleFormat::I16);
    frame.data.push(Buffer::from_vec(vec![0u8; 1024 * 2]));
    frame.pts = Timestamp::new(0);

    let result = encoder.send_frame(&Frame::Audio(frame));
    assert!(result.is_err(), "Should reject wrong channel count");
}

/// Test FLAC encoder with video frame (should fail)
#[test]
fn test_flac_encoder_wrong_frame_type() {
    use zvd_lib::codec::VideoFrame;
    use zvd_lib::util::PixelFormat;

    let mut encoder = FlacEncoder::new(44100, 2).unwrap();
    let video_frame = VideoFrame::new(640, 480, PixelFormat::YUV420P);

    let result = encoder.send_frame(&Frame::Video(video_frame));
    assert!(result.is_err(), "Should reject video frames");
}

/// Test FLAC encoder flush without sending frames
#[test]
fn test_flac_encoder_flush_empty() {
    let mut encoder = FlacEncoder::new(44100, 2).unwrap();

    // Flush without encoding anything
    assert!(encoder.flush().is_ok());

    // Should still get header
    let result = encoder.receive_packet();
    assert!(result.is_ok(), "Should have stream header even when empty");
}

/// Test FLAC encoder receive without send
#[test]
fn test_flac_encoder_receive_without_send() {
    let mut encoder = FlacEncoder::new(44100, 2).unwrap();

    // Try to receive without sending
    let result = encoder.receive_packet();
    assert!(result.is_err(), "Should error when no packets available");
}

/// Test FLAC encoder with custom config
#[test]
fn test_flac_encoder_custom_config() {
    let config = FlacEncoderConfig {
        sample_rate: 96000,
        channels: 6,
        bits_per_sample: 24,
        compression_level: 8,
        block_size: 8192,
    };

    let encoder = FlacEncoder::with_config(config);
    assert!(encoder.is_ok(), "Should create encoder with custom config");

    let encoder = encoder.unwrap();
    assert_eq!(encoder.config().sample_rate, 96000);
    assert_eq!(encoder.config().channels, 6);
    assert_eq!(encoder.config().bits_per_sample, 24);
    assert_eq!(encoder.config().compression_level, 8);
}

/// Test FLAC encoder mono configuration
#[test]
fn test_flac_encoder_mono() {
    let mut encoder = FlacEncoder::new(44100, 1).unwrap();

    // Create mono frame
    let mut frame = AudioFrame::new(4096, 1, SampleFormat::I16);

    let mut samples = Vec::new();
    for i in 0..4096 {
        let sample = ((i as f32 * 0.01).sin() * 16384.0) as i16;
        samples.extend_from_slice(&sample.to_le_bytes());
    }

    frame.data.push(Buffer::from_vec(samples));
    frame.pts = Timestamp::new(0);

    encoder.send_frame(&Frame::Audio(frame)).expect("Failed to encode");
    encoder.flush().expect("Failed to flush");

    assert!(encoder.receive_packet().is_ok());
}

/// Test FLAC encoder 5.1 surround
#[test]
fn test_flac_encoder_surround() {
    let mut encoder = FlacEncoder::new(48000, 6).unwrap();

    // Create 5.1 frame
    let mut frame = AudioFrame::new(4096, 6, SampleFormat::I16);

    let mut samples = Vec::new();
    for i in 0..4096 * 6 {
        let sample = ((i as f32 * 0.01).sin() * 16384.0) as i16;
        samples.extend_from_slice(&sample.to_le_bytes());
    }

    frame.data.push(Buffer::from_vec(samples));
    frame.pts = Timestamp::new(0);

    encoder.send_frame(&Frame::Audio(frame)).expect("Failed to encode");
    encoder.flush().expect("Failed to flush");

    assert!(encoder.receive_packet().is_ok());
}

/// Test FLAC encoder with empty frame data
#[test]
fn test_flac_encoder_empty_frame_data() {
    let mut encoder = FlacEncoder::new(44100, 2).unwrap();

    // Create frame with no data
    let frame = AudioFrame::new(1024, 2, SampleFormat::I16);

    let result = encoder.send_frame(&Frame::Audio(frame));
    assert!(result.is_err(), "Should reject frame with no data");
}

/// Test FLAC encoder packet timestamps
#[test]
fn test_flac_encoder_timestamps() {
    let mut encoder = FlacEncoder::new(44100, 2).unwrap();

    // Encode frame with specific timestamp
    let mut frame = AudioFrame::new(4096, 2, SampleFormat::I16);

    let mut samples = Vec::new();
    for i in 0..4096 * 2 {
        let sample = ((i as f32 * 0.01).sin() * 16384.0) as i16;
        samples.extend_from_slice(&sample.to_le_bytes());
    }

    frame.data.push(Buffer::from_vec(samples));
    frame.pts = Timestamp::new(12345);

    encoder.send_frame(&Frame::Audio(frame)).expect("Failed to encode");
    encoder.flush().expect("Failed to flush");

    // Get header packet
    let header = encoder.receive_packet().expect("Should have header");
    assert!(header.keyframe, "Header should be marked as keyframe");

    // Get audio packet
    if let Ok(audio) = encoder.receive_packet() {
        assert!(audio.pts.is_valid(), "Timestamp should be valid");
    }
}
