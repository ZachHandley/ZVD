//! Integration tests for Vorbis encoder
//!
//! These tests verify the Vorbis encoder implementation.
//!
//! Note: For new projects, use Opus instead of Vorbis (better quality, lower latency).
//! Vorbis is provided for compatibility with existing workflows.

use zvd_lib::codec::vorbis::{VorbisEncoder, VorbisEncoderConfig};
use zvd_lib::codec::{AudioFrame, Encoder, Frame};
use zvd_lib::util::{Buffer, SampleFormat, Timestamp};

/// Test Vorbis encoder creation
#[test]
fn test_vorbis_encoder_creation() {
    let encoder = VorbisEncoder::new(48000, 2);
    assert!(encoder.is_ok(), "Should create Vorbis encoder");

    let encoder = encoder.unwrap();
    assert_eq!(encoder.config().sample_rate, 48000);
    assert_eq!(encoder.config().channels, 2);
}

/// Test Vorbis encoder with various sample rates
#[test]
fn test_vorbis_encoder_sample_rates() {
    let rates = [8000, 16000, 22050, 44100, 48000, 96000];

    for &rate in &rates {
        let encoder = VorbisEncoder::new(rate, 2);
        assert!(
            encoder.is_ok(),
            "Should support sample rate {} Hz",
            rate
        );
    }
}

/// Test Vorbis encoder with various channel counts
#[test]
fn test_vorbis_encoder_channels() {
    let channels = [1, 2, 6, 8];

    for &ch in &channels {
        let encoder = VorbisEncoder::new(48000, ch);
        assert!(
            encoder.is_ok(),
            "Should support {} channels",
            ch
        );
    }
}

/// Test Vorbis encoder configuration validation
#[test]
fn test_vorbis_encoder_config_validation() {
    // Valid config
    let config = VorbisEncoderConfig::default();
    assert!(config.validate().is_ok());

    // Invalid sample rate (too low)
    let mut config = VorbisEncoderConfig::default();
    config.sample_rate = 7999;
    assert!(config.validate().is_err());

    // Invalid sample rate (too high)
    let mut config = VorbisEncoderConfig::default();
    config.sample_rate = 200_000;
    assert!(config.validate().is_err());

    // Invalid channels
    let mut config = VorbisEncoderConfig::default();
    config.channels = 0;
    assert!(config.validate().is_err());

    // Invalid quality
    let mut config = VorbisEncoderConfig::default();
    config.quality = -2.0;
    assert!(config.validate().is_err());

    let mut config = VorbisEncoderConfig::default();
    config.quality = 11.0;
    assert!(config.validate().is_err());
}

/// Test Vorbis encoder quality levels
#[test]
fn test_vorbis_encoder_quality_levels() {
    let mut encoder = VorbisEncoder::new(48000, 2).unwrap();

    // Valid quality levels
    assert!(encoder.set_quality(-1.0).is_ok()); // ~45 kbps
    assert!(encoder.set_quality(0.0).is_ok());  // ~64 kbps
    assert!(encoder.set_quality(3.0).is_ok());  // ~112 kbps
    assert!(encoder.set_quality(5.0).is_ok());  // ~160 kbps
    assert!(encoder.set_quality(8.0).is_ok());  // ~256 kbps
    assert!(encoder.set_quality(10.0).is_ok()); // ~500 kbps

    // Invalid quality levels
    assert!(encoder.set_quality(-2.0).is_err());
    assert!(encoder.set_quality(11.0).is_err());
}

/// Test Vorbis encoder bitrate settings
#[test]
fn test_vorbis_encoder_bitrate() {
    let mut encoder = VorbisEncoder::new(48000, 2).unwrap();

    // Valid bitrates
    assert!(encoder.set_bitrate(64_000).is_ok());
    assert!(encoder.set_bitrate(128_000).is_ok());
    assert!(encoder.set_bitrate(192_000).is_ok());
    assert!(encoder.set_bitrate(256_000).is_ok());

    // Invalid bitrates
    assert!(encoder.set_bitrate(10_000).is_err());
    assert!(encoder.set_bitrate(1_000_000).is_err());
}

/// Test Vorbis encoder headers
#[test]
fn test_vorbis_encoder_headers() {
    let mut encoder = VorbisEncoder::new(48000, 2).unwrap();

    // Flush to generate headers
    assert!(encoder.flush().is_ok());

    // Should receive 3 header packets
    let header1 = encoder.receive_packet();
    assert!(header1.is_ok(), "Should have identification header");
    let header1 = header1.unwrap();
    assert!(header1.keyframe, "Headers should be keyframes");
    // Check for Vorbis identification header signature
    assert_eq!(header1.data.as_slice()[0], 1); // Packet type
    assert!(header1.data.as_slice()[1..7].starts_with(b"vorbis"));

    let header2 = encoder.receive_packet();
    assert!(header2.is_ok(), "Should have comment header");
    let header2 = header2.unwrap();
    assert_eq!(header2.data.as_slice()[0], 3); // Comment packet type

    let header3 = encoder.receive_packet();
    assert!(header3.is_ok(), "Should have setup header");
    let header3 = header3.unwrap();
    assert_eq!(header3.data.as_slice()[0], 5); // Setup packet type
}

/// Test Vorbis encoder with F32 samples
#[test]
fn test_vorbis_encoder_f32_samples() {
    let mut encoder = VorbisEncoder::new(48000, 2).unwrap();

    // Create frame with F32 samples
    let mut frame = AudioFrame::new(1024, 2, SampleFormat::F32);

    // Generate sine wave test data
    let mut samples = Vec::new();
    for i in 0..1024 * 2 {
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

    // Should have header packets
    assert!(encoder.receive_packet().is_ok());
}

/// Test Vorbis encoder with I16 samples
#[test]
fn test_vorbis_encoder_i16_samples() {
    let mut encoder = VorbisEncoder::new(44100, 2).unwrap();

    // Create frame with I16 samples
    let mut frame = AudioFrame::new(1024, 2, SampleFormat::I16);

    // Generate test data
    let mut samples = Vec::new();
    for i in 0..1024 * 2 {
        let sample = ((i as f32 * 0.01).sin() * 16384.0) as i16;
        samples.extend_from_slice(&sample.to_le_bytes());
    }

    frame.data.push(Buffer::from_vec(samples));
    frame.pts = Timestamp::new(0);

    // Encode frame
    let result = encoder.send_frame(&Frame::Audio(frame));
    assert!(result.is_ok(), "Should encode I16 samples");

    encoder.flush().expect("Failed to flush");
    assert!(encoder.receive_packet().is_ok());
}

/// Test Vorbis encoder with multiple frames
#[test]
fn test_vorbis_encoder_multiple_frames() {
    let mut encoder = VorbisEncoder::new(48000, 2).unwrap();

    // Encode 5 frames
    for i in 0..5 {
        let mut frame = AudioFrame::new(1024, 2, SampleFormat::F32);

        let mut samples = Vec::new();
        for j in 0..1024 * 2 {
            let sample = ((j as f32 * 0.01 + i as f32).sin() * 0.5) as f32;
            samples.extend_from_slice(&sample.to_le_bytes());
        }

        frame.data.push(Buffer::from_vec(samples));
        frame.pts = Timestamp::new(i * 1024);

        encoder.send_frame(&Frame::Audio(frame)).expect("Failed to encode");
    }

    encoder.flush().expect("Failed to flush");

    // Should have multiple packets (headers + audio frames)
    let mut packet_count = 0;
    for _ in 0..10 {
        match encoder.receive_packet() {
            Ok(_) => packet_count += 1,
            Err(_) => break,
        }
    }

    assert!(packet_count >= 3, "Should have at least 3 packets (headers)");
}

/// Test Vorbis encoder wrong channel count
#[test]
fn test_vorbis_encoder_wrong_channels() {
    let mut encoder = VorbisEncoder::new(48000, 2).unwrap();

    // Create mono frame for stereo encoder
    let mut frame = AudioFrame::new(1024, 1, SampleFormat::F32);
    let samples = vec![0u8; 1024 * 4]; // F32 is 4 bytes per sample
    frame.data.push(Buffer::from_vec(samples));
    frame.pts = Timestamp::new(0);

    let result = encoder.send_frame(&Frame::Audio(frame));
    assert!(result.is_err(), "Should reject wrong channel count");
}

/// Test Vorbis encoder with video frame (should fail)
#[test]
fn test_vorbis_encoder_wrong_frame_type() {
    use zvd_lib::codec::VideoFrame;
    use zvd_lib::util::PixelFormat;

    let mut encoder = VorbisEncoder::new(48000, 2).unwrap();
    let video_frame = VideoFrame::new(640, 480, PixelFormat::YUV420P);

    let result = encoder.send_frame(&Frame::Video(video_frame));
    assert!(result.is_err(), "Should reject video frames");
}

/// Test Vorbis encoder mono configuration
#[test]
fn test_vorbis_encoder_mono() {
    let mut encoder = VorbisEncoder::new(48000, 1).unwrap();

    // Create mono frame
    let mut frame = AudioFrame::new(1024, 1, SampleFormat::F32);

    let mut samples = Vec::new();
    for i in 0..1024 {
        let sample = (i as f32 * 0.01).sin() * 0.5;
        samples.extend_from_slice(&sample.to_le_bytes());
    }

    frame.data.push(Buffer::from_vec(samples));
    frame.pts = Timestamp::new(0);

    encoder.send_frame(&Frame::Audio(frame)).expect("Failed to encode");
    encoder.flush().expect("Failed to flush");

    assert!(encoder.receive_packet().is_ok());
}

/// Test Vorbis encoder surround sound
#[test]
fn test_vorbis_encoder_surround() {
    let mut encoder = VorbisEncoder::new(48000, 6).unwrap();

    // Create 5.1 frame
    let mut frame = AudioFrame::new(1024, 6, SampleFormat::F32);

    let mut samples = Vec::new();
    for i in 0..1024 * 6 {
        let sample = (i as f32 * 0.01).sin() * 0.5;
        samples.extend_from_slice(&sample.to_le_bytes());
    }

    frame.data.push(Buffer::from_vec(samples));
    frame.pts = Timestamp::new(0);

    encoder.send_frame(&Frame::Audio(frame)).expect("Failed to encode");
    encoder.flush().expect("Failed to flush");

    assert!(encoder.receive_packet().is_ok());
}

/// Test Vorbis encoder with custom config
#[test]
fn test_vorbis_encoder_custom_config() {
    let config = VorbisEncoderConfig {
        sample_rate: 96000,
        channels: 2,
        quality: 8.0,
        bitrate: Some(256_000),
    };

    let encoder = VorbisEncoder::with_config(config);
    assert!(encoder.is_ok(), "Should create encoder with custom config");

    let encoder = encoder.unwrap();
    assert_eq!(encoder.config().sample_rate, 96000);
    assert_eq!(encoder.config().channels, 2);
    assert_eq!(encoder.config().quality, 8.0);
    assert_eq!(encoder.config().bitrate, Some(256_000));
}

/// Test Vorbis encoder receive without send
#[test]
fn test_vorbis_encoder_receive_without_send() {
    let mut encoder = VorbisEncoder::new(48000, 2).unwrap();

    // Try to receive without sending (before flush)
    let result = encoder.receive_packet();
    assert!(result.is_err(), "Should error when no packets available");
}

/// Test Vorbis encoder flush without sending frames
#[test]
fn test_vorbis_encoder_flush_empty() {
    let mut encoder = VorbisEncoder::new(48000, 2).unwrap();

    // Flush without encoding anything
    assert!(encoder.flush().is_ok());

    // Should still get headers
    let result = encoder.receive_packet();
    assert!(result.is_ok(), "Should have headers even when empty");
}

/// Test Vorbis encoder different quality levels produce different configs
#[test]
fn test_vorbis_encoder_quality_affects_config() {
    let mut encoder_low = VorbisEncoder::new(48000, 2).unwrap();
    encoder_low.set_quality(-1.0).unwrap();

    let mut encoder_high = VorbisEncoder::new(48000, 2).unwrap();
    encoder_high.set_quality(10.0).unwrap();

    assert_ne!(
        encoder_low.config().quality,
        encoder_high.config().quality,
        "Different quality settings should be reflected in config"
    );
}
