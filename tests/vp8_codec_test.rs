//! Comprehensive integration tests for VP8 encoder and decoder
//!
//! These tests verify that the libvpx VP8 encoder and decoder work together
//! correctly, testing round-trip encoding/decoding with various configurations.

#![cfg(feature = "vp8-codec")]

use zvd_lib::codec::vp8::{RateControlMode, Vp8Decoder, Vp8Encoder, Vp8EncoderConfig};
use zvd_lib::codec::{Decoder, Encoder, Frame};
use zvd_lib::error::{Error, Result};
use zvd_lib::format::Packet;
use zvd_lib::util::{Buffer, PixelFormat, Rational, Timestamp};

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a test video frame with specified dimensions and fill pattern
/// VP8 only supports YUV420P
fn create_test_frame(
    width: u32,
    height: u32,
    pts: i64,
    fill_pattern: u8,
) -> zvd_lib::codec::frame::VideoFrame {
    use zvd_lib::codec::frame::VideoFrame;

    let mut frame = VideoFrame::new(width, height, PixelFormat::YUV420P);
    frame.pts = Timestamp::new(pts);
    frame.keyframe = pts == 0;

    // Y plane (full resolution)
    let y_size = (width * height) as usize;
    frame
        .data
        .push(Buffer::from_vec(vec![fill_pattern; y_size]));
    frame.linesize.push(width as usize);

    // U plane (half width, half height)
    let uv_width = (width / 2) as usize;
    let uv_height = (height / 2) as usize;
    let uv_size = uv_width * uv_height;
    frame.data.push(Buffer::from_vec(vec![128u8; uv_size]));
    frame.linesize.push(uv_width);

    // V plane (half width, half height)
    frame.data.push(Buffer::from_vec(vec![128u8; uv_size]));
    frame.linesize.push(uv_width);

    frame
}

/// Verify that a decoded frame matches expected dimensions
fn verify_frame_properties(
    frame: &zvd_lib::codec::frame::VideoFrame,
    expected_width: u32,
    expected_height: u32,
) -> Result<()> {
    if frame.width != expected_width {
        return Err(Error::codec(format!(
            "Width mismatch: expected {}, got {}",
            expected_width, frame.width
        )));
    }

    if frame.height != expected_height {
        return Err(Error::codec(format!(
            "Height mismatch: expected {}, got {}",
            expected_height, frame.height
        )));
    }

    // VP8 outputs YUV420P
    if frame.format != PixelFormat::YUV420P {
        return Err(Error::codec(format!(
            "Format mismatch: expected YUV420P, got {:?}",
            frame.format
        )));
    }

    Ok(())
}

/// Encode multiple frames and collect all packets
fn encode_frames(config: Vp8EncoderConfig, frames: Vec<Frame>) -> Result<Vec<Packet>> {
    let mut encoder = Vp8Encoder::with_config(config)?;
    let mut packets = Vec::new();

    // Send all frames
    for frame in &frames {
        encoder.send_frame(frame)?;

        // Collect any available packets
        loop {
            match encoder.receive_packet() {
                Ok(packet) => packets.push(packet),
                Err(Error::TryAgain) => break,
                Err(e) => return Err(e),
            }
        }
    }

    // Flush encoder
    encoder.flush()?;

    // Collect remaining packets
    loop {
        match encoder.receive_packet() {
            Ok(packet) => packets.push(packet),
            Err(Error::TryAgain) => break,
            Err(e) => return Err(e),
        }
    }

    Ok(packets)
}

/// Decode all packets and collect frames
fn decode_packets(packets: Vec<Packet>) -> Result<Vec<zvd_lib::codec::frame::VideoFrame>> {
    let mut decoder = Vp8Decoder::new()?;
    let mut frames = Vec::new();

    for packet in &packets {
        decoder.send_packet(packet)?;

        // Collect any available frames
        loop {
            match decoder.receive_frame() {
                Ok(Frame::Video(frame)) => frames.push(frame),
                Ok(Frame::Audio(_)) => {
                    return Err(Error::codec("Unexpected audio frame from VP8 decoder"))
                }
                Err(Error::TryAgain) => break,
                Err(e) => return Err(e),
            }
        }
    }

    // Flush decoder
    decoder.flush()?;

    Ok(frames)
}

// ============================================================================
// Unit Tests - Encoder
// ============================================================================

#[test]
fn test_vp8_encoder_creation() {
    let encoder = Vp8Encoder::new(640, 480);
    assert!(
        encoder.is_ok(),
        "Failed to create VP8 encoder. Make sure libvpx is installed."
    );
}

#[test]
fn test_vp8_encoder_with_config() {
    let config = Vp8EncoderConfig {
        width: 1920,
        height: 1080,
        bitrate: 5_000_000,
        framerate: Rational::new(60, 1),
        keyframe_interval: 120,
        threads: 4,
        rc_mode: RateControlMode::VBR,
        quality: 20,
    };

    let encoder = Vp8Encoder::with_config(config);
    assert!(
        encoder.is_ok(),
        "Failed to create VP8 encoder with custom config"
    );
}

#[test]
fn test_vp8_encoder_different_rate_control_modes() {
    let modes = vec![
        RateControlMode::VBR,
        RateControlMode::CBR,
        RateControlMode::CQ,
    ];

    for mode in modes {
        let config = Vp8EncoderConfig {
            width: 640,
            height: 480,
            bitrate: 1_000_000,
            framerate: Rational::new(30, 1),
            keyframe_interval: 60,
            threads: 0,
            rc_mode: mode,
            quality: 20,
        };

        let encoder = Vp8Encoder::with_config(config);
        assert!(
            encoder.is_ok(),
            "Failed to create encoder with rate control mode {:?}",
            mode
        );
    }
}

#[test]
fn test_vp8_encoder_flush() {
    let mut encoder = Vp8Encoder::new(640, 480).expect("Failed to create encoder");
    assert!(encoder.flush().is_ok(), "Flush should not error");
}

// ============================================================================
// Unit Tests - Decoder
// ============================================================================

#[test]
fn test_vp8_decoder_creation() {
    let decoder = Vp8Decoder::new();
    assert!(
        decoder.is_ok(),
        "Failed to create VP8 decoder. Make sure libvpx is installed."
    );
}

#[test]
fn test_vp8_decoder_with_threads() {
    let decoder = Vp8Decoder::with_threads(4);
    assert!(decoder.is_ok(), "Failed to create VP8 decoder with threads");
}

#[test]
fn test_vp8_decoder_flush() {
    let mut decoder = Vp8Decoder::new().expect("Failed to create decoder");
    assert!(decoder.flush().is_ok(), "Flush should not error");
}

// ============================================================================
// Integration Tests - Round-trip Encoding/Decoding
// ============================================================================

#[test]
fn test_vp8_roundtrip_single_frame_640x480() {
    let width = 640;
    let height = 480;

    // Create a test frame
    let test_frame = create_test_frame(width, height, 0, 100);
    let frames = vec![Frame::Video(test_frame)];

    // Encode
    let config = Vp8EncoderConfig {
        width,
        height,
        bitrate: 1_000_000,
        framerate: Rational::new(30, 1),
        keyframe_interval: 1,
        threads: 0,
        rc_mode: RateControlMode::VBR,
        quality: 20,
    };

    let packets = encode_frames(config, frames).expect("Encoding failed");
    assert!(!packets.is_empty(), "No packets generated");
    assert!(packets[0].is_keyframe, "First packet should be keyframe");

    // Decode
    let decoded_frames = decode_packets(packets).expect("Decoding failed");
    assert_eq!(decoded_frames.len(), 1, "Should decode exactly 1 frame");

    // Verify
    verify_frame_properties(&decoded_frames[0], width, height).expect("Frame verification failed");
}

#[test]
fn test_vp8_roundtrip_multiple_frames() {
    let width = 320;
    let height = 240;
    let frame_count = 10;

    // Create test frames with different patterns
    let mut frames = Vec::new();
    for i in 0..frame_count {
        let pattern = (i * 25) as u8;
        frames.push(Frame::Video(create_test_frame(width, height, i, pattern)));
    }

    // Encode
    let config = Vp8EncoderConfig {
        width,
        height,
        bitrate: 500_000,
        framerate: Rational::new(30, 1),
        keyframe_interval: 30,
        threads: 0,
        rc_mode: RateControlMode::VBR,
        quality: 25,
    };

    let packets = encode_frames(config, frames).expect("Encoding failed");
    assert!(
        !packets.is_empty(),
        "No packets generated from {} frames",
        frame_count
    );

    // Verify at least one keyframe
    let keyframe_count = packets.iter().filter(|p| p.is_keyframe).count();
    assert!(keyframe_count >= 1, "Should have at least one keyframe");

    // Decode
    let decoded_frames = decode_packets(packets).expect("Decoding failed");
    assert!(
        !decoded_frames.is_empty(),
        "No frames decoded from {} input frames",
        frame_count
    );

    // Verify all frames
    for frame in &decoded_frames {
        verify_frame_properties(frame, width, height).expect("Frame verification failed");
    }
}

#[test]
fn test_vp8_roundtrip_hd_resolution() {
    let width = 1920;
    let height = 1080;

    // Create a single HD frame
    let test_frame = create_test_frame(width, height, 0, 150);
    let frames = vec![Frame::Video(test_frame)];

    // Encode with higher bitrate for HD
    let config = Vp8EncoderConfig {
        width,
        height,
        bitrate: 5_000_000,
        framerate: Rational::new(30, 1),
        keyframe_interval: 1,
        threads: 4, // Use threading for HD
        rc_mode: RateControlMode::VBR,
        quality: 15,
    };

    let packets = encode_frames(config, frames).expect("HD encoding failed");
    assert!(!packets.is_empty(), "No packets from HD encoding");

    // Decode
    let decoded_frames = decode_packets(packets).expect("HD decoding failed");
    assert_eq!(decoded_frames.len(), 1, "Should decode 1 HD frame");

    // Verify
    verify_frame_properties(&decoded_frames[0], width, height)
        .expect("HD frame verification failed");
}

#[test]
fn test_vp8_keyframe_intervals() {
    let width = 320;
    let height = 240;
    let frame_count = 30;
    let keyframe_interval = 10;

    // Create frames
    let mut frames = Vec::new();
    for i in 0..frame_count {
        frames.push(Frame::Video(create_test_frame(width, height, i, 100)));
    }

    // Encode with specific keyframe interval
    let config = Vp8EncoderConfig {
        width,
        height,
        bitrate: 500_000,
        framerate: Rational::new(30, 1),
        keyframe_interval,
        threads: 0,
        rc_mode: RateControlMode::VBR,
        quality: 25,
    };

    let packets = encode_frames(config, frames).expect("Encoding with keyframe interval failed");

    // Count keyframes
    let keyframe_count = packets.iter().filter(|p| p.is_keyframe).count();

    // Should have at least 1 keyframe (first frame)
    // May have more depending on encoder decisions
    assert!(
        keyframe_count >= 1,
        "Should have at least 1 keyframe, got {}",
        keyframe_count
    );

    // First packet should always be a keyframe
    assert!(packets[0].is_keyframe, "First packet must be a keyframe");
}

#[test]
fn test_vp8_different_qualities() {
    let width = 320;
    let height = 240;

    for quality in [10, 30, 50].iter() {
        let test_frame = create_test_frame(width, height, 0, 100);
        let frames = vec![Frame::Video(test_frame)];

        let config = Vp8EncoderConfig {
            width,
            height,
            bitrate: 500_000,
            framerate: Rational::new(30, 1),
            keyframe_interval: 1,
            threads: 0,
            rc_mode: RateControlMode::CQ,
            quality: *quality,
        };

        let packets = encode_frames(config, frames)
            .expect(&format!("Encoding with quality {} failed", quality));
        assert!(!packets.is_empty(), "No packets with quality {}", quality);

        let decoded_frames =
            decode_packets(packets).expect(&format!("Decoding quality {} failed", quality));
        assert_eq!(
            decoded_frames.len(),
            1,
            "Should decode 1 frame at quality {}",
            quality
        );
    }
}

#[test]
fn test_vp8_cbr_mode() {
    let width = 640;
    let height = 480;
    let frame_count = 5;

    let mut frames = Vec::new();
    for i in 0..frame_count {
        frames.push(Frame::Video(create_test_frame(width, height, i, 100)));
    }

    let config = Vp8EncoderConfig {
        width,
        height,
        bitrate: 1_000_000,
        framerate: Rational::new(30, 1),
        keyframe_interval: 30,
        threads: 0,
        rc_mode: RateControlMode::CBR, // Constant bitrate
        quality: 20,
    };

    let packets = encode_frames(config, frames).expect("CBR encoding failed");
    assert!(!packets.is_empty(), "No packets from CBR encoding");

    let decoded_frames = decode_packets(packets).expect("CBR decoding failed");
    assert!(!decoded_frames.is_empty(), "No frames from CBR decode");
}

#[test]
fn test_vp8_error_handling_empty_packet() {
    let mut decoder = Vp8Decoder::new().expect("Failed to create decoder");

    let empty_packet = Packet {
        data: Buffer::empty(),
        pts: Timestamp::new(0),
        dts: Timestamp::new(0),
        duration: 1,
        is_keyframe: false,
        stream_index: 0,
    };

    let result = decoder.send_packet(&empty_packet);
    assert!(result.is_err(), "Should reject empty packet");
}

#[test]
fn test_vp8_encoder_dimension_validation() {
    // Test that encoder validates frame dimensions
    let mut encoder = Vp8Encoder::new(640, 480).expect("Failed to create encoder");

    // Try to encode a frame with wrong dimensions
    let wrong_frame = create_test_frame(320, 240, 0, 100); // Wrong size
    let result = encoder.send_frame(&Frame::Video(wrong_frame));

    assert!(
        result.is_err(),
        "Should reject frame with mismatched dimensions"
    );
}

// ============================================================================
// Performance/Stress Tests
// ============================================================================

#[test]
fn test_vp8_many_frames() {
    let width = 320;
    let height = 240;
    let frame_count = 100;

    let mut frames = Vec::new();
    for i in 0..frame_count {
        frames.push(Frame::Video(create_test_frame(
            width,
            height,
            i,
            (i % 256) as u8,
        )));
    }

    let config = Vp8EncoderConfig {
        width,
        height,
        bitrate: 500_000,
        framerate: Rational::new(30, 1),
        keyframe_interval: 30,
        threads: 0,
        rc_mode: RateControlMode::VBR,
        quality: 30,
    };

    let packets = encode_frames(config, frames).expect("Failed to encode many frames");
    assert!(!packets.is_empty(), "No packets from many frames");

    let decoded_frames = decode_packets(packets).expect("Failed to decode many frames");
    assert!(
        decoded_frames.len() >= frame_count as usize / 2,
        "Decoded significantly fewer frames than expected"
    );
}
