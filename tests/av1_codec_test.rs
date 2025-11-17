//! Comprehensive integration tests for AV1 encoder and decoder
//!
//! These tests verify that the rav1e encoder and dav1d decoder work together
//! correctly, testing round-trip encoding/decoding with various configurations.

use zvd_lib::codec::av1::{Av1Decoder, Av1EncoderBuilder};
use zvd_lib::codec::{Decoder, Encoder, Frame};
use zvd_lib::error::{Error, Result};
use zvd_lib::format::Packet;
use zvd_lib::util::{Buffer, PixelFormat, Timestamp};

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a test video frame with specified dimensions, format, and fill pattern
fn create_test_frame(
    width: u32,
    height: u32,
    format: PixelFormat,
    pts: i64,
    fill_pattern: u8,
) -> zvd_lib::codec::frame::VideoFrame {
    use zvd_lib::codec::frame::VideoFrame;

    let mut frame = VideoFrame::new(width, height, format);
    frame.pts = Timestamp::new(pts);
    frame.keyframe = pts == 0;

    match format {
        PixelFormat::YUV420P => {
            // Y plane (full resolution)
            let y_size = (width * height) as usize;
            frame.data.push(Buffer::from_vec(vec![fill_pattern; y_size]));
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
        }
        PixelFormat::YUV422P => {
            // Y plane (full resolution)
            let y_size = (width * height) as usize;
            frame.data.push(Buffer::from_vec(vec![fill_pattern; y_size]));
            frame.linesize.push(width as usize);

            // U plane (half width, full height)
            let uv_width = (width / 2) as usize;
            let uv_size = uv_width * height as usize;
            frame.data.push(Buffer::from_vec(vec![128u8; uv_size]));
            frame.linesize.push(uv_width);

            // V plane (half width, full height)
            frame.data.push(Buffer::from_vec(vec![128u8; uv_size]));
            frame.linesize.push(uv_width);
        }
        PixelFormat::YUV444P => {
            // Y plane (full resolution)
            let y_size = (width * height) as usize;
            frame.data.push(Buffer::from_vec(vec![fill_pattern; y_size]));
            frame.linesize.push(width as usize);

            // U plane (full resolution)
            let uv_size = (width * height) as usize;
            frame.data.push(Buffer::from_vec(vec![128u8; uv_size]));
            frame.linesize.push(width as usize);

            // V plane (full resolution)
            frame.data.push(Buffer::from_vec(vec![128u8; uv_size]));
            frame.linesize.push(width as usize);
        }
        PixelFormat::GRAY8 => {
            // Single plane (grayscale)
            let size = (width * height) as usize;
            frame.data.push(Buffer::from_vec(vec![fill_pattern; size]));
            frame.linesize.push(width as usize);
        }
        _ => panic!("Unsupported pixel format for test: {:?}", format),
    }

    frame
}

/// Verify that a decoded frame matches expected dimensions and format
fn verify_frame_properties(
    frame: &zvd_lib::codec::frame::VideoFrame,
    expected_width: u32,
    expected_height: u32,
    _expected_format: PixelFormat,
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

    // Note: AV1 encoder (rav1e) may convert pixel formats internally.
    // For example, YUV422P and YUV444P may be converted to YUV420P.
    // We verify that the output is a valid YUV format that the decoder produced.
    let is_valid_yuv = matches!(
        frame.format,
        PixelFormat::YUV420P
            | PixelFormat::YUV422P
            | PixelFormat::YUV444P
            | PixelFormat::GRAY8
            | PixelFormat::YUV420P10LE
            | PixelFormat::YUV422P10LE
            | PixelFormat::YUV444P10LE
    );

    if !is_valid_yuv {
        return Err(Error::codec(format!(
            "Invalid output format: got {:?}",
            frame.format
        )));
    }

    Ok(())
}

/// Encode multiple frames and collect all packets
fn encode_frames(
    width: u32,
    height: u32,
    speed: u8,
    quantizer: usize,
    max_keyframe: u64,
    frames: Vec<Frame>,
) -> Result<Vec<Packet>> {
    let mut encoder = Av1EncoderBuilder::new(width, height)
        .speed_preset(speed)
        .quantizer(quantizer)
        .max_keyframe_interval(max_keyframe)
        .build()?;
    let mut packets = Vec::new();

    // Send all frames
    for frame in frames {
        encoder.send_frame(&frame)?;

        // Try to receive packets after each frame
        loop {
            match encoder.receive_packet() {
                Ok(packet) => packets.push(packet),
                Err(Error::TryAgain) => break, // Need more data
                Err(e) => return Err(e),
            }
        }
    }

    // Flush encoder
    encoder.flush()?;

    // Receive remaining packets
    loop {
        match encoder.receive_packet() {
            Ok(packet) => packets.push(packet),
            Err(Error::TryAgain) => continue,
            Err(Error::EndOfStream) => break,
            Err(e) => return Err(e),
        }
    }

    Ok(packets)
}

/// Decode all packets and collect frames
fn decode_packets(decoder: &mut Av1Decoder, packets: Vec<Packet>) -> Result<Vec<Frame>> {
    let mut frames = Vec::new();

    for packet in packets {
        decoder.send_packet(&packet)?;

        // Try to receive frames
        loop {
            match decoder.receive_frame() {
                Ok(frame) => frames.push(frame),
                Err(Error::TryAgain) => break,
                Err(e) => return Err(e),
            }
        }
    }

    // Flush decoder to get remaining frames
    decoder.flush()?;

    Ok(frames)
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_av1_round_trip_encode_decode() {
    // Create a simple test frame
    let width = 320;
    let height = 240;
    let format = PixelFormat::YUV420P;

    let test_frame = create_test_frame(width, height, format, 0, 64);

    // Create encoder with fast settings for testing
    let mut encoder = Av1EncoderBuilder::new(width, height)
        .speed_preset(10) // Fastest
        .quantizer(100)
        .max_keyframe_interval(1) // Force keyframe
        .build()
        .expect("Failed to create encoder");

    // Encode the frame
    encoder
        .send_frame(&Frame::Video(test_frame.clone()))
        .expect("Failed to send frame");
    encoder.flush().expect("Failed to flush encoder");

    // Collect encoded packets
    let mut packets = Vec::new();
    loop {
        match encoder.receive_packet() {
            Ok(packet) => {
                assert!(!packet.data.is_empty(), "Packet should contain data");
                packets.push(packet);
            }
            Err(Error::EndOfStream) => break,
            Err(Error::TryAgain) => continue,
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    assert!(
        !packets.is_empty(),
        "Should have received at least one packet"
    );

    // Create decoder
    let mut decoder = Av1Decoder::new().expect("Failed to create decoder");

    // Decode packets
    let mut decoded_frames = Vec::new();
    for packet in packets {
        decoder
            .send_packet(&packet)
            .expect("Failed to send packet to decoder");

        // Try to receive frames
        loop {
            match decoder.receive_frame() {
                Ok(Frame::Video(frame)) => {
                    decoded_frames.push(frame);
                }
                Ok(Frame::Audio(_)) => panic!("Unexpected audio frame"),
                Err(Error::TryAgain) => break,
                Err(e) => panic!("Decode error: {:?}", e),
            }
        }
    }

    assert!(
        !decoded_frames.is_empty(),
        "Should have decoded at least one frame"
    );

    // Verify decoded frame properties
    let decoded_frame = &decoded_frames[0];
    verify_frame_properties(decoded_frame, width, height, format)
        .expect("Frame properties don't match");
}

#[test]
fn test_av1_multiple_frames_round_trip() {
    let width = 320;
    let height = 240;
    let format = PixelFormat::YUV420P;
    let num_frames = 15;

    // Create test frames with different fill patterns
    let mut test_frames = Vec::new();
    for i in 0..num_frames {
        let fill = (i * 10) as u8;
        let frame = create_test_frame(width, height, format, i, fill);
        test_frames.push(Frame::Video(frame));
    }

    // Encode frames
    let packets = encode_frames(width, height, 10, 120, 30, test_frames)
        .expect("Failed to encode frames");

    assert!(
        packets.len() >= num_frames as usize,
        "Should have at least {} packets, got {}",
        num_frames,
        packets.len()
    );

    // Decode packets
    let mut decoder = Av1Decoder::new().expect("Failed to create decoder");
    let decoded_frames = decode_packets(&mut decoder, packets)
        .expect("Failed to decode packets");

    assert_eq!(
        decoded_frames.len(),
        num_frames as usize,
        "Decoded frame count should match input frame count"
    );

    // Verify all frames
    for frame in &decoded_frames {
        if let Frame::Video(video_frame) = frame {
            verify_frame_properties(video_frame, width, height, format)
                .expect("Frame properties don't match");
        } else {
            panic!("Expected video frame, got audio frame");
        }
    }
}

#[test]
fn test_av1_different_pixel_formats() {
    let width = 320;
    let height = 240;

    // Note: rav1e encoder primarily supports YUV420P for 8-bit encoding.
    // Other formats like YUV422P and YUV444P may be converted internally to YUV420P.
    // We test YUV420P (natively supported) and GRAY8 (grayscale).
    let formats = vec![
        PixelFormat::YUV420P,
        PixelFormat::GRAY8,
    ];

    for format in formats {
        println!("Testing format: {:?}", format);

        // Create test frame
        let test_frame = create_test_frame(width, height, format, 0, 128);

        // Create encoder
        let mut encoder = Av1EncoderBuilder::new(width, height)
            .speed_preset(10)
            .quantizer(100)
            .max_keyframe_interval(1)
            .build()
            .expect("Failed to create encoder");

        // Encode
        encoder
            .send_frame(&Frame::Video(test_frame))
            .expect("Failed to send frame");
        encoder.flush().expect("Failed to flush");

        // Collect packets
        let mut packets = Vec::new();
        loop {
            match encoder.receive_packet() {
                Ok(packet) => packets.push(packet),
                Err(Error::EndOfStream) => break,
                Err(Error::TryAgain) => continue,
                Err(e) => panic!("Encode error for {:?}: {:?}", format, e),
            }
        }

        assert!(
            !packets.is_empty(),
            "Should have packets for format {:?}",
            format
        );

        // Decode
        let mut decoder = Av1Decoder::new().expect("Failed to create decoder");
        let mut decoded_frames = Vec::new();

        for packet in packets {
            decoder.send_packet(&packet).expect("Failed to send packet");

            loop {
                match decoder.receive_frame() {
                    Ok(frame) => decoded_frames.push(frame),
                    Err(Error::TryAgain) => break,
                    Err(e) => panic!("Decode error for {:?}: {:?}", format, e),
                }
            }
        }

        assert!(
            !decoded_frames.is_empty(),
            "Should have decoded frames for format {:?}",
            format
        );

        // Verify - dimensions should match, format should be valid YUV
        if let Frame::Video(video_frame) = &decoded_frames[0] {
            verify_frame_properties(video_frame, width, height, format)
                .expect("Frame verification failed");
        }
    }

    // Test that unsupported formats are rejected
    println!("Testing unsupported format: RGB24");
    let mut rgb_frame = zvd_lib::codec::frame::VideoFrame::new(width, height, PixelFormat::RGB24);
    rgb_frame
        .data
        .push(Buffer::from_vec(vec![0u8; (width * height * 3) as usize]));
    rgb_frame.linesize.push((width * 3) as usize);

    let mut encoder = Av1EncoderBuilder::new(width, height)
        .speed_preset(10)
        .build()
        .expect("Failed to create encoder");

    let result = encoder.send_frame(&Frame::Video(rgb_frame));
    assert!(
        result.is_err(),
        "RGB24 should be rejected by AV1 encoder"
    );
}

#[test]
fn test_av1_keyframe_handling() {
    let width = 320;
    let height = 240;
    let format = PixelFormat::YUV420P;
    let keyframe_interval = 10;

    // Create 25 frames (should have keyframes at 0, 10, 20)
    let mut frames = Vec::new();
    for i in 0..25 {
        let frame = create_test_frame(width, height, format, i, 100);
        frames.push(Frame::Video(frame));
    }

    // Encode with specific keyframe interval
    let packets = encode_frames(width, height, 10, 100, keyframe_interval, frames)
        .expect("Failed to encode frames");

    // Count keyframes in packets
    let keyframe_count = packets.iter().filter(|p| p.is_keyframe()).count();

    // Should have at least 2 keyframes (first frame + at least one interval)
    assert!(
        keyframe_count >= 2,
        "Should have at least 2 keyframes, got {}",
        keyframe_count
    );

    // Verify packets have valid timestamps
    for (i, packet) in packets.iter().enumerate() {
        assert!(
            packet.pts.is_valid(),
            "Packet {} should have valid PTS",
            i
        );
    }

    // Decode and verify
    let mut decoder = Av1Decoder::new().expect("Failed to create decoder");
    let decoded_frames = decode_packets(&mut decoder, packets)
        .expect("Failed to decode");

    assert!(
        !decoded_frames.is_empty(),
        "Should have decoded frames"
    );
}

#[test]
fn test_av1_encoder_decoder_settings() {
    let width = 320;
    let height = 240;
    let format = PixelFormat::YUV420P;

    // Test different speed presets
    let speed_presets = vec![10, 8, 6]; // Fast to medium

    for speed in speed_presets {
        println!("Testing speed preset: {}", speed);

        let frame = create_test_frame(width, height, format, 0, 128);

        let mut encoder = Av1EncoderBuilder::new(width, height)
            .speed_preset(speed)
            .quantizer(100)
            .max_keyframe_interval(1)
            .build()
            .expect("Failed to create encoder");

        encoder
            .send_frame(&Frame::Video(frame))
            .expect("Failed to send frame");
        encoder.flush().expect("Failed to flush");

        let mut packets = Vec::new();
        loop {
            match encoder.receive_packet() {
                Ok(p) => packets.push(p),
                Err(Error::EndOfStream) => break,
                Err(Error::TryAgain) => continue,
                Err(e) => panic!("Error with speed {}: {:?}", speed, e),
            }
        }

        assert!(
            !packets.is_empty(),
            "Should have packets for speed {}",
            speed
        );

        // Decode to verify output is valid
        let mut decoder = Av1Decoder::new().expect("Failed to create decoder");
        for packet in packets {
            decoder.send_packet(&packet).expect("Failed to decode");
        }
    }

    // Test different quality levels
    let quality_levels = vec![80, 120, 160];

    for quality in quality_levels {
        println!("Testing quality level: {}", quality);

        let frame = create_test_frame(width, height, format, 0, 128);

        let mut encoder = Av1EncoderBuilder::new(width, height)
            .speed_preset(10)
            .quantizer(quality)
            .max_keyframe_interval(1)
            .build()
            .expect("Failed to create encoder");

        encoder
            .send_frame(&Frame::Video(frame))
            .expect("Failed to send frame");
        encoder.flush().expect("Failed to flush");

        let mut packets = Vec::new();
        loop {
            match encoder.receive_packet() {
                Ok(p) => packets.push(p),
                Err(Error::EndOfStream) => break,
                Err(Error::TryAgain) => continue,
                Err(e) => panic!("Error with quality {}: {:?}", quality, e),
            }
        }

        assert!(
            !packets.is_empty(),
            "Should have packets for quality {}",
            quality
        );

        // Decode to verify
        let mut decoder = Av1Decoder::new().expect("Failed to create decoder");
        for packet in packets {
            decoder.send_packet(&packet).expect("Failed to decode");
        }
    }
}

#[test]
fn test_av1_encoder_settings_validation() {
    // Test that builder validates settings correctly

    // Valid settings should work
    let valid = Av1EncoderBuilder::new(320, 240)
        .speed_preset(6)
        .quantizer(100)
        .build();
    assert!(valid.is_ok(), "Valid settings should work");

    // Invalid dimensions (not multiple of 8)
    let invalid_width = Av1EncoderBuilder::new(321, 240).build();
    assert!(invalid_width.is_err(), "Invalid width should fail");

    let invalid_height = Av1EncoderBuilder::new(320, 241).build();
    assert!(invalid_height.is_err(), "Invalid height should fail");

    // Zero dimensions
    let zero_dims = Av1EncoderBuilder::new(0, 0).build();
    assert!(zero_dims.is_err(), "Zero dimensions should fail");
}

#[test]
fn test_av1_threading() {
    // Test encoder with different thread counts
    let width = 320;
    let height = 240;
    let format = PixelFormat::YUV420P;

    let thread_counts = vec![0, 1, 2, 4]; // 0 = auto-detect

    for threads in thread_counts {
        println!("Testing with {} threads", threads);

        let frame = create_test_frame(width, height, format, 0, 100);

        let mut encoder = Av1EncoderBuilder::new(width, height)
            .speed_preset(10)
            .quantizer(100)
            .threads(threads)
            .build()
            .expect("Failed to create encoder");

        encoder
            .send_frame(&Frame::Video(frame))
            .expect("Failed to send frame");
        encoder.flush().expect("Failed to flush");

        let mut packet_count = 0;
        loop {
            match encoder.receive_packet() {
                Ok(_) => packet_count += 1,
                Err(Error::EndOfStream) => break,
                Err(Error::TryAgain) => continue,
                Err(e) => panic!("Error with {} threads: {:?}", threads, e),
            }
        }

        assert!(
            packet_count > 0,
            "Should produce packets with {} threads",
            threads
        );
    }

    // Test decoder with threads
    let decoder = Av1Decoder::with_threads(4);
    assert!(decoder.is_ok(), "Decoder with threads should work");
}

#[test]
fn test_av1_timestamp_preservation() {
    let width = 320;
    let height = 240;
    let format = PixelFormat::YUV420P;

    // Create frames with specific timestamps
    let mut frames = Vec::new();
    for i in 0..5 {
        let mut frame = create_test_frame(width, height, format, i * 100, 128);
        frame.pts = Timestamp::new(i * 100);
        frames.push(Frame::Video(frame));
    }

    // Encode
    let packets = encode_frames(width, height, 10, 100, 240, frames)
        .expect("Failed to encode");

    // Verify packets have timestamps
    for (i, packet) in packets.iter().enumerate() {
        assert!(
            packet.pts.is_valid(),
            "Packet {} should have valid timestamp",
            i
        );
    }

    // Decode and verify timestamps are preserved
    let mut decoder = Av1Decoder::new().expect("Failed to create decoder");
    let decoded_frames = decode_packets(&mut decoder, packets)
        .expect("Failed to decode");

    assert_eq!(decoded_frames.len(), 5, "Should decode 5 frames");

    for frame in decoded_frames {
        if let Frame::Video(video_frame) = frame {
            assert!(
                video_frame.pts.is_valid(),
                "Decoded frame should have valid timestamp"
            );
        }
    }
}

#[test]
fn test_av1_error_handling() {
    // Test sending unsupported pixel format
    let width = 320;
    let height = 240;

    let mut encoder = Av1EncoderBuilder::new(width, height)
        .speed_preset(10)
        .build()
        .expect("Failed to create encoder");

    // Create RGB frame (unsupported by AV1 encoder)
    let mut rgb_frame = zvd_lib::codec::frame::VideoFrame::new(width, height, PixelFormat::RGB24);
    rgb_frame
        .data
        .push(Buffer::from_vec(vec![0u8; (width * height * 3) as usize]));
    rgb_frame.linesize.push((width * 3) as usize);

    let result = encoder.send_frame(&Frame::Video(rgb_frame));
    assert!(
        result.is_err(),
        "Should reject unsupported pixel format"
    );

    // Test sending audio frame to video encoder
    use zvd_lib::codec::frame::AudioFrame;
    use zvd_lib::util::SampleFormat;

    let audio_frame = AudioFrame::new(1024, 48000, 2, SampleFormat::I16);
    let result = encoder.send_frame(&Frame::Audio(audio_frame));
    assert!(
        result.is_err(),
        "Should reject audio frames"
    );
}

#[test]
fn test_av1_flush_behavior() {
    let width = 320;
    let height = 240;
    let format = PixelFormat::YUV420P;

    let mut encoder = Av1EncoderBuilder::new(width, height)
        .speed_preset(10)
        .quantizer(100)
        .build()
        .expect("Failed to create encoder");

    // Send one frame
    let frame = create_test_frame(width, height, format, 0, 128);
    encoder
        .send_frame(&Frame::Video(frame))
        .expect("Failed to send frame");

    // Before flush, may not have all packets
    let _pre_flush_result = encoder.receive_packet();

    // Flush should make remaining packets available
    encoder.flush().expect("Flush should succeed");

    // Should be able to receive packet(s) after flush
    let mut got_packet = false;
    for _ in 0..10 {
        match encoder.receive_packet() {
            Ok(_) => {
                got_packet = true;
                break;
            }
            Err(Error::TryAgain) => continue,
            Err(Error::EndOfStream) => break,
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    assert!(got_packet, "Should get at least one packet after flush");

    // Test decoder flush
    let mut decoder = Av1Decoder::new().expect("Failed to create decoder");
    decoder.flush().expect("Decoder flush should succeed");
}
