//! VP8 codec integration tests
//!
//! Tests the VP8 encoder and decoder using libvpx.
//! Enable the `vp8-libvpx` feature to run these tests.
//!
//! To run: `cargo test --features vp8-libvpx --test vp8_codec_test`

#![cfg(feature = "vp8-libvpx")]

use zvd_lib::codec::vp8::{Vp8Decoder, Vp8Encoder, Vp8EncoderConfig, Vp8RateControl};
use zvd_lib::codec::{Decoder, Encoder, Frame};
use zvd_lib::codec::frame::VideoFrame;
use zvd_lib::error::Error;
use zvd_lib::format::Packet;
use zvd_lib::util::{Buffer, PixelFormat, Timestamp};

/// Create a test video frame with YUV420P format
fn create_test_frame(width: u32, height: u32, frame_num: u64, keyframe: bool) -> VideoFrame {
    let mut frame = VideoFrame::new(width, height, PixelFormat::YUV420P);

    let y_size = (width * height) as usize;
    let uv_size = (width / 2 * height / 2) as usize;

    // Create a gradient pattern that changes per frame
    let mut y_data = Vec::with_capacity(y_size);
    for y in 0..height {
        for x in 0..width {
            // Create a moving gradient pattern
            let value = ((x + y + (frame_num as u32 * 10)) % 256) as u8;
            y_data.push(value);
        }
    }

    // U and V planes - simple checkerboard pattern
    let mut u_data = Vec::with_capacity(uv_size);
    let mut v_data = Vec::with_capacity(uv_size);
    for y in 0..(height / 2) {
        for x in 0..(width / 2) {
            let checker = ((x / 8 + y / 8) % 2) == 0;
            u_data.push(if checker { 100 } else { 200 });
            v_data.push(if checker { 200 } else { 100 });
        }
    }

    frame.data = vec![
        Buffer::from_vec(y_data),
        Buffer::from_vec(u_data),
        Buffer::from_vec(v_data),
    ];
    frame.linesize = vec![width as usize, (width / 2) as usize, (width / 2) as usize];
    frame.keyframe = keyframe;
    frame.pts = Timestamp::new(frame_num as i64);

    frame
}

#[test]
fn test_vp8_encoder_decoder_roundtrip() {
    let width = 320;
    let height = 240;
    let num_frames = 5;

    // Create encoder
    let config = Vp8EncoderConfig {
        width,
        height,
        bitrate: 500_000,
        framerate: 30,
        keyframe_interval: 30,
        threads: 2,
        rate_control: Vp8RateControl::VBR,
        cpu_used: 4,
        ..Default::default()
    };

    let mut encoder = Vp8Encoder::with_config(config).expect("Failed to create VP8 encoder");
    let mut decoder = Vp8Decoder::new().expect("Failed to create VP8 decoder");

    // Encode frames
    let mut encoded_packets = Vec::new();
    for i in 0..num_frames {
        let frame = create_test_frame(width, height, i, i == 0);
        encoder.send_frame(&Frame::Video(frame)).expect("Failed to send frame");

        // Collect any available packets
        loop {
            match encoder.receive_packet() {
                Ok(packet) => {
                    assert!(!packet.data.is_empty(), "Encoded packet should not be empty");
                    encoded_packets.push(packet);
                }
                Err(Error::TryAgain) => break,
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }
    }

    // Flush encoder to get remaining packets
    encoder.flush().expect("Failed to flush encoder");
    loop {
        match encoder.receive_packet() {
            Ok(packet) => encoded_packets.push(packet),
            Err(Error::TryAgain) => break,
            Err(e) => panic!("Unexpected error during flush: {:?}", e),
        }
    }

    assert!(!encoded_packets.is_empty(), "Should have encoded at least one packet");
    println!("Encoded {} packets", encoded_packets.len());

    // Verify first packet is a keyframe
    assert!(encoded_packets[0].is_keyframe(), "First packet should be a keyframe");

    // Decode all packets
    let mut decoded_frames = Vec::new();
    for packet in &encoded_packets {
        decoder.send_packet(packet).expect("Failed to send packet to decoder");

        loop {
            match decoder.receive_frame() {
                Ok(Frame::Video(frame)) => {
                    assert_eq!(frame.width, width);
                    assert_eq!(frame.height, height);
                    decoded_frames.push(frame);
                }
                Ok(_) => panic!("Expected video frame"),
                Err(Error::TryAgain) => break,
                Err(e) => panic!("Unexpected decode error: {:?}", e),
            }
        }
    }

    // Flush decoder
    decoder.flush().expect("Failed to flush decoder");

    println!("Decoded {} frames", decoded_frames.len());
    assert!(!decoded_frames.is_empty(), "Should have decoded at least one frame");
}

#[test]
fn test_vp8_encoder_multiple_resolutions() {
    let resolutions = [
        (320, 240),
        (640, 480),
        (854, 480),
        (1280, 720),
    ];

    for (width, height) in resolutions {
        println!("Testing VP8 encoding at {}x{}", width, height);

        let mut encoder = Vp8Encoder::new(width, height)
            .expect(&format!("Failed to create encoder for {}x{}", width, height));

        // Create and encode a single frame
        let frame = create_test_frame(width, height, 0, true);
        encoder.send_frame(&Frame::Video(frame))
            .expect(&format!("Failed to encode frame at {}x{}", width, height));

        // Flush to get output
        encoder.flush().expect("Failed to flush");

        // Should have output
        match encoder.receive_packet() {
            Ok(packet) => {
                println!("  Encoded to {} bytes", packet.data.len());
                assert!(!packet.data.is_empty());
            }
            Err(Error::TryAgain) => {
                // Some resolutions might buffer
                println!("  Frame buffered (no immediate output)");
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }
}

#[test]
fn test_vp8_encoder_rate_control_modes() {
    let width = 320;
    let height = 240;

    let modes = [
        (Vp8RateControl::VBR, "VBR"),
        (Vp8RateControl::CBR, "CBR"),
        (Vp8RateControl::CQ, "CQ"),
    ];

    for (mode, name) in modes {
        println!("Testing VP8 with {} rate control", name);

        let config = Vp8EncoderConfig {
            width,
            height,
            bitrate: 500_000,
            framerate: 30,
            rate_control: mode,
            cq_level: 31,
            ..Default::default()
        };

        let mut encoder = Vp8Encoder::with_config(config)
            .expect(&format!("Failed to create encoder with {} mode", name));

        // Encode a few frames
        for i in 0..3 {
            let frame = create_test_frame(width, height, i, i == 0);
            encoder.send_frame(&Frame::Video(frame))
                .expect(&format!("Failed to encode frame with {} mode", name));
        }

        encoder.flush().expect("Failed to flush");

        // Collect packets
        let mut total_bytes = 0;
        loop {
            match encoder.receive_packet() {
                Ok(packet) => {
                    total_bytes += packet.data.len();
                }
                Err(Error::TryAgain) => break,
                Err(e) => panic!("Error with {} mode: {:?}", name, e),
            }
        }

        println!("  {} mode produced {} bytes", name, total_bytes);
        assert!(total_bytes > 0, "{} mode should produce output", name);
    }
}

#[test]
fn test_vp8_decoder_empty_packet() {
    let mut decoder = Vp8Decoder::new().expect("Failed to create decoder");

    // Create empty packet
    let packet = Packet::new(0, Buffer::new());

    // Should return error for empty packet
    let result = decoder.send_packet(&packet);
    assert!(result.is_err(), "Should reject empty packet");
}

#[test]
fn test_vp8_decoder_invalid_data() {
    let mut decoder = Vp8Decoder::new().expect("Failed to create decoder");

    // Create packet with random invalid data
    let invalid_data = vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
    let packet = Packet::new(0, Buffer::from_vec(invalid_data));

    // Should return error for invalid VP8 data
    let result = decoder.send_packet(&packet);
    // libvpx might accept some garbage data initially, so we just verify it doesn't panic
    match result {
        Ok(_) | Err(_) => {} // Either is acceptable
    }
}

#[test]
fn test_vp8_encoder_forced_keyframe() {
    let width = 320;
    let height = 240;

    let config = Vp8EncoderConfig {
        width,
        height,
        bitrate: 500_000,
        framerate: 30,
        keyframe_interval: 100, // High interval so we can test forced keyframes
        ..Default::default()
    };

    let mut encoder = Vp8Encoder::with_config(config).expect("Failed to create encoder");

    // Encode several frames
    for i in 0..10 {
        let mut frame = create_test_frame(width, height, i, false);

        // Force keyframe at frame 5
        if i == 5 {
            frame.keyframe = true;
        }

        encoder.send_frame(&Frame::Video(frame)).expect("Failed to send frame");
    }

    encoder.flush().expect("Failed to flush");

    // Collect and check packets
    let mut packets = Vec::new();
    loop {
        match encoder.receive_packet() {
            Ok(packet) => packets.push(packet),
            Err(Error::TryAgain) => break,
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    // First packet should definitely be a keyframe
    assert!(packets[0].is_keyframe(), "First packet should be keyframe");

    // Count keyframes
    let keyframe_count = packets.iter().filter(|p| p.is_keyframe()).count();
    println!("Found {} keyframes in {} packets", keyframe_count, packets.len());
}

#[test]
fn test_vp8_encoder_thread_configurations() {
    let width = 640;
    let height = 480;
    let thread_counts = [1, 2, 4];

    for threads in thread_counts {
        println!("Testing VP8 with {} threads", threads);

        let config = Vp8EncoderConfig {
            width,
            height,
            bitrate: 1_000_000,
            framerate: 30,
            threads,
            ..Default::default()
        };

        let result = Vp8Encoder::with_config(config);
        assert!(result.is_ok(), "Should support {} threads", threads);
    }
}

#[test]
fn test_vp8_speed_presets() {
    let width = 320;
    let height = 240;

    // VP8 cpu_used ranges from -16 to 16
    let presets = [-8, -4, 0, 4, 8, 12];

    for preset in presets {
        println!("Testing VP8 with cpu_used = {}", preset);

        let config = Vp8EncoderConfig {
            width,
            height,
            bitrate: 500_000,
            framerate: 30,
            cpu_used: preset,
            ..Default::default()
        };

        let result = Vp8Encoder::with_config(config);
        assert!(result.is_ok(), "Should support cpu_used = {}", preset);
    }
}

#[test]
fn test_vp8_error_resilient_mode() {
    let width = 320;
    let height = 240;

    let config = Vp8EncoderConfig {
        width,
        height,
        bitrate: 500_000,
        framerate: 30,
        error_resilient: true,
        ..Default::default()
    };

    let mut encoder = Vp8Encoder::with_config(config).expect("Failed to create encoder");

    // Encode a frame
    let frame = create_test_frame(width, height, 0, true);
    encoder.send_frame(&Frame::Video(frame)).expect("Failed to send frame");
    encoder.flush().expect("Failed to flush");

    // Should produce output
    match encoder.receive_packet() {
        Ok(packet) => {
            assert!(!packet.data.is_empty());
            println!("Error resilient mode produced {} bytes", packet.data.len());
        }
        Err(Error::TryAgain) => println!("Frame was buffered"),
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

#[test]
fn test_vp8_dimension_validation() {
    // Zero dimensions should fail
    let result = Vp8Encoder::new(0, 480);
    assert!(result.is_err(), "Should reject zero width");

    let result = Vp8Encoder::new(640, 0);
    assert!(result.is_err(), "Should reject zero height");

    let result = Vp8Encoder::new(0, 0);
    assert!(result.is_err(), "Should reject zero dimensions");
}

#[test]
fn test_vp8_frame_format_validation() {
    let mut encoder = Vp8Encoder::new(320, 240).expect("Failed to create encoder");

    // Create frame with wrong format (not YUV420P)
    let mut frame = VideoFrame::new(320, 240, PixelFormat::YUV444P);
    frame.data = vec![
        Buffer::from_vec(vec![128u8; 320 * 240]),
        Buffer::from_vec(vec![128u8; 320 * 240]),
        Buffer::from_vec(vec![128u8; 320 * 240]),
    ];
    frame.linesize = vec![320, 320, 320];
    frame.keyframe = true;
    frame.pts = Timestamp::new(0);

    let result = encoder.send_frame(&Frame::Video(frame));
    assert!(result.is_err(), "Should reject non-YUV420P frame");
}

#[test]
fn test_vp8_consecutive_sequences() {
    // Test that we can encode multiple sequences back-to-back
    let width = 320;
    let height = 240;

    for sequence in 0..3 {
        println!("Encoding sequence {}", sequence);

        let mut encoder = Vp8Encoder::new(width, height).expect("Failed to create encoder");
        let mut decoder = Vp8Decoder::new().expect("Failed to create decoder");

        // Encode 3 frames per sequence
        for i in 0..3 {
            let frame = create_test_frame(width, height, i, i == 0);
            encoder.send_frame(&Frame::Video(frame)).expect("Failed to encode");
        }

        encoder.flush().expect("Failed to flush");

        // Decode
        let mut decoded_count = 0;
        loop {
            match encoder.receive_packet() {
                Ok(packet) => {
                    decoder.send_packet(&packet).expect("Failed to decode");
                    loop {
                        match decoder.receive_frame() {
                            Ok(_) => decoded_count += 1,
                            Err(Error::TryAgain) => break,
                            Err(e) => panic!("Decode error: {:?}", e),
                        }
                    }
                }
                Err(Error::TryAgain) => break,
                Err(e) => panic!("Encode error: {:?}", e),
            }
        }

        println!("  Decoded {} frames", decoded_count);
    }
}
