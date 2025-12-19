//! Comprehensive integration tests for H.264 encoder and decoder
//!
//! These tests verify that the OpenH264 encoder and decoder work together
//! correctly, testing round-trip encoding/decoding, SPS/PPS extraction,
//! avcC format, and keyframe detection.

#[cfg(feature = "h264")]
use zvd_lib::codec::h264::{
    build_avcc, contains_keyframe, extract_sps_pps, parse_avcc, H264Decoder, H264Encoder,
};
#[cfg(feature = "h264")]
use zvd_lib::codec::{Decoder, Encoder, Frame};
#[cfg(feature = "h264")]
use zvd_lib::error::Error;
#[cfg(feature = "h264")]
use zvd_lib::format::Packet;
#[cfg(feature = "h264")]
use zvd_lib::util::{Buffer, PixelFormat, Timestamp};

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a test video frame with specified dimensions and fill pattern
#[cfg(feature = "h264")]
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
#[cfg(feature = "h264")]
fn verify_frame_properties(
    frame: &zvd_lib::codec::frame::VideoFrame,
    expected_width: u32,
    expected_height: u32,
) -> Result<(), String> {
    if frame.width != expected_width {
        return Err(format!(
            "Width mismatch: expected {}, got {}",
            expected_width, frame.width
        ));
    }

    if frame.height != expected_height {
        return Err(format!(
            "Height mismatch: expected {}, got {}",
            expected_height, frame.height
        ));
    }

    // Verify it's YUV420P format (what H.264 decoder outputs)
    if frame.format != PixelFormat::YUV420P {
        return Err(format!(
            "Format mismatch: expected YUV420P, got {:?}",
            frame.format
        ));
    }

    // Verify we have 3 planes
    if frame.data.len() != 3 {
        return Err(format!(
            "Plane count mismatch: expected 3, got {}",
            frame.data.len()
        ));
    }

    Ok(())
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
#[cfg(feature = "h264")]
fn test_h264_encode_decode_roundtrip() {
    println!("=== H.264 Encode/Decode Round-Trip Test ===");

    let width = 320;
    let height = 240;
    let num_frames = 5;

    // Create test frames with different fill patterns
    let mut test_frames = Vec::new();
    for i in 0..num_frames {
        let fill = 64 + (i * 20) as u8;
        let frame = create_test_frame(width, height, i, fill);
        test_frames.push(frame);
    }

    // Create encoder
    let mut encoder = H264Encoder::new(width, height).expect("Failed to create H.264 encoder");

    println!("Created encoder for {}x{}", width, height);

    // Encode all frames
    let mut encoded_packets = Vec::new();
    for (i, frame) in test_frames.iter().enumerate() {
        encoder
            .send_frame(&Frame::Video(frame.clone()))
            .expect("Failed to send frame to encoder");

        // Try to receive packets after each frame
        loop {
            match encoder.receive_packet() {
                Ok(packet) => {
                    println!(
                        "Encoded frame {}: {} bytes, keyframe={}",
                        i,
                        packet.data.len(),
                        packet.is_keyframe()
                    );
                    encoded_packets.push(packet);
                }
                Err(Error::TryAgain) => break,
                Err(e) => panic!("Unexpected encoder error: {:?}", e),
            }
        }
    }

    assert!(
        !encoded_packets.is_empty(),
        "Should have received encoded packets"
    );
    println!("Encoded {} packets total", encoded_packets.len());

    // Create decoder
    let mut decoder = H264Decoder::new().expect("Failed to create H.264 decoder");

    // Decode all packets
    let mut decoded_frames = Vec::new();
    for (i, packet) in encoded_packets.iter().enumerate() {
        decoder
            .send_packet(packet)
            .expect("Failed to send packet to decoder");

        // Try to receive frames
        loop {
            match decoder.receive_frame() {
                Ok(Frame::Video(frame)) => {
                    println!(
                        "Decoded frame from packet {}: {}x{}",
                        i, frame.width, frame.height
                    );
                    decoded_frames.push(frame);
                }
                Ok(Frame::Audio(_)) => panic!("Unexpected audio frame from H.264 decoder"),
                Err(Error::TryAgain) => break,
                Err(e) => panic!("Decode error: {:?}", e),
            }
        }
    }

    assert!(
        !decoded_frames.is_empty(),
        "Should have decoded at least one frame"
    );
    println!("Decoded {} frames total", decoded_frames.len());

    // Verify decoded frame properties
    for (i, frame) in decoded_frames.iter().enumerate() {
        verify_frame_properties(frame, width, height)
            .unwrap_or_else(|e| panic!("Frame {} verification failed: {}", i, e));
    }

    println!("=== Round-trip test PASSED ===\n");
}

#[test]
#[cfg(feature = "h264")]
fn test_h264_sps_pps_extraction() {
    println!("=== H.264 SPS/PPS Extraction Test ===");

    let width = 320;
    let height = 240;

    // Create encoder
    let mut encoder = H264Encoder::new(width, height).expect("Failed to create H.264 encoder");

    // Initially should have no parameter sets
    assert!(
        !encoder.has_parameter_sets(),
        "Should not have parameter sets before encoding"
    );
    assert!(
        encoder.sps().is_none(),
        "SPS should be None before encoding"
    );
    assert!(
        encoder.pps().is_none(),
        "PPS should be None before encoding"
    );

    // Encode a frame to trigger SPS/PPS extraction
    let frame = create_test_frame(width, height, 0, 128);
    encoder
        .send_frame(&Frame::Video(frame))
        .expect("Failed to send frame");

    // Receive the packet
    loop {
        match encoder.receive_packet() {
            Ok(packet) => {
                println!("Got packet: {} bytes", packet.data.len());

                // Extract SPS/PPS from the encoded data directly
                if let Some((sps, pps)) = extract_sps_pps(packet.data.as_slice()) {
                    println!("Extracted SPS from packet: {} bytes", sps.len());
                    println!("Extracted PPS from packet: {} bytes", pps.len());

                    // Verify SPS NAL type
                    let sps_nal_type = sps[0] & 0x1F;
                    assert_eq!(sps_nal_type, 7, "SPS NAL type should be 7");

                    // Verify PPS NAL type
                    let pps_nal_type = pps[0] & 0x1F;
                    assert_eq!(pps_nal_type, 8, "PPS NAL type should be 8");
                }
            }
            Err(Error::TryAgain) => break,
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    // Now encoder should have parameter sets cached
    assert!(
        encoder.has_parameter_sets(),
        "Should have parameter sets after encoding keyframe"
    );

    let sps = encoder.sps().expect("SPS should be available");
    let pps = encoder.pps().expect("PPS should be available");

    println!("Cached SPS: {} bytes", sps.len());
    println!("Cached PPS: {} bytes", pps.len());

    // Verify SPS format (first byte should indicate NAL type 7)
    let sps_nal_type = sps[0] & 0x1F;
    assert_eq!(sps_nal_type, 7, "SPS NAL type should be 7");

    // Verify PPS format (first byte should indicate NAL type 8)
    let pps_nal_type = pps[0] & 0x1F;
    assert_eq!(pps_nal_type, 8, "PPS NAL type should be 8");

    // SPS should have at least profile, level info (minimum ~4 bytes)
    assert!(sps.len() >= 4, "SPS should have at least 4 bytes");

    // PPS is typically smaller but should have content
    assert!(!pps.is_empty(), "PPS should not be empty");

    println!("=== SPS/PPS extraction test PASSED ===\n");
}

#[test]
#[cfg(feature = "h264")]
fn test_h264_extradata_avcc_format() {
    println!("=== H.264 Extradata avcC Format Test ===");

    let width = 320;
    let height = 240;

    // Create encoder
    let mut encoder = H264Encoder::new(width, height).expect("Failed to create H.264 encoder");

    // Encode a frame to generate extradata
    let frame = create_test_frame(width, height, 0, 100);
    encoder
        .send_frame(&Frame::Video(frame))
        .expect("Failed to send frame");

    // Receive packet
    loop {
        match encoder.receive_packet() {
            Ok(_) => {}
            Err(Error::TryAgain) => break,
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    // Get extradata
    let extradata = encoder.extradata().expect("Extradata should be available");
    println!("Extradata length: {} bytes", extradata.len());

    // avcC format verification:
    // Byte 0: Configuration version (should be 1)
    // Byte 1: AVC profile indication
    // Byte 2: Profile compatibility
    // Byte 3: AVC level indication
    // Byte 4: Length size minus 1 (lower 2 bits)
    // Byte 5: Number of SPS (lower 5 bits)
    // Then SPS length (2 bytes) + SPS data
    // Then number of PPS
    // Then PPS length (2 bytes) + PPS data

    assert!(extradata.len() >= 7, "avcC should be at least 7 bytes");
    assert_eq!(extradata[0], 1, "avcC version should be 1");

    let profile = extradata[1];
    let level = extradata[3];
    println!("Profile: {}, Level: {}", profile, level);

    // Parse the extradata to verify it's valid
    let parsed = parse_avcc(extradata).expect("Should be able to parse avcC");

    assert_eq!(parsed.configuration_version, 1);
    assert!(!parsed.sps.is_empty(), "Should have at least one SPS");
    assert!(!parsed.pps.is_empty(), "Should have at least one PPS");

    println!("Parsed avcC:");
    println!("  - Version: {}", parsed.configuration_version);
    println!("  - Profile: {}", parsed.avc_profile_indication);
    println!("  - Level: {}", parsed.avc_level_indication);
    println!("  - Length size: {}", parsed.length_size_minus_one + 1);
    println!("  - SPS count: {}", parsed.sps.len());
    println!("  - PPS count: {}", parsed.pps.len());

    // Verify we can reconstruct the avcC
    let sps = encoder.sps().expect("SPS should exist");
    let pps = encoder.pps().expect("PPS should exist");
    let reconstructed = build_avcc(sps, pps);

    // The reconstructed avcC should have the same structure
    let reparsed = parse_avcc(&reconstructed).expect("Should be able to parse reconstructed avcC");
    assert_eq!(reparsed.sps.len(), parsed.sps.len());
    assert_eq!(reparsed.pps.len(), parsed.pps.len());

    println!("=== Extradata avcC format test PASSED ===\n");
}

#[test]
#[cfg(feature = "h264")]
fn test_h264_keyframe_detection() {
    println!("=== H.264 Keyframe Detection Test ===");

    let width = 320;
    let height = 240;
    let num_frames = 10;

    // Create encoder
    let mut encoder = H264Encoder::new(width, height).expect("Failed to create H.264 encoder");

    // Encode multiple frames
    let mut packets = Vec::new();
    for i in 0..num_frames {
        let fill = 80 + (i * 10) as u8;
        let frame = create_test_frame(width, height, i, fill);
        encoder
            .send_frame(&Frame::Video(frame))
            .expect("Failed to send frame");

        // Receive packets
        loop {
            match encoder.receive_packet() {
                Ok(packet) => {
                    packets.push(packet);
                }
                Err(Error::TryAgain) => break,
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }
    }

    assert!(!packets.is_empty(), "Should have encoded packets");

    // Check keyframe flags
    let mut keyframe_count = 0;
    let mut non_keyframe_count = 0;

    for (i, packet) in packets.iter().enumerate() {
        let is_keyframe_flag = packet.is_keyframe();
        let contains_idr = contains_keyframe(packet.data.as_slice());

        println!(
            "Packet {}: {} bytes, keyframe_flag={}, contains_IDR={}",
            i,
            packet.data.len(),
            is_keyframe_flag,
            contains_idr
        );

        // The keyframe flag should match whether the packet contains an IDR NAL
        assert_eq!(
            is_keyframe_flag, contains_idr,
            "Packet {} keyframe flag mismatch: flag={}, contains_IDR={}",
            i, is_keyframe_flag, contains_idr
        );

        if is_keyframe_flag {
            keyframe_count += 1;
        } else {
            non_keyframe_count += 1;
        }
    }

    println!("\nKeyframe statistics:");
    println!("  - Total packets: {}", packets.len());
    println!("  - Keyframes: {}", keyframe_count);
    println!("  - Non-keyframes: {}", non_keyframe_count);

    // First packet should be a keyframe (IDR)
    assert!(
        packets[0].is_keyframe(),
        "First encoded packet should be a keyframe"
    );
    assert!(
        contains_keyframe(packets[0].data.as_slice()),
        "First packet should contain IDR NAL"
    );

    // Should have at least one keyframe
    assert!(keyframe_count >= 1, "Should have at least one keyframe");

    println!("=== Keyframe detection test PASSED ===\n");
}

#[test]
#[cfg(feature = "h264")]
fn test_h264_encoder_creation_variants() {
    println!("=== H.264 Encoder Creation Variants Test ===");

    // Test default creation
    let encoder1 = H264Encoder::new(640, 480);
    assert!(encoder1.is_ok(), "Default encoder creation should succeed");
    println!("Created 640x480 encoder with default settings");

    // Test with custom bitrate
    let encoder2 = H264Encoder::with_bitrate(1920, 1080, 5_000_000);
    assert!(
        encoder2.is_ok(),
        "Encoder with custom bitrate should succeed"
    );
    println!("Created 1920x1080 encoder with 5 Mbps bitrate");

    // Test with small dimensions
    let encoder3 = H264Encoder::new(64, 64);
    assert!(encoder3.is_ok(), "Small dimension encoder should succeed");
    println!("Created 64x64 encoder");

    println!("=== Encoder creation variants test PASSED ===\n");
}

#[test]
#[cfg(feature = "h264")]
fn test_h264_decoder_creation() {
    println!("=== H.264 Decoder Creation Test ===");

    let decoder = H264Decoder::new();
    assert!(decoder.is_ok(), "Decoder creation should succeed");
    println!("Created H.264 decoder");

    println!("=== Decoder creation test PASSED ===\n");
}

#[test]
#[cfg(feature = "h264")]
fn test_h264_timestamp_preservation() {
    println!("=== H.264 Timestamp Preservation Test ===");

    let width = 320;
    let height = 240;

    // Create encoder
    let mut encoder = H264Encoder::new(width, height).expect("Failed to create encoder");

    // Create frames with specific timestamps
    let timestamps: Vec<i64> = vec![0, 1000, 2000, 3000, 4000];

    for &pts in &timestamps {
        let mut frame = create_test_frame(width, height, pts, 128);
        frame.pts = Timestamp::new(pts);

        encoder
            .send_frame(&Frame::Video(frame))
            .expect("Failed to send frame");
    }

    // Collect packets
    let mut packets = Vec::new();
    loop {
        match encoder.receive_packet() {
            Ok(packet) => {
                println!("Packet PTS: {:?}", packet.pts);
                packets.push(packet);
            }
            Err(Error::TryAgain) => break,
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    // Verify packets have valid timestamps
    for (i, packet) in packets.iter().enumerate() {
        assert!(packet.pts.is_valid(), "Packet {} should have valid PTS", i);
    }

    // Decode and verify timestamps are preserved
    let mut decoder = H264Decoder::new().expect("Failed to create decoder");

    for packet in &packets {
        decoder.send_packet(packet).expect("Failed to send packet");

        loop {
            match decoder.receive_frame() {
                Ok(Frame::Video(frame)) => {
                    println!("Decoded frame PTS: {:?}", frame.pts);
                    assert!(frame.pts.is_valid(), "Decoded frame should have valid PTS");
                }
                Ok(Frame::Audio(_)) => panic!("Unexpected audio frame"),
                Err(Error::TryAgain) => break,
                Err(e) => panic!("Decode error: {:?}", e),
            }
        }
    }

    println!("=== Timestamp preservation test PASSED ===\n");
}

#[test]
#[cfg(feature = "h264")]
fn test_h264_error_handling() {
    println!("=== H.264 Error Handling Test ===");

    let width = 320;
    let height = 240;

    let mut encoder = H264Encoder::new(width, height).expect("Failed to create encoder");

    // Test sending audio frame to video encoder
    use zvd_lib::codec::frame::AudioFrame;
    use zvd_lib::util::SampleFormat;

    let audio_frame = AudioFrame::new(1024, 48000, 2, SampleFormat::I16);
    let result = encoder.send_frame(&Frame::Audio(audio_frame));
    assert!(result.is_err(), "Should reject audio frames");
    println!("Correctly rejected audio frame");

    // Test sending unsupported pixel format
    let mut rgb_frame = zvd_lib::codec::frame::VideoFrame::new(width, height, PixelFormat::RGB24);
    rgb_frame
        .data
        .push(Buffer::from_vec(vec![0u8; (width * height * 3) as usize]));
    rgb_frame.linesize.push((width * 3) as usize);

    let result = encoder.send_frame(&Frame::Video(rgb_frame));
    assert!(result.is_err(), "Should reject RGB24 format");
    println!("Correctly rejected RGB24 pixel format");

    println!("=== Error handling test PASSED ===\n");
}

#[test]
#[cfg(feature = "h264")]
fn test_h264_flush_behavior() {
    println!("=== H.264 Flush Behavior Test ===");

    let width = 320;
    let height = 240;

    // Test encoder flush
    let mut encoder = H264Encoder::new(width, height).expect("Failed to create encoder");

    let frame = create_test_frame(width, height, 0, 128);
    encoder
        .send_frame(&Frame::Video(frame))
        .expect("Failed to send frame");

    // Receive any available packets
    loop {
        match encoder.receive_packet() {
            Ok(_) => {}
            Err(Error::TryAgain) => break,
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    // Flush should succeed
    encoder.flush().expect("Encoder flush should succeed");
    println!("Encoder flush succeeded");

    // Test decoder flush
    let mut decoder = H264Decoder::new().expect("Failed to create decoder");
    decoder.flush().expect("Decoder flush should succeed");
    println!("Decoder flush succeeded");

    println!("=== Flush behavior test PASSED ===\n");
}
