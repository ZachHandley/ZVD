//! Integration tests for H.264 codec
//!
//! These tests verify the complete H.264 encoder/decoder pipeline and integration
//! with the codec factory system.

#[cfg(feature = "h264")]
use zvd_lib::codec::{create_decoder, create_encoder, Decoder, Encoder, Frame};
#[cfg(feature = "h264")]
use zvd_lib::format::Packet;
#[cfg(feature = "h264")]
use zvd_lib::util::{Buffer, PixelFormat, Timestamp};

/// Test creating H.264 decoder through factory function
#[test]
#[cfg(feature = "h264")]
fn test_h264_decoder_factory() {
    let decoder = create_decoder("h264");
    assert!(decoder.is_ok(), "Factory should create H.264 decoder");

    let mut decoder = decoder.unwrap();
    // Should be able to flush immediately
    assert!(decoder.flush().is_ok());
}

/// Test creating H.264 encoder through factory function
#[test]
#[cfg(feature = "h264")]
fn test_h264_encoder_factory() {
    let encoder = create_encoder("h264", 640, 480);
    assert!(encoder.is_ok(), "Factory should create H.264 encoder");

    let mut encoder = encoder.unwrap();
    // Should be able to flush immediately
    assert!(encoder.flush().is_ok());
}

/// Test creating H.264 encoder with invalid dimensions
#[test]
#[cfg(feature = "h264")]
fn test_h264_encoder_invalid_dimensions() {
    // Width not multiple of 2
    let encoder = create_encoder("h264", 641, 480);
    assert!(encoder.is_err(), "Should reject non-even width");

    // Height not multiple of 2
    let encoder = create_encoder("h264", 640, 481);
    assert!(encoder.is_err(), "Should reject non-even height");

    // Zero dimensions
    let encoder = create_encoder("h264", 0, 480);
    assert!(encoder.is_err(), "Should reject zero width");

    let encoder = create_encoder("h264", 640, 0);
    assert!(encoder.is_err(), "Should reject zero height");
}

/// Test basic H.264 encoding pipeline
#[test]
#[cfg(feature = "h264")]
fn test_h264_encoding_pipeline() {
    use zvd_lib::codec::VideoFrame;

    // Create encoder
    let mut encoder = create_encoder("h264", 320, 240).expect("Failed to create encoder");

    // Send multiple frames
    for i in 0..10 {
        let mut frame = VideoFrame::new(320, 240, PixelFormat::YUV420P);

        // Y plane (320x240)
        let y_size = 320 * 240;
        frame.data.push(Buffer::from_vec(vec![100u8; y_size]));
        frame.linesize.push(320);

        // U plane (160x120)
        let uv_size = 160 * 120;
        frame.data.push(Buffer::from_vec(vec![128u8; uv_size]));
        frame.linesize.push(160);

        // V plane (160x120)
        frame.data.push(Buffer::from_vec(vec![128u8; uv_size]));
        frame.linesize.push(160);

        frame.pts = Timestamp::new(i);
        frame.keyframe = i == 0;

        encoder
            .send_frame(&Frame::Video(frame))
            .expect("Failed to send frame");
    }

    // Flush encoder
    encoder.flush().expect("Failed to flush encoder");

    // Receive at least one packet
    let mut packet_count = 0;
    for _ in 0..20 {
        match encoder.receive_packet() {
            Ok(packet) => {
                assert!(!packet.data.is_empty(), "Packet should contain data");
                assert!(packet.pts.is_valid(), "Packet should have valid PTS");
                packet_count += 1;
            }
            Err(_) => break,
        }
    }

    assert!(
        packet_count > 0,
        "Should receive at least one encoded packet"
    );
}

/// Test H.264 encode-decode roundtrip
#[test]
#[cfg(feature = "h264")]
fn test_h264_roundtrip() {
    use zvd_lib::codec::VideoFrame;

    // Create encoder and decoder
    let mut encoder = create_encoder("h264", 320, 240).expect("Failed to create encoder");
    let mut decoder = create_decoder("h264").expect("Failed to create decoder");

    // Create and encode a frame
    let mut frame = VideoFrame::new(320, 240, PixelFormat::YUV420P);
    frame.data.push(Buffer::from_vec(vec![100u8; 320 * 240]));
    frame.linesize.push(320);
    frame.data.push(Buffer::from_vec(vec![128u8; 160 * 120]));
    frame.linesize.push(160);
    frame.data.push(Buffer::from_vec(vec![128u8; 160 * 120]));
    frame.linesize.push(160);
    frame.pts = Timestamp::new(0);
    frame.keyframe = true;

    encoder
        .send_frame(&Frame::Video(frame))
        .expect("Failed to encode frame");
    encoder.flush().expect("Failed to flush encoder");

    // Get encoded packet
    let packet = encoder.receive_packet().expect("Failed to receive packet");

    // Decode packet
    decoder
        .send_packet(&packet)
        .expect("Failed to send packet to decoder");

    let decoded_frame = decoder.receive_frame().expect("Failed to receive frame");

    // Verify frame properties
    if let Frame::Video(vf) = decoded_frame {
        assert_eq!(vf.width, 320);
        assert_eq!(vf.height, 240);
        assert_eq!(vf.format, PixelFormat::YUV420P);
    } else {
        panic!("Expected video frame");
    }
}

/// Test H.264 module-level API
#[test]
#[cfg(feature = "h264")]
fn test_h264_module_api() {
    use zvd_lib::codec::h264::{H264Decoder, H264Encoder, H264EncoderConfig};
    use zvd_lib::util::Rational;

    // Test decoder creation
    let decoder = H264Decoder::new();
    assert!(decoder.is_ok(), "Should create decoder");

    // Test encoder with default config
    let config = H264EncoderConfig {
        width: 640,
        height: 480,
        bitrate: 500_000,
        framerate: Rational::new(30, 1),
        keyframe_interval: 60,
    };
    let encoder = H264Encoder::new(config);
    assert!(encoder.is_ok(), "Should create encoder with config");
}

/// Test H.264 encoder configuration
#[test]
#[cfg(feature = "h264")]
fn test_h264_encoder_config() {
    use zvd_lib::codec::h264::{H264Encoder, H264EncoderConfig};
    use zvd_lib::util::Rational;

    // Test various bitrates
    let config = H264EncoderConfig {
        width: 1920,
        height: 1080,
        bitrate: 5_000_000, // 5 Mbps
        framerate: Rational::new(30, 1),
        keyframe_interval: 120,
    };
    assert!(H264Encoder::new(config).is_ok(), "Should support 5 Mbps");

    // Test various framerates
    let config = H264EncoderConfig {
        width: 1280,
        height: 720,
        bitrate: 2_000_000,
        framerate: Rational::new(60, 1), // 60 fps
        keyframe_interval: 120,
    };
    assert!(H264Encoder::new(config).is_ok(), "Should support 60 fps");

    // Test different resolutions
    let config = H264EncoderConfig {
        width: 1920,
        height: 1080,
        bitrate: 4_000_000,
        framerate: Rational::new(24, 1),
        keyframe_interval: 240,
    };
    assert!(H264Encoder::new(config).is_ok(), "Should support 1080p");

    let config = H264EncoderConfig {
        width: 3840,
        height: 2160,
        bitrate: 15_000_000,
        framerate: Rational::new(30, 1),
        keyframe_interval: 120,
    };
    assert!(H264Encoder::new(config).is_ok(), "Should support 4K");
}

/// Test H.264 keyframe interval
#[test]
#[cfg(feature = "h264")]
fn test_h264_keyframe_interval() {
    use zvd_lib::codec::h264::{H264Encoder, H264EncoderConfig};
    use zvd_lib::codec::{Encoder, Frame, VideoFrame};
    use zvd_lib::util::Rational;

    let config = H264EncoderConfig {
        width: 320,
        height: 240,
        bitrate: 500_000,
        framerate: Rational::new(30, 1),
        keyframe_interval: 5, // Keyframe every 5 frames
    };
    let mut encoder = H264Encoder::new(config).expect("Failed to create encoder");

    // Send 20 frames
    for i in 0..20 {
        let mut frame = VideoFrame::new(320, 240, PixelFormat::YUV420P);
        frame.data.push(Buffer::from_vec(vec![100u8; 320 * 240]));
        frame.linesize.push(320);
        frame.data.push(Buffer::from_vec(vec![128u8; 160 * 120]));
        frame.linesize.push(160);
        frame.data.push(Buffer::from_vec(vec![128u8; 160 * 120]));
        frame.linesize.push(160);
        frame.pts = Timestamp::new(i);
        frame.keyframe = i % 5 == 0;

        encoder
            .send_frame(&Frame::Video(frame))
            .expect("Failed to send frame");
    }

    encoder.flush().expect("Failed to flush");

    // Verify we got packets
    let mut packet_count = 0;
    let mut keyframe_count = 0;
    for _ in 0..30 {
        match encoder.receive_packet() {
            Ok(packet) => {
                packet_count += 1;
                if packet.keyframe {
                    keyframe_count += 1;
                }
            }
            Err(_) => break,
        }
    }

    assert!(packet_count > 0, "Should receive packets");
    assert!(keyframe_count > 0, "Should have keyframes");
}

/// Test codec info for H.264
#[test]
#[cfg(feature = "h264")]
fn test_h264_codec_info() {
    use zvd_lib::codec::get_codec_info;
    use zvd_lib::util::MediaType;

    let info = get_codec_info("h264");
    assert!(info.is_some(), "Should have codec info for H.264");

    let info = info.unwrap();
    assert_eq!(info.id, "h264");
    assert_eq!(info.name, "H.264");
    assert_eq!(info.long_name, "H.264 / AVC / MPEG-4 AVC");
    assert_eq!(info.media_type, MediaType::Video);
    assert!(info.capabilities.lossy, "H.264 supports lossy compression");
    assert!(info.capabilities.inter, "H.264 supports inter-frame coding");
}

/// Verify that H.264 is listed in available codecs when feature is enabled
#[test]
#[cfg(feature = "h264")]
fn test_h264_in_codec_list() {
    use zvd_lib::codec::get_codec_info;

    assert!(
        get_codec_info("h264").is_some(),
        "H.264 should be in codec registry"
    );
}

/// Test that H.264 is not available without feature flag
#[test]
#[cfg(not(feature = "h264"))]
fn test_h264_not_available_without_feature() {
    use zvd_lib::codec::create_encoder;

    let encoder = create_encoder("h264", 640, 480);
    assert!(
        encoder.is_err(),
        "H.264 should not be available without feature flag"
    );
}
