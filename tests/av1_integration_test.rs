//! Integration tests for AV1 codec
//!
//! These tests verify the complete AV1 encoder/decoder pipeline and integration
//! with the codec factory system.

use zvd_lib::codec::{create_decoder, create_encoder, Decoder, Encoder, Frame};
use zvd_lib::format::Packet;
use zvd_lib::util::{Buffer, PixelFormat, Timestamp};

/// Test creating AV1 decoder through factory function
#[test]
fn test_av1_decoder_factory() {
    let decoder = create_decoder("av1");
    assert!(decoder.is_ok(), "Factory should create AV1 decoder");

    let mut decoder = decoder.unwrap();
    // Should be able to flush immediately
    assert!(decoder.flush().is_ok());
}

/// Test creating AV1 encoder through factory function
#[test]
fn test_av1_encoder_factory() {
    let encoder = create_encoder("av1", 640, 480);
    assert!(encoder.is_ok(), "Factory should create AV1 encoder");

    let mut encoder = encoder.unwrap();
    // Should be able to flush immediately
    assert!(encoder.flush().is_ok());
}

/// Test creating AV1 encoder with invalid dimensions
#[test]
fn test_av1_encoder_invalid_dimensions() {
    // Width not multiple of 8
    let encoder = create_encoder("av1", 641, 480);
    assert!(encoder.is_err(), "Should reject non-multiple-of-8 width");

    // Height not multiple of 8
    let encoder = create_encoder("av1", 640, 481);
    assert!(encoder.is_err(), "Should reject non-multiple-of-8 height");
}

/// Test basic encoding pipeline
#[test]
fn test_av1_encoding_pipeline() {
    use zvd_lib::codec::VideoFrame;

    // Create encoder with fast settings for testing
    let mut encoder = create_encoder("av1", 320, 240).expect("Failed to create encoder");

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

/// Test direct AV1 module API
#[test]
fn test_av1_module_api() {
    use zvd_lib::codec::av1::{create_decoder, create_encoder, Av1EncoderBuilder};

    // Test module-level factory functions
    let decoder = create_decoder();
    assert!(decoder.is_ok(), "Module factory should create decoder");

    let encoder = create_encoder(640, 480);
    assert!(encoder.is_ok(), "Module factory should create encoder");

    // Test builder API
    let encoder = Av1EncoderBuilder::new(1920, 1080)
        .speed_preset(6)
        .quantizer(100)
        .max_keyframe_interval(240)
        .threads(4)
        .build();
    assert!(encoder.is_ok(), "Builder should create encoder");
}

/// Test codec info for AV1
#[test]
fn test_av1_codec_info() {
    use zvd_lib::codec::get_codec_info;
    use zvd_lib::util::MediaType;

    let info = get_codec_info("av1");
    assert!(info.is_some(), "Should have codec info for AV1");

    let info = info.unwrap();
    assert_eq!(info.id, "av1");
    assert_eq!(info.name, "AV1");
    assert_eq!(info.long_name, "AOMedia Video 1");
    assert_eq!(info.media_type, MediaType::Video);
    assert!(info.capabilities.lossy, "AV1 supports lossy compression");
    assert!(
        info.capabilities.lossless,
        "AV1 supports lossless compression"
    );
    assert!(info.capabilities.inter, "AV1 supports inter-frame coding");
}

/// Test AV1 with various pixel formats
#[test]
fn test_av1_pixel_formats() {
    use zvd_lib::codec::av1::Av1EncoderBuilder;
    use zvd_lib::codec::VideoFrame;

    let mut encoder = Av1EncoderBuilder::new(320, 240)
        .speed_preset(10) // Fastest
        .build()
        .expect("Failed to create encoder");

    // Test YUV420P (should work)
    let mut frame_420 = VideoFrame::new(320, 240, PixelFormat::YUV420P);
    frame_420.data.push(Buffer::from_vec(vec![0u8; 320 * 240]));
    frame_420.linesize.push(320);
    frame_420
        .data
        .push(Buffer::from_vec(vec![128u8; 160 * 120]));
    frame_420.linesize.push(160);
    frame_420
        .data
        .push(Buffer::from_vec(vec![128u8; 160 * 120]));
    frame_420.linesize.push(160);

    let result = encoder.send_frame(&Frame::Video(frame_420));
    assert!(result.is_ok(), "YUV420P should be supported");
}

/// Test AV1 encoder builder advanced options
#[test]
fn test_av1_encoder_builder_options() {
    use zvd_lib::codec::av1::Av1EncoderBuilder;

    // Test with bitrate mode
    let encoder = Av1EncoderBuilder::new(1280, 720)
        .speed_preset(7)
        .bitrate(2000) // 2 Mbps
        .max_keyframe_interval(120)
        .threads(4)
        .build();
    assert!(encoder.is_ok(), "Bitrate mode should work");

    // Test with low latency
    let encoder = Av1EncoderBuilder::new(1280, 720)
        .speed_preset(9)
        .low_latency(true)
        .rdo_lookahead_frames(5)
        .build();
    assert!(encoder.is_ok(), "Low latency mode should work");

    // Test with tiles
    let encoder = Av1EncoderBuilder::new(1920, 1080)
        .speed_preset(6)
        .tile_cols(2)
        .tile_rows(2)
        .threads(8)
        .build();
    assert!(encoder.is_ok(), "Tile-based encoding should work");
}

/// Test AV1 encoder metadata
#[test]
fn test_av1_encoder_metadata() {
    use zvd_lib::codec::av1::Av1EncoderBuilder;

    let encoder = Av1EncoderBuilder::new(640, 480)
        .speed_preset(7)
        .time_base(1, 60)
        .build()
        .unwrap();

    assert_eq!(encoder.speed_preset(), 7);
    assert_eq!(encoder.time_base().num, 1);
    assert_eq!(encoder.time_base().den, 60);
}

/// Test AV1 decoder threading
#[test]
fn test_av1_decoder_threading() {
    use zvd_lib::codec::av1::Av1Decoder;

    // Default (auto threads)
    let decoder = Av1Decoder::new();
    assert!(decoder.is_ok(), "Auto-threading should work");

    // Specific thread count
    let decoder = Av1Decoder::with_threads(4);
    assert!(decoder.is_ok(), "4 threads should work");

    let decoder = Av1Decoder::with_threads(8);
    assert!(decoder.is_ok(), "8 threads should work");

    // Single-threaded
    let decoder = Av1Decoder::with_threads(1);
    assert!(decoder.is_ok(), "Single-threaded should work");
}

/// Verify that AV1 is listed in available codecs
#[test]
fn test_av1_in_codec_list() {
    use zvd_lib::codec::get_codec_info;

    // Verify AV1 is recognized as a valid codec
    assert!(
        get_codec_info("av1").is_some(),
        "AV1 should be in codec registry"
    );
}
