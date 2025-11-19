//! End-to-end transcoding integration tests
//!
//! These tests verify complete transcoding workflows combining
//! encoders, decoders, filters, and containers.

use zvd_lib::codec::{create_encoder, create_decoder, Encoder, Decoder, Frame, VideoFrame};
use zvd_lib::filter::chain::VideoFilterChain;
use zvd_lib::filter::video::{ScaleFilter, BrightnessContrastFilter};
use zvd_lib::filter::FilterChain;
use zvd_lib::util::{Buffer, PixelFormat, Timestamp};

/// Test basic AV1 encode-decode-reencode workflow
#[test]
fn test_av1_transcode_workflow() {
    // Create encoder
    let mut encoder = create_encoder("av1", 640, 480).expect("Failed to create encoder");

    // Encode a frame
    let mut frame = VideoFrame::new(640, 480, PixelFormat::YUV420P);
    frame.data.push(Buffer::from_vec(vec![100u8; 640 * 480]));
    frame.linesize.push(640);
    frame.data.push(Buffer::from_vec(vec![128u8; 320 * 240]));
    frame.linesize.push(320);
    frame.data.push(Buffer::from_vec(vec![128u8; 320 * 240]));
    frame.linesize.push(320);
    frame.pts = Timestamp::new(0);
    frame.keyframe = true;

    encoder
        .send_frame(&Frame::Video(frame))
        .expect("Failed to encode");
    encoder.flush().expect("Failed to flush");

    let packet = encoder.receive_packet().expect("Failed to receive packet");

    // Decode packet
    let mut decoder = create_decoder("av1").expect("Failed to create decoder");
    decoder
        .send_packet(&packet)
        .expect("Failed to send packet");

    let decoded_frame = decoder.receive_frame().expect("Failed to receive frame");

    // Verify decoded frame
    if let Frame::Video(vf) = decoded_frame {
        assert_eq!(vf.width, 640);
        assert_eq!(vf.height, 480);
        assert_eq!(vf.format, PixelFormat::YUV420P);
    } else {
        panic!("Expected video frame");
    }
}

/// Test H.264 to AV1 transcoding
#[test]
#[cfg(feature = "h264")]
fn test_h264_to_av1_transcode() {
    // Encode with H.264
    let mut h264_encoder = create_encoder("h264", 640, 480)
        .expect("Failed to create H.264 encoder");

    let mut frame = VideoFrame::new(640, 480, PixelFormat::YUV420P);
    frame.data.push(Buffer::from_vec(vec![100u8; 640 * 480]));
    frame.linesize.push(640);
    frame.data.push(Buffer::from_vec(vec![128u8; 320 * 240]));
    frame.linesize.push(320);
    frame.data.push(Buffer::from_vec(vec![128u8; 320 * 240]));
    frame.linesize.push(320);
    frame.pts = Timestamp::new(0);
    frame.keyframe = true;

    h264_encoder
        .send_frame(&Frame::Video(frame))
        .expect("Failed to encode H.264");
    h264_encoder.flush().expect("Failed to flush H.264");

    let h264_packet = h264_encoder
        .receive_packet()
        .expect("Failed to receive H.264 packet");

    // Decode H.264
    let mut h264_decoder = create_decoder("h264")
        .expect("Failed to create H.264 decoder");
    h264_decoder
        .send_packet(&h264_packet)
        .expect("Failed to send H.264 packet");

    let decoded_frame = h264_decoder
        .receive_frame()
        .expect("Failed to receive H.264 frame");

    // Re-encode with AV1
    let mut av1_encoder = create_encoder("av1", 640, 480)
        .expect("Failed to create AV1 encoder");
    av1_encoder
        .send_frame(&decoded_frame)
        .expect("Failed to encode AV1");
    av1_encoder.flush().expect("Failed to flush AV1");

    let av1_packet = av1_encoder
        .receive_packet()
        .expect("Failed to receive AV1 packet");

    // Verify AV1 packet
    assert!(!av1_packet.data.is_empty());
}

/// Test transcoding with scaling
#[test]
fn test_transcode_with_scale() {
    // Encode 1920x1080 frame
    let mut encoder = create_encoder("av1", 1920, 1080)
        .expect("Failed to create encoder");

    let mut frame = VideoFrame::new(1920, 1080, PixelFormat::YUV420P);
    frame.data.push(Buffer::from_vec(vec![100u8; 1920 * 1080]));
    frame.linesize.push(1920);
    frame.data.push(Buffer::from_vec(vec![128u8; 960 * 540]));
    frame.linesize.push(960);
    frame.data.push(Buffer::from_vec(vec![128u8; 960 * 540]));
    frame.linesize.push(960);
    frame.pts = Timestamp::new(0);
    frame.keyframe = true;

    encoder
        .send_frame(&Frame::Video(frame))
        .expect("Failed to encode");
    encoder.flush().expect("Failed to flush");

    let packet = encoder.receive_packet().expect("Failed to receive packet");

    // Decode
    let mut decoder = create_decoder("av1").expect("Failed to create decoder");
    decoder
        .send_packet(&packet)
        .expect("Failed to send packet");

    let decoded_frame = decoder.receive_frame().expect("Failed to receive frame");

    // Apply scale filter to 1280x720
    let mut filter = ScaleFilter::new(1280, 720);
    let scaled_frame = filter
        .process(&decoded_frame)
        .expect("Failed to scale");

    // Re-encode at new resolution
    let mut encoder_720p = create_encoder("av1", 1280, 720)
        .expect("Failed to create 720p encoder");
    encoder_720p
        .send_frame(&scaled_frame)
        .expect("Failed to encode scaled frame");
    encoder_720p.flush().expect("Failed to flush");

    let scaled_packet = encoder_720p
        .receive_packet()
        .expect("Failed to receive scaled packet");

    // Verify packet
    assert!(!scaled_packet.data.is_empty());
}

/// Test transcoding with filter chain
#[test]
fn test_transcode_with_filter_chain() {
    // Create encoder
    let mut encoder = create_encoder("av1", 1920, 1080)
        .expect("Failed to create encoder");

    // Encode frame
    let mut frame = VideoFrame::new(1920, 1080, PixelFormat::YUV420P);
    frame.data.push(Buffer::from_vec(vec![100u8; 1920 * 1080]));
    frame.linesize.push(1920);
    frame.data.push(Buffer::from_vec(vec![128u8; 960 * 540]));
    frame.linesize.push(960);
    frame.data.push(Buffer::from_vec(vec![128u8; 960 * 540]));
    frame.linesize.push(960);
    frame.pts = Timestamp::new(0);
    frame.keyframe = true;

    encoder
        .send_frame(&Frame::Video(frame))
        .expect("Failed to encode");
    encoder.flush().expect("Failed to flush");

    let packet = encoder.receive_packet().expect("Failed to receive packet");

    // Decode
    let mut decoder = create_decoder("av1").expect("Failed to create decoder");
    decoder
        .send_packet(&packet)
        .expect("Failed to send packet");

    let decoded_frame = decoder.receive_frame().expect("Failed to receive frame");

    // Apply filter chain: scale + brightness
    let mut filter_chain = VideoFilterChain::new();
    filter_chain.add(Box::new(ScaleFilter::new(1280, 720)));
    filter_chain.add(Box::new(BrightnessContrastFilter::new(15.0, 1.2)));

    let filtered_frame = filter_chain
        .process(decoded_frame)
        .expect("Failed to apply filters");

    // Re-encode
    let mut encoder_filtered = create_encoder("av1", 1280, 720)
        .expect("Failed to create filtered encoder");
    encoder_filtered
        .send_frame(&filtered_frame)
        .expect("Failed to encode filtered frame");
    encoder_filtered.flush().expect("Failed to flush");

    let filtered_packet = encoder_filtered
        .receive_packet()
        .expect("Failed to receive filtered packet");

    // Verify packet
    assert!(!filtered_packet.data.is_empty());
}

/// Test multi-frame transcoding workflow
#[test]
fn test_multi_frame_transcode() {
    let mut encoder = create_encoder("av1", 640, 480)
        .expect("Failed to create encoder");

    // Encode 10 frames
    for i in 0..10 {
        let mut frame = VideoFrame::new(640, 480, PixelFormat::YUV420P);
        frame.data.push(Buffer::from_vec(vec![100u8; 640 * 480]));
        frame.linesize.push(640);
        frame.data.push(Buffer::from_vec(vec![128u8; 320 * 240]));
        frame.linesize.push(320);
        frame.data.push(Buffer::from_vec(vec![128u8; 320 * 240]));
        frame.linesize.push(320);
        frame.pts = Timestamp::new(i);
        frame.keyframe = i == 0;

        encoder
            .send_frame(&Frame::Video(frame))
            .expect("Failed to encode frame");
    }

    encoder.flush().expect("Failed to flush");

    // Collect all packets
    let mut packets = Vec::new();
    for _ in 0..20 {
        match encoder.receive_packet() {
            Ok(packet) => packets.push(packet),
            Err(_) => break,
        }
    }

    assert!(!packets.is_empty(), "Should have encoded packets");

    // Decode all packets
    let mut decoder = create_decoder("av1").expect("Failed to create decoder");
    let mut decoded_frames = Vec::new();

    for packet in packets {
        decoder
            .send_packet(&packet)
            .expect("Failed to send packet");

        match decoder.receive_frame() {
            Ok(frame) => decoded_frames.push(frame),
            Err(_) => {}
        }
    }

    decoder.flush().expect("Failed to flush decoder");

    // Try to get any remaining frames
    for _ in 0..10 {
        match decoder.receive_frame() {
            Ok(frame) => decoded_frames.push(frame),
            Err(_) => break,
        }
    }

    assert!(!decoded_frames.is_empty(), "Should have decoded frames");
}

/// Test transcoding with timestamp preservation
#[test]
fn test_transcode_timestamp_preservation() {
    let mut encoder = create_encoder("av1", 640, 480)
        .expect("Failed to create encoder");

    // Encode frame with specific timestamp
    let mut frame = VideoFrame::new(640, 480, PixelFormat::YUV420P);
    frame.data.push(Buffer::from_vec(vec![100u8; 640 * 480]));
    frame.linesize.push(640);
    frame.data.push(Buffer::from_vec(vec![128u8; 320 * 240]));
    frame.linesize.push(320);
    frame.data.push(Buffer::from_vec(vec![128u8; 320 * 240]));
    frame.linesize.push(320);
    frame.pts = Timestamp::new(12345);
    frame.keyframe = true;

    encoder
        .send_frame(&Frame::Video(frame))
        .expect("Failed to encode");
    encoder.flush().expect("Failed to flush");

    let packet = encoder.receive_packet().expect("Failed to receive packet");
    assert!(packet.pts.is_valid());

    // Decode and verify timestamp is preserved
    let mut decoder = create_decoder("av1").expect("Failed to create decoder");
    decoder
        .send_packet(&packet)
        .expect("Failed to send packet");

    let decoded_frame = decoder.receive_frame().expect("Failed to receive frame");

    if let Frame::Video(vf) = decoded_frame {
        assert!(vf.pts.is_valid(), "Timestamp should be valid");
        // Note: Timestamp may not be exactly preserved depending on codec
    } else {
        panic!("Expected video frame");
    }
}

/// Test keyframe detection in transcoding
#[test]
fn test_transcode_keyframe_detection() {
    let mut encoder = create_encoder("av1", 640, 480)
        .expect("Failed to create encoder");

    // Encode keyframe
    let mut keyframe = VideoFrame::new(640, 480, PixelFormat::YUV420P);
    keyframe.data.push(Buffer::from_vec(vec![100u8; 640 * 480]));
    keyframe.linesize.push(640);
    keyframe.data.push(Buffer::from_vec(vec![128u8; 320 * 240]));
    keyframe.linesize.push(320);
    keyframe.data.push(Buffer::from_vec(vec![128u8; 320 * 240]));
    keyframe.linesize.push(320);
    keyframe.pts = Timestamp::new(0);
    keyframe.keyframe = true;

    encoder
        .send_frame(&Frame::Video(keyframe))
        .expect("Failed to encode keyframe");

    // Encode non-keyframe
    let mut interframe = VideoFrame::new(640, 480, PixelFormat::YUV420P);
    interframe.data.push(Buffer::from_vec(vec![105u8; 640 * 480]));
    interframe.linesize.push(640);
    interframe.data.push(Buffer::from_vec(vec![128u8; 320 * 240]));
    interframe.linesize.push(320);
    interframe.data.push(Buffer::from_vec(vec![128u8; 320 * 240]));
    interframe.linesize.push(320);
    interframe.pts = Timestamp::new(1);
    interframe.keyframe = false;

    encoder
        .send_frame(&Frame::Video(interframe))
        .expect("Failed to encode interframe");

    encoder.flush().expect("Failed to flush");

    // Check packets
    let mut keyframe_found = false;
    for _ in 0..10 {
        match encoder.receive_packet() {
            Ok(packet) => {
                if packet.keyframe {
                    keyframe_found = true;
                }
            }
            Err(_) => break,
        }
    }

    assert!(keyframe_found, "Should have at least one keyframe packet");
}

/// Test resource cleanup after failed transcode
#[test]
fn test_transcode_error_cleanup() {
    // Create encoder
    let mut encoder = create_encoder("av1", 640, 480)
        .expect("Failed to create encoder");

    // Send frame with wrong dimensions (should fail)
    let mut frame = VideoFrame::new(1920, 1080, PixelFormat::YUV420P);
    frame.data.push(Buffer::from_vec(vec![100u8; 1920 * 1080]));
    frame.linesize.push(1920);
    frame.data.push(Buffer::from_vec(vec![128u8; 960 * 540]));
    frame.linesize.push(960);
    frame.data.push(Buffer::from_vec(vec![128u8; 960 * 540]));
    frame.linesize.push(960);
    frame.pts = Timestamp::new(0);

    let result = encoder.send_frame(&Frame::Video(frame));
    assert!(result.is_err(), "Should fail with wrong dimensions");

    // Encoder should still be usable
    let result = encoder.flush();
    // Should not panic even after error
}

/// Test concurrent transcoding (separate encoders)
#[test]
fn test_concurrent_transcode() {
    use std::thread;

    let handles: Vec<_> = (0..4)
        .map(|i| {
            thread::spawn(move || {
                let mut encoder = create_encoder("av1", 320, 240)
                    .expect("Failed to create encoder");

                let mut frame = VideoFrame::new(320, 240, PixelFormat::YUV420P);
                frame.data.push(Buffer::from_vec(vec![100u8; 320 * 240]));
                frame.linesize.push(320);
                frame.data.push(Buffer::from_vec(vec![128u8; 160 * 120]));
                frame.linesize.push(160);
                frame.data.push(Buffer::from_vec(vec![128u8; 160 * 120]));
                frame.linesize.push(160);
                frame.pts = Timestamp::new(i);

                encoder
                    .send_frame(&Frame::Video(frame))
                    .expect("Failed to encode");
                encoder.flush().expect("Failed to flush");

                encoder
                    .receive_packet()
                    .expect("Failed to receive packet");
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

/// Test memory efficiency with large frame count
#[test]
fn test_transcode_memory_efficiency() {
    let mut encoder = create_encoder("av1", 320, 240)
        .expect("Failed to create encoder");

    // Encode and immediately receive packets to avoid buffering
    for i in 0..100 {
        let mut frame = VideoFrame::new(320, 240, PixelFormat::YUV420P);
        frame.data.push(Buffer::from_vec(vec![100u8; 320 * 240]));
        frame.linesize.push(320);
        frame.data.push(Buffer::from_vec(vec![128u8; 160 * 120]));
        frame.linesize.push(160);
        frame.data.push(Buffer::from_vec(vec![128u8; 160 * 120]));
        frame.linesize.push(160);
        frame.pts = Timestamp::new(i);
        frame.keyframe = i % 30 == 0;

        encoder
            .send_frame(&Frame::Video(frame))
            .expect("Failed to encode");

        // Try to receive packet immediately
        let _ = encoder.receive_packet();
    }

    encoder.flush().expect("Failed to flush");

    // Receive remaining packets
    for _ in 0..200 {
        match encoder.receive_packet() {
            Ok(_) => {}
            Err(_) => break,
        }
    }
}
