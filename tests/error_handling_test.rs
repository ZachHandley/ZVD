//! Integration tests for error handling and edge cases
//!
//! These tests verify that ZVD handles malformed data, invalid inputs,
//! and error conditions gracefully without panicking.

use zvd_lib::codec::{create_encoder, create_decoder, Encoder, Decoder, Frame, VideoFrame, AudioFrame};
use zvd_lib::format::Packet;
use zvd_lib::util::{Buffer, PixelFormat, SampleFormat, Timestamp};

/// Test encoder with invalid codec name
#[test]
fn test_invalid_codec_name() {
    let result = create_encoder("nonexistent_codec", 640, 480);
    assert!(result.is_err(), "Should reject invalid codec name");

    let result = create_decoder("fake_codec");
    assert!(result.is_err(), "Should reject invalid codec name");
}

/// Test encoder with zero dimensions
#[test]
fn test_zero_dimensions() {
    let result = create_encoder("av1", 0, 480);
    assert!(result.is_err(), "Should reject zero width");

    let result = create_encoder("av1", 640, 0);
    assert!(result.is_err(), "Should reject zero height");

    let result = create_encoder("av1", 0, 0);
    assert!(result.is_err(), "Should reject zero dimensions");
}

/// Test encoder with excessive dimensions
#[test]
fn test_excessive_dimensions() {
    // Very large dimensions (should fail gracefully)
    let result = create_encoder("av1", 100000, 100000);
    // May fail due to memory or codec limits, but shouldn't panic
}

/// Test decoder with empty packet
#[test]
fn test_decoder_empty_packet() {
    let mut decoder = create_decoder("av1").expect("Failed to create decoder");

    let empty_packet = Packet {
        stream_index: 0,
        data: Buffer::from_vec(vec![]),
        pts: Timestamp::new(0),
        dts: Timestamp::new(0),
        duration: Timestamp::new(1),
        keyframe: false,
    };

    // Should handle empty packet without panicking
    let result = decoder.send_packet(&empty_packet);
    // May error, but shouldn't panic
}

/// Test decoder with malformed packet data
#[test]
fn test_decoder_malformed_packet() {
    let mut decoder = create_decoder("av1").expect("Failed to create decoder");

    let malformed_packet = Packet {
        stream_index: 0,
        data: Buffer::from_vec(vec![0xFF; 100]), // Random data
        pts: Timestamp::new(0),
        dts: Timestamp::new(0),
        duration: Timestamp::new(1),
        keyframe: true,
    };

    // Should handle malformed data without panicking
    let result = decoder.send_packet(&malformed_packet);
    // Will likely error, but shouldn't panic
}

/// Test encoder with invalid frame dimensions
#[test]
fn test_encoder_wrong_dimensions() {
    let mut encoder = create_encoder("av1", 640, 480).expect("Failed to create encoder");

    // Create frame with wrong dimensions
    let mut frame = VideoFrame::new(1920, 1080, PixelFormat::YUV420P);
    frame.data.push(Buffer::from_vec(vec![0u8; 1920 * 1080]));
    frame.linesize.push(1920);
    frame.data.push(Buffer::from_vec(vec![0u8; 960 * 540]));
    frame.linesize.push(960);
    frame.data.push(Buffer::from_vec(vec![0u8; 960 * 540]));
    frame.linesize.push(960);
    frame.pts = Timestamp::new(0);

    // Should reject frame with wrong dimensions
    let result = encoder.send_frame(&Frame::Video(frame));
    assert!(result.is_err(), "Should reject frame with wrong dimensions");
}

/// Test encoder with incomplete frame data
#[test]
fn test_encoder_incomplete_frame() {
    let mut encoder = create_encoder("av1", 640, 480).expect("Failed to create encoder");

    // Create frame with missing planes
    let mut frame = VideoFrame::new(640, 480, PixelFormat::YUV420P);
    frame.data.push(Buffer::from_vec(vec![0u8; 640 * 480]));
    frame.linesize.push(640);
    // Missing U and V planes
    frame.pts = Timestamp::new(0);

    // Should handle incomplete frame data
    let result = encoder.send_frame(&Frame::Video(frame));
    // May error, but shouldn't panic
}

/// Test encoder with invalid timestamps
#[test]
fn test_encoder_invalid_timestamps() {
    let mut encoder = create_encoder("av1", 320, 240).expect("Failed to create encoder");

    // Create frame with invalid timestamp
    let mut frame = VideoFrame::new(320, 240, PixelFormat::YUV420P);
    frame.data.push(Buffer::from_vec(vec![0u8; 320 * 240]));
    frame.linesize.push(320);
    frame.data.push(Buffer::from_vec(vec![0u8; 160 * 120]));
    frame.linesize.push(160);
    frame.data.push(Buffer::from_vec(vec![0u8; 160 * 120]));
    frame.linesize.push(160);
    frame.pts = Timestamp::invalid();

    // Should handle invalid timestamp
    let result = encoder.send_frame(&Frame::Video(frame));
    // May work or error, but shouldn't panic
}

/// Test encoder with negative timestamps
#[test]
fn test_encoder_negative_timestamps() {
    let mut encoder = create_encoder("av1", 320, 240).expect("Failed to create encoder");

    // Create frame with negative timestamp
    let mut frame = VideoFrame::new(320, 240, PixelFormat::YUV420P);
    frame.data.push(Buffer::from_vec(vec![0u8; 320 * 240]));
    frame.linesize.push(320);
    frame.data.push(Buffer::from_vec(vec![0u8; 160 * 120]));
    frame.linesize.push(160);
    frame.data.push(Buffer::from_vec(vec![0u8; 160 * 120]));
    frame.linesize.push(160);
    frame.pts = Timestamp::new(-1);

    // Should handle negative timestamp
    let result = encoder.send_frame(&Frame::Video(frame));
    // May work or error depending on codec, but shouldn't panic
}

/// Test receiving frame before sending any
#[test]
fn test_receive_without_send() {
    let mut decoder = create_decoder("av1").expect("Failed to create decoder");

    // Try to receive frame without sending packet
    let result = decoder.receive_frame();
    assert!(result.is_err(), "Should error when no frames available");
}

/// Test receiving packet before sending any frames
#[test]
fn test_receive_packet_without_send() {
    let mut encoder = create_encoder("av1", 640, 480).expect("Failed to create encoder");

    // Try to receive packet without sending frame
    let result = encoder.receive_packet();
    assert!(result.is_err(), "Should error when no packets available");
}

/// Test multiple flushes
#[test]
fn test_multiple_flushes() {
    let mut encoder = create_encoder("av1", 640, 480).expect("Failed to create encoder");

    // Flush multiple times
    for _ in 0..5 {
        let result = encoder.flush();
        assert!(result.is_ok(), "Multiple flushes should not panic");
    }
}

/// Test filter with invalid dimensions
#[test]
fn test_scale_filter_invalid_dimensions() {
    use zvd_lib::filter::video::ScaleFilter;

    // Zero dimensions
    let filter = ScaleFilter::new(0, 480);
    // Should create but may error on processing

    let filter = ScaleFilter::new(640, 0);
    // Should create but may error on processing
}

/// Test crop filter with out-of-bounds region
#[test]
fn test_crop_filter_out_of_bounds() {
    use zvd_lib::filter::video::CropFilter;

    // Crop region larger than any reasonable frame
    let filter = CropFilter::new(0, 0, 100000, 100000);
    // Should create but may error on processing
}

/// Test volume filter with extreme values
#[test]
fn test_volume_filter_extreme_values() {
    use zvd_lib::filter::audio::VolumeFilter;

    // Very high gain
    let filter = VolumeFilter::new(1000.0);

    // Negative gain (should still create)
    let filter = VolumeFilter::new(-1.0);

    // NaN (should handle gracefully)
    let filter = VolumeFilter::new(f32::NAN);
}

/// Test resample filter with invalid rates
#[test]
fn test_resample_filter_invalid_rates() {
    use zvd_lib::filter::audio::ResampleFilter;

    // Zero sample rates
    let filter = ResampleFilter::new(0, 48000);

    let filter = ResampleFilter::new(48000, 0);

    // Extreme sample rates
    let filter = ResampleFilter::new(1_000_000, 48000);
}

/// Test audio frame with zero samples
#[test]
fn test_audio_frame_zero_samples() {
    let frame = AudioFrame::new(0, 2, SampleFormat::F32);
    assert_eq!(frame.nb_samples, 0);
    // Should create without panicking
}

/// Test audio frame with zero channels
#[test]
fn test_audio_frame_zero_channels() {
    let frame = AudioFrame::new(1024, 0, SampleFormat::F32);
    assert_eq!(frame.channels, 0);
    // Should create without panicking
}

/// Test video frame with mismatched linesize
#[test]
fn test_video_frame_mismatched_linesize() {
    let mut frame = VideoFrame::new(640, 480, PixelFormat::YUV420P);

    // Add data with wrong linesize
    frame.data.push(Buffer::from_vec(vec![0u8; 640 * 480]));
    frame.linesize.push(1920); // Wrong linesize
    frame.data.push(Buffer::from_vec(vec![0u8; 320 * 240]));
    frame.linesize.push(960); // Wrong linesize
    frame.data.push(Buffer::from_vec(vec![0u8; 320 * 240]));
    frame.linesize.push(960); // Wrong linesize

    // Frame is created but may cause issues during processing
}

/// Test packet with mismatched stream index
#[test]
fn test_packet_invalid_stream_index() {
    let packet = Packet {
        stream_index: 999, // Invalid stream index
        data: Buffer::from_vec(vec![0u8; 100]),
        pts: Timestamp::new(0),
        dts: Timestamp::new(0),
        duration: Timestamp::new(1),
        keyframe: false,
    };

    // Packet is created but may cause issues during demuxing
}

/// Test timestamp overflow
#[test]
fn test_timestamp_overflow() {
    let ts = Timestamp::new(i64::MAX);
    assert!(ts.is_valid());
    assert_eq!(ts.value(), i64::MAX);

    let ts = Timestamp::new(i64::MIN);
    assert_eq!(ts.value(), i64::MIN);
}

/// Test buffer with excessive size
#[test]
fn test_buffer_large_allocation() {
    // Try to create very large buffer (may fail due to memory)
    // Should not panic, just return error or succeed
    let size = 1024 * 1024; // 1 MB (reasonable for testing)
    let buffer = Buffer::from_vec(vec![0u8; size]);
    assert_eq!(buffer.len(), size);
}

/// Test codec info for nonexistent codec
#[test]
fn test_codec_info_nonexistent() {
    use zvd_lib::codec::get_codec_info;

    let info = get_codec_info("nonexistent");
    assert!(info.is_none(), "Should return None for nonexistent codec");

    let info = get_codec_info("");
    assert!(info.is_none(), "Should return None for empty codec name");
}

/// Test H.264 encoder with invalid configuration
#[test]
#[cfg(feature = "h264")]
fn test_h264_invalid_config() {
    use zvd_lib::codec::h264::{H264Encoder, H264EncoderConfig};
    use zvd_lib::util::Rational;

    // Zero bitrate
    let config = H264EncoderConfig {
        width: 640,
        height: 480,
        bitrate: 0,
        framerate: Rational::new(30, 1),
        keyframe_interval: 60,
    };
    let result = H264Encoder::new(config);
    // May error or use default, but shouldn't panic

    // Invalid framerate
    let config = H264EncoderConfig {
        width: 640,
        height: 480,
        bitrate: 500_000,
        framerate: Rational::new(0, 1),
        keyframe_interval: 60,
    };
    let result = H264Encoder::new(config);
    // May error or use default, but shouldn't panic
}

/// Test that errors are properly propagated
#[test]
fn test_error_propagation() {
    use zvd_lib::error::Error;

    // Create an error
    let err = Error::invalid_input("test error");
    let err_str = format!("{}", err);
    assert!(err_str.contains("Invalid input") || err_str.contains("test error"));

    // Verify error types exist
    let _err = Error::TryAgain;
    let _err = Error::EndOfFile;
}

/// Test concurrent encoder access (thread safety)
#[test]
fn test_encoder_thread_safety() {
    use std::sync::{Arc, Mutex};
    use std::thread;

    let encoder = create_encoder("av1", 320, 240).expect("Failed to create encoder");
    let encoder = Arc::new(Mutex::new(encoder));

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let encoder = Arc::clone(&encoder);
            thread::spawn(move || {
                let mut encoder = encoder.lock().unwrap();

                let mut frame = VideoFrame::new(320, 240, PixelFormat::YUV420P);
                frame.data.push(Buffer::from_vec(vec![100u8; 320 * 240]));
                frame.linesize.push(320);
                frame.data.push(Buffer::from_vec(vec![128u8; 160 * 120]));
                frame.linesize.push(160);
                frame.data.push(Buffer::from_vec(vec![128u8; 160 * 120]));
                frame.linesize.push(160);
                frame.pts = Timestamp::new(i);

                let _ = encoder.send_frame(&Frame::Video(frame));
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

/// Test memory safety with clone and drop
#[test]
fn test_buffer_clone_and_drop() {
    let buffer1 = Buffer::from_vec(vec![1, 2, 3, 4, 5]);
    let buffer2 = buffer1.clone();

    assert_eq!(buffer1.len(), buffer2.len());
    assert_eq!(buffer1.as_slice(), buffer2.as_slice());

    // Both buffers should be dropped safely
    drop(buffer1);
    drop(buffer2);
}

/// Test pixel format edge cases
#[test]
fn test_pixel_format_edge_cases() {
    // Test all pixel formats can be created
    let formats = [
        PixelFormat::YUV420P,
        PixelFormat::YUV422P,
        PixelFormat::YUV444P,
        PixelFormat::YUV420P10LE,
        PixelFormat::YUV422P10LE,
        PixelFormat::YUV444P10LE,
        PixelFormat::YUV420P12LE,
        PixelFormat::RGB24,
    ];

    for &format in &formats {
        let frame = VideoFrame::new(640, 480, format);
        assert_eq!(frame.format, format);
    }
}

/// Test sample format edge cases
#[test]
fn test_sample_format_edge_cases() {
    let formats = [
        SampleFormat::U8,
        SampleFormat::I16,
        SampleFormat::I32,
        SampleFormat::F32,
        SampleFormat::F64,
        SampleFormat::U8P,
        SampleFormat::I16P,
        SampleFormat::F32P,
    ];

    for &format in &formats {
        let frame = AudioFrame::new(1024, 2, format);
        assert_eq!(frame.format, format);
    }
}
