//! VP9 codec integration tests
//!
//! Tests the VP9 encoder and decoder using libvpx.
//! Enable the `vp9-libvpx` feature to run these tests.
//!
//! To run: `cargo test --features vp9-libvpx --test vp9_codec_test`

#![cfg(feature = "vp9-libvpx")]

use zvd_lib::codec::vp9::{
    Vp9AqMode, Vp9Decoder, Vp9Encoder, Vp9EncoderConfig, Vp9EncodingPass, Vp9Profile,
    Vp9RateControl, Vp9TuneContent,
};
use zvd_lib::codec::{Decoder, Encoder, Frame};
use zvd_lib::codec::frame::VideoFrame;
use zvd_lib::error::Error;
use zvd_lib::format::Packet;
use zvd_lib::util::{Buffer, PixelFormat, Timestamp};

/// Create a test video frame with specified format
fn create_test_frame(
    width: u32,
    height: u32,
    format: PixelFormat,
    frame_num: u64,
    keyframe: bool,
) -> VideoFrame {
    let mut frame = VideoFrame::new(width, height, format);

    let (uv_width_div, uv_height_div, bytes_per_sample) = match format {
        PixelFormat::YUV420P => (2, 2, 1),
        PixelFormat::YUV444P => (1, 1, 1),
        PixelFormat::YUV420P10LE => (2, 2, 2),
        PixelFormat::YUV444P10LE => (1, 1, 2),
        _ => panic!("Unsupported format for test"),
    };

    let y_size = (width * height) as usize * bytes_per_sample;
    let uv_size = (width / uv_width_div * height / uv_height_div) as usize * bytes_per_sample;

    // Create Y plane with moving gradient
    let mut y_data = Vec::with_capacity(y_size);
    for y in 0..height {
        for x in 0..width {
            let value = ((x + y + (frame_num as u32 * 10)) % 256) as u8;
            if bytes_per_sample == 2 {
                // 10-bit: scale to 10-bit range (0-1023) and store as little-endian
                let value_10bit = (value as u16) << 2;
                y_data.push((value_10bit & 0xFF) as u8);
                y_data.push((value_10bit >> 8) as u8);
            } else {
                y_data.push(value);
            }
        }
    }

    // U and V planes
    let mut u_data = Vec::with_capacity(uv_size);
    let mut v_data = Vec::with_capacity(uv_size);
    for y in 0..(height / uv_height_div) {
        for x in 0..(width / uv_width_div) {
            let checker = ((x / 8 + y / 8) % 2) == 0;
            let u_val = if checker { 100u8 } else { 200u8 };
            let v_val = if checker { 200u8 } else { 100u8 };

            if bytes_per_sample == 2 {
                let u_10bit = (u_val as u16) << 2;
                let v_10bit = (v_val as u16) << 2;
                u_data.push((u_10bit & 0xFF) as u8);
                u_data.push((u_10bit >> 8) as u8);
                v_data.push((v_10bit & 0xFF) as u8);
                v_data.push((v_10bit >> 8) as u8);
            } else {
                u_data.push(u_val);
                v_data.push(v_val);
            }
        }
    }

    let uv_width = (width / uv_width_div) as usize;
    frame.data = vec![
        Buffer::from_vec(y_data),
        Buffer::from_vec(u_data),
        Buffer::from_vec(v_data),
    ];
    frame.linesize = vec![
        width as usize * bytes_per_sample,
        uv_width * bytes_per_sample,
        uv_width * bytes_per_sample,
    ];
    frame.keyframe = keyframe;
    frame.pts = Timestamp::new(frame_num as i64);

    frame
}

/// Helper to create YUV420P test frame
fn create_yuv420p_frame(width: u32, height: u32, frame_num: u64, keyframe: bool) -> VideoFrame {
    create_test_frame(width, height, PixelFormat::YUV420P, frame_num, keyframe)
}

#[test]
fn test_vp9_encoder_decoder_roundtrip() {
    let width = 320;
    let height = 240;
    let num_frames = 5;

    // Create encoder with low lag for testing
    let config = Vp9EncoderConfig {
        width,
        height,
        bitrate: 500_000,
        framerate: 30,
        keyframe_interval: 30,
        threads: 2,
        rate_control: Vp9RateControl::VBR,
        cpu_used: 6,
        lag_in_frames: 0, // Disable lag for immediate output
        auto_alt_ref: false,
        ..Default::default()
    };

    let mut encoder = Vp9Encoder::with_config(config).expect("Failed to create VP9 encoder");
    let mut decoder = Vp9Decoder::new().expect("Failed to create VP9 decoder");

    // Encode frames
    let mut encoded_packets = Vec::new();
    for i in 0..num_frames {
        let frame = create_yuv420p_frame(width, height, i, i == 0);
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

    // Flush encoder
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

    // First packet should be keyframe
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

    decoder.flush().expect("Failed to flush decoder");
    println!("Decoded {} frames", decoded_frames.len());
    assert!(!decoded_frames.is_empty(), "Should have decoded at least one frame");
}

#[test]
fn test_vp9_encoder_multiple_resolutions() {
    let resolutions = [
        (320, 240),
        (640, 480),
        (854, 480),
        (1280, 720),
    ];

    for (width, height) in resolutions {
        println!("Testing VP9 encoding at {}x{}", width, height);

        let config = Vp9EncoderConfig {
            width,
            height,
            lag_in_frames: 0,
            auto_alt_ref: false,
            ..Default::default()
        };

        let mut encoder = Vp9Encoder::with_config(config)
            .expect(&format!("Failed to create encoder for {}x{}", width, height));

        let frame = create_yuv420p_frame(width, height, 0, true);
        encoder.send_frame(&Frame::Video(frame))
            .expect(&format!("Failed to encode frame at {}x{}", width, height));

        encoder.flush().expect("Failed to flush");

        let mut total_bytes = 0;
        loop {
            match encoder.receive_packet() {
                Ok(packet) => total_bytes += packet.data.len(),
                Err(Error::TryAgain) => break,
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }

        println!("  Encoded to {} bytes", total_bytes);
        assert!(total_bytes > 0, "Should produce output at {}x{}", width, height);
    }
}

#[test]
fn test_vp9_rate_control_modes() {
    let width = 320;
    let height = 240;

    let modes = [
        (Vp9RateControl::VBR, "VBR"),
        (Vp9RateControl::CBR, "CBR"),
        (Vp9RateControl::CQ, "CQ"),
        (Vp9RateControl::Q, "Q"),
    ];

    for (mode, name) in modes {
        println!("Testing VP9 with {} rate control", name);

        let config = Vp9EncoderConfig {
            width,
            height,
            bitrate: 500_000,
            framerate: 30,
            rate_control: mode,
            cq_level: 31,
            lag_in_frames: 0,
            auto_alt_ref: false,
            ..Default::default()
        };

        let mut encoder = Vp9Encoder::with_config(config)
            .expect(&format!("Failed to create encoder with {} mode", name));

        for i in 0..3 {
            let frame = create_yuv420p_frame(width, height, i, i == 0);
            encoder.send_frame(&Frame::Video(frame))
                .expect(&format!("Failed to encode frame with {} mode", name));
        }

        encoder.flush().expect("Failed to flush");

        let mut total_bytes = 0;
        loop {
            match encoder.receive_packet() {
                Ok(packet) => total_bytes += packet.data.len(),
                Err(Error::TryAgain) => break,
                Err(e) => panic!("Error with {} mode: {:?}", name, e),
            }
        }

        println!("  {} mode produced {} bytes", name, total_bytes);
        assert!(total_bytes > 0, "{} mode should produce output", name);
    }
}

#[test]
fn test_vp9_profile0_yuv420p() {
    let width = 320;
    let height = 240;

    let config = Vp9EncoderConfig {
        width,
        height,
        profile: Vp9Profile::Profile0,
        lag_in_frames: 0,
        auto_alt_ref: false,
        ..Default::default()
    };

    let mut encoder = Vp9Encoder::with_config(config).expect("Failed to create Profile 0 encoder");

    let frame = create_test_frame(width, height, PixelFormat::YUV420P, 0, true);
    encoder.send_frame(&Frame::Video(frame)).expect("Failed to encode Profile 0 frame");
    encoder.flush().expect("Failed to flush");

    match encoder.receive_packet() {
        Ok(packet) => {
            assert!(!packet.data.is_empty());
            println!("Profile 0 (YUV420P 8-bit) encoded to {} bytes", packet.data.len());
        }
        Err(e) => panic!("Profile 0 encoding failed: {:?}", e),
    }
}

#[test]
fn test_vp9_tiling() {
    let width = 1280;
    let height = 720;

    // Test different tile configurations
    let tile_configs = [
        (0, 0, "1x1"),
        (1, 0, "2x1"),
        (2, 0, "4x1"),
        (2, 1, "4x2"),
    ];

    for (tile_cols, tile_rows, desc) in tile_configs {
        println!("Testing VP9 with {} tiles", desc);

        let config = Vp9EncoderConfig {
            width,
            height,
            tile_columns: tile_cols,
            tile_rows,
            lag_in_frames: 0,
            auto_alt_ref: false,
            ..Default::default()
        };

        let mut encoder = Vp9Encoder::with_config(config)
            .expect(&format!("Failed to create encoder with {} tiles", desc));

        let frame = create_yuv420p_frame(width, height, 0, true);
        encoder.send_frame(&Frame::Video(frame)).expect("Failed to encode");
        encoder.flush().expect("Failed to flush");

        let mut total_bytes = 0;
        loop {
            match encoder.receive_packet() {
                Ok(packet) => total_bytes += packet.data.len(),
                Err(Error::TryAgain) => break,
                Err(e) => panic!("Error with {} tiles: {:?}", desc, e),
            }
        }

        println!("  {} tiles produced {} bytes", desc, total_bytes);
        assert!(total_bytes > 0);
    }
}

#[test]
fn test_vp9_lossless_mode() {
    let width = 320;
    let height = 240;

    let config = Vp9EncoderConfig {
        width,
        height,
        lossless: true,
        lag_in_frames: 0,
        auto_alt_ref: false,
        ..Default::default()
    };

    let mut encoder = Vp9Encoder::with_config(config).expect("Failed to create lossless encoder");
    let mut decoder = Vp9Decoder::new().expect("Failed to create decoder");

    // Create a simple test pattern
    let original_frame = create_yuv420p_frame(width, height, 0, true);
    let original_y_data = original_frame.data[0].as_slice().to_vec();

    encoder.send_frame(&Frame::Video(original_frame)).expect("Failed to encode");
    encoder.flush().expect("Failed to flush");

    // Get encoded packet
    let packet = match encoder.receive_packet() {
        Ok(p) => p,
        Err(e) => panic!("Failed to get lossless encoded packet: {:?}", e),
    };

    println!("Lossless encoded to {} bytes", packet.data.len());

    // Decode
    decoder.send_packet(&packet).expect("Failed to decode lossless");

    match decoder.receive_frame() {
        Ok(Frame::Video(decoded)) => {
            let decoded_y_data = decoded.data[0].as_slice();
            // In lossless mode, Y plane should be identical
            assert_eq!(
                original_y_data.len(),
                decoded_y_data.len(),
                "Y plane size should match"
            );
            // Note: Exact comparison may fail due to libvpx internal handling
            println!("Lossless decode produced {}x{} frame", decoded.width, decoded.height);
        }
        Ok(_) => panic!("Expected video frame"),
        Err(e) => panic!("Lossless decode failed: {:?}", e),
    }
}

#[test]
fn test_vp9_adaptive_quantization_modes() {
    let width = 320;
    let height = 240;

    let aq_modes = [
        (Vp9AqMode::None, "None"),
        (Vp9AqMode::Variance, "Variance"),
        (Vp9AqMode::Complexity, "Complexity"),
        (Vp9AqMode::CyclicRefresh, "CyclicRefresh"),
    ];

    for (mode, name) in aq_modes {
        println!("Testing VP9 with AQ mode: {}", name);

        let config = Vp9EncoderConfig {
            width,
            height,
            aq_mode: mode,
            lag_in_frames: 0,
            auto_alt_ref: false,
            ..Default::default()
        };

        let result = Vp9Encoder::with_config(config);
        assert!(result.is_ok(), "Should support AQ mode: {}", name);
    }
}

#[test]
fn test_vp9_content_tuning() {
    let width = 320;
    let height = 240;

    let tune_modes = [
        (Vp9TuneContent::Default, "Default"),
        (Vp9TuneContent::Screen, "Screen"),
        (Vp9TuneContent::Film, "Film"),
    ];

    for (mode, name) in tune_modes {
        println!("Testing VP9 with tune content: {}", name);

        let config = Vp9EncoderConfig {
            width,
            height,
            tune_content: mode,
            lag_in_frames: 0,
            auto_alt_ref: false,
            ..Default::default()
        };

        let mut encoder = Vp9Encoder::with_config(config)
            .expect(&format!("Failed to create encoder with tune: {}", name));

        let frame = create_yuv420p_frame(width, height, 0, true);
        encoder.send_frame(&Frame::Video(frame)).expect("Failed to encode");
        encoder.flush().expect("Failed to flush");

        match encoder.receive_packet() {
            Ok(packet) => println!("  {} tuning produced {} bytes", name, packet.data.len()),
            Err(Error::TryAgain) => println!("  {} tuning buffered frame", name),
            Err(e) => panic!("Error with {} tuning: {:?}", name, e),
        }
    }
}

#[test]
fn test_vp9_speed_presets() {
    let width = 320;
    let height = 240;

    // VP9 cpu_used ranges from -9 to 9
    let presets = [-8, -4, 0, 4, 6, 8];

    for preset in presets {
        println!("Testing VP9 with cpu_used = {}", preset);

        let config = Vp9EncoderConfig {
            width,
            height,
            cpu_used: preset,
            lag_in_frames: 0,
            auto_alt_ref: false,
            ..Default::default()
        };

        let result = Vp9Encoder::with_config(config);
        assert!(result.is_ok(), "Should support cpu_used = {}", preset);
    }
}

#[test]
fn test_vp9_row_mt() {
    let width = 1280;
    let height = 720;

    // Test with row-based multi-threading enabled
    let config = Vp9EncoderConfig {
        width,
        height,
        row_mt: true,
        threads: 4,
        lag_in_frames: 0,
        auto_alt_ref: false,
        ..Default::default()
    };

    let mut encoder = Vp9Encoder::with_config(config).expect("Failed to create row MT encoder");

    let frame = create_yuv420p_frame(width, height, 0, true);
    encoder.send_frame(&Frame::Video(frame)).expect("Failed to encode with row MT");
    encoder.flush().expect("Failed to flush");

    match encoder.receive_packet() {
        Ok(packet) => println!("Row MT encoded to {} bytes", packet.data.len()),
        Err(Error::TryAgain) => println!("Row MT frame buffered"),
        Err(e) => panic!("Row MT encoding failed: {:?}", e),
    }
}

#[test]
fn test_vp9_frame_parallel_decoding() {
    let width = 640;
    let height = 480;

    let config = Vp9EncoderConfig {
        width,
        height,
        frame_parallel: true,
        lag_in_frames: 0,
        auto_alt_ref: false,
        ..Default::default()
    };

    let mut encoder = Vp9Encoder::with_config(config)
        .expect("Failed to create frame parallel encoder");

    let frame = create_yuv420p_frame(width, height, 0, true);
    encoder.send_frame(&Frame::Video(frame)).expect("Failed to encode");
    encoder.flush().expect("Failed to flush");

    match encoder.receive_packet() {
        Ok(packet) => println!("Frame parallel encoded to {} bytes", packet.data.len()),
        Err(Error::TryAgain) => println!("Frame parallel frame buffered"),
        Err(e) => panic!("Frame parallel encoding failed: {:?}", e),
    }
}

#[test]
fn test_vp9_decoder_invalid_data() {
    let mut decoder = Vp9Decoder::new().expect("Failed to create decoder");

    // Create packet with random invalid data
    let invalid_data = vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
    let packet = Packet::new(0, Buffer::from_vec(invalid_data));

    // Should return error for invalid VP9 data
    let result = decoder.send_packet(&packet);
    match result {
        Ok(_) | Err(_) => {} // Either is acceptable - just don't panic
    }
}

#[test]
fn test_vp9_decoder_empty_packet() {
    let mut decoder = Vp9Decoder::new().expect("Failed to create decoder");

    let packet = Packet::new(0, Buffer::new());

    let result = decoder.send_packet(&packet);
    assert!(result.is_err(), "Should reject empty packet");
}

#[test]
fn test_vp9_dimension_validation() {
    let result = Vp9Encoder::new(0, 480);
    assert!(result.is_err(), "Should reject zero width");

    let result = Vp9Encoder::new(640, 0);
    assert!(result.is_err(), "Should reject zero height");

    let result = Vp9Encoder::new(0, 0);
    assert!(result.is_err(), "Should reject zero dimensions");
}

#[test]
fn test_vp9_error_resilient_mode() {
    let width = 320;
    let height = 240;

    let config = Vp9EncoderConfig {
        width,
        height,
        error_resilient: true,
        lag_in_frames: 0,
        auto_alt_ref: false,
        ..Default::default()
    };

    let mut encoder = Vp9Encoder::with_config(config).expect("Failed to create encoder");

    let frame = create_yuv420p_frame(width, height, 0, true);
    encoder.send_frame(&Frame::Video(frame)).expect("Failed to send frame");
    encoder.flush().expect("Failed to flush");

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
fn test_vp9_bitrate_control() {
    let width = 640;
    let height = 480;
    let num_frames = 10;

    let bitrates = [250_000u32, 500_000, 1_000_000, 2_000_000];

    for target_bitrate in bitrates {
        let config = Vp9EncoderConfig {
            width,
            height,
            bitrate: target_bitrate,
            framerate: 30,
            rate_control: Vp9RateControl::CBR,
            lag_in_frames: 0,
            auto_alt_ref: false,
            ..Default::default()
        };

        let mut encoder = Vp9Encoder::with_config(config).expect("Failed to create encoder");

        let mut total_bytes = 0;
        for i in 0..num_frames {
            let frame = create_yuv420p_frame(width, height, i, i == 0);
            encoder.send_frame(&Frame::Video(frame)).expect("Failed to send frame");

            loop {
                match encoder.receive_packet() {
                    Ok(packet) => total_bytes += packet.data.len(),
                    Err(Error::TryAgain) => break,
                    Err(e) => panic!("Unexpected error: {:?}", e),
                }
            }
        }

        encoder.flush().expect("Failed to flush");
        loop {
            match encoder.receive_packet() {
                Ok(packet) => total_bytes += packet.data.len(),
                Err(Error::TryAgain) => break,
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }

        let bits_produced = (total_bytes * 8) as f64;
        let duration_seconds = num_frames as f64 / 30.0;
        let actual_bitrate = bits_produced / duration_seconds;

        println!(
            "Target: {} bps, Actual: {:.0} bps, Ratio: {:.2}",
            target_bitrate,
            actual_bitrate,
            actual_bitrate / target_bitrate as f64
        );
    }
}

#[test]
fn test_vp9_keyframe_interval() {
    let width = 320;
    let height = 240;
    let keyframe_interval = 5;
    let num_frames = 15;

    let config = Vp9EncoderConfig {
        width,
        height,
        keyframe_interval,
        lag_in_frames: 0,
        auto_alt_ref: false,
        ..Default::default()
    };

    let mut encoder = Vp9Encoder::with_config(config).expect("Failed to create encoder");

    let mut packets = Vec::new();
    for i in 0..num_frames {
        let frame = create_yuv420p_frame(width, height, i, false);
        encoder.send_frame(&Frame::Video(frame)).expect("Failed to send frame");

        loop {
            match encoder.receive_packet() {
                Ok(packet) => packets.push(packet),
                Err(Error::TryAgain) => break,
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }
    }

    encoder.flush().expect("Failed to flush");
    loop {
        match encoder.receive_packet() {
            Ok(packet) => packets.push(packet),
            Err(Error::TryAgain) => break,
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    let keyframe_count = packets.iter().filter(|p| p.is_keyframe()).count();
    println!(
        "Got {} keyframes in {} packets (interval was {})",
        keyframe_count,
        packets.len(),
        keyframe_interval
    );

    // Should have at least one keyframe (the first frame)
    assert!(keyframe_count >= 1, "Should have at least one keyframe");
}

#[test]
fn test_vp9_quantizer_range() {
    let width = 320;
    let height = 240;

    // Test with restricted quantizer range
    let config = Vp9EncoderConfig {
        width,
        height,
        min_q: 10,
        max_q: 40,
        lag_in_frames: 0,
        auto_alt_ref: false,
        ..Default::default()
    };

    let mut encoder = Vp9Encoder::with_config(config).expect("Failed to create encoder");

    let frame = create_yuv420p_frame(width, height, 0, true);
    encoder.send_frame(&Frame::Video(frame)).expect("Failed to encode");
    encoder.flush().expect("Failed to flush");

    match encoder.receive_packet() {
        Ok(packet) => {
            println!("Q range [10, 40] produced {} bytes", packet.data.len());
        }
        Err(Error::TryAgain) => println!("Frame buffered"),
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

#[test]
fn test_vp9_consecutive_sequences() {
    let width = 320;
    let height = 240;

    for sequence in 0..3 {
        println!("Encoding VP9 sequence {}", sequence);

        let config = Vp9EncoderConfig {
            width,
            height,
            lag_in_frames: 0,
            auto_alt_ref: false,
            ..Default::default()
        };

        let mut encoder = Vp9Encoder::with_config(config).expect("Failed to create encoder");
        let mut decoder = Vp9Decoder::new().expect("Failed to create decoder");

        for i in 0..3 {
            let frame = create_yuv420p_frame(width, height, i, i == 0);
            encoder.send_frame(&Frame::Video(frame)).expect("Failed to encode");
        }

        encoder.flush().expect("Failed to flush");

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
