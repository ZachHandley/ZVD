//! Integration tests for MP4 demuxer
//!
//! These tests verify the MP4 demuxer correctly extracts streams,
//! extradata, and packets from real MP4 files.

#![cfg(feature = "mp4-support")]

use std::path::Path;
use zvd_lib::error::Error;
use zvd_lib::format::mp4::Mp4Demuxer;
use zvd_lib::format::Demuxer;
use zvd_lib::util::MediaType;

/// Path to test MP4 file (H.264 video + AAC audio)
const TEST_MP4_PATH: &str = "tests/files/HELLDIVERS2.mp4";

/// Test that MP4 demuxer correctly detects H.264 video and AAC audio streams
#[test]
fn test_mp4_demux_streams() {
    let path = Path::new(TEST_MP4_PATH);
    if !path.exists() {
        eprintln!("Test file not found: {}", TEST_MP4_PATH);
        return;
    }

    let mut demuxer = Mp4Demuxer::new();
    demuxer.open(path).expect("Failed to open MP4 file");

    let streams = demuxer.streams();
    println!("Found {} streams", streams.len());

    // Should have at least 2 streams (video + audio)
    assert!(
        streams.len() >= 2,
        "Expected at least 2 streams, found {}",
        streams.len()
    );

    // Find video stream
    let video_stream = streams
        .iter()
        .find(|s| s.info.media_type == MediaType::Video);
    assert!(video_stream.is_some(), "Expected to find a video stream");

    let video = video_stream.unwrap();
    println!(
        "Video stream: index={}, codec={}",
        video.info.index, video.info.codec_id
    );

    // Verify H.264 codec
    assert_eq!(
        video.info.codec_id, "h264",
        "Expected H.264 video codec, found {}",
        video.info.codec_id
    );

    // Verify video info is populated
    let video_info = video.info.video_info.as_ref().expect("Missing video info");
    println!(
        "Video: {}x{} @ {:?} fps",
        video_info.width, video_info.height, video_info.frame_rate
    );
    assert!(video_info.width > 0, "Video width should be > 0");
    assert!(video_info.height > 0, "Video height should be > 0");

    // Find audio stream
    let audio_stream = streams
        .iter()
        .find(|s| s.info.media_type == MediaType::Audio);
    assert!(audio_stream.is_some(), "Expected to find an audio stream");

    let audio = audio_stream.unwrap();
    println!(
        "Audio stream: index={}, codec={}",
        audio.info.index, audio.info.codec_id
    );

    // Verify AAC codec
    assert_eq!(
        audio.info.codec_id, "aac",
        "Expected AAC audio codec, found {}",
        audio.info.codec_id
    );

    // Verify audio info is populated
    let audio_info = audio.info.audio_info.as_ref().expect("Missing audio info");
    println!(
        "Audio: {} Hz, {} channels",
        audio_info.sample_rate, audio_info.channels
    );
    assert!(audio_info.sample_rate > 0, "Sample rate should be > 0");
    assert!(audio_info.channels > 0, "Channel count should be > 0");

    demuxer.close().expect("Failed to close demuxer");
}

/// Test that extradata is extracted for both H.264 and AAC streams
#[test]
fn test_mp4_extradata_extraction() {
    let path = Path::new(TEST_MP4_PATH);
    if !path.exists() {
        eprintln!("Test file not found: {}", TEST_MP4_PATH);
        return;
    }

    let mut demuxer = Mp4Demuxer::new();
    demuxer.open(path).expect("Failed to open MP4 file");

    let streams = demuxer.streams();

    // Find and verify video stream extradata
    let video_stream = streams
        .iter()
        .find(|s| s.info.media_type == MediaType::Video)
        .expect("Missing video stream");

    let video_extradata = video_stream
        .extradata
        .as_ref()
        .expect("Video stream missing extradata (avcC)");

    println!(
        "H.264 extradata (avcC) size: {} bytes",
        video_extradata.len()
    );

    // avcC should be at least 7 bytes (header) + SPS/PPS data
    assert!(
        video_extradata.len() >= 7,
        "avcC extradata too small: {} bytes",
        video_extradata.len()
    );

    // Find and verify audio stream extradata
    let audio_stream = streams
        .iter()
        .find(|s| s.info.media_type == MediaType::Audio)
        .expect("Missing audio stream");

    let audio_extradata = audio_stream
        .extradata
        .as_ref()
        .expect("Audio stream missing extradata (AudioSpecificConfig)");

    println!(
        "AAC extradata (AudioSpecificConfig) size: {} bytes",
        audio_extradata.len()
    );

    // AudioSpecificConfig is typically 2 bytes for basic AAC
    assert!(
        audio_extradata.len() >= 2,
        "AAC extradata too small: {} bytes",
        audio_extradata.len()
    );

    demuxer.close().expect("Failed to close demuxer");
}

/// Test that H.264 extradata (avcC) contains valid SPS and PPS NAL units
#[test]
fn test_mp4_h264_extradata_valid() {
    let path = Path::new(TEST_MP4_PATH);
    if !path.exists() {
        eprintln!("Test file not found: {}", TEST_MP4_PATH);
        return;
    }

    let mut demuxer = Mp4Demuxer::new();
    demuxer.open(path).expect("Failed to open MP4 file");

    let streams = demuxer.streams();

    let video_stream = streams
        .iter()
        .find(|s| s.info.media_type == MediaType::Video)
        .expect("Missing video stream");

    let extradata = video_stream
        .extradata
        .as_ref()
        .expect("Video stream missing extradata");

    // Parse avcC structure
    // Byte 0: configurationVersion (should be 1)
    // Byte 1: AVCProfileIndication
    // Byte 2: profile_compatibility
    // Byte 3: AVCLevelIndication
    // Byte 4: (6 bits reserved) + (2 bits lengthSizeMinusOne)
    // Byte 5: (3 bits reserved) + (5 bits numOfSequenceParameterSets)
    // Then SPS NALs, then numOfPictureParameterSets, then PPS NALs

    println!("Parsing avcC extradata ({} bytes):", extradata.len());
    println!(
        "  Raw bytes: {:02x?}",
        &extradata[..extradata.len().min(20)]
    );

    // Check configuration version
    let config_version = extradata[0];
    println!("  configurationVersion: {}", config_version);
    assert_eq!(config_version, 1, "avcC configurationVersion should be 1");

    // Extract profile and level
    let profile = extradata[1];
    let level = extradata[3];
    println!("  AVCProfileIndication: {} (0x{:02x})", profile, profile);
    println!("  AVCLevelIndication: {} (0x{:02x})", level, level);

    // Decode common profile names
    let profile_name = match profile {
        66 => "Baseline",
        77 => "Main",
        88 => "Extended",
        100 => "High",
        110 => "High 10",
        122 => "High 4:2:2",
        244 => "High 4:4:4 Predictive",
        _ => "Other",
    };
    println!("  Profile: {} ({})", profile_name, profile);

    // Length size minus one (bits 0-1 of byte 4)
    let length_size_minus_one = extradata[4] & 0x03;
    println!(
        "  NAL unit length size: {} bytes",
        length_size_minus_one + 1
    );

    // Number of SPS (bits 0-4 of byte 5)
    let num_sps = extradata[5] & 0x1F;
    println!("  Number of SPS: {}", num_sps);
    assert!(num_sps >= 1, "Should have at least 1 SPS NAL unit");

    // Parse SPS entries
    let mut offset = 6;
    for i in 0..num_sps {
        if offset + 2 > extradata.len() {
            panic!("Unexpected end of avcC data while reading SPS length");
        }
        let sps_len = u16::from_be_bytes([extradata[offset], extradata[offset + 1]]) as usize;
        offset += 2;

        if offset + sps_len > extradata.len() {
            panic!("Unexpected end of avcC data while reading SPS data");
        }

        let sps_data = &extradata[offset..offset + sps_len];
        offset += sps_len;

        // SPS NAL unit type should be 7 (bits 0-4 of first byte)
        let nal_type = sps_data[0] & 0x1F;
        println!("  SPS[{}]: {} bytes, NAL type: {}", i, sps_len, nal_type);
        assert_eq!(nal_type, 7, "SPS NAL type should be 7");
    }

    // Number of PPS
    if offset >= extradata.len() {
        panic!("Unexpected end of avcC data while reading PPS count");
    }
    let num_pps = extradata[offset];
    offset += 1;
    println!("  Number of PPS: {}", num_pps);
    assert!(num_pps >= 1, "Should have at least 1 PPS NAL unit");

    // Parse PPS entries
    for i in 0..num_pps {
        if offset + 2 > extradata.len() {
            panic!("Unexpected end of avcC data while reading PPS length");
        }
        let pps_len = u16::from_be_bytes([extradata[offset], extradata[offset + 1]]) as usize;
        offset += 2;

        if offset + pps_len > extradata.len() {
            panic!("Unexpected end of avcC data while reading PPS data");
        }

        let pps_data = &extradata[offset..offset + pps_len];
        offset += pps_len;

        // PPS NAL unit type should be 8 (bits 0-4 of first byte)
        let nal_type = pps_data[0] & 0x1F;
        println!("  PPS[{}]: {} bytes, NAL type: {}", i, pps_len, nal_type);
        assert_eq!(nal_type, 8, "PPS NAL type should be 8");
    }

    println!("avcC parsing successful - valid H.264 decoder configuration");

    demuxer.close().expect("Failed to close demuxer");
}

/// Test reading packets from the MP4 file
#[test]
fn test_mp4_read_packets() {
    let path = Path::new(TEST_MP4_PATH);
    if !path.exists() {
        eprintln!("Test file not found: {}", TEST_MP4_PATH);
        return;
    }

    let mut demuxer = Mp4Demuxer::new();
    demuxer.open(path).expect("Failed to open MP4 file");

    let mut video_packet_count = 0;
    let mut audio_packet_count = 0;
    let mut keyframe_count = 0;
    let max_packets = 100;

    // Read first N packets
    for i in 0..max_packets {
        match demuxer.read_packet() {
            Ok(packet) => {
                // Verify packet has data
                assert!(!packet.data.is_empty(), "Packet {} has empty data", i);

                // Check which stream this packet belongs to
                let stream = demuxer
                    .streams()
                    .iter()
                    .find(|s| s.info.index == packet.stream_index);

                if let Some(s) = stream {
                    match s.info.media_type {
                        MediaType::Video => {
                            video_packet_count += 1;
                            if packet.flags.keyframe {
                                keyframe_count += 1;
                            }
                        }
                        MediaType::Audio => {
                            audio_packet_count += 1;
                        }
                        _ => {}
                    }
                }

                // Print details for first few packets
                if i < 10 {
                    println!(
                        "Packet {}: stream={}, size={}, pts={:?}, keyframe={}",
                        i,
                        packet.stream_index,
                        packet.data.len(),
                        packet.pts,
                        packet.flags.keyframe
                    );
                }
            }
            Err(Error::EndOfStream) => {
                println!("End of stream after {} packets", i);
                break;
            }
            Err(e) => {
                panic!("Error reading packet {}: {}", i, e);
            }
        }
    }

    println!(
        "Read {} video packets, {} audio packets, {} keyframes",
        video_packet_count, audio_packet_count, keyframe_count
    );

    // Verify we got packets from both streams
    assert!(
        video_packet_count > 0,
        "Expected to read at least one video packet"
    );
    assert!(
        audio_packet_count > 0,
        "Expected to read at least one audio packet"
    );

    // Should have at least one keyframe (first video frame should be one)
    assert!(keyframe_count > 0, "Expected to find at least one keyframe");

    demuxer.close().expect("Failed to close demuxer");
}

/// Test seeking in the MP4 file
#[test]
fn test_mp4_seek() {
    let path = Path::new(TEST_MP4_PATH);
    if !path.exists() {
        eprintln!("Test file not found: {}", TEST_MP4_PATH);
        return;
    }

    let mut demuxer = Mp4Demuxer::new();
    demuxer.open(path).expect("Failed to open MP4 file");

    // Find the video stream index
    let video_stream = demuxer
        .streams()
        .iter()
        .find(|s| s.info.media_type == MediaType::Video)
        .expect("No video stream found");
    let video_stream_index = video_stream.info.index;
    let time_base = video_stream.info.time_base;
    println!(
        "Video stream index: {}, time_base: {:?}",
        video_stream_index, time_base
    );

    // Read first packet to get initial state
    let first_packet = demuxer.read_packet().expect("Failed to read first packet");
    let first_pts = first_packet.pts;
    println!(
        "First packet: stream={}, pts={:?}, keyframe={}",
        first_packet.stream_index, first_pts, first_packet.flags.keyframe
    );

    // Seek to the beginning (timestamp 0)
    demuxer
        .seek(video_stream_index, 0)
        .expect("Failed to seek to beginning");
    println!("Seeked to beginning");

    // Read packet after seeking to verify we're at the start
    let after_seek_packet = demuxer
        .read_packet()
        .expect("Failed to read packet after seek to beginning");
    println!(
        "After seek to 0: stream={}, pts={:?}, keyframe={}",
        after_seek_packet.stream_index, after_seek_packet.pts, after_seek_packet.flags.keyframe
    );

    // Read a bunch of packets to get somewhere in the middle
    let mut last_video_pts: i64 = 0;
    for _ in 0..50 {
        match demuxer.read_packet() {
            Ok(packet) => {
                if packet.codec_type == MediaType::Video {
                    last_video_pts = packet.pts.value;
                }
            }
            Err(Error::EndOfStream) => break,
            Err(e) => panic!("Error reading packet: {}", e),
        }
    }
    println!("After reading 50 packets, last video PTS: {}", last_video_pts);

    // Seek to a timestamp we've seen before (roughly half of what we've read)
    let seek_target = last_video_pts / 2;
    println!("Seeking to timestamp: {}", seek_target);

    demuxer
        .seek(video_stream_index, seek_target)
        .expect("Failed to seek to middle");

    // Read packet after seeking - should be at or before the seek target (keyframe)
    let mut found_video = false;
    for _ in 0..10 {
        match demuxer.read_packet() {
            Ok(packet) => {
                if packet.codec_type == MediaType::Video {
                    println!(
                        "After seek to {}: stream={}, pts={:?}, keyframe={}",
                        seek_target, packet.stream_index, packet.pts, packet.flags.keyframe
                    );
                    // After seeking, the first video packet should be a keyframe
                    assert!(
                        packet.flags.keyframe,
                        "First video packet after seek should be a keyframe"
                    );
                    // The PTS should be at or before the seek target
                    assert!(
                        packet.pts.value <= seek_target,
                        "First video packet after seek should be at or before seek target"
                    );
                    found_video = true;
                    break;
                }
            }
            Err(Error::EndOfStream) => break,
            Err(e) => panic!("Error reading packet after seek: {}", e),
        }
    }
    assert!(found_video, "Should have found a video packet after seek");

    // Test seeking to a very large timestamp (past EOF) - should seek to the last keyframe
    let very_large_timestamp = i64::MAX / 2;
    println!("Seeking to very large timestamp: {}", very_large_timestamp);

    match demuxer.seek(video_stream_index, very_large_timestamp) {
        Ok(()) => {
            // This should either work and position near the end, or the binary search
            // should find the last sample
            println!("Seek to large timestamp succeeded");
        }
        Err(e) => {
            // This is acceptable - seeking past EOF might fail
            println!("Seek to large timestamp returned error (acceptable): {}", e);
        }
    }

    demuxer.close().expect("Failed to close demuxer");
    println!("MP4 seek test passed!");
}
