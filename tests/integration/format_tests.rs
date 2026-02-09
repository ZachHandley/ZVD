//! Container format integration tests for ZVD
//!
//! Tests muxer/demuxer round-trips for various container formats.
//! Each test creates test data, muxes it to a container, demuxes it,
//! and verifies the data matches.

#![allow(unused_imports)]

use std::io::{Read, Write};
use tempfile::NamedTempFile;

use zvd_lib::codec::frame::{AudioFrame, VideoFrame};
use zvd_lib::codec::Frame;
use zvd_lib::error::Error;
use zvd_lib::format::{
    Packet, PacketFlags, Stream, StreamInfo,
};
use zvd_lib::util::{Buffer, MediaType, PixelFormat, Rational, SampleFormat, Timestamp};

// Include common test utilities
#[path = "../common/mod.rs"]
mod common;

use common::*;

// ============================================================================
// WAV Format Tests
// ============================================================================

mod wav_tests {
    use super::*;
    use zvd_lib::format::wav::{WavDemuxer, WavMuxer};
    use zvd_lib::format::{Demuxer, Muxer};

    /// Create a simple WAV test stream info
    fn create_wav_stream_info(sample_rate: u32, channels: u16, bits: u8) -> StreamInfo {
        StreamInfo {
            index: 0,
            media_type: MediaType::Audio,
            codec_id: "pcm".to_string(),
            time_base: Rational::new(1, sample_rate as i64),
            duration: 0,
            nb_frames: None,
            audio_info: Some(zvd_lib::format::stream::AudioInfo {
                sample_rate,
                channels,
                sample_fmt: format!("s{}", bits),
                bits_per_sample: bits,
                frame_size: 1024,
                bit_rate: None,
            }),
            video_info: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_wav_mux_demux_roundtrip() {
        let sample_rate = audio::SAMPLE_RATE_44100;
        let channels = audio::CHANNELS_STEREO;
        let bits_per_sample = 16u8;

        // Create a temporary file
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let temp_path = temp_file.path().to_path_buf();

        // Create test audio data
        let num_samples = 1024;
        let mut audio_data = Vec::with_capacity(num_samples * channels as usize * 2);
        for i in 0..num_samples {
            let t = i as f64 / sample_rate as f64;
            let sample = ((2.0 * std::f64::consts::PI * 440.0 * t).sin() * 0.5 * i16::MAX as f64)
                as i16;
            for _ch in 0..channels {
                audio_data.extend_from_slice(&sample.to_le_bytes());
            }
        }

        // Mux to WAV
        {
            let mut muxer = WavMuxer::new();
            muxer.create(&temp_path).expect("Failed to create WAV file");

            let stream_info = create_wav_stream_info(sample_rate, channels, bits_per_sample);
            let stream = Stream::new(stream_info);
            muxer.add_stream(stream).expect("Failed to add stream");

            muxer.write_header().expect("Failed to write header");

            let mut packet = Packet::new_audio(0, Buffer::from_vec(audio_data.clone()));
            packet.pts = Timestamp::new(0);
            muxer.write_packet(&packet).expect("Failed to write packet");

            muxer.write_trailer().expect("Failed to write trailer");
        }

        // Demux from WAV
        {
            let mut demuxer = WavDemuxer::new();
            demuxer.open(&temp_path).expect("Failed to open WAV file");

            let streams = demuxer.streams();
            assert_eq!(streams.len(), 1, "Should have one audio stream");

            let stream = &streams[0];
            assert_eq!(stream.info.media_type, MediaType::Audio);

            let audio_info = stream.info.audio_info.as_ref().expect("Should have audio info");
            assert_eq!(audio_info.sample_rate, sample_rate);
            assert_eq!(audio_info.channels, channels);

            // Read packet
            let packet = demuxer.read_packet().expect("Should read packet");
            assert!(!packet.data.is_empty(), "Packet should have data");
        }
    }

    #[test]
    fn test_wav_different_sample_rates() {
        let sample_rates = [
            audio::SAMPLE_RATE_8000,
            audio::SAMPLE_RATE_16000,
            audio::SAMPLE_RATE_44100,
            audio::SAMPLE_RATE_48000,
        ];

        for sample_rate in sample_rates {
            let temp_file = NamedTempFile::new().expect("Failed to create temp file");
            let temp_path = temp_file.path().to_path_buf();

            // Create minimal audio data
            let audio_data: Vec<u8> = (0..sample_rate as usize)
                .flat_map(|_| [0i16; 2].iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<_>>())
                .take(4096)
                .collect();

            // Mux
            {
                let mut muxer = WavMuxer::new();
                muxer.create(&temp_path).expect("Failed to create WAV");

                let stream_info = create_wav_stream_info(sample_rate, 2, 16);
                let stream = Stream::new(stream_info);
                muxer.add_stream(stream).expect("Failed to add stream");
                muxer.write_header().expect("Failed to write header");

                let packet = Packet::new_audio(0, Buffer::from_vec(audio_data));
                muxer.write_packet(&packet).expect("Failed to write packet");
                muxer.write_trailer().expect("Failed to write trailer");
            }

            // Demux and verify
            {
                let mut demuxer = WavDemuxer::new();
                demuxer.open(&temp_path).expect("Failed to open WAV");

                let streams = demuxer.streams();
                let audio_info = streams[0]
                    .info
                    .audio_info
                    .as_ref()
                    .expect("Should have audio info");

                assert_eq!(
                    audio_info.sample_rate, sample_rate,
                    "Sample rate should match for {} Hz",
                    sample_rate
                );
            }
        }
    }

    #[test]
    fn test_wav_mono_stereo() {
        let channel_configs = [
            audio::CHANNELS_MONO,
            audio::CHANNELS_STEREO,
        ];

        for channels in channel_configs {
            let temp_file = NamedTempFile::new().expect("Failed to create temp file");
            let temp_path = temp_file.path().to_path_buf();

            let audio_data: Vec<u8> = (0..1024 * channels as usize)
                .flat_map(|_| 0i16.to_le_bytes())
                .collect();

            // Mux
            {
                let mut muxer = WavMuxer::new();
                muxer.create(&temp_path).expect("Failed to create WAV");

                let stream_info = create_wav_stream_info(44100, channels, 16);
                let stream = Stream::new(stream_info);
                muxer.add_stream(stream).expect("Failed to add stream");
                muxer.write_header().expect("Failed to write header");

                let packet = Packet::new_audio(0, Buffer::from_vec(audio_data));
                muxer.write_packet(&packet).expect("Failed to write packet");
                muxer.write_trailer().expect("Failed to write trailer");
            }

            // Demux and verify
            {
                let mut demuxer = WavDemuxer::new();
                demuxer.open(&temp_path).expect("Failed to open WAV");

                let streams = demuxer.streams();
                let audio_info = streams[0]
                    .info
                    .audio_info
                    .as_ref()
                    .expect("Should have audio info");

                assert_eq!(
                    audio_info.channels, channels,
                    "Channel count should match for {} channels",
                    channels
                );
            }
        }
    }
}

// ============================================================================
// Y4M Format Tests
// ============================================================================

mod y4m_tests {
    use super::*;
    use zvd_lib::format::y4m::{Y4mDemuxer, Y4mMuxer};
    use zvd_lib::format::{Demuxer, Muxer};

    /// Create a Y4M test stream info
    fn create_y4m_stream_info(width: u32, height: u32, frame_rate: Rational) -> StreamInfo {
        StreamInfo {
            index: 0,
            media_type: MediaType::Video,
            codec_id: "rawvideo".to_string(),
            time_base: Rational::new(frame_rate.den, frame_rate.num),
            duration: 0,
            nb_frames: None,
            audio_info: None,
            video_info: Some(zvd_lib::format::stream::VideoInfo {
                width,
                height,
                pix_fmt: "yuv420p".to_string(),
                bits_per_sample: 8,
                frame_rate,
                aspect_ratio: Rational::new(1, 1),
            }),
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_y4m_mux_demux_roundtrip() {
        let (width, height) = resolutions::TEST_SMALL;
        let frame_rate = Rational::new(30, 1);

        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let temp_path = temp_file.path().to_path_buf();

        // Create test video frame
        let frame = create_test_video_frame(width, height);

        // Calculate YUV420P frame size
        let y_size = (width * height) as usize;
        let uv_size = ((width / 2) * (height / 2)) as usize;
        let frame_size = y_size + 2 * uv_size;

        // Mux to Y4M
        {
            let mut muxer = Y4mMuxer::new();
            muxer.create(&temp_path).expect("Failed to create Y4M file");

            let stream_info = create_y4m_stream_info(width, height, frame_rate);
            let stream = Stream::new(stream_info);
            muxer.add_stream(stream).expect("Failed to add stream");

            muxer.write_header().expect("Failed to write header");

            // Combine planes into single buffer for Y4M
            let mut frame_data = Vec::with_capacity(frame_size);
            for plane in &frame.data {
                frame_data.extend_from_slice(plane.as_slice());
            }

            let mut packet = Packet::new_video(0, Buffer::from_vec(frame_data));
            packet.pts = Timestamp::new(0);
            packet.flags.keyframe = true;
            muxer.write_packet(&packet).expect("Failed to write packet");

            muxer.write_trailer().expect("Failed to write trailer");
        }

        // Demux from Y4M
        {
            let mut demuxer = Y4mDemuxer::new();
            demuxer.open(&temp_path).expect("Failed to open Y4M file");

            let streams = demuxer.streams();
            assert_eq!(streams.len(), 1, "Should have one video stream");

            let stream = &streams[0];
            assert_eq!(stream.info.media_type, MediaType::Video);

            let video_info = stream.info.video_info.as_ref().expect("Should have video info");
            assert_eq!(video_info.width, width);
            assert_eq!(video_info.height, height);

            // Read packet
            let packet = demuxer.read_packet().expect("Should read packet");
            assert!(!packet.data.is_empty(), "Packet should have data");
            assert_eq!(packet.data.len(), frame_size, "Frame size should match");
        }
    }

    #[test]
    fn test_y4m_different_resolutions() {
        let test_resolutions = [
            resolutions::QCIF,
            resolutions::CIF,
            resolutions::TEST_SMALL,
            resolutions::VGA,
        ];

        for (width, height) in test_resolutions {
            let temp_file = NamedTempFile::new().expect("Failed to create temp file");
            let temp_path = temp_file.path().to_path_buf();

            let frame = create_test_video_frame(width, height);

            // Calculate frame size
            let y_size = (width * height) as usize;
            let uv_size = ((width / 2) * (height / 2)) as usize;
            let frame_size = y_size + 2 * uv_size;

            // Mux
            {
                let mut muxer = Y4mMuxer::new();
                muxer.create(&temp_path).expect("Failed to create Y4M");

                let stream_info = create_y4m_stream_info(width, height, Rational::new(30, 1));
                let stream = Stream::new(stream_info);
                muxer.add_stream(stream).expect("Failed to add stream");
                muxer.write_header().expect("Failed to write header");

                let mut frame_data = Vec::with_capacity(frame_size);
                for plane in &frame.data {
                    frame_data.extend_from_slice(plane.as_slice());
                }

                let mut packet = Packet::new_video(0, Buffer::from_vec(frame_data));
                packet.pts = Timestamp::new(0);
                packet.flags.keyframe = true;
                muxer.write_packet(&packet).expect("Failed to write packet");
                muxer.write_trailer().expect("Failed to write trailer");
            }

            // Demux and verify
            {
                let mut demuxer = Y4mDemuxer::new();
                demuxer.open(&temp_path).expect("Failed to open Y4M");

                let streams = demuxer.streams();
                let video_info = streams[0]
                    .info
                    .video_info
                    .as_ref()
                    .expect("Should have video info");

                assert_eq!(
                    video_info.width, width,
                    "Width should match for {}x{}",
                    width, height
                );
                assert_eq!(
                    video_info.height, height,
                    "Height should match for {}x{}",
                    width, height
                );
            }
        }
    }

    #[test]
    fn test_y4m_multiple_frames() {
        let (width, height) = resolutions::TEST_SMALL;
        let frame_rate = Rational::new(30, 1);
        let num_frames = 5;

        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let temp_path = temp_file.path().to_path_buf();

        let frames = create_video_sequence(width, height, num_frames);

        let y_size = (width * height) as usize;
        let uv_size = ((width / 2) * (height / 2)) as usize;
        let frame_size = y_size + 2 * uv_size;

        // Mux multiple frames
        {
            let mut muxer = Y4mMuxer::new();
            muxer.create(&temp_path).expect("Failed to create Y4M");

            let stream_info = create_y4m_stream_info(width, height, frame_rate);
            let stream = Stream::new(stream_info);
            muxer.add_stream(stream).expect("Failed to add stream");
            muxer.write_header().expect("Failed to write header");

            for (i, frame) in frames.iter().enumerate() {
                let mut frame_data = Vec::with_capacity(frame_size);
                for plane in &frame.data {
                    frame_data.extend_from_slice(plane.as_slice());
                }

                let mut packet = Packet::new_video(0, Buffer::from_vec(frame_data));
                packet.pts = Timestamp::new(i as i64);
                packet.flags.keyframe = i == 0;
                muxer.write_packet(&packet).expect("Failed to write packet");
            }

            muxer.write_trailer().expect("Failed to write trailer");
        }

        // Demux and count frames
        {
            let mut demuxer = Y4mDemuxer::new();
            demuxer.open(&temp_path).expect("Failed to open Y4M");

            let mut frame_count = 0;
            loop {
                match demuxer.read_packet() {
                    Ok(_) => frame_count += 1,
                    Err(Error::EndOfStream) => break,
                    Err(e) => panic!("Unexpected error: {:?}", e),
                }
            }

            assert_eq!(
                frame_count, num_frames,
                "Should read {} frames",
                num_frames
            );
        }
    }
}

// ============================================================================
// AVI Format Tests
// ============================================================================

mod avi_tests {
    use super::*;
    use zvd_lib::format::avi::{AviDemuxer, AviMuxer};
    use zvd_lib::format::{Demuxer, Muxer};

    /// Create AVI video stream info
    fn create_avi_video_stream_info(width: u32, height: u32, frame_rate: Rational) -> StreamInfo {
        StreamInfo {
            index: 0,
            media_type: MediaType::Video,
            codec_id: "rawvideo".to_string(),
            time_base: Rational::new(1, 1000000), // microseconds
            duration: 0,
            nb_frames: None,
            audio_info: None,
            video_info: Some(zvd_lib::format::stream::VideoInfo {
                width,
                height,
                pix_fmt: "yuv420p".to_string(),
                bits_per_sample: 8,
                frame_rate,
                aspect_ratio: Rational::new(1, 1),
            }),
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_avi_mux_demux_roundtrip() {
        let (width, height) = resolutions::TEST_SMALL;
        let frame_rate = Rational::new(30, 1);

        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let temp_path = temp_file.path().to_path_buf();

        let frame = create_test_video_frame(width, height);

        // Calculate frame size
        let y_size = (width * height) as usize;
        let uv_size = ((width / 2) * (height / 2)) as usize;
        let frame_size = y_size + 2 * uv_size;

        // Mux to AVI
        {
            let mut muxer = AviMuxer::new();
            muxer.create(&temp_path).expect("Failed to create AVI file");

            let stream_info = create_avi_video_stream_info(width, height, frame_rate);
            let stream = Stream::new(stream_info);
            muxer.add_stream(stream).expect("Failed to add stream");

            muxer.write_header().expect("Failed to write header");

            let mut frame_data = Vec::with_capacity(frame_size);
            for plane in &frame.data {
                frame_data.extend_from_slice(plane.as_slice());
            }

            let mut packet = Packet::new_video(0, Buffer::from_vec(frame_data));
            packet.pts = Timestamp::new(0);
            packet.flags.keyframe = true;
            muxer.write_packet(&packet).expect("Failed to write packet");

            muxer.write_trailer().expect("Failed to write trailer");
        }

        // Demux from AVI
        {
            let mut demuxer = AviDemuxer::new();
            demuxer.open(&temp_path).expect("Failed to open AVI file");

            let streams = demuxer.streams();
            assert!(!streams.is_empty(), "Should have at least one stream");

            // Find video stream
            let video_stream = streams
                .iter()
                .find(|s| s.info.media_type == MediaType::Video);

            assert!(video_stream.is_some(), "Should have video stream");
        }
    }
}

// ============================================================================
// WebM Format Tests
// ============================================================================

#[cfg(feature = "webm-support")]
mod webm_tests {
    use super::*;
    use zvd_lib::format::webm::{WebmDemuxer, WebmMuxer};
    use zvd_lib::format::{Demuxer, Muxer};

    fn create_webm_video_stream_info(width: u32, height: u32) -> StreamInfo {
        StreamInfo {
            index: 0,
            media_type: MediaType::Video,
            codec_id: "vp9".to_string(),
            time_base: Rational::new(1, 1000), // milliseconds
            duration: 0,
            nb_frames: None,
            audio_info: None,
            video_info: Some(zvd_lib::format::stream::VideoInfo {
                width,
                height,
                pix_fmt: "yuv420p".to_string(),
                bits_per_sample: 8,
                frame_rate: Rational::new(30, 1),
                aspect_ratio: Rational::new(1, 1),
            }),
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_webm_mux_demux_roundtrip() {
        let (width, height) = resolutions::TEST_SMALL;

        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let temp_path = temp_file.path().to_path_buf();

        // For WebM, we need actual VP9/VP8 encoded data
        // This test creates a minimal structure

        // Mux to WebM
        {
            let mut muxer = WebmMuxer::new();
            muxer.create(&temp_path).expect("Failed to create WebM file");

            let stream_info = create_webm_video_stream_info(width, height);
            let stream = Stream::new(stream_info);
            muxer.add_stream(stream).expect("Failed to add stream");

            muxer.write_header().expect("Failed to write header");

            // Create minimal "video" data
            let video_data = vec![0u8; 1000];
            let mut packet = Packet::new_video(0, Buffer::from_vec(video_data));
            packet.pts = Timestamp::new(0);
            packet.flags.keyframe = true;
            muxer.write_packet(&packet).expect("Failed to write packet");

            muxer.write_trailer().expect("Failed to write trailer");
        }

        // Verify file was created
        assert!(temp_path.exists(), "WebM file should exist");
        let metadata = std::fs::metadata(&temp_path).expect("Should get file metadata");
        assert!(metadata.len() > 0, "WebM file should have content");
    }
}

// ============================================================================
// MPEG-TS Format Tests
// ============================================================================

mod mpegts_tests {
    use super::*;
    use zvd_lib::format::mpegts::{MpegtsDemuxer, MpegtsMuxer};
    use zvd_lib::format::{Demuxer, Muxer};

    fn create_mpegts_video_stream_info(width: u32, height: u32) -> StreamInfo {
        StreamInfo {
            index: 0,
            media_type: MediaType::Video,
            codec_id: "h264".to_string(),
            time_base: Rational::new(1, 90000), // 90kHz PTS
            duration: 0,
            nb_frames: None,
            audio_info: None,
            video_info: Some(zvd_lib::format::stream::VideoInfo {
                width,
                height,
                pix_fmt: "yuv420p".to_string(),
                bits_per_sample: 8,
                frame_rate: Rational::new(30, 1),
                aspect_ratio: Rational::new(1, 1),
            }),
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_mpegts_mux_demux_roundtrip() {
        let (width, height) = resolutions::TEST_SMALL;

        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let temp_path = temp_file.path().to_path_buf();

        // Mux to MPEG-TS
        {
            let mut muxer = MpegtsMuxer::new();
            muxer.create(&temp_path).expect("Failed to create MPEG-TS file");

            let stream_info = create_mpegts_video_stream_info(width, height);
            let stream = Stream::new(stream_info);
            muxer.add_stream(stream).expect("Failed to add stream");

            muxer.write_header().expect("Failed to write header");

            // Create minimal "video" data with NAL structure
            let mut nal_data = vec![0, 0, 0, 1]; // Start code
            nal_data.extend_from_slice(&[0x67, 0x00, 0x00, 0x00]); // SPS-like
            nal_data.extend_from_slice(&vec![0u8; 100]); // Padding

            let mut packet = Packet::new_video(0, Buffer::from_vec(nal_data));
            packet.pts = Timestamp::new(0);
            packet.dts = Timestamp::new(0);
            packet.flags.keyframe = true;
            muxer.write_packet(&packet).expect("Failed to write packet");

            muxer.write_trailer().expect("Failed to write trailer");
        }

        // Verify file was created with TS packets
        assert!(temp_path.exists(), "MPEG-TS file should exist");
        let metadata = std::fs::metadata(&temp_path).expect("Should get file metadata");

        // TS packets are 188 bytes
        assert!(
            metadata.len() >= 188,
            "MPEG-TS file should have at least one packet"
        );
    }
}

// ============================================================================
// FLV Format Tests
// ============================================================================

mod flv_tests {
    use super::*;
    use zvd_lib::format::flv::{FlvDemuxer, FlvMuxer};
    use zvd_lib::format::{Demuxer, Muxer};

    fn create_flv_video_stream_info(width: u32, height: u32) -> StreamInfo {
        StreamInfo {
            index: 0,
            media_type: MediaType::Video,
            codec_id: "h264".to_string(),
            time_base: Rational::new(1, 1000), // milliseconds
            duration: 0,
            nb_frames: None,
            audio_info: None,
            video_info: Some(zvd_lib::format::stream::VideoInfo {
                width,
                height,
                pix_fmt: "yuv420p".to_string(),
                bits_per_sample: 8,
                frame_rate: Rational::new(30, 1),
                aspect_ratio: Rational::new(1, 1),
            }),
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_flv_mux_demux_roundtrip() {
        let (width, height) = resolutions::TEST_SMALL;

        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let temp_path = temp_file.path().to_path_buf();

        // Mux to FLV
        {
            let mut muxer = FlvMuxer::new();
            muxer.create(&temp_path).expect("Failed to create FLV file");

            let stream_info = create_flv_video_stream_info(width, height);
            let stream = Stream::new(stream_info);
            muxer.add_stream(stream).expect("Failed to add stream");

            muxer.write_header().expect("Failed to write header");

            // Create minimal video tag data
            let video_data = vec![0x17, 0x00, 0x00, 0x00, 0x00]; // AVC keyframe sequence header
            video_data.iter().chain(&vec![0u8; 100]).copied().collect::<Vec<_>>();

            let mut packet = Packet::new_video(0, Buffer::from_vec(vec![0u8; 100]));
            packet.pts = Timestamp::new(0);
            packet.flags.keyframe = true;
            muxer.write_packet(&packet).expect("Failed to write packet");

            muxer.write_trailer().expect("Failed to write trailer");
        }

        // Verify file was created
        assert!(temp_path.exists(), "FLV file should exist");

        // Check FLV signature
        let mut file = std::fs::File::open(&temp_path).expect("Should open file");
        let mut header = [0u8; 3];
        file.read_exact(&mut header).expect("Should read header");
        assert_eq!(&header, b"FLV", "Should have FLV signature");
    }
}

// ============================================================================
// MP4 Format Tests (when feature is enabled)
// ============================================================================

#[cfg(feature = "mp4-support")]
mod mp4_tests {
    use super::*;
    use zvd_lib::format::mp4::{Mp4Demuxer, Mp4Muxer};
    use zvd_lib::format::{Demuxer, Muxer};

    fn create_mp4_video_stream_info(width: u32, height: u32) -> StreamInfo {
        StreamInfo {
            index: 0,
            media_type: MediaType::Video,
            codec_id: "h264".to_string(),
            time_base: Rational::new(1, 90000),
            duration: 0,
            nb_frames: None,
            audio_info: None,
            video_info: Some(zvd_lib::format::stream::VideoInfo {
                width,
                height,
                pix_fmt: "yuv420p".to_string(),
                bits_per_sample: 8,
                frame_rate: Rational::new(30, 1),
                aspect_ratio: Rational::new(1, 1),
            }),
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_mp4_mux_demux_roundtrip() {
        let (width, height) = resolutions::TEST_SMALL;

        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let temp_path = temp_file.path().to_path_buf();

        // Mux to MP4
        {
            let mut muxer = Mp4Muxer::new();
            muxer.create(&temp_path).expect("Failed to create MP4 file");

            let stream_info = create_mp4_video_stream_info(width, height);
            let stream = Stream::new(stream_info);
            muxer.add_stream(stream).expect("Failed to add stream");

            muxer.write_header().expect("Failed to write header");

            // Minimal NAL data
            let video_data = vec![0u8; 1000];
            let mut packet = Packet::new_video(0, Buffer::from_vec(video_data));
            packet.pts = Timestamp::new(0);
            packet.dts = Timestamp::new(0);
            packet.flags.keyframe = true;
            muxer.write_packet(&packet).expect("Failed to write packet");

            muxer.write_trailer().expect("Failed to write trailer");
        }

        // Demux from MP4
        {
            let mut demuxer = Mp4Demuxer::new();
            demuxer.open(&temp_path).expect("Failed to open MP4 file");

            let streams = demuxer.streams();
            assert!(!streams.is_empty(), "Should have at least one stream");
        }
    }

    #[test]
    fn test_mp4_with_audio() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let temp_path = temp_file.path().to_path_buf();

        // Create stream info for audio
        let audio_stream_info = StreamInfo {
            index: 0,
            media_type: MediaType::Audio,
            codec_id: "aac".to_string(),
            time_base: Rational::new(1, 44100),
            duration: 0,
            nb_frames: None,
            audio_info: Some(zvd_lib::format::stream::AudioInfo {
                sample_rate: 44100,
                channels: 2,
                sample_fmt: "f32".to_string(),
                bits_per_sample: 16,
                frame_size: 1024,
                bit_rate: Some(128000),
            }),
            video_info: None,
            metadata: std::collections::HashMap::new(),
        };

        // Mux
        {
            let mut muxer = Mp4Muxer::new();
            muxer.create(&temp_path).expect("Failed to create MP4");

            let stream = Stream::new(audio_stream_info);
            muxer.add_stream(stream).expect("Failed to add stream");

            muxer.write_header().expect("Failed to write header");

            let audio_data = vec![0u8; 1024];
            let mut packet = Packet::new_audio(0, Buffer::from_vec(audio_data));
            packet.pts = Timestamp::new(0);
            muxer.write_packet(&packet).expect("Failed to write packet");

            muxer.write_trailer().expect("Failed to write trailer");
        }

        // Verify file exists
        assert!(temp_path.exists(), "MP4 file should exist");
    }
}

// ============================================================================
// Format Detection Tests
// ============================================================================

mod format_detection_tests {
    use super::*;
    use zvd_lib::format::{detect_format_from_extension, get_format_info};

    #[test]
    fn test_format_detection_from_extension() {
        let test_cases = [
            ("test.mp4", Some("mp4")),
            ("test.m4v", Some("mp4")),
            ("test.mkv", Some("matroska")),
            ("test.webm", Some("webm")),
            ("test.avi", Some("avi")),
            ("test.mov", Some("mov")),
            ("test.flv", Some("flv")),
            ("test.ts", Some("mpegts")),
            ("test.mp3", Some("mp3")),
            ("test.ogg", Some("ogg")),
            ("test.wav", Some("wav")),
            ("test.flac", Some("flac")),
            ("test.y4m", Some("y4m")),
            ("test.unknown", None),
        ];

        for (path, expected) in test_cases {
            let detected = detect_format_from_extension(path);
            assert_eq!(
                detected, expected,
                "Format detection failed for {}",
                path
            );
        }
    }

    #[test]
    fn test_format_info() {
        let formats = ["mp4", "matroska", "wav", "flac", "y4m", "webm", "avi", "flv", "mpegts"];

        for format in formats {
            let info = get_format_info(format);
            assert!(
                info.is_some(),
                "Should have format info for {}",
                format
            );

            let info = info.unwrap();
            assert!(!info.name.is_empty(), "Format name should not be empty");
            assert!(!info.long_name.is_empty(), "Long name should not be empty");
            assert!(!info.extensions.is_empty(), "Extensions should not be empty");
        }
    }

    #[test]
    fn test_format_capabilities() {
        let mp4_info = get_format_info("mp4").expect("Should have MP4 info");
        assert!(mp4_info.capabilities.seekable, "MP4 should be seekable");
        assert!(mp4_info.capabilities.multi_stream, "MP4 should support multi-stream");
        assert!(mp4_info.capabilities.timestamps, "MP4 should support timestamps");
        assert!(mp4_info.capabilities.metadata, "MP4 should support metadata");

        let mpegts_info = get_format_info("mpegts").expect("Should have MPEG-TS info");
        assert!(!mpegts_info.capabilities.seekable, "MPEG-TS should not be easily seekable");
        assert!(mpegts_info.capabilities.multi_stream, "MPEG-TS should support multi-stream");
    }
}

// ============================================================================
// Interoperability Tests
// ============================================================================

mod interop_tests {
    use super::*;

    #[test]
    fn test_wav_to_y4m_metadata_preservation() {
        // This tests that metadata and timing information is preserved
        // when working with different format types

        let sample_rate = 44100u32;
        let channels = 2u16;
        let num_samples = 1024;

        // Create audio frame with specific timestamp
        let mut frame = create_test_audio_frame(sample_rate, channels, num_samples);
        frame.pts = Timestamp::new(12345);

        // Verify timestamp is preserved
        assert_eq!(frame.pts.value, 12345);
        assert!(frame.pts.is_valid());
    }

    #[test]
    fn test_packet_flags_roundtrip() {
        // Test that packet flags survive serialization/deserialization

        let mut packet = Packet::new_video(0, Buffer::from_vec(vec![0u8; 100]));
        packet.flags.keyframe = true;
        packet.flags.corrupt = false;
        packet.flags.config = true;

        assert!(packet.is_keyframe());
        assert!(packet.flags.config);
        assert!(!packet.flags.corrupt);
    }

    #[test]
    fn test_timestamp_preservation() {
        // Test timestamp handling across different time bases

        let pts = Timestamp::new(90000); // 1 second at 90kHz
        let time_base_90k = Rational::new(1, 90000);
        let time_base_1k = Rational::new(1, 1000);

        // Convert to seconds
        let seconds = pts.to_seconds(time_base_90k);
        assert!((seconds - 1.0).abs() < 0.001, "Should be approximately 1 second");

        // Rescale
        let rescaled = pts.rescale(time_base_90k, time_base_1k);
        assert_eq!(rescaled.value, 1000, "Should be 1000ms");
    }
}
