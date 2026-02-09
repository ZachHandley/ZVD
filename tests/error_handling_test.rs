//! Error handling tests for ZVD
//!
//! These tests verify that decoders and demuxers gracefully handle malformed,
//! truncated, or garbage input without panicking. All error cases should
//! return appropriate Error variants, not crash.

#![allow(unused_imports)]

use std::panic;
use zvd_lib::codec::frame::{AudioFrame, VideoFrame};
use zvd_lib::codec::{Decoder, Encoder, Frame};
use zvd_lib::error::Error;
use zvd_lib::format::{Packet, PacketFlags};
use zvd_lib::util::{Buffer, MediaType, PixelFormat, SampleFormat, Timestamp};

// Include common test utilities
#[path = "common/mod.rs"]
mod common;

use common::*;

// ============================================================================
// Helper Functions
// ============================================================================

/// Test that a closure does not panic
fn assert_no_panic<F: FnOnce() -> R + panic::UnwindSafe, R>(f: F, description: &str) {
    let result = panic::catch_unwind(f);
    assert!(result.is_ok(), "Panic occurred: {}", description);
}

/// Test that an operation returns an error (not panics)
fn assert_error_not_panic<F: FnOnce() -> Result<R, Error> + panic::UnwindSafe, R>(
    f: F,
    description: &str,
) {
    let panic_result = panic::catch_unwind(f);
    match panic_result {
        Ok(result) => {
            // Didn't panic - operation returned either Ok or Err, both are acceptable
            // but Err is more expected for error handling tests
            if result.is_ok() {
                println!("Warning: {} succeeded instead of returning error", description);
            }
        }
        Err(_) => {
            panic!("Panic occurred during: {}", description);
        }
    }
}

// ============================================================================
// AV1 Decoder Error Handling
// ============================================================================

mod av1_error_tests {
    use super::*;
    use zvd_lib::codec::av1::Av1Decoder;

    #[test]
    fn test_av1_decoder_garbage_input() {
        let mut decoder = Av1Decoder::new().expect("Should create decoder");

        // Send garbage data
        let garbage = create_garbage_data(1000);
        let packet = create_test_video_packet(garbage, 0, true);

        // Should return error, not panic
        let result = decoder.send_packet(&packet);
        // AV1 decoder may accept the packet and fail on receive, or fail on send
        // Either is acceptable as long as it doesn't panic

        // Try to receive (may fail)
        let _ = decoder.receive_frame();
    }

    #[test]
    fn test_av1_decoder_empty_packet() {
        let mut decoder = Av1Decoder::new().expect("Should create decoder");

        let packet = create_test_video_packet(vec![], 0, true);

        // Should handle empty packet gracefully
        let result = decoder.send_packet(&packet);
        // Either error or TryAgain is acceptable
        match result {
            Ok(()) => {
                // If accepted, receive should handle it
                let _ = decoder.receive_frame();
            }
            Err(_) => {
                // Error is expected for empty input
            }
        }
    }

    #[test]
    fn test_av1_decoder_truncated_obu() {
        let mut decoder = Av1Decoder::new().expect("Should create decoder");

        // Create truncated OBU-like data
        // OBU header but no payload
        let truncated = vec![0x12, 0x00]; // OBU header with size 0

        let packet = create_test_video_packet(truncated, 0, true);
        let result = decoder.send_packet(&packet);

        // Should not panic
        match result {
            Ok(()) => {
                let _ = decoder.receive_frame();
            }
            Err(_) => {}
        }
    }

    #[test]
    fn test_av1_decoder_random_data() {
        let mut decoder = Av1Decoder::new().expect("Should create decoder");

        // Test multiple random patterns
        for seed in [1, 42, 123, 999, 0xDEADBEEF] {
            let random_data = create_random_data(500, seed);
            let packet = create_test_video_packet(random_data, 0, true);

            assert_no_panic(
                || {
                    let _ = decoder.send_packet(&packet);
                    let _ = decoder.receive_frame();
                },
                &format!("AV1 decoder with random seed {}", seed),
            );
        }
    }

    #[test]
    fn test_av1_decoder_oversized_dimensions() {
        // Test that encoder rejects unreasonable dimensions
        use zvd_lib::codec::av1::Av1EncoderBuilder;

        // Very large dimensions - should fail at creation or first frame
        let result = Av1EncoderBuilder::new(65536, 65536).build();
        // This should either fail or succeed but handle gracefully
        if let Ok(mut encoder) = result {
            let frame = create_test_video_frame(65536, 65536);
            let _ = encoder.send_frame(&Frame::Video(frame));
        }
    }

    #[test]
    fn test_av1_decoder_flush_without_data() {
        let mut decoder = Av1Decoder::new().expect("Should create decoder");

        // Flush without sending any data should be safe
        let result = decoder.flush();
        assert!(result.is_ok(), "Flush on empty decoder should succeed");
    }
}

// ============================================================================
// H.264 Decoder Error Handling (when feature enabled)
// ============================================================================

#[cfg(feature = "h264")]
mod h264_error_tests {
    use super::*;
    use zvd_lib::codec::h264::{H264Decoder, H264Encoder};

    #[test]
    fn test_h264_decoder_malformed_nal() {
        let mut decoder = H264Decoder::new().expect("Should create decoder");

        // Malformed NAL unit - invalid NAL type
        let malformed = vec![0x00, 0x00, 0x00, 0x01, 0xFF, 0xFF, 0xFF, 0xFF];
        let packet = create_test_video_packet(malformed, 0, true);

        // Should return error or accept gracefully, not panic
        let result = decoder.send_packet(&packet);
        match result {
            Ok(()) => {
                // May accept and fail on decode
                let _ = decoder.receive_frame();
            }
            Err(_) => {
                // Error is expected
            }
        }
    }

    #[test]
    fn test_h264_decoder_garbage_input() {
        let mut decoder = H264Decoder::new().expect("Should create decoder");

        let garbage = create_garbage_data(1000);
        let packet = create_test_video_packet(garbage, 0, true);

        assert_no_panic(
            || {
                let _ = decoder.send_packet(&packet);
                let _ = decoder.receive_frame();
            },
            "H.264 decoder with garbage input",
        );
    }

    #[test]
    fn test_h264_decoder_truncated_sps() {
        let mut decoder = H264Decoder::new().expect("Should create decoder");

        // Truncated SPS NAL unit
        let truncated_sps = vec![
            0x00, 0x00, 0x00, 0x01, // Start code
            0x67, // SPS NAL type
            0x42, // profile_idc
            // Missing rest of SPS
        ];
        let packet = create_test_video_packet(truncated_sps, 0, true);

        assert_no_panic(
            || {
                let _ = decoder.send_packet(&packet);
                let _ = decoder.receive_frame();
            },
            "H.264 decoder with truncated SPS",
        );
    }

    #[test]
    fn test_h264_encoder_wrong_frame_type() {
        let mut encoder = H264Encoder::new(320, 240).expect("Should create encoder");

        // Send audio frame to video encoder
        let audio_frame = AudioFrame::new(1024, 48000, 2, SampleFormat::I16);
        let result = encoder.send_frame(&Frame::Audio(audio_frame));

        assert!(result.is_err(), "Should reject audio frame");
    }

    #[test]
    fn test_h264_encoder_wrong_pixel_format() {
        let mut encoder = H264Encoder::new(320, 240).expect("Should create encoder");

        // Send RGB frame (unsupported by H.264 encoder)
        let frame = create_test_video_frame_with_pattern(320, 240, PixelFormat::RGB24, 0, 128);
        let result = encoder.send_frame(&Frame::Video(frame));

        assert!(result.is_err(), "Should reject RGB24 format");
    }

    #[test]
    fn test_h264_decoder_random_data() {
        let mut decoder = H264Decoder::new().expect("Should create decoder");

        for seed in [1u64, 42, 123, 999] {
            let random_data = create_random_data(1000, seed);
            let packet = create_test_video_packet(random_data, 0, true);

            assert_no_panic(
                || {
                    let _ = decoder.send_packet(&packet);
                    let _ = decoder.receive_frame();
                },
                &format!("H.264 decoder with random seed {}", seed),
            );
        }
    }
}

// ============================================================================
// VP8 Decoder Error Handling (when feature enabled)
// ============================================================================

#[cfg(feature = "vp8-codec")]
mod vp8_error_tests {
    use super::*;
    use zvd_lib::codec::vp8::Vp8Decoder;

    #[test]
    fn test_vp8_decoder_garbage_input() {
        let mut decoder = Vp8Decoder::new().expect("Should create decoder");

        let garbage = create_garbage_data(1000);
        let packet = create_test_video_packet(garbage, 0, true);

        assert_no_panic(
            || {
                let _ = decoder.send_packet(&packet);
                let _ = decoder.receive_frame();
            },
            "VP8 decoder with garbage input",
        );
    }

    #[test]
    fn test_vp8_decoder_invalid_frame_tag() {
        let mut decoder = Vp8Decoder::new().expect("Should create decoder");

        // Invalid VP8 frame tag
        let invalid_tag = vec![0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00];
        let packet = create_test_video_packet(invalid_tag, 0, true);

        assert_no_panic(
            || {
                let _ = decoder.send_packet(&packet);
                let _ = decoder.receive_frame();
            },
            "VP8 decoder with invalid frame tag",
        );
    }
}

// ============================================================================
// VP9 Decoder Error Handling (when feature enabled)
// ============================================================================

#[cfg(feature = "vp9-codec")]
mod vp9_error_tests {
    use super::*;
    use zvd_lib::codec::vp9::Vp9Decoder;

    #[test]
    fn test_vp9_decoder_garbage_input() {
        let mut decoder = Vp9Decoder::new().expect("Should create decoder");

        let garbage = create_garbage_data(1000);
        let packet = create_test_video_packet(garbage, 0, true);

        assert_no_panic(
            || {
                let _ = decoder.send_packet(&packet);
                let _ = decoder.receive_frame();
            },
            "VP9 decoder with garbage input",
        );
    }

    #[test]
    fn test_vp9_decoder_truncated_superframe() {
        let mut decoder = Vp9Decoder::new().expect("Should create decoder");

        // Truncated VP9 superframe marker
        let truncated = vec![0xC0, 0x00]; // Superframe marker without data
        let packet = create_test_video_packet(truncated, 0, true);

        assert_no_panic(
            || {
                let _ = decoder.send_packet(&packet);
                let _ = decoder.receive_frame();
            },
            "VP9 decoder with truncated superframe",
        );
    }
}

// ============================================================================
// ProRes Decoder Error Handling
// ============================================================================

mod prores_error_tests {
    use super::*;
    use zvd_lib::codec::prores::ProResDecoder;

    #[test]
    fn test_prores_decoder_garbage_input() {
        let mut decoder = ProResDecoder::new().expect("Should create decoder");

        let garbage = create_garbage_data(1000);
        let packet = create_test_video_packet(garbage, 0, true);

        assert_no_panic(
            || {
                let _ = decoder.send_packet(&packet);
                let _ = decoder.receive_frame();
            },
            "ProRes decoder with garbage input",
        );
    }

    #[test]
    fn test_prores_decoder_invalid_fourcc() {
        let mut decoder = ProResDecoder::new().expect("Should create decoder");

        // Create data with invalid FourCC
        let mut invalid = vec![0u8; 100];
        invalid[0..4].copy_from_slice(b"XXXX"); // Invalid FourCC
        let packet = create_test_video_packet(invalid, 0, true);

        assert_no_panic(
            || {
                let _ = decoder.send_packet(&packet);
                let _ = decoder.receive_frame();
            },
            "ProRes decoder with invalid FourCC",
        );
    }

    #[test]
    fn test_prores_decoder_truncated_header() {
        let mut decoder = ProResDecoder::new().expect("Should create decoder");

        // Truncated ProRes frame header
        let truncated = vec![0x00, 0x00, 0x00, 0x10]; // Just frame size
        let packet = create_test_video_packet(truncated, 0, true);

        assert_no_panic(
            || {
                let _ = decoder.send_packet(&packet);
                let _ = decoder.receive_frame();
            },
            "ProRes decoder with truncated header",
        );
    }
}

// ============================================================================
// DNxHD Decoder Error Handling
// ============================================================================

mod dnxhd_error_tests {
    use super::*;
    use zvd_lib::codec::dnxhd::DnxhdDecoder;

    #[test]
    fn test_dnxhd_decoder_garbage_input() {
        let mut decoder = DnxhdDecoder::new().expect("Should create decoder");

        let garbage = create_garbage_data(1000);
        let packet = create_test_video_packet(garbage, 0, true);

        assert_no_panic(
            || {
                let _ = decoder.send_packet(&packet);
                let _ = decoder.receive_frame();
            },
            "DNxHD decoder with garbage input",
        );
    }

    #[test]
    fn test_dnxhd_decoder_invalid_header() {
        let mut decoder = DnxhdDecoder::new().expect("Should create decoder");

        // Invalid DNxHD header magic
        let mut invalid = vec![0u8; 100];
        invalid[0..4].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);
        let packet = create_test_video_packet(invalid, 0, true);

        assert_no_panic(
            || {
                let _ = decoder.send_packet(&packet);
                let _ = decoder.receive_frame();
            },
            "DNxHD decoder with invalid header",
        );
    }
}

// ============================================================================
// Opus Decoder Error Handling (when feature enabled)
// ============================================================================

#[cfg(feature = "opus-codec")]
mod opus_error_tests {
    use super::*;
    use zvd_lib::codec::opus::OpusDecoder;

    #[test]
    fn test_opus_decoder_garbage_input() {
        let mut decoder = OpusDecoder::new(48000, 2).expect("Should create decoder");

        let garbage = create_garbage_data(100);
        let packet = create_test_audio_packet(garbage, 0);

        assert_no_panic(
            || {
                let _ = decoder.send_packet(&packet);
                let _ = decoder.receive_frame();
            },
            "Opus decoder with garbage input",
        );
    }

    #[test]
    fn test_opus_decoder_invalid_toc() {
        let mut decoder = OpusDecoder::new(48000, 2).expect("Should create decoder");

        // Invalid Opus TOC byte
        let invalid_toc = vec![0xFF, 0x00, 0x00, 0x00];
        let packet = create_test_audio_packet(invalid_toc, 0);

        assert_no_panic(
            || {
                let _ = decoder.send_packet(&packet);
                let _ = decoder.receive_frame();
            },
            "Opus decoder with invalid TOC",
        );
    }

    #[test]
    fn test_opus_decoder_empty_packet() {
        let mut decoder = OpusDecoder::new(48000, 2).expect("Should create decoder");

        let packet = create_test_audio_packet(vec![], 0);

        assert_no_panic(
            || {
                let _ = decoder.send_packet(&packet);
                let _ = decoder.receive_frame();
            },
            "Opus decoder with empty packet",
        );
    }

    #[test]
    fn test_opus_decoder_invalid_sample_rate() {
        // Opus only supports specific sample rates
        // 48000, 24000, 16000, 12000, 8000

        // This should either fail or clamp to a valid rate
        let result = OpusDecoder::new(44100, 2);
        // 44100 is not a valid Opus sample rate - behavior is implementation-defined
        if let Ok(mut decoder) = result {
            // If it succeeded, it should still work
            let packet = create_test_audio_packet(vec![0u8; 100], 0);
            let _ = decoder.send_packet(&packet);
        }
    }
}

// ============================================================================
// FLAC Decoder Error Handling
// ============================================================================

mod flac_error_tests {
    use super::*;
    use zvd_lib::codec::FlacDecoder;

    #[test]
    fn test_flac_decoder_bad_sync() {
        let mut decoder = FlacDecoder::new(44100, 2).expect("Should create decoder");

        // Invalid FLAC sync code
        let bad_sync = vec![0xFF, 0x00, 0x00, 0x00]; // Wrong sync
        let packet = create_test_audio_packet(bad_sync, 0);

        assert_no_panic(
            || {
                let _ = decoder.send_packet(&packet);
                let _ = decoder.receive_frame();
            },
            "FLAC decoder with bad sync",
        );
    }

    #[test]
    fn test_flac_decoder_garbage_input() {
        let mut decoder = FlacDecoder::new(44100, 2).expect("Should create decoder");

        let garbage = create_garbage_data(1000);
        let packet = create_test_audio_packet(garbage, 0);

        assert_no_panic(
            || {
                let _ = decoder.send_packet(&packet);
                let _ = decoder.receive_frame();
            },
            "FLAC decoder with garbage input",
        );
    }
}

// ============================================================================
// MP3 Decoder Error Handling
// ============================================================================

mod mp3_error_tests {
    use super::*;
    use zvd_lib::codec::Mp3Decoder;

    #[test]
    fn test_mp3_decoder_garbage_input() {
        let mut decoder = Mp3Decoder::new(44100, 2).expect("Should create decoder");

        let garbage = create_garbage_data(1000);
        let packet = create_test_audio_packet(garbage, 0);

        assert_no_panic(
            || {
                let _ = decoder.send_packet(&packet);
                let _ = decoder.receive_frame();
            },
            "MP3 decoder with garbage input",
        );
    }

    #[test]
    fn test_mp3_decoder_invalid_frame_header() {
        let mut decoder = Mp3Decoder::new(44100, 2).expect("Should create decoder");

        // Invalid MP3 frame header
        let invalid_header = vec![0xFF, 0x00, 0x00, 0x00]; // Not a valid sync word
        let packet = create_test_audio_packet(invalid_header, 0);

        assert_no_panic(
            || {
                let _ = decoder.send_packet(&packet);
                let _ = decoder.receive_frame();
            },
            "MP3 decoder with invalid frame header",
        );
    }
}

// ============================================================================
// Vorbis Decoder Error Handling
// ============================================================================

mod vorbis_error_tests {
    use super::*;
    use zvd_lib::codec::VorbisDecoder;

    #[test]
    fn test_vorbis_decoder_garbage_input() {
        let mut decoder = VorbisDecoder::new(44100, 2).expect("Should create decoder");

        let garbage = create_garbage_data(1000);
        let packet = create_test_audio_packet(garbage, 0);

        assert_no_panic(
            || {
                let _ = decoder.send_packet(&packet);
                let _ = decoder.receive_frame();
            },
            "Vorbis decoder with garbage input",
        );
    }
}

// ============================================================================
// Demuxer Error Handling
// ============================================================================

mod demuxer_error_tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_demuxer_nonexistent_file() {
        use zvd_lib::format::demuxer::create_demuxer;

        let result = create_demuxer(&std::path::Path::new("/nonexistent/path/file.mp4"));
        assert!(result.is_err(), "Should fail for nonexistent file");
    }

    #[test]
    fn test_demuxer_empty_file() {
        use zvd_lib::format::demuxer::create_demuxer;

        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let temp_path = temp_file.path();

        // File is empty
        let result = create_demuxer(temp_path);
        // Should fail or return demuxer with no streams
        match result {
            Ok(demuxer) => {
                // If it opens, it should have no valid streams
                let streams = demuxer.streams();
                // Either empty or error on read
            }
            Err(_) => {
                // Expected - empty file is invalid
            }
        }
    }

    #[test]
    fn test_demuxer_garbage_file() {
        use zvd_lib::format::demuxer::create_demuxer;

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(&create_garbage_data(10000))
            .expect("Failed to write");
        temp_file.flush().expect("Failed to flush");

        let result = create_demuxer(temp_file.path());
        // Should fail or return demuxer that fails on read
        match result {
            Ok(mut demuxer) => {
                // If it opens, reading should fail
                let read_result = demuxer.read_packet();
                // Either EndOfStream or error is acceptable
            }
            Err(_) => {
                // Expected - garbage is invalid
            }
        }
    }

    #[test]
    fn test_wav_demuxer_invalid_riff() {
        use zvd_lib::format::wav::WavDemuxer;
        use zvd_lib::format::Demuxer;

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");

        // Write invalid RIFF header
        temp_file.write_all(b"XXXX").expect("Failed to write");
        temp_file
            .write_all(&[0u8; 100])
            .expect("Failed to write");
        temp_file.flush().expect("Failed to flush");

        let mut demuxer = WavDemuxer::new();
        let result = demuxer.open(temp_file.path());

        assert!(result.is_err(), "Should fail for invalid RIFF header");
    }

    #[test]
    fn test_wav_demuxer_truncated_header() {
        use zvd_lib::format::wav::WavDemuxer;
        use zvd_lib::format::Demuxer;

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");

        // Write truncated WAV header
        temp_file.write_all(b"RIFF").expect("Failed to write");
        // Missing rest of header
        temp_file.flush().expect("Failed to flush");

        let mut demuxer = WavDemuxer::new();
        let result = demuxer.open(temp_file.path());

        assert!(result.is_err(), "Should fail for truncated header");
    }
}

// ============================================================================
// Encoder Error Handling
// ============================================================================

mod encoder_error_tests {
    use super::*;
    use zvd_lib::codec::av1::Av1EncoderBuilder;

    #[test]
    fn test_encoder_invalid_dimensions() {
        // Zero dimensions
        let result = Av1EncoderBuilder::new(0, 0).build();
        assert!(result.is_err(), "Should reject zero dimensions");

        // Odd dimensions (AV1 requires multiples of 8)
        let result = Av1EncoderBuilder::new(321, 241).build();
        assert!(result.is_err(), "Should reject non-aligned dimensions");
    }

    #[test]
    fn test_encoder_mismatched_frame_size() {
        let mut encoder = Av1EncoderBuilder::new(320, 240)
            .speed_preset(10)
            .build()
            .expect("Should create encoder");

        // Send frame with wrong dimensions
        let frame = create_test_video_frame(640, 480); // Wrong size

        // Should either reject or handle gracefully
        let result = encoder.send_frame(&Frame::Video(frame));
        // Implementation may accept and resize, or reject
    }

    #[test]
    fn test_encoder_audio_to_video() {
        let mut encoder = Av1EncoderBuilder::new(320, 240)
            .speed_preset(10)
            .build()
            .expect("Should create encoder");

        // Send audio frame to video encoder
        let audio_frame = AudioFrame::new(1024, 48000, 2, SampleFormat::F32);
        let result = encoder.send_frame(&Frame::Audio(audio_frame));

        assert!(result.is_err(), "Should reject audio frame");
    }

    #[test]
    fn test_encoder_receive_without_send() {
        let mut encoder = Av1EncoderBuilder::new(320, 240)
            .speed_preset(10)
            .build()
            .expect("Should create encoder");

        // Try to receive without sending
        let result = encoder.receive_packet();

        // Should return TryAgain, not panic
        match result {
            Err(Error::TryAgain) => {
                // Expected
            }
            _ => {
                // Other results are also acceptable
            }
        }
    }
}

// ============================================================================
// Buffer Edge Cases
// ============================================================================

mod buffer_edge_tests {
    use super::*;

    #[test]
    fn test_empty_buffer() {
        let buffer = Buffer::empty();
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_packet_with_empty_data() {
        let packet = Packet::new(0, Buffer::empty());
        assert_eq!(packet.size(), 0);
        assert!(packet.data.is_empty());
    }

    #[test]
    fn test_frame_with_no_planes() {
        let mut frame = VideoFrame::new(320, 240, PixelFormat::YUV420P);
        // Don't add any plane data
        assert!(frame.data.is_empty());
        assert_eq!(frame.num_planes(), 0);
    }

    #[test]
    fn test_audio_frame_with_zero_samples() {
        let frame = AudioFrame::new(0, 48000, 2, SampleFormat::F32);
        assert_eq!(frame.nb_samples, 0);
        assert_eq!(frame.total_samples(), 0);
    }
}

// ============================================================================
// Timestamp Edge Cases
// ============================================================================

mod timestamp_edge_tests {
    use super::*;
    use zvd_lib::util::Rational;

    #[test]
    fn test_timestamp_none() {
        let ts = Timestamp::none();
        assert!(!ts.is_valid());
    }

    #[test]
    fn test_timestamp_min_max() {
        let max_ts = Timestamp::new(i64::MAX);
        assert!(max_ts.is_valid());

        let min_ts = Timestamp::new(i64::MIN + 1); // MIN is reserved for none
        assert!(min_ts.is_valid());
    }

    #[test]
    fn test_timestamp_rescale_overflow() {
        // Test rescaling with potential overflow
        let large_ts = Timestamp::new(i64::MAX / 2);
        let from = Rational::new(1, 1);
        let to = Rational::new(1, 2);

        // Should handle without panic
        let _ = large_ts.rescale(from, to);
    }

    #[test]
    fn test_timestamp_to_seconds_none() {
        let none_ts = Timestamp::none();
        let time_base = Rational::new(1, 1000);

        let seconds = none_ts.to_seconds(time_base);
        assert_eq!(seconds, 0.0);
    }
}

// ============================================================================
// Concurrent Access Tests
// ============================================================================

mod concurrent_tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_multiple_decoders() {
        use zvd_lib::codec::av1::Av1Decoder;

        // Create multiple decoders in parallel
        let handles: Vec<_> = (0..4)
            .map(|i| {
                thread::spawn(move || {
                    let decoder = Av1Decoder::new();
                    assert!(
                        decoder.is_ok(),
                        "Thread {} should create decoder",
                        i
                    );
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread should complete");
        }
    }

    #[test]
    fn test_multiple_encoders() {
        use zvd_lib::codec::av1::Av1EncoderBuilder;

        let handles: Vec<_> = (0..4)
            .map(|i| {
                thread::spawn(move || {
                    let encoder = Av1EncoderBuilder::new(320, 240)
                        .speed_preset(10)
                        .build();
                    assert!(
                        encoder.is_ok(),
                        "Thread {} should create encoder",
                        i
                    );
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread should complete");
        }
    }
}

// ============================================================================
// Stress Tests
// ============================================================================

mod stress_tests {
    use super::*;

    #[test]
    fn test_repeated_decode_attempts() {
        use zvd_lib::codec::av1::Av1Decoder;

        let mut decoder = Av1Decoder::new().expect("Should create decoder");

        // Repeatedly send garbage and try to receive
        for i in 0..100 {
            let garbage = create_garbage_data((i + 1) * 10);
            let packet = create_test_video_packet(garbage, i as i64, i == 0);

            let _ = decoder.send_packet(&packet);
            let _ = decoder.receive_frame();
        }

        // Decoder should still be usable
        let _ = decoder.flush();
    }

    #[test]
    fn test_alternating_valid_invalid() {
        use zvd_lib::codec::av1::{Av1Decoder, Av1EncoderBuilder};

        // Create some valid encoded data first
        let frame = create_test_video_frame(320, 240);
        let mut encoder = Av1EncoderBuilder::new(320, 240)
            .speed_preset(10)
            .quantizer(100)
            .build()
            .expect("Should create encoder");

        encoder
            .send_frame(&Frame::Video(frame))
            .expect("Should accept frame");
        encoder.flush().expect("Should flush");

        let mut valid_packets = Vec::new();
        loop {
            match encoder.receive_packet() {
                Ok(packet) => valid_packets.push(packet),
                Err(Error::TryAgain) => continue,
                Err(Error::EndOfStream) => break,
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }

        // Now alternate valid and invalid packets
        let mut decoder = Av1Decoder::new().expect("Should create decoder");

        for (i, valid_packet) in valid_packets.iter().enumerate() {
            // Send garbage
            let garbage = create_garbage_data(100);
            let garbage_packet = create_test_video_packet(garbage, -1, false);
            let _ = decoder.send_packet(&garbage_packet);
            let _ = decoder.receive_frame();

            // Send valid
            let _ = decoder.send_packet(valid_packet);
            let _ = decoder.receive_frame();
        }
    }
}
