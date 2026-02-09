//! Transcode integration tests for ZVD
//!
//! Tests codec-to-codec transcoding for both video and audio codecs.
//! Each test encodes test data with one codec, decodes it, then re-encodes
//! with another codec to verify interoperability.

#![allow(unused_imports)]

use zvd_lib::codec::frame::{AudioFrame, VideoFrame};
use zvd_lib::codec::{Decoder, Encoder, Frame};
use zvd_lib::error::Error;
use zvd_lib::format::Packet;
use zvd_lib::util::{Buffer, PixelFormat, SampleFormat, Timestamp};

// Include common test utilities
#[path = "../common/mod.rs"]
mod common;

use common::*;

// ============================================================================
// AV1 Codec Tests
// ============================================================================

/// Helper to encode frames with AV1 and collect packets
fn av1_encode_frames(
    width: u32,
    height: u32,
    frames: Vec<Frame>,
    speed: u8,
    quantizer: usize,
) -> Result<Vec<Packet>, Error> {
    use zvd_lib::codec::av1::Av1EncoderBuilder;

    let mut encoder = Av1EncoderBuilder::new(width, height)
        .speed_preset(speed)
        .quantizer(quantizer)
        .max_keyframe_interval(30)
        .build()?;

    let mut packets = Vec::new();

    for frame in frames {
        encoder.send_frame(&frame)?;

        loop {
            match encoder.receive_packet() {
                Ok(packet) => packets.push(packet),
                Err(Error::TryAgain) => break,
                Err(e) => return Err(e),
            }
        }
    }

    encoder.flush()?;

    loop {
        match encoder.receive_packet() {
            Ok(packet) => packets.push(packet),
            Err(Error::TryAgain) => continue,
            Err(Error::EndOfStream) => break,
            Err(e) => return Err(e),
        }
    }

    Ok(packets)
}

/// Helper to decode AV1 packets
fn av1_decode_packets(packets: Vec<Packet>) -> Result<Vec<Frame>, Error> {
    use zvd_lib::codec::av1::Av1Decoder;

    let mut decoder = Av1Decoder::new()?;
    let mut frames = Vec::new();

    for packet in packets {
        decoder.send_packet(&packet)?;

        loop {
            match decoder.receive_frame() {
                Ok(frame) => frames.push(frame),
                Err(Error::TryAgain) => break,
                Err(e) => return Err(e),
            }
        }
    }

    decoder.flush()?;

    Ok(frames)
}

#[test]
fn test_av1_encode_decode_roundtrip() {
    let (width, height) = resolutions::TEST_SMALL;
    let num_frames = 5;

    // Create test frames
    let test_frames: Vec<Frame> = create_video_sequence(width, height, num_frames)
        .into_iter()
        .map(Frame::Video)
        .collect();

    // Encode with AV1
    let packets = av1_encode_frames(width, height, test_frames, 10, 100)
        .expect("AV1 encoding should succeed");

    assert!(!packets.is_empty(), "Should produce encoded packets");

    // Decode AV1
    let decoded_frames = av1_decode_packets(packets).expect("AV1 decoding should succeed");

    assert!(!decoded_frames.is_empty(), "Should decode frames");

    // Verify decoded frame properties
    for frame in &decoded_frames {
        if let Frame::Video(video_frame) = frame {
            verify_video_frame_properties(video_frame, width, height)
                .expect("Frame properties should match");
        }
    }
}

#[test]
fn test_av1_different_resolutions() {
    let test_resolutions = [
        resolutions::QCIF,
        resolutions::CIF,
        resolutions::TEST_SMALL,
    ];

    for (width, height) in test_resolutions {
        // Ensure dimensions are multiples of 8 for AV1
        let width = (width / 8) * 8;
        let height = (height / 8) * 8;

        let frame = create_test_video_frame_with_pattern(width, height, PixelFormat::YUV420P, 0, 128);

        let packets = av1_encode_frames(width, height, vec![Frame::Video(frame)], 10, 120)
            .unwrap_or_else(|e| panic!("AV1 encoding failed for {}x{}: {:?}", width, height, e));

        assert!(
            !packets.is_empty(),
            "Should produce packets for {}x{}",
            width,
            height
        );

        let decoded = av1_decode_packets(packets)
            .unwrap_or_else(|e| panic!("AV1 decoding failed for {}x{}: {:?}", width, height, e));

        assert!(
            !decoded.is_empty(),
            "Should decode frames for {}x{}",
            width,
            height
        );
    }
}

// ============================================================================
// H.264 Codec Tests (when feature is enabled)
// ============================================================================

#[cfg(feature = "h264")]
mod h264_tests {
    use super::*;
    use zvd_lib::codec::h264::{H264Decoder, H264Encoder};

    fn h264_encode_frames(
        width: u32,
        height: u32,
        frames: Vec<Frame>,
    ) -> Result<Vec<Packet>, Error> {
        let mut encoder = H264Encoder::new(width, height)?;
        let mut packets = Vec::new();

        for frame in frames {
            encoder.send_frame(&frame)?;

            loop {
                match encoder.receive_packet() {
                    Ok(packet) => packets.push(packet),
                    Err(Error::TryAgain) => break,
                    Err(e) => return Err(e),
                }
            }
        }

        encoder.flush()?;

        loop {
            match encoder.receive_packet() {
                Ok(packet) => packets.push(packet),
                Err(Error::TryAgain) => continue,
                Err(Error::EndOfStream) => break,
                Err(e) => return Err(e),
            }
        }

        Ok(packets)
    }

    fn h264_decode_packets(packets: Vec<Packet>) -> Result<Vec<Frame>, Error> {
        let mut decoder = H264Decoder::new()?;
        let mut frames = Vec::new();

        for packet in packets {
            decoder.send_packet(&packet)?;

            loop {
                match decoder.receive_frame() {
                    Ok(frame) => frames.push(frame),
                    Err(Error::TryAgain) => break,
                    Err(e) => return Err(e),
                }
            }
        }

        decoder.flush()?;

        Ok(frames)
    }

    #[test]
    fn test_h264_encode_decode_roundtrip() {
        let (width, height) = resolutions::TEST_SMALL;
        let num_frames = 5;

        let test_frames: Vec<Frame> = create_video_sequence(width, height, num_frames)
            .into_iter()
            .map(Frame::Video)
            .collect();

        let packets =
            h264_encode_frames(width, height, test_frames).expect("H.264 encoding should succeed");

        assert!(!packets.is_empty(), "Should produce encoded packets");

        let decoded_frames =
            h264_decode_packets(packets).expect("H.264 decoding should succeed");

        assert!(!decoded_frames.is_empty(), "Should decode frames");

        for frame in &decoded_frames {
            if let Frame::Video(video_frame) = frame {
                verify_video_frame_properties(video_frame, width, height)
                    .expect("Frame properties should match");
            }
        }
    }

    #[test]
    fn test_h264_to_av1_transcode() {
        let (width, height) = (320, 240);
        let num_frames = 3;

        // Step 1: Encode with H.264
        let test_frames: Vec<Frame> = create_video_sequence(width, height, num_frames)
            .into_iter()
            .map(Frame::Video)
            .collect();

        let h264_packets = h264_encode_frames(width, height, test_frames)
            .expect("H.264 encoding should succeed");

        // Step 2: Decode H.264
        let decoded_frames =
            h264_decode_packets(h264_packets).expect("H.264 decoding should succeed");

        assert!(!decoded_frames.is_empty(), "Should have decoded frames");

        // Step 3: Re-encode with AV1
        let av1_packets = av1_encode_frames(width, height, decoded_frames, 10, 120)
            .expect("AV1 re-encoding should succeed");

        assert!(!av1_packets.is_empty(), "Should produce AV1 packets");

        // Step 4: Decode AV1 to verify
        let final_frames =
            av1_decode_packets(av1_packets).expect("AV1 decoding should succeed");

        assert!(
            !final_frames.is_empty(),
            "Should have final decoded frames"
        );

        // Verify final frame dimensions
        if let Some(Frame::Video(frame)) = final_frames.first() {
            verify_video_frame_properties(frame, width, height)
                .expect("Final frame dimensions should match");
        }
    }

    #[test]
    fn test_av1_to_h264_transcode() {
        let (width, height) = (320, 240);
        let num_frames = 3;

        // Step 1: Encode with AV1
        let test_frames: Vec<Frame> = create_video_sequence(width, height, num_frames)
            .into_iter()
            .map(Frame::Video)
            .collect();

        let av1_packets = av1_encode_frames(width, height, test_frames, 10, 120)
            .expect("AV1 encoding should succeed");

        // Step 2: Decode AV1
        let decoded_frames =
            av1_decode_packets(av1_packets).expect("AV1 decoding should succeed");

        // Step 3: Re-encode with H.264
        let h264_packets = h264_encode_frames(width, height, decoded_frames)
            .expect("H.264 re-encoding should succeed");

        assert!(!h264_packets.is_empty(), "Should produce H.264 packets");

        // Step 4: Decode H.264 to verify
        let final_frames =
            h264_decode_packets(h264_packets).expect("H.264 decoding should succeed");

        assert!(
            !final_frames.is_empty(),
            "Should have final decoded frames"
        );
    }
}

// ============================================================================
// VP8 Codec Tests (when feature is enabled)
// ============================================================================

#[cfg(feature = "vp8-codec")]
mod vp8_tests {
    use super::*;
    use zvd_lib::codec::vp8::{Vp8Decoder, Vp8Encoder};

    fn vp8_encode_frames(
        width: u32,
        height: u32,
        frames: Vec<Frame>,
    ) -> Result<Vec<Packet>, Error> {
        let mut encoder = Vp8Encoder::new(width, height)?;
        let mut packets = Vec::new();

        for frame in frames {
            encoder.send_frame(&frame)?;

            loop {
                match encoder.receive_packet() {
                    Ok(packet) => packets.push(packet),
                    Err(Error::TryAgain) => break,
                    Err(e) => return Err(e),
                }
            }
        }

        encoder.flush()?;

        loop {
            match encoder.receive_packet() {
                Ok(packet) => packets.push(packet),
                Err(Error::TryAgain) => continue,
                Err(Error::EndOfStream) => break,
                Err(e) => return Err(e),
            }
        }

        Ok(packets)
    }

    fn vp8_decode_packets(packets: Vec<Packet>) -> Result<Vec<Frame>, Error> {
        let mut decoder = Vp8Decoder::new()?;
        let mut frames = Vec::new();

        for packet in packets {
            decoder.send_packet(&packet)?;

            loop {
                match decoder.receive_frame() {
                    Ok(frame) => frames.push(frame),
                    Err(Error::TryAgain) => break,
                    Err(e) => return Err(e),
                }
            }
        }

        decoder.flush()?;

        Ok(frames)
    }

    #[test]
    fn test_vp8_encode_decode_roundtrip() {
        let (width, height) = resolutions::TEST_SMALL;
        let num_frames = 5;

        let test_frames: Vec<Frame> = create_video_sequence(width, height, num_frames)
            .into_iter()
            .map(Frame::Video)
            .collect();

        let packets =
            vp8_encode_frames(width, height, test_frames).expect("VP8 encoding should succeed");

        assert!(!packets.is_empty(), "Should produce encoded packets");

        let decoded_frames =
            vp8_decode_packets(packets).expect("VP8 decoding should succeed");

        assert!(!decoded_frames.is_empty(), "Should decode frames");
    }

    #[test]
    fn test_vp8_to_av1_transcode() {
        let (width, height) = (320, 240);
        let num_frames = 3;

        // Encode with VP8
        let test_frames: Vec<Frame> = create_video_sequence(width, height, num_frames)
            .into_iter()
            .map(Frame::Video)
            .collect();

        let vp8_packets =
            vp8_encode_frames(width, height, test_frames).expect("VP8 encoding should succeed");

        // Decode VP8
        let decoded_frames =
            vp8_decode_packets(vp8_packets).expect("VP8 decoding should succeed");

        // Re-encode with AV1
        let av1_packets = av1_encode_frames(width, height, decoded_frames, 10, 120)
            .expect("AV1 re-encoding should succeed");

        assert!(!av1_packets.is_empty(), "Should produce AV1 packets");

        // Verify AV1 output
        let final_frames =
            av1_decode_packets(av1_packets).expect("AV1 decoding should succeed");

        assert!(
            !final_frames.is_empty(),
            "Should have final decoded frames"
        );
    }
}

// ============================================================================
// VP9 Codec Tests (when feature is enabled)
// ============================================================================

#[cfg(feature = "vp9-codec")]
mod vp9_tests {
    use super::*;
    use zvd_lib::codec::vp9::{Vp9Decoder, Vp9Encoder};

    fn vp9_encode_frames(
        width: u32,
        height: u32,
        frames: Vec<Frame>,
    ) -> Result<Vec<Packet>, Error> {
        let mut encoder = Vp9Encoder::new(width, height)?;
        let mut packets = Vec::new();

        for frame in frames {
            encoder.send_frame(&frame)?;

            loop {
                match encoder.receive_packet() {
                    Ok(packet) => packets.push(packet),
                    Err(Error::TryAgain) => break,
                    Err(e) => return Err(e),
                }
            }
        }

        encoder.flush()?;

        loop {
            match encoder.receive_packet() {
                Ok(packet) => packets.push(packet),
                Err(Error::TryAgain) => continue,
                Err(Error::EndOfStream) => break,
                Err(e) => return Err(e),
            }
        }

        Ok(packets)
    }

    fn vp9_decode_packets(packets: Vec<Packet>) -> Result<Vec<Frame>, Error> {
        let mut decoder = Vp9Decoder::new()?;
        let mut frames = Vec::new();

        for packet in packets {
            decoder.send_packet(&packet)?;

            loop {
                match decoder.receive_frame() {
                    Ok(frame) => frames.push(frame),
                    Err(Error::TryAgain) => break,
                    Err(e) => return Err(e),
                }
            }
        }

        decoder.flush()?;

        Ok(frames)
    }

    #[test]
    fn test_vp9_encode_decode_roundtrip() {
        let (width, height) = resolutions::TEST_SMALL;
        let num_frames = 5;

        let test_frames: Vec<Frame> = create_video_sequence(width, height, num_frames)
            .into_iter()
            .map(Frame::Video)
            .collect();

        let packets =
            vp9_encode_frames(width, height, test_frames).expect("VP9 encoding should succeed");

        assert!(!packets.is_empty(), "Should produce encoded packets");

        let decoded_frames =
            vp9_decode_packets(packets).expect("VP9 decoding should succeed");

        assert!(!decoded_frames.is_empty(), "Should decode frames");
    }

    #[test]
    fn test_vp9_to_av1_transcode() {
        let (width, height) = (320, 240);
        let num_frames = 3;

        // Encode with VP9
        let test_frames: Vec<Frame> = create_video_sequence(width, height, num_frames)
            .into_iter()
            .map(Frame::Video)
            .collect();

        let vp9_packets =
            vp9_encode_frames(width, height, test_frames).expect("VP9 encoding should succeed");

        // Decode VP9
        let decoded_frames =
            vp9_decode_packets(vp9_packets).expect("VP9 decoding should succeed");

        // Re-encode with AV1
        let av1_packets = av1_encode_frames(width, height, decoded_frames, 10, 120)
            .expect("AV1 re-encoding should succeed");

        assert!(!av1_packets.is_empty(), "Should produce AV1 packets");
    }

    #[test]
    fn test_av1_to_vp9_transcode() {
        let (width, height) = (320, 240);
        let num_frames = 3;

        // Encode with AV1
        let test_frames: Vec<Frame> = create_video_sequence(width, height, num_frames)
            .into_iter()
            .map(Frame::Video)
            .collect();

        let av1_packets = av1_encode_frames(width, height, test_frames, 10, 120)
            .expect("AV1 encoding should succeed");

        // Decode AV1
        let decoded_frames =
            av1_decode_packets(av1_packets).expect("AV1 decoding should succeed");

        // Re-encode with VP9
        let vp9_packets = vp9_encode_frames(width, height, decoded_frames)
            .expect("VP9 re-encoding should succeed");

        assert!(!vp9_packets.is_empty(), "Should produce VP9 packets");
    }
}

// ============================================================================
// ProRes Codec Tests
// ============================================================================

mod prores_tests {
    use super::*;
    use zvd_lib::codec::prores::{ProResDecoder, ProResEncoder, ProResProfile};

    fn prores_encode_frames(
        width: u32,
        height: u32,
        frames: Vec<Frame>,
        profile: ProResProfile,
    ) -> Result<Vec<Packet>, Error> {
        let mut encoder = ProResEncoder::new(width as u16, height as u16, profile)?;
        let mut packets = Vec::new();

        for frame in frames {
            encoder.send_frame(&frame)?;

            loop {
                match encoder.receive_packet() {
                    Ok(packet) => packets.push(packet),
                    Err(Error::TryAgain) => break,
                    Err(e) => return Err(e),
                }
            }
        }

        encoder.flush()?;

        loop {
            match encoder.receive_packet() {
                Ok(packet) => packets.push(packet),
                Err(Error::TryAgain) => continue,
                Err(Error::EndOfStream) => break,
                Err(e) => return Err(e),
            }
        }

        Ok(packets)
    }

    fn prores_decode_packets(packets: Vec<Packet>) -> Result<Vec<Frame>, Error> {
        let mut decoder = ProResDecoder::new()?;
        let mut frames = Vec::new();

        for packet in packets {
            decoder.send_packet(&packet)?;

            loop {
                match decoder.receive_frame() {
                    Ok(frame) => frames.push(frame),
                    Err(Error::TryAgain) => break,
                    Err(e) => return Err(e),
                }
            }
        }

        decoder.flush()?;

        Ok(frames)
    }

    #[test]
    fn test_prores_encode_decode_roundtrip() {
        let (width, height) = resolutions::TEST_SMALL;
        let num_frames = 3;

        let test_frames: Vec<Frame> = create_video_sequence(width, height, num_frames)
            .into_iter()
            .map(Frame::Video)
            .collect();

        let packets = prores_encode_frames(width, height, test_frames, ProResProfile::Standard)
            .expect("ProRes encoding should succeed");

        assert!(!packets.is_empty(), "Should produce encoded packets");

        let decoded_frames =
            prores_decode_packets(packets).expect("ProRes decoding should succeed");

        assert!(!decoded_frames.is_empty(), "Should decode frames");
    }

    #[test]
    fn test_prores_different_profiles() {
        let (width, height) = resolutions::TEST_SMALL;

        let profiles = [
            ProResProfile::Proxy,
            ProResProfile::Lt,
            ProResProfile::Standard,
            ProResProfile::Hq,
        ];

        for profile in profiles {
            let frame = create_test_video_frame_with_pattern(
                width,
                height,
                PixelFormat::YUV422P,
                0,
                128,
            );

            let packets = prores_encode_frames(width, height, vec![Frame::Video(frame)], profile)
                .unwrap_or_else(|e| panic!("ProRes {:?} encoding failed: {:?}", profile, e));

            assert!(
                !packets.is_empty(),
                "Should produce packets for ProRes {:?}",
                profile
            );
        }
    }

    #[test]
    fn test_prores_to_av1_transcode() {
        let (width, height) = (320, 240);

        // Encode with ProRes
        let frame = create_test_video_frame_with_pattern(width, height, PixelFormat::YUV422P, 0, 128);

        let prores_packets =
            prores_encode_frames(width, height, vec![Frame::Video(frame)], ProResProfile::Standard)
                .expect("ProRes encoding should succeed");

        // Decode ProRes
        let decoded_frames =
            prores_decode_packets(prores_packets).expect("ProRes decoding should succeed");

        // Re-encode with AV1 (may need to convert 422 to 420)
        // Note: AV1 encoder may handle this conversion internally
        let av1_packets = av1_encode_frames(width, height, decoded_frames, 10, 120)
            .expect("AV1 re-encoding should succeed");

        assert!(!av1_packets.is_empty(), "Should produce AV1 packets");
    }
}

// ============================================================================
// DNxHD Codec Tests
// ============================================================================

mod dnxhd_tests {
    use super::*;
    use zvd_lib::codec::dnxhd::{DnxhdDecoder, DnxhdEncoder, DnxhdProfile};

    fn dnxhd_encode_frames(
        width: u32,
        height: u32,
        frames: Vec<Frame>,
        profile: DnxhdProfile,
    ) -> Result<Vec<Packet>, Error> {
        let mut encoder = DnxhdEncoder::new(width as u16, height as u16, profile)?;
        let mut packets = Vec::new();

        for frame in frames {
            encoder.send_frame(&frame)?;

            loop {
                match encoder.receive_packet() {
                    Ok(packet) => packets.push(packet),
                    Err(Error::TryAgain) => break,
                    Err(e) => return Err(e),
                }
            }
        }

        encoder.flush()?;

        loop {
            match encoder.receive_packet() {
                Ok(packet) => packets.push(packet),
                Err(Error::TryAgain) => continue,
                Err(Error::EndOfStream) => break,
                Err(e) => return Err(e),
            }
        }

        Ok(packets)
    }

    fn dnxhd_decode_packets(packets: Vec<Packet>) -> Result<Vec<Frame>, Error> {
        let mut decoder = DnxhdDecoder::new()?;
        let mut frames = Vec::new();

        for packet in packets {
            decoder.send_packet(&packet)?;

            loop {
                match decoder.receive_frame() {
                    Ok(frame) => frames.push(frame),
                    Err(Error::TryAgain) => break,
                    Err(e) => return Err(e),
                }
            }
        }

        decoder.flush()?;

        Ok(frames)
    }

    #[test]
    fn test_dnxhd_encode_decode_roundtrip() {
        // DNxHD requires specific resolutions - use 1920x1080
        let (width, height) = (1920, 1080);
        let num_frames = 2;

        let test_frames: Vec<Frame> = create_video_sequence(width, height, num_frames)
            .into_iter()
            .map(Frame::Video)
            .collect();

        let packets = dnxhd_encode_frames(width, height, test_frames, DnxhdProfile::DnxHd185x)
            .expect("DNxHD encoding should succeed");

        assert!(!packets.is_empty(), "Should produce encoded packets");

        let decoded_frames =
            dnxhd_decode_packets(packets).expect("DNxHD decoding should succeed");

        assert!(!decoded_frames.is_empty(), "Should decode frames");
    }

    #[test]
    fn test_dnxhd_to_prores_transcode() {
        let (width, height) = (1920, 1080);

        // Encode with DNxHD
        let frame = create_test_video_frame_with_pattern(width, height, PixelFormat::YUV422P, 0, 128);

        let dnxhd_packets =
            dnxhd_encode_frames(width, height, vec![Frame::Video(frame)], DnxhdProfile::DnxHd185x)
                .expect("DNxHD encoding should succeed");

        // Decode DNxHD
        let decoded_frames =
            dnxhd_decode_packets(dnxhd_packets).expect("DNxHD decoding should succeed");

        // Re-encode with ProRes
        use zvd_lib::codec::prores::{ProResEncoder, ProResProfile};
        let mut prores_encoder =
            ProResEncoder::new(width as u16, height as u16, ProResProfile::Standard)
                .expect("ProRes encoder creation should succeed");

        for frame in decoded_frames {
            prores_encoder
                .send_frame(&frame)
                .expect("ProRes encoding should succeed");
        }
    }
}

// ============================================================================
// Audio Codec Tests
// ============================================================================

#[cfg(feature = "opus-codec")]
mod opus_tests {
    use super::*;
    use zvd_lib::codec::opus::{OpusDecoder, OpusEncoder};

    fn opus_encode_frames(
        sample_rate: u32,
        channels: u16,
        frames: Vec<Frame>,
    ) -> Result<Vec<Packet>, Error> {
        let mut encoder = OpusEncoder::new(sample_rate, channels)?;
        let mut packets = Vec::new();

        for frame in frames {
            encoder.send_frame(&frame)?;

            loop {
                match encoder.receive_packet() {
                    Ok(packet) => packets.push(packet),
                    Err(Error::TryAgain) => break,
                    Err(e) => return Err(e),
                }
            }
        }

        encoder.flush()?;

        loop {
            match encoder.receive_packet() {
                Ok(packet) => packets.push(packet),
                Err(Error::TryAgain) => continue,
                Err(Error::EndOfStream) => break,
                Err(e) => return Err(e),
            }
        }

        Ok(packets)
    }

    fn opus_decode_packets(
        packets: Vec<Packet>,
        sample_rate: u32,
        channels: u16,
    ) -> Result<Vec<Frame>, Error> {
        let mut decoder = OpusDecoder::new(sample_rate, channels)?;
        let mut frames = Vec::new();

        for packet in packets {
            decoder.send_packet(&packet)?;

            loop {
                match decoder.receive_frame() {
                    Ok(frame) => frames.push(frame),
                    Err(Error::TryAgain) => break,
                    Err(e) => return Err(e),
                }
            }
        }

        decoder.flush()?;

        Ok(frames)
    }

    #[test]
    fn test_opus_encode_decode_roundtrip() {
        let sample_rate = audio::SAMPLE_RATE_48000;
        let channels = audio::CHANNELS_STEREO;
        let num_frames = 5;

        let test_frames: Vec<Frame> =
            create_audio_sequence(sample_rate, channels, audio::FRAME_SIZE, num_frames)
                .into_iter()
                .map(Frame::Audio)
                .collect();

        let packets = opus_encode_frames(sample_rate, channels, test_frames)
            .expect("Opus encoding should succeed");

        assert!(!packets.is_empty(), "Should produce encoded packets");

        let decoded_frames = opus_decode_packets(packets, sample_rate, channels)
            .expect("Opus decoding should succeed");

        assert!(!decoded_frames.is_empty(), "Should decode frames");

        for frame in &decoded_frames {
            if let Frame::Audio(audio_frame) = frame {
                verify_audio_frame_properties(audio_frame, sample_rate, channels)
                    .expect("Audio frame properties should match");
            }
        }
    }

    #[test]
    fn test_opus_different_sample_rates() {
        let sample_rates = [
            audio::SAMPLE_RATE_8000,
            audio::SAMPLE_RATE_16000,
            audio::SAMPLE_RATE_48000,
        ];

        for sample_rate in sample_rates {
            let frame = create_test_audio_frame(sample_rate, 2, 960);

            let packets = opus_encode_frames(sample_rate, 2, vec![Frame::Audio(frame)])
                .unwrap_or_else(|e| panic!("Opus encoding failed for {} Hz: {:?}", sample_rate, e));

            assert!(
                !packets.is_empty(),
                "Should produce packets for {} Hz",
                sample_rate
            );
        }
    }
}

// ============================================================================
// PCM Codec Tests (Always Available)
// ============================================================================

mod pcm_tests {
    use super::*;
    use zvd_lib::codec::pcm::{PcmConfig, PcmDecoder, PcmEncoder};

    #[test]
    fn test_pcm_encode_decode_roundtrip() {
        let sample_rate = audio::SAMPLE_RATE_48000;
        let channels = audio::CHANNELS_STEREO;
        let format = SampleFormat::I16;

        let config = PcmConfig::new(format, channels, sample_rate);
        let encoder = PcmEncoder::new(config.clone());
        let decoder = PcmDecoder::new(config);

        // Create test audio frame
        let frame = create_test_audio_frame_i16(sample_rate, channels, 1024);

        // Encode
        let packet = encoder
            .encode_frame(&Frame::Audio(frame.clone()))
            .expect("PCM encoding should succeed");

        // Decode
        let decoded = decoder
            .decode_packet(&packet)
            .expect("PCM decoding should succeed");

        // Verify
        if let Frame::Audio(audio_frame) = decoded {
            verify_audio_frame_properties(&audio_frame, sample_rate, channels)
                .expect("Audio properties should match");
        } else {
            panic!("Expected audio frame");
        }
    }

    #[test]
    fn test_pcm_different_formats() {
        let sample_rate = audio::SAMPLE_RATE_44100;
        let channels = audio::CHANNELS_STEREO;

        let formats = [
            SampleFormat::U8,
            SampleFormat::I16,
            SampleFormat::I32,
            SampleFormat::F32,
        ];

        for format in formats {
            let config = PcmConfig::new(format, channels, sample_rate);
            let encoder = PcmEncoder::new(config.clone());
            let decoder = PcmDecoder::new(config);

            let frame = create_silence_audio_frame(sample_rate, channels, 256, format);

            let packet = encoder
                .encode_frame(&Frame::Audio(frame))
                .unwrap_or_else(|e| panic!("PCM {:?} encoding failed: {:?}", format, e));

            let decoded = decoder
                .decode_packet(&packet)
                .unwrap_or_else(|e| panic!("PCM {:?} decoding failed: {:?}", format, e));

            assert!(
                matches!(decoded, Frame::Audio(_)),
                "Should decode audio for {:?}",
                format
            );
        }
    }
}

// ============================================================================
// Cross-Codec Transcode Tests
// ============================================================================

#[test]
fn test_av1_quality_levels() {
    let (width, height) = resolutions::TEST_SMALL;
    let frame = create_test_video_frame(width, height);

    let quality_levels = [60, 100, 150, 200];

    for quantizer in quality_levels {
        let packets = av1_encode_frames(width, height, vec![Frame::Video(frame.clone())], 10, quantizer)
            .unwrap_or_else(|e| panic!("AV1 encoding failed for quantizer {}: {:?}", quantizer, e));

        assert!(
            !packets.is_empty(),
            "Should produce packets for quantizer {}",
            quantizer
        );

        // Higher quantizer should produce smaller packets (lower quality)
        let total_size: usize = packets.iter().map(|p| p.data.len()).sum();
        println!("Quantizer {}: {} bytes", quantizer, total_size);
    }
}

#[test]
fn test_av1_speed_presets() {
    let (width, height) = resolutions::TEST_SMALL;
    let frame = create_test_video_frame(width, height);

    let speed_presets = [10, 8, 6]; // Fast to medium

    for speed in speed_presets {
        let start = std::time::Instant::now();

        let packets = av1_encode_frames(width, height, vec![Frame::Video(frame.clone())], speed, 100)
            .unwrap_or_else(|e| panic!("AV1 encoding failed for speed {}: {:?}", speed, e));

        let elapsed = start.elapsed();

        assert!(
            !packets.is_empty(),
            "Should produce packets for speed {}",
            speed
        );

        println!("Speed {}: {} ms", speed, elapsed.as_millis());
    }
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn test_av1_many_frames() {
    let (width, height) = resolutions::QCIF;
    let num_frames = 30; // 1 second at 30fps

    let test_frames: Vec<Frame> = create_video_sequence(width, height, num_frames)
        .into_iter()
        .map(Frame::Video)
        .collect();

    let packets = av1_encode_frames(width, height, test_frames, 10, 120)
        .expect("AV1 encoding of many frames should succeed");

    assert!(
        packets.len() >= num_frames,
        "Should produce at least {} packets, got {}",
        num_frames,
        packets.len()
    );

    let decoded_frames = av1_decode_packets(packets).expect("AV1 decoding should succeed");

    assert_eq!(
        decoded_frames.len(),
        num_frames,
        "Should decode exactly {} frames",
        num_frames
    );
}

#[test]
fn test_encoder_flush_behavior() {
    use zvd_lib::codec::av1::Av1EncoderBuilder;

    let (width, height) = resolutions::TEST_SMALL;

    let mut encoder = Av1EncoderBuilder::new(width, height)
        .speed_preset(10)
        .quantizer(100)
        .build()
        .expect("Encoder creation should succeed");

    // Send one frame
    let frame = create_test_video_frame(width, height);
    encoder
        .send_frame(&Frame::Video(frame))
        .expect("Should accept frame");

    // Flush
    encoder.flush().expect("Flush should succeed");

    // Should eventually get packets
    let mut packet_count = 0;
    loop {
        match encoder.receive_packet() {
            Ok(_) => packet_count += 1,
            Err(Error::TryAgain) => continue,
            Err(Error::EndOfStream) => break,
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    assert!(packet_count > 0, "Should get at least one packet after flush");
}
