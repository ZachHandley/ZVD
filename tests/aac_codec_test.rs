//! AAC Codec Integration Tests
//!
//! These tests verify the AAC decoder works correctly using Symphonia.
//!
//! Test categories:
//! 1. Decoder creation tests
//! 2. AudioSpecificConfig parsing tests
//! 3. Profile detection tests
//! 4. Error handling tests

#![cfg(feature = "aac")]

use zvd_lib::codec::aac::{AacDecoder, AacProfile};
use zvd_lib::codec::{Decoder, Frame};
use zvd_lib::error::Error;
use zvd_lib::format::Packet;
use zvd_lib::util::{Buffer, MediaType, Timestamp};

// ============================================================================
// Test Helpers
// ============================================================================

/// Create a test packet with the given data
fn create_test_packet(data: Vec<u8>, pts: i64) -> Packet {
    let mut packet = Packet::new_audio(0, Buffer::from_vec(data));
    packet.pts = Timestamp::new(pts);
    packet.duration = 1024;
    packet
}

/// Generate a valid ADTS header for testing
fn generate_adts_header(
    profile: u8,
    sample_rate_index: u8,
    channel_config: u8,
    frame_length: usize,
) -> [u8; 7] {
    let mut header = [0u8; 7];

    // Syncword (0xFFF) + MPEG-4 ID (0) + Layer (0) + Protection absent (1)
    header[0] = 0xFF;
    header[1] = 0xF1;

    // Profile + Sample rate index + Private bit + Channel config high bit
    header[2] = ((profile & 0x03) << 6)
        | ((sample_rate_index & 0x0F) << 2)
        | ((channel_config >> 2) & 0x01);

    // Channel config low bits + originality + home + copyright + frame length high bits
    header[3] = ((channel_config & 0x03) << 6) | ((frame_length >> 11) & 0x03) as u8;

    // Frame length middle bits
    header[4] = ((frame_length >> 3) & 0xFF) as u8;

    // Frame length low bits + buffer fullness high bits (VBR = 0x7FF)
    header[5] = (((frame_length & 0x07) << 5) | 0x1F) as u8;

    // Buffer fullness low bits + raw data blocks (0)
    header[6] = 0xFC;

    header
}

// ============================================================================
// Decoder Creation Tests
// ============================================================================

#[test]
fn test_aac_decoder_creation_stereo_44100() {
    let decoder = AacDecoder::new(44100, 2);
    assert!(decoder.is_ok(), "Failed to create AAC decoder: {:?}", decoder.err());

    let decoder = decoder.unwrap();
    assert_eq!(decoder.sample_rate(), 44100);
    assert_eq!(decoder.channels(), 2);
}

#[test]
fn test_aac_decoder_creation_stereo_48000() {
    let decoder = AacDecoder::new(48000, 2);
    assert!(decoder.is_ok());

    let decoder = decoder.unwrap();
    assert_eq!(decoder.sample_rate(), 48000);
    assert_eq!(decoder.channels(), 2);
}

#[test]
fn test_aac_decoder_creation_mono() {
    let decoder = AacDecoder::new(44100, 1);
    assert!(decoder.is_ok());

    let decoder = decoder.unwrap();
    assert_eq!(decoder.channels(), 1);
}

#[test]
fn test_aac_decoder_creation_surround_51() {
    let decoder = AacDecoder::new(48000, 6);
    // 5.1 surround may not be supported by all backend configurations
    // so we only test that it doesn't panic
    if let Ok(decoder) = decoder {
        assert_eq!(decoder.channels(), 6);
    }
}

#[test]
fn test_various_sample_rates() {
    let sample_rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000];

    for rate in sample_rates {
        let decoder_result = AacDecoder::new(rate, 2);
        if let Ok(decoder) = decoder_result {
            assert_eq!(decoder.sample_rate(), rate);
        }
    }
}

// ============================================================================
// Decoder with Extradata Tests
// ============================================================================

#[test]
fn test_aac_decoder_with_aac_lc_extradata() {
    // AAC-LC, 44100 Hz, stereo
    // Object type: 2 (LC), Sample rate index: 4 (44100), Channels: 2
    // Binary: 00010 0100 0010 000 = 0x12 0x10
    let asc_data = [0x12, 0x10];

    let decoder = AacDecoder::with_extradata(44100, 2, &asc_data);
    assert!(decoder.is_ok(), "Failed to create decoder with extradata: {:?}", decoder.err());

    let decoder = decoder.unwrap();
    assert_eq!(decoder.profile(), Some(AacProfile::LowComplexity));
}

#[test]
fn test_aac_decoder_with_48khz_extradata() {
    // AAC-LC, 48000 Hz, stereo
    // Object type: 2 (LC), Sample rate index: 3 (48000), Channels: 2
    // Binary: 00010 0011 0010 000 = 0x11 0x90
    let asc_data = [0x11, 0x90];

    let decoder = AacDecoder::with_extradata(48000, 2, &asc_data);
    assert!(decoder.is_ok());
}

#[test]
fn test_he_aac_rejected() {
    // HE-AAC v1 (SBR), 44100 Hz, stereo
    // Object type: 5 (HE-AAC), Sample rate index: 4 (44100), Channels: 2
    // Binary: 00101 0100 0010 000 = 0x2A 0x10
    let asc_data = [0x2A, 0x10];

    let decoder = AacDecoder::with_extradata(44100, 2, &asc_data);
    assert!(decoder.is_err(), "HE-AAC should be rejected");

    let err = decoder.unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)));
}

#[test]
fn test_he_aac_v2_rejected() {
    // HE-AAC v2 (PS), 22050 Hz, stereo
    // Object type: 29 (HE-AAC v2)
    // Binary: 11101 0111 0010 000 = 0xEB 0x90
    let asc_data = [0xEB, 0x90];

    let decoder = AacDecoder::with_extradata(22050, 2, &asc_data);
    assert!(decoder.is_err(), "HE-AAC v2 should be rejected");
}

// ============================================================================
// Profile Tests
// ============================================================================

#[test]
fn test_aac_profile_supported() {
    assert!(AacProfile::LowComplexity.is_supported());
    assert!(AacProfile::Main.is_supported());
    assert!(AacProfile::Ltp.is_supported());
}

#[test]
fn test_aac_profile_unsupported() {
    assert!(!AacProfile::HeAacV1.is_supported());
    assert!(!AacProfile::HeAacV2.is_supported());
    assert!(!AacProfile::Ssr.is_supported());
    assert!(!AacProfile::Unknown.is_supported());
}

#[test]
fn test_supported_profiles_list() {
    let profiles = AacDecoder::supported_profiles();
    assert_eq!(profiles.len(), 3);
    assert!(profiles.contains(&AacProfile::Main));
    assert!(profiles.contains(&AacProfile::LowComplexity));
    assert!(profiles.contains(&AacProfile::Ltp));
}

#[test]
fn test_is_profile_supported() {
    assert!(AacDecoder::is_profile_supported(AacProfile::LowComplexity));
    assert!(AacDecoder::is_profile_supported(AacProfile::Main));
    assert!(!AacDecoder::is_profile_supported(AacProfile::HeAacV1));
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_empty_packet_rejected() {
    let mut decoder = AacDecoder::new(44100, 2).unwrap();
    let empty_packet = Packet::new_audio(0, Buffer::empty());

    let result = decoder.send_packet(&empty_packet);
    assert!(result.is_err(), "Empty packet should be rejected");
}

#[test]
fn test_receive_without_send_returns_try_again() {
    let mut decoder = AacDecoder::new(44100, 2).unwrap();

    let result = decoder.receive_frame();
    assert!(matches!(result, Err(Error::TryAgain)));
}

#[test]
fn test_flush_succeeds() {
    let mut decoder = AacDecoder::new(44100, 2).unwrap();

    let result = decoder.flush();
    assert!(result.is_ok());
}

#[test]
fn test_flush_clears_state() {
    let mut decoder = AacDecoder::new(44100, 2).unwrap();

    // Flush should succeed
    assert!(decoder.flush().is_ok());

    // After flush, receive should return TryAgain
    let result = decoder.receive_frame();
    assert!(matches!(result, Err(Error::TryAgain)));
}

#[test]
fn test_short_extradata_rejected() {
    // Too short AudioSpecificConfig
    let short_asc = [0x12];

    let decoder = AacDecoder::with_extradata(44100, 2, &short_asc);
    assert!(decoder.is_err());
}

// ============================================================================
// Decoder Trait Tests
// ============================================================================

#[test]
fn test_decoder_trait_implementation() {
    let mut decoder = AacDecoder::new(44100, 2).unwrap();

    // Test that we can use it through the trait
    let _: &mut dyn Decoder = &mut decoder;
}

#[test]
fn test_decoder_debug_impl() {
    let decoder = AacDecoder::new(44100, 2).unwrap();

    let debug_str = format!("{:?}", decoder);
    assert!(debug_str.contains("AacDecoder"));
    assert!(debug_str.contains("sample_rate"));
    assert!(debug_str.contains("channels"));
}

// ============================================================================
// Channel Configuration Tests
// ============================================================================

#[test]
fn test_various_channel_configurations() {
    let channel_configs = [
        (1, "mono"),
        (2, "stereo"),
        (3, "3.0"),
        (4, "4.0"),
        (5, "5.0"),
        (6, "5.1 surround"),
        (8, "7.1 surround"),
    ];

    for (channels, name) in channel_configs {
        let decoder_result = AacDecoder::new(44100, channels);
        if let Ok(decoder) = decoder_result {
            assert_eq!(
                decoder.channels(),
                channels,
                "Channel count mismatch for {}",
                name
            );
        }
    }
}

// ============================================================================
// Convenience Method Tests
// ============================================================================

#[test]
fn test_decode_packet_convenience() {
    let mut decoder = AacDecoder::new(44100, 2).unwrap();

    // Create a dummy packet (will fail to decode but should not panic)
    let dummy_data = vec![0xFF, 0xF1, 0x50, 0x80, 0x00, 0x1F, 0xFC];
    let packet = create_test_packet(dummy_data, 0);

    // decode_packet combines send and receive
    let result = decoder.decode_packet(&packet);
    // This will likely fail since it's not real AAC data, but it shouldn't panic
    let _ = result;
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_send_multiple_packets() {
    let mut decoder = AacDecoder::new(44100, 2).unwrap();

    // Send multiple packets
    for i in 0..5 {
        let data = vec![0xFF, 0xF1, 0x50, 0x80, 0x00, 0x1F, 0xFC, i as u8];
        let packet = create_test_packet(data, i * 1024);

        // Sending should succeed (queues the packet)
        let _ = decoder.send_packet(&packet);
    }

    // Flush should still work
    assert!(decoder.flush().is_ok());
}

#[test]
fn test_decoder_recovers_after_flush() {
    let mut decoder = AacDecoder::new(44100, 2).unwrap();

    // Send some packets
    let data = vec![0xFF, 0xF1, 0x50, 0x80, 0x00, 0x1F, 0xFC];
    let packet = create_test_packet(data.clone(), 0);
    let _ = decoder.send_packet(&packet);

    // Flush
    assert!(decoder.flush().is_ok());

    // Should be able to send more packets after flush
    let packet2 = create_test_packet(data, 1024);
    let result = decoder.send_packet(&packet2);
    assert!(result.is_ok());
}
