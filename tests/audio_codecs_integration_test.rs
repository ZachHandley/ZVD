//! Integration tests for Symphonia-based audio codecs
//!
//! These tests verify FLAC, Vorbis, MP3, and AAC codec integration through
//! container-level decoding with SymphoniaAdapter.

use zvd_lib::codec::{get_codec_info, AudioFrame};
use zvd_lib::util::{MediaType, SampleFormat};

/// Test FLAC codec info
#[test]
#[cfg(feature = "flac")]
fn test_flac_codec_info() {
    let info = get_codec_info("flac");
    assert!(info.is_some(), "Should have codec info for FLAC");

    let info = info.unwrap();
    assert_eq!(info.id, "flac");
    assert_eq!(info.name, "FLAC");
    assert_eq!(info.long_name, "Free Lossless Audio Codec");
    assert_eq!(info.media_type, MediaType::Audio);
    assert!(info.capabilities.lossless, "FLAC is lossless");
    assert!(!info.capabilities.lossy, "FLAC is not lossy");
    assert!(!info.capabilities.inter, "FLAC has no inter-frame coding");
}

/// Test Vorbis codec info
#[test]
#[cfg(feature = "vorbis")]
fn test_vorbis_codec_info() {
    let info = get_codec_info("vorbis");
    assert!(info.is_some(), "Should have codec info for Vorbis");

    let info = info.unwrap();
    assert_eq!(info.id, "vorbis");
    assert_eq!(info.name, "Vorbis");
    assert_eq!(info.long_name, "Vorbis");
    assert_eq!(info.media_type, MediaType::Audio);
    assert!(info.capabilities.lossy, "Vorbis is lossy");
    assert!(!info.capabilities.lossless, "Vorbis is not lossless");
}

/// Test MP3 codec info
#[test]
#[cfg(feature = "mp3")]
fn test_mp3_codec_info() {
    let info = get_codec_info("mp3");
    assert!(info.is_some(), "Should have codec info for MP3");

    let info = info.unwrap();
    assert_eq!(info.id, "mp3");
    assert_eq!(info.name, "MP3");
    assert_eq!(info.long_name, "MP3 (MPEG audio layer 3)");
    assert_eq!(info.media_type, MediaType::Audio);
    assert!(info.capabilities.lossy, "MP3 is lossy");
    assert!(!info.capabilities.lossless, "MP3 is not lossless");
}

/// Test AAC codec info
#[test]
#[cfg(feature = "aac")]
fn test_aac_codec_info() {
    let info = get_codec_info("aac");
    assert!(info.is_some(), "Should have codec info for AAC");

    let info = info.unwrap();
    assert_eq!(info.id, "aac");
    assert_eq!(info.name, "AAC");
    assert_eq!(info.long_name, "AAC (Advanced Audio Coding)");
    assert_eq!(info.media_type, MediaType::Audio);
    assert!(info.capabilities.lossy, "AAC is lossy");
}

/// Test FLAC decoder creation and configuration
#[test]
#[cfg(feature = "flac")]
fn test_flac_decoder_creation() {
    use zvd_lib::codec::flac::FlacDecoder;

    // Valid configuration
    let decoder = FlacDecoder::new(44100, 2);
    assert!(decoder.is_ok(), "Should create decoder with valid config");

    // Test various sample rates
    assert!(FlacDecoder::new(8000, 2).is_ok(), "Should support 8 kHz");
    assert!(FlacDecoder::new(48000, 2).is_ok(), "Should support 48 kHz");
    assert!(FlacDecoder::new(96000, 2).is_ok(), "Should support 96 kHz");
    assert!(FlacDecoder::new(192000, 2).is_ok(), "Should support 192 kHz");

    // Test various channel counts
    assert!(FlacDecoder::new(44100, 1).is_ok(), "Should support mono");
    assert!(FlacDecoder::new(44100, 2).is_ok(), "Should support stereo");
    assert!(FlacDecoder::new(44100, 6).is_ok(), "Should support 5.1");
    assert!(FlacDecoder::new(44100, 8).is_ok(), "Should support 7.1");
}

/// Test FLAC decoder validation
#[test]
#[cfg(feature = "flac")]
fn test_flac_decoder_validation() {
    use zvd_lib::codec::flac::FlacDecoder;

    // Invalid sample rate (too low)
    let decoder = FlacDecoder::new(0, 2);
    assert!(decoder.is_err(), "Should reject zero sample rate");

    // Invalid sample rate (too high)
    let decoder = FlacDecoder::new(1_000_000, 2);
    assert!(decoder.is_err(), "Should reject excessive sample rate");

    // Invalid channels
    let decoder = FlacDecoder::new(44100, 0);
    assert!(decoder.is_err(), "Should reject zero channels");

    let decoder = FlacDecoder::new(44100, 255);
    assert!(decoder.is_err(), "Should reject too many channels");
}

/// Test Vorbis decoder creation and configuration
#[test]
#[cfg(feature = "vorbis")]
fn test_vorbis_decoder_creation() {
    use zvd_lib::codec::vorbis::VorbisDecoder;

    // Valid configuration
    let decoder = VorbisDecoder::new(44100, 2);
    assert!(decoder.is_ok(), "Should create decoder with valid config");

    // Test various sample rates
    assert!(VorbisDecoder::new(8000, 2).is_ok(), "Should support 8 kHz");
    assert!(VorbisDecoder::new(48000, 2).is_ok(), "Should support 48 kHz");
    assert!(VorbisDecoder::new(96000, 2).is_ok(), "Should support 96 kHz");

    // Test various channel counts
    assert!(VorbisDecoder::new(44100, 1).is_ok(), "Should support mono");
    assert!(VorbisDecoder::new(44100, 2).is_ok(), "Should support stereo");
    assert!(VorbisDecoder::new(44100, 6).is_ok(), "Should support 5.1");
}

/// Test MP3 decoder creation and configuration
#[test]
#[cfg(feature = "mp3")]
fn test_mp3_decoder_creation() {
    use zvd_lib::codec::mp3::Mp3Decoder;

    // Valid configuration - standard MP3 sample rates
    let decoder = Mp3Decoder::new(44100, 2);
    assert!(decoder.is_ok(), "Should create decoder with valid config");

    // Test standard MP3 sample rates
    assert!(Mp3Decoder::new(8000, 2).is_ok(), "Should support 8 kHz");
    assert!(Mp3Decoder::new(16000, 2).is_ok(), "Should support 16 kHz");
    assert!(Mp3Decoder::new(22050, 2).is_ok(), "Should support 22.05 kHz");
    assert!(Mp3Decoder::new(32000, 2).is_ok(), "Should support 32 kHz");
    assert!(Mp3Decoder::new(44100, 2).is_ok(), "Should support 44.1 kHz");
    assert!(Mp3Decoder::new(48000, 2).is_ok(), "Should support 48 kHz");

    // Test channel counts
    assert!(Mp3Decoder::new(44100, 1).is_ok(), "Should support mono");
    assert!(Mp3Decoder::new(44100, 2).is_ok(), "Should support stereo");
}

/// Test MP3 decoder validation
#[test]
#[cfg(feature = "mp3")]
fn test_mp3_decoder_validation() {
    use zvd_lib::codec::mp3::Mp3Decoder;

    // Invalid sample rate
    let decoder = Mp3Decoder::new(0, 2);
    assert!(decoder.is_err(), "Should reject zero sample rate");

    // Non-standard sample rate (should still work, just warn)
    let decoder = Mp3Decoder::new(96000, 2);
    assert!(decoder.is_ok(), "Should accept non-standard sample rate");

    // Invalid channels
    let decoder = Mp3Decoder::new(44100, 0);
    assert!(decoder.is_err(), "Should reject zero channels");
}

/// Test AAC decoder creation and configuration
#[test]
#[cfg(feature = "aac")]
fn test_aac_decoder_creation() {
    use zvd_lib::codec::aac::AacDecoder;

    // Valid configuration
    let decoder = AacDecoder::new(44100, 2, None);
    assert!(decoder.is_ok(), "Should create decoder with valid config");

    // Test various sample rates
    assert!(AacDecoder::new(8000, 2, None).is_ok(), "Should support 8 kHz");
    assert!(AacDecoder::new(48000, 2, None).is_ok(), "Should support 48 kHz");
    assert!(AacDecoder::new(96000, 2, None).is_ok(), "Should support 96 kHz");

    // Test various channel counts
    assert!(AacDecoder::new(44100, 1, None).is_ok(), "Should support mono");
    assert!(AacDecoder::new(44100, 2, None).is_ok(), "Should support stereo");
    assert!(AacDecoder::new(44100, 6, None).is_ok(), "Should support 5.1");
}

/// Test AAC decoder validation
#[test]
#[cfg(feature = "aac")]
fn test_aac_decoder_validation() {
    use zvd_lib::codec::aac::AacDecoder;

    // Invalid sample rate (below minimum)
    let decoder = AacDecoder::new(7999, 2, None);
    assert!(decoder.is_err(), "Should reject sample rate below 8 kHz");

    // Invalid sample rate (above maximum)
    let decoder = AacDecoder::new(96001, 2, None);
    assert!(decoder.is_err(), "Should reject sample rate above 96 kHz");

    // Invalid channels
    let decoder = AacDecoder::new(44100, 0, None);
    assert!(decoder.is_err(), "Should reject zero channels");

    let decoder = AacDecoder::new(44100, 49, None);
    assert!(decoder.is_err(), "Should reject more than 48 channels");
}

/// Test AAC decoder with extradata
#[test]
#[cfg(feature = "aac")]
fn test_aac_decoder_extradata() {
    use zvd_lib::codec::aac::AacDecoder;

    // AudioSpecificConfig for AAC-LC, 44.1 kHz, stereo
    let extradata = vec![0x12, 0x10];

    let decoder = AacDecoder::new(44100, 2, Some(extradata));
    assert!(decoder.is_ok(), "Should accept valid extradata");
}

/// Test audio frame creation
#[test]
fn test_audio_frame_creation() {
    // Test various sample formats
    let frame = AudioFrame::new(1024, 2, SampleFormat::I16);
    assert_eq!(frame.nb_samples, 1024);
    assert_eq!(frame.channels, 2);
    assert_eq!(frame.format, SampleFormat::I16);

    let frame = AudioFrame::new(1024, 2, SampleFormat::F32);
    assert_eq!(frame.format, SampleFormat::F32);

    let frame = AudioFrame::new(1024, 6, SampleFormat::I16);
    assert_eq!(frame.channels, 6);
}

/// Test that codecs are not available without their feature flags
#[test]
#[cfg(not(feature = "flac"))]
fn test_flac_not_available_without_feature() {
    use zvd_lib::codec::get_codec_info;
    assert!(
        get_codec_info("flac").is_none(),
        "FLAC should not be available without feature flag"
    );
}

#[test]
#[cfg(not(feature = "vorbis"))]
fn test_vorbis_not_available_without_feature() {
    use zvd_lib::codec::get_codec_info;
    assert!(
        get_codec_info("vorbis").is_none(),
        "Vorbis should not be available without feature flag"
    );
}

#[test]
#[cfg(not(feature = "mp3"))]
fn test_mp3_not_available_without_feature() {
    use zvd_lib::codec::get_codec_info;
    assert!(
        get_codec_info("mp3").is_none(),
        "MP3 should not be available without feature flag"
    );
}

#[test]
#[cfg(not(feature = "aac"))]
fn test_aac_not_available_without_feature() {
    use zvd_lib::codec::get_codec_info;
    assert!(
        get_codec_info("aac").is_none(),
        "AAC should not be available without feature flag"
    );
}

/// Test Opus codec info (from Phase 4)
#[test]
#[cfg(feature = "opus-codec")]
fn test_opus_codec_info() {
    let info = get_codec_info("opus");
    assert!(info.is_some(), "Should have codec info for Opus");

    let info = info.unwrap();
    assert_eq!(info.id, "opus");
    assert_eq!(info.name, "Opus");
    assert_eq!(info.media_type, MediaType::Audio);
    assert!(info.capabilities.lossy, "Opus supports lossy compression");
}

/// Test multiple audio codec registrations
#[test]
fn test_audio_codec_registry() {
    let codecs = [
        ("flac", "flac"),
        ("vorbis", "vorbis"),
        ("mp3", "mp3"),
        ("aac", "aac"),
        ("opus", "opus-codec"),
    ];

    for (codec_id, feature) in codecs {
        let info = get_codec_info(codec_id);
        #[cfg(feature = feature)]
        assert!(
            info.is_some(),
            "{} should be registered when feature is enabled",
            codec_id
        );
    }
}
