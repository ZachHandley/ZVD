//! Audio Decoder Integration Tests
//!
//! Tests for standalone packet-level audio decoding.
//!
//! These tests verify that FLAC, Vorbis, MP3, and AAC decoders
//! work correctly with the send_packet/receive_frame interface.

use zvd_lib::codec::{Decoder, Frame};
use zvd_lib::error::Error;
use zvd_lib::format::Packet;
use zvd_lib::util::{Buffer, Timestamp};

// ============================================================================
// Test Helpers
// ============================================================================

/// Create a test audio packet with the given data
fn create_audio_packet(data: Vec<u8>, pts: i64, duration: i64) -> Packet {
    let mut packet = Packet::new_audio(0, Buffer::from_vec(data));
    packet.pts = Timestamp::new(pts);
    packet.duration = duration;
    packet
}

// ============================================================================
// FLAC Decoder Tests
// ============================================================================

mod flac_tests {
    use super::*;
    use zvd_lib::codec::flac::{FlacDecoder, FlacStreamInfo};

    #[test]
    fn test_flac_decoder_creation_basic() {
        let decoder = FlacDecoder::new(44100, 2);
        assert!(decoder.is_ok(), "Failed to create FLAC decoder: {:?}", decoder.err());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.sample_rate(), 44100);
        assert_eq!(decoder.channels(), 2);
    }

    #[test]
    fn test_flac_decoder_various_sample_rates() {
        let sample_rates = [8000u32, 11025, 22050, 44100, 48000, 88200, 96000, 176400, 192000];
        for rate in sample_rates {
            let decoder = FlacDecoder::new(rate, 2);
            assert!(decoder.is_ok(), "Failed for sample rate {}", rate);
            assert_eq!(decoder.unwrap().sample_rate(), rate);
        }
    }

    #[test]
    fn test_flac_decoder_various_channel_counts() {
        // FLAC supports 1-8 channels
        for channels in 1..=8 {
            let decoder = FlacDecoder::new(44100, channels);
            assert!(decoder.is_ok(), "Failed for {} channels", channels);
            assert_eq!(decoder.unwrap().channels(), channels);
        }
    }

    #[test]
    fn test_flac_decoder_bit_depths() {
        let bit_depths = [16u8, 24, 32];
        for bits in bit_depths {
            let decoder = FlacDecoder::with_bits_per_sample(44100, 2, bits);
            assert!(decoder.is_ok(), "Failed for {} bits", bits);
            assert_eq!(decoder.unwrap().bits_per_sample(), bits);
        }
    }

    #[test]
    fn test_flac_decoder_empty_packet_rejected() {
        let mut decoder = FlacDecoder::new(44100, 2).unwrap();
        let empty_packet = Packet::new_audio(0, Buffer::empty());
        let result = decoder.send_packet(&empty_packet);
        assert!(result.is_err());
    }

    #[test]
    fn test_flac_decoder_receive_without_send() {
        let mut decoder = FlacDecoder::new(44100, 2).unwrap();
        let result = decoder.receive_frame();
        assert!(matches!(result, Err(Error::TryAgain)));
    }

    #[test]
    fn test_flac_decoder_flush() {
        let mut decoder = FlacDecoder::new(44100, 2).unwrap();
        assert!(decoder.flush().is_ok());
        // After flush, samples_decoded should reset
        assert_eq!(decoder.samples_decoded(), 0);
    }

    #[test]
    fn test_flac_stream_info_valid() {
        // Construct a valid FLAC STREAMINFO
        let mut streaminfo = [0u8; 34];
        // min_block_size = 4096
        streaminfo[0] = 0x10;
        streaminfo[1] = 0x00;
        // max_block_size = 4096
        streaminfo[2] = 0x10;
        streaminfo[3] = 0x00;
        // sample_rate=44100, channels=2, bits=16
        streaminfo[10] = 0x0A;
        streaminfo[11] = 0xC4;
        streaminfo[12] = 0x42;
        streaminfo[13] = 0xF0;

        let info = FlacStreamInfo::parse(&streaminfo);
        assert!(info.is_ok());
        let info = info.unwrap();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
        assert_eq!(info.bits_per_sample, 16);
    }

    #[test]
    fn test_flac_stream_info_too_short() {
        let short_data = [0u8; 10];
        let info = FlacStreamInfo::parse(&short_data);
        assert!(info.is_err());
    }

    #[test]
    fn test_flac_decoder_with_extradata() {
        // Valid STREAMINFO for 48kHz stereo 24-bit
        let mut streaminfo = [0u8; 34];
        streaminfo[0] = 0x10;
        streaminfo[1] = 0x00;
        streaminfo[2] = 0x10;
        streaminfo[3] = 0x00;
        // 48000 = 0xBB80
        // channels-1 = 1 (stereo)
        // bits-1 = 23 (24-bit)
        streaminfo[10] = 0x0B;
        streaminfo[11] = 0xB8;
        streaminfo[12] = 0x02; // 0x0 | 0x1<<1 = channels, last bit of sample rate
        streaminfo[13] = 0xF0; // bits = 23+1 = 24

        let decoder = FlacDecoder::with_extradata(48000, 2, &streaminfo);
        // Should use values from STREAMINFO
        if let Ok(dec) = decoder {
            // The extradata parsing may override initial parameters
            assert!(dec.sample_rate() > 0);
            assert!(dec.channels() > 0);
        }
    }

    #[test]
    fn test_flac_decoder_debug() {
        let decoder = FlacDecoder::new(44100, 2).unwrap();
        let debug_str = format!("{:?}", decoder);
        assert!(debug_str.contains("FlacDecoder"));
        assert!(debug_str.contains("sample_rate"));
    }
}

// ============================================================================
// Vorbis Decoder Tests
// ============================================================================

mod vorbis_tests {
    use super::*;
    use zvd_lib::codec::vorbis::{VorbisDecoder, VorbisIdHeader};

    #[test]
    fn test_vorbis_decoder_creation_basic() {
        let decoder = VorbisDecoder::new(44100, 2);
        assert!(decoder.is_ok(), "Failed to create Vorbis decoder: {:?}", decoder.err());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.sample_rate(), 44100);
        assert_eq!(decoder.channels(), 2);
    }

    #[test]
    fn test_vorbis_decoder_various_sample_rates() {
        let sample_rates = [8000u32, 11025, 22050, 44100, 48000, 96000, 192000];
        for rate in sample_rates {
            let decoder = VorbisDecoder::new(rate, 2);
            assert!(decoder.is_ok(), "Failed for sample rate {}", rate);
            assert_eq!(decoder.unwrap().sample_rate(), rate);
        }
    }

    #[test]
    fn test_vorbis_decoder_various_channel_counts() {
        // Vorbis supports up to 255 channels, but we test common configs
        for channels in [1u16, 2, 3, 4, 5, 6, 7, 8] {
            let decoder = VorbisDecoder::new(44100, channels);
            assert!(decoder.is_ok(), "Failed for {} channels", channels);
            assert_eq!(decoder.unwrap().channels(), channels);
        }
    }

    #[test]
    fn test_vorbis_decoder_empty_packet_rejected() {
        let mut decoder = VorbisDecoder::new(44100, 2).unwrap();
        let empty_packet = Packet::new_audio(0, Buffer::empty());
        let result = decoder.send_packet(&empty_packet);
        assert!(result.is_err());
    }

    #[test]
    fn test_vorbis_decoder_receive_without_send() {
        let mut decoder = VorbisDecoder::new(44100, 2).unwrap();
        let result = decoder.receive_frame();
        assert!(matches!(result, Err(Error::TryAgain)));
    }

    #[test]
    fn test_vorbis_decoder_flush() {
        let mut decoder = VorbisDecoder::new(44100, 2).unwrap();
        assert!(decoder.flush().is_ok());
        assert_eq!(decoder.samples_decoded(), 0);
    }

    #[test]
    fn test_vorbis_id_header_valid() {
        // Build a valid Vorbis identification header
        let mut header = vec![0x01]; // Packet type
        header.extend_from_slice(b"vorbis"); // Magic
        header.extend_from_slice(&0u32.to_le_bytes()); // Version
        header.push(2); // Channels
        header.extend_from_slice(&44100u32.to_le_bytes()); // Sample rate
        header.extend_from_slice(&0i32.to_le_bytes()); // Max bitrate
        header.extend_from_slice(&128000i32.to_le_bytes()); // Nominal bitrate
        header.extend_from_slice(&0i32.to_le_bytes()); // Min bitrate
        header.push(0x88); // blocksizes: 8 | (8 << 4)
        header.push(0x01); // Framing bit

        let id_header = VorbisIdHeader::parse(&header);
        assert!(id_header.is_ok());
        let id = id_header.unwrap();
        assert_eq!(id.vorbis_version, 0);
        assert_eq!(id.audio_channels, 2);
        assert_eq!(id.audio_sample_rate, 44100);
        assert_eq!(id.bitrate_nominal, 128000);
    }

    #[test]
    fn test_vorbis_id_header_invalid_magic() {
        let mut header = vec![0x01];
        header.extend_from_slice(b"vorbiX"); // Invalid magic
        header.extend_from_slice(&[0u8; 23]);
        let result = VorbisIdHeader::parse(&header);
        assert!(result.is_err());
    }

    #[test]
    fn test_vorbis_id_header_invalid_type() {
        let mut header = vec![0x02]; // Invalid type (not 0x01)
        header.extend_from_slice(b"vorbis");
        header.extend_from_slice(&[0u8; 23]);
        let result = VorbisIdHeader::parse(&header);
        assert!(result.is_err());
    }

    #[test]
    fn test_vorbis_is_header_packet() {
        // Test identification header detection
        let mut id_header = vec![0x01];
        id_header.extend_from_slice(b"vorbis");

        // Comment header
        let mut comment_header = vec![0x03];
        comment_header.extend_from_slice(b"vorbis");

        // Setup header
        let mut setup_header = vec![0x05];
        setup_header.extend_from_slice(b"vorbis");

        // Audio packet (not a header)
        let audio_packet = vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06];

        // These checks happen internally in the decoder
        // We just verify the decoder can handle them
        let mut decoder = VorbisDecoder::new(44100, 2).unwrap();

        // Sending header packets should succeed
        let pkt1 = create_audio_packet(id_header, 0, 0);
        assert!(decoder.send_packet(&pkt1).is_ok());
    }

    #[test]
    fn test_vorbis_decoder_debug() {
        let decoder = VorbisDecoder::new(44100, 2).unwrap();
        let debug_str = format!("{:?}", decoder);
        assert!(debug_str.contains("VorbisDecoder"));
        assert!(debug_str.contains("sample_rate"));
    }
}

// ============================================================================
// MP3 Decoder Tests
// ============================================================================

mod mp3_tests {
    use super::*;
    use zvd_lib::codec::mp3::{Mp3Decoder, Mp3FrameHeader, MpegVersion};

    #[test]
    fn test_mp3_decoder_creation_basic() {
        let decoder = Mp3Decoder::new(44100, 2);
        assert!(decoder.is_ok(), "Failed to create MP3 decoder: {:?}", decoder.err());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.sample_rate(), 44100);
        assert_eq!(decoder.channels(), 2);
    }

    #[test]
    fn test_mp3_decoder_various_sample_rates() {
        // MP3 supports specific sample rates
        let sample_rates = [
            8000u32, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000,
        ];
        for rate in sample_rates {
            let decoder = Mp3Decoder::new(rate, 2);
            assert!(decoder.is_ok(), "Failed for sample rate {}", rate);
            assert_eq!(decoder.unwrap().sample_rate(), rate);
        }
    }

    #[test]
    fn test_mp3_decoder_mono() {
        let decoder = Mp3Decoder::new(44100, 1);
        assert!(decoder.is_ok());
        assert_eq!(decoder.unwrap().channels(), 1);
    }

    #[test]
    fn test_mp3_decoder_stereo() {
        let decoder = Mp3Decoder::new(44100, 2);
        assert!(decoder.is_ok());
        assert_eq!(decoder.unwrap().channels(), 2);
    }

    #[test]
    fn test_mp3_decoder_empty_packet_rejected() {
        let mut decoder = Mp3Decoder::new(44100, 2).unwrap();
        let empty_packet = Packet::new_audio(0, Buffer::empty());
        let result = decoder.send_packet(&empty_packet);
        assert!(result.is_err());
    }

    #[test]
    fn test_mp3_decoder_receive_without_send() {
        let mut decoder = Mp3Decoder::new(44100, 2).unwrap();
        let result = decoder.receive_frame();
        assert!(matches!(result, Err(Error::TryAgain)));
    }

    #[test]
    fn test_mp3_decoder_flush() {
        let mut decoder = Mp3Decoder::new(44100, 2).unwrap();
        assert!(decoder.flush().is_ok());
        assert_eq!(decoder.samples_decoded(), 0);
        assert!(decoder.last_header().is_none());
    }

    #[test]
    fn test_mp3_frame_header_mpeg1_layer3() {
        // Valid MPEG-1 Layer III header
        // Sync: 0xFF 0xFB (MPEG-1, Layer III, no CRC)
        // 0x90 = bitrate index 9 (128kbps), sample rate 0 (44100Hz), no padding
        let header_data = [0xFF, 0xFB, 0x90, 0x00];
        let header = Mp3FrameHeader::parse(&header_data);
        assert!(header.is_ok());
        let header = header.unwrap();
        assert_eq!(header.version, MpegVersion::Mpeg1);
        assert_eq!(header.layer, 3);
        assert_eq!(header.bitrate, 128);
        assert_eq!(header.sample_rate, 44100);
        assert_eq!(header.channels, 2);
        assert_eq!(header.samples_per_frame, 1152);
    }

    #[test]
    fn test_mp3_frame_header_mpeg2() {
        // MPEG-2 Layer III header
        let header_data = [0xFF, 0xF3, 0x90, 0x00];
        let header = Mp3FrameHeader::parse(&header_data);
        assert!(header.is_ok());
        let header = header.unwrap();
        assert_eq!(header.version, MpegVersion::Mpeg2);
        assert_eq!(header.samples_per_frame, 576);
    }

    #[test]
    fn test_mp3_frame_header_mono() {
        // Mono channel mode (0xC0 in byte 3)
        let header_data = [0xFF, 0xFB, 0x90, 0xC0];
        let header = Mp3FrameHeader::parse(&header_data);
        assert!(header.is_ok());
        assert_eq!(header.unwrap().channels, 1);
    }

    #[test]
    fn test_mp3_frame_header_invalid_sync() {
        let header_data = [0x00, 0x00, 0x00, 0x00];
        let header = Mp3FrameHeader::parse(&header_data);
        assert!(header.is_err());
    }

    #[test]
    fn test_mp3_frame_header_too_short() {
        let header_data = [0xFF, 0xFB];
        let header = Mp3FrameHeader::parse(&header_data);
        assert!(header.is_err());
    }

    #[test]
    fn test_mpeg_version_from_header() {
        assert_eq!(MpegVersion::from_header(0b11), Some(MpegVersion::Mpeg1));
        assert_eq!(MpegVersion::from_header(0b10), Some(MpegVersion::Mpeg2));
        assert_eq!(MpegVersion::from_header(0b00), Some(MpegVersion::Mpeg25));
        assert_eq!(MpegVersion::from_header(0b01), None); // Reserved
    }

    #[test]
    fn test_mp3_is_layer3() {
        let header_data = [0xFF, 0xFB, 0x90, 0x00];
        let header = Mp3FrameHeader::parse(&header_data).unwrap();
        assert!(header.is_layer3());
    }

    #[test]
    fn test_mp3_decoder_debug() {
        let decoder = Mp3Decoder::new(44100, 2).unwrap();
        let debug_str = format!("{:?}", decoder);
        assert!(debug_str.contains("Mp3Decoder"));
        assert!(debug_str.contains("sample_rate"));
    }
}

// ============================================================================
// AAC Decoder Tests (requires aac feature)
// ============================================================================

#[cfg(feature = "aac")]
mod aac_tests {
    use super::*;
    use zvd_lib::codec::aac::{AacDecoder, AacProfile};

    #[test]
    fn test_aac_decoder_creation_basic() {
        let decoder = AacDecoder::new(44100, 2);
        assert!(decoder.is_ok(), "Failed to create AAC decoder: {:?}", decoder.err());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.sample_rate(), 44100);
        assert_eq!(decoder.channels(), 2);
    }

    #[test]
    fn test_aac_decoder_various_sample_rates() {
        let sample_rates = [8000u32, 16000, 22050, 32000, 44100, 48000, 96000];
        for rate in sample_rates {
            let decoder = AacDecoder::new(rate, 2);
            if let Ok(d) = decoder {
                assert_eq!(d.sample_rate(), rate);
            }
        }
    }

    #[test]
    fn test_aac_decoder_various_channel_counts() {
        for channels in [1u16, 2, 3, 4, 5, 6, 8] {
            let decoder = AacDecoder::new(44100, channels);
            if let Ok(d) = decoder {
                assert_eq!(d.channels(), channels);
            }
        }
    }

    #[test]
    fn test_aac_decoder_empty_packet_rejected() {
        let mut decoder = AacDecoder::new(44100, 2).unwrap();
        let empty_packet = Packet::new_audio(0, Buffer::empty());
        let result = decoder.send_packet(&empty_packet);
        assert!(result.is_err());
    }

    #[test]
    fn test_aac_decoder_receive_without_send() {
        let mut decoder = AacDecoder::new(44100, 2).unwrap();
        let result = decoder.receive_frame();
        assert!(matches!(result, Err(Error::TryAgain)));
    }

    #[test]
    fn test_aac_decoder_flush() {
        let mut decoder = AacDecoder::new(44100, 2).unwrap();
        assert!(decoder.flush().is_ok());
    }

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
    }

    #[test]
    fn test_aac_decoder_with_asc_lc() {
        // AAC-LC, 44100 Hz, stereo
        let asc = [0x12, 0x10];
        let decoder = AacDecoder::with_extradata(44100, 2, &asc);
        assert!(decoder.is_ok());
        let d = decoder.unwrap();
        assert_eq!(d.profile(), Some(AacProfile::LowComplexity));
    }

    #[test]
    fn test_aac_decoder_he_aac_rejected() {
        // HE-AAC v1 (SBR)
        let asc = [0x2A, 0x10];
        let decoder = AacDecoder::with_extradata(44100, 2, &asc);
        assert!(decoder.is_err());
        let err = decoder.unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }

    #[test]
    fn test_aac_decoder_debug() {
        let decoder = AacDecoder::new(44100, 2).unwrap();
        let debug_str = format!("{:?}", decoder);
        assert!(debug_str.contains("AacDecoder"));
        assert!(debug_str.contains("sample_rate"));
    }
}

// ============================================================================
// Cross-Decoder Tests
// ============================================================================

mod cross_decoder_tests {
    use super::*;
    use zvd_lib::codec::flac::FlacDecoder;
    use zvd_lib::codec::mp3::Mp3Decoder;
    use zvd_lib::codec::vorbis::VorbisDecoder;

    #[test]
    fn test_all_decoders_implement_decoder_trait() {
        let mut flac: Box<dyn Decoder> = Box::new(FlacDecoder::new(44100, 2).unwrap());
        let mut vorbis: Box<dyn Decoder> = Box::new(VorbisDecoder::new(44100, 2).unwrap());
        let mut mp3: Box<dyn Decoder> = Box::new(Mp3Decoder::new(44100, 2).unwrap());

        // All should return TryAgain when no packets sent
        assert!(matches!(flac.receive_frame(), Err(Error::TryAgain)));
        assert!(matches!(vorbis.receive_frame(), Err(Error::TryAgain)));
        assert!(matches!(mp3.receive_frame(), Err(Error::TryAgain)));

        // All should flush successfully
        assert!(flac.flush().is_ok());
        assert!(vorbis.flush().is_ok());
        assert!(mp3.flush().is_ok());
    }

    #[test]
    fn test_all_decoders_reject_empty_packets() {
        let empty = Packet::new_audio(0, Buffer::empty());

        let mut flac = FlacDecoder::new(44100, 2).unwrap();
        let mut vorbis = VorbisDecoder::new(44100, 2).unwrap();
        let mut mp3 = Mp3Decoder::new(44100, 2).unwrap();

        assert!(flac.send_packet(&empty).is_err());
        assert!(vorbis.send_packet(&empty).is_err());
        assert!(mp3.send_packet(&empty).is_err());
    }

    #[test]
    fn test_factory_functions() {
        use zvd_lib::codec::flac;
        use zvd_lib::codec::mp3;
        use zvd_lib::codec::vorbis;

        assert!(flac::create_decoder(44100, 2).is_ok());
        assert!(vorbis::create_decoder(44100, 2).is_ok());
        assert!(mp3::create_decoder(44100, 2).is_ok());
    }

    #[test]
    #[cfg(feature = "aac")]
    fn test_aac_factory_function() {
        use zvd_lib::codec::aac;
        assert!(aac::create_decoder(44100, 2).is_ok());
    }
}

// ============================================================================
// Decoder Workflow Tests
// ============================================================================

mod workflow_tests {
    use super::*;
    use zvd_lib::codec::flac::FlacDecoder;
    use zvd_lib::codec::mp3::Mp3Decoder;
    use zvd_lib::codec::vorbis::VorbisDecoder;

    #[test]
    fn test_multiple_flush_calls() {
        let mut flac = FlacDecoder::new(44100, 2).unwrap();

        // Multiple flushes should be safe
        assert!(flac.flush().is_ok());
        assert!(flac.flush().is_ok());
        assert!(flac.flush().is_ok());
    }

    #[test]
    fn test_send_after_flush() {
        let mut mp3 = Mp3Decoder::new(44100, 2).unwrap();

        // Flush
        assert!(mp3.flush().is_ok());

        // Should be able to send packets after flush
        let data = vec![0xFF, 0xFB, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00];
        let packet = create_audio_packet(data, 0, 1152);
        assert!(mp3.send_packet(&packet).is_ok());
    }

    #[test]
    fn test_receive_after_flush() {
        let mut vorbis = VorbisDecoder::new(44100, 2).unwrap();

        // Send a packet
        let data = vec![0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let packet = create_audio_packet(data, 0, 1024);
        let _ = vorbis.send_packet(&packet);

        // Flush clears pending packets
        assert!(vorbis.flush().is_ok());

        // Receive should return TryAgain (queue is empty)
        assert!(matches!(vorbis.receive_frame(), Err(Error::TryAgain)));
    }

    #[test]
    fn test_decoder_state_after_invalid_packet() {
        let mut flac = FlacDecoder::new(44100, 2).unwrap();

        // Send an invalid packet (random data, not valid FLAC)
        let invalid_data = vec![0x00, 0x01, 0x02, 0x03];
        let packet = create_audio_packet(invalid_data, 0, 1024);
        let _ = flac.send_packet(&packet);

        // Receive will fail but decoder should recover
        let _ = flac.receive_frame();

        // Should still be able to flush
        assert!(flac.flush().is_ok());

        // And receive should return TryAgain
        assert!(matches!(flac.receive_frame(), Err(Error::TryAgain)));
    }
}
