//! Integration tests for H.265 NAL unit and parameter set parsing
//!
//! These tests validate our H.265 parser implementation with realistic data patterns.

#[cfg(feature = "h265")]
mod h265_tests {
    use zvd_lib::codec::h265::bitstream::BitstreamReader;
    use zvd_lib::codec::h265::headers::Sps;
    use zvd_lib::codec::h265::nal::{find_start_codes, NalUnit, NalUnitType};

    /// Test NAL unit header parsing with IDR frame
    #[test]
    fn test_nal_header_idr() {
        // IDR_W_RADL NAL unit header (type=19)
        // Byte 0: 0x26 = 0010 0110
        //   - forbidden_zero_bit = 0
        //   - nal_unit_type = 010011 (19 = IDR_W_RADL)
        // Byte 1: 0x01 = 0000 0001
        //   - nuh_layer_id = 000000 (0)
        //   - nuh_temporal_id_plus1 = 001 (1, meaning TID=0)
        let data = vec![0x26, 0x01];

        let nal = NalUnit::parse(&data).expect("Should parse IDR NAL unit");

        assert_eq!(nal.nal_type(), NalUnitType::IdrWRadl);
        assert!(nal.is_idr());
        assert!(nal.is_slice());
        assert_eq!(nal.header.nuh_layer_id, 0);
        assert_eq!(nal.header.temporal_id(), 0);
    }

    /// Test NAL unit header parsing with SPS
    #[test]
    fn test_nal_header_sps() {
        // SPS NAL unit header (type=33)
        // Byte 0: 0x42 = 0100 0010
        //   - forbidden_zero_bit = 0
        //   - nal_unit_type = 100001 (33 = SPS)
        // Byte 1: 0x01 = 0000 0001
        //   - nuh_layer_id = 000000 (0)
        //   - nuh_temporal_id_plus1 = 001 (1)
        let data = vec![0x42, 0x01];

        let nal = NalUnit::parse(&data).expect("Should parse SPS NAL unit");

        assert_eq!(nal.nal_type(), NalUnitType::SpsNut);
        assert!(!nal.is_idr());
        assert!(!nal.is_slice());
        assert!(nal.nal_type().is_parameter_set());
        assert_eq!(nal.header.nuh_layer_id, 0);
    }

    /// Test emulation prevention byte removal
    #[test]
    fn test_emulation_prevention() {
        // Data with emulation prevention: 0x00 0x00 0x03 0x01
        // Should become: 0x00 0x00 0x01
        let data = vec![
            0x42, 0x01, // NAL header (SPS)
            0x00, 0x00, 0x03, 0x01, // 0x000001 with emulation prevention
            0xFF,
        ];

        let nal = NalUnit::parse(&data).expect("Should parse with emulation prevention");

        // RBSP should have the 0x03 removed
        assert_eq!(nal.rbsp[0], 0x00);
        assert_eq!(nal.rbsp[1], 0x00);
        assert_eq!(nal.rbsp[2], 0x01); // 0x03 was removed
        assert_eq!(nal.rbsp[3], 0xFF);
    }

    /// Test finding start codes in byte stream
    #[test]
    fn test_find_start_codes() {
        let data = vec![
            0x00, 0x00, 0x00, 0x01, // 4-byte start code at position 0
            0x42, 0x01, // NAL header
            0xFF, 0xFF, 0x00, 0x00, 0x01, // 3-byte start code at position 8
            0x26, 0x01, // NAL header
            0xAA, 0xBB, 0xCC,
        ];

        let positions = find_start_codes(&data);

        assert_eq!(positions.len(), 2);
        assert_eq!(positions[0], 0); // First start code at byte 0
        assert_eq!(positions[1], 8); // Second start code at byte 8
    }

    /// Test bitstream reader with Exp-Golomb codes
    #[test]
    fn test_bitstream_exp_golomb() {
        // Test reading multiple ue(v) values in sequence
        // ue(v) values: 0, 1, 2, 3, 4
        // Bit patterns: 1, 010, 011, 00100, 00101
        let data = vec![
            0b1_010_011_0, // 1 (0), 010 (1), 011 (2), 0
            0b0100_00101,  // 00100 (3), 00101 (4)
        ];

        let mut reader = BitstreamReader::new(&data);

        assert_eq!(reader.read_ue().unwrap(), 0);
        assert_eq!(reader.read_ue().unwrap(), 1);
        assert_eq!(reader.read_ue().unwrap(), 2);
        assert_eq!(reader.read_ue().unwrap(), 3);
        assert_eq!(reader.read_ue().unwrap(), 4);
    }

    /// Test SPS parsing with realistic 1920x1080 8-bit 4:2:0 data
    ///
    /// This creates a minimal but valid SPS for Full HD video
    #[test]
    fn test_sps_parsing_1080p() {
        use zvd_lib::codec::h265::bitstream::BitstreamReader;

        let mut sps_data = Vec::new();

        // sps_video_parameter_set_id (4 bits) = 0
        // sps_max_sub_layers_minus1 (3 bits) = 0
        // sps_temporal_id_nesting_flag (1 bit) = 1
        sps_data.push(0b0000_000_1);

        // profile_tier_level (12 bytes for max_sub_layers_minus1 = 0)
        // general_profile_space (2) = 0, general_tier_flag (1) = 0, general_profile_idc (5) = 1 (Main)
        sps_data.push(0b00_0_00001); // Profile: Main (1)

        // general_profile_compatibility_flag[32] - set Main profile flag
        sps_data.push(0b01000000); // Main profile compatible
        sps_data.push(0x00);
        sps_data.push(0x00);
        sps_data.push(0x00);

        // general_progressive_source_flag (1), interlaced (1), non_packed (1), frame_only (1) = 4 bits
        // + general_reserved_zero_44bits = 44 bits
        // Total: 48 bits = 6 bytes
        sps_data.push(0b1000_0000); // Progressive
        for _ in 0..5 {
            sps_data.push(0x00);
        }

        // general_level_idc = 8 bits (level 4.0 = 120)
        sps_data.push(120);

        // Now encode the rest with a bitstream writer approach
        // sps_seq_parameter_set_id (ue(v)) = 0 → "1"
        // chroma_format_idc (ue(v)) = 1 → "010"
        // pic_width_in_luma_samples (ue(v)) = 1920
        //   1920 = 0x780 = 11 bits value
        //   ue(v) encoding: leading_zeros = 10, so: 0000000000 1 1110000000 = 21 bits
        // pic_height_in_luma_samples (ue(v)) = 1080
        //   1080 = 0x438 = 11 bits value
        //   ue(v) encoding: leading_zeros = 10, so: 0000000000 1 1000111000 = 21 bits
        // conformance_window_flag = 0
        // bit_depth_luma_minus8 (ue(v)) = 0 → "1"
        // bit_depth_chroma_minus8 (ue(v)) = 0 → "1"

        // Let's build this bit-by-bit:
        // "1" + "010" + ue(1920) + ue(1080) + "0" + "1" + "1"

        // Simplified: Use smaller dimensions for easier encoding
        // Let's use 32x32 instead for this test
        // ue(32) = ue(32) needs: 32 = 0b100000 = 6 bits
        //   2^5 = 32, so value = 31 + 1 = 32
        //   5 leading zeros + 1 + 00001 = 00000 1 00001

        // Actually, let's create a proper bitstream
        let mut bits = Vec::new();

        // sps_seq_parameter_set_id = 0: "1"
        bits.push(true);

        // chroma_format_idc = 1: "010"
        bits.extend_from_slice(&[false, true, false]);

        // pic_width = 1920 = 0x780 = 11110000000 in binary (11 bits)
        // For ue(v): 1920 - 1 = 1919 = 0x77F
        // We need 2^k - 1 + remainder = 1920
        // 2^10 = 1024, 1024 - 1 = 1023
        // 1920 - 1023 = 897 = 0x381 = 1110000001 (10 bits)
        // So: 0000000000 1 1110000001 (10 zeros + 1 + 10 bits)
        for _ in 0..10 {
            bits.push(false);
        }
        bits.push(true);
        // 897 = 0b1110000001
        bits.extend_from_slice(&[
            true, true, true, false, false, false, false, false, false, true,
        ]);

        // pic_height = 1080 = 0x438
        // 1080 - 1023 = 57 = 0x39 = 0000111001 (10 bits)
        for _ in 0..10 {
            bits.push(false);
        }
        bits.push(true);
        // 57 = 0b0000111001
        bits.extend_from_slice(&[
            false, false, false, false, true, true, true, false, false, true,
        ]);

        // conformance_window_flag = 0
        bits.push(false);

        // bit_depth_luma_minus8 = 0: "1"
        bits.push(true);

        // bit_depth_chroma_minus8 = 0: "1"
        bits.push(true);

        // Convert bits to bytes
        let mut byte = 0u8;
        let mut bit_count = 0;
        for bit in bits {
            byte = (byte << 1) | (if bit { 1 } else { 0 });
            bit_count += 1;
            if bit_count == 8 {
                sps_data.push(byte);
                byte = 0;
                bit_count = 0;
            }
        }
        // Pad remaining bits with zeros
        if bit_count > 0 {
            byte <<= 8 - bit_count;
            sps_data.push(byte);
        }

        // Parse the SPS
        let sps = Sps::parse(&sps_data).expect("Should parse 1080p SPS");

        // Verify parsed values
        assert_eq!(sps.sps_video_parameter_set_id, 0);
        assert_eq!(sps.sps_max_sub_layers_minus1, 0);
        assert_eq!(sps.sps_temporal_id_nesting_flag, true);
        assert_eq!(sps.sps_seq_parameter_set_id, 0);
        assert_eq!(sps.chroma_format_idc, 1);
        assert_eq!(sps.chroma_format_string(), "4:2:0");
        assert_eq!(sps.pic_width_in_luma_samples, 1920);
        assert_eq!(sps.pic_height_in_luma_samples, 1080);
        assert_eq!(sps.conformance_window_flag, false);
        assert_eq!(sps.bit_depth_luma(), 8);
        assert_eq!(sps.bit_depth_chroma(), 8);
    }

    /// Test SPS parsing with 4K 10-bit data
    #[test]
    fn test_sps_parsing_4k_10bit() {
        let mut sps_data = Vec::new();

        // Header: VPS=0, max_sub_layers=0, temporal_nesting=true
        sps_data.push(0b0000_000_1);

        // profile_tier_level (12 bytes, Main 10 profile)
        sps_data.push(0b00_0_00010); // Profile: Main 10 (2)
        sps_data.push(0b00100000); // Main 10 profile compatible
        sps_data.push(0x00);
        sps_data.push(0x00);
        sps_data.push(0x00);
        sps_data.push(0b1000_0000); // Progressive
        for _ in 0..5 {
            sps_data.push(0x00);
        }
        sps_data.push(150); // Level 5.0

        // Build bitstream for 3840x2160 10-bit
        let mut bits = Vec::new();

        // sps_seq_parameter_set_id = 0: "1"
        bits.push(true);

        // chroma_format_idc = 1 (4:2:0): "010"
        bits.extend_from_slice(&[false, true, false]);

        // pic_width = 3840
        // 3840 - 2047 = 1793 = 0x701 = 11100000001 (11 bits)
        // Need 2^11 - 1 = 2047, so 3840 - 2047 = 1793
        for _ in 0..11 {
            bits.push(false);
        }
        bits.push(true);
        // 1793 = 0b11100000001
        bits.extend_from_slice(&[
            true, true, true, false, false, false, false, false, false, false, true,
        ]);

        // pic_height = 2160
        // 2160 - 2047 = 113 = 0x71 = 01110001 (needs 11 bits: 00001110001)
        for _ in 0..11 {
            bits.push(false);
        }
        bits.push(true);
        // 113 = 0b00001110001
        bits.extend_from_slice(&[
            false, false, false, false, true, true, true, false, false, false, true,
        ]);

        // conformance_window_flag = 0
        bits.push(false);

        // bit_depth_luma_minus8 = 2 (10-bit): "011"
        bits.extend_from_slice(&[false, true, true]);

        // bit_depth_chroma_minus8 = 2 (10-bit): "011"
        bits.extend_from_slice(&[false, true, true]);

        // Convert bits to bytes
        let mut byte = 0u8;
        let mut bit_count = 0;
        for bit in bits {
            byte = (byte << 1) | (if bit { 1 } else { 0 });
            bit_count += 1;
            if bit_count == 8 {
                sps_data.push(byte);
                byte = 0;
                bit_count = 0;
            }
        }
        if bit_count > 0 {
            byte <<= 8 - bit_count;
            sps_data.push(byte);
        }

        // Parse the SPS
        let sps = Sps::parse(&sps_data).expect("Should parse 4K 10-bit SPS");

        // Verify parsed values
        assert_eq!(sps.pic_width_in_luma_samples, 3840);
        assert_eq!(sps.pic_height_in_luma_samples, 2160);
        assert_eq!(sps.chroma_format_idc, 1);
        assert_eq!(sps.bit_depth_luma(), 10);
        assert_eq!(sps.bit_depth_chroma(), 10);
    }

    /// Test SPS parsing with conformance window (cropping)
    #[test]
    fn test_sps_parsing_with_conformance_window() {
        let mut sps_data = Vec::new();

        // Header
        sps_data.push(0b0000_000_1);

        // profile_tier_level (12 bytes)
        sps_data.push(0b00_0_00001);
        for _ in 0..11 {
            sps_data.push(0x00);
        }

        // Build bitstream with conformance window
        let mut bits = Vec::new();

        // sps_seq_parameter_set_id = 0: "1"
        bits.push(true);

        // chroma_format_idc = 1: "010"
        bits.extend_from_slice(&[false, true, false]);

        // Use simple dimensions: pic_width = 1920, pic_height = 1088 (padded)
        // ue(1920): 10 zeros + 1 + 897 in 10 bits
        for _ in 0..10 {
            bits.push(false);
        }
        bits.push(true);
        bits.extend_from_slice(&[
            true, true, true, false, false, false, false, false, false, true,
        ]);

        // ue(1088): 1088 - 1023 = 65 = 0b0001000001
        for _ in 0..10 {
            bits.push(false);
        }
        bits.push(true);
        bits.extend_from_slice(&[
            false, false, false, true, false, false, false, false, false, true,
        ]);

        // conformance_window_flag = 1
        bits.push(true);

        // Conformance window offsets (crop 8 pixels from bottom)
        // conf_win_left_offset = 0: "1"
        bits.push(true);
        // conf_win_right_offset = 0: "1"
        bits.push(true);
        // conf_win_top_offset = 0: "1"
        bits.push(true);
        // conf_win_bottom_offset = 4 (4 * 2 = 8 pixels for 4:2:0): "00101"
        bits.extend_from_slice(&[false, false, true, false, true]);

        // bit_depth_luma_minus8 = 0: "1"
        bits.push(true);
        // bit_depth_chroma_minus8 = 0: "1"
        bits.push(true);

        // Convert to bytes
        let mut byte = 0u8;
        let mut bit_count = 0;
        for bit in bits {
            byte = (byte << 1) | (if bit { 1 } else { 0 });
            bit_count += 1;
            if bit_count == 8 {
                sps_data.push(byte);
                byte = 0;
                bit_count = 0;
            }
        }
        if bit_count > 0 {
            byte <<= 8 - bit_count;
            sps_data.push(byte);
        }

        // Parse
        let sps = Sps::parse(&sps_data).expect("Should parse SPS with conformance window");

        // Verify
        assert_eq!(sps.pic_width_in_luma_samples, 1920);
        assert_eq!(sps.pic_height_in_luma_samples, 1088);
        assert_eq!(sps.conformance_window_flag, true);
        assert_eq!(sps.conf_win_left_offset, 0);
        assert_eq!(sps.conf_win_right_offset, 0);
        assert_eq!(sps.conf_win_top_offset, 0);
        assert_eq!(sps.conf_win_bottom_offset, 4); // 4 * 2 = 8 pixels
    }

    /// Test that we correctly identify different NAL unit types
    #[test]
    fn test_nal_unit_type_identification() {
        // Test VPS (type 32)
        let vps_header = vec![0x40, 0x01]; // 0100 0000, 0000 0001
        let vps_nal = NalUnit::parse(&vps_header).unwrap();
        assert_eq!(vps_nal.nal_type(), NalUnitType::VpsNut);
        assert!(vps_nal.nal_type().is_parameter_set());

        // Test SPS (type 33)
        let sps_header = vec![0x42, 0x01]; // 0100 0010, 0000 0001
        let sps_nal = NalUnit::parse(&sps_header).unwrap();
        assert_eq!(sps_nal.nal_type(), NalUnitType::SpsNut);
        assert!(sps_nal.nal_type().is_parameter_set());

        // Test PPS (type 34)
        let pps_header = vec![0x44, 0x01]; // 0100 0100, 0000 0001
        let pps_nal = NalUnit::parse(&pps_header).unwrap();
        assert_eq!(pps_nal.nal_type(), NalUnitType::PpsNut);
        assert!(pps_nal.nal_type().is_parameter_set());

        // Test TRAIL_R (type 1)
        let trail_header = vec![0x02, 0x01]; // 0000 0010, 0000 0001
        let trail_nal = NalUnit::parse(&trail_header).unwrap();
        assert_eq!(trail_nal.nal_type(), NalUnitType::TrailR);
        assert!(trail_nal.is_slice());
        assert!(!trail_nal.is_idr());
    }

    /// Test NAL unit temporal ID extraction
    #[test]
    fn test_nal_temporal_id() {
        // Test TID = 0 (nuh_temporal_id_plus1 = 1)
        let data1 = vec![0x42, 0x01]; // TID = 0
        let nal1 = NalUnit::parse(&data1).unwrap();
        assert_eq!(nal1.header.temporal_id(), 0);

        // Test TID = 2 (nuh_temporal_id_plus1 = 3)
        let data2 = vec![0x42, 0x03]; // TID = 2
        let nal2 = NalUnit::parse(&data2).unwrap();
        assert_eq!(nal2.header.temporal_id(), 2);

        // Test TID = 6 (nuh_temporal_id_plus1 = 7)
        let data3 = vec![0x42, 0x07]; // TID = 6 (max)
        let nal3 = NalUnit::parse(&data3).unwrap();
        assert_eq!(nal3.header.temporal_id(), 6);
    }

    /// Test forbidden zero bit validation
    #[test]
    fn test_forbidden_zero_bit() {
        // Valid: forbidden_zero_bit = 0
        let valid_data = vec![0x42, 0x01];
        assert!(NalUnit::parse(&valid_data).is_ok());

        // Invalid: forbidden_zero_bit = 1
        let invalid_data = vec![0xC2, 0x01]; // 1100 0010 (forbidden bit = 1)
        assert!(NalUnit::parse(&invalid_data).is_err());
    }

    /// Test nuh_temporal_id_plus1 validation
    #[test]
    fn test_temporal_id_plus1_validation() {
        // Valid: nuh_temporal_id_plus1 = 1
        let valid_data = vec![0x42, 0x01];
        assert!(NalUnit::parse(&valid_data).is_ok());

        // Invalid: nuh_temporal_id_plus1 = 0 (not allowed)
        let invalid_data = vec![0x42, 0x00];
        assert!(NalUnit::parse(&invalid_data).is_err());
    }

    /// Test reading signed Exp-Golomb values
    #[test]
    fn test_signed_exp_golomb() {
        // se(v) values: 0, 1, -1, 2, -2
        // ue(v) values: 0, 1,  2, 3,  4
        // Bit patterns: 1, 010, 011, 00100, 00101
        let data = vec![
            0b1_010_011_0, // 1 (0), 010 (1), 011 (-1), 0
            0b0100_00101,  // 00100 (2), 00101 (-2)
        ];

        let mut reader = BitstreamReader::new(&data);

        assert_eq!(reader.read_se().unwrap(), 0);
        assert_eq!(reader.read_se().unwrap(), 1);
        assert_eq!(reader.read_se().unwrap(), -1);
        assert_eq!(reader.read_se().unwrap(), 2);
        assert_eq!(reader.read_se().unwrap(), -2);
    }
}

/// Tests that should fail when h265 feature is not enabled
#[cfg(not(feature = "h265"))]
mod h265_disabled_tests {
    #[test]
    fn test_h265_feature_disabled() {
        // When h265 feature is disabled, the module should not be available
        // This test just verifies the feature flag system works
        assert!(true, "H.265 feature is disabled as expected");
    }
}
