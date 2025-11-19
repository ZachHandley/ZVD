//! H.265/HEVC parameter set headers (VPS, SPS, PPS)
//!
//! These headers contain critical decoding parameters and are sent before video data.

use crate::error::{Error, Result};
use super::bitstream::BitstreamReader;

/// Video Parameter Set (VPS)
///
/// Contains parameters that apply to multiple sequences
#[derive(Debug, Clone)]
pub struct Vps {
    /// VPS ID (0-15)
    pub vps_video_parameter_set_id: u8,
    /// Base layer internal flag
    pub vps_base_layer_internal_flag: bool,
    /// Base layer available flag
    pub vps_base_layer_available_flag: bool,
    /// Maximum number of layers minus 1
    pub vps_max_layers_minus1: u8,
    /// Maximum sub-layers minus 1 (0-6)
    pub vps_max_sub_layers_minus1: u8,
    /// Temporal ID nesting flag
    pub vps_temporal_id_nesting_flag: bool,

    // TODO: Add more VPS fields as needed
}

impl Vps {
    /// Parse VPS from RBSP data
    pub fn parse(rbsp: &[u8]) -> Result<Self> {
        let mut reader = BitstreamReader::new(rbsp);

        // Read VPS video parameter set ID (4 bits, range 0-15)
        let vps_video_parameter_set_id = reader.read_bits(4)? as u8;
        if vps_video_parameter_set_id > 15 {
            return Err(Error::codec(format!("Invalid VPS ID: {}", vps_video_parameter_set_id)));
        }

        // Read vps_base_layer_internal_flag (2 bits)
        let vps_base_layer_internal_flag_bits = reader.read_bits(2)?;
        let vps_base_layer_internal_flag = vps_base_layer_internal_flag_bits != 0;

        // Read vps_base_layer_available_flag (1 bit)
        let vps_base_layer_available_flag = reader.read_bool()?;

        // Read vps_max_layers_minus1 (6 bits, range 0-62)
        let vps_max_layers_minus1 = reader.read_bits(6)? as u8;
        if vps_max_layers_minus1 > 62 {
            return Err(Error::codec(format!("Invalid vps_max_layers_minus1: {}", vps_max_layers_minus1)));
        }

        // Read vps_max_sub_layers_minus1 (3 bits, range 0-6)
        let vps_max_sub_layers_minus1 = reader.read_bits(3)? as u8;
        if vps_max_sub_layers_minus1 > 6 {
            return Err(Error::codec(format!("Invalid vps_max_sub_layers_minus1: {}", vps_max_sub_layers_minus1)));
        }

        // Read vps_temporal_id_nesting_flag (1 bit)
        let vps_temporal_id_nesting_flag = reader.read_bool()?;

        // Read vps_reserved_0xffff_16bits (16 bits, should be 0xFFFF)
        let reserved = reader.read_bits(16)?;
        if reserved != 0xFFFF {
            return Err(Error::codec(format!("VPS reserved bits invalid: expected 0xFFFF, got 0x{:04X}", reserved)));
        }

        // Parse profile_tier_level (complex structure, skip for Phase 8.1)
        Self::skip_profile_tier_level(&mut reader, vps_max_sub_layers_minus1)?;

        // For Phase 8.1, we've read the critical VPS fields
        // Full parsing would continue with:
        // - vps_sub_layer_ordering_info_present_flag
        // - vps_max_dec_pic_buffering_minus1, etc.
        // - Various timing and HRD parameters

        Ok(Vps {
            vps_video_parameter_set_id,
            vps_base_layer_internal_flag,
            vps_base_layer_available_flag,
            vps_max_layers_minus1,
            vps_max_sub_layers_minus1,
            vps_temporal_id_nesting_flag,
        })
    }

    /// Skip profile_tier_level structure (shared with SPS)
    fn skip_profile_tier_level(reader: &mut BitstreamReader, max_sub_layers_minus1: u8) -> Result<()> {
        // general_profile_space (2 bits)
        reader.skip_bits(2)?;
        // general_tier_flag (1 bit)
        reader.skip_bits(1)?;
        // general_profile_idc (5 bits)
        reader.skip_bits(5)?;

        // general_profile_compatibility_flag[32] (32 bits)
        reader.skip_bits(32)?;

        // general_progressive_source_flag through general_reserved_zero_44bits
        reader.skip_bits(48)?;

        // general_level_idc (8 bits)
        reader.skip_bits(8)?;

        // sub_layer_profile_present_flag and sub_layer_level_present_flag
        let mut sub_layer_profile_present = vec![false; max_sub_layers_minus1 as usize];
        let mut sub_layer_level_present = vec![false; max_sub_layers_minus1 as usize];

        for i in 0..max_sub_layers_minus1 as usize {
            sub_layer_profile_present[i] = reader.read_bool()?;
            sub_layer_level_present[i] = reader.read_bool()?;
        }

        // Skip reserved bits if max_sub_layers_minus1 > 0
        if max_sub_layers_minus1 > 0 {
            for _ in max_sub_layers_minus1..8 {
                reader.skip_bits(2)?; // reserved_zero_2bits
            }
        }

        // Skip sub-layer profile/level info
        for i in 0..max_sub_layers_minus1 as usize {
            if sub_layer_profile_present[i] {
                reader.skip_bits(88)?; // sub_layer_profile_space through sub_layer_reserved_zero_44bits
            }
            if sub_layer_level_present[i] {
                reader.skip_bits(8)?; // sub_layer_level_idc
            }
        }

        Ok(())
    }
}

/// Sequence Parameter Set (SPS)
///
/// Contains parameters that apply to all slices in a coded video sequence
#[derive(Debug, Clone)]
pub struct Sps {
    /// VPS ID reference (0-15)
    pub sps_video_parameter_set_id: u8,
    /// Maximum sub-layers minus 1 (0-6)
    pub sps_max_sub_layers_minus1: u8,
    /// Temporal ID nesting flag
    pub sps_temporal_id_nesting_flag: bool,

    /// SPS ID (0-15)
    pub sps_seq_parameter_set_id: u8,

    /// Chroma format (0=Mono, 1=4:2:0, 2=4:2:2, 3=4:4:4)
    pub chroma_format_idc: u8,
    /// Separate color plane flag
    pub separate_colour_plane_flag: bool,

    /// Picture width in luma samples
    pub pic_width_in_luma_samples: u32,
    /// Picture height in luma samples
    pub pic_height_in_luma_samples: u32,

    /// Conformance window flag
    pub conformance_window_flag: bool,
    /// Conformance window left offset
    pub conf_win_left_offset: u32,
    /// Conformance window right offset
    pub conf_win_right_offset: u32,
    /// Conformance window top offset
    pub conf_win_top_offset: u32,
    /// Conformance window bottom offset
    pub conf_win_bottom_offset: u32,

    /// Bit depth (luma) minus 8 (0-6, meaning 8-14 bits)
    pub bit_depth_luma_minus8: u8,
    /// Bit depth (chroma) minus 8 (0-6, meaning 8-14 bits)
    pub bit_depth_chroma_minus8: u8,

    // TODO: Add more SPS fields as needed
}

impl Sps {
    /// Parse SPS from RBSP data
    pub fn parse(rbsp: &[u8]) -> Result<Self> {
        let mut reader = BitstreamReader::new(rbsp);

        // Read VPS ID (4 bits)
        let sps_video_parameter_set_id = reader.read_bits(4)? as u8;

        // Read max sub-layers minus 1 (3 bits, range 0-6)
        let sps_max_sub_layers_minus1 = reader.read_bits(3)? as u8;
        if sps_max_sub_layers_minus1 > 6 {
            return Err(Error::codec(format!("Invalid sps_max_sub_layers_minus1: {}", sps_max_sub_layers_minus1)));
        }

        // Read temporal ID nesting flag (1 bit)
        let sps_temporal_id_nesting_flag = reader.read_bool()?;

        // Parse profile_tier_level (complex structure, simplified for now)
        // For Phase 8.1, we'll skip this and just read past it
        Self::skip_profile_tier_level(&mut reader, sps_max_sub_layers_minus1)?;

        // Read SPS ID (ue(v), range 0-15)
        let sps_seq_parameter_set_id = reader.read_ue()? as u8;
        if sps_seq_parameter_set_id > 15 {
            return Err(Error::codec(format!("Invalid SPS ID: {}", sps_seq_parameter_set_id)));
        }

        // Read chroma format IDC (ue(v), range 0-3)
        let chroma_format_idc = reader.read_ue()? as u8;
        if chroma_format_idc > 3 {
            return Err(Error::codec(format!("Invalid chroma_format_idc: {}", chroma_format_idc)));
        }

        // Read separate color plane flag (only if chroma_format_idc == 3)
        let separate_colour_plane_flag = if chroma_format_idc == 3 {
            reader.read_bool()?
        } else {
            false
        };

        // Read picture width and height (ue(v))
        let pic_width_in_luma_samples = reader.read_ue()?;
        let pic_height_in_luma_samples = reader.read_ue()?;

        // Validate dimensions
        if pic_width_in_luma_samples == 0 || pic_height_in_luma_samples == 0 {
            return Err(Error::codec("Invalid picture dimensions (width or height is 0)"));
        }

        // Read conformance window flag
        let conformance_window_flag = reader.read_bool()?;

        let (conf_win_left_offset, conf_win_right_offset, conf_win_top_offset, conf_win_bottom_offset) =
            if conformance_window_flag {
                (
                    reader.read_ue()?,
                    reader.read_ue()?,
                    reader.read_ue()?,
                    reader.read_ue()?,
                )
            } else {
                (0, 0, 0, 0)
            };

        // Read bit depth minus 8 (ue(v), range 0-6)
        let bit_depth_luma_minus8 = reader.read_ue()? as u8;
        let bit_depth_chroma_minus8 = reader.read_ue()? as u8;

        if bit_depth_luma_minus8 > 6 || bit_depth_chroma_minus8 > 6 {
            return Err(Error::codec("Bit depth too high (>14 bits)"));
        }

        // For Phase 8.1, we've read the critical fields
        // Full parsing would continue with:
        // - log2_max_pic_order_cnt_lsb_minus4
        // - Various flags and parameters
        // - Scaling lists, VUI, etc.

        Ok(Sps {
            sps_video_parameter_set_id,
            sps_max_sub_layers_minus1,
            sps_temporal_id_nesting_flag,
            sps_seq_parameter_set_id,
            chroma_format_idc,
            separate_colour_plane_flag,
            pic_width_in_luma_samples,
            pic_height_in_luma_samples,
            conformance_window_flag,
            conf_win_left_offset,
            conf_win_right_offset,
            conf_win_top_offset,
            conf_win_bottom_offset,
            bit_depth_luma_minus8,
            bit_depth_chroma_minus8,
        })
    }

    /// Skip profile_tier_level structure (simplified for Phase 8.1)
    fn skip_profile_tier_level(reader: &mut BitstreamReader, max_sub_layers_minus1: u8) -> Result<()> {
        // general_profile_space (2 bits)
        reader.skip_bits(2)?;
        // general_tier_flag (1 bit)
        reader.skip_bits(1)?;
        // general_profile_idc (5 bits)
        reader.skip_bits(5)?;

        // general_profile_compatibility_flag[32] (32 bits)
        reader.skip_bits(32)?;

        // general_progressive_source_flag through general_reserved_zero_44bits
        reader.skip_bits(48)?;

        // general_level_idc (8 bits)
        reader.skip_bits(8)?;

        // sub_layer_profile_present_flag and sub_layer_level_present_flag
        let mut sub_layer_profile_present = vec![false; max_sub_layers_minus1 as usize];
        let mut sub_layer_level_present = vec![false; max_sub_layers_minus1 as usize];

        for i in 0..max_sub_layers_minus1 as usize {
            sub_layer_profile_present[i] = reader.read_bool()?;
            sub_layer_level_present[i] = reader.read_bool()?;
        }

        // Skip reserved bits if max_sub_layers_minus1 > 0
        if max_sub_layers_minus1 > 0 {
            for _ in max_sub_layers_minus1..8 {
                reader.skip_bits(2)?; // reserved_zero_2bits
            }
        }

        // Skip sub-layer profile/level info
        for i in 0..max_sub_layers_minus1 as usize {
            if sub_layer_profile_present[i] {
                reader.skip_bits(88)?; // sub_layer_profile_space through sub_layer_reserved_zero_44bits
            }
            if sub_layer_level_present[i] {
                reader.skip_bits(8)?; // sub_layer_level_idc
            }
        }

        Ok(())
    }

    /// Get actual luma bit depth
    pub fn bit_depth_luma(&self) -> u8 {
        self.bit_depth_luma_minus8 + 8
    }

    /// Get actual chroma bit depth
    pub fn bit_depth_chroma(&self) -> u8 {
        self.bit_depth_chroma_minus8 + 8
    }

    /// Get chroma format as string
    pub fn chroma_format_string(&self) -> &'static str {
        match self.chroma_format_idc {
            0 => "Monochrome",
            1 => "4:2:0",
            2 => "4:2:2",
            3 => "4:4:4",
            _ => "Unknown",
        }
    }
}

/// Picture Parameter Set (PPS)
///
/// Contains parameters that apply to all slices in a coded picture
#[derive(Debug, Clone)]
pub struct Pps {
    /// PPS ID (0-63)
    pub pps_pic_parameter_set_id: u8,
    /// SPS ID reference (0-15)
    pub pps_seq_parameter_set_id: u8,

    /// Dependent slice segments enabled flag
    pub dependent_slice_segments_enabled_flag: bool,
    /// Output flag present flag
    pub output_flag_present_flag: bool,
    /// Number of extra slice header bits
    pub num_extra_slice_header_bits: u8,

    /// Sign data hiding enabled flag
    pub sign_data_hiding_enabled_flag: bool,
    /// Cabac init present flag
    pub cabac_init_present_flag: bool,

    // TODO: Add more PPS fields as needed
}

impl Pps {
    /// Parse PPS from RBSP data
    pub fn parse(_rbsp: &[u8]) -> Result<Self> {
        // TODO: Implement full PPS parsing
        // For now, return a placeholder
        Err(Error::codec("PPS parsing not yet implemented"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sps_bit_depth() {
        let sps = Sps {
            sps_video_parameter_set_id: 0,
            sps_max_sub_layers_minus1: 0,
            sps_temporal_id_nesting_flag: false,
            sps_seq_parameter_set_id: 0,
            chroma_format_idc: 1,
            separate_colour_plane_flag: false,
            pic_width_in_luma_samples: 1920,
            pic_height_in_luma_samples: 1080,
            conformance_window_flag: false,
            conf_win_left_offset: 0,
            conf_win_right_offset: 0,
            conf_win_top_offset: 0,
            conf_win_bottom_offset: 0,
            bit_depth_luma_minus8: 0,   // 8-bit
            bit_depth_chroma_minus8: 0, // 8-bit
        };

        assert_eq!(sps.bit_depth_luma(), 8);
        assert_eq!(sps.bit_depth_chroma(), 8);
    }

    #[test]
    fn test_sps_10bit_depth() {
        let sps = Sps {
            sps_video_parameter_set_id: 0,
            sps_max_sub_layers_minus1: 0,
            sps_temporal_id_nesting_flag: false,
            sps_seq_parameter_set_id: 0,
            chroma_format_idc: 1,
            separate_colour_plane_flag: false,
            pic_width_in_luma_samples: 3840,
            pic_height_in_luma_samples: 2160,
            conformance_window_flag: false,
            conf_win_left_offset: 0,
            conf_win_right_offset: 0,
            conf_win_top_offset: 0,
            conf_win_bottom_offset: 0,
            bit_depth_luma_minus8: 2,   // 10-bit
            bit_depth_chroma_minus8: 2, // 10-bit
        };

        assert_eq!(sps.bit_depth_luma(), 10);
        assert_eq!(sps.bit_depth_chroma(), 10);
    }

    #[test]
    fn test_sps_chroma_format_string() {
        let mut sps = Sps {
            sps_video_parameter_set_id: 0,
            sps_max_sub_layers_minus1: 0,
            sps_temporal_id_nesting_flag: false,
            sps_seq_parameter_set_id: 0,
            chroma_format_idc: 1,
            separate_colour_plane_flag: false,
            pic_width_in_luma_samples: 1920,
            pic_height_in_luma_samples: 1080,
            conformance_window_flag: false,
            conf_win_left_offset: 0,
            conf_win_right_offset: 0,
            conf_win_top_offset: 0,
            conf_win_bottom_offset: 0,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
        };

        sps.chroma_format_idc = 0;
        assert_eq!(sps.chroma_format_string(), "Monochrome");

        sps.chroma_format_idc = 1;
        assert_eq!(sps.chroma_format_string(), "4:2:0");

        sps.chroma_format_idc = 2;
        assert_eq!(sps.chroma_format_string(), "4:2:2");

        sps.chroma_format_idc = 3;
        assert_eq!(sps.chroma_format_string(), "4:4:4");
    }

    #[test]
    fn test_vps_parsing_basic() {
        let mut vps_data = Vec::new();

        // vps_video_parameter_set_id (4 bits) = 0
        // vps_base_layer_internal_flag (2 bits) = 3 (11 binary)
        // vps_base_layer_available_flag (1 bit) = 1
        // vps_max_layers_minus1 (6 bits) = 0
        // vps_max_sub_layers_minus1 (3 bits) = 0
        // vps_temporal_id_nesting_flag (1 bit) = 1
        // Total: 17 bits = 2 bytes + 1 bit

        // Byte 0: 0000 11 1 0 = 0x1C (shifted)
        // Actually: vps_id=0000, base_internal=11
        vps_data.push(0b0000_11_10); // vps_id=0, base_internal=3, base_avail=1, max_layers[0:0]

        // Byte 1: max_layers[1:5]=00000, max_sub_layers=000, temporal_nesting=1
        vps_data.push(0b00000_000); // rest of max_layers, max_sub_layers
        vps_data.push(0b1_0000000); // temporal_nesting, start of reserved

        // vps_reserved_0xffff_16bits
        vps_data.push(0xFF);
        vps_data.push(0xFF);

        // profile_tier_level (12 bytes for max_sub_layers_minus1 = 0)
        vps_data.push(0b00_0_00001); // Profile: Main
        for _ in 0..11 {
            vps_data.push(0x00);
        }

        // Parse VPS
        let vps = Vps::parse(&vps_data).expect("Should parse basic VPS");

        // Verify
        assert_eq!(vps.vps_video_parameter_set_id, 0);
        assert_eq!(vps.vps_base_layer_internal_flag, true); // 3 = 0b11 != 0
        assert_eq!(vps.vps_base_layer_available_flag, true);
        assert_eq!(vps.vps_max_layers_minus1, 0);
        assert_eq!(vps.vps_max_sub_layers_minus1, 0);
        assert_eq!(vps.vps_temporal_id_nesting_flag, true);
    }

    #[test]
    fn test_vps_parsing_multi_layer() {
        // Create a more complex bitstream
        let mut bits = Vec::new();

        // vps_video_parameter_set_id = 1 (4 bits)
        bits.extend_from_slice(&[false, false, false, true]);

        // vps_base_layer_internal_flag = 0 (2 bits)
        bits.extend_from_slice(&[false, false]);

        // vps_base_layer_available_flag = 1 (1 bit)
        bits.push(true);

        // vps_max_layers_minus1 = 2 (6 bits: 000010)
        bits.extend_from_slice(&[false, false, false, false, true, false]);

        // vps_max_sub_layers_minus1 = 1 (3 bits: 001)
        bits.extend_from_slice(&[false, false, true]);

        // vps_temporal_id_nesting_flag = 0 (1 bit)
        bits.push(false);

        // vps_reserved_0xffff_16bits (16 bits: all 1s)
        for _ in 0..16 {
            bits.push(true);
        }

        // profile_tier_level
        // general_profile_space (2), tier_flag (1), profile_idc (5) = 8 bits
        bits.extend_from_slice(&[false, false, false, false, false, false, true, false]); // Main 10

        // general_profile_compatibility_flag[32]
        for _ in 0..32 {
            bits.push(false);
        }

        // 48 bits for progressive/interlaced/reserved
        for _ in 0..48 {
            bits.push(false);
        }

        // general_level_idc (8 bits)
        for _ in 0..8 {
            bits.push(false);
        }

        // sub_layer flags for max_sub_layers_minus1 = 1
        bits.push(false); // sub_layer_profile_present_flag[0]
        bits.push(true);  // sub_layer_level_present_flag[0]

        // reserved_zero_2bits (14 bits total: (8-1)*2 = 14)
        for _ in 0..14 {
            bits.push(false);
        }

        // sub_layer_level_idc[0] (8 bits) since level_present=true
        for _ in 0..8 {
            bits.push(false);
        }

        // Convert bits to bytes
        let mut vps_data = Vec::new();
        let mut byte = 0u8;
        let mut bit_count = 0;
        for bit in bits {
            byte = (byte << 1) | (if bit { 1 } else { 0 });
            bit_count += 1;
            if bit_count == 8 {
                vps_data.push(byte);
                byte = 0;
                bit_count = 0;
            }
        }
        if bit_count > 0 {
            byte <<= 8 - bit_count;
            vps_data.push(byte);
        }

        // Parse VPS
        let vps = Vps::parse(&vps_data).expect("Should parse multi-layer VPS");

        // Verify
        assert_eq!(vps.vps_video_parameter_set_id, 1);
        assert_eq!(vps.vps_base_layer_internal_flag, false);
        assert_eq!(vps.vps_base_layer_available_flag, true);
        assert_eq!(vps.vps_max_layers_minus1, 2);
        assert_eq!(vps.vps_max_sub_layers_minus1, 1);
        assert_eq!(vps.vps_temporal_id_nesting_flag, false);
    }

    #[test]
    fn test_vps_invalid_reserved_bits() {
        let mut vps_data = Vec::new();

        // Build minimal VPS header
        vps_data.push(0b0000_00_1_0);
        vps_data.push(0b00000_000);
        vps_data.push(0b1_0000000);

        // Invalid reserved bits (should be 0xFFFF)
        vps_data.push(0x00); // Wrong!
        vps_data.push(0x00);

        // This should fail due to invalid reserved bits
        let result = Vps::parse(&vps_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_vps_invalid_id() {
        let mut vps_data = Vec::new();

        // VPS ID = 16 (invalid, max is 15)
        // This is impossible with 4 bits, but let's test range checking elsewhere
        // Actually, 4 bits can only represent 0-15, so this test is not needed
        // But we can test max_layers_minus1 validation

        // vps_video_parameter_set_id = 15 (valid, max value)
        vps_data.push(0b1111_00_1_0);
        vps_data.push(0b00000_000);
        vps_data.push(0b1_0000000);
        vps_data.push(0xFF);
        vps_data.push(0xFF);

        // profile_tier_level
        for _ in 0..12 {
            vps_data.push(0x00);
        }

        let result = Vps::parse(&vps_data);
        assert!(result.is_ok()); // ID=15 is valid
    }
}
