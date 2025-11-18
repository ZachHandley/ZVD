//! H.265/HEVC parameter set headers (VPS, SPS, PPS)
//!
//! These headers contain critical decoding parameters and are sent before video data.

use crate::error::{Error, Result};

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
    pub fn parse(_rbsp: &[u8]) -> Result<Self> {
        // TODO: Implement full VPS parsing
        // For now, return a placeholder
        Err(Error::codec("VPS parsing not yet implemented"))
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
    pub fn parse(_rbsp: &[u8]) -> Result<Self> {
        // TODO: Implement full SPS parsing with bitstream reader
        // For now, return a placeholder
        Err(Error::codec("SPS parsing not yet implemented"))
    }

    /// Get actual luma bit depth
    pub fn bit_depth_luma(&self) -> u8 {
        self.bit_depth_luma_minus8 + 8
    }

    /// Get actual chroma bit depth
    pub fn bit_depth_chroma(&self) -> u8 {
        self.bit_depth_chroma_minus8 + 8
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
}
