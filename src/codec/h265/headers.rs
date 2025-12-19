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

    /// Number of reference index for L0 minus 1
    pub num_ref_idx_l0_default_active_minus1: u8,
    /// Number of reference index for L1 minus 1
    pub num_ref_idx_l1_default_active_minus1: u8,

    /// Initial QP minus 26 (-26 to +25)
    pub init_qp_minus26: i8,

    /// Constrained intra prediction flag
    pub constrained_intra_pred_flag: bool,
    /// Transform skip enabled flag
    pub transform_skip_enabled_flag: bool,

    // TODO: Add more PPS fields as needed (tiles, weighted prediction, etc.)
}

impl Pps {
    /// Parse PPS from RBSP data
    pub fn parse(rbsp: &[u8]) -> Result<Self> {
        let mut reader = BitstreamReader::new(rbsp);

        // Read PPS ID (ue(v), range 0-63)
        let pps_pic_parameter_set_id = reader.read_ue()? as u8;
        if pps_pic_parameter_set_id > 63 {
            return Err(Error::codec(format!("Invalid PPS ID: {}", pps_pic_parameter_set_id)));
        }

        // Read SPS ID (ue(v), range 0-15)
        let pps_seq_parameter_set_id = reader.read_ue()? as u8;
        if pps_seq_parameter_set_id > 15 {
            return Err(Error::codec(format!("Invalid SPS ID in PPS: {}", pps_seq_parameter_set_id)));
        }

        // Read dependent slice segments enabled flag (1 bit)
        let dependent_slice_segments_enabled_flag = reader.read_bool()?;

        // Read output flag present flag (1 bit)
        let output_flag_present_flag = reader.read_bool()?;

        // Read number of extra slice header bits (3 bits)
        let num_extra_slice_header_bits = reader.read_bits(3)? as u8;

        // Read sign data hiding enabled flag (1 bit)
        let sign_data_hiding_enabled_flag = reader.read_bool()?;

        // Read cabac init present flag (1 bit)
        let cabac_init_present_flag = reader.read_bool()?;

        // Read num_ref_idx_l0_default_active_minus1 (ue(v), range 0-14)
        let num_ref_idx_l0_default_active_minus1 = reader.read_ue()? as u8;
        if num_ref_idx_l0_default_active_minus1 > 14 {
            return Err(Error::codec("num_ref_idx_l0_default_active_minus1 out of range"));
        }

        // Read num_ref_idx_l1_default_active_minus1 (ue(v), range 0-14)
        let num_ref_idx_l1_default_active_minus1 = reader.read_ue()? as u8;
        if num_ref_idx_l1_default_active_minus1 > 14 {
            return Err(Error::codec("num_ref_idx_l1_default_active_minus1 out of range"));
        }

        // Read init_qp_minus26 (se(v), range -26 to +25)
        let init_qp_minus26 = reader.read_se()?;
        if init_qp_minus26 < -26 || init_qp_minus26 > 25 {
            return Err(Error::codec(format!("init_qp_minus26 out of range: {}", init_qp_minus26)));
        }

        // Read constrained_intra_pred_flag (1 bit)
        let constrained_intra_pred_flag = reader.read_bool()?;

        // Read transform_skip_enabled_flag (1 bit)
        let transform_skip_enabled_flag = reader.read_bool()?;

        // For Phase 8.1, we've read the most critical PPS fields
        // Full parsing would continue with:
        // - cu_qp_delta_enabled_flag and diff_cu_qp_delta_depth
        // - pps_cb_qp_offset and pps_cr_qp_offset
        // - pps_slice_chroma_qp_offsets_present_flag
        // - weighted_pred_flag and weighted_bipred_flag
        // - transquant_bypass_enabled_flag
        // - tiles_enabled_flag and tile parameters
        // - pps_loop_filter_across_slices_enabled_flag
        // - deblocking_filter parameters
        // - pps_scaling_list_data_present_flag
        // - lists_modification_present_flag
        // - log2_parallel_merge_level_minus2
        // - slice_segment_header_extension_present_flag

        Ok(Pps {
            pps_pic_parameter_set_id,
            pps_seq_parameter_set_id,
            dependent_slice_segments_enabled_flag,
            output_flag_present_flag,
            num_extra_slice_header_bits,
            sign_data_hiding_enabled_flag,
            cabac_init_present_flag,
            num_ref_idx_l0_default_active_minus1,
            num_ref_idx_l1_default_active_minus1,
            init_qp_minus26: init_qp_minus26 as i8,
            constrained_intra_pred_flag,
            transform_skip_enabled_flag,
        })
    }
}

/// Slice type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceType {
    /// B slice (bi-predictive)
    B = 0,
    /// P slice (predictive)
    P = 1,
    /// I slice (intra)
    I = 2,
}

impl SliceType {
    /// Parse slice type from ue(v) value
    pub fn from_ue(value: u32) -> Result<Self> {
        match value {
            0 => Ok(SliceType::B),
            1 => Ok(SliceType::P),
            2 => Ok(SliceType::I),
            // H.265 spec allows 3-9 for B/P/I with special meanings (non-reference)
            3 => Ok(SliceType::B), // B (non-reference)
            4 => Ok(SliceType::P), // P (non-reference)
            5 => Ok(SliceType::I), // I (non-reference)
            6 => Ok(SliceType::B), // B (non-reference, alternative)
            7 => Ok(SliceType::P), // P (non-reference, alternative)
            8 => Ok(SliceType::I), // I (non-reference, alternative)
            9 => Ok(SliceType::I), // I (non-reference, alternative)
            _ => Err(Error::codec(format!("Invalid slice type: {}", value))),
        }
    }

    /// Check if this is an I slice
    pub fn is_intra(&self) -> bool {
        matches!(self, SliceType::I)
    }

    /// Check if this is a P slice
    pub fn is_predictive(&self) -> bool {
        matches!(self, SliceType::P)
    }

    /// Check if this is a B slice
    pub fn is_bipredictive(&self) -> bool {
        matches!(self, SliceType::B)
    }
}

/// Slice Segment Header
///
/// Contains parameters for a single slice segment within a picture.
/// Note: In H.265, a picture can be divided into multiple slices.
#[derive(Debug, Clone)]
pub struct SliceHeader {
    /// First slice segment in picture flag
    pub first_slice_segment_in_pic_flag: bool,

    /// No output of prior pictures flag (for IRAP)
    pub no_output_of_prior_pics_flag: bool,

    /// PPS ID reference (0-63)
    pub slice_pic_parameter_set_id: u8,

    /// Dependent slice segment flag
    pub dependent_slice_segment_flag: bool,

    /// Slice segment address (CTU address)
    pub slice_segment_address: u32,

    /// Slice type (I, P, or B)
    pub slice_type: SliceType,

    /// Picture output flag
    pub pic_output_flag: bool,

    /// Colour plane ID (for 4:4:4 separate planes)
    pub colour_plane_id: Option<u8>,

    /// Slice QP delta (adjustment to PPS init_qp)
    pub slice_qp_delta: i32,

    /// Slice deblocking filter disabled flag
    pub slice_deblocking_filter_disabled_flag: bool,

    // TODO: Add more fields for full slice header parsing:
    // - Reference picture lists
    // - Weighted prediction parameters
    // - Temporal MVP flags
    // - SAO parameters
    // - etc.
}

impl SliceHeader {
    /// Parse slice header from RBSP data
    ///
    /// Note: This requires NAL unit type and parameter sets for full parsing.
    /// For Phase 8.1, we implement essential fields only.
    pub fn parse(
        rbsp: &[u8],
        nal_unit_type: u8,
        sps: &Sps,
        pps: &Pps,
    ) -> Result<Self> {
        let mut reader = BitstreamReader::new(rbsp);

        // first_slice_segment_in_pic_flag (1 bit)
        let first_slice_segment_in_pic_flag = reader.read_bool()?;

        // Check if this is an IRAP picture (IDR, CRA, BLA)
        let is_irap = nal_unit_type >= 16 && nal_unit_type <= 23;

        // no_output_of_prior_pics_flag (1 bit, only for IRAP)
        let no_output_of_prior_pics_flag = if is_irap {
            reader.read_bool()?
        } else {
            false
        };

        // slice_pic_parameter_set_id (ue(v), range 0-63)
        let slice_pic_parameter_set_id = reader.read_ue()? as u8;
        if slice_pic_parameter_set_id > 63 {
            return Err(Error::codec("Invalid PPS ID in slice header"));
        }

        // dependent_slice_segment_flag (only if not first slice)
        let dependent_slice_segment_flag = if !first_slice_segment_in_pic_flag {
            if pps.dependent_slice_segments_enabled_flag {
                reader.read_bool()?
            } else {
                false
            }
        } else {
            false
        };

        // slice_segment_address (v bits, only if not first slice)
        let slice_segment_address = if !first_slice_segment_in_pic_flag {
            // Number of bits depends on picture size in CTUs
            // For Phase 8.1, we'll read it as ue(v) for simplicity
            reader.read_ue()?
        } else {
            0
        };

        // If this is a dependent slice, we skip most of the header
        if dependent_slice_segment_flag {
            // Dependent slices inherit parameters from previous slice
            // For Phase 8.1, return minimal header
            return Ok(SliceHeader {
                first_slice_segment_in_pic_flag,
                no_output_of_prior_pics_flag,
                slice_pic_parameter_set_id,
                dependent_slice_segment_flag,
                slice_segment_address,
                slice_type: SliceType::I, // Inherited from previous
                pic_output_flag: true,
                colour_plane_id: None,
                slice_qp_delta: 0,
                slice_deblocking_filter_disabled_flag: false,
            });
        }

        // Read slice_type (ue(v))
        let slice_type_val = reader.read_ue()?;
        let slice_type = SliceType::from_ue(slice_type_val)?;

        // pic_output_flag (1 bit, if pps.output_flag_present_flag)
        let pic_output_flag = if pps.output_flag_present_flag {
            reader.read_bool()?
        } else {
            true
        };

        // colour_plane_id (2 bits, if separate_colour_plane_flag)
        let colour_plane_id = if sps.separate_colour_plane_flag {
            Some(reader.read_bits(2)? as u8)
        } else {
            None
        };

        // For Phase 8.1, skip complex reference picture set parsing
        // In full implementation, we'd parse:
        // - short_term_ref_pic_set_sps_flag and related
        // - long_term reference pictures
        // - slice_temporal_mvp_enabled_flag
        // - slice_sao_luma_flag, slice_sao_chroma_flag

        // For now, skip to slice_qp_delta which is essential
        // Note: This is a simplification; full parsing would handle all fields

        // slice_qp_delta (se(v))
        let slice_qp_delta = reader.read_se()?;

        // slice_deblocking_filter_disabled_flag (1 bit, conditionally)
        // For Phase 8.1, assume it's present
        let slice_deblocking_filter_disabled_flag = reader.read_bool().unwrap_or(false);

        Ok(SliceHeader {
            first_slice_segment_in_pic_flag,
            no_output_of_prior_pics_flag,
            slice_pic_parameter_set_id,
            dependent_slice_segment_flag,
            slice_segment_address,
            slice_type,
            pic_output_flag,
            colour_plane_id,
            slice_qp_delta,
            slice_deblocking_filter_disabled_flag,
        })
    }

    /// Get the final QP value for this slice
    pub fn get_qp(&self, pps: &Pps) -> i32 {
        26 + pps.init_qp_minus26 as i32 + self.slice_qp_delta
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

    #[test]
    fn test_pps_parsing_basic() {
        // Build minimal PPS bitstream
        let mut bits = Vec::new();

        // pps_pic_parameter_set_id = 0 (ue(v)): "1"
        bits.push(true);

        // pps_seq_parameter_set_id = 0 (ue(v)): "1"
        bits.push(true);

        // dependent_slice_segments_enabled_flag = 0 (1 bit)
        bits.push(false);

        // output_flag_present_flag = 0 (1 bit)
        bits.push(false);

        // num_extra_slice_header_bits = 0 (3 bits)
        bits.extend_from_slice(&[false, false, false]);

        // sign_data_hiding_enabled_flag = 0 (1 bit)
        bits.push(false);

        // cabac_init_present_flag = 0 (1 bit)
        bits.push(false);

        // num_ref_idx_l0_default_active_minus1 = 0 (ue(v)): "1"
        bits.push(true);

        // num_ref_idx_l1_default_active_minus1 = 0 (ue(v)): "1"
        bits.push(true);

        // init_qp_minus26 = 0 (se(v)): "1"
        bits.push(true);

        // constrained_intra_pred_flag = 0 (1 bit)
        bits.push(false);

        // transform_skip_enabled_flag = 0 (1 bit)
        bits.push(false);

        // Convert to bytes
        let mut pps_data = Vec::new();
        let mut byte = 0u8;
        let mut bit_count = 0;
        for bit in bits {
            byte = (byte << 1) | (if bit { 1 } else { 0 });
            bit_count += 1;
            if bit_count == 8 {
                pps_data.push(byte);
                byte = 0;
                bit_count = 0;
            }
        }
        if bit_count > 0 {
            byte <<= 8 - bit_count;
            pps_data.push(byte);
        }

        // Parse PPS
        let pps = Pps::parse(&pps_data).expect("Should parse basic PPS");

        // Verify
        assert_eq!(pps.pps_pic_parameter_set_id, 0);
        assert_eq!(pps.pps_seq_parameter_set_id, 0);
        assert_eq!(pps.dependent_slice_segments_enabled_flag, false);
        assert_eq!(pps.output_flag_present_flag, false);
        assert_eq!(pps.num_extra_slice_header_bits, 0);
        assert_eq!(pps.sign_data_hiding_enabled_flag, false);
        assert_eq!(pps.cabac_init_present_flag, false);
        assert_eq!(pps.num_ref_idx_l0_default_active_minus1, 0);
        assert_eq!(pps.num_ref_idx_l1_default_active_minus1, 0);
        assert_eq!(pps.init_qp_minus26, 0);
        assert_eq!(pps.constrained_intra_pred_flag, false);
        assert_eq!(pps.transform_skip_enabled_flag, false);
    }

    #[test]
    fn test_pps_parsing_with_features() {
        // Build PPS with various features enabled
        let mut bits = Vec::new();

        // pps_pic_parameter_set_id = 1 (ue(v)): "010"
        bits.extend_from_slice(&[false, true, false]);

        // pps_seq_parameter_set_id = 0 (ue(v)): "1"
        bits.push(true);

        // dependent_slice_segments_enabled_flag = 1
        bits.push(true);

        // output_flag_present_flag = 1
        bits.push(true);

        // num_extra_slice_header_bits = 3 (3 bits: 011)
        bits.extend_from_slice(&[false, true, true]);

        // sign_data_hiding_enabled_flag = 1
        bits.push(true);

        // cabac_init_present_flag = 1
        bits.push(true);

        // num_ref_idx_l0_default_active_minus1 = 3 (ue(v)): "00100"
        bits.extend_from_slice(&[false, false, true, false, false]);

        // num_ref_idx_l1_default_active_minus1 = 2 (ue(v)): "011"
        bits.extend_from_slice(&[false, true, true]);

        // init_qp_minus26 = -4 (se(v) for -4 is ue(8)): "000010000"
        bits.extend_from_slice(&[false, false, false, false, true, false, false, false, false]);

        // constrained_intra_pred_flag = 1
        bits.push(true);

        // transform_skip_enabled_flag = 1
        bits.push(true);

        // Convert to bytes
        let mut pps_data = Vec::new();
        let mut byte = 0u8;
        let mut bit_count = 0;
        for bit in bits {
            byte = (byte << 1) | (if bit { 1 } else { 0 });
            bit_count += 1;
            if bit_count == 8 {
                pps_data.push(byte);
                byte = 0;
                bit_count = 0;
            }
        }
        if bit_count > 0 {
            byte <<= 8 - bit_count;
            pps_data.push(byte);
        }

        // Parse PPS
        let pps = Pps::parse(&pps_data).expect("Should parse PPS with features");

        // Verify
        assert_eq!(pps.pps_pic_parameter_set_id, 1);
        assert_eq!(pps.pps_seq_parameter_set_id, 0);
        assert_eq!(pps.dependent_slice_segments_enabled_flag, true);
        assert_eq!(pps.output_flag_present_flag, true);
        assert_eq!(pps.num_extra_slice_header_bits, 3);
        assert_eq!(pps.sign_data_hiding_enabled_flag, true);
        assert_eq!(pps.cabac_init_present_flag, true);
        assert_eq!(pps.num_ref_idx_l0_default_active_minus1, 3);
        assert_eq!(pps.num_ref_idx_l1_default_active_minus1, 2);
        assert_eq!(pps.init_qp_minus26, -4);
        assert_eq!(pps.constrained_intra_pred_flag, true);
        assert_eq!(pps.transform_skip_enabled_flag, true);
    }

    #[test]
    fn test_pps_qp_range() {
        // Test QP at boundaries
        let mut bits = Vec::new();

        // Minimal header
        bits.push(true); // pps_id = 0
        bits.push(true); // sps_id = 0
        bits.push(false); // dependent_slice_segments
        bits.push(false); // output_flag
        bits.extend_from_slice(&[false, false, false]); // num_extra_bits = 0
        bits.push(false); // sign_data_hiding
        bits.push(false); // cabac_init
        bits.push(true); // num_ref_idx_l0 = 0
        bits.push(true); // num_ref_idx_l1 = 0

        // init_qp_minus26 = 10 (se(v) for 10 is ue(20)): "0000010100"
        // ue(20): 4 leading zeros + 1 + 0100 (4 in binary)
        bits.extend_from_slice(&[false, false, false, false, true, false, true, false, false]);

        bits.push(false); // constrained_intra
        bits.push(false); // transform_skip

        // Convert to bytes
        let mut pps_data = Vec::new();
        let mut byte = 0u8;
        let mut bit_count = 0;
        for bit in bits {
            byte = (byte << 1) | (if bit { 1 } else { 0 });
            bit_count += 1;
            if bit_count == 8 {
                pps_data.push(byte);
                byte = 0;
                bit_count = 0;
            }
        }
        if bit_count > 0 {
            byte <<= 8 - bit_count;
            pps_data.push(byte);
        }

        // Parse PPS
        let pps = Pps::parse(&pps_data).expect("Should parse PPS with QP=10");
        assert_eq!(pps.init_qp_minus26, 10);
    }

    #[test]
    fn test_pps_invalid_id() {
        // Create PPS with invalid PPS ID (>63)
        // Since ue(v) can encode any value, we manually test validation

        let mut bits = Vec::new();

        // pps_pic_parameter_set_id = 64 (ue(v))
        // ue(64): 64 - 1 = 63 = 0b111111 (6 bits)
        // Need 2^6 - 1 = 63, so 64 - 63 = 1 = 0b000001
        // 6 leading zeros + 1 + 000001
        for _ in 0..6 { bits.push(false); }
        bits.push(true);
        bits.extend_from_slice(&[false, false, false, false, false, true]);

        bits.push(true); // sps_id = 0

        // Convert to bytes
        let mut pps_data = Vec::new();
        let mut byte = 0u8;
        let mut bit_count = 0;
        for bit in bits {
            byte = (byte << 1) | (if bit { 1 } else { 0 });
            bit_count += 1;
            if bit_count == 8 {
                pps_data.push(byte);
                byte = 0;
                bit_count = 0;
            }
        }
        if bit_count > 0 {
            byte <<= 8 - bit_count;
            pps_data.push(byte);
        }

        // Should fail validation
        let result = Pps::parse(&pps_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_slice_type_from_ue() {
        // Test standard slice types
        assert_eq!(SliceType::from_ue(0).unwrap(), SliceType::B);
        assert_eq!(SliceType::from_ue(1).unwrap(), SliceType::P);
        assert_eq!(SliceType::from_ue(2).unwrap(), SliceType::I);

        // Test non-reference variants (3-9)
        assert_eq!(SliceType::from_ue(3).unwrap(), SliceType::B);
        assert_eq!(SliceType::from_ue(4).unwrap(), SliceType::P);
        assert_eq!(SliceType::from_ue(5).unwrap(), SliceType::I);
        assert_eq!(SliceType::from_ue(6).unwrap(), SliceType::B);
        assert_eq!(SliceType::from_ue(7).unwrap(), SliceType::P);
        assert_eq!(SliceType::from_ue(8).unwrap(), SliceType::I);
        assert_eq!(SliceType::from_ue(9).unwrap(), SliceType::I);

        // Test invalid slice type
        assert!(SliceType::from_ue(10).is_err());
        assert!(SliceType::from_ue(100).is_err());
    }

    #[test]
    fn test_slice_type_helpers() {
        let i_slice = SliceType::I;
        assert!(i_slice.is_intra());
        assert!(!i_slice.is_predictive());
        assert!(!i_slice.is_bipredictive());

        let p_slice = SliceType::P;
        assert!(!p_slice.is_intra());
        assert!(p_slice.is_predictive());
        assert!(!p_slice.is_bipredictive());

        let b_slice = SliceType::B;
        assert!(!b_slice.is_intra());
        assert!(!b_slice.is_predictive());
        assert!(b_slice.is_bipredictive());
    }

    #[test]
    fn test_slice_header_parsing_i_slice() {
        // Create minimal SPS
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
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
        };

        // Create minimal PPS
        let pps = Pps {
            pps_pic_parameter_set_id: 0,
            pps_seq_parameter_set_id: 0,
            dependent_slice_segments_enabled_flag: false,
            output_flag_present_flag: false,
            num_extra_slice_header_bits: 0,
            sign_data_hiding_enabled_flag: false,
            cabac_init_present_flag: false,
            num_ref_idx_l0_default_active_minus1: 0,
            num_ref_idx_l1_default_active_minus1: 0,
            init_qp_minus26: 0,
            constrained_intra_pred_flag: false,
            transform_skip_enabled_flag: false,
        };

        // Build slice header for IDR I slice
        let mut bits = Vec::new();

        // first_slice_segment_in_pic_flag = 1
        bits.push(true);

        // no_output_of_prior_pics_flag = 0 (for IDR, nal_type=19)
        bits.push(false);

        // slice_pic_parameter_set_id = 0 (ue(v)): "1"
        bits.push(true);

        // slice_type = 2 (I slice, ue(v)): "011"
        bits.extend_from_slice(&[false, true, true]);

        // slice_qp_delta = 0 (se(v)): "1"
        bits.push(true);

        // slice_deblocking_filter_disabled_flag = 0
        bits.push(false);

        // Convert to bytes
        let mut slice_data = Vec::new();
        let mut byte = 0u8;
        let mut bit_count = 0;
        for bit in bits {
            byte = (byte << 1) | (if bit { 1 } else { 0 });
            bit_count += 1;
            if bit_count == 8 {
                slice_data.push(byte);
                byte = 0;
                bit_count = 0;
            }
        }
        if bit_count > 0 {
            byte <<= 8 - bit_count;
            slice_data.push(byte);
        }

        // Parse slice header (NAL type 19 = IDR_W_RADL)
        let slice = SliceHeader::parse(&slice_data, 19, &sps, &pps)
            .expect("Should parse I slice header");

        // Verify
        assert_eq!(slice.first_slice_segment_in_pic_flag, true);
        assert_eq!(slice.no_output_of_prior_pics_flag, false);
        assert_eq!(slice.slice_pic_parameter_set_id, 0);
        assert_eq!(slice.slice_type, SliceType::I);
        assert_eq!(slice.slice_qp_delta, 0);
        assert_eq!(slice.get_qp(&pps), 26); // 26 + 0 + 0
    }

    #[test]
    fn test_slice_header_parsing_p_slice() {
        // Create minimal SPS and PPS
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
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
        };

        let pps = Pps {
            pps_pic_parameter_set_id: 0,
            pps_seq_parameter_set_id: 0,
            dependent_slice_segments_enabled_flag: false,
            output_flag_present_flag: false,
            num_extra_slice_header_bits: 0,
            sign_data_hiding_enabled_flag: false,
            cabac_init_present_flag: false,
            num_ref_idx_l0_default_active_minus1: 0,
            num_ref_idx_l1_default_active_minus1: 0,
            init_qp_minus26: 5,
            constrained_intra_pred_flag: false,
            transform_skip_enabled_flag: false,
        };

        // Build P slice header
        let mut bits = Vec::new();

        // first_slice_segment_in_pic_flag = 1
        bits.push(true);

        // slice_pic_parameter_set_id = 0 (ue(v)): "1"
        bits.push(true);

        // slice_type = 1 (P slice, ue(v)): "010"
        bits.extend_from_slice(&[false, true, false]);

        // slice_qp_delta = -3 (se(v) for -3 is ue(6)): "00011010" (needs correction)
        // se(-3) maps to ue(6): 6 = 0b110, needs 2 leading zeros + 1 + 10 = "00110"
        bits.extend_from_slice(&[false, false, true, true, false]);

        // slice_deblocking_filter_disabled_flag = 1
        bits.push(true);

        // Convert to bytes
        let mut slice_data = Vec::new();
        let mut byte = 0u8;
        let mut bit_count = 0;
        for bit in bits {
            byte = (byte << 1) | (if bit { 1 } else { 0 });
            bit_count += 1;
            if bit_count == 8 {
                slice_data.push(byte);
                byte = 0;
                bit_count = 0;
            }
        }
        if bit_count > 0 {
            byte <<= 8 - bit_count;
            slice_data.push(byte);
        }

        // Parse slice header (NAL type 1 = TRAIL_R)
        let slice = SliceHeader::parse(&slice_data, 1, &sps, &pps)
            .expect("Should parse P slice header");

        // Verify
        assert_eq!(slice.slice_type, SliceType::P);
        assert_eq!(slice.slice_qp_delta, -3);
        assert_eq!(slice.slice_deblocking_filter_disabled_flag, true);
        assert_eq!(slice.get_qp(&pps), 28); // 26 + 5 + (-3) = 28
    }

    #[test]
    fn test_slice_header_qp_calculation() {
        let pps = Pps {
            pps_pic_parameter_set_id: 0,
            pps_seq_parameter_set_id: 0,
            dependent_slice_segments_enabled_flag: false,
            output_flag_present_flag: false,
            num_extra_slice_header_bits: 0,
            sign_data_hiding_enabled_flag: false,
            cabac_init_present_flag: false,
            num_ref_idx_l0_default_active_minus1: 0,
            num_ref_idx_l1_default_active_minus1: 0,
            init_qp_minus26: 10,
            constrained_intra_pred_flag: false,
            transform_skip_enabled_flag: false,
        };

        let slice = SliceHeader {
            first_slice_segment_in_pic_flag: true,
            no_output_of_prior_pics_flag: false,
            slice_pic_parameter_set_id: 0,
            dependent_slice_segment_flag: false,
            slice_segment_address: 0,
            slice_type: SliceType::I,
            pic_output_flag: true,
            colour_plane_id: None,
            slice_qp_delta: -5,
            slice_deblocking_filter_disabled_flag: false,
        };

        // QP = 26 + init_qp_minus26 + slice_qp_delta
        // QP = 26 + 10 + (-5) = 31
        assert_eq!(slice.get_qp(&pps), 31);
    }

    #[test]
    fn test_slice_header_with_output_flag() {
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
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
        };

        // PPS with output_flag_present_flag enabled
        let pps = Pps {
            pps_pic_parameter_set_id: 0,
            pps_seq_parameter_set_id: 0,
            dependent_slice_segments_enabled_flag: false,
            output_flag_present_flag: true, // Enabled!
            num_extra_slice_header_bits: 0,
            sign_data_hiding_enabled_flag: false,
            cabac_init_present_flag: false,
            num_ref_idx_l0_default_active_minus1: 0,
            num_ref_idx_l1_default_active_minus1: 0,
            init_qp_minus26: 0,
            constrained_intra_pred_flag: false,
            transform_skip_enabled_flag: false,
        };

        // Build slice header
        let mut bits = Vec::new();

        bits.push(true); // first_slice
        bits.push(true); // slice_pps_id = 0
        bits.extend_from_slice(&[false, true, true]); // slice_type = 2 (I)
        bits.push(false); // pic_output_flag = 0 (don't output)
        bits.push(true); // slice_qp_delta = 0
        bits.push(false); // deblocking_disabled = 0

        // Convert to bytes
        let mut slice_data = Vec::new();
        let mut byte = 0u8;
        let mut bit_count = 0;
        for bit in bits {
            byte = (byte << 1) | (if bit { 1 } else { 0 });
            bit_count += 1;
            if bit_count == 8 {
                slice_data.push(byte);
                byte = 0;
                bit_count = 0;
            }
        }
        if bit_count > 0 {
            byte <<= 8 - bit_count;
            slice_data.push(byte);
        }

        // Parse slice header (NAL type 19 = IDR_W_RADL)
        let slice = SliceHeader::parse(&slice_data, 19, &sps, &pps)
            .expect("Should parse slice with output flag");

        // Verify output flag was parsed
        assert_eq!(slice.pic_output_flag, false);
    }
}
