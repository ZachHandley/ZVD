//! VP8 frame header parsing
//!
//! This module handles parsing the VP8 frame header from the bitstream,
//! including both the uncompressed header and the compressed header data.

use super::bool_decoder::BoolDecoder;
use super::entropy::CoeffProbs;
use super::tables::{
    ColorSpace, FilterType, FrameType, IntraMode16x16, IntraMode4x4, SegmentFeatureMode,
    COEFF_PROBS_UPDATE_PROBS, DEFAULT_SEGMENT_PROBS, MV_PROB_DEFAULT, MV_UPDATE_PROBS,
};
use crate::error::{Error, Result};

/// VP8 frame header containing all parsed header information
#[derive(Clone)]
pub struct FrameHeader {
    // Uncompressed data chunk (3 bytes)
    pub frame_type: FrameType,
    pub version: u8,
    pub show_frame: bool,
    pub first_partition_size: u32,

    // Key frame specific (uncompressed)
    pub width: u16,
    pub horizontal_scale: u8,
    pub height: u16,
    pub vertical_scale: u8,

    // Color space and clamping (key frames only)
    pub color_space: ColorSpace,
    pub clamping_required: bool,

    // Segmentation
    pub segmentation_enabled: bool,
    pub update_mb_segmentation_map: bool,
    pub update_segment_feature_data: bool,
    pub segment_feature_mode: SegmentFeatureMode,
    pub segment_quant_update: [Option<i8>; 4],
    pub segment_lf_update: [Option<i8>; 4],
    pub mb_segment_tree_probs: [u8; 3],

    // Loop filter
    pub filter_type: FilterType,
    pub loop_filter_level: u8,
    pub sharpness_level: u8,
    pub loop_filter_adj_enable: bool,
    pub mode_ref_lf_delta_enabled: bool,
    pub mode_ref_lf_delta_update: bool,
    pub ref_lf_deltas: [i8; 4],
    pub mode_lf_deltas: [i8; 4],

    // Partitions
    pub num_token_partitions: u8,

    // Quantization
    pub y_ac_qi: u8,
    pub y_dc_delta: i8,
    pub y2_dc_delta: i8,
    pub y2_ac_delta: i8,
    pub uv_dc_delta: i8,
    pub uv_ac_delta: i8,

    // Reference frame management (inter frames)
    pub refresh_golden_frame: bool,
    pub refresh_altref_frame: bool,
    pub copy_buffer_to_golden: u8,
    pub copy_buffer_to_altref: u8,
    pub sign_bias_golden: bool,
    pub sign_bias_altref: bool,
    pub refresh_last: bool,
    pub refresh_entropy_probs: bool,

    // Probability updates
    pub coeff_prob_update: bool,
    pub mv_prob_update: bool,
}

impl Default for FrameHeader {
    fn default() -> Self {
        FrameHeader {
            frame_type: FrameType::KeyFrame,
            version: 0,
            show_frame: true,
            first_partition_size: 0,
            width: 0,
            horizontal_scale: 0,
            height: 0,
            vertical_scale: 0,
            color_space: ColorSpace::YCbCr,
            clamping_required: false,
            segmentation_enabled: false,
            update_mb_segmentation_map: false,
            update_segment_feature_data: false,
            segment_feature_mode: SegmentFeatureMode::Delta,
            segment_quant_update: [None; 4],
            segment_lf_update: [None; 4],
            mb_segment_tree_probs: DEFAULT_SEGMENT_PROBS,
            filter_type: FilterType::Normal,
            loop_filter_level: 0,
            sharpness_level: 0,
            loop_filter_adj_enable: false,
            mode_ref_lf_delta_enabled: false,
            mode_ref_lf_delta_update: false,
            ref_lf_deltas: [0; 4],
            mode_lf_deltas: [0; 4],
            num_token_partitions: 1,
            y_ac_qi: 0,
            y_dc_delta: 0,
            y2_dc_delta: 0,
            y2_ac_delta: 0,
            uv_dc_delta: 0,
            uv_ac_delta: 0,
            refresh_golden_frame: false,
            refresh_altref_frame: false,
            copy_buffer_to_golden: 0,
            copy_buffer_to_altref: 0,
            sign_bias_golden: false,
            sign_bias_altref: false,
            refresh_last: true,
            refresh_entropy_probs: false,
            coeff_prob_update: false,
            mv_prob_update: false,
        }
    }
}

impl FrameHeader {
    /// Parse a VP8 frame header from data
    ///
    /// # Returns
    /// Tuple of (header, offset to first partition data)
    pub fn parse(data: &[u8]) -> Result<(Self, usize)> {
        if data.len() < 3 {
            return Err(Error::codec("VP8 frame too small for header"));
        }

        // Parse uncompressed chunk (first 3 bytes)
        let frame_tag = (data[2] as u32) << 16 | (data[1] as u32) << 8 | data[0] as u32;

        let frame_type = if frame_tag & 1 == 0 {
            FrameType::KeyFrame
        } else {
            FrameType::InterFrame
        };
        let version = ((frame_tag >> 1) & 0x7) as u8;
        let show_frame = (frame_tag >> 4) & 1 != 0;
        let first_partition_size = frame_tag >> 5;

        let mut offset = 3;

        // Key frame has additional uncompressed data
        let (width, horizontal_scale, height, vertical_scale) = if frame_type == FrameType::KeyFrame
        {
            // Check signature (0x9D 0x01 0x2A)
            if data.len() < 10 || data[3] != 0x9D || data[4] != 0x01 || data[5] != 0x2A {
                return Err(Error::codec("Invalid VP8 key frame signature"));
            }

            let w = (data[7] as u16) << 8 | data[6] as u16;
            let h = (data[9] as u16) << 8 | data[8] as u16;

            let width = w & 0x3FFF;
            let horizontal_scale = (w >> 14) as u8;
            let height = h & 0x3FFF;
            let vertical_scale = (h >> 14) as u8;

            offset = 10;
            (width, horizontal_scale, height, vertical_scale)
        } else {
            (0, 0, 0, 0) // Inter frames use previous dimensions
        };

        // Check if we have enough data for the first partition
        if data.len() < offset + first_partition_size as usize {
            return Err(Error::codec("VP8 frame truncated"));
        }

        // Initialize header
        let mut header = FrameHeader {
            frame_type,
            version,
            show_frame,
            first_partition_size,
            width,
            horizontal_scale,
            height,
            vertical_scale,
            ..Default::default()
        };

        // Parse compressed header using boolean decoder
        let header_data = &data[offset..offset + first_partition_size as usize];
        let mut bd = BoolDecoder::new(header_data);

        header.parse_compressed_header(&mut bd)?;

        Ok((header, offset))
    }

    /// Parse the compressed portion of the header
    fn parse_compressed_header(&mut self, bd: &mut BoolDecoder) -> Result<()> {
        // Key frame specific
        if self.frame_type == FrameType::KeyFrame {
            self.color_space = if bd.read_bool(128) {
                ColorSpace::Reserved
            } else {
                ColorSpace::YCbCr
            };
            self.clamping_required = bd.read_bool(128);
        }

        // Parse segmentation
        self.segmentation_enabled = bd.read_bool(128);
        if self.segmentation_enabled {
            self.parse_segmentation(bd)?;
        }

        // Filter type and level
        self.filter_type = if bd.read_bool(128) {
            FilterType::Simple
        } else {
            FilterType::Normal
        };
        self.loop_filter_level = bd.read_literal(6) as u8;
        self.sharpness_level = bd.read_literal(3) as u8;

        // Loop filter adjustments
        self.parse_loop_filter_adjustments(bd)?;

        // Token partitions
        self.num_token_partitions = 1 << bd.read_literal(2);

        // Quantization parameters
        self.parse_quantization(bd)?;

        // Reference frame updates (inter frames only)
        if self.frame_type == FrameType::InterFrame {
            self.parse_inter_frame_refs(bd)?;
        } else {
            // Key frames always refresh all references
            self.refresh_golden_frame = true;
            self.refresh_altref_frame = true;
            self.refresh_last = true;
        }

        // Refresh entropy probs
        self.refresh_entropy_probs = bd.read_bool(128);

        Ok(())
    }

    /// Parse segmentation parameters
    fn parse_segmentation(&mut self, bd: &mut BoolDecoder) -> Result<()> {
        self.update_mb_segmentation_map = bd.read_bool(128);
        self.update_segment_feature_data = bd.read_bool(128);

        if self.update_segment_feature_data {
            self.segment_feature_mode = if bd.read_bool(128) {
                SegmentFeatureMode::Absolute
            } else {
                SegmentFeatureMode::Delta
            };

            // Quantizer updates
            for i in 0..4 {
                if bd.read_bool(128) {
                    let value = bd.read_literal(7) as i8;
                    let sign = if bd.read_bool(128) { -1 } else { 1 };
                    self.segment_quant_update[i] = Some(value * sign);
                }
            }

            // Loop filter updates
            for i in 0..4 {
                if bd.read_bool(128) {
                    let value = bd.read_literal(6) as i8;
                    let sign = if bd.read_bool(128) { -1 } else { 1 };
                    self.segment_lf_update[i] = Some(value * sign);
                }
            }
        }

        // Segment tree probabilities
        if self.update_mb_segmentation_map {
            for i in 0..3 {
                if bd.read_bool(128) {
                    self.mb_segment_tree_probs[i] = bd.read_literal(8) as u8;
                } else {
                    self.mb_segment_tree_probs[i] = 255;
                }
            }
        }

        Ok(())
    }

    /// Parse loop filter adjustments
    fn parse_loop_filter_adjustments(&mut self, bd: &mut BoolDecoder) -> Result<()> {
        self.loop_filter_adj_enable = bd.read_bool(128);
        if self.loop_filter_adj_enable {
            self.mode_ref_lf_delta_enabled = bd.read_bool(128);
            if self.mode_ref_lf_delta_enabled {
                self.mode_ref_lf_delta_update = bd.read_bool(128);
                if self.mode_ref_lf_delta_update {
                    // Reference frame deltas
                    for i in 0..4 {
                        if bd.read_bool(128) {
                            let value = bd.read_literal(6) as i8;
                            self.ref_lf_deltas[i] = if bd.read_bool(128) { -value } else { value };
                        }
                    }
                    // Mode deltas
                    for i in 0..4 {
                        if bd.read_bool(128) {
                            let value = bd.read_literal(6) as i8;
                            self.mode_lf_deltas[i] = if bd.read_bool(128) { -value } else { value };
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Parse quantization parameters
    fn parse_quantization(&mut self, bd: &mut BoolDecoder) -> Result<()> {
        self.y_ac_qi = bd.read_literal(7) as u8;

        self.y_dc_delta = if bd.read_bool(128) {
            let val = bd.read_literal(4) as i8;
            if bd.read_bool(128) {
                -val
            } else {
                val
            }
        } else {
            0
        };

        self.y2_dc_delta = if bd.read_bool(128) {
            let val = bd.read_literal(4) as i8;
            if bd.read_bool(128) {
                -val
            } else {
                val
            }
        } else {
            0
        };

        self.y2_ac_delta = if bd.read_bool(128) {
            let val = bd.read_literal(4) as i8;
            if bd.read_bool(128) {
                -val
            } else {
                val
            }
        } else {
            0
        };

        self.uv_dc_delta = if bd.read_bool(128) {
            let val = bd.read_literal(4) as i8;
            if bd.read_bool(128) {
                -val
            } else {
                val
            }
        } else {
            0
        };

        self.uv_ac_delta = if bd.read_bool(128) {
            let val = bd.read_literal(4) as i8;
            if bd.read_bool(128) {
                -val
            } else {
                val
            }
        } else {
            0
        };

        Ok(())
    }

    /// Parse inter frame reference updates
    fn parse_inter_frame_refs(&mut self, bd: &mut BoolDecoder) -> Result<()> {
        self.refresh_golden_frame = bd.read_bool(128);
        self.refresh_altref_frame = bd.read_bool(128);

        if !self.refresh_golden_frame {
            self.copy_buffer_to_golden = bd.read_literal(2) as u8;
        }

        if !self.refresh_altref_frame {
            self.copy_buffer_to_altref = bd.read_literal(2) as u8;
        }

        self.sign_bias_golden = bd.read_bool(128);
        self.sign_bias_altref = bd.read_bool(128);
        self.refresh_last = bd.read_bool(128);

        Ok(())
    }

    /// Check if this is a keyframe
    pub fn is_keyframe(&self) -> bool {
        self.frame_type == FrameType::KeyFrame
    }

    /// Get frame dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width as u32, self.height as u32)
    }

    /// Get number of macroblocks
    pub fn macroblock_dimensions(&self) -> (u32, u32) {
        let mb_width = (self.width as u32 + 15) >> 4;
        let mb_height = (self.height as u32 + 15) >> 4;
        (mb_width, mb_height)
    }
}

/// Parse coefficient probability updates from the bitstream
pub fn parse_coeff_prob_updates(bd: &mut BoolDecoder, probs: &mut CoeffProbs) {
    probs.update(bd, &COEFF_PROBS_UPDATE_PROBS);
}

/// Motion vector probabilities
#[derive(Clone)]
pub struct MvProbs {
    pub probs: [[u8; 19]; 2],
}

impl Default for MvProbs {
    fn default() -> Self {
        MvProbs {
            probs: MV_PROB_DEFAULT,
        }
    }
}

impl MvProbs {
    /// Update MV probabilities from bitstream
    pub fn update(&mut self, bd: &mut BoolDecoder) {
        for comp in 0..2 {
            for i in 0..19 {
                if bd.read_bool(MV_UPDATE_PROBS[comp][i]) {
                    self.probs[comp][i] = bd.read_literal(7) as u8;
                }
            }
        }
    }
}

/// Mode probabilities for intra prediction
#[derive(Clone)]
pub struct ModeProbs {
    pub y_mode_probs: [u8; 4],
    pub uv_mode_probs: [u8; 3],
    pub bmode_probs: [[[u8; 9]; 10]; 10],
}

impl Default for ModeProbs {
    fn default() -> Self {
        use super::tables::{DEFAULT_INTRA_16X16_PROBS, KF_BMODE_PROBS, KF_UVMODE_PROBS};
        ModeProbs {
            y_mode_probs: [
                DEFAULT_INTRA_16X16_PROBS[0],
                DEFAULT_INTRA_16X16_PROBS[1],
                DEFAULT_INTRA_16X16_PROBS[2],
                128,
            ],
            uv_mode_probs: KF_UVMODE_PROBS,
            bmode_probs: KF_BMODE_PROBS,
        }
    }
}

/// Decode 16x16 intra mode using key frame probabilities
pub fn decode_kf_y_mode(bd: &mut BoolDecoder) -> IntraMode16x16 {
    use super::tables::KF_YMODE_PROBS;

    if !bd.read_bool(KF_YMODE_PROBS[0]) {
        IntraMode16x16::DcPred
    } else if !bd.read_bool(KF_YMODE_PROBS[1]) {
        IntraMode16x16::VPred
    } else if !bd.read_bool(KF_YMODE_PROBS[2]) {
        IntraMode16x16::HPred
    } else {
        IntraMode16x16::TmPred
    }
}

/// Decode chroma intra mode using key frame probabilities
pub fn decode_kf_uv_mode(bd: &mut BoolDecoder) -> IntraMode16x16 {
    use super::tables::KF_UVMODE_PROBS;

    if !bd.read_bool(KF_UVMODE_PROBS[0]) {
        IntraMode16x16::DcPred
    } else if !bd.read_bool(KF_UVMODE_PROBS[1]) {
        IntraMode16x16::VPred
    } else if !bd.read_bool(KF_UVMODE_PROBS[2]) {
        IntraMode16x16::HPred
    } else {
        IntraMode16x16::TmPred
    }
}

/// Decode 4x4 intra mode using key frame probabilities
pub fn decode_kf_b_mode(
    bd: &mut BoolDecoder,
    above_mode: IntraMode4x4,
    left_mode: IntraMode4x4,
) -> IntraMode4x4 {
    use super::tables::KF_BMODE_PROBS;

    let probs = &KF_BMODE_PROBS[above_mode as usize][left_mode as usize];

    if !bd.read_bool(probs[0]) {
        IntraMode4x4::BDcPred
    } else if !bd.read_bool(probs[1]) {
        IntraMode4x4::BTmPred
    } else if !bd.read_bool(probs[2]) {
        IntraMode4x4::BVePred
    } else if !bd.read_bool(probs[3]) {
        if !bd.read_bool(probs[4]) {
            IntraMode4x4::BHePred
        } else {
            IntraMode4x4::BLdPred
        }
    } else if !bd.read_bool(probs[5]) {
        IntraMode4x4::BRdPred
    } else if !bd.read_bool(probs[6]) {
        IntraMode4x4::BVrPred
    } else if !bd.read_bool(probs[7]) {
        IntraMode4x4::BVlPred
    } else if !bd.read_bool(probs[8]) {
        IntraMode4x4::BHdPred
    } else {
        IntraMode4x4::BHuPred
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_header_default() {
        let header = FrameHeader::default();
        assert_eq!(header.frame_type, FrameType::KeyFrame);
        assert!(header.show_frame);
    }

    #[test]
    fn test_macroblock_dimensions() {
        let mut header = FrameHeader::default();
        header.width = 640;
        header.height = 480;

        let (mb_w, mb_h) = header.macroblock_dimensions();
        assert_eq!(mb_w, 40); // 640 / 16
        assert_eq!(mb_h, 30); // 480 / 16
    }

    #[test]
    fn test_macroblock_dimensions_not_aligned() {
        let mut header = FrameHeader::default();
        header.width = 641;
        header.height = 481;

        let (mb_w, mb_h) = header.macroblock_dimensions();
        assert_eq!(mb_w, 41); // ceil(641 / 16)
        assert_eq!(mb_h, 31); // ceil(481 / 16)
    }

    #[test]
    fn test_mv_probs_default() {
        let probs = MvProbs::default();
        assert_eq!(probs.probs[0][0], 162);
        assert_eq!(probs.probs[1][0], 164);
    }
}
