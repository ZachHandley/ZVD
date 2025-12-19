//! VP9 Frame Header Parsing
//!
//! This module handles parsing of VP9 frame headers, including:
//! - Uncompressed header (bit-packed)
//! - Compressed header (range coded)
//! - Tile information

use crate::error::{Error, Result};

use super::range_coder::BitReader;
use super::tables::{
    ColorSpace, InterpFilter, Profile, DEFAULT_COMP_INTER_PROB, DEFAULT_COMP_REF_PROB,
    DEFAULT_INTER_MODE_PROBS, DEFAULT_INTRA_INTER_PROB, DEFAULT_KF_UV_MODE_PROBS,
    DEFAULT_KF_Y_MODE_PROBS, DEFAULT_PARTITION_PROBS, DEFAULT_SINGLE_REF_PROB, DEFAULT_SKIP_PROBS,
};

/// Frame types in VP9
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    KeyFrame = 0,
    InterFrame = 1,
}

/// Loop filter parameters
#[derive(Debug, Clone, Default)]
pub struct LoopFilterParams {
    /// Filter level (0-63)
    pub level: u8,
    /// Sharpness level (0-7)
    pub sharpness: u8,
    /// Delta values enabled
    pub delta_enabled: bool,
    /// Delta update in this frame
    pub delta_update: bool,
    /// Reference frame deltas (INTRA, LAST, GOLDEN, ALTREF)
    pub ref_deltas: [i8; 4],
    /// Mode deltas (0=zero_mv, 1=non_zero_mv)
    pub mode_deltas: [i8; 2],
}

/// Quantization parameters
#[derive(Debug, Clone, Default)]
pub struct QuantizationParams {
    /// Base quantizer index (0-255)
    pub base_q_idx: u8,
    /// Y DC delta
    pub y_dc_delta: i8,
    /// UV DC delta
    pub uv_dc_delta: i8,
    /// UV AC delta
    pub uv_ac_delta: i8,
    /// Lossless mode (all deltas zero and base_q_idx == 0)
    pub lossless: bool,
}

impl QuantizationParams {
    pub fn is_lossless(&self) -> bool {
        self.base_q_idx == 0
            && self.y_dc_delta == 0
            && self.uv_dc_delta == 0
            && self.uv_ac_delta == 0
    }
}

/// Segmentation parameters
#[derive(Debug, Clone, Default)]
pub struct SegmentationParams {
    /// Segmentation enabled
    pub enabled: bool,
    /// Update map this frame
    pub update_map: bool,
    /// Temporal update enabled
    pub temporal_update: bool,
    /// Update data this frame
    pub update_data: bool,
    /// Absolute or delta values
    pub abs_or_delta_update: bool,
    /// Segment feature data
    pub features: [[SegmentFeature; 4]; 8], // 8 segments, 4 features each
    /// Segment prediction probabilities
    pub pred_probs: [u8; 3],
    /// Tree probabilities
    pub tree_probs: [u8; 7],
}

/// Segment feature types
#[derive(Debug, Clone, Copy, Default)]
pub struct SegmentFeature {
    pub enabled: bool,
    pub value: i16,
}

/// Tile configuration
#[derive(Debug, Clone)]
pub struct TileInfo {
    /// Log2 of tile columns
    pub cols_log2: u8,
    /// Log2 of tile rows
    pub rows_log2: u8,
    /// Number of tile columns
    pub num_cols: usize,
    /// Number of tile rows
    pub num_rows: usize,
}

impl Default for TileInfo {
    fn default() -> Self {
        TileInfo {
            cols_log2: 0,
            rows_log2: 0,
            num_cols: 1,
            num_rows: 1,
        }
    }
}

/// VP9 Uncompressed Frame Header
#[derive(Debug, Clone)]
pub struct FrameHeader {
    /// Frame marker (should be 2)
    pub frame_marker: u8,
    /// Profile (0-3)
    pub profile: Profile,
    /// Show existing frame flag
    pub show_existing_frame: bool,
    /// Frame to show map index (if show_existing_frame)
    pub frame_to_show_map_idx: u8,
    /// Frame type
    pub frame_type: FrameType,
    /// Show frame
    pub show_frame: bool,
    /// Error resilient mode
    pub error_resilient_mode: bool,

    // Key frame specific
    /// Bit depth (8, 10, or 12)
    pub bit_depth: u8,
    /// Color space
    pub color_space: ColorSpace,
    /// Color range (false = studio, true = full)
    pub color_range: bool,
    /// Subsampling X (false = 4:4:4, true = 4:2:x)
    pub subsampling_x: bool,
    /// Subsampling Y (false = 4:2:2 or 4:4:4, true = 4:2:0)
    pub subsampling_y: bool,

    /// Frame width in pixels
    pub width: u32,
    /// Frame height in pixels
    pub height: u32,
    /// Render width
    pub render_width: u32,
    /// Render height
    pub render_height: u32,

    // Inter frame specific
    /// Intra only flag (for non-key frames)
    pub intra_only: bool,
    /// Reset frame context
    pub reset_frame_context: u8,
    /// Refresh frame flags (bitmask for 8 slots)
    pub refresh_frame_flags: u8,
    /// Reference frame indices
    pub ref_frame_idx: [u8; 3],
    /// Reference frame sign bias
    pub ref_frame_sign_bias: [bool; 4],
    /// Allow high precision motion vectors
    pub allow_high_precision_mv: bool,
    /// Interpolation filter
    pub interp_filter: InterpFilter,

    /// Refresh frame context
    pub refresh_frame_context: bool,
    /// Frame parallel decoding mode
    pub frame_parallel_decoding_mode: bool,
    /// Frame context index
    pub frame_context_idx: u8,

    /// Loop filter parameters
    pub loop_filter: LoopFilterParams,

    /// Quantization parameters
    pub quantization: QuantizationParams,

    /// Segmentation parameters
    pub segmentation: SegmentationParams,

    /// Tile info
    pub tile_info: TileInfo,

    /// Compressed header size in bytes
    pub header_size: u16,

    /// Total uncompressed header size consumed
    pub uncompressed_header_size: usize,
}

impl Default for FrameHeader {
    fn default() -> Self {
        FrameHeader {
            frame_marker: 2,
            profile: Profile::Profile0,
            show_existing_frame: false,
            frame_to_show_map_idx: 0,
            frame_type: FrameType::KeyFrame,
            show_frame: true,
            error_resilient_mode: false,
            bit_depth: 8,
            color_space: ColorSpace::Unknown,
            color_range: false,
            subsampling_x: true,
            subsampling_y: true,
            width: 0,
            height: 0,
            render_width: 0,
            render_height: 0,
            intra_only: false,
            reset_frame_context: 0,
            refresh_frame_flags: 0,
            ref_frame_idx: [0; 3],
            ref_frame_sign_bias: [false; 4],
            allow_high_precision_mv: false,
            interp_filter: InterpFilter::EightTap,
            refresh_frame_context: true,
            frame_parallel_decoding_mode: false,
            frame_context_idx: 0,
            loop_filter: LoopFilterParams::default(),
            quantization: QuantizationParams::default(),
            segmentation: SegmentationParams::default(),
            tile_info: TileInfo::default(),
            header_size: 0,
            uncompressed_header_size: 0,
        }
    }
}

impl FrameHeader {
    /// Parse VP9 frame header from data
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 3 {
            return Err(Error::codec("VP9 frame too short"));
        }

        let mut reader = BitReader::new(data);
        let mut header = FrameHeader::default();

        // Frame marker (2 bits, must be 2)
        header.frame_marker = reader
            .read_bits(2)
            .ok_or_else(|| Error::codec("Cannot read frame marker"))?
            as u8;
        if header.frame_marker != 2 {
            return Err(Error::codec(format!(
                "Invalid VP9 frame marker: {} (expected 2)",
                header.frame_marker
            )));
        }

        // Profile (2 bits, but encoded specially)
        let profile_low = reader
            .read_bit()
            .ok_or_else(|| Error::codec("Cannot read profile"))? as u8;
        let profile_high = reader
            .read_bit()
            .ok_or_else(|| Error::codec("Cannot read profile"))? as u8;
        let profile = (profile_high << 1) | profile_low;
        header.profile = Profile::from_u8(profile);

        // Reserved bit for profile 3
        if profile == 3 {
            let reserved = reader
                .read_bit()
                .ok_or_else(|| Error::codec("Cannot read reserved bit"))?;
            if reserved {
                return Err(Error::codec("Invalid VP9 profile reserved bit"));
            }
        }

        // Show existing frame
        header.show_existing_frame = reader
            .read_bit()
            .ok_or_else(|| Error::codec("Cannot read show_existing_frame"))?;
        if header.show_existing_frame {
            header.frame_to_show_map_idx = reader
                .read_bits(3)
                .ok_or_else(|| Error::codec("Cannot read frame_to_show_map_idx"))?
                as u8;
            header.uncompressed_header_size = (reader.bit_position() + 7) / 8;
            return Ok(header);
        }

        // Frame type
        header.frame_type = if reader
            .read_bit()
            .ok_or_else(|| Error::codec("Cannot read frame_type"))?
        {
            FrameType::InterFrame
        } else {
            FrameType::KeyFrame
        };

        header.show_frame = reader
            .read_bit()
            .ok_or_else(|| Error::codec("Cannot read show_frame"))?;
        header.error_resilient_mode = reader
            .read_bit()
            .ok_or_else(|| Error::codec("Cannot read error_resilient_mode"))?;

        if header.frame_type == FrameType::KeyFrame {
            // Key frame sync code
            Self::parse_sync_code(&mut reader)?;

            // Color config
            Self::parse_color_config(&mut reader, &mut header)?;

            // Frame size
            Self::parse_frame_size(&mut reader, &mut header)?;

            // Render size
            Self::parse_render_size(&mut reader, &mut header)?;
        } else {
            // Inter frame
            header.intra_only = if !header.show_frame {
                reader
                    .read_bit()
                    .ok_or_else(|| Error::codec("Cannot read intra_only"))?
            } else {
                false
            };

            if !header.error_resilient_mode {
                header.reset_frame_context = reader
                    .read_bits(2)
                    .ok_or_else(|| Error::codec("Cannot read reset_frame_context"))?
                    as u8;
            }

            if header.intra_only {
                // Sync code for intra-only frames
                Self::parse_sync_code(&mut reader)?;

                // Color config for profiles > 0
                if header.profile != Profile::Profile0 {
                    Self::parse_color_config(&mut reader, &mut header)?;
                } else {
                    header.bit_depth = 8;
                    header.color_space = ColorSpace::Bt601;
                    header.subsampling_x = true;
                    header.subsampling_y = true;
                }

                // Refresh frame flags
                header.refresh_frame_flags = reader
                    .read_bits(8)
                    .ok_or_else(|| Error::codec("Cannot read refresh_frame_flags"))?
                    as u8;

                // Frame size
                Self::parse_frame_size(&mut reader, &mut header)?;
                Self::parse_render_size(&mut reader, &mut header)?;
            } else {
                // Normal inter frame
                header.refresh_frame_flags = reader
                    .read_bits(8)
                    .ok_or_else(|| Error::codec("Cannot read refresh_frame_flags"))?
                    as u8;

                // Reference frame indices
                for i in 0..3 {
                    header.ref_frame_idx[i] = reader
                        .read_bits(3)
                        .ok_or_else(|| Error::codec("Cannot read ref_frame_idx"))?
                        as u8;
                    header.ref_frame_sign_bias[i + 1] = reader
                        .read_bit()
                        .ok_or_else(|| Error::codec("Cannot read ref_frame_sign_bias"))?;
                }

                // Frame size with refs
                let use_ref_frame_size = reader
                    .read_bit()
                    .ok_or_else(|| Error::codec("Cannot read frame_size_from_refs"))?;
                if use_ref_frame_size {
                    // Size comes from reference frame - for now, error
                    return Err(Error::codec("Frame size from refs not yet supported"));
                } else {
                    Self::parse_frame_size(&mut reader, &mut header)?;
                }
                Self::parse_render_size(&mut reader, &mut header)?;

                // High precision MV
                header.allow_high_precision_mv = reader
                    .read_bit()
                    .ok_or_else(|| Error::codec("Cannot read allow_high_precision_mv"))?;

                // Interpolation filter
                Self::parse_interp_filter(&mut reader, &mut header)?;
            }
        }

        // Refresh frame context
        if !header.error_resilient_mode {
            header.refresh_frame_context = reader
                .read_bit()
                .ok_or_else(|| Error::codec("Cannot read refresh_frame_context"))?;
            header.frame_parallel_decoding_mode = reader
                .read_bit()
                .ok_or_else(|| Error::codec("Cannot read frame_parallel_decoding_mode"))?;
        } else {
            header.refresh_frame_context = false;
            header.frame_parallel_decoding_mode = true;
        }

        // Frame context index
        header.frame_context_idx = reader
            .read_bits(2)
            .ok_or_else(|| Error::codec("Cannot read frame_context_idx"))?
            as u8;

        // Loop filter params
        Self::parse_loop_filter(&mut reader, &mut header)?;

        // Quantization params
        Self::parse_quantization(&mut reader, &mut header)?;

        // Segmentation params
        Self::parse_segmentation(&mut reader, &mut header)?;

        // Tile info
        Self::parse_tile_info(&mut reader, &mut header)?;

        // Compressed header size
        header.header_size = reader
            .read_bits(16)
            .ok_or_else(|| Error::codec("Cannot read header_size"))?
            as u16;

        // Byte-align
        reader.byte_align();
        header.uncompressed_header_size = reader.position();

        Ok(header)
    }

    fn parse_sync_code(reader: &mut BitReader) -> Result<()> {
        let sync = reader
            .read_bits(24)
            .ok_or_else(|| Error::codec("Cannot read sync code"))?;
        if sync != 0x498342 {
            return Err(Error::codec(format!(
                "Invalid VP9 sync code: 0x{:06X} (expected 0x498342)",
                sync
            )));
        }
        Ok(())
    }

    fn parse_color_config(reader: &mut BitReader, header: &mut FrameHeader) -> Result<()> {
        // Bit depth
        if header.profile == Profile::Profile2 || header.profile == Profile::Profile3 {
            let ten_or_twelve = reader
                .read_bit()
                .ok_or_else(|| Error::codec("Cannot read bit depth flag"))?;
            header.bit_depth = if ten_or_twelve { 12 } else { 10 };
        } else {
            header.bit_depth = 8;
        }

        // Color space
        header.color_space = ColorSpace::from_u8(
            reader
                .read_bits(3)
                .ok_or_else(|| Error::codec("Cannot read color space"))? as u8,
        );

        if header.color_space != ColorSpace::Srgb {
            // Color range
            header.color_range = reader
                .read_bit()
                .ok_or_else(|| Error::codec("Cannot read color range"))?;

            // Subsampling
            if header.profile == Profile::Profile1 || header.profile == Profile::Profile3 {
                header.subsampling_x = reader
                    .read_bit()
                    .ok_or_else(|| Error::codec("Cannot read subsampling_x"))?;
                header.subsampling_y = reader
                    .read_bit()
                    .ok_or_else(|| Error::codec("Cannot read subsampling_y"))?;
                let reserved = reader
                    .read_bit()
                    .ok_or_else(|| Error::codec("Cannot read reserved bit"))?;
                if reserved {
                    return Err(Error::codec("Reserved bit set in color config"));
                }
            } else {
                header.subsampling_x = true;
                header.subsampling_y = true;
            }
        } else {
            header.color_range = true;
            if header.profile == Profile::Profile1 || header.profile == Profile::Profile3 {
                header.subsampling_x = false;
                header.subsampling_y = false;
                let reserved = reader
                    .read_bit()
                    .ok_or_else(|| Error::codec("Cannot read reserved bit"))?;
                if reserved {
                    return Err(Error::codec("Reserved bit set in color config"));
                }
            } else {
                return Err(Error::codec("sRGB not allowed in profile 0 or 2"));
            }
        }

        Ok(())
    }

    fn parse_frame_size(reader: &mut BitReader, header: &mut FrameHeader) -> Result<()> {
        // Width minus 1 (16 bits)
        header.width = reader
            .read_bits(16)
            .ok_or_else(|| Error::codec("Cannot read width"))?
            + 1;
        // Height minus 1 (16 bits)
        header.height = reader
            .read_bits(16)
            .ok_or_else(|| Error::codec("Cannot read height"))?
            + 1;
        Ok(())
    }

    fn parse_render_size(reader: &mut BitReader, header: &mut FrameHeader) -> Result<()> {
        let render_and_frame_size_different = reader
            .read_bit()
            .ok_or_else(|| Error::codec("Cannot read render_and_frame_size_different"))?;
        if render_and_frame_size_different {
            header.render_width = reader
                .read_bits(16)
                .ok_or_else(|| Error::codec("Cannot read render_width"))?
                + 1;
            header.render_height = reader
                .read_bits(16)
                .ok_or_else(|| Error::codec("Cannot read render_height"))?
                + 1;
        } else {
            header.render_width = header.width;
            header.render_height = header.height;
        }
        Ok(())
    }

    fn parse_interp_filter(reader: &mut BitReader, header: &mut FrameHeader) -> Result<()> {
        let is_filter_switchable = reader
            .read_bit()
            .ok_or_else(|| Error::codec("Cannot read is_filter_switchable"))?;
        if is_filter_switchable {
            header.interp_filter = InterpFilter::Switchable;
        } else {
            let raw_filter = reader
                .read_bits(2)
                .ok_or_else(|| Error::codec("Cannot read interp_filter"))?
                as u8;
            header.interp_filter = match raw_filter {
                0 => InterpFilter::EightTapSmooth,
                1 => InterpFilter::EightTap,
                2 => InterpFilter::EightTapSharp,
                3 => InterpFilter::Bilinear,
                _ => InterpFilter::EightTap,
            };
        }
        Ok(())
    }

    fn parse_loop_filter(reader: &mut BitReader, header: &mut FrameHeader) -> Result<()> {
        header.loop_filter.level = reader
            .read_bits(6)
            .ok_or_else(|| Error::codec("Cannot read loop_filter level"))?
            as u8;
        header.loop_filter.sharpness = reader
            .read_bits(3)
            .ok_or_else(|| Error::codec("Cannot read loop_filter sharpness"))?
            as u8;

        header.loop_filter.delta_enabled = reader
            .read_bit()
            .ok_or_else(|| Error::codec("Cannot read loop_filter delta_enabled"))?;
        if header.loop_filter.delta_enabled {
            header.loop_filter.delta_update = reader
                .read_bit()
                .ok_or_else(|| Error::codec("Cannot read loop_filter delta_update"))?;
            if header.loop_filter.delta_update {
                // Read ref deltas
                for i in 0..4 {
                    if reader
                        .read_bit()
                        .ok_or_else(|| Error::codec("Cannot read ref_delta update flag"))?
                    {
                        header.loop_filter.ref_deltas[i] = reader
                            .read_signed_bits(6)
                            .ok_or_else(|| Error::codec("Cannot read ref_delta"))?
                            as i8;
                    }
                }
                // Read mode deltas
                for i in 0..2 {
                    if reader
                        .read_bit()
                        .ok_or_else(|| Error::codec("Cannot read mode_delta update flag"))?
                    {
                        header.loop_filter.mode_deltas[i] = reader
                            .read_signed_bits(6)
                            .ok_or_else(|| Error::codec("Cannot read mode_delta"))?
                            as i8;
                    }
                }
            }
        }
        Ok(())
    }

    fn parse_quantization(reader: &mut BitReader, header: &mut FrameHeader) -> Result<()> {
        header.quantization.base_q_idx = reader
            .read_bits(8)
            .ok_or_else(|| Error::codec("Cannot read base_q_idx"))?
            as u8;

        // Read delta_coded values
        fn read_delta_q(reader: &mut BitReader) -> Result<i8> {
            if reader
                .read_bit()
                .ok_or_else(|| Error::codec("Cannot read delta_q present"))?
            {
                let magnitude = reader
                    .read_bits(4)
                    .ok_or_else(|| Error::codec("Cannot read delta_q magnitude"))?
                    as i8;
                let sign = reader
                    .read_bit()
                    .ok_or_else(|| Error::codec("Cannot read delta_q sign"))?;
                Ok(if sign { -magnitude } else { magnitude })
            } else {
                Ok(0)
            }
        }

        header.quantization.y_dc_delta = read_delta_q(reader)?;
        header.quantization.uv_dc_delta = read_delta_q(reader)?;
        header.quantization.uv_ac_delta = read_delta_q(reader)?;
        header.quantization.lossless = header.quantization.is_lossless();

        Ok(())
    }

    fn parse_segmentation(reader: &mut BitReader, header: &mut FrameHeader) -> Result<()> {
        header.segmentation.enabled = reader
            .read_bit()
            .ok_or_else(|| Error::codec("Cannot read segmentation enabled"))?;

        if !header.segmentation.enabled {
            return Ok(());
        }

        // Update map
        header.segmentation.update_map = reader
            .read_bit()
            .ok_or_else(|| Error::codec("Cannot read segmentation update_map"))?;
        if header.segmentation.update_map {
            // Tree probs
            for i in 0..7 {
                if reader
                    .read_bit()
                    .ok_or_else(|| Error::codec("Cannot read tree_prob update"))?
                {
                    header.segmentation.tree_probs[i] = reader
                        .read_bits(8)
                        .ok_or_else(|| Error::codec("Cannot read tree_prob"))?
                        as u8;
                } else {
                    header.segmentation.tree_probs[i] = 255;
                }
            }

            // Temporal update
            header.segmentation.temporal_update = reader
                .read_bit()
                .ok_or_else(|| Error::codec("Cannot read temporal_update"))?;
            if header.segmentation.temporal_update {
                for i in 0..3 {
                    if reader
                        .read_bit()
                        .ok_or_else(|| Error::codec("Cannot read pred_prob update"))?
                    {
                        header.segmentation.pred_probs[i] = reader
                            .read_bits(8)
                            .ok_or_else(|| Error::codec("Cannot read pred_prob"))?
                            as u8;
                    } else {
                        header.segmentation.pred_probs[i] = 255;
                    }
                }
            }
        }

        // Update data
        header.segmentation.update_data = reader
            .read_bit()
            .ok_or_else(|| Error::codec("Cannot read segmentation update_data"))?;
        if header.segmentation.update_data {
            header.segmentation.abs_or_delta_update = reader
                .read_bit()
                .ok_or_else(|| Error::codec("Cannot read abs_or_delta_update"))?;

            // Segment features: 0=alt_q, 1=alt_lf, 2=ref_frame, 3=skip
            let feature_bits = [8, 6, 2, 0]; // bits for each feature
            let feature_signed = [true, true, false, false];

            for seg_id in 0..8 {
                for feat_id in 0..4 {
                    let enabled = reader
                        .read_bit()
                        .ok_or_else(|| Error::codec("Cannot read feature enabled"))?;
                    header.segmentation.features[seg_id][feat_id].enabled = enabled;
                    if enabled && feature_bits[feat_id] > 0 {
                        let value = reader
                            .read_bits(feature_bits[feat_id])
                            .ok_or_else(|| Error::codec("Cannot read feature value"))?
                            as i16;
                        if feature_signed[feat_id] {
                            let sign = reader
                                .read_bit()
                                .ok_or_else(|| Error::codec("Cannot read feature sign"))?;
                            header.segmentation.features[seg_id][feat_id].value =
                                if sign { -value } else { value };
                        } else {
                            header.segmentation.features[seg_id][feat_id].value = value;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn parse_tile_info(reader: &mut BitReader, header: &mut FrameHeader) -> Result<()> {
        // Calculate min/max tile columns based on frame width
        let sb_cols = (header.width + 63) / 64;
        let min_log2 = Self::tile_log2(64, sb_cols as usize);
        let max_log2 = Self::tile_log2(1, (sb_cols as usize).min(64));

        header.tile_info.cols_log2 = min_log2 as u8;
        while header.tile_info.cols_log2 < max_log2 as u8 {
            if reader
                .read_bit()
                .ok_or_else(|| Error::codec("Cannot read tile_cols increment"))?
            {
                header.tile_info.cols_log2 += 1;
            } else {
                break;
            }
        }

        header.tile_info.rows_log2 = if reader
            .read_bit()
            .ok_or_else(|| Error::codec("Cannot read tile_rows_log2 bit"))?
        {
            1 + if reader
                .read_bit()
                .ok_or_else(|| Error::codec("Cannot read tile_rows_log2 bit"))?
            {
                1
            } else {
                0
            }
        } else {
            0
        };

        header.tile_info.num_cols = 1 << header.tile_info.cols_log2;
        header.tile_info.num_rows = 1 << header.tile_info.rows_log2;

        Ok(())
    }

    /// Calculate log2 of tile count
    fn tile_log2(blk_size: usize, target: usize) -> usize {
        let mut k = 0;
        while (blk_size << k) < target {
            k += 1;
        }
        k
    }

    /// Check if this is a key frame
    pub fn is_keyframe(&self) -> bool {
        self.frame_type == FrameType::KeyFrame
    }

    /// Check if this is an intra-only frame
    pub fn is_intra_only(&self) -> bool {
        self.frame_type == FrameType::KeyFrame || self.intra_only
    }

    /// Get dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Get number of superblocks in width
    pub fn sb_cols(&self) -> usize {
        ((self.width + 63) / 64) as usize
    }

    /// Get number of superblocks in height
    pub fn sb_rows(&self) -> usize {
        ((self.height + 63) / 64) as usize
    }

    /// Get number of mode info blocks (4x4) in width
    pub fn mi_cols(&self) -> usize {
        ((self.width + 7) / 8) as usize
    }

    /// Get number of mode info blocks (4x4) in height
    pub fn mi_rows(&self) -> usize {
        ((self.height + 7) / 8) as usize
    }
}

/// Frame probability context
/// Contains all probability tables that can be updated per-frame
#[derive(Clone)]
pub struct FrameContext {
    /// Partition probabilities [context][partition]
    pub partition_probs: [[u8; 3]; 16],
    /// Skip probabilities
    pub skip_probs: [u8; 3],
    /// Intra/inter probabilities
    pub intra_inter_probs: [u8; 4],
    /// Compound inter probabilities
    pub comp_inter_probs: [u8; 5],
    /// Single reference probabilities
    pub single_ref_probs: [[u8; 2]; 5],
    /// Compound reference probabilities
    pub comp_ref_probs: [u8; 5],
    /// Inter mode probabilities
    pub inter_mode_probs: [[u8; 3]; 7],
    /// Y mode probabilities for keyframes
    pub kf_y_mode_probs: [[u8; 9]; 10],
    /// UV mode probabilities for keyframes
    pub kf_uv_mode_probs: [[u8; 9]; 10],
}

impl Default for FrameContext {
    fn default() -> Self {
        FrameContext {
            partition_probs: DEFAULT_PARTITION_PROBS,
            skip_probs: DEFAULT_SKIP_PROBS,
            intra_inter_probs: DEFAULT_INTRA_INTER_PROB,
            comp_inter_probs: DEFAULT_COMP_INTER_PROB,
            single_ref_probs: DEFAULT_SINGLE_REF_PROB,
            comp_ref_probs: DEFAULT_COMP_REF_PROB,
            inter_mode_probs: DEFAULT_INTER_MODE_PROBS,
            kf_y_mode_probs: DEFAULT_KF_Y_MODE_PROBS,
            kf_uv_mode_probs: DEFAULT_KF_UV_MODE_PROBS,
        }
    }
}

impl FrameContext {
    /// Reset to default probabilities
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_header_default() {
        let header = FrameHeader::default();
        assert_eq!(header.frame_marker, 2);
        assert_eq!(header.bit_depth, 8);
        assert!(header.is_keyframe());
    }

    #[test]
    fn test_tile_log2() {
        assert_eq!(FrameHeader::tile_log2(64, 1), 0);
        assert_eq!(FrameHeader::tile_log2(64, 64), 0);
        assert_eq!(FrameHeader::tile_log2(64, 65), 1);
        assert_eq!(FrameHeader::tile_log2(64, 128), 1);
    }

    #[test]
    fn test_quantization_lossless() {
        let mut params = QuantizationParams::default();
        assert!(params.is_lossless()); // All zeros = lossless

        params.base_q_idx = 1;
        assert!(!params.is_lossless());
    }

    #[test]
    fn test_frame_dimensions() {
        let mut header = FrameHeader::default();
        header.width = 1920;
        header.height = 1080;

        assert_eq!(header.sb_cols(), 30); // ceil(1920/64) = 30
        assert_eq!(header.sb_rows(), 17); // ceil(1080/64) = 17
        assert_eq!(header.mi_cols(), 240); // ceil(1920/8) = 240
        assert_eq!(header.mi_rows(), 135); // ceil(1080/8) = 135
    }
}
