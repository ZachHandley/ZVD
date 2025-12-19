//! VP9 Frame Header and Bitstream Writing
//!
//! This module handles writing VP9 frame headers and organizing
//! the compressed bitstream for output.

use super::frame::{FrameType, LoopFilterParams, QuantizationParams, SegmentationParams, TileInfo};
use super::range_encoder::BitWriter;
use super::tables::{ColorSpace, InterpFilter, Profile};

/// VP9 frame header for encoding
#[derive(Debug, Clone)]
pub struct FrameHeaderWriter {
    /// Profile (0-3)
    pub profile: Profile,
    /// Frame type
    pub frame_type: FrameType,
    /// Show frame flag
    pub show_frame: bool,
    /// Error resilient mode
    pub error_resilient_mode: bool,
    /// Bit depth
    pub bit_depth: u8,
    /// Color space
    pub color_space: ColorSpace,
    /// Color range
    pub color_range: bool,
    /// Subsampling X
    pub subsampling_x: bool,
    /// Subsampling Y
    pub subsampling_y: bool,
    /// Frame width
    pub width: u32,
    /// Frame height
    pub height: u32,
    /// Render width
    pub render_width: u32,
    /// Render height
    pub render_height: u32,
    /// Intra only (for non-key inter frames)
    pub intra_only: bool,
    /// Refresh frame flags
    pub refresh_frame_flags: u8,
    /// Reference frame indices
    pub ref_frame_idx: [u8; 3],
    /// Reference frame sign bias
    pub ref_frame_sign_bias: [bool; 4],
    /// Allow high precision MV
    pub allow_high_precision_mv: bool,
    /// Interpolation filter
    pub interp_filter: InterpFilter,
    /// Refresh frame context
    pub refresh_frame_context: bool,
    /// Frame parallel decoding
    pub frame_parallel_decoding_mode: bool,
    /// Frame context index
    pub frame_context_idx: u8,
    /// Loop filter params
    pub loop_filter: LoopFilterParams,
    /// Quantization params
    pub quantization: QuantizationParams,
    /// Segmentation params
    pub segmentation: SegmentationParams,
    /// Tile info
    pub tile_info: TileInfo,
}

impl Default for FrameHeaderWriter {
    fn default() -> Self {
        FrameHeaderWriter {
            profile: Profile::Profile0,
            frame_type: FrameType::KeyFrame,
            show_frame: true,
            error_resilient_mode: false,
            bit_depth: 8,
            color_space: ColorSpace::Bt709,
            color_range: false,
            subsampling_x: true,
            subsampling_y: true,
            width: 0,
            height: 0,
            render_width: 0,
            render_height: 0,
            intra_only: false,
            refresh_frame_flags: 0xFF,
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
        }
    }
}

impl FrameHeaderWriter {
    /// Create a new frame header for a keyframe
    pub fn new_keyframe(width: u32, height: u32, q_index: u8) -> Self {
        FrameHeaderWriter {
            width,
            height,
            render_width: width,
            render_height: height,
            frame_type: FrameType::KeyFrame,
            quantization: QuantizationParams {
                base_q_idx: q_index,
                y_dc_delta: 0,
                uv_dc_delta: 0,
                uv_ac_delta: 0,
                lossless: q_index == 0,
            },
            ..Default::default()
        }
    }

    /// Write the uncompressed header to a bit writer
    pub fn write_uncompressed_header(&self, compressed_header_size: u16) -> Vec<u8> {
        let mut writer = BitWriter::new();

        // Frame marker (2 bits, always 2)
        writer.write_bits(2, 2);

        // Profile (2 bits, low then high)
        let profile = self.profile as u8;
        writer.write_bit((profile & 1) != 0);
        writer.write_bit((profile & 2) != 0);

        // Reserved bit for profile 3
        if profile == 3 {
            writer.write_bit(false);
        }

        // Show existing frame (always false for new frames)
        writer.write_bit(false);

        // Frame type
        writer.write_bit(self.frame_type == FrameType::InterFrame);

        // Show frame
        writer.write_bit(self.show_frame);

        // Error resilient mode
        writer.write_bit(self.error_resilient_mode);

        if self.frame_type == FrameType::KeyFrame {
            // Sync code
            writer.write_bits(0x49, 8); // 'I'
            writer.write_bits(0x83, 8);
            writer.write_bits(0x42, 8);

            // Color config
            self.write_color_config(&mut writer);

            // Frame size
            self.write_frame_size(&mut writer);

            // Render size
            self.write_render_size(&mut writer);
        } else {
            // Inter frame
            if !self.show_frame {
                writer.write_bit(self.intra_only);
            }

            if !self.error_resilient_mode {
                writer.write_bits(0, 2); // reset_frame_context
            }

            if self.intra_only {
                // Sync code
                writer.write_bits(0x49, 8);
                writer.write_bits(0x83, 8);
                writer.write_bits(0x42, 8);

                if self.profile != Profile::Profile0 {
                    self.write_color_config(&mut writer);
                }

                writer.write_bits(self.refresh_frame_flags as u32, 8);
                self.write_frame_size(&mut writer);
                self.write_render_size(&mut writer);
            } else {
                writer.write_bits(self.refresh_frame_flags as u32, 8);

                for i in 0..3 {
                    writer.write_bits(self.ref_frame_idx[i] as u32, 3);
                    writer.write_bit(self.ref_frame_sign_bias[i + 1]);
                }

                // Frame size from refs (false = explicit size)
                writer.write_bit(false);
                self.write_frame_size(&mut writer);
                self.write_render_size(&mut writer);

                writer.write_bit(self.allow_high_precision_mv);
                self.write_interp_filter(&mut writer);
            }
        }

        // Refresh frame context
        if !self.error_resilient_mode {
            writer.write_bit(self.refresh_frame_context);
            writer.write_bit(self.frame_parallel_decoding_mode);
        }

        // Frame context index
        writer.write_bits(self.frame_context_idx as u32, 2);

        // Loop filter params
        self.write_loop_filter(&mut writer);

        // Quantization params
        self.write_quantization(&mut writer);

        // Segmentation params
        self.write_segmentation(&mut writer);

        // Tile info
        self.write_tile_info(&mut writer);

        // Compressed header size
        writer.write_bits(compressed_header_size as u32, 16);

        writer.finalize()
    }

    /// Write color configuration
    fn write_color_config(&self, writer: &mut BitWriter) {
        // Bit depth for profiles 2 and 3
        if self.profile == Profile::Profile2 || self.profile == Profile::Profile3 {
            writer.write_bit(self.bit_depth == 12);
        }

        // Color space
        writer.write_bits(self.color_space as u32, 3);

        if self.color_space != ColorSpace::Srgb {
            // Color range
            writer.write_bit(self.color_range);

            // Subsampling for profiles 1 and 3
            if self.profile == Profile::Profile1 || self.profile == Profile::Profile3 {
                writer.write_bit(self.subsampling_x);
                writer.write_bit(self.subsampling_y);
                writer.write_bit(false); // Reserved
            }
        } else {
            // sRGB
            if self.profile == Profile::Profile1 || self.profile == Profile::Profile3 {
                writer.write_bit(false); // Reserved
            }
        }
    }

    /// Write frame size
    fn write_frame_size(&self, writer: &mut BitWriter) {
        writer.write_bits(self.width - 1, 16);
        writer.write_bits(self.height - 1, 16);
    }

    /// Write render size
    fn write_render_size(&self, writer: &mut BitWriter) {
        let different = self.render_width != self.width || self.render_height != self.height;
        writer.write_bit(different);
        if different {
            writer.write_bits(self.render_width - 1, 16);
            writer.write_bits(self.render_height - 1, 16);
        }
    }

    /// Write interpolation filter
    fn write_interp_filter(&self, writer: &mut BitWriter) {
        let is_switchable = self.interp_filter == InterpFilter::Switchable;
        writer.write_bit(is_switchable);
        if !is_switchable {
            let filter_idx = match self.interp_filter {
                InterpFilter::EightTapSmooth => 0,
                InterpFilter::EightTap => 1,
                InterpFilter::EightTapSharp => 2,
                InterpFilter::Bilinear => 3,
                InterpFilter::Switchable => 0,
            };
            writer.write_bits(filter_idx, 2);
        }
    }

    /// Write loop filter params
    fn write_loop_filter(&self, writer: &mut BitWriter) {
        writer.write_bits(self.loop_filter.level as u32, 6);
        writer.write_bits(self.loop_filter.sharpness as u32, 3);

        writer.write_bit(self.loop_filter.delta_enabled);
        if self.loop_filter.delta_enabled {
            writer.write_bit(self.loop_filter.delta_update);
            if self.loop_filter.delta_update {
                // Ref deltas
                for i in 0..4 {
                    let delta = self.loop_filter.ref_deltas[i];
                    writer.write_bit(delta != 0);
                    if delta != 0 {
                        write_signed_6(writer, delta);
                    }
                }
                // Mode deltas
                for i in 0..2 {
                    let delta = self.loop_filter.mode_deltas[i];
                    writer.write_bit(delta != 0);
                    if delta != 0 {
                        write_signed_6(writer, delta);
                    }
                }
            }
        }
    }

    /// Write quantization params
    fn write_quantization(&self, writer: &mut BitWriter) {
        writer.write_bits(self.quantization.base_q_idx as u32, 8);

        write_delta_q(writer, self.quantization.y_dc_delta);
        write_delta_q(writer, self.quantization.uv_dc_delta);
        write_delta_q(writer, self.quantization.uv_ac_delta);
    }

    /// Write segmentation params
    fn write_segmentation(&self, writer: &mut BitWriter) {
        writer.write_bit(self.segmentation.enabled);

        if !self.segmentation.enabled {
            return;
        }

        writer.write_bit(self.segmentation.update_map);
        if self.segmentation.update_map {
            // Tree probs
            for i in 0..7 {
                let prob = self.segmentation.tree_probs[i];
                let update = prob != 255;
                writer.write_bit(update);
                if update {
                    writer.write_bits(prob as u32, 8);
                }
            }

            writer.write_bit(self.segmentation.temporal_update);
            if self.segmentation.temporal_update {
                for i in 0..3 {
                    let prob = self.segmentation.pred_probs[i];
                    let update = prob != 255;
                    writer.write_bit(update);
                    if update {
                        writer.write_bits(prob as u32, 8);
                    }
                }
            }
        }

        writer.write_bit(self.segmentation.update_data);
        if self.segmentation.update_data {
            writer.write_bit(self.segmentation.abs_or_delta_update);

            let feature_bits = [8u8, 6, 2, 0];
            let feature_signed = [true, true, false, false];

            for seg_id in 0..8 {
                for feat_id in 0..4 {
                    let feature = &self.segmentation.features[seg_id][feat_id];
                    writer.write_bit(feature.enabled);
                    if feature.enabled && feature_bits[feat_id] > 0 {
                        if feature_signed[feat_id] {
                            let value = feature.value.abs() as u32;
                            writer.write_bits(value, feature_bits[feat_id]);
                            writer.write_bit(feature.value < 0);
                        } else {
                            writer.write_bits(feature.value as u32, feature_bits[feat_id]);
                        }
                    }
                }
            }
        }
    }

    /// Write tile info
    fn write_tile_info(&self, writer: &mut BitWriter) {
        // Calculate min/max tile columns
        let sb_cols = (self.width + 63) / 64;
        let min_log2 = tile_log2(64, sb_cols as usize);
        let max_log2 = tile_log2(1, (sb_cols as usize).min(64));

        // Write tile_cols_log2
        let mut cols_log2 = self.tile_info.cols_log2 as usize;
        while cols_log2 > min_log2 {
            writer.write_bit(true);
            cols_log2 -= 1;
        }
        if cols_log2 < max_log2 {
            writer.write_bit(false);
        }

        // Write tile_rows_log2
        let rows_log2 = self.tile_info.rows_log2;
        if rows_log2 > 0 {
            writer.write_bit(true);
            if rows_log2 > 1 {
                writer.write_bit(true);
            } else {
                writer.write_bit(false);
            }
        } else {
            writer.write_bit(false);
        }
    }

    /// Get number of superblocks in width
    pub fn sb_cols(&self) -> usize {
        ((self.width + 63) / 64) as usize
    }

    /// Get number of superblocks in height
    pub fn sb_rows(&self) -> usize {
        ((self.height + 63) / 64) as usize
    }
}

/// Write a delta Q value
fn write_delta_q(writer: &mut BitWriter, delta: i8) {
    if delta != 0 {
        writer.write_bit(true);
        let magnitude = delta.unsigned_abs() as u32;
        writer.write_bits(magnitude & 0xF, 4);
        writer.write_bit(delta < 0);
    } else {
        writer.write_bit(false);
    }
}

/// Write a signed 6-bit value
fn write_signed_6(writer: &mut BitWriter, value: i8) {
    let magnitude = value.abs() as u32;
    writer.write_bits(magnitude & 0x3F, 6);
    // Note: VP9 uses separate sign bit after magnitude
}

/// Calculate tile log2
fn tile_log2(blk_size: usize, target: usize) -> usize {
    let mut k = 0;
    while (blk_size << k) < target {
        k += 1;
    }
    k
}

/// VP9 frame builder for complete frame output
pub struct FrameBuilder {
    /// Uncompressed header
    uncompressed_header: Vec<u8>,
    /// Compressed header (probability updates)
    compressed_header: Vec<u8>,
    /// Tile data
    tile_data: Vec<Vec<u8>>,
}

impl FrameBuilder {
    /// Create a new frame builder
    pub fn new() -> Self {
        FrameBuilder {
            uncompressed_header: Vec::new(),
            compressed_header: Vec::new(),
            tile_data: Vec::new(),
        }
    }

    /// Set the uncompressed header
    pub fn set_uncompressed_header(&mut self, header: Vec<u8>) {
        self.uncompressed_header = header;
    }

    /// Set the compressed header
    pub fn set_compressed_header(&mut self, header: Vec<u8>) {
        self.compressed_header = header;
    }

    /// Add tile data
    pub fn add_tile(&mut self, data: Vec<u8>) {
        self.tile_data.push(data);
    }

    /// Build the complete frame
    pub fn build(self) -> Vec<u8> {
        let mut frame = Vec::new();

        // Uncompressed header
        frame.extend_from_slice(&self.uncompressed_header);

        // Compressed header
        frame.extend_from_slice(&self.compressed_header);

        // Tile data
        // For single tile, just append data
        // For multiple tiles, need tile size prefixes
        if self.tile_data.len() == 1 {
            frame.extend_from_slice(&self.tile_data[0]);
        } else {
            for (i, tile) in self.tile_data.iter().enumerate() {
                if i < self.tile_data.len() - 1 {
                    // Tile size (4 bytes, little-endian)
                    let size = tile.len() as u32;
                    frame.extend_from_slice(&size.to_le_bytes());
                }
                frame.extend_from_slice(tile);
            }
        }

        frame
    }
}

impl Default for FrameBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Write compressed header (probability updates)
/// For keyframes with default probabilities, this is minimal
pub fn write_compressed_header_keyframe() -> Vec<u8> {
    use super::range_encoder::RangeEncoder;

    let mut encoder = RangeEncoder::new();

    // TX mode (for keyframes, typically TX_MODE_SELECT or a fixed size)
    // Write TX mode: 0 = ONLY_4X4, 1 = ALLOW_8X8, etc.
    // For simplicity, use ONLY_4X4
    encoder.write_literal(0, 2); // ONLY_4X4 mode

    // No probability updates for keyframe with defaults
    // This produces a minimal compressed header

    encoder.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_header_writer_default() {
        let header = FrameHeaderWriter::default();
        assert_eq!(header.frame_type, FrameType::KeyFrame);
        assert_eq!(header.profile, Profile::Profile0);
    }

    #[test]
    fn test_frame_header_keyframe() {
        let header = FrameHeaderWriter::new_keyframe(640, 480, 50);
        assert_eq!(header.width, 640);
        assert_eq!(header.height, 480);
        assert_eq!(header.quantization.base_q_idx, 50);
    }

    #[test]
    fn test_write_uncompressed_header() {
        let header = FrameHeaderWriter::new_keyframe(640, 480, 50);
        let data = header.write_uncompressed_header(10);

        // Should start with frame marker (0b10 = 2)
        assert!(!data.is_empty());
        assert_eq!(data[0] >> 6, 2); // Frame marker in top 2 bits
    }

    #[test]
    fn test_frame_builder() {
        let mut builder = FrameBuilder::new();
        builder.set_uncompressed_header(vec![0x82, 0x49, 0x83, 0x42]);
        builder.set_compressed_header(vec![0x00]);
        builder.add_tile(vec![0x01, 0x02, 0x03]);

        let frame = builder.build();
        assert_eq!(frame.len(), 8);
    }

    #[test]
    fn test_tile_log2() {
        assert_eq!(tile_log2(64, 1), 0);
        assert_eq!(tile_log2(64, 64), 0);
        assert_eq!(tile_log2(64, 65), 1);
        assert_eq!(tile_log2(64, 128), 1);
    }

    #[test]
    fn test_compressed_header_keyframe() {
        let data = write_compressed_header_keyframe();
        assert!(!data.is_empty());
    }
}
