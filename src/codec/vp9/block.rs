//! VP9 Superblock and Block Structure
//!
//! VP9 uses a hierarchical block structure:
//! - 64x64 superblocks are the largest coding unit
//! - Recursive quadtree partitioning down to 4x4 blocks
//! - Each block has associated mode info

use super::tables::{
    BlockSize, InterMode, IntraMode, MotionVector, Partition, RefFrame, TxSize, TxType,
};

/// Mode information for a single block
#[derive(Debug, Clone, Default)]
pub struct ModeInfo {
    /// Block size
    pub block_size: BlockSize,
    /// Is this an intra-predicted block
    pub is_intra: bool,
    /// Skip flag (no residual)
    pub skip: bool,
    /// Segment ID (0-7)
    pub segment_id: u8,

    // Intra prediction info
    /// Y prediction mode
    pub y_mode: IntraMode,
    /// UV prediction mode
    pub uv_mode: IntraMode,
    /// Sub-block Y modes for 4x4 intra (if applicable)
    pub sub_modes: [IntraMode; 4],

    // Inter prediction info
    /// Inter prediction mode
    pub inter_mode: InterMode,
    /// Reference frames (up to 2 for compound)
    pub ref_frames: [RefFrame; 2],
    /// Motion vectors (up to 2 for compound)
    pub mv: [MotionVector; 2],
    /// Is compound prediction
    pub is_compound: bool,

    // Transform info
    /// Transform size
    pub tx_size: TxSize,
    /// Transform type
    pub tx_type: TxType,
}

impl ModeInfo {
    pub fn new_intra(block_size: BlockSize, y_mode: IntraMode, uv_mode: IntraMode) -> Self {
        ModeInfo {
            block_size,
            is_intra: true,
            y_mode,
            uv_mode,
            ..Default::default()
        }
    }

    pub fn new_inter(
        block_size: BlockSize,
        inter_mode: InterMode,
        ref_frame: RefFrame,
        mv: MotionVector,
    ) -> Self {
        ModeInfo {
            block_size,
            is_intra: false,
            inter_mode,
            ref_frames: [ref_frame, RefFrame::None],
            mv: [mv, MotionVector::zero()],
            ..Default::default()
        }
    }
}

impl Default for BlockSize {
    fn default() -> Self {
        BlockSize::Block8x8
    }
}

impl Default for IntraMode {
    fn default() -> Self {
        IntraMode::DcPred
    }
}

impl Default for InterMode {
    fn default() -> Self {
        InterMode::ZeroMv
    }
}

impl Default for RefFrame {
    fn default() -> Self {
        RefFrame::None
    }
}

impl Default for TxSize {
    fn default() -> Self {
        TxSize::Tx4x4
    }
}

impl Default for TxType {
    fn default() -> Self {
        TxType::DctDct
    }
}

/// Block information grid
/// Stores mode info for every 8x8 block in the frame
pub struct ModeInfoGrid {
    /// Grid of mode info blocks
    pub grid: Vec<ModeInfo>,
    /// Width in 8x8 units
    pub cols: usize,
    /// Height in 8x8 units
    pub rows: usize,
}

impl ModeInfoGrid {
    /// Create a new mode info grid
    pub fn new(width: u32, height: u32) -> Self {
        let cols = ((width + 7) / 8) as usize;
        let rows = ((height + 7) / 8) as usize;
        let size = cols * rows;

        ModeInfoGrid {
            grid: vec![ModeInfo::default(); size],
            cols,
            rows,
        }
    }

    /// Get mode info at position
    pub fn get(&self, mi_col: usize, mi_row: usize) -> Option<&ModeInfo> {
        if mi_col < self.cols && mi_row < self.rows {
            Some(&self.grid[mi_row * self.cols + mi_col])
        } else {
            None
        }
    }

    /// Get mutable mode info at position
    pub fn get_mut(&mut self, mi_col: usize, mi_row: usize) -> Option<&mut ModeInfo> {
        if mi_col < self.cols && mi_row < self.rows {
            Some(&mut self.grid[mi_row * self.cols + mi_col])
        } else {
            None
        }
    }

    /// Set mode info for a block
    pub fn set(&mut self, mi_col: usize, mi_row: usize, info: ModeInfo) {
        if mi_col < self.cols && mi_row < self.rows {
            self.grid[mi_row * self.cols + mi_col] = info;
        }
    }

    /// Fill a block region with mode info
    pub fn fill_block(
        &mut self,
        mi_col: usize,
        mi_row: usize,
        block_size: BlockSize,
        info: &ModeInfo,
    ) {
        let w = block_size.width_mi();
        let h = block_size.height_mi();

        for dy in 0..h {
            for dx in 0..w {
                let col = mi_col + dx;
                let row = mi_row + dy;
                if col < self.cols && row < self.rows {
                    self.grid[row * self.cols + col] = info.clone();
                }
            }
        }
    }

    /// Get above mode info (row - 1)
    pub fn get_above(&self, mi_col: usize, mi_row: usize) -> Option<&ModeInfo> {
        if mi_row > 0 {
            self.get(mi_col, mi_row - 1)
        } else {
            None
        }
    }

    /// Get left mode info (col - 1)
    pub fn get_left(&self, mi_col: usize, mi_row: usize) -> Option<&ModeInfo> {
        if mi_col > 0 {
            self.get(mi_col - 1, mi_row)
        } else {
            None
        }
    }

    /// Clear all mode info
    pub fn clear(&mut self) {
        for info in &mut self.grid {
            *info = ModeInfo::default();
        }
    }
}

/// Partition context calculator
pub struct PartitionContext {
    /// Above partition context (one per 8x8 column)
    above: Vec<u8>,
    /// Left partition context (one per 8x8 row in current superblock column)
    left: Vec<u8>,
}

impl PartitionContext {
    pub fn new(mi_cols: usize, mi_rows: usize) -> Self {
        PartitionContext {
            above: vec![0; mi_cols],
            left: vec![0; 8], // One superblock height worth
        }
    }

    /// Get partition context for a block
    pub fn get_context(&self, mi_col: usize, mi_row: usize, block_size: BlockSize) -> usize {
        let bsl = block_size_to_log2(block_size);
        let bsl = bsl.saturating_sub(2); // Normalize to 0-4 range

        let above = if mi_col < self.above.len() {
            (self.above[mi_col] != 0) as usize
        } else {
            0
        };

        let left_idx = mi_row & 7; // Within superblock
        let left = if left_idx < self.left.len() {
            (self.left[left_idx] != 0) as usize
        } else {
            0
        };

        // Context = bsl * 4 + left * 2 + above
        (bsl as usize) * 4 + left * 2 + above
    }

    /// Update context after decoding a partition
    pub fn update(
        &mut self,
        mi_col: usize,
        mi_row: usize,
        block_size: BlockSize,
        partition: Partition,
    ) {
        let w = block_size.width_mi();
        let h = block_size.height_mi();

        let ctx_value = match partition {
            Partition::None => 0,
            _ => 1,
        };

        // Update above context
        for i in 0..w {
            if mi_col + i < self.above.len() {
                self.above[mi_col + i] = ctx_value;
            }
        }

        // Update left context
        for i in 0..h {
            let left_idx = (mi_row + i) & 7;
            if left_idx < self.left.len() {
                self.left[left_idx] = ctx_value;
            }
        }
    }

    /// Clear left context at start of new superblock row
    pub fn clear_left(&mut self) {
        self.left.fill(0);
    }
}

/// Get log2 of block size (for square blocks)
fn block_size_to_log2(bs: BlockSize) -> u8 {
    match bs {
        BlockSize::Block4x4 | BlockSize::Block4x8 | BlockSize::Block8x4 => 2,
        BlockSize::Block8x8 | BlockSize::Block8x16 | BlockSize::Block16x8 => 3,
        BlockSize::Block16x16 | BlockSize::Block16x32 | BlockSize::Block32x16 => 4,
        BlockSize::Block32x32 | BlockSize::Block32x64 | BlockSize::Block64x32 => 5,
        BlockSize::Block64x64 => 6,
        BlockSize::Invalid => 0,
    }
}

/// Superblock decoder state
pub struct SuperblockState {
    /// Current superblock row
    pub sb_row: usize,
    /// Current superblock column
    pub sb_col: usize,
    /// Above context for this superblock
    pub above_context: [u8; 16],
    /// Left context for this superblock
    pub left_context: [u8; 16],
}

impl SuperblockState {
    pub fn new(sb_row: usize, sb_col: usize) -> Self {
        SuperblockState {
            sb_row,
            sb_col,
            above_context: [0; 16],
            left_context: [0; 16],
        }
    }

    /// Get pixel position of top-left corner
    pub fn pixel_x(&self) -> usize {
        self.sb_col * 64
    }

    pub fn pixel_y(&self) -> usize {
        self.sb_row * 64
    }

    /// Get mode info position
    pub fn mi_col(&self) -> usize {
        self.sb_col * 8
    }

    pub fn mi_row(&self) -> usize {
        self.sb_row * 8
    }
}

/// Transform block info
#[derive(Debug, Clone, Default)]
pub struct TransformBlockInfo {
    /// Transform size
    pub tx_size: TxSize,
    /// Transform type
    pub tx_type: TxType,
    /// Number of non-zero coefficients
    pub eob: u16,
    /// Block has non-zero coefficients
    pub has_coeffs: bool,
}

/// Coefficient context calculation
pub fn get_coeff_context(
    plane: usize,
    tx_size: TxSize,
    above_nonzero: bool,
    left_nonzero: bool,
) -> usize {
    // Simple context based on above/left neighbors
    let above = above_nonzero as usize;
    let left = left_nonzero as usize;
    above + left
}

/// Get maximum transform size for a block size
pub fn max_tx_size_for_block(block_size: BlockSize) -> TxSize {
    match block_size {
        BlockSize::Block4x4 | BlockSize::Block4x8 | BlockSize::Block8x4 => TxSize::Tx4x4,
        BlockSize::Block8x8 | BlockSize::Block8x16 | BlockSize::Block16x8 => TxSize::Tx8x8,
        BlockSize::Block16x16 | BlockSize::Block16x32 | BlockSize::Block32x16 => TxSize::Tx16x16,
        BlockSize::Block32x32
        | BlockSize::Block32x64
        | BlockSize::Block64x32
        | BlockSize::Block64x64 => TxSize::Tx32x32,
        BlockSize::Invalid => TxSize::Tx4x4,
    }
}

/// Get transform type based on intra mode (for 4x4 blocks)
pub fn get_tx_type_4x4(mode: IntraMode) -> TxType {
    match mode {
        IntraMode::VPred | IntraMode::D63Pred | IntraMode::D117Pred => TxType::AdstDct,
        IntraMode::HPred | IntraMode::D153Pred | IntraMode::D207Pred => TxType::DctAdst,
        IntraMode::D45Pred | IntraMode::D135Pred | IntraMode::TmPred => TxType::AdstAdst,
        IntraMode::DcPred => TxType::DctDct,
    }
}

/// Number of 4x4 blocks in each transform size
pub fn num_4x4_blocks(tx_size: TxSize) -> usize {
    match tx_size {
        TxSize::Tx4x4 => 1,
        TxSize::Tx8x8 => 4,
        TxSize::Tx16x16 => 16,
        TxSize::Tx32x32 => 64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mode_info_grid() {
        let mut grid = ModeInfoGrid::new(640, 480);
        assert_eq!(grid.cols, 80); // 640/8
        assert_eq!(grid.rows, 60); // 480/8

        let info = ModeInfo::new_intra(BlockSize::Block16x16, IntraMode::VPred, IntraMode::DcPred);
        grid.set(0, 0, info.clone());

        let retrieved = grid.get(0, 0).unwrap();
        assert!(retrieved.is_intra);
        assert_eq!(retrieved.y_mode, IntraMode::VPred);
    }

    #[test]
    fn test_fill_block() {
        let mut grid = ModeInfoGrid::new(128, 128);
        let info = ModeInfo::new_intra(BlockSize::Block16x16, IntraMode::HPred, IntraMode::HPred);

        // Fill a 16x16 block (2x2 in 8x8 units)
        grid.fill_block(0, 0, BlockSize::Block16x16, &info);

        // Check all 4 8x8 positions
        for row in 0..2 {
            for col in 0..2 {
                let mi = grid.get(col, row).unwrap();
                assert_eq!(mi.y_mode, IntraMode::HPred);
            }
        }
    }

    #[test]
    fn test_partition_context() {
        let mut ctx = PartitionContext::new(80, 60);

        // Initial context should be 0
        let c = ctx.get_context(0, 0, BlockSize::Block64x64);
        assert!(c < 20); // Valid context range

        // Update with a split
        ctx.update(0, 0, BlockSize::Block64x64, Partition::Split);

        // Context should change
        let c2 = ctx.get_context(1, 0, BlockSize::Block32x32);
        assert!(c2 < 20);
    }

    #[test]
    fn test_max_tx_size() {
        assert_eq!(max_tx_size_for_block(BlockSize::Block4x4), TxSize::Tx4x4);
        assert_eq!(max_tx_size_for_block(BlockSize::Block8x8), TxSize::Tx8x8);
        assert_eq!(
            max_tx_size_for_block(BlockSize::Block16x16),
            TxSize::Tx16x16
        );
        assert_eq!(
            max_tx_size_for_block(BlockSize::Block64x64),
            TxSize::Tx32x32
        );
    }

    #[test]
    fn test_tx_type_for_mode() {
        assert_eq!(get_tx_type_4x4(IntraMode::DcPred), TxType::DctDct);
        assert_eq!(get_tx_type_4x4(IntraMode::VPred), TxType::AdstDct);
        assert_eq!(get_tx_type_4x4(IntraMode::HPred), TxType::DctAdst);
        assert_eq!(get_tx_type_4x4(IntraMode::TmPred), TxType::AdstAdst);
    }
}
