//! VP9 Tables and Constants
//!
//! This module contains all the static tables used by the VP9 decoder:
//! - Probability tables for entropy coding
//! - Quantization tables
//! - Scan orders for coefficients
//! - Block size and partition information

// =============================================================================
// Block Sizes and Partitions
// =============================================================================

/// VP9 block sizes
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum BlockSize {
    Block4x4 = 0,
    Block4x8 = 1,
    Block8x4 = 2,
    Block8x8 = 3,
    Block8x16 = 4,
    Block16x8 = 5,
    Block16x16 = 6,
    Block16x32 = 7,
    Block32x16 = 8,
    Block32x32 = 9,
    Block32x64 = 10,
    Block64x32 = 11,
    Block64x64 = 12,
    Invalid = 13,
}

impl BlockSize {
    /// Width in pixels
    pub const fn width(&self) -> usize {
        match self {
            BlockSize::Block4x4 | BlockSize::Block4x8 => 4,
            BlockSize::Block8x4 | BlockSize::Block8x8 | BlockSize::Block8x16 => 8,
            BlockSize::Block16x8 | BlockSize::Block16x16 | BlockSize::Block16x32 => 16,
            BlockSize::Block32x16 | BlockSize::Block32x32 | BlockSize::Block32x64 => 32,
            BlockSize::Block64x32 | BlockSize::Block64x64 => 64,
            BlockSize::Invalid => 0,
        }
    }

    /// Height in pixels
    pub const fn height(&self) -> usize {
        match self {
            BlockSize::Block4x4 | BlockSize::Block8x4 => 4,
            BlockSize::Block4x8 | BlockSize::Block8x8 | BlockSize::Block16x8 => 8,
            BlockSize::Block8x16 | BlockSize::Block16x16 | BlockSize::Block32x16 => 16,
            BlockSize::Block16x32 | BlockSize::Block32x32 | BlockSize::Block64x32 => 32,
            BlockSize::Block32x64 | BlockSize::Block64x64 => 64,
            BlockSize::Invalid => 0,
        }
    }

    /// Width in 4x4 units (mi_col units)
    pub const fn width_mi(&self) -> usize {
        self.width() >> 2
    }

    /// Height in 4x4 units (mi_row units)
    pub const fn height_mi(&self) -> usize {
        self.height() >> 2
    }

    /// Log2 of size (for square blocks)
    pub const fn log2(&self) -> u8 {
        match self {
            BlockSize::Block4x4 => 2,
            BlockSize::Block8x8 => 3,
            BlockSize::Block16x16 => 4,
            BlockSize::Block32x32 => 5,
            BlockSize::Block64x64 => 6,
            _ => 0, // Non-square blocks
        }
    }

    /// From block dimensions
    pub fn from_dimensions(width: usize, height: usize) -> Self {
        match (width, height) {
            (4, 4) => BlockSize::Block4x4,
            (4, 8) => BlockSize::Block4x8,
            (8, 4) => BlockSize::Block8x4,
            (8, 8) => BlockSize::Block8x8,
            (8, 16) => BlockSize::Block8x16,
            (16, 8) => BlockSize::Block16x8,
            (16, 16) => BlockSize::Block16x16,
            (16, 32) => BlockSize::Block16x32,
            (32, 16) => BlockSize::Block32x16,
            (32, 32) => BlockSize::Block32x32,
            (32, 64) => BlockSize::Block32x64,
            (64, 32) => BlockSize::Block64x32,
            (64, 64) => BlockSize::Block64x64,
            _ => BlockSize::Invalid,
        }
    }
}

/// Partition types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Partition {
    None = 0,       // Use whole block
    Horizontal = 1, // Split into top/bottom
    Vertical = 2,   // Split into left/right
    Split = 3,      // Split into 4 quadrants
}

/// Get subblock size after partition
pub const fn get_subsize(block_size: BlockSize, partition: Partition) -> BlockSize {
    match (block_size, partition) {
        (BlockSize::Block64x64, Partition::None) => BlockSize::Block64x64,
        (BlockSize::Block64x64, Partition::Horizontal) => BlockSize::Block64x32,
        (BlockSize::Block64x64, Partition::Vertical) => BlockSize::Block32x64,
        (BlockSize::Block64x64, Partition::Split) => BlockSize::Block32x32,

        (BlockSize::Block64x32, Partition::None) => BlockSize::Block64x32,
        (BlockSize::Block64x32, Partition::Horizontal) => BlockSize::Block64x32, // Invalid
        (BlockSize::Block64x32, Partition::Vertical) => BlockSize::Block32x32,
        (BlockSize::Block64x32, Partition::Split) => BlockSize::Block32x32,

        (BlockSize::Block32x64, Partition::None) => BlockSize::Block32x64,
        (BlockSize::Block32x64, Partition::Horizontal) => BlockSize::Block32x32,
        (BlockSize::Block32x64, Partition::Vertical) => BlockSize::Block32x64, // Invalid
        (BlockSize::Block32x64, Partition::Split) => BlockSize::Block32x32,

        (BlockSize::Block32x32, Partition::None) => BlockSize::Block32x32,
        (BlockSize::Block32x32, Partition::Horizontal) => BlockSize::Block32x16,
        (BlockSize::Block32x32, Partition::Vertical) => BlockSize::Block16x32,
        (BlockSize::Block32x32, Partition::Split) => BlockSize::Block16x16,

        (BlockSize::Block32x16, Partition::None) => BlockSize::Block32x16,
        (BlockSize::Block32x16, Partition::Horizontal) => BlockSize::Block32x16,
        (BlockSize::Block32x16, Partition::Vertical) => BlockSize::Block16x16,
        (BlockSize::Block32x16, Partition::Split) => BlockSize::Block16x16,

        (BlockSize::Block16x32, Partition::None) => BlockSize::Block16x32,
        (BlockSize::Block16x32, Partition::Horizontal) => BlockSize::Block16x16,
        (BlockSize::Block16x32, Partition::Vertical) => BlockSize::Block16x32,
        (BlockSize::Block16x32, Partition::Split) => BlockSize::Block16x16,

        (BlockSize::Block16x16, Partition::None) => BlockSize::Block16x16,
        (BlockSize::Block16x16, Partition::Horizontal) => BlockSize::Block16x8,
        (BlockSize::Block16x16, Partition::Vertical) => BlockSize::Block8x16,
        (BlockSize::Block16x16, Partition::Split) => BlockSize::Block8x8,

        (BlockSize::Block16x8, Partition::None) => BlockSize::Block16x8,
        (BlockSize::Block16x8, Partition::Horizontal) => BlockSize::Block16x8,
        (BlockSize::Block16x8, Partition::Vertical) => BlockSize::Block8x8,
        (BlockSize::Block16x8, Partition::Split) => BlockSize::Block8x8,

        (BlockSize::Block8x16, Partition::None) => BlockSize::Block8x16,
        (BlockSize::Block8x16, Partition::Horizontal) => BlockSize::Block8x8,
        (BlockSize::Block8x16, Partition::Vertical) => BlockSize::Block8x16,
        (BlockSize::Block8x16, Partition::Split) => BlockSize::Block8x8,

        (BlockSize::Block8x8, Partition::None) => BlockSize::Block8x8,
        (BlockSize::Block8x8, Partition::Horizontal) => BlockSize::Block8x4,
        (BlockSize::Block8x8, Partition::Vertical) => BlockSize::Block4x8,
        (BlockSize::Block8x8, Partition::Split) => BlockSize::Block4x4,

        (BlockSize::Block8x4, Partition::None) => BlockSize::Block8x4,
        (BlockSize::Block4x8, Partition::None) => BlockSize::Block4x8,
        (BlockSize::Block4x4, Partition::None) => BlockSize::Block4x4,

        _ => BlockSize::Invalid,
    }
}

// =============================================================================
// Transform Sizes
// =============================================================================

/// Transform sizes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TxSize {
    Tx4x4 = 0,
    Tx8x8 = 1,
    Tx16x16 = 2,
    Tx32x32 = 3,
}

impl TxSize {
    pub const fn size(&self) -> usize {
        match self {
            TxSize::Tx4x4 => 4,
            TxSize::Tx8x8 => 8,
            TxSize::Tx16x16 => 16,
            TxSize::Tx32x32 => 32,
        }
    }

    /// Get transform width (same as size for square transforms)
    pub const fn width(&self) -> usize {
        self.size()
    }

    /// Get transform height (same as size for square transforms)
    pub const fn height(&self) -> usize {
        self.size()
    }

    pub const fn log2(&self) -> u8 {
        match self {
            TxSize::Tx4x4 => 2,
            TxSize::Tx8x8 => 3,
            TxSize::Tx16x16 => 4,
            TxSize::Tx32x32 => 5,
        }
    }

    pub const fn num_coeffs(&self) -> usize {
        let s = self.size();
        s * s
    }
}

/// Transform type (DCT, ADST combinations)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TxType {
    DctDct = 0,   // DCT in both directions
    AdstDct = 1,  // ADST vertical, DCT horizontal
    DctAdst = 2,  // DCT vertical, ADST horizontal
    AdstAdst = 3, // ADST in both directions
}

/// Transform mode from frame header
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TxMode {
    Only4x4 = 0,
    Allow8x8 = 1,
    Allow16x16 = 2,
    Allow32x32 = 3,
    TxModeSelect = 4, // Per-block selection
}

// =============================================================================
// Intra Prediction Modes
// =============================================================================

/// VP9 intra prediction modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum IntraMode {
    DcPred = 0,   // DC prediction
    VPred = 1,    // Vertical
    HPred = 2,    // Horizontal
    D45Pred = 3,  // 45-degree diagonal (down-right)
    D135Pred = 4, // 135-degree diagonal (down-left)
    D117Pred = 5, // 117-degree
    D153Pred = 6, // 153-degree
    D207Pred = 7, // 207-degree (horizontal-up)
    D63Pred = 8,  // 63-degree (vertical-right)
    TmPred = 9,   // True Motion (gradient)
}

impl IntraMode {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => IntraMode::DcPred,
            1 => IntraMode::VPred,
            2 => IntraMode::HPred,
            3 => IntraMode::D45Pred,
            4 => IntraMode::D135Pred,
            5 => IntraMode::D117Pred,
            6 => IntraMode::D153Pred,
            7 => IntraMode::D207Pred,
            8 => IntraMode::D63Pred,
            9 => IntraMode::TmPred,
            _ => IntraMode::DcPred,
        }
    }
}

// =============================================================================
// Inter Prediction
// =============================================================================

/// Reference frame types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RefFrame {
    None = 0,
    Last = 1,
    Golden = 2,
    AltRef = 3,
}

impl RefFrame {
    /// Check if this is an intra frame (None reference)
    pub fn is_intra(&self) -> bool {
        *self == RefFrame::None
    }
}

/// Inter prediction modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum InterMode {
    NearestMv = 0,
    NearMv = 1,
    ZeroMv = 2,
    NewMv = 3,
}

/// Motion vector with 1/8-pel precision
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MotionVector {
    pub row: i16, // Vertical component in 1/8-pel units
    pub col: i16, // Horizontal component in 1/8-pel units
}

impl MotionVector {
    pub const fn new(row: i16, col: i16) -> Self {
        MotionVector { row, col }
    }

    pub const fn zero() -> Self {
        MotionVector { row: 0, col: 0 }
    }

    /// Full-pel row
    pub const fn row_int(&self) -> i16 {
        self.row >> 3
    }

    /// Full-pel col
    pub const fn col_int(&self) -> i16 {
        self.col >> 3
    }

    /// Sub-pel row (0-7)
    pub const fn row_frac(&self) -> u8 {
        (self.row & 7) as u8
    }

    /// Sub-pel col (0-7)
    pub const fn col_frac(&self) -> u8 {
        (self.col & 7) as u8
    }
}

/// Interpolation filter types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum InterpFilter {
    EightTap = 0,
    EightTapSmooth = 1,
    EightTapSharp = 2,
    Bilinear = 3,
    Switchable = 4,
}

// =============================================================================
// Quantization Tables
// =============================================================================

/// DC quantization table for 8-bit depth
pub const DC_QUANT_8BIT: [i16; 256] = [
    4, 8, 8, 9, 10, 11, 12, 12, 13, 14, 15, 16, 17, 18, 19, 19, 20, 21, 22, 23, 24, 25, 26, 26, 27,
    28, 29, 30, 31, 32, 32, 33, 34, 35, 36, 37, 38, 38, 39, 40, 41, 42, 43, 43, 44, 45, 46, 47, 48,
    48, 49, 50, 51, 52, 53, 53, 54, 55, 56, 57, 57, 58, 59, 60, 61, 62, 62, 63, 64, 65, 66, 66, 67,
    68, 69, 70, 70, 71, 72, 73, 74, 74, 75, 76, 77, 78, 78, 79, 80, 81, 81, 82, 83, 84, 85, 85, 87,
    88, 90, 92, 93, 95, 96, 98, 99, 101, 102, 104, 105, 107, 108, 110, 111, 113, 114, 116, 117,
    118, 120, 121, 123, 125, 127, 129, 131, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154,
    156, 158, 161, 164, 166, 169, 172, 174, 177, 180, 182, 185, 187, 190, 192, 195, 199, 202, 205,
    208, 211, 214, 217, 220, 223, 226, 230, 233, 237, 240, 243, 247, 250, 253, 257, 261, 265, 269,
    272, 276, 280, 284, 288, 292, 296, 300, 304, 309, 313, 317, 322, 326, 330, 335, 340, 344, 349,
    354, 359, 364, 369, 374, 379, 384, 389, 395, 400, 406, 411, 417, 423, 429, 435, 441, 447, 454,
    461, 467, 475, 482, 489, 497, 505, 513, 522, 530, 539, 549, 559, 569, 579, 590, 602, 614, 626,
    640, 654, 668, 684, 700, 717, 736, 755, 775, 796, 819, 843, 869, 896, 925, 955, 988, 1022,
    1058, 1098, 1139, 1184, 1232, 1282, 1336,
];

/// AC quantization table for 8-bit depth
pub const AC_QUANT_8BIT: [i16; 256] = [
    4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
    79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
    102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138,
    140, 142, 144, 146, 148, 150, 152, 155, 158, 161, 164, 167, 170, 173, 176, 179, 182, 185, 188,
    191, 194, 197, 200, 203, 207, 211, 215, 219, 223, 227, 231, 235, 239, 243, 247, 251, 255, 260,
    265, 270, 275, 280, 285, 290, 295, 300, 305, 311, 317, 323, 329, 335, 341, 347, 353, 359, 366,
    373, 380, 387, 394, 401, 408, 416, 424, 432, 440, 448, 456, 465, 474, 483, 492, 501, 510, 520,
    530, 540, 550, 560, 571, 582, 593, 604, 615, 627, 639, 651, 663, 676, 689, 702, 715, 729, 743,
    757, 771, 786, 801, 816, 832, 848, 864, 881, 898, 915, 933, 951, 969, 988, 1007, 1026, 1046,
    1066, 1087, 1108, 1129, 1151, 1173, 1196, 1219, 1243, 1267, 1292, 1317, 1343, 1369, 1396, 1423,
    1451, 1479, 1508, 1537, 1567, 1597, 1628, 1660, 1692, 1725, 1759, 1793, 1828,
];

// =============================================================================
// Probability Tables
// =============================================================================

/// Default partition probabilities
/// Indexed by [context][partition_type]
/// Context is based on above/left partition state
pub const DEFAULT_PARTITION_PROBS: [[u8; 3]; 16] = [
    [199, 122, 141], // 0
    [147, 63, 159],  // 1
    [148, 133, 118], // 2
    [121, 104, 114], // 3
    [174, 73, 87],   // 4
    [92, 41, 83],    // 5
    [82, 99, 50],    // 6
    [53, 39, 39],    // 7
    [177, 58, 59],   // 8
    [68, 26, 63],    // 9
    [52, 79, 25],    // 10
    [17, 14, 12],    // 11
    [222, 34, 30],   // 12
    [72, 16, 44],    // 13
    [58, 32, 12],    // 14
    [10, 7, 6],      // 15
];

/// Default skip probabilities
pub const DEFAULT_SKIP_PROBS: [u8; 3] = [192, 128, 64];

/// Default intra inter probabilities
pub const DEFAULT_INTRA_INTER_PROB: [u8; 4] = [9, 102, 187, 225];

/// Default compound mode probabilities
pub const DEFAULT_COMP_INTER_PROB: [u8; 5] = [239, 183, 119, 96, 41];

/// Default single reference probabilities
pub const DEFAULT_SINGLE_REF_PROB: [[u8; 2]; 5] =
    [[33, 16], [77, 74], [142, 142], [172, 170], [238, 247]];

/// Default compound reference probabilities
pub const DEFAULT_COMP_REF_PROB: [u8; 5] = [50, 126, 123, 221, 226];

/// Default inter mode probabilities
pub const DEFAULT_INTER_MODE_PROBS: [[u8; 3]; 7] = [
    [2, 173, 34],
    [7, 145, 85],
    [7, 166, 63],
    [7, 94, 66],
    [8, 64, 46],
    [17, 81, 31],
    [25, 29, 30],
];

/// Default Y mode probabilities for key frames
pub const DEFAULT_KF_Y_MODE_PROBS: [[u8; 9]; 10] = [
    [137, 30, 42, 148, 151, 207, 70, 52, 91],
    [92, 45, 102, 136, 116, 180, 74, 90, 100],
    [73, 32, 19, 187, 222, 215, 46, 34, 100],
    [91, 30, 32, 116, 121, 186, 93, 86, 94],
    [72, 35, 36, 149, 68, 206, 68, 63, 105],
    [73, 31, 28, 138, 57, 124, 55, 122, 151],
    [67, 23, 21, 140, 126, 197, 40, 37, 171],
    [86, 27, 28, 128, 154, 212, 45, 43, 53],
    [74, 32, 27, 107, 86, 160, 63, 134, 102],
    [59, 67, 44, 140, 161, 202, 78, 67, 119],
];

/// Default UV mode probabilities for key frames
pub const DEFAULT_KF_UV_MODE_PROBS: [[u8; 9]; 10] = [
    [121, 30, 54, 128, 164, 158, 45, 26, 70],
    [132, 44, 68, 128, 138, 165, 55, 46, 68],
    [68, 28, 65, 128, 200, 163, 32, 21, 77],
    [123, 29, 67, 128, 155, 178, 45, 35, 77],
    [90, 29, 55, 128, 145, 174, 38, 32, 116],
    [95, 34, 44, 128, 147, 163, 31, 31, 143],
    [94, 22, 42, 128, 156, 171, 32, 24, 51],
    [115, 37, 49, 128, 157, 178, 38, 28, 47],
    [102, 33, 48, 128, 155, 177, 36, 27, 64],
    [122, 37, 44, 128, 154, 162, 32, 23, 65],
];

/// Default Y mode probabilities for inter frames
pub const DEFAULT_IF_Y_MODE_PROBS: [[u8; 9]; 4] = [
    [65, 32, 18, 144, 162, 194, 41, 51, 98],
    [132, 68, 18, 165, 217, 196, 45, 40, 78],
    [173, 80, 19, 176, 240, 193, 64, 35, 46],
    [221, 135, 38, 194, 248, 121, 96, 85, 29],
];

/// Default UV mode probabilities for inter frames
pub const DEFAULT_IF_UV_MODE_PROBS: [[u8; 9]; 10] = [
    [120, 7, 76, 176, 208, 126, 28, 54, 103],
    [48, 12, 154, 155, 139, 90, 34, 117, 119],
    [67, 6, 25, 204, 243, 158, 13, 21, 96],
    [97, 5, 44, 131, 176, 139, 48, 68, 97],
    [83, 5, 42, 156, 111, 152, 26, 49, 152],
    [80, 5, 58, 178, 74, 83, 33, 62, 145],
    [86, 5, 32, 154, 192, 168, 14, 22, 163],
    [85, 5, 32, 156, 216, 148, 19, 29, 73],
    [77, 7, 64, 116, 132, 122, 37, 126, 120],
    [101, 21, 107, 181, 192, 103, 19, 67, 125],
];

// =============================================================================
// Interpolation Filters
// =============================================================================

/// 8-tap regular interpolation filter coefficients
/// Indexed by [subpel_position][tap]
pub const SUBPEL_FILTERS_REGULAR: [[i16; 8]; 16] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 1, -5, 126, 8, -3, 1, 0],
    [-1, 3, -10, 122, 18, -6, 2, 0],
    [-1, 4, -13, 118, 27, -9, 3, -1],
    [-1, 4, -16, 112, 37, -11, 4, -1],
    [-1, 5, -18, 105, 48, -14, 4, -1],
    [-1, 5, -19, 97, 58, -16, 5, -1],
    [-1, 6, -19, 88, 68, -18, 5, -1],
    [-1, 6, -19, 78, 78, -19, 6, -1],
    [-1, 5, -18, 68, 88, -19, 6, -1],
    [-1, 5, -16, 58, 97, -19, 5, -1],
    [-1, 4, -14, 48, 105, -18, 5, -1],
    [-1, 4, -11, 37, 112, -16, 4, -1],
    [-1, 3, -9, 27, 118, -13, 4, -1],
    [0, 2, -6, 18, 122, -10, 3, -1],
    [0, 1, -3, 8, 126, -5, 1, 0],
];

/// 8-tap smooth interpolation filter
pub const SUBPEL_FILTERS_SMOOTH: [[i16; 8]; 16] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [-3, -1, 32, 64, 38, 1, -3, 0],
    [-2, -2, 29, 63, 41, 2, -3, 0],
    [-2, -2, 26, 63, 44, 3, -4, 0],
    [-2, -3, 24, 62, 46, 4, -4, 1],
    [-2, -3, 21, 60, 49, 6, -4, 1],
    [-1, -4, 18, 59, 51, 7, -4, 2],
    [-1, -4, 16, 57, 53, 9, -4, 2],
    [-1, -4, 14, 55, 55, 14, -4, -1],
    [2, -4, 9, 53, 57, 16, -4, -1],
    [2, -4, 7, 51, 59, 18, -4, -1],
    [1, -4, 6, 49, 60, 21, -3, -2],
    [1, -4, 4, 46, 62, 24, -3, -2],
    [0, -4, 3, 44, 63, 26, -2, -2],
    [0, -3, 2, 41, 63, 29, -2, -2],
    [0, -3, 1, 38, 64, 32, -1, -3],
];

/// 8-tap sharp interpolation filter
pub const SUBPEL_FILTERS_SHARP: [[i16; 8]; 16] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [-1, 3, -7, 127, 8, -3, 1, 0],
    [-2, 5, -13, 125, 17, -6, 3, -1],
    [-3, 7, -17, 121, 27, -10, 5, -2],
    [-4, 9, -20, 115, 37, -13, 6, -2],
    [-4, 10, -23, 108, 48, -16, 8, -3],
    [-4, 10, -24, 100, 59, -19, 9, -3],
    [-4, 11, -24, 90, 70, -21, 10, -4],
    [-4, 11, -23, 80, 80, -23, 11, -4],
    [-4, 10, -21, 70, 90, -24, 11, -4],
    [-3, 9, -19, 59, 100, -24, 10, -4],
    [-3, 8, -16, 48, 108, -23, 10, -4],
    [-2, 6, -13, 37, 115, -20, 9, -4],
    [-2, 5, -10, 27, 121, -17, 7, -3],
    [-1, 3, -6, 17, 125, -13, 5, -2],
    [0, 1, -3, 8, 127, -7, 3, -1],
];

/// Bilinear filter coefficients (2-tap)
pub const BILINEAR_FILTERS: [[i16; 2]; 16] = [
    [128, 0],
    [120, 8],
    [112, 16],
    [104, 24],
    [96, 32],
    [88, 40],
    [80, 48],
    [72, 56],
    [64, 64],
    [56, 72],
    [48, 80],
    [40, 88],
    [32, 96],
    [24, 104],
    [16, 112],
    [8, 120],
];

// =============================================================================
// Scan Orders
// =============================================================================

/// Default scan order for 4x4 blocks
pub const DEFAULT_SCAN_4X4: [u8; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// Default scan order for 8x8 blocks
pub const DEFAULT_SCAN_8X8: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Default scan order for 16x16 blocks (first 64 shown, pattern continues)
pub const DEFAULT_SCAN_16X16: [u16; 256] = {
    let mut scan = [0u16; 256];
    let mut i = 0;
    let mut diag = 0;
    while i < 256 {
        let mut x = if diag < 16 { 0 } else { diag - 15 };
        let mut y = if diag < 16 { diag } else { 15 };
        while x <= (if diag < 16 { diag } else { 15 })
            && y >= (if diag < 16 { 0 } else { diag - 15 })
        {
            if i < 256 {
                scan[i] = (y * 16 + x) as u16;
                i += 1;
            }
            x += 1;
            if y > 0 {
                y -= 1;
            } else {
                break;
            }
        }
        diag += 1;
    }
    scan
};

/// Default scan order for 32x32 blocks (first 64 shown, pattern continues)
pub const DEFAULT_SCAN_32X32: [u16; 1024] = {
    let mut scan = [0u16; 1024];
    let mut i = 0;
    let mut diag = 0;
    while i < 1024 {
        let mut x = if diag < 32 { 0 } else { diag - 31 };
        let mut y = if diag < 32 { diag } else { 31 };
        while x <= (if diag < 32 { diag } else { 31 })
            && y >= (if diag < 32 { 0 } else { diag - 31 })
        {
            if i < 1024 {
                scan[i] = (y * 32 + x) as u16;
                i += 1;
            }
            x += 1;
            if y > 0 {
                y -= 1;
            } else {
                break;
            }
        }
        diag += 1;
    }
    scan
};

// =============================================================================
// Coefficient Band Structure
// =============================================================================

/// Coefficient band index for token decoding
/// Maps coefficient index to band (0-5)
pub const COEF_BANDS_4X4: [u8; 16] = [0, 1, 2, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5];

pub const COEF_BANDS_8X8: [u8; 64] = [
    0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
];

pub const COEF_BANDS_16X16: [u8; 256] = {
    let mut bands = [5u8; 256];
    bands[0] = 0;
    bands[1] = 1;
    bands[2] = 2;
    bands[3] = 2;
    bands[4] = 3;
    bands[5] = 3;
    bands[6] = 3;
    bands[7] = 3;
    let mut i = 8;
    while i < 16 {
        bands[i] = 4;
        i += 1;
    }
    bands
};

pub const COEF_BANDS_32X32: [u8; 1024] = {
    let mut bands = [5u8; 1024];
    bands[0] = 0;
    bands[1] = 1;
    bands[2] = 2;
    bands[3] = 2;
    bands[4] = 3;
    bands[5] = 3;
    bands[6] = 3;
    bands[7] = 3;
    let mut i = 8;
    while i < 32 {
        bands[i] = 4;
        i += 1;
    }
    bands
};

// =============================================================================
// Color Space and Profile
// =============================================================================

/// VP9 color spaces
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ColorSpace {
    Unknown = 0,
    Bt601 = 1,
    Bt709 = 2,
    Smpte170 = 3,
    Smpte240 = 4,
    Bt2020 = 5,
    Reserved = 6,
    Srgb = 7,
}

impl ColorSpace {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => ColorSpace::Unknown,
            1 => ColorSpace::Bt601,
            2 => ColorSpace::Bt709,
            3 => ColorSpace::Smpte170,
            4 => ColorSpace::Smpte240,
            5 => ColorSpace::Bt2020,
            6 => ColorSpace::Reserved,
            7 => ColorSpace::Srgb,
            _ => ColorSpace::Unknown,
        }
    }
}

/// VP9 profiles
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Profile {
    Profile0 = 0, // 8-bit 4:2:0
    Profile1 = 1, // 8-bit 4:2:2, 4:4:4
    Profile2 = 2, // 10/12-bit 4:2:0
    Profile3 = 3, // 10/12-bit 4:2:2, 4:4:4
}

impl Profile {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Profile::Profile0,
            1 => Profile::Profile1,
            2 => Profile::Profile2,
            3 => Profile::Profile3,
            _ => Profile::Profile0,
        }
    }

    pub fn bit_depth(&self) -> u8 {
        match self {
            Profile::Profile0 | Profile::Profile1 => 8,
            Profile::Profile2 | Profile::Profile3 => 10, // Can be 10 or 12
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_size_dimensions() {
        assert_eq!(BlockSize::Block64x64.width(), 64);
        assert_eq!(BlockSize::Block64x64.height(), 64);
        assert_eq!(BlockSize::Block32x16.width(), 32);
        assert_eq!(BlockSize::Block32x16.height(), 16);
        assert_eq!(BlockSize::Block4x8.width(), 4);
        assert_eq!(BlockSize::Block4x8.height(), 8);
    }

    #[test]
    fn test_get_subsize() {
        assert_eq!(
            get_subsize(BlockSize::Block64x64, Partition::Split),
            BlockSize::Block32x32
        );
        assert_eq!(
            get_subsize(BlockSize::Block32x32, Partition::Horizontal),
            BlockSize::Block32x16
        );
        assert_eq!(
            get_subsize(BlockSize::Block16x16, Partition::Vertical),
            BlockSize::Block8x16
        );
        assert_eq!(
            get_subsize(BlockSize::Block8x8, Partition::None),
            BlockSize::Block8x8
        );
    }

    #[test]
    fn test_tx_size() {
        assert_eq!(TxSize::Tx4x4.size(), 4);
        assert_eq!(TxSize::Tx8x8.num_coeffs(), 64);
        assert_eq!(TxSize::Tx16x16.log2(), 4);
        assert_eq!(TxSize::Tx32x32.size(), 32);
    }

    #[test]
    fn test_motion_vector() {
        let mv = MotionVector::new(13, 27);
        assert_eq!(mv.row_int(), 1); // 13 >> 3 = 1
        assert_eq!(mv.col_int(), 3); // 27 >> 3 = 3
        assert_eq!(mv.row_frac(), 5); // 13 & 7 = 5
        assert_eq!(mv.col_frac(), 3); // 27 & 7 = 3
    }

    #[test]
    fn test_quant_tables_length() {
        assert_eq!(DC_QUANT_8BIT.len(), 256);
        assert_eq!(AC_QUANT_8BIT.len(), 256);
    }
}
