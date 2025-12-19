//! H.265/HEVC Coding Tree Unit (CTU) structures
//!
//! The CTU is the basic unit of H.265 encoding/decoding. Each CTU is recursively
//! subdivided using a quadtree structure into smaller Coding Units (CUs).
//!
//! CTU sizes: 64×64, 32×32, or 16×16 (specified in SPS)
//! CU sizes: 64×64 down to 8×8 (quadtree partitioning)
//! PU sizes: Various sizes for prediction
//! TU sizes: 32×32 down to 4×4 for transforms

use crate::error::{Error, Result};

/// CTU size enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CtuSize {
    /// 16×16 CTU
    Size16 = 16,
    /// 32×32 CTU
    Size32 = 32,
    /// 64×64 CTU (most common)
    Size64 = 64,
}

impl CtuSize {
    /// Get CTU size from log2 value (from SPS)
    pub fn from_log2(log2_size: u8) -> Result<Self> {
        match log2_size {
            4 => Ok(CtuSize::Size16),
            5 => Ok(CtuSize::Size32),
            6 => Ok(CtuSize::Size64),
            _ => Err(Error::codec(format!("Invalid CTU log2 size: {}", log2_size))),
        }
    }

    /// Get the size in pixels
    pub fn size(&self) -> usize {
        *self as usize
    }

    /// Get log2 of size
    pub fn log2(&self) -> u8 {
        match self {
            CtuSize::Size16 => 4,
            CtuSize::Size32 => 5,
            CtuSize::Size64 => 6,
        }
    }
}

/// Coding Unit (CU) - leaf node of CTU quadtree
#[derive(Debug, Clone)]
pub struct CodingUnit {
    /// Position within CTU (x in pixels)
    pub x: usize,
    /// Position within CTU (y in pixels)
    pub y: usize,
    /// Size of this CU (8, 16, 32, or 64)
    pub size: usize,
    /// Log2 of size
    pub log2_size: u8,

    /// Prediction mode
    pub pred_mode: PredMode,
    /// Intra prediction mode (if intra)
    pub intra_mode: Option<IntraMode>,

    /// Skip flag (for inter prediction)
    pub skip_flag: bool,
    /// Transquant bypass flag
    pub transquant_bypass_flag: bool,
}

/// Prediction mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredMode {
    /// Intra prediction (use spatial prediction from neighboring blocks)
    Intra,
    /// Inter prediction (use motion compensation from reference frames)
    Inter,
}

/// Intra prediction mode (H.265 has 35 modes)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntraMode {
    /// Planar mode (mode 0) - smooth gradient prediction
    Planar = 0,
    /// DC mode (mode 1) - average of neighboring pixels
    Dc = 1,
    /// Angular mode 2-34 - directional prediction
    Angular(u8),
}

impl IntraMode {
    /// Create intra mode from mode index (0-34)
    pub fn from_index(index: u8) -> Result<Self> {
        match index {
            0 => Ok(IntraMode::Planar),
            1 => Ok(IntraMode::Dc),
            2..=34 => Ok(IntraMode::Angular(index)),
            _ => Err(Error::codec(format!("Invalid intra mode: {}", index))),
        }
    }

    /// Get mode index
    pub fn index(&self) -> u8 {
        match self {
            IntraMode::Planar => 0,
            IntraMode::Dc => 1,
            IntraMode::Angular(idx) => *idx,
        }
    }

    /// Check if this is a horizontal-ish mode
    pub fn is_horizontal(&self) -> bool {
        matches!(self, IntraMode::Angular(10))
    }

    /// Check if this is a vertical-ish mode
    pub fn is_vertical(&self) -> bool {
        matches!(self, IntraMode::Angular(26))
    }

    /// Check if this is a diagonal mode
    pub fn is_diagonal(&self) -> bool {
        matches!(self, IntraMode::Angular(2) | IntraMode::Angular(18) | IntraMode::Angular(34))
    }
}

impl CodingUnit {
    /// Create a new coding unit
    pub fn new(x: usize, y: usize, size: usize) -> Self {
        let log2_size = (size as f32).log2() as u8;

        CodingUnit {
            x,
            y,
            size,
            log2_size,
            pred_mode: PredMode::Intra,
            intra_mode: Some(IntraMode::Dc), // Default to DC mode
            skip_flag: false,
            transquant_bypass_flag: false,
        }
    }

    /// Check if this CU can be split further
    pub fn can_split(&self, min_cu_size: usize) -> bool {
        self.size > min_cu_size
    }

    /// Get the 4 sub-CUs if this CU is split (quadtree)
    pub fn split(&self) -> [CodingUnit; 4] {
        let half_size = self.size / 2;

        [
            // Top-left
            CodingUnit::new(self.x, self.y, half_size),
            // Top-right
            CodingUnit::new(self.x + half_size, self.y, half_size),
            // Bottom-left
            CodingUnit::new(self.x, self.y + half_size, half_size),
            // Bottom-right
            CodingUnit::new(self.x + half_size, self.y + half_size, half_size),
        ]
    }
}

/// Coding Tree Unit (CTU) - the root of the quadtree
#[derive(Debug, Clone)]
pub struct CodingTreeUnit {
    /// CTU position in frame (in CTU units)
    pub ctu_x: usize,
    pub ctu_y: usize,

    /// CTU size (16, 32, or 64)
    pub size: CtuSize,

    /// Coding units in this CTU (quadtree leaves)
    pub coding_units: Vec<CodingUnit>,

    /// SAO (Sample Adaptive Offset) parameters
    pub sao_enabled: bool,
}

impl CodingTreeUnit {
    /// Create a new CTU
    pub fn new(ctu_x: usize, ctu_y: usize, size: CtuSize) -> Self {
        CodingTreeUnit {
            ctu_x,
            ctu_y,
            size,
            coding_units: Vec::new(),
            sao_enabled: false,
        }
    }

    /// Get pixel position of this CTU in the frame
    pub fn pixel_x(&self) -> usize {
        self.ctu_x * self.size.size()
    }

    /// Get pixel position of this CTU in the frame
    pub fn pixel_y(&self) -> usize {
        self.ctu_y * self.size.size()
    }

    /// Add a coding unit to this CTU
    pub fn add_cu(&mut self, cu: CodingUnit) {
        self.coding_units.push(cu);
    }

    /// Get all CUs in raster scan order
    pub fn get_cus(&self) -> &[CodingUnit] {
        &self.coding_units
    }
}

/// Frame buffer for storing decoded pixels
#[derive(Debug, Clone)]
pub struct FrameBuffer {
    /// Width in pixels
    pub width: usize,
    /// Height in pixels
    pub height: usize,
    /// Bit depth (8, 10, or 12)
    pub bit_depth: u8,

    /// Luma plane (Y)
    pub luma: Vec<u16>,
    /// Chroma Cb plane (U)
    pub chroma_cb: Vec<u16>,
    /// Chroma Cr plane (V)
    pub chroma_cr: Vec<u16>,

    /// Chroma format (0=mono, 1=4:2:0, 2=4:2:2, 3=4:4:4)
    pub chroma_format: u8,
}

impl FrameBuffer {
    /// Create a new frame buffer
    pub fn new(width: usize, height: usize, bit_depth: u8, chroma_format: u8) -> Self {
        let luma_size = width * height;

        // Calculate chroma dimensions based on format
        let (chroma_width, chroma_height) = match chroma_format {
            0 => (0, 0),           // Monochrome
            1 => (width / 2, height / 2),  // 4:2:0
            2 => (width / 2, height),      // 4:2:2
            3 => (width, height),          // 4:4:4
            _ => (width / 2, height / 2),  // Default to 4:2:0
        };

        let chroma_size = chroma_width * chroma_height;

        FrameBuffer {
            width,
            height,
            bit_depth,
            luma: vec![0; luma_size],
            chroma_cb: vec![0; chroma_size],
            chroma_cr: vec![0; chroma_size],
            chroma_format,
        }
    }

    /// Get luma pixel at (x, y)
    pub fn get_luma(&self, x: usize, y: usize) -> u16 {
        if x >= self.width || y >= self.height {
            return 0;
        }
        self.luma[y * self.width + x]
    }

    /// Set luma pixel at (x, y)
    pub fn set_luma(&mut self, x: usize, y: usize, value: u16) {
        if x < self.width && y < self.height {
            self.luma[y * self.width + x] = value;
        }
    }

    /// Get chroma width
    pub fn chroma_width(&self) -> usize {
        match self.chroma_format {
            0 => 0,
            1 | 2 => self.width / 2,
            3 => self.width,
            _ => self.width / 2,
        }
    }

    /// Get chroma height
    pub fn chroma_height(&self) -> usize {
        match self.chroma_format {
            0 => 0,
            1 => self.height / 2,
            2 | 3 => self.height,
            _ => self.height / 2,
        }
    }

    /// Clear the frame buffer
    pub fn clear(&mut self) {
        let mid_gray = 1 << (self.bit_depth - 1);

        self.luma.fill(0);
        self.chroma_cb.fill(mid_gray);
        self.chroma_cr.fill(mid_gray);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ctu_size_from_log2() {
        assert_eq!(CtuSize::from_log2(4).unwrap(), CtuSize::Size16);
        assert_eq!(CtuSize::from_log2(5).unwrap(), CtuSize::Size32);
        assert_eq!(CtuSize::from_log2(6).unwrap(), CtuSize::Size64);
        assert!(CtuSize::from_log2(7).is_err());
    }

    #[test]
    fn test_ctu_size_properties() {
        let size = CtuSize::Size64;
        assert_eq!(size.size(), 64);
        assert_eq!(size.log2(), 6);

        let size = CtuSize::Size32;
        assert_eq!(size.size(), 32);
        assert_eq!(size.log2(), 5);
    }

    #[test]
    fn test_intra_mode_from_index() {
        assert_eq!(IntraMode::from_index(0).unwrap(), IntraMode::Planar);
        assert_eq!(IntraMode::from_index(1).unwrap(), IntraMode::Dc);
        assert_eq!(IntraMode::from_index(10).unwrap(), IntraMode::Angular(10));
        assert_eq!(IntraMode::from_index(26).unwrap(), IntraMode::Angular(26));
        assert!(IntraMode::from_index(35).is_err());
    }

    #[test]
    fn test_intra_mode_helpers() {
        let planar = IntraMode::Planar;
        assert_eq!(planar.index(), 0);
        assert!(!planar.is_horizontal());
        assert!(!planar.is_vertical());

        let horizontal = IntraMode::Angular(10);
        assert!(horizontal.is_horizontal());
        assert!(!horizontal.is_vertical());

        let vertical = IntraMode::Angular(26);
        assert!(!vertical.is_horizontal());
        assert!(vertical.is_vertical());
    }

    #[test]
    fn test_cu_creation() {
        let cu = CodingUnit::new(0, 0, 32);
        assert_eq!(cu.x, 0);
        assert_eq!(cu.y, 0);
        assert_eq!(cu.size, 32);
        assert_eq!(cu.log2_size, 5);
        assert_eq!(cu.pred_mode, PredMode::Intra);
    }

    #[test]
    fn test_cu_split() {
        let cu = CodingUnit::new(0, 0, 64);
        assert!(cu.can_split(8));

        let split = cu.split();

        // Check top-left
        assert_eq!(split[0].x, 0);
        assert_eq!(split[0].y, 0);
        assert_eq!(split[0].size, 32);

        // Check top-right
        assert_eq!(split[1].x, 32);
        assert_eq!(split[1].y, 0);
        assert_eq!(split[1].size, 32);

        // Check bottom-left
        assert_eq!(split[2].x, 0);
        assert_eq!(split[2].y, 32);
        assert_eq!(split[2].size, 32);

        // Check bottom-right
        assert_eq!(split[3].x, 32);
        assert_eq!(split[3].y, 32);
        assert_eq!(split[3].size, 32);
    }

    #[test]
    fn test_ctu_creation() {
        let ctu = CodingTreeUnit::new(2, 3, CtuSize::Size64);
        assert_eq!(ctu.ctu_x, 2);
        assert_eq!(ctu.ctu_y, 3);
        assert_eq!(ctu.pixel_x(), 128);  // 2 * 64
        assert_eq!(ctu.pixel_y(), 192);  // 3 * 64
    }

    #[test]
    fn test_frame_buffer_4_2_0() {
        let fb = FrameBuffer::new(1920, 1080, 8, 1);
        assert_eq!(fb.width, 1920);
        assert_eq!(fb.height, 1080);
        assert_eq!(fb.chroma_width(), 960);  // 1920 / 2
        assert_eq!(fb.chroma_height(), 540); // 1080 / 2
        assert_eq!(fb.luma.len(), 1920 * 1080);
        assert_eq!(fb.chroma_cb.len(), 960 * 540);
    }

    #[test]
    fn test_frame_buffer_4_4_4() {
        let fb = FrameBuffer::new(1920, 1080, 10, 3);
        assert_eq!(fb.chroma_width(), 1920);
        assert_eq!(fb.chroma_height(), 1080);
        assert_eq!(fb.luma.len(), 1920 * 1080);
        assert_eq!(fb.chroma_cb.len(), 1920 * 1080);
    }

    #[test]
    fn test_frame_buffer_get_set() {
        let mut fb = FrameBuffer::new(16, 16, 8, 1);

        fb.set_luma(5, 7, 200);
        assert_eq!(fb.get_luma(5, 7), 200);

        // Out of bounds should return 0
        assert_eq!(fb.get_luma(20, 20), 0);
    }

    #[test]
    fn test_frame_buffer_clear() {
        let mut fb = FrameBuffer::new(16, 16, 8, 1);

        fb.set_luma(5, 5, 255);
        fb.clear();

        assert_eq!(fb.get_luma(5, 5), 0);
        // Chroma should be mid-gray (128 for 8-bit)
        assert_eq!(fb.chroma_cb[0], 128);
        assert_eq!(fb.chroma_cr[0], 128);
    }
}
