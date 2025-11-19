//! DNxHD Macroblock Processing
//!
//! Handles 16×16 macroblock structure and 8×8 block extraction/insertion

use crate::error::Result;

/// Macroblock structure (16×16 pixels)
#[derive(Debug, Clone)]
pub struct Macroblock {
    /// Y blocks (4 blocks for 4:2:2, 4 blocks for 4:4:4)
    pub y_blocks: Vec<[i16; 64]>,
    /// Cb blocks (2 blocks for 4:2:2, 4 blocks for 4:4:4)
    pub cb_blocks: Vec<[i16; 64]>,
    /// Cr blocks (2 blocks for 4:2:2, 4 blocks for 4:4:4)
    pub cr_blocks: Vec<[i16; 64]>,
    /// Quantization scale for this macroblock
    pub qscale: u16,
}

impl Macroblock {
    /// Create a new macroblock for 4:2:2
    pub fn new_422() -> Self {
        Self {
            y_blocks: vec![[0i16; 64]; 4],  // 4 Y blocks (2×2 in 16×16)
            cb_blocks: vec![[0i16; 64]; 2], // 2 Cb blocks (1×2 in 8×16)
            cr_blocks: vec![[0i16; 64]; 2], // 2 Cr blocks (1×2 in 8×16)
            qscale: 1024,
        }
    }

    /// Create a new macroblock for 4:4:4
    pub fn new_444() -> Self {
        Self {
            y_blocks: vec![[0i16; 64]; 4],  // 4 Y blocks
            cb_blocks: vec![[0i16; 64]; 4], // 4 Cb blocks
            cr_blocks: vec![[0i16; 64]; 4], // 4 Cr blocks
            qscale: 1024,
        }
    }
}

/// Macroblock processor
pub struct MacroblockProcessor {
    width: usize,
    height: usize,
    is_444: bool,
}

impl MacroblockProcessor {
    /// Create a new macroblock processor
    pub fn new(width: usize, height: usize, is_444: bool) -> Self {
        Self {
            width,
            height,
            is_444,
        }
    }

    /// Extract macroblock from frame at position (mb_x, mb_y)
    pub fn extract_macroblock(
        &self,
        y_plane: &[u8],
        cb_plane: &[u8],
        cr_plane: &[u8],
        mb_x: usize,
        mb_y: usize,
    ) -> Result<Macroblock> {
        let mb = if self.is_444 {
            Macroblock::new_444()
        } else {
            Macroblock::new_422()
        };

        // Extract Y blocks (4 blocks in 2×2 arrangement)
        let mut mb = mb;
        self.extract_y_blocks(y_plane, mb_x, mb_y, &mut mb)?;

        // Extract chroma blocks
        if self.is_444 {
            self.extract_chroma_444(cb_plane, cr_plane, mb_x, mb_y, &mut mb)?;
        } else {
            self.extract_chroma_422(cb_plane, cr_plane, mb_x, mb_y, &mut mb)?;
        }

        Ok(mb)
    }

    /// Extract Y blocks from frame
    fn extract_y_blocks(
        &self,
        y_plane: &[u8],
        mb_x: usize,
        mb_y: usize,
        mb: &mut Macroblock,
    ) -> Result<()> {
        let mb_start_x = mb_x * 16;
        let mb_start_y = mb_y * 16;

        // Extract 4 8×8 blocks in 2×2 arrangement
        for block_idx in 0..4 {
            let block_x = mb_start_x + (block_idx % 2) * 8;
            let block_y = mb_start_y + (block_idx / 2) * 8;

            self.extract_8x8_block(y_plane, self.width, block_x, block_y, &mut mb.y_blocks[block_idx])?;
        }

        Ok(())
    }

    /// Extract chroma blocks for 4:2:2
    fn extract_chroma_422(
        &self,
        cb_plane: &[u8],
        cr_plane: &[u8],
        mb_x: usize,
        mb_y: usize,
        mb: &mut Macroblock,
    ) -> Result<()> {
        let chroma_width = self.width / 2;
        let mb_start_x = mb_x * 8; // Half width for 4:2:2
        let mb_start_y = mb_y * 16;

        // Extract 2 Cb blocks vertically
        for block_idx in 0..2 {
            let block_x = mb_start_x;
            let block_y = mb_start_y + block_idx * 8;
            self.extract_8x8_block(cb_plane, chroma_width, block_x, block_y, &mut mb.cb_blocks[block_idx])?;
        }

        // Extract 2 Cr blocks vertically
        for block_idx in 0..2 {
            let block_x = mb_start_x;
            let block_y = mb_start_y + block_idx * 8;
            self.extract_8x8_block(cr_plane, chroma_width, block_x, block_y, &mut mb.cr_blocks[block_idx])?;
        }

        Ok(())
    }

    /// Extract chroma blocks for 4:4:4
    fn extract_chroma_444(
        &self,
        cb_plane: &[u8],
        cr_plane: &[u8],
        mb_x: usize,
        mb_y: usize,
        mb: &mut Macroblock,
    ) -> Result<()> {
        let mb_start_x = mb_x * 16;
        let mb_start_y = mb_y * 16;

        // Extract 4 Cb blocks in 2×2 arrangement (same as Y)
        for block_idx in 0..4 {
            let block_x = mb_start_x + (block_idx % 2) * 8;
            let block_y = mb_start_y + (block_idx / 2) * 8;
            self.extract_8x8_block(cb_plane, self.width, block_x, block_y, &mut mb.cb_blocks[block_idx])?;
        }

        // Extract 4 Cr blocks in 2×2 arrangement
        for block_idx in 0..4 {
            let block_x = mb_start_x + (block_idx % 2) * 8;
            let block_y = mb_start_y + (block_idx / 2) * 8;
            self.extract_8x8_block(cr_plane, self.width, block_x, block_y, &mut mb.cr_blocks[block_idx])?;
        }

        Ok(())
    }

    /// Extract a single 8×8 block from plane
    fn extract_8x8_block(
        &self,
        plane: &[u8],
        plane_width: usize,
        block_x: usize,
        block_y: usize,
        output: &mut [i16; 64],
    ) -> Result<()> {
        for y in 0..8 {
            for x in 0..8 {
                let px = block_x + x;
                let py = block_y + y;

                let value = if px < plane_width && py < self.height {
                    let idx = py * plane_width + px;
                    if idx < plane.len() {
                        plane[idx] as i16 - 128 // Center around 0
                    } else {
                        0
                    }
                } else {
                    0
                };

                output[y * 8 + x] = value;
            }
        }

        Ok(())
    }

    /// Insert macroblock back into frame
    pub fn insert_macroblock(
        &self,
        mb: &Macroblock,
        y_plane: &mut [u8],
        cb_plane: &mut [u8],
        cr_plane: &mut [u8],
        mb_x: usize,
        mb_y: usize,
    ) -> Result<()> {
        // Insert Y blocks
        self.insert_y_blocks(mb, y_plane, mb_x, mb_y)?;

        // Insert chroma blocks
        if self.is_444 {
            self.insert_chroma_444(mb, cb_plane, cr_plane, mb_x, mb_y)?;
        } else {
            self.insert_chroma_422(mb, cb_plane, cr_plane, mb_x, mb_y)?;
        }

        Ok(())
    }

    /// Insert Y blocks into frame
    fn insert_y_blocks(
        &self,
        mb: &Macroblock,
        y_plane: &mut [u8],
        mb_x: usize,
        mb_y: usize,
    ) -> Result<()> {
        let mb_start_x = mb_x * 16;
        let mb_start_y = mb_y * 16;

        for block_idx in 0..4 {
            let block_x = mb_start_x + (block_idx % 2) * 8;
            let block_y = mb_start_y + (block_idx / 2) * 8;

            self.insert_8x8_block(&mb.y_blocks[block_idx], y_plane, self.width, block_x, block_y)?;
        }

        Ok(())
    }

    /// Insert chroma blocks for 4:2:2
    fn insert_chroma_422(
        &self,
        mb: &Macroblock,
        cb_plane: &mut [u8],
        cr_plane: &mut [u8],
        mb_x: usize,
        mb_y: usize,
    ) -> Result<()> {
        let chroma_width = self.width / 2;
        let mb_start_x = mb_x * 8;
        let mb_start_y = mb_y * 16;

        for block_idx in 0..2 {
            let block_x = mb_start_x;
            let block_y = mb_start_y + block_idx * 8;
            self.insert_8x8_block(&mb.cb_blocks[block_idx], cb_plane, chroma_width, block_x, block_y)?;
            self.insert_8x8_block(&mb.cr_blocks[block_idx], cr_plane, chroma_width, block_x, block_y)?;
        }

        Ok(())
    }

    /// Insert chroma blocks for 4:4:4
    fn insert_chroma_444(
        &self,
        mb: &Macroblock,
        cb_plane: &mut [u8],
        cr_plane: &mut [u8],
        mb_x: usize,
        mb_y: usize,
    ) -> Result<()> {
        let mb_start_x = mb_x * 16;
        let mb_start_y = mb_y * 16;

        for block_idx in 0..4 {
            let block_x = mb_start_x + (block_idx % 2) * 8;
            let block_y = mb_start_y + (block_idx / 2) * 8;
            self.insert_8x8_block(&mb.cb_blocks[block_idx], cb_plane, self.width, block_x, block_y)?;
            self.insert_8x8_block(&mb.cr_blocks[block_idx], cr_plane, self.width, block_x, block_y)?;
        }

        Ok(())
    }

    /// Insert a single 8×8 block into plane
    fn insert_8x8_block(
        &self,
        block: &[i16; 64],
        plane: &mut [u8],
        plane_width: usize,
        block_x: usize,
        block_y: usize,
    ) -> Result<()> {
        for y in 0..8 {
            for x in 0..8 {
                let px = block_x + x;
                let py = block_y + y;

                if px < plane_width && py < self.height {
                    let idx = py * plane_width + px;
                    if idx < plane.len() {
                        let value = (block[y * 8 + x] + 128).clamp(0, 255);
                        plane[idx] = value as u8;
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macroblock_new_422() {
        let mb = Macroblock::new_422();
        assert_eq!(mb.y_blocks.len(), 4);
        assert_eq!(mb.cb_blocks.len(), 2);
        assert_eq!(mb.cr_blocks.len(), 2);
    }

    #[test]
    fn test_macroblock_new_444() {
        let mb = Macroblock::new_444();
        assert_eq!(mb.y_blocks.len(), 4);
        assert_eq!(mb.cb_blocks.len(), 4);
        assert_eq!(mb.cr_blocks.len(), 4);
    }

    #[test]
    fn test_extract_insert_roundtrip_422() {
        let width = 64;
        let height = 64;
        let processor = MacroblockProcessor::new(width, height, false);

        let mut y_plane = vec![100u8; width * height];
        let mut cb_plane = vec![128u8; (width / 2) * height];
        let mut cr_plane = vec![128u8; (width / 2) * height];

        let mb = processor.extract_macroblock(&y_plane, &cb_plane, &cr_plane, 0, 0).unwrap();

        let mut y_out = vec![0u8; width * height];
        let mut cb_out = vec![0u8; (width / 2) * height];
        let mut cr_out = vec![0u8; (width / 2) * height];

        processor.insert_macroblock(&mb, &mut y_out, &mut cb_out, &mut cr_out, 0, 0).unwrap();

        // Check first macroblock area
        for i in 0..256 {
            let diff = (y_plane[i] as i16 - y_out[i] as i16).abs();
            assert!(diff <= 1, "Y plane mismatch at {}: {} vs {}", i, y_plane[i], y_out[i]);
        }
    }
}
