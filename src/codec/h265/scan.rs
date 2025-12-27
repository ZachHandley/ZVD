//! Coefficient scanning for H.265/HEVC
//!
//! This module implements the scanning patterns and coefficient decoding
//! used in H.265 to read transform coefficients from the bitstream.
//!
//! # Scanning Patterns
//!
//! H.265 uses three main scanning patterns:
//! - **Diagonal**: Default scan (zig-zag from top-left to bottom-right)
//! - **Horizontal**: Row-by-row scanning
//! - **Vertical**: Column-by-column scanning
//!
//! # Coefficient Groups
//!
//! Coefficients are organized into 4×4 coefficient groups (CGs) for efficient
//! context modeling in CABAC.

use crate::error::{Error, Result};

/// Coefficient scanning pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanPattern {
    /// Diagonal scan (default, zig-zag)
    Diagonal = 0,
    /// Horizontal scan
    Horizontal = 1,
    /// Vertical scan
    Vertical = 2,
}

/// Coefficient scanner for transform blocks
pub struct CoefficientScanner {
    /// Transform size (log2)
    log2_size: u8,
    /// Scan pattern
    pattern: ScanPattern,
    /// Scan order lookup table
    scan_order: Vec<(usize, usize)>,
}

impl CoefficientScanner {
    /// Create a new coefficient scanner
    pub fn new(log2_size: u8, pattern: ScanPattern) -> Result<Self> {
        if log2_size < 2 || log2_size > 5 {
            return Err(Error::Codec(format!(
                "Invalid transform size log2: {}",
                log2_size
            )));
        }

        let size = 1 << log2_size;
        let scan_order = Self::generate_scan_order(size, pattern);

        Ok(Self {
            log2_size,
            pattern,
            scan_order,
        })
    }

    /// Generate scanning order for given size and pattern
    fn generate_scan_order(size: usize, pattern: ScanPattern) -> Vec<(usize, usize)> {
        let mut order = Vec::with_capacity(size * size);

        match pattern {
            ScanPattern::Diagonal => {
                // Diagonal zig-zag scan
                Self::generate_diagonal_scan(size, &mut order);
            }
            ScanPattern::Horizontal => {
                // Row-by-row scan
                for y in 0..size {
                    for x in 0..size {
                        order.push((x, y));
                    }
                }
            }
            ScanPattern::Vertical => {
                // Column-by-column scan
                for x in 0..size {
                    for y in 0..size {
                        order.push((x, y));
                    }
                }
            }
        }

        order
    }

    /// Generate diagonal (zig-zag) scanning pattern
    fn generate_diagonal_scan(size: usize, order: &mut Vec<(usize, usize)>) {
        // Diagonal scan from top-left to bottom-right
        // Processes diagonals in order, alternating direction

        for diag in 0..(2 * size - 1) {
            let mut positions = Vec::new();

            // Collect positions on this diagonal
            for y in 0..size {
                let x = diag.wrapping_sub(y);
                if x < size {
                    positions.push((x, y));
                }
            }

            // Reverse every other diagonal for zig-zag
            if diag % 2 == 1 {
                positions.reverse();
            }

            order.extend(positions);
        }
    }

    /// Get position at scan index
    pub fn get_position(&self, scan_idx: usize) -> Option<(usize, usize)> {
        self.scan_order.get(scan_idx).copied()
    }

    /// Get scan index for position
    pub fn get_scan_index(&self, x: usize, y: usize) -> Option<usize> {
        self.scan_order
            .iter()
            .position(|&(sx, sy)| sx == x && sy == y)
    }

    /// Get total number of coefficients
    pub fn num_coeffs(&self) -> usize {
        self.scan_order.len()
    }

    /// Iterate over scan order
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.scan_order.iter().copied()
    }
}

/// Coefficient group (CG) scanner for 4×4 groups
pub struct CoefficientGroupScanner {
    /// Number of CGs in each dimension
    cg_size: usize,
    /// Scan pattern for CGs
    pattern: ScanPattern,
    /// CG scan order
    cg_scan_order: Vec<(usize, usize)>,
}

impl CoefficientGroupScanner {
    /// Create a new CG scanner
    pub fn new(log2_size: u8, pattern: ScanPattern) -> Result<Self> {
        if log2_size < 2 {
            return Err(Error::Codec(format!(
                "Transform size too small for CG scanning: {}",
                log2_size
            )));
        }

        // CG size in log2 (always 2 for 4×4 groups)
        let log2_cg_size = 2u8;

        // Number of CGs = transform_size / 4
        let cg_size = if log2_size >= log2_cg_size {
            1 << (log2_size - log2_cg_size)
        } else {
            1
        };

        let cg_scan_order = if cg_size > 1 {
            CoefficientScanner::generate_scan_order(cg_size, pattern)
        } else {
            vec![(0, 0)]
        };

        Ok(Self {
            cg_size,
            pattern,
            cg_scan_order,
        })
    }

    /// Get CG position at scan index
    pub fn get_cg_position(&self, cg_scan_idx: usize) -> Option<(usize, usize)> {
        self.cg_scan_order.get(cg_scan_idx).copied()
    }

    /// Get number of coefficient groups
    pub fn num_cgs(&self) -> usize {
        self.cg_scan_order.len()
    }

    /// Iterate over CG scan order
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.cg_scan_order.iter().copied()
    }
}

/// Significance map for coefficients
///
/// Tracks which coefficients are non-zero
pub struct SignificanceMap {
    /// Size of transform block
    size: usize,
    /// Significance flags (true = non-zero coefficient)
    significant: Vec<bool>,
}

impl SignificanceMap {
    /// Create a new significance map
    pub fn new(size: usize) -> Self {
        Self {
            size,
            significant: vec![false; size * size],
        }
    }

    /// Set coefficient as significant
    pub fn set_significant(&mut self, x: usize, y: usize, significant: bool) -> Result<()> {
        let idx = y * self.size + x;
        if idx >= self.significant.len() {
            return Err(Error::Codec(format!(
                "Position out of bounds: ({}, {})",
                x, y
            )));
        }
        self.significant[idx] = significant;
        Ok(())
    }

    /// Check if coefficient is significant
    pub fn is_significant(&self, x: usize, y: usize) -> bool {
        let idx = y * self.size + x;
        self.significant.get(idx).copied().unwrap_or(false)
    }

    /// Count significant coefficients in a region
    pub fn count_significant(&self, x_start: usize, y_start: usize, width: usize, height: usize) -> usize {
        let mut count = 0;
        for y in y_start..y_start + height {
            for x in x_start..x_start + width {
                if self.is_significant(x, y) {
                    count += 1;
                }
            }
        }
        count
    }

    /// Get total number of significant coefficients
    pub fn total_significant(&self) -> usize {
        self.significant.iter().filter(|&&s| s).count()
    }
}

/// Coefficient levels (quantized values)
pub struct CoefficientLevels {
    /// Size of transform block
    size: usize,
    /// Coefficient level values
    levels: Vec<i16>,
}

impl CoefficientLevels {
    /// Create new coefficient levels
    pub fn new(size: usize) -> Self {
        Self {
            size,
            levels: vec![0; size * size],
        }
    }

    /// Set coefficient level
    pub fn set_level(&mut self, x: usize, y: usize, level: i16) -> Result<()> {
        let idx = y * self.size + x;
        if idx >= self.levels.len() {
            return Err(Error::Codec(format!(
                "Position out of bounds: ({}, {})",
                x, y
            )));
        }
        self.levels[idx] = level;
        Ok(())
    }

    /// Get coefficient level
    pub fn get_level(&self, x: usize, y: usize) -> i16 {
        let idx = y * self.size + x;
        self.levels.get(idx).copied().unwrap_or(0)
    }

    /// Get all levels as slice
    pub fn as_slice(&self) -> &[i16] {
        &self.levels
    }

    /// Get mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [i16] {
        &mut self.levels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coefficient_scanner_4x4_diagonal() {
        let scanner = CoefficientScanner::new(2, ScanPattern::Diagonal).unwrap();
        assert_eq!(scanner.num_coeffs(), 16);

        // First position should be (0, 0)
        assert_eq!(scanner.get_position(0), Some((0, 0)));
    }

    #[test]
    fn test_coefficient_scanner_4x4_horizontal() {
        let scanner = CoefficientScanner::new(2, ScanPattern::Horizontal).unwrap();
        assert_eq!(scanner.num_coeffs(), 16);

        // Horizontal scan: row by row
        assert_eq!(scanner.get_position(0), Some((0, 0)));
        assert_eq!(scanner.get_position(1), Some((1, 0)));
        assert_eq!(scanner.get_position(4), Some((0, 1)));
    }

    #[test]
    fn test_coefficient_scanner_4x4_vertical() {
        let scanner = CoefficientScanner::new(2, ScanPattern::Vertical).unwrap();
        assert_eq!(scanner.num_coeffs(), 16);

        // Vertical scan: column by column
        assert_eq!(scanner.get_position(0), Some((0, 0)));
        assert_eq!(scanner.get_position(1), Some((0, 1)));
        assert_eq!(scanner.get_position(4), Some((1, 0)));
    }

    #[test]
    fn test_coefficient_scanner_8x8() {
        let scanner = CoefficientScanner::new(3, ScanPattern::Diagonal).unwrap();
        assert_eq!(scanner.num_coeffs(), 64);
    }

    #[test]
    fn test_coefficient_scanner_invalid_size() {
        let scanner = CoefficientScanner::new(1, ScanPattern::Diagonal);
        assert!(scanner.is_err());

        let scanner = CoefficientScanner::new(6, ScanPattern::Diagonal);
        assert!(scanner.is_err());
    }

    #[test]
    fn test_scan_index_lookup() {
        let scanner = CoefficientScanner::new(2, ScanPattern::Horizontal).unwrap();

        // (0, 0) should be index 0
        assert_eq!(scanner.get_scan_index(0, 0), Some(0));

        // (1, 0) should be index 1
        assert_eq!(scanner.get_scan_index(1, 0), Some(1));
    }

    #[test]
    fn test_coefficient_group_scanner() {
        let cg_scanner = CoefficientGroupScanner::new(4, ScanPattern::Diagonal).unwrap();
        // 16×16 block has 4×4 = 16 CGs
        assert_eq!(cg_scanner.num_cgs(), 16);
    }

    #[test]
    fn test_coefficient_group_scanner_small() {
        let cg_scanner = CoefficientGroupScanner::new(2, ScanPattern::Diagonal).unwrap();
        // 4×4 block has 1 CG
        assert_eq!(cg_scanner.num_cgs(), 1);
    }

    #[test]
    fn test_significance_map() {
        let mut sig_map = SignificanceMap::new(4);

        // Initially all zeros
        assert!(!sig_map.is_significant(0, 0));

        // Set (1, 1) as significant
        sig_map.set_significant(1, 1, true).unwrap();
        assert!(sig_map.is_significant(1, 1));
        assert_eq!(sig_map.total_significant(), 1);
    }

    #[test]
    fn test_significance_map_count() {
        let mut sig_map = SignificanceMap::new(4);

        sig_map.set_significant(0, 0, true).unwrap();
        sig_map.set_significant(1, 1, true).unwrap();
        sig_map.set_significant(2, 2, true).unwrap();

        assert_eq!(sig_map.total_significant(), 3);
    }

    #[test]
    fn test_significance_map_region_count() {
        let mut sig_map = SignificanceMap::new(4);

        // Set top-left 2×2 region
        sig_map.set_significant(0, 0, true).unwrap();
        sig_map.set_significant(1, 0, true).unwrap();
        sig_map.set_significant(0, 1, true).unwrap();

        let count = sig_map.count_significant(0, 0, 2, 2);
        assert_eq!(count, 3);
    }

    #[test]
    fn test_coefficient_levels() {
        let mut levels = CoefficientLevels::new(4);

        // Set some levels
        levels.set_level(0, 0, 10).unwrap();
        levels.set_level(1, 1, -5).unwrap();

        assert_eq!(levels.get_level(0, 0), 10);
        assert_eq!(levels.get_level(1, 1), -5);
        assert_eq!(levels.get_level(2, 2), 0); // Unset
    }

    #[test]
    fn test_coefficient_levels_slice() {
        let levels = CoefficientLevels::new(4);
        assert_eq!(levels.as_slice().len(), 16);
    }

    #[test]
    fn test_scan_patterns() {
        assert_eq!(ScanPattern::Diagonal as u8, 0);
        assert_eq!(ScanPattern::Horizontal as u8, 1);
        assert_eq!(ScanPattern::Vertical as u8, 2);
    }

    #[test]
    fn test_diagonal_scan_covers_all() {
        let scanner = CoefficientScanner::new(2, ScanPattern::Diagonal).unwrap();

        // Ensure all 16 positions are unique
        let mut positions = std::collections::HashSet::new();
        for (x, y) in scanner.iter() {
            positions.insert((x, y));
        }
        assert_eq!(positions.len(), 16);
    }

    #[test]
    fn test_horizontal_scan_order() {
        let scanner = CoefficientScanner::new(2, ScanPattern::Horizontal).unwrap();

        // Check first row is scanned first
        for i in 0..4 {
            let (_, y) = scanner.get_position(i).unwrap();
            assert_eq!(y, 0); // First row
        }
    }

    #[test]
    fn test_vertical_scan_order() {
        let scanner = CoefficientScanner::new(2, ScanPattern::Vertical).unwrap();

        // Check first column is scanned first
        for i in 0..4 {
            let (x, _) = scanner.get_position(i).unwrap();
            assert_eq!(x, 0); // First column
        }
    }

    #[test]
    fn test_significance_map_bounds() {
        let mut sig_map = SignificanceMap::new(4);

        // Out of bounds should error
        let result = sig_map.set_significant(5, 5, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_coefficient_levels_bounds() {
        let mut levels = CoefficientLevels::new(4);

        // Out of bounds should error
        let result = levels.set_level(5, 5, 10);
        assert!(result.is_err());
    }
}
