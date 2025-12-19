//! 3D LUT (Look-Up Table) Support
//!
//! This module provides professional color grading using 3D LUTs (Look-Up Tables).
//! LUTs are widely used in film, video production, and post-production for:
//!
//! - Color grading and correction
//! - Film stock emulation
//! - Creative color transformations
//! - Camera-to-display color mapping
//! - HDR to SDR tone mapping
//!
//! ## Supported Formats
//!
//! - **.cube** - Adobe Cube LUT format (most common)
//! - **1D LUTs** - Simple input-output curves
//! - **3D LUTs** - Full RGB color space transformation
//!
//! ## Usage
//!
//! ```rust
//! use zvd_lib::util::lut::Lut3D;
//!
//! // Load LUT from file
//! let lut = Lut3D::from_cube_file("film_emulation.cube")?;
//!
//! // Apply to RGB pixel
//! let output = lut.apply(r, g, b);
//!
//! // Apply to entire frame
//! let transformed = lut.apply_to_buffer(&rgb_data, width, height)?;
//! ```

use crate::error::{Error, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// 1D LUT for single-channel transformations
#[derive(Debug, Clone)]
pub struct Lut1D {
    /// LUT size (number of entries)
    size: usize,
    /// R channel mapping
    r_table: Vec<f32>,
    /// G channel mapping
    g_table: Vec<f32>,
    /// B channel mapping
    b_table: Vec<f32>,
}

impl Lut1D {
    /// Create a new 1D LUT
    pub fn new(size: usize) -> Self {
        Lut1D {
            size,
            r_table: vec![0.0; size],
            g_table: vec![0.0; size],
            b_table: vec![0.0; size],
        }
    }

    /// Apply 1D LUT to RGB values (0.0-1.0 range)
    pub fn apply(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let r_out = self.lookup_1d(&self.r_table, r);
        let g_out = self.lookup_1d(&self.g_table, g);
        let b_out = self.lookup_1d(&self.b_table, b);

        (r_out, g_out, b_out)
    }

    /// Lookup value in 1D table with linear interpolation
    fn lookup_1d(&self, table: &[f32], value: f32) -> f32 {
        let value = value.clamp(0.0, 1.0);
        let scaled = value * (self.size - 1) as f32;
        let index = scaled.floor() as usize;
        let fraction = scaled - index as f32;

        if index >= self.size - 1 {
            table[self.size - 1]
        } else {
            // Linear interpolation
            let v0 = table[index];
            let v1 = table[index + 1];
            v0 + (v1 - v0) * fraction
        }
    }
}

/// 3D LUT for full RGB color space transformation
#[derive(Debug, Clone)]
pub struct Lut3D {
    /// LUT size (e.g., 32 for 32x32x32 cube)
    size: usize,
    /// 3D color mapping table (R, G, B) -> (R', G', B')
    /// Stored as flat array: [r][g][b] = index
    table: Vec<(f32, f32, f32)>,
    /// LUT title/name
    title: Option<String>,
    /// Input domain (min, max)
    domain_min: (f32, f32, f32),
    domain_max: (f32, f32, f32),
}

impl Lut3D {
    /// Create a new 3D LUT with identity mapping
    pub fn new(size: usize) -> Self {
        let total_entries = size * size * size;
        let mut table = Vec::with_capacity(total_entries);

        // Initialize with identity mapping
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let r_val = r as f32 / (size - 1) as f32;
                    let g_val = g as f32 / (size - 1) as f32;
                    let b_val = b as f32 / (size - 1) as f32;
                    table.push((r_val, g_val, b_val));
                }
            }
        }

        Lut3D {
            size,
            table,
            title: None,
            domain_min: (0.0, 0.0, 0.0),
            domain_max: (1.0, 1.0, 1.0),
        }
    }

    /// Load 3D LUT from Adobe .cube file
    pub fn from_cube_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref()).map_err(|e| Error::Io(e))?;
        let reader = BufReader::new(file);

        let mut size = None;
        let mut title = None;
        let mut domain_min = (0.0, 0.0, 0.0);
        let mut domain_max = (1.0, 1.0, 1.0);
        let mut table = Vec::new();

        for line in reader.lines() {
            let line = line.map_err(|e| Error::Io(e))?;
            let line = line.trim();

            // Skip comments and empty lines
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse title
            if line.starts_with("TITLE") {
                title = line
                    .split_once('"')
                    .and_then(|(_, rest)| rest.rsplit_once('"'))
                    .map(|(t, _)| t.to_string());
                continue;
            }

            // Parse LUT size
            if line.starts_with("LUT_3D_SIZE") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    size = parts[1].parse::<usize>().ok();
                }
                continue;
            }

            // Parse domain min
            if line.starts_with("DOMAIN_MIN") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    domain_min = (
                        parts[1].parse().unwrap_or(0.0),
                        parts[2].parse().unwrap_or(0.0),
                        parts[3].parse().unwrap_or(0.0),
                    );
                }
                continue;
            }

            // Parse domain max
            if line.starts_with("DOMAIN_MAX") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    domain_max = (
                        parts[1].parse().unwrap_or(1.0),
                        parts[2].parse().unwrap_or(1.0),
                        parts[3].parse().unwrap_or(1.0),
                    );
                }
                continue;
            }

            // Parse LUT data (RGB triplets)
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                if let (Ok(r), Ok(g), Ok(b)) = (
                    parts[0].parse::<f32>(),
                    parts[1].parse::<f32>(),
                    parts[2].parse::<f32>(),
                ) {
                    table.push((r, g, b));
                }
            }
        }

        let size = size
            .ok_or_else(|| Error::InvalidInput("Missing LUT_3D_SIZE in .cube file".to_string()))?;

        let expected_entries = size * size * size;
        if table.len() != expected_entries {
            return Err(Error::InvalidInput(format!(
                "Invalid .cube file: expected {} entries, got {}",
                expected_entries,
                table.len()
            )));
        }

        Ok(Lut3D {
            size,
            table,
            title,
            domain_min,
            domain_max,
        })
    }

    /// Save 3D LUT to Adobe .cube file
    pub fn save_cube_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        use std::io::Write;

        let mut file = File::create(path.as_ref()).map_err(|e| Error::Io(e))?;

        // Write header
        if let Some(ref title) = self.title {
            writeln!(file, "TITLE \"{}\"", title).map_err(|e| Error::Io(e))?;
        }

        writeln!(file, "LUT_3D_SIZE {}", self.size).map_err(|e| Error::Io(e))?;

        if self.domain_min != (0.0, 0.0, 0.0) {
            writeln!(
                file,
                "DOMAIN_MIN {:.6} {:.6} {:.6}",
                self.domain_min.0, self.domain_min.1, self.domain_min.2
            )
            .map_err(|e| Error::Io(e))?;
        }

        if self.domain_max != (1.0, 1.0, 1.0) {
            writeln!(
                file,
                "DOMAIN_MAX {:.6} {:.6} {:.6}",
                self.domain_max.0, self.domain_max.1, self.domain_max.2
            )
            .map_err(|e| Error::Io(e))?;
        }

        // Write LUT data
        for &(r, g, b) in &self.table {
            writeln!(file, "{:.6} {:.6} {:.6}", r, g, b).map_err(|e| Error::Io(e))?;
        }

        Ok(())
    }

    /// Apply 3D LUT to RGB values (0.0-1.0 range) using trilinear interpolation
    pub fn apply(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        // Normalize input to domain
        let r = self.normalize_input(r, self.domain_min.0, self.domain_max.0);
        let g = self.normalize_input(g, self.domain_min.1, self.domain_max.1);
        let b = self.normalize_input(b, self.domain_min.2, self.domain_max.2);

        // Clamp to valid range
        let r = r.clamp(0.0, 1.0);
        let g = g.clamp(0.0, 1.0);
        let b = b.clamp(0.0, 1.0);

        // Convert to LUT coordinates
        let size_f = (self.size - 1) as f32;
        let r_scaled = r * size_f;
        let g_scaled = g * size_f;
        let b_scaled = b * size_f;

        // Get integer indices and fractional parts
        let r_idx = r_scaled.floor() as usize;
        let g_idx = g_scaled.floor() as usize;
        let b_idx = b_scaled.floor() as usize;

        let r_frac = r_scaled - r_idx as f32;
        let g_frac = g_scaled - g_idx as f32;
        let b_frac = b_scaled - b_idx as f32;

        // Trilinear interpolation
        // Get 8 corner values of the cube
        let c000 = self.get_value(r_idx, g_idx, b_idx);
        let c001 = self.get_value(r_idx, g_idx, b_idx + 1);
        let c010 = self.get_value(r_idx, g_idx + 1, b_idx);
        let c011 = self.get_value(r_idx, g_idx + 1, b_idx + 1);
        let c100 = self.get_value(r_idx + 1, g_idx, b_idx);
        let c101 = self.get_value(r_idx + 1, g_idx, b_idx + 1);
        let c110 = self.get_value(r_idx + 1, g_idx + 1, b_idx);
        let c111 = self.get_value(r_idx + 1, g_idx + 1, b_idx + 1);

        // Interpolate along R axis (4 values)
        let c00 = Self::lerp_rgb(c000, c100, r_frac);
        let c01 = Self::lerp_rgb(c001, c101, r_frac);
        let c10 = Self::lerp_rgb(c010, c110, r_frac);
        let c11 = Self::lerp_rgb(c011, c111, r_frac);

        // Interpolate along G axis (2 values)
        let c0 = Self::lerp_rgb(c00, c10, g_frac);
        let c1 = Self::lerp_rgb(c01, c11, g_frac);

        // Interpolate along B axis (final value)
        Self::lerp_rgb(c0, c1, b_frac)
    }

    /// Apply LUT to RGB buffer (interleaved RGB)
    pub fn apply_to_buffer(&self, rgb_data: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
        if rgb_data.len() != width * height * 3 {
            return Err(Error::InvalidInput("Invalid RGB buffer size".to_string()));
        }

        let mut output = vec![0u8; rgb_data.len()];

        for i in 0..(width * height) {
            let idx = i * 3;

            // Normalize to 0.0-1.0
            let r = rgb_data[idx] as f32 / 255.0;
            let g = rgb_data[idx + 1] as f32 / 255.0;
            let b = rgb_data[idx + 2] as f32 / 255.0;

            // Apply LUT
            let (r_out, g_out, b_out) = self.apply(r, g, b);

            // Convert back to 0-255
            output[idx] = (r_out * 255.0).clamp(0.0, 255.0) as u8;
            output[idx + 1] = (g_out * 255.0).clamp(0.0, 255.0) as u8;
            output[idx + 2] = (b_out * 255.0).clamp(0.0, 255.0) as u8;
        }

        Ok(output)
    }

    /// Get value from 3D table with boundary checking
    fn get_value(&self, r: usize, g: usize, b: usize) -> (f32, f32, f32) {
        let r = r.min(self.size - 1);
        let g = g.min(self.size - 1);
        let b = b.min(self.size - 1);

        let index = r * self.size * self.size + g * self.size + b;
        self.table[index]
    }

    /// Linear interpolation between two RGB values
    fn lerp_rgb(a: (f32, f32, f32), b: (f32, f32, f32), t: f32) -> (f32, f32, f32) {
        (
            a.0 + (b.0 - a.0) * t,
            a.1 + (b.1 - a.1) * t,
            a.2 + (b.2 - a.2) * t,
        )
    }

    /// Normalize input value based on domain
    fn normalize_input(&self, value: f32, min: f32, max: f32) -> f32 {
        if max - min == 0.0 {
            value
        } else {
            (value - min) / (max - min)
        }
    }

    /// Set LUT title
    pub fn set_title(&mut self, title: String) {
        self.title = Some(title);
    }

    /// Get LUT size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get LUT title
    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    /// Create a common LUT (film emulation, color correction, etc.)
    pub fn preset(name: &str, size: usize) -> Result<Self> {
        let mut lut = Lut3D::new(size);

        match name.to_lowercase().as_str() {
            "sepia" => {
                lut.title = Some("Sepia Tone".to_string());
                // Apply sepia transformation to each entry
                for (r, g, b) in &mut lut.table {
                    let new_r = (*r * 0.393 + *g * 0.769 + *b * 0.189).min(1.0);
                    let new_g = (*r * 0.349 + *g * 0.686 + *b * 0.168).min(1.0);
                    let new_b = (*r * 0.272 + *g * 0.534 + *b * 0.131).min(1.0);
                    *r = new_r;
                    *g = new_g;
                    *b = new_b;
                }
            }
            "cool" => {
                lut.title = Some("Cool Tone".to_string());
                // Boost blues, reduce reds
                for (r, g, b) in &mut lut.table {
                    *r *= 0.9;
                    *b *= 1.1;
                }
            }
            "warm" => {
                lut.title = Some("Warm Tone".to_string());
                // Boost reds, reduce blues
                for (r, g, b) in &mut lut.table {
                    *r *= 1.1;
                    *b *= 0.9;
                }
            }
            "desaturate" => {
                lut.title = Some("Desaturate 50%".to_string());
                // Reduce saturation by 50%
                for (r, g, b) in &mut lut.table {
                    let gray = *r * 0.299 + *g * 0.587 + *b * 0.114;
                    *r = *r * 0.5 + gray * 0.5;
                    *g = *g * 0.5 + gray * 0.5;
                    *b = *b * 0.5 + gray * 0.5;
                }
            }
            _ => {
                return Err(Error::InvalidInput(format!("Unknown preset: {}", name)));
            }
        }

        Ok(lut)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lut3d_creation() {
        let lut = Lut3D::new(8);
        assert_eq!(lut.size(), 8);
        assert_eq!(lut.table.len(), 8 * 8 * 8);
    }

    #[test]
    fn test_lut3d_identity() {
        let lut = Lut3D::new(32);

        // Identity LUT should return input values
        let (r, g, b) = lut.apply(0.5, 0.7, 0.3);
        assert!((r - 0.5).abs() < 0.01);
        assert!((g - 0.7).abs() < 0.01);
        assert!((b - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_lut3d_sepia() {
        let lut = Lut3D::preset("sepia", 16).unwrap();

        // Apply to mid-gray
        let (r, g, b) = lut.apply(0.5, 0.5, 0.5);

        // Sepia should be warmer: red > green > blue
        // For 0.5 input: r=0.675, g=0.601, b=0.468 (standard sepia coefficients)
        assert!(r > g);
        assert!(g > b);
        assert!(r > 0.6);
        assert!(b < r);
    }

    #[test]
    fn test_lut3d_cool_warm() {
        let cool = Lut3D::preset("cool", 16).unwrap();
        let warm = Lut3D::preset("warm", 16).unwrap();

        let (r_cool, _, b_cool) = cool.apply(0.5, 0.5, 0.5);
        let (r_warm, _, b_warm) = warm.apply(0.5, 0.5, 0.5);

        // Cool should have less red, more blue
        assert!(r_cool < r_warm);
        assert!(b_cool > b_warm);
    }

    #[test]
    fn test_lut3d_buffer_application() {
        let lut = Lut3D::new(8);

        // Create simple RGB buffer (2x2 pixels)
        let rgb_data = vec![
            255, 0, 0, // Red
            0, 255, 0, // Green
            0, 0, 255, // Blue
            128, 128, 128, // Gray
        ];

        let output = lut.apply_to_buffer(&rgb_data, 2, 2).unwrap();

        // Identity LUT should preserve values (with small rounding errors)
        for i in 0..rgb_data.len() {
            assert!((rgb_data[i] as i32 - output[i] as i32).abs() <= 2);
        }
    }

    #[test]
    fn test_lut3d_clamping() {
        let lut = Lut3D::new(8);

        // Test values outside 0-1 range
        let (r, g, b) = lut.apply(-0.5, 1.5, 0.5);

        // Should be clamped to 0-1
        assert!(r >= 0.0 && r <= 1.0);
        assert!(g >= 0.0 && g <= 1.0);
        assert!(b >= 0.0 && b <= 1.0);
    }

    #[test]
    fn test_lut1d_creation() {
        let lut = Lut1D::new(256);
        assert_eq!(lut.size, 256);
    }

    #[test]
    fn test_lut1d_application() {
        let mut lut = Lut1D::new(256);

        // Create simple gamma curve (gamma = 2.2)
        for i in 0..256 {
            let value = (i as f32 / 255.0).powf(1.0 / 2.2);
            lut.r_table[i] = value;
            lut.g_table[i] = value;
            lut.b_table[i] = value;
        }

        let (r, g, b) = lut.apply(0.5, 0.5, 0.5);

        // Gamma 2.2 should brighten mid-tones
        assert!(r > 0.5);
        assert!(g > 0.5);
        assert!(b > 0.5);
    }

    #[test]
    fn test_lut3d_trilinear_interpolation() {
        let lut = Lut3D::new(2); // Minimal size for testing interpolation

        // Test midpoint interpolation
        let (r, g, b) = lut.apply(0.5, 0.5, 0.5);

        // Should be approximately 0.5 for all channels
        assert!((r - 0.5).abs() < 0.01);
        assert!((g - 0.5).abs() < 0.01);
        assert!((b - 0.5).abs() < 0.01);
    }
}
