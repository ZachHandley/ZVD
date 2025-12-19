//! Professional Video Scopes (Waveform, Vectorscope, Histogram)
//!
//! Industry-standard video analysis and monitoring tools for broadcast,
//! color grading, and quality control workflows.
//!
//! ## Scopes Implemented
//!
//! - **RGB Parade**: Separate R, G, B waveforms side-by-side
//! - **Luma Waveform**: Luminance (Y) levels across frame
//! - **YUV Vectorscope**: Chrominance (U/V) phase and amplitude
//! - **RGB Histogram**: Red, Green, Blue distribution
//! - **Luma Histogram**: Brightness distribution
//!
//! ## Use Cases
//!
//! - Exposure checking (clipping detection)
//! - Color balance verification
//! - Broadcast legal range compliance
//! - Skin tone accuracy
//! - White/black point setting
//! - HDR/SDR level verification
//!
//! ## Usage
//!
//! ```rust
//! use zvd_lib::util::scopes::{WaveformScope, VectorscopeYUV, Histogram};
//!
//! // RGB Parade
//! let waveform = WaveformScope::rgb_parade(&rgb_data, width, height);
//! waveform.save_image("rgb_parade.png")?;
//!
//! // Vectorscope
//! let vectorscope = VectorscopeYUV::new(&yuv_data, width, height)?;
//! vectorscope.save_image("vectorscope.png")?;
//!
//! // Histogram
//! let histogram = Histogram::from_rgb(&rgb_data);
//! println!("Peak R: {}", histogram.r_peak());
//! ```

use crate::error::{Error, Result};
use image::{ImageBuffer, Rgb, RgbImage};
use std::path::Path;

/// Waveform display mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveformMode {
    /// RGB Parade (R, G, B side-by-side)
    RGBParade,
    /// Luma only (Y channel)
    Luma,
    /// RGB Overlay (all channels overlaid)
    RGBOverlay,
}

/// Waveform scope
pub struct WaveformScope {
    width: usize,
    height: usize,
    mode: WaveformMode,
    data: Vec<Vec<u32>>, // Accumulator: [channel][position][value]
}

impl WaveformScope {
    /// Create RGB Parade waveform
    pub fn rgb_parade(rgb_data: &[u8], frame_width: usize, frame_height: usize) -> Self {
        let scope_width = frame_width;
        let scope_height = 256; // 0-255 range

        let mut scope = WaveformScope {
            width: scope_width,
            height: scope_height,
            mode: WaveformMode::RGBParade,
            data: vec![vec![0u32; scope_height]; scope_width * 3], // R, G, B sections
        };

        scope.process_rgb(rgb_data, frame_width, frame_height);
        scope
    }

    /// Create Luma waveform
    pub fn luma(rgb_data: &[u8], frame_width: usize, frame_height: usize) -> Self {
        let scope_width = frame_width;
        let scope_height = 256;

        let mut scope = WaveformScope {
            width: scope_width,
            height: scope_height,
            mode: WaveformMode::Luma,
            data: vec![vec![0u32; scope_height]; scope_width],
        };

        scope.process_luma(rgb_data, frame_width, frame_height);
        scope
    }

    fn process_rgb(&mut self, rgb_data: &[u8], frame_width: usize, frame_height: usize) {
        let section_width = self.width / 3;

        for y in 0..frame_height {
            for x in 0..frame_width {
                let idx = (y * frame_width + x) * 3;
                if idx + 2 >= rgb_data.len() {
                    continue;
                }

                let r = rgb_data[idx] as usize;
                let g = rgb_data[idx + 1] as usize;
                let b = rgb_data[idx + 2] as usize;

                // Map to scope position
                let scope_x = (x * self.width) / frame_width;

                // R channel (left third)
                if scope_x < section_width && r < self.height {
                    self.data[scope_x][r] += 1;
                }

                // G channel (middle third)
                let g_x = section_width + ((scope_x * section_width) / self.width);
                if g_x < section_width * 2 && g < self.height {
                    self.data[g_x][g] += 1;
                }

                // B channel (right third)
                let b_x = section_width * 2 + ((scope_x * section_width) / self.width);
                if b_x < self.width && b < self.height {
                    self.data[b_x][b] += 1;
                }
            }
        }
    }

    fn process_luma(&mut self, rgb_data: &[u8], frame_width: usize, frame_height: usize) {
        for y in 0..frame_height {
            for x in 0..frame_width {
                let idx = (y * frame_width + x) * 3;
                if idx + 2 >= rgb_data.len() {
                    continue;
                }

                // Calculate luma (ITU-R BT.709)
                let r = rgb_data[idx] as f32;
                let g = rgb_data[idx + 1] as f32;
                let b = rgb_data[idx + 2] as f32;
                let luma = (0.2126 * r + 0.7152 * g + 0.0722 * b) as usize;

                let scope_x = (x * self.width) / frame_width;
                if luma < self.height {
                    self.data[scope_x][luma] += 1;
                }
            }
        }
    }

    /// Generate scope image
    pub fn to_image(&self) -> RgbImage {
        let mut img =
            ImageBuffer::from_pixel(self.width as u32, self.height as u32, Rgb([0u8, 0u8, 0u8]));

        // Find max value for normalization
        let max_val = self
            .data
            .iter()
            .flat_map(|col| col.iter())
            .max()
            .copied()
            .unwrap_or(1)
            .max(1);

        match self.mode {
            WaveformMode::RGBParade => {
                let section_width = self.width / 3;

                for x in 0..self.width {
                    for y in 0..self.height {
                        let value = self.data[x][y];
                        let intensity = ((value as f32 / max_val as f32) * 255.0) as u8;

                        let pixel = if x < section_width {
                            // R section
                            Rgb([intensity, 0, 0])
                        } else if x < section_width * 2 {
                            // G section
                            Rgb([0, intensity, 0])
                        } else {
                            // B section
                            Rgb([0, 0, intensity])
                        };

                        // Flip Y (top = 255, bottom = 0)
                        img.put_pixel(x as u32, (self.height - 1 - y) as u32, pixel);
                    }
                }
            }
            WaveformMode::Luma => {
                for x in 0..self.width {
                    for y in 0..self.height {
                        let value = self.data[x][y];
                        let intensity = ((value as f32 / max_val as f32) * 255.0) as u8;

                        // Flip Y
                        img.put_pixel(
                            x as u32,
                            (self.height - 1 - y) as u32,
                            Rgb([intensity, intensity, intensity]),
                        );
                    }
                }
            }
            WaveformMode::RGBOverlay => {
                // Not implemented in this version
            }
        }

        img
    }

    /// Save scope to image file
    pub fn save_image(&self, path: &Path) -> Result<()> {
        let img = self.to_image();
        img.save(path)
            .map_err(|e| Error::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
        Ok(())
    }
}

/// YUV Vectorscope
pub struct VectorscopeYUV {
    size: usize,
    data: Vec<Vec<u32>>, // [u][v] accumulator
}

impl VectorscopeYUV {
    /// Create vectorscope from YUV420P data
    pub fn new(yuv_data: &[u8], width: usize, height: usize) -> Result<Self> {
        let size = 256;
        let mut scope = VectorscopeYUV {
            size,
            data: vec![vec![0u32; size]; size],
        };

        scope.process_yuv420p(yuv_data, width, height)?;
        Ok(scope)
    }

    fn process_yuv420p(&mut self, yuv_data: &[u8], width: usize, height: usize) -> Result<()> {
        let y_size = width * height;
        let uv_width = width / 2;
        let uv_height = height / 2;
        let uv_size = uv_width * uv_height;

        if yuv_data.len() < y_size + 2 * uv_size {
            return Err(Error::InvalidInput("Invalid YUV420P data size".to_string()));
        }

        let u_plane = &yuv_data[y_size..y_size + uv_size];
        let v_plane = &yuv_data[y_size + uv_size..y_size + 2 * uv_size];

        for i in 0..uv_size {
            let u = u_plane[i] as usize;
            let v = v_plane[i] as usize;

            if u < self.size && v < self.size {
                self.data[u][v] += 1;
            }
        }

        Ok(())
    }

    /// Generate vectorscope image with graticule
    pub fn to_image(&self) -> RgbImage {
        let mut img =
            ImageBuffer::from_pixel(self.size as u32, self.size as u32, Rgb([0u8, 0u8, 0u8]));

        // Find max for normalization
        let max_val = self
            .data
            .iter()
            .flat_map(|row| row.iter())
            .max()
            .copied()
            .unwrap_or(1)
            .max(1);

        // Draw scope data
        for u in 0..self.size {
            for v in 0..self.size {
                let value = self.data[u][v];
                let intensity = ((value as f32 / max_val as f32) * 255.0) as u8;

                img.put_pixel(u as u32, v as u32, Rgb([intensity, intensity, intensity]));
            }
        }

        // Draw graticule (center crosshair and skin tone line)
        self.draw_graticule(&mut img);

        img
    }

    fn draw_graticule(&self, img: &mut RgbImage) {
        let center = self.size / 2;
        let gray = Rgb([64u8, 64u8, 64u8]);

        // Center crosshair
        for i in 0..self.size {
            img.put_pixel(center as u32, i as u32, gray);
            img.put_pixel(i as u32, center as u32, gray);
        }

        // Skin tone line (approximately 11 o'clock, 123 degrees)
        let skin_tone_angle = 123.0_f64.to_radians();
        for r in 0..self.size / 2 {
            let u = center as f64 + r as f64 * skin_tone_angle.cos();
            let v = center as f64 + r as f64 * skin_tone_angle.sin();

            if u >= 0.0 && u < self.size as f64 && v >= 0.0 && v < self.size as f64 {
                img.put_pixel(u as u32, v as u32, Rgb([128, 64, 64]));
            }
        }
    }

    /// Save vectorscope to image file
    pub fn save_image(&self, path: &Path) -> Result<()> {
        let img = self.to_image();
        img.save(path)
            .map_err(|e| Error::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
        Ok(())
    }

    /// Check if colors are within broadcast safe range
    pub fn broadcast_safe_percentage(&self) -> f64 {
        let safe_radius = 118; // Broadcast safe U/V range
        let center = self.size / 2;

        let mut total_pixels = 0u64;
        let mut safe_pixels = 0u64;

        for u in 0..self.size {
            for v in 0..self.size {
                let count = self.data[u][v] as u64;
                if count > 0 {
                    total_pixels += count;

                    let du = (u as i32 - center as i32).abs();
                    let dv = (v as i32 - center as i32).abs();
                    let distance = ((du * du + dv * dv) as f64).sqrt();

                    if distance <= safe_radius as f64 {
                        safe_pixels += count;
                    }
                }
            }
        }

        if total_pixels == 0 {
            100.0
        } else {
            (safe_pixels as f64 / total_pixels as f64) * 100.0
        }
    }
}

/// RGB Histogram
pub struct Histogram {
    r_bins: Vec<u32>,
    g_bins: Vec<u32>,
    b_bins: Vec<u32>,
    luma_bins: Vec<u32>,
}

impl Histogram {
    /// Create histogram from RGB data
    pub fn from_rgb(rgb_data: &[u8]) -> Self {
        let mut histogram = Histogram {
            r_bins: vec![0u32; 256],
            g_bins: vec![0u32; 256],
            b_bins: vec![0u32; 256],
            luma_bins: vec![0u32; 256],
        };

        for rgb in rgb_data.chunks_exact(3) {
            let r = rgb[0] as usize;
            let g = rgb[1] as usize;
            let b = rgb[2] as usize;

            histogram.r_bins[r] += 1;
            histogram.g_bins[g] += 1;
            histogram.b_bins[b] += 1;

            // Calculate luma
            let luma = (0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32) as usize;
            histogram.luma_bins[luma.min(255)] += 1;
        }

        histogram
    }

    /// Get R channel peak
    pub fn r_peak(&self) -> usize {
        self.r_bins
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Get G channel peak
    pub fn g_peak(&self) -> usize {
        self.g_bins
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Get B channel peak
    pub fn b_peak(&self) -> usize {
        self.b_bins
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Check if any channel is clipping (>1% at 0 or 255)
    pub fn is_clipping(&self) -> (bool, bool, bool) {
        let total: u32 = self.r_bins.iter().sum();
        let threshold = (total as f32 * 0.01) as u32;

        let r_clip = self.r_bins[0] > threshold || self.r_bins[255] > threshold;
        let g_clip = self.g_bins[0] > threshold || self.g_bins[255] > threshold;
        let b_clip = self.b_bins[0] > threshold || self.b_bins[255] > threshold;

        (r_clip, g_clip, b_clip)
    }

    /// Generate histogram image
    pub fn to_image(&self, width: usize, height: usize) -> RgbImage {
        let mut img = ImageBuffer::from_pixel(width as u32, height as u32, Rgb([0u8, 0u8, 0u8]));

        let max_r = *self.r_bins.iter().max().unwrap_or(&1);
        let max_g = *self.g_bins.iter().max().unwrap_or(&1);
        let max_b = *self.b_bins.iter().max().unwrap_or(&1);

        for x in 0..width {
            let bin = (x * 256) / width;

            let r_height = ((self.r_bins[bin] as f32 / max_r as f32) * height as f32) as usize;
            let g_height = ((self.g_bins[bin] as f32 / max_g as f32) * height as f32) as usize;
            let b_height = ((self.b_bins[bin] as f32 / max_b as f32) * height as f32) as usize;

            for y in 0..height {
                let from_bottom = height - 1 - y;

                let mut pixel = Rgb([0u8, 0u8, 0u8]);

                if from_bottom < r_height {
                    pixel[0] = 255;
                }
                if from_bottom < g_height {
                    pixel[1] = 255;
                }
                if from_bottom < b_height {
                    pixel[2] = 255;
                }

                img.put_pixel(x as u32, y as u32, pixel);
            }
        }

        img
    }

    /// Save histogram to image file
    pub fn save_image(&self, path: &Path, width: usize, height: usize) -> Result<()> {
        let img = self.to_image(width, height);
        img.save(path)
            .map_err(|e| Error::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_rgb(width: usize, height: usize, r: u8, g: u8, b: u8) -> Vec<u8> {
        vec![r, g, b].repeat(width * height)
    }

    #[test]
    fn test_waveform_rgb_parade() {
        let rgb = create_test_rgb(100, 100, 128, 64, 192);
        let waveform = WaveformScope::rgb_parade(&rgb, 100, 100);

        assert_eq!(waveform.mode, WaveformMode::RGBParade);
        assert_eq!(waveform.height, 256);
    }

    #[test]
    fn test_waveform_luma() {
        let rgb = create_test_rgb(100, 100, 128, 128, 128);
        let waveform = WaveformScope::luma(&rgb, 100, 100);

        assert_eq!(waveform.mode, WaveformMode::Luma);
    }

    #[test]
    fn test_vectorscope_creation() {
        let width = 64;
        let height = 64;
        let y_size = width * height;
        let uv_size = (width / 2) * (height / 2);
        let mut yuv = vec![128u8; y_size + 2 * uv_size];

        // Set U/V to center
        for i in y_size..y_size + 2 * uv_size {
            yuv[i] = 128;
        }

        let vectorscope = VectorscopeYUV::new(&yuv, width, height);
        assert!(vectorscope.is_ok());
    }

    #[test]
    fn test_vectorscope_broadcast_safe() {
        let width = 64;
        let height = 64;
        let y_size = width * height;
        let uv_size = (width / 2) * (height / 2);
        let mut yuv = vec![128u8; y_size + 2 * uv_size];

        // All chroma at center (very safe)
        for i in y_size..y_size + 2 * uv_size {
            yuv[i] = 128;
        }

        let vectorscope = VectorscopeYUV::new(&yuv, width, height).unwrap();
        let safe_pct = vectorscope.broadcast_safe_percentage();

        assert!(safe_pct > 99.0); // Should be 100% safe
    }

    #[test]
    fn test_histogram_creation() {
        let rgb = create_test_rgb(100, 100, 128, 64, 192);
        let histogram = Histogram::from_rgb(&rgb);

        assert_eq!(histogram.r_bins.len(), 256);
        assert_eq!(histogram.g_bins.len(), 256);
        assert_eq!(histogram.b_bins.len(), 256);
    }

    #[test]
    fn test_histogram_peaks() {
        let rgb = create_test_rgb(100, 100, 100, 150, 200);
        let histogram = Histogram::from_rgb(&rgb);

        assert_eq!(histogram.r_peak(), 100);
        assert_eq!(histogram.g_peak(), 150);
        assert_eq!(histogram.b_peak(), 200);
    }

    #[test]
    fn test_histogram_clipping() {
        // Create image with clipping
        let mut rgb = vec![255u8, 255u8, 255u8].repeat(1000); // Bright pixels
        rgb.extend(vec![0u8, 0u8, 0u8].repeat(10)); // Some black

        let histogram = Histogram::from_rgb(&rgb);
        let (r_clip, g_clip, b_clip) = histogram.is_clipping();

        assert!(r_clip);
        assert!(g_clip);
        assert!(b_clip);
    }

    #[test]
    fn test_histogram_no_clipping() {
        let rgb = create_test_rgb(100, 100, 128, 128, 128);
        let histogram = Histogram::from_rgb(&rgb);
        let (r_clip, g_clip, b_clip) = histogram.is_clipping();

        assert!(!r_clip);
        assert!(!g_clip);
        assert!(!b_clip);
    }
}
