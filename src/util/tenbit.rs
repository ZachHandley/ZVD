//! 10-bit Color Depth Processing
//!
//! Professional video processing with 10-bit color depth for:
//! - HDR workflows (HDR10, HLG)
//! - High-end color grading
//! - Broadcast production
//! - Reduced banding in gradients
//!
//! ## 10-bit Advantages
//!
//! - **1024 levels** per channel (vs 256 in 8-bit)
//! - **4x better** gradient smoothness
//! - Essential for HDR (PQ, HLG)
//! - Professional broadcast standard (Rec. 2020)
//!
//! ## Pixel Formats
//!
//! - **YUV420P10LE**: 10-bit 4:2:0 chroma subsampling
//! - **YUV444P10LE**: 10-bit 4:4:4 full chroma
//! - **RGB48LE**: 10-bit RGB (16-bit storage per channel)
//!
//! ## Usage
//!
//! ```rust
//! use zvd_lib::util::tenbit::{TenBitConverter, DitherMethod};
//!
//! // Convert 8-bit to 10-bit
//! let converter = TenBitConverter::new();
//! let frame_10bit = converter.upscale_8bit_to_10bit(&frame_8bit)?;
//!
//! // Convert 10-bit to 8-bit with dithering
//! let frame_8bit = converter.downscale_10bit_to_8bit(&frame_10bit, DitherMethod::Ordered)?;
//! ```

use crate::error::{Error, Result};

/// 10-bit pixel format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TenBitFormat {
    /// YUV 4:2:0 10-bit (most common)
    YUV420P10LE,
    /// YUV 4:4:4 10-bit (full chroma)
    YUV444P10LE,
    /// RGB 10-bit (stored as 16-bit per channel)
    RGB48LE,
}

impl TenBitFormat {
    /// Get bytes per pixel
    pub fn bytes_per_pixel(&self) -> usize {
        match self {
            TenBitFormat::YUV420P10LE => 2, // 2 bytes per sample (10 bits + padding)
            TenBitFormat::YUV444P10LE => 2,
            TenBitFormat::RGB48LE => 6, // 2 bytes × 3 channels
        }
    }

    /// Calculate buffer size for resolution
    pub fn buffer_size(&self, width: usize, height: usize) -> usize {
        match self {
            TenBitFormat::YUV420P10LE => {
                let y_size = width * height * 2; // 16-bit per pixel
                let uv_size = (width / 2) * (height / 2) * 2 * 2; // 2 planes, 16-bit
                y_size + uv_size
            }
            TenBitFormat::YUV444P10LE => {
                width * height * 3 * 2 // 3 planes, 16-bit per pixel
            }
            TenBitFormat::RGB48LE => {
                width * height * 6 // 3 channels × 2 bytes
            }
        }
    }
}

/// Dithering method for bit depth reduction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DitherMethod {
    /// No dithering (simple truncation)
    None,
    /// Ordered dithering (Bayer matrix)
    Ordered,
    /// Floyd-Steinberg error diffusion
    FloydSteinberg,
    /// Simple random noise
    Random,
}

/// 10-bit frame data
#[derive(Debug, Clone)]
pub struct TenBitFrame {
    pub width: usize,
    pub height: usize,
    pub format: TenBitFormat,
    /// Data stored as u16 (10 bits + 6 bits padding)
    pub data: Vec<u16>,
}

impl TenBitFrame {
    /// Create new 10-bit frame
    pub fn new(width: usize, height: usize, format: TenBitFormat) -> Self {
        let sample_count = match format {
            TenBitFormat::YUV420P10LE => {
                let y_samples = width * height;
                let uv_samples = (width / 2) * (height / 2) * 2;
                y_samples + uv_samples
            }
            TenBitFormat::YUV444P10LE => width * height * 3,
            TenBitFormat::RGB48LE => width * height * 3,
        };

        TenBitFrame {
            width,
            height,
            format,
            data: vec![0u16; sample_count],
        }
    }

    /// Get Y plane (for YUV formats)
    pub fn y_plane(&self) -> &[u16] {
        match self.format {
            TenBitFormat::YUV420P10LE | TenBitFormat::YUV444P10LE => {
                &self.data[0..self.width * self.height]
            }
            _ => &[],
        }
    }

    /// Get Y plane mutable
    pub fn y_plane_mut(&mut self) -> &mut [u16] {
        let size = self.width * self.height;
        match self.format {
            TenBitFormat::YUV420P10LE | TenBitFormat::YUV444P10LE => &mut self.data[0..size],
            _ => &mut [],
        }
    }
}

/// 10-bit converter
pub struct TenBitConverter {
    /// Random state for dithering
    random_seed: u32,
}

impl TenBitConverter {
    /// Create new converter
    pub fn new() -> Self {
        TenBitConverter { random_seed: 12345 }
    }

    /// Upscale 8-bit to 10-bit
    ///
    /// Simple bit shifting: value * 4 (or value << 2)
    /// Maps 0-255 to 0-1020 (full 10-bit range is 0-1023)
    pub fn upscale_8bit_to_10bit(&self, data_8bit: &[u8]) -> Vec<u16> {
        data_8bit
            .iter()
            .map(|&v| {
                // Scale 8-bit to 10-bit: multiply by 4.011764 ≈ 4
                // Use u32 to avoid overflow: 255 * 1023 = 260,865 > u16::MAX
                ((v as u32 * 1023) / 255) as u16
            })
            .collect()
    }

    /// Downscale 10-bit to 8-bit with dithering
    pub fn downscale_10bit_to_8bit(
        &mut self,
        data_10bit: &[u16],
        width: usize,
        height: usize,
        method: DitherMethod,
    ) -> Vec<u8> {
        match method {
            DitherMethod::None => self.downscale_truncate(data_10bit),
            DitherMethod::Ordered => self.downscale_ordered(data_10bit, width, height),
            DitherMethod::FloydSteinberg => {
                self.downscale_floyd_steinberg(data_10bit, width, height)
            }
            DitherMethod::Random => self.downscale_random(data_10bit),
        }
    }

    /// Simple truncation (no dithering)
    fn downscale_truncate(&self, data_10bit: &[u16]) -> Vec<u8> {
        data_10bit
            .iter()
            .map(|&v| {
                // Scale 10-bit to 8-bit: divide by 4.011764 ≈ 4
                // Better: (v * 255) / 1023
                ((v.min(1023) as u32 * 255) / 1023) as u8
            })
            .collect()
    }

    /// Ordered dithering (Bayer 4x4 matrix)
    fn downscale_ordered(&self, data_10bit: &[u16], width: usize, height: usize) -> Vec<u8> {
        // Bayer 4x4 threshold matrix (scaled to 0-15)
        const BAYER_MATRIX: [[u8; 4]; 4] =
            [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];

        let mut result = Vec::with_capacity(data_10bit.len());

        for (idx, &value_10bit) in data_10bit.iter().enumerate() {
            let x = idx % width;
            let y = idx / width;

            // Get threshold from Bayer matrix
            let threshold = BAYER_MATRIX[y % 4][x % 4] as u16;

            // Add dither before scaling
            let dithered = value_10bit.saturating_add(threshold);

            // Scale to 8-bit
            let value_8bit = ((dithered.min(1023) as u32 * 255) / 1023) as u8;
            result.push(value_8bit);
        }

        result
    }

    /// Floyd-Steinberg error diffusion
    fn downscale_floyd_steinberg(
        &self,
        data_10bit: &[u16],
        width: usize,
        height: usize,
    ) -> Vec<u8> {
        let mut buffer: Vec<i32> = data_10bit.iter().map(|&v| v as i32).collect();
        let mut result = Vec::with_capacity(data_10bit.len());

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let old_value = buffer[idx];

                // Quantize to 8-bit
                let new_value = ((old_value.max(0).min(1023) as u32 * 255) / 1023) as u8;
                result.push(new_value);

                // Calculate error
                let new_value_10bit = ((new_value as i32 * 1023) / 255) as i32;
                let error = old_value - new_value_10bit;

                // Distribute error to neighbors (Floyd-Steinberg)
                if x + 1 < width {
                    buffer[idx + 1] += (error * 7) / 16;
                }
                if y + 1 < height {
                    if x > 0 {
                        buffer[idx + width - 1] += (error * 3) / 16;
                    }
                    buffer[idx + width] += (error * 5) / 16;
                    if x + 1 < width {
                        buffer[idx + width + 1] += error / 16;
                    }
                }
            }
        }

        result
    }

    /// Random dithering
    fn downscale_random(&mut self, data_10bit: &[u16]) -> Vec<u8> {
        data_10bit
            .iter()
            .map(|&v| {
                // Simple LCG random number generator
                self.random_seed = self
                    .random_seed
                    .wrapping_mul(1103515245)
                    .wrapping_add(12345);
                let noise = (self.random_seed >> 16) & 0xF; // 0-15

                let dithered = v.saturating_add(noise as u16);
                ((dithered.min(1023) as u32 * 255) / 1023) as u8
            })
            .collect()
    }

    /// Convert 8-bit YUV420P to 10-bit YUV420P10LE
    pub fn convert_yuv420p_8bit_to_10bit(
        &self,
        data_8bit: &[u8],
        width: usize,
        height: usize,
    ) -> Result<TenBitFrame> {
        let y_size = width * height;
        let uv_size = (width / 2) * (height / 2);

        if data_8bit.len() < y_size + 2 * uv_size {
            return Err(Error::InvalidInput("Invalid YUV420P data size".to_string()));
        }

        let mut frame = TenBitFrame::new(width, height, TenBitFormat::YUV420P10LE);

        // Convert Y plane
        for i in 0..y_size {
            frame.data[i] = (data_8bit[i] as u16 * 1023) / 255;
        }

        // Convert U and V planes
        for i in 0..uv_size * 2 {
            frame.data[y_size + i] = (data_8bit[y_size + i] as u16 * 1023) / 255;
        }

        Ok(frame)
    }

    /// Convert 10-bit YUV420P10LE to 8-bit YUV420P
    pub fn convert_yuv420p_10bit_to_8bit(
        &mut self,
        frame: &TenBitFrame,
        method: DitherMethod,
    ) -> Result<Vec<u8>> {
        if frame.format != TenBitFormat::YUV420P10LE {
            return Err(Error::InvalidInput(
                "Expected YUV420P10LE format".to_string(),
            ));
        }

        Ok(self.downscale_10bit_to_8bit(&frame.data, frame.width, frame.height, method))
    }
}

impl Default for TenBitConverter {
    fn default() -> Self {
        Self::new()
    }
}

/// 10-bit statistics and analysis
pub struct TenBitAnalyzer;

impl TenBitAnalyzer {
    /// Calculate effective bit depth usage
    ///
    /// Returns how many unique values are actually used (0-1023)
    pub fn effective_bit_depth(data: &[u16]) -> f64 {
        let unique_values: std::collections::HashSet<u16> =
            data.iter().map(|&v| v.min(1023)).collect();

        let used_levels = unique_values.len() as f64;
        let max_levels = 1024.0_f64;

        (used_levels.log2() / max_levels.log2()) * 10.0
    }

    /// Detect banding in gradients
    ///
    /// Returns percentage of adjacent pixels with large jumps
    pub fn detect_banding(data: &[u16], threshold: u16) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }

        let mut jumps = 0;
        for i in 0..data.len() - 1 {
            let diff = (data[i] as i32 - data[i + 1] as i32).unsigned_abs() as u16;
            if diff > threshold {
                jumps += 1;
            }
        }

        (jumps as f64 / (data.len() - 1) as f64) * 100.0
    }

    /// Check if data is actually 10-bit or upscaled 8-bit
    pub fn is_true_10bit(data: &[u16]) -> bool {
        // True 10-bit should use values not divisible by 4
        // (8-bit upscaled is always divisible by ~4)
        let non_divisible = data.iter().filter(|&&v| v % 4 != 0).count();
        let ratio = non_divisible as f64 / data.len() as f64;

        ratio > 0.1 // If >10% values are not divisible by 4, likely true 10-bit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upscale_8bit_to_10bit() {
        let converter = TenBitConverter::new();
        let data_8bit = vec![0u8, 128, 255];
        let data_10bit = converter.upscale_8bit_to_10bit(&data_8bit);

        assert_eq!(data_10bit[0], 0); // 0 -> 0
        assert_eq!(data_10bit[1], 513); // 128 -> ~512
        assert_eq!(data_10bit[2], 1023); // 255 -> 1023
    }

    #[test]
    fn test_downscale_truncate() {
        let converter = TenBitConverter::new();
        let data_10bit = vec![0u16, 512, 1023];
        let data_8bit = converter.downscale_truncate(&data_10bit);

        assert_eq!(data_8bit[0], 0); // 0 -> 0
        assert_eq!(data_8bit[1], 127); // 512 -> ~127
        assert_eq!(data_8bit[2], 255); // 1023 -> 255
    }

    #[test]
    fn test_roundtrip_conversion() {
        let converter = TenBitConverter::new();
        let original = vec![0u8, 64, 128, 192, 255];

        let upscaled = converter.upscale_8bit_to_10bit(&original);
        let downscaled = converter.downscale_truncate(&upscaled);

        // Should be very close (within rounding error)
        for (orig, result) in original.iter().zip(downscaled.iter()) {
            assert!((*orig as i16 - *result as i16).abs() <= 1);
        }
    }

    #[test]
    fn test_tenbit_format_buffer_size() {
        let yuv420 = TenBitFormat::YUV420P10LE;
        let size = yuv420.buffer_size(1920, 1080);

        let expected_y = 1920 * 1080 * 2;
        let expected_uv = 960 * 540 * 2 * 2;
        assert_eq!(size, expected_y + expected_uv);
    }

    #[test]
    fn test_tenbit_frame_creation() {
        let frame = TenBitFrame::new(1920, 1080, TenBitFormat::YUV420P10LE);

        assert_eq!(frame.width, 1920);
        assert_eq!(frame.height, 1080);
        assert!(!frame.data.is_empty());
    }

    #[test]
    fn test_dither_ordered() {
        let mut converter = TenBitConverter::new();
        let data_10bit = vec![512u16; 64 * 64]; // Uniform value

        let dithered =
            converter.downscale_10bit_to_8bit(&data_10bit, 64, 64, DitherMethod::Ordered);

        // Should have some variation due to dithering
        let unique_values: std::collections::HashSet<u8> = dithered.iter().copied().collect();
        assert!(unique_values.len() > 1);
    }

    #[test]
    fn test_dither_random() {
        let mut converter = TenBitConverter::new();
        let data_10bit = vec![512u16; 100];

        let dithered = converter.downscale_10bit_to_8bit(&data_10bit, 100, 1, DitherMethod::Random);

        // Should have some variation
        let unique_values: std::collections::HashSet<u8> = dithered.iter().copied().collect();
        assert!(unique_values.len() > 1);
    }

    #[test]
    fn test_effective_bit_depth() {
        // Test with 8-bit upscaled (only 256 values)
        let data_8bit_upscaled: Vec<u16> = (0..256).map(|v| v * 4).collect();
        let depth = TenBitAnalyzer::effective_bit_depth(&data_8bit_upscaled);
        assert!(depth < 9.0); // Should be close to 8-bit

        // Test with true 10-bit (1024 values)
        let data_10bit: Vec<u16> = (0..1024).collect();
        let depth_10 = TenBitAnalyzer::effective_bit_depth(&data_10bit);
        assert!(depth_10 > 9.5); // Should be close to 10-bit
    }

    #[test]
    fn test_is_true_10bit() {
        // Upscaled 8-bit (divisible by 4)
        let fake_10bit: Vec<u16> = (0..256).map(|v| v * 4).collect();
        assert!(!TenBitAnalyzer::is_true_10bit(&fake_10bit));

        // True 10-bit (not divisible by 4)
        let true_10bit: Vec<u16> = (0..1024).collect();
        assert!(TenBitAnalyzer::is_true_10bit(&true_10bit));
    }

    #[test]
    fn test_banding_detection() {
        // Smooth gradient (no banding)
        let smooth: Vec<u16> = (0..1024).collect();
        let banding_smooth = TenBitAnalyzer::detect_banding(&smooth, 10);
        assert!(banding_smooth < 1.0);

        // Banded gradient (large jumps)
        let banded: Vec<u16> = (0..100).map(|v| v * 10).collect();
        let banding_bad = TenBitAnalyzer::detect_banding(&banded, 5);
        assert!(banding_bad > 50.0);
    }
}
