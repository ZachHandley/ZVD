//! Video Denoising and Sharpening
//!
//! Advanced noise reduction and detail enhancement for video processing,
//! restoration, and quality improvement.
//!
//! ## Denoising Methods
//!
//! - **Spatial**: Process single frames (Gaussian blur, bilateral filter)
//! - **Temporal**: Process across frames (motion-compensated averaging)
//! - **Spatio-Temporal**: Combined approach for best results
//!
//! ## Sharpening Methods
//!
//! - **Unsharp Mask**: Classic sharpening (blur + subtract + add)
//! - **High-Pass Filter**: Enhance edges and detail
//! - **Adaptive Sharpening**: Vary strength based on content
//!
//! ## Common Use Cases
//!
//! - Low-light footage cleanup
//! - Vintage film restoration
//! - Compression artifact reduction
//! - Detail enhancement
//! - Broadcast quality improvement
//!
//! ## Usage
//!
//! ```rust
//! use zvd_lib::filter::denoise::{TemporalDenoiser, UnsharpMask};
//!
//! // Temporal denoising
//! let mut denoiser = TemporalDenoiser::new(3); // 3-frame window
//! denoiser.set_strength(0.5);
//! let denoised = denoiser.process(&frames)?;
//!
//! // Sharpening
//! let sharpen = UnsharpMask::new(1.0, 0.5, 0.0);
//! let sharpened = sharpen.process(&frame_data, width, height)?;
//! ```

use crate::error::{Error, Result};
use std::collections::VecDeque;

/// Spatial denoiser (single frame)
pub struct SpatialDenoiser {
    /// Filter strength (0.0 to 1.0)
    strength: f32,
    /// Filter radius (pixels)
    radius: usize,
}

impl SpatialDenoiser {
    /// Create new spatial denoiser
    pub fn new(strength: f32, radius: usize) -> Self {
        SpatialDenoiser {
            strength: strength.clamp(0.0, 1.0),
            radius,
        }
    }

    /// Process frame with Gaussian blur
    pub fn process(&self, frame_rgb: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
        if frame_rgb.len() != width * height * 3 {
            return Err(Error::InvalidInput("Invalid frame dimensions".to_string()));
        }

        // Apply separable Gaussian blur
        let kernel = self.gaussian_kernel(self.radius);
        let horizontal = self.convolve_horizontal(frame_rgb, width, height, &kernel)?;
        let blurred = self.convolve_vertical(&horizontal, width, height, &kernel)?;

        // Blend original with blurred based on strength
        let mut output = vec![0u8; frame_rgb.len()];
        for i in 0..frame_rgb.len() {
            let orig = frame_rgb[i] as f32;
            let blur = blurred[i] as f32;
            output[i] = (orig * (1.0 - self.strength) + blur * self.strength) as u8;
        }

        Ok(output)
    }

    /// Generate Gaussian kernel
    fn gaussian_kernel(&self, radius: usize) -> Vec<f32> {
        let sigma = radius as f32 / 2.0;
        let size = radius * 2 + 1;
        let mut kernel = vec![0.0; size];

        let mut sum = 0.0;
        for i in 0..size {
            let x = i as f32 - radius as f32;
            kernel[i] = (-x * x / (2.0 * sigma * sigma)).exp();
            sum += kernel[i];
        }

        // Normalize
        kernel.iter_mut().for_each(|k| *k /= sum);
        kernel
    }

    /// Horizontal convolution
    fn convolve_horizontal(
        &self,
        src: &[u8],
        width: usize,
        height: usize,
        kernel: &[f32],
    ) -> Result<Vec<u8>> {
        let mut output = vec![0u8; src.len()];
        let radius = kernel.len() / 2;

        for y in 0..height {
            for x in 0..width {
                for c in 0..3 {
                    let mut sum = 0.0;

                    for k in 0..kernel.len() {
                        let sample_x = (x as i32 + k as i32 - radius as i32)
                            .max(0)
                            .min(width as i32 - 1) as usize;

                        let idx = (y * width + sample_x) * 3 + c;
                        sum += src[idx] as f32 * kernel[k];
                    }

                    let out_idx = (y * width + x) * 3 + c;
                    output[out_idx] = sum.round().clamp(0.0, 255.0) as u8;
                }
            }
        }

        Ok(output)
    }

    /// Vertical convolution
    fn convolve_vertical(
        &self,
        src: &[u8],
        width: usize,
        height: usize,
        kernel: &[f32],
    ) -> Result<Vec<u8>> {
        let mut output = vec![0u8; src.len()];
        let radius = kernel.len() / 2;

        for y in 0..height {
            for x in 0..width {
                for c in 0..3 {
                    let mut sum = 0.0;

                    for k in 0..kernel.len() {
                        let sample_y = (y as i32 + k as i32 - radius as i32)
                            .max(0)
                            .min(height as i32 - 1) as usize;

                        let idx = (sample_y * width + x) * 3 + c;
                        sum += src[idx] as f32 * kernel[k];
                    }

                    let out_idx = (y * width + x) * 3 + c;
                    output[out_idx] = sum.round().clamp(0.0, 255.0) as u8;
                }
            }
        }

        Ok(output)
    }
}

/// Temporal denoiser (multi-frame)
pub struct TemporalDenoiser {
    /// Number of frames to average
    window_size: usize,
    /// Strength (0.0 to 1.0)
    strength: f32,
    /// Frame buffer
    frame_buffer: VecDeque<Vec<u8>>,
    /// Motion threshold
    motion_threshold: f32,
}

impl TemporalDenoiser {
    /// Create new temporal denoiser
    pub fn new(window_size: usize) -> Self {
        TemporalDenoiser {
            window_size: window_size.max(2),
            strength: 0.5,
            frame_buffer: VecDeque::with_capacity(window_size),
            motion_threshold: 20.0,
        }
    }

    /// Set denoising strength
    pub fn set_strength(&mut self, strength: f32) {
        self.strength = strength.clamp(0.0, 1.0);
    }

    /// Set motion threshold (for motion-adaptive filtering)
    pub fn set_motion_threshold(&mut self, threshold: f32) {
        self.motion_threshold = threshold.max(0.0);
    }

    /// Process frame
    pub fn process(&mut self, frame: &[u8]) -> Vec<u8> {
        // Add frame to buffer
        if self.frame_buffer.len() >= self.window_size {
            self.frame_buffer.pop_front();
        }
        self.frame_buffer.push_back(frame.to_vec());

        if self.frame_buffer.len() < 2 {
            return frame.to_vec();
        }

        // Temporal averaging with motion detection
        let mut output = vec![0u8; frame.len()];

        for i in 0..frame.len() {
            let current = frame[i] as f32;

            // Calculate average of buffer
            let mut sum = 0.0;
            let mut count = 0.0;

            for buffered_frame in &self.frame_buffer {
                let pixel = buffered_frame[i] as f32;
                let diff = (pixel - current).abs();

                // Weight by similarity (motion-adaptive)
                let weight = if diff < self.motion_threshold {
                    1.0
                } else {
                    1.0 / (1.0 + diff / self.motion_threshold)
                };

                sum += pixel * weight;
                count += weight;
            }

            let avg = sum / count;

            // Blend with original based on strength
            output[i] = (current * (1.0 - self.strength) + avg * self.strength)
                .round()
                .clamp(0.0, 255.0) as u8;
        }

        output
    }

    /// Reset buffer
    pub fn reset(&mut self) {
        self.frame_buffer.clear();
    }
}

/// Unsharp mask sharpening
pub struct UnsharpMask {
    /// Amount (0.0 to 2.0 typical)
    amount: f32,
    /// Radius (1.0 to 5.0 typical)
    radius: f32,
    /// Threshold (0.0 to 255.0)
    threshold: f32,
}

impl UnsharpMask {
    /// Create new unsharp mask
    ///
    /// # Arguments
    /// * `amount` - Sharpening strength (0.0-2.0)
    /// * `radius` - Blur radius (1.0-5.0)
    /// * `threshold` - Minimum difference to sharpen
    pub fn new(amount: f32, radius: f32, threshold: f32) -> Self {
        UnsharpMask {
            amount: amount.max(0.0),
            radius: radius.max(0.1),
            threshold: threshold.clamp(0.0, 255.0),
        }
    }

    /// Process frame
    pub fn process(&self, frame_rgb: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
        if frame_rgb.len() != width * height * 3 {
            return Err(Error::InvalidInput("Invalid frame dimensions".to_string()));
        }

        // Create blurred version
        let denoiser = SpatialDenoiser::new(1.0, self.radius as usize);
        let blurred = denoiser.process(frame_rgb, width, height)?;

        // Unsharp mask = original + amount * (original - blurred)
        let mut output = vec![0u8; frame_rgb.len()];

        for i in 0..frame_rgb.len() {
            let original = frame_rgb[i] as f32;
            let blur = blurred[i] as f32;
            let diff = original - blur;

            // Apply threshold
            let sharpened = if diff.abs() < self.threshold {
                original
            } else {
                original + self.amount * diff
            };

            output[i] = sharpened.round().clamp(0.0, 255.0) as u8;
        }

        Ok(output)
    }
}

/// Adaptive sharpening (content-aware)
pub struct AdaptiveSharpen {
    /// Base amount
    base_amount: f32,
    /// Edge threshold
    edge_threshold: f32,
}

impl AdaptiveSharpen {
    /// Create new adaptive sharpener
    pub fn new(base_amount: f32) -> Self {
        AdaptiveSharpen {
            base_amount: base_amount.max(0.0),
            edge_threshold: 30.0,
        }
    }

    /// Process frame with adaptive sharpening
    pub fn process(&self, frame_rgb: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
        if frame_rgb.len() != width * height * 3 {
            return Err(Error::InvalidInput("Invalid frame dimensions".to_string()));
        }

        let mut output = vec![0u8; frame_rgb.len()];

        // Laplacian kernel for edge detection
        let kernel = [-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0];

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                for c in 0..3 {
                    let center_idx = (y * width + x) * 3 + c;
                    let center = frame_rgb[center_idx] as f32;

                    // Apply Laplacian
                    let mut laplacian = 0.0;
                    let mut k_idx = 0;

                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let sample_y = (y as i32 + dy) as usize;
                            let sample_x = (x as i32 + dx) as usize;
                            let idx = (sample_y * width + sample_x) * 3 + c;

                            laplacian += frame_rgb[idx] as f32 * kernel[k_idx];
                            k_idx += 1;
                        }
                    }

                    // Adaptive amount based on edge strength
                    let edge_strength = laplacian.abs() / 8.0;
                    let adaptive_amount = if edge_strength > self.edge_threshold {
                        self.base_amount
                    } else {
                        self.base_amount * (edge_strength / self.edge_threshold)
                    };

                    let sharpened = center + adaptive_amount * laplacian;
                    output[center_idx] = sharpened.round().clamp(0.0, 255.0) as u8;
                }
            }
        }

        // Copy borders
        for y in 0..height {
            for x in 0..width {
                if y == 0 || y == height - 1 || x == 0 || x == width - 1 {
                    let idx = (y * width + x) * 3;
                    output[idx..idx + 3].copy_from_slice(&frame_rgb[idx..idx + 3]);
                }
            }
        }

        Ok(output)
    }
}

/// Bilateral filter (edge-preserving blur)
pub struct BilateralFilter {
    /// Spatial sigma
    spatial_sigma: f32,
    /// Range sigma (color similarity)
    range_sigma: f32,
    /// Filter radius
    radius: usize,
}

impl BilateralFilter {
    /// Create new bilateral filter
    pub fn new(spatial_sigma: f32, range_sigma: f32) -> Self {
        let radius = (spatial_sigma * 2.0) as usize;

        BilateralFilter {
            spatial_sigma,
            range_sigma,
            radius,
        }
    }

    /// Process frame
    pub fn process(&self, frame_rgb: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
        if frame_rgb.len() != width * height * 3 {
            return Err(Error::InvalidInput("Invalid frame dimensions".to_string()));
        }

        let mut output = vec![0u8; frame_rgb.len()];

        for y in 0..height {
            for x in 0..width {
                for c in 0..3 {
                    let center_idx = (y * width + x) * 3 + c;
                    let center_value = frame_rgb[center_idx] as f32;

                    let mut sum_weight = 0.0;
                    let mut sum_value = 0.0;

                    // Iterate over neighborhood
                    for dy in -(self.radius as i32)..=self.radius as i32 {
                        for dx in -(self.radius as i32)..=self.radius as i32 {
                            let sample_y = (y as i32 + dy).max(0).min(height as i32 - 1) as usize;
                            let sample_x = (x as i32 + dx).max(0).min(width as i32 - 1) as usize;

                            let sample_idx = (sample_y * width + sample_x) * 3 + c;
                            let sample_value = frame_rgb[sample_idx] as f32;

                            // Spatial weight (Gaussian)
                            let spatial_dist = ((dx * dx + dy * dy) as f32).sqrt();
                            let spatial_weight = (-spatial_dist * spatial_dist
                                / (2.0 * self.spatial_sigma * self.spatial_sigma))
                                .exp();

                            // Range weight (color similarity)
                            let color_diff = sample_value - center_value;
                            let range_weight = (-color_diff * color_diff
                                / (2.0 * self.range_sigma * self.range_sigma))
                                .exp();

                            let weight = spatial_weight * range_weight;

                            sum_weight += weight;
                            sum_value += sample_value * weight;
                        }
                    }

                    output[center_idx] = (sum_value / sum_weight).round().clamp(0.0, 255.0) as u8;
                }
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_denoiser() {
        let denoiser = SpatialDenoiser::new(0.5, 2);

        let frame = vec![128u8; 100 * 100 * 3];
        let result = denoiser.process(&frame, 100, 100);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), frame.len());
    }

    #[test]
    fn test_gaussian_kernel() {
        let denoiser = SpatialDenoiser::new(0.5, 2);
        let kernel = denoiser.gaussian_kernel(2);

        assert_eq!(kernel.len(), 5);

        // Sum should be approximately 1.0
        let sum: f32 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_temporal_denoiser() {
        let mut denoiser = TemporalDenoiser::new(3);
        denoiser.set_strength(0.5);

        let frame1 = vec![100u8; 100 * 100 * 3];
        let frame2 = vec![110u8; 100 * 100 * 3];
        let frame3 = vec![105u8; 100 * 100 * 3];

        denoiser.process(&frame1);
        denoiser.process(&frame2);
        let result = denoiser.process(&frame3);

        assert_eq!(result.len(), frame3.len());
    }

    #[test]
    fn test_temporal_reset() {
        let mut denoiser = TemporalDenoiser::new(3);

        let frame = vec![128u8; 100];
        denoiser.process(&frame);

        denoiser.reset();
        assert_eq!(denoiser.frame_buffer.len(), 0);
    }

    #[test]
    fn test_unsharp_mask() {
        let sharpen = UnsharpMask::new(1.0, 2.0, 0.0);

        let frame = vec![128u8; 100 * 100 * 3];
        let result = sharpen.process(&frame, 100, 100);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), frame.len());
    }

    #[test]
    fn test_unsharp_threshold() {
        let sharpen = UnsharpMask::new(1.0, 2.0, 50.0);

        let frame = vec![128u8; 50 * 50 * 3];
        let result = sharpen.process(&frame, 50, 50);

        assert!(result.is_ok());
    }

    #[test]
    fn test_adaptive_sharpen() {
        let sharpen = AdaptiveSharpen::new(0.5);

        let frame = vec![128u8; 100 * 100 * 3];
        let result = sharpen.process(&frame, 100, 100);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), frame.len());
    }

    #[test]
    fn test_bilateral_filter() {
        let filter = BilateralFilter::new(2.0, 25.0);

        let frame = vec![128u8; 50 * 50 * 3];
        let result = filter.process(&frame, 50, 50);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), frame.len());
    }

    #[test]
    fn test_invalid_dimensions() {
        let denoiser = SpatialDenoiser::new(0.5, 2);

        let frame = vec![128u8; 100];
        let result = denoiser.process(&frame, 100, 100);

        assert!(result.is_err());
    }

    #[test]
    fn test_strength_clamping() {
        let denoiser = SpatialDenoiser::new(2.0, 2); // Should clamp to 1.0
        assert_eq!(denoiser.strength, 1.0);

        let denoiser2 = SpatialDenoiser::new(-0.5, 2); // Should clamp to 0.0
        assert_eq!(denoiser2.strength, 0.0);
    }

    #[test]
    fn test_motion_threshold() {
        let mut denoiser = TemporalDenoiser::new(3);
        denoiser.set_motion_threshold(30.0);

        assert_eq!(denoiser.motion_threshold, 30.0);
    }

    #[test]
    fn test_window_size() {
        let denoiser = TemporalDenoiser::new(1); // Should be at least 2
        assert_eq!(denoiser.window_size, 2);
    }
}
