//! Video Quality Metrics
//!
//! This module implements industry-standard objective video quality metrics
//! for comparing original and compressed/processed video frames.
//!
//! ## Metrics Implemented
//!
//! - **PSNR** (Peak Signal-to-Noise Ratio): Simple MSE-based metric
//! - **SSIM** (Structural Similarity Index): Perceptually-weighted metric
//! - **MS-SSIM** (Multi-Scale SSIM): Enhanced SSIM with multi-scale analysis
//!
//! ## Standards
//!
//! - ITU-R BT.500: Methodology for subjective assessment
//! - ITU-T J.340: Transmission of multimedia signals
//! - SSIM: Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity"
//!
//! ## Usage
//!
//! ```rust
//! use zvd_lib::util::quality::{QualityMetrics, calculate_psnr, calculate_ssim};
//!
//! // Calculate PSNR
//! let psnr = calculate_psnr(&original, &compressed, width, height)?;
//! println!("PSNR: {:.2} dB", psnr);
//!
//! // Calculate SSIM
//! let ssim = calculate_ssim(&original, &compressed, width, height)?;
//! println!("SSIM: {:.4}", ssim);
//!
//! // Full quality report
//! let metrics = QualityMetrics::analyze(&original, &compressed, width, height)?;
//! println!("{}", metrics.summary());
//! ```

use crate::error::{Error, Result};

/// Complete quality metrics for a frame
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Peak Signal-to-Noise Ratio (dB)
    pub psnr_y: f64,
    pub psnr_u: f64,
    pub psnr_v: f64,
    pub psnr_avg: f64,

    /// Structural Similarity Index (0.0-1.0)
    pub ssim_y: f64,
    pub ssim_u: f64,
    pub ssim_v: f64,
    pub ssim_avg: f64,

    /// Multi-Scale SSIM (0.0-1.0)
    pub ms_ssim: Option<f64>,

    /// Mean Squared Error
    pub mse_y: f64,
    pub mse_u: f64,
    pub mse_v: f64,

    /// Frame dimensions
    pub width: usize,
    pub height: usize,
}

impl QualityMetrics {
    /// Analyze quality between reference and distorted frames
    ///
    /// # Arguments
    /// * `reference` - Original YUV420P data (Y plane, then U plane, then V plane)
    /// * `distorted` - Compressed/processed YUV420P data
    /// * `width` - Frame width
    /// * `height` - Frame height
    pub fn analyze(
        reference: &[u8],
        distorted: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Self> {
        if reference.len() != distorted.len() {
            return Err(Error::InvalidInput(
                "Reference and distorted frames must have same size".to_string(),
            ));
        }

        // Calculate plane sizes for YUV420P
        let y_size = width * height;
        let uv_size = (width / 2) * (height / 2);

        if reference.len() < y_size + 2 * uv_size {
            return Err(Error::InvalidInput(
                "Frame data too small for YUV420P format".to_string(),
            ));
        }

        // Extract planes
        let ref_y = &reference[0..y_size];
        let ref_u = &reference[y_size..y_size + uv_size];
        let ref_v = &reference[y_size + uv_size..y_size + 2 * uv_size];

        let dist_y = &distorted[0..y_size];
        let dist_u = &distorted[y_size..y_size + uv_size];
        let dist_v = &distorted[y_size + uv_size..y_size + 2 * uv_size];

        // Calculate MSE for each plane
        let mse_y = calculate_mse(ref_y, dist_y);
        let mse_u = calculate_mse(ref_u, dist_u);
        let mse_v = calculate_mse(ref_v, dist_v);

        // Calculate PSNR for each plane
        let psnr_y = mse_to_psnr(mse_y);
        let psnr_u = mse_to_psnr(mse_u);
        let psnr_v = mse_to_psnr(mse_v);

        // YUV PSNR average (weighted: Y has more importance)
        let psnr_avg = (6.0 * psnr_y + psnr_u + psnr_v) / 8.0;

        // Calculate SSIM for each plane
        let ssim_y = calculate_ssim_plane(ref_y, dist_y, width, height)?;
        let ssim_u = calculate_ssim_plane(ref_u, dist_u, width / 2, height / 2)?;
        let ssim_v = calculate_ssim_plane(ref_v, dist_v, width / 2, height / 2)?;

        // YUV SSIM average (weighted)
        let ssim_avg = (6.0 * ssim_y + ssim_u + ssim_v) / 8.0;

        // Calculate MS-SSIM (Y plane only for performance)
        let ms_ssim = calculate_ms_ssim(ref_y, dist_y, width, height).ok();

        Ok(QualityMetrics {
            psnr_y,
            psnr_u,
            psnr_v,
            psnr_avg,
            ssim_y,
            ssim_u,
            ssim_v,
            ssim_avg,
            ms_ssim,
            mse_y,
            mse_u,
            mse_v,
            width,
            height,
        })
    }

    /// Generate quality summary report
    pub fn summary(&self) -> String {
        let mut s = String::from("Video Quality Metrics:\n");
        s.push_str(&format!("Resolution: {}x{}\n\n", self.width, self.height));

        s.push_str("PSNR (Peak Signal-to-Noise Ratio):\n");
        s.push_str(&format!("  Y:   {:.2} dB\n", self.psnr_y));
        s.push_str(&format!("  U:   {:.2} dB\n", self.psnr_u));
        s.push_str(&format!("  V:   {:.2} dB\n", self.psnr_v));
        s.push_str(&format!("  Avg: {:.2} dB\n\n", self.psnr_avg));

        s.push_str("SSIM (Structural Similarity Index):\n");
        s.push_str(&format!("  Y:   {:.4}\n", self.ssim_y));
        s.push_str(&format!("  U:   {:.4}\n", self.ssim_u));
        s.push_str(&format!("  V:   {:.4}\n", self.ssim_v));
        s.push_str(&format!("  Avg: {:.4}\n", self.ssim_avg));

        if let Some(ms_ssim) = self.ms_ssim {
            s.push_str(&format!("\nMS-SSIM: {:.4}\n", ms_ssim));
        }

        s.push_str(&format!(
            "\nQuality Assessment: {}\n",
            self.quality_rating()
        ));

        s
    }

    /// Get quality rating based on PSNR and SSIM
    pub fn quality_rating(&self) -> &'static str {
        // Combined rating based on PSNR and SSIM
        if self.psnr_avg >= 40.0 && self.ssim_avg >= 0.98 {
            "Excellent (visually lossless)"
        } else if self.psnr_avg >= 35.0 && self.ssim_avg >= 0.95 {
            "Very Good (imperceptible quality loss)"
        } else if self.psnr_avg >= 30.0 && self.ssim_avg >= 0.90 {
            "Good (acceptable quality)"
        } else if self.psnr_avg >= 25.0 && self.ssim_avg >= 0.80 {
            "Fair (visible quality loss)"
        } else {
            "Poor (significant quality degradation)"
        }
    }
}

/// Calculate Mean Squared Error between two planes
fn calculate_mse(reference: &[u8], distorted: &[u8]) -> f64 {
    if reference.is_empty() || reference.len() != distorted.len() {
        return 0.0;
    }

    let sum: u64 = reference
        .iter()
        .zip(distorted.iter())
        .map(|(&r, &d)| {
            let diff = r as i32 - d as i32;
            (diff * diff) as u64
        })
        .sum();

    sum as f64 / reference.len() as f64
}

/// Convert MSE to PSNR (dB)
fn mse_to_psnr(mse: f64) -> f64 {
    if mse == 0.0 {
        f64::INFINITY // Perfect match
    } else {
        // PSNR = 10 * log10(MAX^2 / MSE)
        // For 8-bit data, MAX = 255
        10.0 * ((255.0 * 255.0) / mse).log10()
    }
}

/// Calculate PSNR for YUV420P frame
pub fn calculate_psnr(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
) -> Result<f64> {
    let metrics = QualityMetrics::analyze(reference, distorted, width, height)?;
    Ok(metrics.psnr_avg)
}

/// Calculate SSIM for a single plane
fn calculate_ssim_plane(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
) -> Result<f64> {
    if width < 8 || height < 8 {
        return Err(Error::InvalidInput(
            "Frame too small for SSIM calculation (min 8x8)".to_string(),
        ));
    }

    // SSIM parameters
    const K1: f64 = 0.01;
    const K2: f64 = 0.03;
    const L: f64 = 255.0; // Dynamic range for 8-bit

    let c1 = (K1 * L) * (K1 * L);
    let c2 = (K2 * L) * (K2 * L);

    // Use 8x8 blocks with stride of 4 (50% overlap)
    const BLOCK_SIZE: usize = 8;
    const STRIDE: usize = 4;

    let mut ssim_values = Vec::new();

    for y in (0..height - BLOCK_SIZE + 1).step_by(STRIDE) {
        for x in (0..width - BLOCK_SIZE + 1).step_by(STRIDE) {
            let ssim_block =
                calculate_ssim_block(reference, distorted, width, x, y, BLOCK_SIZE, c1, c2);
            ssim_values.push(ssim_block);
        }
    }

    if ssim_values.is_empty() {
        return Ok(1.0);
    }

    // Mean SSIM
    Ok(ssim_values.iter().sum::<f64>() / ssim_values.len() as f64)
}

/// Calculate SSIM for a single block
fn calculate_ssim_block(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    x: usize,
    y: usize,
    block_size: usize,
    c1: f64,
    c2: f64,
) -> f64 {
    let mut ref_sum = 0.0;
    let mut dist_sum = 0.0;
    let mut ref_sq_sum = 0.0;
    let mut dist_sq_sum = 0.0;
    let mut ref_dist_sum = 0.0;
    let mut count = 0.0;

    for dy in 0..block_size {
        for dx in 0..block_size {
            let idx = (y + dy) * width + (x + dx);
            let r = reference[idx] as f64;
            let d = distorted[idx] as f64;

            ref_sum += r;
            dist_sum += d;
            ref_sq_sum += r * r;
            dist_sq_sum += d * d;
            ref_dist_sum += r * d;
            count += 1.0;
        }
    }

    // Calculate means
    let mu_ref = ref_sum / count;
    let mu_dist = dist_sum / count;

    // Calculate variances and covariance
    let var_ref = (ref_sq_sum / count) - (mu_ref * mu_ref);
    let var_dist = (dist_sq_sum / count) - (mu_dist * mu_dist);
    let covar = (ref_dist_sum / count) - (mu_ref * mu_dist);

    // SSIM formula
    let numerator = (2.0 * mu_ref * mu_dist + c1) * (2.0 * covar + c2);
    let denominator = (mu_ref * mu_ref + mu_dist * mu_dist + c1) * (var_ref + var_dist + c2);

    if denominator == 0.0 {
        1.0
    } else {
        (numerator / denominator).clamp(0.0, 1.0)
    }
}

/// Calculate SSIM for YUV420P frame
pub fn calculate_ssim(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
) -> Result<f64> {
    let metrics = QualityMetrics::analyze(reference, distorted, width, height)?;
    Ok(metrics.ssim_avg)
}

/// Calculate Multi-Scale SSIM
///
/// MS-SSIM provides better correlation with human perception by
/// analyzing quality at multiple scales (resolutions).
fn calculate_ms_ssim(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
) -> Result<f64> {
    // MS-SSIM uses 5 scales with different weights
    const SCALES: usize = 5;
    const WEIGHTS: [f64; SCALES] = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333];

    let mut current_ref = reference.to_vec();
    let mut current_dist = distorted.to_vec();
    let mut current_width = width;
    let mut current_height = height;

    let mut ms_ssim_product = 1.0;

    for scale in 0..SCALES {
        if current_width < 8 || current_height < 8 {
            break; // Can't downsample further
        }

        // Calculate SSIM components at this scale
        let ssim =
            calculate_ssim_plane(&current_ref, &current_dist, current_width, current_height)?;

        // Apply weight (exponential averaging)
        ms_ssim_product *= ssim.powf(WEIGHTS[scale]);

        // Downsample by 2x for next scale (unless last scale)
        if scale < SCALES - 1 {
            current_ref = downsample_2x(&current_ref, current_width, current_height);
            current_dist = downsample_2x(&current_dist, current_width, current_height);
            current_width /= 2;
            current_height /= 2;
        }
    }

    Ok(ms_ssim_product.clamp(0.0, 1.0))
}

/// Downsample image by 2x using simple averaging
fn downsample_2x(data: &[u8], width: usize, height: usize) -> Vec<u8> {
    let new_width = width / 2;
    let new_height = height / 2;
    let mut result = vec![0u8; new_width * new_height];

    for y in 0..new_height {
        for x in 0..new_width {
            let src_y = y * 2;
            let src_x = x * 2;

            // Average 2x2 block
            let p00 = data[src_y * width + src_x] as u32;
            let p01 = data[src_y * width + src_x + 1] as u32;
            let p10 = data[(src_y + 1) * width + src_x] as u32;
            let p11 = data[(src_y + 1) * width + src_x + 1] as u32;

            let avg = (p00 + p01 + p10 + p11) / 4;
            result[y * new_width + x] = avg as u8;
        }
    }

    result
}

/// Quality comparison between multiple encodings
#[derive(Debug, Clone)]
pub struct QualityComparison {
    /// Labels for each encoding
    pub labels: Vec<String>,
    /// Metrics for each encoding
    pub metrics: Vec<QualityMetrics>,
}

impl QualityComparison {
    /// Create new quality comparison
    pub fn new() -> Self {
        QualityComparison {
            labels: Vec::new(),
            metrics: Vec::new(),
        }
    }

    /// Add an encoding to comparison
    pub fn add(
        &mut self,
        label: String,
        reference: &[u8],
        distorted: &[u8],
        width: usize,
        height: usize,
    ) -> Result<()> {
        let metrics = QualityMetrics::analyze(reference, distorted, width, height)?;
        self.labels.push(label);
        self.metrics.push(metrics);
        Ok(())
    }

    /// Generate comparison table
    pub fn summary(&self) -> String {
        if self.metrics.is_empty() {
            return "No encodings to compare".to_string();
        }

        let mut s = String::from("Quality Comparison:\n");
        s.push_str(&format!(
            "{:<20} {:>10} {:>10} {:>10}\n",
            "Encoding", "PSNR (dB)", "SSIM", "MS-SSIM"
        ));
        s.push_str(&"-".repeat(60));
        s.push('\n');

        for (label, metrics) in self.labels.iter().zip(self.metrics.iter()) {
            let ms_ssim_str = metrics
                .ms_ssim
                .map(|v| format!("{:.4}", v))
                .unwrap_or_else(|| "N/A".to_string());

            s.push_str(&format!(
                "{:<20} {:>10.2} {:>10.4} {:>10}\n",
                label, metrics.psnr_avg, metrics.ssim_avg, ms_ssim_str
            ));
        }

        // Find best encoding
        if let Some((best_idx, _)) = self
            .metrics
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.ssim_avg.partial_cmp(&b.ssim_avg).unwrap())
        {
            s.push_str(&format!("\nBest Quality: {}\n", self.labels[best_idx]));
        }

        s
    }

    /// Get best encoding by SSIM
    pub fn best_by_ssim(&self) -> Option<(String, &QualityMetrics)> {
        self.metrics
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.ssim_avg.partial_cmp(&b.ssim_avg).unwrap())
            .map(|(idx, metrics)| (self.labels[idx].clone(), metrics))
    }

    /// Get best encoding by PSNR
    pub fn best_by_psnr(&self) -> Option<(String, &QualityMetrics)> {
        self.metrics
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.psnr_avg.partial_cmp(&b.psnr_avg).unwrap())
            .map(|(idx, metrics)| (self.labels[idx].clone(), metrics))
    }
}

impl Default for QualityComparison {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_yuv420p(width: usize, height: usize, value: u8) -> Vec<u8> {
        let y_size = width * height;
        let uv_size = (width / 2) * (height / 2);
        vec![value; y_size + 2 * uv_size]
    }

    #[test]
    fn test_mse_calculation() {
        let ref_data = vec![100u8; 1000];
        let dist_data = vec![105u8; 1000]; // Difference of 5

        let mse = calculate_mse(&ref_data, &dist_data);
        assert_eq!(mse, 25.0); // (5^2) = 25
    }

    #[test]
    fn test_mse_to_psnr() {
        let mse = 25.0;
        let psnr = mse_to_psnr(mse);
        // PSNR = 10 * log10(255^2 / 25) = 10 * log10(2601) ≈ 34.15 dB
        assert!((psnr - 34.15).abs() < 0.01);

        // Perfect match
        let psnr_perfect = mse_to_psnr(0.0);
        assert!(psnr_perfect.is_infinite() && psnr_perfect.is_sign_positive());
    }

    #[test]
    fn test_psnr_calculation() {
        let width = 64;
        let height = 64;

        // Identical frames
        let reference = create_test_yuv420p(width, height, 128);
        let distorted = reference.clone();

        let psnr = calculate_psnr(&reference, &distorted, width, height).unwrap();
        assert!(psnr.is_infinite()); // Perfect match
    }

    #[test]
    fn test_psnr_with_noise() {
        let width = 64;
        let height = 64;

        let reference = create_test_yuv420p(width, height, 128);
        let mut distorted = reference.clone();

        // Add small noise
        for i in (0..distorted.len()).step_by(10) {
            distorted[i] = distorted[i].saturating_add(5);
        }

        let psnr = calculate_psnr(&reference, &distorted, width, height).unwrap();
        assert!(psnr > 20.0 && psnr < 50.0); // Reasonable range
    }

    #[test]
    fn test_ssim_identical() {
        let width = 64;
        let height = 64;

        let reference = create_test_yuv420p(width, height, 128);
        let distorted = reference.clone();

        let ssim = calculate_ssim(&reference, &distorted, width, height).unwrap();
        assert!((ssim - 1.0).abs() < 0.01); // Should be ~1.0 for identical
    }

    #[test]
    fn test_ssim_with_distortion() {
        let width = 64;
        let height = 64;

        let reference = create_test_yuv420p(width, height, 128);
        let mut distorted = reference.clone();

        // Add distortion
        for i in (0..distorted.len() / 2).step_by(5) {
            distorted[i] = distorted[i].saturating_add(10);
        }

        let ssim = calculate_ssim(&reference, &distorted, width, height).unwrap();
        assert!(ssim > 0.5 && ssim < 1.0); // Lower than perfect, but not terrible
    }

    #[test]
    fn test_quality_metrics_analyze() {
        let width = 64;
        let height = 64;

        let reference = create_test_yuv420p(width, height, 128);
        let distorted = reference.clone();

        let metrics = QualityMetrics::analyze(&reference, &distorted, width, height).unwrap();

        assert!(metrics.psnr_avg.is_infinite());
        assert!((metrics.ssim_avg - 1.0).abs() < 0.01);
        assert_eq!(metrics.mse_y, 0.0);
        assert_eq!(metrics.width, width);
        assert_eq!(metrics.height, height);
    }

    #[test]
    fn test_quality_rating() {
        let width = 64;
        let height = 64;

        let reference = create_test_yuv420p(width, height, 128);

        // Excellent quality (identical)
        let metrics = QualityMetrics::analyze(&reference, &reference, width, height).unwrap();
        assert!(metrics.quality_rating().contains("Excellent"));

        // Lower quality
        let mut distorted = reference.clone();
        for i in (0..distorted.len()).step_by(2) {
            distorted[i] = distorted[i].saturating_add(30);
        }
        let metrics = QualityMetrics::analyze(&reference, &distorted, width, height).unwrap();
        assert!(!metrics.quality_rating().contains("Excellent"));
    }

    #[test]
    fn test_downsample_2x() {
        let data = vec![
            100, 100, 150, 150, 100, 100, 150, 150, 200, 200, 250, 250, 200, 200, 250, 250,
        ];
        let downsampled = downsample_2x(&data, 4, 4);

        assert_eq!(downsampled.len(), 4); // 2x2 result
        assert_eq!(downsampled[0], 100); // Average of top-left 2x2
        assert_eq!(downsampled[1], 150); // Average of top-right 2x2
    }

    #[test]
    fn test_ms_ssim() {
        let width = 128; // Larger for multi-scale
        let height = 128;

        let reference = create_test_yuv420p(width, height, 128);
        let distorted = reference.clone();

        let y_size = width * height;
        let ms_ssim =
            calculate_ms_ssim(&reference[..y_size], &distorted[..y_size], width, height).unwrap();

        assert!((ms_ssim - 1.0).abs() < 0.01); // Should be ~1.0 for identical
    }

    #[test]
    fn test_quality_comparison() {
        let width = 64;
        let height = 64;

        let reference = create_test_yuv420p(width, height, 128);
        let encoding1 = reference.clone();
        let mut encoding2 = reference.clone();
        encoding2[0] = 130; // Slightly different

        let mut comparison = QualityComparison::new();
        comparison
            .add("Perfect".to_string(), &reference, &encoding1, width, height)
            .unwrap();
        comparison
            .add(
                "Slightly Distorted".to_string(),
                &reference,
                &encoding2,
                width,
                height,
            )
            .unwrap();

        assert_eq!(comparison.labels.len(), 2);
        assert_eq!(comparison.metrics.len(), 2);

        let (best_label, _) = comparison.best_by_ssim().unwrap();
        assert_eq!(best_label, "Perfect");
    }
}
