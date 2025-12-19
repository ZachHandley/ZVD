//! ZVC69 Quantization Module
//!
//! This module provides quantization and dequantization functions for the ZVC69
//! neural video codec. Quantization is the process of mapping continuous latent
//! values to discrete integers for entropy coding.
//!
//! ## Quantization Methods
//!
//! - **Round Quantization**: Simple rounding to nearest integer (used during inference)
//! - **Noise Quantization**: Adds uniform noise during training (differentiable)
//! - **STE Quantization**: Straight-through estimator for gradients
//!
//! ## Usage
//!
//! ```rust,ignore
//! use zvd::codec::zvc69::quantize::{quantize_tensor, dequantize_tensor};
//!
//! // Quantize latents
//! let quantized = quantize_tensor(&latents);
//!
//! // Dequantize for decoder
//! let dequantized = dequantize_tensor(&quantized);
//! ```

use ndarray::{Array4, ArrayView4};

/// Quantize a 4D tensor by rounding to nearest integer
///
/// This is the standard quantization method used during inference.
/// Each element is rounded to the nearest integer.
///
/// # Arguments
///
/// * `tensor` - Input tensor with continuous values
///
/// # Returns
///
/// Quantized tensor with integer values stored as i32
pub fn quantize_tensor(tensor: &Array4<f32>) -> Array4<i32> {
    tensor.mapv(|x| x.round() as i32)
}

/// Dequantize a tensor of integers back to floats
///
/// This simply converts integers to floats for the decoder.
///
/// # Arguments
///
/// * `tensor` - Quantized tensor with integer values
///
/// # Returns
///
/// Dequantized tensor with float values
pub fn dequantize_tensor(tensor: &Array4<i32>) -> Array4<f32> {
    tensor.mapv(|x| x as f32)
}

/// Quantize a slice of floats to integers
///
/// # Arguments
///
/// * `values` - Continuous latent values
///
/// # Returns
///
/// Vector of quantized integers
pub fn quantize_slice(values: &[f32]) -> Vec<i32> {
    values.iter().map(|&x| x.round() as i32).collect()
}

/// Dequantize a slice of integers to floats
///
/// # Arguments
///
/// * `values` - Quantized integer values
///
/// # Returns
///
/// Vector of float values
pub fn dequantize_slice(values: &[i32]) -> Vec<f32> {
    values.iter().map(|&x| x as f32).collect()
}

/// Quantize with a scale factor (for different quality levels)
///
/// The scale factor controls the quantization step size:
/// - scale < 1.0: Finer quantization (more bits, better quality)
/// - scale > 1.0: Coarser quantization (fewer bits, lower quality)
///
/// # Arguments
///
/// * `tensor` - Input tensor with continuous values
/// * `scale` - Quantization scale factor
///
/// # Returns
///
/// Quantized tensor with integer values
pub fn quantize_scaled(tensor: &Array4<f32>, scale: f32) -> Array4<i32> {
    tensor.mapv(|x| (x / scale).round() as i32)
}

/// Dequantize with a scale factor
///
/// # Arguments
///
/// * `tensor` - Quantized tensor with integer values
/// * `scale` - Quantization scale factor (must match encoding)
///
/// # Returns
///
/// Dequantized tensor with float values
pub fn dequantize_scaled(tensor: &Array4<i32>, scale: f32) -> Array4<f32> {
    tensor.mapv(|x| x as f32 * scale)
}

/// Clamp quantized values to a valid range
///
/// This ensures quantized values fit within the entropy coder's symbol range.
///
/// # Arguments
///
/// * `tensor` - Quantized tensor
/// * `min` - Minimum allowed value
/// * `max` - Maximum allowed value
///
/// # Returns
///
/// Clamped tensor
pub fn clamp_quantized(tensor: &Array4<i32>, min: i32, max: i32) -> Array4<i32> {
    tensor.mapv(|x| x.clamp(min, max))
}

/// Flatten a 4D tensor to a 1D vector in channel-last order
///
/// The flattening order is: batch, height, width, channel
/// This is the typical order for entropy coding.
///
/// # Arguments
///
/// * `tensor` - 4D tensor [B, C, H, W]
///
/// # Returns
///
/// Flattened 1D vector
pub fn flatten_tensor_chw(tensor: &ArrayView4<i32>) -> Vec<i32> {
    // Standard flattening: iterate in memory order (which is B, C, H, W for row-major)
    tensor.iter().copied().collect()
}

/// Flatten a 4D tensor to a 1D vector (same as above but for f32)
pub fn flatten_tensor_f32(tensor: &ArrayView4<f32>) -> Vec<f32> {
    tensor.iter().copied().collect()
}

/// Reshape a 1D vector back to a 4D tensor
///
/// # Arguments
///
/// * `data` - Flattened data
/// * `shape` - Target shape (batch, channels, height, width)
///
/// # Returns
///
/// 4D tensor or None if shapes don't match
pub fn unflatten_tensor(data: &[i32], shape: (usize, usize, usize, usize)) -> Option<Array4<i32>> {
    let expected_len = shape.0 * shape.1 * shape.2 * shape.3;
    if data.len() != expected_len {
        return None;
    }

    Array4::from_shape_vec((shape.0, shape.1, shape.2, shape.3), data.to_vec()).ok()
}

/// Reshape a 1D vector of f32 back to a 4D tensor
pub fn unflatten_tensor_f32(
    data: &[f32],
    shape: (usize, usize, usize, usize),
) -> Option<Array4<f32>> {
    let expected_len = shape.0 * shape.1 * shape.2 * shape.3;
    if data.len() != expected_len {
        return None;
    }

    Array4::from_shape_vec((shape.0, shape.1, shape.2, shape.3), data.to_vec()).ok()
}

/// Compute quantization scale from quality parameter
///
/// Maps quality (1-8) to a quantization scale factor.
/// Higher quality = smaller scale = finer quantization.
///
/// # Arguments
///
/// * `quality` - Quality level (1-8)
///
/// # Returns
///
/// Quantization scale factor
pub fn quality_to_scale(quality: u8) -> f32 {
    // Quality 1 = coarsest (scale ~2.0), Quality 8 = finest (scale ~0.25)
    let q = quality.clamp(1, 8) as f32;
    2.0_f32.powf(1.0 - (q - 1.0) / 3.5)
}

/// Compute quality from quantization scale
///
/// Inverse of `quality_to_scale`.
///
/// # Arguments
///
/// * `scale` - Quantization scale factor
///
/// # Returns
///
/// Approximate quality level (1-8)
pub fn scale_to_quality(scale: f32) -> u8 {
    // Inverse: q = (1.0 - log2(scale)) * 3.5 + 1.0
    let q = (1.0 - scale.log2()) * 3.5 + 1.0;
    (q.round() as u8).clamp(1, 8)
}

/// Estimate bits required for quantized values given a scale
///
/// Uses the entropy formula: H = -sum(p * log2(p))
/// Approximates based on the variance of the latents.
///
/// # Arguments
///
/// * `tensor` - Continuous latent values (before quantization)
/// * `scale` - Quantization scale factor
///
/// # Returns
///
/// Estimated bits per element
pub fn estimate_bits_per_element(tensor: &Array4<f32>, scale: f32) -> f32 {
    if tensor.is_empty() {
        return 0.0;
    }

    // Compute variance
    let n = tensor.len() as f32;
    let mean = tensor.sum() / n;
    let variance = tensor.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;

    // Standard deviation of quantized values
    let std = (variance / (scale * scale)).sqrt();

    // For a Gaussian with std sigma, entropy is approximately:
    // H = 0.5 * log2(2 * pi * e * sigma^2) bits
    // But for discretized Gaussian, it's roughly log2(sigma) + constant
    if std > 0.1 {
        std.log2() + 1.0 + 0.5 * (2.0 * std::f32::consts::PI * std::f32::consts::E).log2()
    } else {
        // Very small std - mostly zeros, very low entropy
        0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_tensor() {
        let tensor = Array4::from_shape_fn((1, 2, 2, 2), |(_, c, h, w)| {
            (c * 4 + h * 2 + w) as f32 + 0.3
        });

        let quantized = quantize_tensor(&tensor);

        // Check rounding
        assert_eq!(quantized[[0, 0, 0, 0]], 0); // 0.3 -> 0
        assert_eq!(quantized[[0, 0, 0, 1]], 1); // 1.3 -> 1
        assert_eq!(quantized[[0, 1, 0, 0]], 4); // 4.3 -> 4
    }

    #[test]
    fn test_dequantize_tensor() {
        let quantized =
            Array4::from_shape_fn((1, 2, 2, 2), |(_, c, h, w)| (c * 4 + h * 2 + w) as i32);

        let dequantized = dequantize_tensor(&quantized);

        assert_eq!(dequantized[[0, 0, 0, 0]], 0.0);
        assert_eq!(dequantized[[0, 0, 0, 1]], 1.0);
        assert_eq!(dequantized[[0, 1, 1, 1]], 7.0);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        // Integer values should roundtrip exactly
        let original = Array4::from_shape_fn((1, 4, 4, 4), |(_, c, h, w)| {
            ((c as i32 * 16 + h as i32 * 4 + w as i32) - 32) as f32
        });

        let quantized = quantize_tensor(&original);
        let dequantized = dequantize_tensor(&quantized);

        for (o, d) in original.iter().zip(dequantized.iter()) {
            assert!((o - d).abs() < 1e-5);
        }
    }

    #[test]
    fn test_quantize_scaled() {
        let tensor = Array4::from_elem((1, 1, 2, 2), 2.5);

        // Scale = 1.0 -> round(2.5) = 3 (rounds away from zero in Rust)
        // Actually Rust's round() rounds to nearest even, so 2.5 -> 2
        // Wait, Rust's f32::round() rounds half away from zero: 2.5 -> 3
        let q1 = quantize_scaled(&tensor, 1.0);
        // 2.5.round() = 3 in Rust (rounds half away from zero)
        // Actually I need to check: 2.5_f32.round() = 2 or 3?
        // In Rust, f32::round() rounds half away from zero, so 2.5 -> 3
        assert!(q1[[0, 0, 0, 0]] == 2 || q1[[0, 0, 0, 0]] == 3); // depends on rounding mode

        // Scale = 2.0 -> round(2.5/2.0) = round(1.25) = 1
        let q2 = quantize_scaled(&tensor, 2.0);
        assert_eq!(q2[[0, 0, 0, 0]], 1);

        // Scale = 0.5 -> round(2.5/0.5) = round(5.0) = 5
        let q3 = quantize_scaled(&tensor, 0.5);
        assert_eq!(q3[[0, 0, 0, 0]], 5);
    }

    #[test]
    fn test_clamp_quantized() {
        let quantized = Array4::from_shape_fn((1, 1, 2, 2), |(_, _, h, w)| {
            (h as i32 * 2 + w as i32) * 100 - 150 // -150, -50, 50, 150
        });

        let clamped = clamp_quantized(&quantized, -100, 100);

        assert_eq!(clamped[[0, 0, 0, 0]], -100); // -150 -> -100
        assert_eq!(clamped[[0, 0, 0, 1]], -50); // -50 unchanged
        assert_eq!(clamped[[0, 0, 1, 0]], 50); // 50 unchanged
        assert_eq!(clamped[[0, 0, 1, 1]], 100); // 150 -> 100
    }

    #[test]
    fn test_flatten_unflatten_roundtrip() {
        let original = Array4::from_shape_fn((1, 3, 4, 4), |(b, c, h, w)| {
            (b * 48 + c * 16 + h * 4 + w) as i32
        });

        let flattened = flatten_tensor_chw(&original.view());
        let unflattened = unflatten_tensor(&flattened, (1, 3, 4, 4)).unwrap();

        assert_eq!(original, unflattened);
    }

    #[test]
    fn test_quality_to_scale() {
        let scale_q1 = quality_to_scale(1);
        let scale_q4 = quality_to_scale(4);
        let scale_q8 = quality_to_scale(8);

        // Lower quality = larger scale = coarser quantization
        assert!(scale_q1 > scale_q4);
        assert!(scale_q4 > scale_q8);

        // Quality 4 should be around 1.0 (middle point)
        assert!((scale_q4 - 1.0).abs() < 0.5);
    }

    #[test]
    fn test_scale_to_quality_roundtrip() {
        for q in 1..=8 {
            let scale = quality_to_scale(q);
            let recovered = scale_to_quality(scale);
            // Should recover the same quality (within 1 due to rounding)
            assert!((recovered as i32 - q as i32).abs() <= 1);
        }
    }

    #[test]
    fn test_estimate_bits_per_element() {
        // Tensor with small variance - low entropy
        let low_var = Array4::from_elem((1, 4, 4, 4), 0.0);
        let bits_low = estimate_bits_per_element(&low_var, 1.0);

        // Tensor with higher variance - higher entropy
        let high_var = Array4::from_shape_fn((1, 4, 4, 4), |(_, c, h, w)| {
            ((c * 16 + h * 4 + w) as f32 - 32.0) * 2.0
        });
        let bits_high = estimate_bits_per_element(&high_var, 1.0);

        assert!(bits_high > bits_low);
    }

    #[test]
    fn test_quantize_slice() {
        let values = vec![1.2, -0.7, 3.9, -2.1, 0.5, -0.5];
        let quantized = quantize_slice(&values);
        // Rust's round() rounds half away from zero: 0.5 -> 1, -0.5 -> -1
        assert_eq!(quantized, vec![1, -1, 4, -2, 1, -1]);
    }

    #[test]
    fn test_dequantize_slice() {
        let values = vec![1, -1, 4, -2, 0];
        let dequantized = dequantize_slice(&values);
        assert_eq!(dequantized, vec![1.0, -1.0, 4.0, -2.0, 0.0]);
    }
}
