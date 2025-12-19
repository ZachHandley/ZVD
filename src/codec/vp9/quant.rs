//! VP9 Quantization and Dequantization
//!
//! VP9 uses separate quantization for DC and AC coefficients,
//! with different tables for luma and chroma planes.

use super::frame::QuantizationParams;
use super::tables::{TxSize, AC_QUANT_8BIT, DC_QUANT_8BIT};

/// Quantization factors for a single plane
#[derive(Debug, Clone, Copy, Default)]
pub struct PlaneQuantFactors {
    /// DC dequantization factor
    pub dc: i16,
    /// AC dequantization factor
    pub ac: i16,
}

/// Complete quantization state for a frame
#[derive(Debug, Clone)]
pub struct QuantizerState {
    /// Y plane factors
    pub y: PlaneQuantFactors,
    /// UV plane factors
    pub uv: PlaneQuantFactors,
    /// Base Q index
    pub base_q_idx: u8,
    /// Is lossless mode
    pub lossless: bool,
}

impl QuantizerState {
    /// Create quantizer state from frame parameters
    pub fn new(params: &QuantizationParams) -> Self {
        let base_q = params.base_q_idx as i16;

        // Calculate effective Q indices for each coefficient type
        let y_dc_q = (base_q + params.y_dc_delta as i16).clamp(0, 255) as usize;
        let y_ac_q = base_q.clamp(0, 255) as usize;
        let uv_dc_q = (base_q + params.uv_dc_delta as i16).clamp(0, 255) as usize;
        let uv_ac_q = (base_q + params.uv_ac_delta as i16).clamp(0, 255) as usize;

        QuantizerState {
            y: PlaneQuantFactors {
                dc: DC_QUANT_8BIT[y_dc_q],
                ac: AC_QUANT_8BIT[y_ac_q],
            },
            uv: PlaneQuantFactors {
                dc: DC_QUANT_8BIT[uv_dc_q],
                ac: AC_QUANT_8BIT[uv_ac_q],
            },
            base_q_idx: params.base_q_idx,
            lossless: params.is_lossless(),
        }
    }

    /// Create quantizer for a specific segment (with segment delta)
    pub fn for_segment(&self, segment_delta: i16) -> Self {
        let new_base = (self.base_q_idx as i16 + segment_delta).clamp(0, 255) as u8;

        let y_dc_q = new_base as usize;
        let y_ac_q = new_base as usize;
        let uv_dc_q = new_base as usize;
        let uv_ac_q = new_base as usize;

        QuantizerState {
            y: PlaneQuantFactors {
                dc: DC_QUANT_8BIT[y_dc_q],
                ac: AC_QUANT_8BIT[y_ac_q],
            },
            uv: PlaneQuantFactors {
                dc: DC_QUANT_8BIT[uv_dc_q],
                ac: AC_QUANT_8BIT[uv_ac_q],
            },
            base_q_idx: new_base,
            lossless: new_base == 0,
        }
    }

    /// Get quantizer for a specific plane (0=Y, 1=U, 2=V)
    pub fn for_plane(&self, plane: usize) -> &PlaneQuantFactors {
        if plane == 0 {
            &self.y
        } else {
            &self.uv
        }
    }
}

/// Dequantize a single coefficient
#[inline]
pub fn dequantize_coeff(coeff: i16, dequant: i16) -> i32 {
    coeff as i32 * dequant as i32
}

/// Dequantize a block of coefficients
///
/// # Arguments
/// * `coeffs` - Quantized coefficients (in scan order)
/// * `dequant` - Quantization factors for this block
/// * `eob` - End of block (number of non-zero coefficients + 1)
/// * `output` - Output buffer for dequantized coefficients
pub fn dequantize_block(
    coeffs: &[i16],
    factors: &PlaneQuantFactors,
    eob: usize,
    output: &mut [i32],
) {
    // Clear output
    output.fill(0);

    if eob == 0 {
        return;
    }

    // DC coefficient
    if !coeffs.is_empty() {
        output[0] = dequantize_coeff(coeffs[0], factors.dc);
    }

    // AC coefficients
    for i in 1..eob.min(coeffs.len()).min(output.len()) {
        output[i] = dequantize_coeff(coeffs[i], factors.ac);
    }
}

/// Dequantize a 4x4 block
pub fn dequantize_4x4(
    coeffs: &[i16; 16],
    factors: &PlaneQuantFactors,
    eob: usize,
    output: &mut [i32; 16],
) {
    output.fill(0);

    if eob == 0 {
        return;
    }

    output[0] = dequantize_coeff(coeffs[0], factors.dc);

    for i in 1..eob.min(16) {
        output[i] = dequantize_coeff(coeffs[i], factors.ac);
    }
}

/// Dequantize an 8x8 block
pub fn dequantize_8x8(
    coeffs: &[i16; 64],
    factors: &PlaneQuantFactors,
    eob: usize,
    output: &mut [i32; 64],
) {
    output.fill(0);

    if eob == 0 {
        return;
    }

    output[0] = dequantize_coeff(coeffs[0], factors.dc);

    for i in 1..eob.min(64) {
        output[i] = dequantize_coeff(coeffs[i], factors.ac);
    }
}

/// Dequantize a 16x16 block
pub fn dequantize_16x16(
    coeffs: &[i16; 256],
    factors: &PlaneQuantFactors,
    eob: usize,
    output: &mut [i32; 256],
) {
    output.fill(0);

    if eob == 0 {
        return;
    }

    output[0] = dequantize_coeff(coeffs[0], factors.dc);

    for i in 1..eob.min(256) {
        output[i] = dequantize_coeff(coeffs[i], factors.ac);
    }
}

/// Dequantize a 32x32 block
pub fn dequantize_32x32(
    coeffs: &[i16; 1024],
    factors: &PlaneQuantFactors,
    eob: usize,
    output: &mut [i32; 1024],
) {
    output.fill(0);

    if eob == 0 {
        return;
    }

    output[0] = dequantize_coeff(coeffs[0], factors.dc);

    for i in 1..eob.min(1024) {
        output[i] = dequantize_coeff(coeffs[i], factors.ac);
    }
}

/// Dequantize based on transform size
pub fn dequantize(
    coeffs: &[i16],
    factors: &PlaneQuantFactors,
    tx_size: TxSize,
    eob: usize,
    output: &mut [i32],
) {
    let size = tx_size.num_coeffs();

    // Clear output
    for i in 0..size.min(output.len()) {
        output[i] = 0;
    }

    if eob == 0 {
        return;
    }

    // DC coefficient
    if !coeffs.is_empty() && !output.is_empty() {
        output[0] = dequantize_coeff(coeffs[0], factors.dc);
    }

    // AC coefficients
    let limit = eob.min(size).min(coeffs.len()).min(output.len());
    for i in 1..limit {
        output[i] = dequantize_coeff(coeffs[i], factors.ac);
    }
}

/// Get dequantization table for a bit depth
pub fn get_dc_quant_table(bit_depth: u8) -> &'static [i16; 256] {
    match bit_depth {
        8 => &DC_QUANT_8BIT,
        // For 10/12 bit, we'd need additional tables
        // For now, scale the 8-bit table
        _ => &DC_QUANT_8BIT,
    }
}

pub fn get_ac_quant_table(bit_depth: u8) -> &'static [i16; 256] {
    match bit_depth {
        8 => &AC_QUANT_8BIT,
        _ => &AC_QUANT_8BIT,
    }
}

/// Calculate QIndex with segment adjustments
pub fn get_qindex(
    base_q_idx: u8,
    segment_id: u8,
    segment_features: &[[super::frame::SegmentFeature; 4]; 8],
    abs_delta: bool,
) -> u8 {
    let seg_features = &segment_features[segment_id as usize];

    // Feature 0 is quantizer adjustment
    if seg_features[0].enabled {
        let delta = seg_features[0].value as i16;
        if abs_delta {
            // Absolute value
            delta.clamp(0, 255) as u8
        } else {
            // Delta from base
            (base_q_idx as i16 + delta).clamp(0, 255) as u8
        }
    } else {
        base_q_idx
    }
}

// =============================================================================
// Forward Quantization (for encoding)
// =============================================================================

/// Quantize a single coefficient
#[inline]
pub fn quantize_coeff(coeff: i32, quant: i16, shift: i32) -> i16 {
    if coeff == 0 {
        return 0;
    }

    let sign = if coeff < 0 { -1 } else { 1 };
    let abs_coeff = coeff.abs();

    // Apply quantization: (coeff * quant_mult + round) >> shift
    // For VP9, we use a simpler approach
    let quant_val = quant as i32;
    let quantized = (abs_coeff + (quant_val >> 1)) / quant_val;

    (quantized as i16) * sign as i16
}

/// Quantize a block of transform coefficients
///
/// # Arguments
/// * `coeffs` - Transform coefficients (in raster scan order)
/// * `factors` - Quantization factors for this block
/// * `output` - Output buffer for quantized coefficients
///
/// # Returns
/// End of block position (last non-zero coefficient index + 1)
pub fn quantize_block(coeffs: &[i32], factors: &PlaneQuantFactors, output: &mut [i16]) -> usize {
    let mut eob = 0;

    if coeffs.is_empty() || output.is_empty() {
        return 0;
    }

    // DC coefficient
    output[0] = quantize_coeff(coeffs[0], factors.dc, 0);
    if output[0] != 0 {
        eob = 1;
    }

    // AC coefficients
    let limit = coeffs.len().min(output.len());
    for i in 1..limit {
        output[i] = quantize_coeff(coeffs[i], factors.ac, 0);
        if output[i] != 0 {
            eob = i + 1;
        }
    }

    eob
}

/// Quantize a 4x4 block
pub fn quantize_4x4(
    coeffs: &[i32; 16],
    factors: &PlaneQuantFactors,
    output: &mut [i16; 16],
) -> usize {
    let mut eob = 0;

    output[0] = quantize_coeff(coeffs[0], factors.dc, 0);
    if output[0] != 0 {
        eob = 1;
    }

    for i in 1..16 {
        output[i] = quantize_coeff(coeffs[i], factors.ac, 0);
        if output[i] != 0 {
            eob = i + 1;
        }
    }

    eob
}

/// Quantize an 8x8 block
pub fn quantize_8x8(
    coeffs: &[i32; 64],
    factors: &PlaneQuantFactors,
    output: &mut [i16; 64],
) -> usize {
    let mut eob = 0;

    output[0] = quantize_coeff(coeffs[0], factors.dc, 0);
    if output[0] != 0 {
        eob = 1;
    }

    for i in 1..64 {
        output[i] = quantize_coeff(coeffs[i], factors.ac, 0);
        if output[i] != 0 {
            eob = i + 1;
        }
    }

    eob
}

/// Quantize a 16x16 block
pub fn quantize_16x16(
    coeffs: &[i32; 256],
    factors: &PlaneQuantFactors,
    output: &mut [i16; 256],
) -> usize {
    let mut eob = 0;

    output[0] = quantize_coeff(coeffs[0], factors.dc, 0);
    if output[0] != 0 {
        eob = 1;
    }

    for i in 1..256 {
        output[i] = quantize_coeff(coeffs[i], factors.ac, 0);
        if output[i] != 0 {
            eob = i + 1;
        }
    }

    eob
}

/// Quantize a 32x32 block
pub fn quantize_32x32(
    coeffs: &[i32; 1024],
    factors: &PlaneQuantFactors,
    output: &mut [i16; 1024],
) -> usize {
    let mut eob = 0;

    output[0] = quantize_coeff(coeffs[0], factors.dc, 0);
    if output[0] != 0 {
        eob = 1;
    }

    for i in 1..1024 {
        output[i] = quantize_coeff(coeffs[i], factors.ac, 0);
        if output[i] != 0 {
            eob = i + 1;
        }
    }

    eob
}

/// Quantize based on transform size
pub fn quantize(
    coeffs: &[i32],
    factors: &PlaneQuantFactors,
    tx_size: TxSize,
    output: &mut [i16],
) -> usize {
    let size = tx_size.num_coeffs();
    let mut eob = 0;

    // Clear output
    for i in 0..size.min(output.len()) {
        output[i] = 0;
    }

    if coeffs.is_empty() {
        return 0;
    }

    // DC coefficient
    if !coeffs.is_empty() && !output.is_empty() {
        output[0] = quantize_coeff(coeffs[0], factors.dc, 0);
        if output[0] != 0 {
            eob = 1;
        }
    }

    // AC coefficients
    let limit = size.min(coeffs.len()).min(output.len());
    for i in 1..limit {
        output[i] = quantize_coeff(coeffs[i], factors.ac, 0);
        if output[i] != 0 {
            eob = i + 1;
        }
    }

    eob
}

/// Create quantizer factors from a Q index
pub fn create_quant_factors(base_q_idx: u8) -> PlaneQuantFactors {
    let q_idx = base_q_idx as usize;
    PlaneQuantFactors {
        dc: DC_QUANT_8BIT[q_idx],
        ac: AC_QUANT_8BIT[q_idx],
    }
}

/// Create quantizer state for encoding
pub fn create_encoder_quantizer(base_q_idx: u8) -> QuantizerState {
    let q_idx = base_q_idx as usize;

    QuantizerState {
        y: PlaneQuantFactors {
            dc: DC_QUANT_8BIT[q_idx],
            ac: AC_QUANT_8BIT[q_idx],
        },
        uv: PlaneQuantFactors {
            dc: DC_QUANT_8BIT[q_idx],
            ac: AC_QUANT_8BIT[q_idx],
        },
        base_q_idx,
        lossless: base_q_idx == 0,
    }
}

/// Count non-zero coefficients in a block
pub fn count_nonzero(coeffs: &[i16]) -> usize {
    coeffs.iter().filter(|&&c| c != 0).count()
}

/// Check if a block has any non-zero coefficients
pub fn has_nonzero(coeffs: &[i16]) -> bool {
    coeffs.iter().any(|&c| c != 0)
}

/// Calculate the sum of absolute values (SAD) for rate-distortion
pub fn calc_coeff_cost(coeffs: &[i16], eob: usize) -> u64 {
    let mut cost = 0u64;
    for i in 0..eob {
        cost += coeffs[i].unsigned_abs() as u64;
    }
    cost
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantizer_state_creation() {
        let params = QuantizationParams {
            base_q_idx: 100,
            y_dc_delta: 0,
            uv_dc_delta: 0,
            uv_ac_delta: 0,
            lossless: false,
        };

        let state = QuantizerState::new(&params);
        assert_eq!(state.base_q_idx, 100);
        assert!(!state.lossless);
        assert!(state.y.dc > 0);
        assert!(state.y.ac > 0);
    }

    #[test]
    fn test_lossless_quantizer() {
        let params = QuantizationParams {
            base_q_idx: 0,
            y_dc_delta: 0,
            uv_dc_delta: 0,
            uv_ac_delta: 0,
            lossless: true,
        };

        let state = QuantizerState::new(&params);
        assert!(state.lossless);
        // At Q=0, dequant values should be 4
        assert_eq!(state.y.dc, 4);
        assert_eq!(state.y.ac, 4);
    }

    #[test]
    fn test_dequantize_block() {
        let coeffs = [10i16, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let factors = PlaneQuantFactors { dc: 8, ac: 4 };
        let mut output = [0i32; 16];

        dequantize_4x4(&coeffs, &factors, 3, &mut output);

        assert_eq!(output[0], 80); // 10 * 8
        assert_eq!(output[1], 20); // 5 * 4
        assert_eq!(output[2], 12); // 3 * 4
        assert_eq!(output[3], 0); // Zero (past EOB but coeff is 0)
    }

    #[test]
    fn test_dequantize_eob_zero() {
        let coeffs = [100i16; 16];
        let factors = PlaneQuantFactors { dc: 8, ac: 4 };
        let mut output = [999i32; 16];

        dequantize_4x4(&coeffs, &factors, 0, &mut output);

        // All should be zero when EOB is 0
        for &v in &output {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn test_for_segment() {
        let params = QuantizationParams {
            base_q_idx: 100,
            y_dc_delta: 0,
            uv_dc_delta: 0,
            uv_ac_delta: 0,
            lossless: false,
        };

        let state = QuantizerState::new(&params);
        let seg_state = state.for_segment(-10);

        assert_eq!(seg_state.base_q_idx, 90);
    }

    #[test]
    fn test_for_segment_clamp() {
        let params = QuantizationParams {
            base_q_idx: 10,
            y_dc_delta: 0,
            uv_dc_delta: 0,
            uv_ac_delta: 0,
            lossless: false,
        };

        let state = QuantizerState::new(&params);
        let seg_state = state.for_segment(-100);

        // Should clamp to 0, not go negative
        assert_eq!(seg_state.base_q_idx, 0);
    }
}
