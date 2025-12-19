//! VP8 quantization and dequantization
//!
//! This module contains quantization lookup tables and functions for
//! converting between quantized and unquantized DCT coefficients.

/// Y plane DC quantizer lookup table (qi 0-127)
pub const DC_QUANT: [i16; 128] = [
    4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17, 17, 18, 19, 20, 20, 21, 21, 22, 22, 23,
    23, 24, 25, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 37, 38, 39, 40, 41, 42, 43, 44,
    45, 46, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
    68, 69, 70, 71, 72, 73, 74, 75, 76, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91,
    93, 95, 96, 98, 100, 101, 102, 104, 106, 108, 110, 112, 114, 116, 118, 122, 124, 126, 128, 130,
    132, 134, 136, 138, 140, 143, 145, 148, 151, 154, 157,
];

/// Y plane AC quantizer lookup table (qi 0-127)
pub const AC_QUANT: [i16; 128] = [
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
    53, 54, 55, 56, 57, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94,
    96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 119, 122, 125, 128, 131, 134, 137, 140,
    143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 173, 177, 181, 185, 189, 193, 197, 201, 205,
    209, 213, 217, 221, 225, 229, 234, 239, 245, 249, 254, 259, 264, 269, 274, 279, 284,
];

/// Y2 (WHT) block DC quantizer lookup table
/// Minimum value is 8 for Y2 DC
pub const Y2_DC_QUANT: [i16; 128] = [
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
    73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
    97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
    116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
];

/// Y2 (WHT) block AC quantizer lookup table
/// Minimum value is 8 for Y2 AC
pub const Y2_AC_QUANT: [i16; 128] = [
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
    73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
    97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
    116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
];

/// Quantizer step sizes for a macroblock
#[derive(Debug, Clone, Copy, Default)]
pub struct QuantFactors {
    pub y_dc: i16,
    pub y_ac: i16,
    pub y2_dc: i16,
    pub y2_ac: i16,
    pub uv_dc: i16,
    pub uv_ac: i16,
}

impl QuantFactors {
    /// Get quantization factors from quantizer indices
    pub fn from_indices(
        y_ac_qi: u8,
        y_dc_delta: i8,
        y2_dc_delta: i8,
        y2_ac_delta: i8,
        uv_dc_delta: i8,
        uv_ac_delta: i8,
    ) -> Self {
        let clamp_qi =
            |base: u8, delta: i8| -> usize { (base as i16 + delta as i16).clamp(0, 127) as usize };

        QuantFactors {
            y_dc: DC_QUANT[clamp_qi(y_ac_qi, y_dc_delta)],
            y_ac: AC_QUANT[y_ac_qi as usize],
            y2_dc: Y2_DC_QUANT[clamp_qi(y_ac_qi, y2_dc_delta)].max(8),
            y2_ac: Y2_AC_QUANT[clamp_qi(y_ac_qi, y2_ac_delta)].max(8),
            uv_dc: DC_QUANT[clamp_qi(y_ac_qi, uv_dc_delta)].min(132),
            uv_ac: AC_QUANT[clamp_qi(y_ac_qi, uv_ac_delta)],
        }
    }
}

/// Dequantize a 4x4 block of DCT coefficients
#[inline]
pub fn dequantize_block(coeffs: &[i16; 16], dc_quant: i16, ac_quant: i16) -> [i16; 16] {
    let mut output = [0i16; 16];
    output[0] = coeffs[0].saturating_mul(dc_quant);
    for i in 1..16 {
        output[i] = coeffs[i].saturating_mul(ac_quant);
    }
    output
}

/// Dequantize Y2 (WHT) block
#[inline]
pub fn dequantize_y2_block(coeffs: &[i16; 16], y2_dc: i16, y2_ac: i16) -> [i16; 16] {
    let mut output = [0i16; 16];
    output[0] = coeffs[0].saturating_mul(y2_dc);
    for i in 1..16 {
        output[i] = coeffs[i].saturating_mul(y2_ac);
    }
    output
}

/// Dequantize DC coefficient only (for Y blocks when Y2 is present)
#[inline]
pub fn dequantize_dc_only(dc: i16, dc_quant: i16) -> i16 {
    dc.saturating_mul(dc_quant)
}

/// Quantize a 4x4 block of DCT coefficients (for encoder)
#[inline]
pub fn quantize_block(coeffs: &[i16; 16], dc_quant: i16, ac_quant: i16) -> [i16; 16] {
    let mut output = [0i16; 16];

    // DC coefficient with rounding
    output[0] = if coeffs[0] >= 0 {
        (coeffs[0] + (dc_quant >> 1)) / dc_quant
    } else {
        (coeffs[0] - (dc_quant >> 1)) / dc_quant
    };

    // AC coefficients with rounding
    for i in 1..16 {
        output[i] = if coeffs[i] >= 0 {
            (coeffs[i] + (ac_quant >> 1)) / ac_quant
        } else {
            (coeffs[i] - (ac_quant >> 1)) / ac_quant
        };
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_factors() {
        let qf = QuantFactors::from_indices(64, 0, 0, 0, 0, 0);
        assert!(qf.y_dc > 0);
        assert!(qf.y_ac > 0);
        assert!(qf.y2_dc >= 8);
        assert!(qf.y2_ac >= 8);
    }

    #[test]
    fn test_dequantize_block() {
        let coeffs = [1i16; 16];
        let result = dequantize_block(&coeffs, 10, 20);
        assert_eq!(result[0], 10);
        assert_eq!(result[1], 20);
    }

    #[test]
    fn test_quantize_roundtrip() {
        let original = [100i16, 50, 25, 12, 6, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let quantized = quantize_block(&original, 10, 20);
        let dequantized = dequantize_block(&quantized, 10, 20);

        // Quantization is lossy, but should preserve order
        for i in 0..8 {
            if original[i] > 0 {
                assert!(dequantized[i] >= 0);
            }
        }
    }
}
