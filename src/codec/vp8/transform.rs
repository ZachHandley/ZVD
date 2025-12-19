//! VP8 transform implementations (DCT and WHT)
//!
//! This module contains the 4x4 DCT (Discrete Cosine Transform) and
//! WHT (Walsh-Hadamard Transform) implementations used in VP8.

/// Constants for inverse DCT
const C1: i32 = 20091; // cos(pi/8) * sqrt(2) * 16384 (approximately)
const C2: i32 = 35468; // sin(pi/8) * sqrt(2) * 16384 (approximately)

/// 4x4 Inverse DCT (IDCT) for decoder
///
/// Takes a 4x4 block of DCT coefficients and outputs a 4x4 block of residuals.
/// The output values are added to the prediction to reconstruct the original block.
#[inline]
pub fn inverse_dct4x4(input: &[i16; 16], output: &mut [i16; 16]) {
    let mut temp = [0i32; 16];

    // 1D IDCT on columns
    for j in 0..4 {
        let a = input[j] as i32;
        let b = input[4 + j] as i32;
        let c = input[8 + j] as i32;
        let d = input[12 + j] as i32;

        // Transform using the VP8-specific integer approximation
        let a1 = a + c;
        let b1 = a - c;

        // Note: The VP8 IDCT uses a specific fixed-point approximation
        let c1 = ((b * C2) >> 16) - d - ((d * C1) >> 16);
        let d1 = b + ((b * C1) >> 16) + ((d * C2) >> 16);

        temp[j] = a1 + d1;
        temp[4 + j] = b1 + c1;
        temp[8 + j] = b1 - c1;
        temp[12 + j] = a1 - d1;
    }

    // 1D IDCT on rows
    for i in 0..4 {
        let row_offset = i * 4;
        let a = temp[row_offset];
        let b = temp[row_offset + 1];
        let c = temp[row_offset + 2];
        let d = temp[row_offset + 3];

        let a1 = a + c;
        let b1 = a - c;
        let c1 = ((b * C2) >> 16) - d - ((d * C1) >> 16);
        let d1 = b + ((b * C1) >> 16) + ((d * C2) >> 16);

        // Add 4 for rounding, shift by 3
        output[row_offset] = ((a1 + d1 + 4) >> 3) as i16;
        output[row_offset + 1] = ((b1 + c1 + 4) >> 3) as i16;
        output[row_offset + 2] = ((b1 - c1 + 4) >> 3) as i16;
        output[row_offset + 3] = ((a1 - d1 + 4) >> 3) as i16;
    }
}

/// Simplified IDCT for DC-only blocks
///
/// When only the DC coefficient is non-zero, we can use a much simpler transform.
#[inline]
pub fn inverse_dct4x4_dc_only(dc: i16, output: &mut [i16; 16]) {
    // DC value is uniformly distributed across the block
    // (dc + 4) >> 3 normalizes the value
    let val = (dc as i32 + 4) >> 3;
    let val = val as i16;
    output.fill(val);
}

/// 4x4 Inverse Walsh-Hadamard Transform (IWHT)
///
/// Used for the Y2 block which contains the DC coefficients of the 16 Y subblocks.
#[inline]
pub fn inverse_wht4x4(input: &[i16; 16], output: &mut [i16; 16]) {
    let mut temp = [0i32; 16];

    // 1D IWHT on rows
    for i in 0..4 {
        let row_offset = i * 4;
        let a = input[row_offset] as i32 + input[row_offset + 3] as i32;
        let b = input[row_offset + 1] as i32 + input[row_offset + 2] as i32;
        let c = input[row_offset + 1] as i32 - input[row_offset + 2] as i32;
        let d = input[row_offset] as i32 - input[row_offset + 3] as i32;

        temp[row_offset] = a + b;
        temp[row_offset + 1] = c + d;
        temp[row_offset + 2] = a - b;
        temp[row_offset + 3] = c - d;
    }

    // 1D IWHT on columns
    for j in 0..4 {
        let a = temp[j] + temp[12 + j];
        let b = temp[4 + j] + temp[8 + j];
        let c = temp[4 + j] - temp[8 + j];
        let d = temp[j] - temp[12 + j];

        // Add 3 for rounding, shift right by 3
        output[j] = ((a + b + 3) >> 3) as i16;
        output[4 + j] = ((c + d + 3) >> 3) as i16;
        output[8 + j] = ((a - b + 3) >> 3) as i16;
        output[12 + j] = ((c - d + 3) >> 3) as i16;
    }
}

/// Simplified IWHT for DC-only Y2 blocks
#[inline]
pub fn inverse_wht4x4_dc_only(dc: i16, output: &mut [i16; 16]) {
    // For DC-only, the value propagates uniformly
    let val = (dc as i32 + 3) >> 3;
    let val = val as i16;
    output.fill(val);
}

/// 4x4 Forward DCT (for encoder)
///
/// Takes a 4x4 block of residuals and outputs DCT coefficients.
#[allow(dead_code)]
pub fn forward_dct4x4(input: &[i16; 16], output: &mut [i16; 16]) {
    let mut temp = [0i32; 16];

    // Constants for forward transform
    const FWD_C1: i32 = 5352;
    const FWD_C2: i32 = 2217;

    // 1D DCT on rows
    for i in 0..4 {
        let row_offset = i * 4;
        let a1 = (input[row_offset] + input[row_offset + 3]) as i32;
        let b1 = (input[row_offset + 1] + input[row_offset + 2]) as i32;
        let c1 = (input[row_offset + 1] - input[row_offset + 2]) as i32;
        let d1 = (input[row_offset] - input[row_offset + 3]) as i32;

        temp[row_offset] = a1 + b1;
        temp[row_offset + 1] = (c1 * FWD_C2 + d1 * FWD_C1 + 14500) >> 12;
        temp[row_offset + 2] = a1 - b1;
        temp[row_offset + 3] = (d1 * FWD_C2 - c1 * FWD_C1 + 7500) >> 12;
    }

    // 1D DCT on columns
    for j in 0..4 {
        let a1 = temp[j] + temp[12 + j];
        let b1 = temp[4 + j] + temp[8 + j];
        let c1 = temp[4 + j] - temp[8 + j];
        let d1 = temp[j] - temp[12 + j];

        output[j] = ((a1 + b1 + 7) >> 4) as i16;
        output[4 + j] = ((c1 * FWD_C2 + d1 * FWD_C1 + 12000) >> 16) as i16;
        output[8 + j] = ((a1 - b1 + 7) >> 4) as i16;
        output[12 + j] = ((d1 * FWD_C2 - c1 * FWD_C1 + 51000) >> 16) as i16;
    }
}

/// 4x4 Forward Walsh-Hadamard Transform (for encoder)
#[allow(dead_code)]
pub fn forward_wht4x4(input: &[i16; 16], output: &mut [i16; 16]) {
    let mut temp = [0i32; 16];

    // 1D WHT on rows
    for i in 0..4 {
        let row_offset = i * 4;
        let a = input[row_offset] as i32 + input[row_offset + 2] as i32;
        let b = input[row_offset + 1] as i32 + input[row_offset + 3] as i32;
        let c = input[row_offset] as i32 - input[row_offset + 2] as i32;
        let d = input[row_offset + 1] as i32 - input[row_offset + 3] as i32;

        temp[row_offset] = a + b;
        temp[row_offset + 1] = c + d;
        temp[row_offset + 2] = a - b;
        temp[row_offset + 3] = c - d;
    }

    // 1D WHT on columns
    for j in 0..4 {
        let a = temp[j] + temp[8 + j];
        let b = temp[4 + j] + temp[12 + j];
        let c = temp[j] - temp[8 + j];
        let d = temp[4 + j] - temp[12 + j];

        // Scale by 1/2 with rounding
        output[j] = ((a + b + 1) >> 1) as i16;
        output[4 + j] = ((c + d + 1) >> 1) as i16;
        output[8 + j] = ((a - b + 1) >> 1) as i16;
        output[12 + j] = ((c - d + 1) >> 1) as i16;
    }
}

/// Add residual to prediction and clamp to [0, 255]
#[inline]
pub fn add_residual_to_prediction(
    pred: &[u8],
    pred_stride: usize,
    residual: &[i16; 16],
    output: &mut [u8],
    output_stride: usize,
) {
    for y in 0..4 {
        for x in 0..4 {
            let pred_val = pred[y * pred_stride + x] as i16;
            let res_val = residual[y * 4 + x];
            let result = (pred_val + res_val).clamp(0, 255) as u8;
            output[y * output_stride + x] = result;
        }
    }
}

/// Add residual to prediction in-place
#[inline]
pub fn add_residual_inplace(output: &mut [u8], stride: usize, residual: &[i16; 16]) {
    for y in 0..4 {
        for x in 0..4 {
            let idx = y * stride + x;
            let pred_val = output[idx] as i16;
            let res_val = residual[y * 4 + x];
            output[idx] = (pred_val + res_val).clamp(0, 255) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idct_dc_only() {
        let input = [16i16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut output = [0i16; 16];
        inverse_dct4x4(&input, &mut output);

        // All values should be (approximately) equal for DC-only input
        let dc_val = output[0];
        for &val in &output {
            assert!((val - dc_val).abs() <= 1);
        }
    }

    #[test]
    fn test_idct_dc_only_simplified() {
        let dc = 16i16;
        let mut output1 = [0i16; 16];
        let mut output2 = [0i16; 16];

        // Full IDCT with DC-only input
        let input = [dc, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        inverse_dct4x4(&input, &mut output1);

        // Simplified DC-only IDCT
        inverse_dct4x4_dc_only(dc, &mut output2);

        // Results should match
        for i in 0..16 {
            assert_eq!(output1[i], output2[i], "Mismatch at position {}", i);
        }
    }

    #[test]
    fn test_iwht_produces_output() {
        // Test that IWHT produces reasonable output
        let input = [64i16, 32, 16, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut output = [0i16; 16];

        inverse_wht4x4(&input, &mut output);

        // Output should have non-zero values
        assert!(
            output.iter().any(|&x| x != 0),
            "IWHT should produce non-zero output"
        );
    }

    #[test]
    fn test_idct_produces_output() {
        // Test that IDCT produces reasonable output from typical coefficients
        let input = [128i16, 64, 32, 16, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut output = [0i16; 16];

        inverse_dct4x4(&input, &mut output);

        // Output should have non-zero values
        assert!(
            output.iter().any(|&x| x != 0),
            "IDCT should produce non-zero output"
        );
    }

    #[test]
    fn test_transforms_deterministic() {
        // Test that transforms are deterministic
        let input = [100i16, 50, 25, 12, 6, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        let mut output1 = [0i16; 16];
        let mut output2 = [0i16; 16];

        inverse_dct4x4(&input, &mut output1);
        inverse_dct4x4(&input, &mut output2);

        assert_eq!(output1, output2, "IDCT should be deterministic");

        inverse_wht4x4(&input, &mut output1);
        inverse_wht4x4(&input, &mut output2);

        assert_eq!(output1, output2, "IWHT should be deterministic");
    }

    #[test]
    fn test_add_residual() {
        let pred = [128u8; 16];
        let residual = [10i16, -10, 20, -20, 0, 0, 0, 0, 127, -128, 0, 0, 0, 0, 0, 0];
        let mut output = [0u8; 16];

        add_residual_to_prediction(&pred, 4, &residual, &mut output, 4);

        assert_eq!(output[0], 138); // 128 + 10
        assert_eq!(output[1], 118); // 128 - 10
        assert_eq!(output[2], 148); // 128 + 20
        assert_eq!(output[3], 108); // 128 - 20
        assert_eq!(output[8], 255); // 128 + 127 clamped
        assert_eq!(output[9], 0); // 128 - 128
    }
}
