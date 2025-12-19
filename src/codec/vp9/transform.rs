//! VP9 Transform Implementations
//!
//! VP9 uses multiple transform types and sizes:
//! - DCT (Discrete Cosine Transform): 4x4, 8x8, 16x16, 32x32
//! - ADST (Asymmetric Discrete Sine Transform): 4x4, 8x8, 16x16
//! - WHT (Walsh-Hadamard Transform): 4x4 only (lossless mode)
//!
//! All transforms use fixed-point arithmetic for precision and speed.

use super::tables::{TxSize, TxType};

// =============================================================================
// Fixed-Point Constants
// =============================================================================

// DCT constants (cos values * 16384)
const COSPI_1_64: i32 = 16364;
const COSPI_2_64: i32 = 16305;
const COSPI_3_64: i32 = 16207;
const COSPI_4_64: i32 = 16069;
const COSPI_5_64: i32 = 15893;
const COSPI_6_64: i32 = 15679;
const COSPI_7_64: i32 = 15426;
const COSPI_8_64: i32 = 15137;
const COSPI_9_64: i32 = 14811;
const COSPI_10_64: i32 = 14449;
const COSPI_11_64: i32 = 14053;
const COSPI_12_64: i32 = 13623;
const COSPI_13_64: i32 = 13160;
const COSPI_14_64: i32 = 12665;
const COSPI_15_64: i32 = 12140;
const COSPI_16_64: i32 = 11585;
const COSPI_17_64: i32 = 11003;
const COSPI_18_64: i32 = 10394;
const COSPI_19_64: i32 = 9760;
const COSPI_20_64: i32 = 9102;
const COSPI_21_64: i32 = 8423;
const COSPI_22_64: i32 = 7723;
const COSPI_23_64: i32 = 7005;
const COSPI_24_64: i32 = 6270;
const COSPI_25_64: i32 = 5520;
const COSPI_26_64: i32 = 4756;
const COSPI_27_64: i32 = 3981;
const COSPI_28_64: i32 = 3196;
const COSPI_29_64: i32 = 2404;
const COSPI_30_64: i32 = 1606;
const COSPI_31_64: i32 = 804;

// ADST constants
const SINPI_1_9: i32 = 5283;
const SINPI_2_9: i32 = 9929;
const SINPI_3_9: i32 = 13377;
const SINPI_4_9: i32 = 15212;

// =============================================================================
// Helper Macros
// =============================================================================

/// Rounding shift for DCT
#[inline(always)]
fn dct_round_shift(x: i32) -> i32 {
    (x + (1 << 13)) >> 14
}

/// Butterfly operation
#[inline(always)]
fn butterfly(a: i32, b: i32, c: i32, d: i32) -> (i32, i32) {
    let e = a * c + b * d;
    let f = a * d - b * c;
    (dct_round_shift(e), dct_round_shift(f))
}

// =============================================================================
// 4x4 Transforms
// =============================================================================

/// 4x4 Inverse DCT
pub fn idct4(input: &[i32], output: &mut [i32]) {
    let x0 = input[0];
    let x1 = input[1];
    let x2 = input[2];
    let x3 = input[3];

    // Stage 1
    let s0 = x0 + x2;
    let s1 = x0 - x2;
    let (s2, s3) = butterfly(x1, x3, COSPI_24_64, COSPI_8_64);
    let s2 = s3;
    let s3 = dct_round_shift(x1 * COSPI_8_64 + x3 * COSPI_24_64);

    // Simpler version that matches VP9 spec better
    let a = dct_round_shift((x0 + x2) * COSPI_16_64);
    let b = dct_round_shift((x0 - x2) * COSPI_16_64);
    let c = dct_round_shift(x1 * COSPI_24_64 - x3 * COSPI_8_64);
    let d = dct_round_shift(x1 * COSPI_8_64 + x3 * COSPI_24_64);

    output[0] = a + d;
    output[1] = b + c;
    output[2] = b - c;
    output[3] = a - d;
}

/// 4x4 Inverse ADST
pub fn iadst4(input: &[i32], output: &mut [i32]) {
    let x0 = input[0];
    let x1 = input[1];
    let x2 = input[2];
    let x3 = input[3];

    // Intermediate calculations
    let s0 = SINPI_1_9 * x0;
    let s1 = SINPI_2_9 * x0;
    let s2 = SINPI_3_9 * x1;
    let s3 = SINPI_4_9 * x2;
    let s4 = SINPI_1_9 * x2;
    let s5 = SINPI_2_9 * x3;
    let s6 = SINPI_4_9 * x3;

    let x0 = s0 + s3 + s5;
    let x1 = s1 - s4 - s6;
    let x2 = SINPI_3_9 * (x0 + x2 - x3);
    let x3 = s2;

    output[0] = dct_round_shift(x0 + x3);
    output[1] = dct_round_shift(x1 + x3);
    output[2] = dct_round_shift(x2);
    output[3] = dct_round_shift(x0 + x1 - x3);
}

/// 4x4 2D Inverse Transform
pub fn idct4x4(input: &[i32; 16], output: &mut [i32; 16], tx_type: TxType) {
    let mut temp = [[0i32; 4]; 4];
    let mut temp2 = [[0i32; 4]; 4];

    // Copy input to 2D array
    for i in 0..4 {
        for j in 0..4 {
            temp[i][j] = input[i * 4 + j];
        }
    }

    // First pass: columns
    for j in 0..4 {
        let col = [temp[0][j], temp[1][j], temp[2][j], temp[3][j]];
        let mut out = [0i32; 4];

        match tx_type {
            TxType::DctDct | TxType::DctAdst => idct4(&col, &mut out),
            TxType::AdstDct | TxType::AdstAdst => iadst4(&col, &mut out),
        }

        for i in 0..4 {
            temp2[i][j] = out[i];
        }
    }

    // Second pass: rows
    for i in 0..4 {
        let row = [temp2[i][0], temp2[i][1], temp2[i][2], temp2[i][3]];
        let mut out = [0i32; 4];

        match tx_type {
            TxType::DctDct | TxType::AdstDct => idct4(&row, &mut out),
            TxType::DctAdst | TxType::AdstAdst => iadst4(&row, &mut out),
        }

        for j in 0..4 {
            // Final rounding shift
            output[i * 4 + j] = (out[j] + 8) >> 4;
        }
    }
}

// =============================================================================
// 8x8 Transforms
// =============================================================================

/// 8x8 Inverse DCT
pub fn idct8(input: &[i32], output: &mut [i32]) {
    // Stage 1
    let x0 = input[0];
    let x1 = input[4];
    let x2 = input[2];
    let x3 = input[6];
    let x4 = input[1];
    let x5 = input[5];
    let x6 = input[3];
    let x7 = input[7];

    // Even part - 4-point IDCT
    let s0 = dct_round_shift((x0 + x1) * COSPI_16_64);
    let s1 = dct_round_shift((x0 - x1) * COSPI_16_64);
    let s2 = dct_round_shift(x2 * COSPI_24_64 - x3 * COSPI_8_64);
    let s3 = dct_round_shift(x2 * COSPI_8_64 + x3 * COSPI_24_64);

    let x0 = s0 + s3;
    let x1 = s1 + s2;
    let x2 = s1 - s2;
    let x3 = s0 - s3;

    // Odd part
    let s4 = dct_round_shift(x4 * COSPI_28_64 - x7 * COSPI_4_64);
    let s7 = dct_round_shift(x4 * COSPI_4_64 + x7 * COSPI_28_64);
    let s5 = dct_round_shift(x5 * COSPI_12_64 - x6 * COSPI_20_64);
    let s6 = dct_round_shift(x5 * COSPI_20_64 + x6 * COSPI_12_64);

    let x4 = s4 + s5;
    let x5 = s4 - s5;
    let x6 = s7 - s6;
    let x7 = s7 + s6;

    let s5 = dct_round_shift((x6 - x5) * COSPI_16_64);
    let s6 = dct_round_shift((x6 + x5) * COSPI_16_64);

    // Final stage
    output[0] = x0 + x7;
    output[1] = x1 + s6;
    output[2] = x2 + s5;
    output[3] = x3 + x4;
    output[4] = x3 - x4;
    output[5] = x2 - s5;
    output[6] = x1 - s6;
    output[7] = x0 - x7;
}

/// 8x8 Inverse ADST
pub fn iadst8(input: &[i32], output: &mut [i32]) {
    let x0 = input[7];
    let x1 = input[0];
    let x2 = input[5];
    let x3 = input[2];
    let x4 = input[3];
    let x5 = input[4];
    let x6 = input[1];
    let x7 = input[6];

    // Stage 1
    let s0 = COSPI_2_64 * x0 + COSPI_30_64 * x1;
    let s1 = COSPI_30_64 * x0 - COSPI_2_64 * x1;
    let s2 = COSPI_10_64 * x2 + COSPI_22_64 * x3;
    let s3 = COSPI_22_64 * x2 - COSPI_10_64 * x3;
    let s4 = COSPI_18_64 * x4 + COSPI_14_64 * x5;
    let s5 = COSPI_14_64 * x4 - COSPI_18_64 * x5;
    let s6 = COSPI_26_64 * x6 + COSPI_6_64 * x7;
    let s7 = COSPI_6_64 * x6 - COSPI_26_64 * x7;

    let x0 = dct_round_shift(s0 + s4);
    let x1 = dct_round_shift(s1 + s5);
    let x2 = dct_round_shift(s2 + s6);
    let x3 = dct_round_shift(s3 + s7);
    let x4 = dct_round_shift(s0 - s4);
    let x5 = dct_round_shift(s1 - s5);
    let x6 = dct_round_shift(s2 - s6);
    let x7 = dct_round_shift(s3 - s7);

    // Stage 2
    let s0 = x0;
    let s1 = x1;
    let s2 = x2;
    let s3 = x3;
    let s4 = COSPI_8_64 * x4 + COSPI_24_64 * x5;
    let s5 = COSPI_24_64 * x4 - COSPI_8_64 * x5;
    let s6 = -COSPI_24_64 * x6 + COSPI_8_64 * x7;
    let s7 = COSPI_8_64 * x6 + COSPI_24_64 * x7;

    let x0 = s0 + s2;
    let x1 = s1 + s3;
    let x2 = s0 - s2;
    let x3 = s1 - s3;
    let x4 = dct_round_shift(s4 + s6);
    let x5 = dct_round_shift(s5 + s7);
    let x6 = dct_round_shift(s4 - s6);
    let x7 = dct_round_shift(s5 - s7);

    // Stage 3
    let s2 = COSPI_16_64 * (x2 + x3);
    let s3 = COSPI_16_64 * (x2 - x3);
    let s6 = COSPI_16_64 * (x6 + x7);
    let s7 = COSPI_16_64 * (x6 - x7);

    let x2 = dct_round_shift(s2);
    let x3 = dct_round_shift(s3);
    let x6 = dct_round_shift(s6);
    let x7 = dct_round_shift(s7);

    output[0] = x0;
    output[1] = -x4;
    output[2] = x6;
    output[3] = -x2;
    output[4] = x3;
    output[5] = -x7;
    output[6] = x5;
    output[7] = -x1;
}

/// 8x8 2D Inverse Transform
pub fn idct8x8(input: &[i32; 64], output: &mut [i32; 64], tx_type: TxType) {
    let mut temp = [[0i32; 8]; 8];
    let mut temp2 = [[0i32; 8]; 8];

    // Copy input to 2D array
    for i in 0..8 {
        for j in 0..8 {
            temp[i][j] = input[i * 8 + j];
        }
    }

    // First pass: columns
    for j in 0..8 {
        let col: Vec<i32> = (0..8).map(|i| temp[i][j]).collect();
        let mut out = [0i32; 8];

        match tx_type {
            TxType::DctDct | TxType::DctAdst => idct8(&col, &mut out),
            TxType::AdstDct | TxType::AdstAdst => iadst8(&col, &mut out),
        }

        for i in 0..8 {
            temp2[i][j] = out[i];
        }
    }

    // Second pass: rows
    for i in 0..8 {
        let row: Vec<i32> = (0..8).map(|j| temp2[i][j]).collect();
        let mut out = [0i32; 8];

        match tx_type {
            TxType::DctDct | TxType::AdstDct => idct8(&row, &mut out),
            TxType::DctAdst | TxType::AdstAdst => iadst8(&row, &mut out),
        }

        for j in 0..8 {
            output[i * 8 + j] = (out[j] + 16) >> 5;
        }
    }
}

// =============================================================================
// 16x16 Transforms
// =============================================================================

/// 16x16 Inverse DCT
pub fn idct16(input: &[i32], output: &mut [i32]) {
    // Simplified 16-point IDCT using recursive structure
    let mut even = [0i32; 8];
    let mut odd = [0i32; 8];

    // Extract even and odd indexed inputs
    for i in 0..8 {
        even[i] = input[i * 2];
        odd[i] = input[i * 2 + 1];
    }

    // 8-point IDCT on even
    let mut even_out = [0i32; 8];
    idct8(&even, &mut even_out);

    // Process odd samples
    let x0 = input[1];
    let x1 = input[15];
    let x2 = input[9];
    let x3 = input[7];
    let x4 = input[5];
    let x5 = input[11];
    let x6 = input[13];
    let x7 = input[3];

    let s0 = dct_round_shift(x0 * COSPI_30_64 - x1 * COSPI_2_64);
    let s1 = dct_round_shift(x0 * COSPI_2_64 + x1 * COSPI_30_64);
    let s2 = dct_round_shift(x2 * COSPI_14_64 - x3 * COSPI_18_64);
    let s3 = dct_round_shift(x2 * COSPI_18_64 + x3 * COSPI_14_64);
    let s4 = dct_round_shift(x4 * COSPI_22_64 - x5 * COSPI_10_64);
    let s5 = dct_round_shift(x4 * COSPI_10_64 + x5 * COSPI_22_64);
    let s6 = dct_round_shift(x6 * COSPI_6_64 - x7 * COSPI_26_64);
    let s7 = dct_round_shift(x6 * COSPI_26_64 + x7 * COSPI_6_64);

    let t0 = s0 + s4;
    let t1 = s1 + s5;
    let t2 = s2 + s6;
    let t3 = s3 + s7;
    let t4 = s0 - s4;
    let t5 = s1 - s5;
    let t6 = s2 - s6;
    let t7 = s3 - s7;

    let s4 = dct_round_shift(t4 * COSPI_8_64 + t5 * COSPI_24_64);
    let s5 = dct_round_shift(t4 * COSPI_24_64 - t5 * COSPI_8_64);
    let s6 = dct_round_shift(-t6 * COSPI_24_64 + t7 * COSPI_8_64);
    let s7 = dct_round_shift(t6 * COSPI_8_64 + t7 * COSPI_24_64);

    let t0 = t0 + t2;
    let t1 = t1 + t3;
    let t2 = t0 - t2;
    let t3 = t1 - t3;
    let t4 = s4 + s6;
    let t5 = s5 + s7;
    let t6 = s4 - s6;
    let t7 = s5 - s7;

    let s2 = dct_round_shift((t2 + t3) * COSPI_16_64);
    let s3 = dct_round_shift((t2 - t3) * COSPI_16_64);
    let s6 = dct_round_shift((t6 + t7) * COSPI_16_64);
    let s7 = dct_round_shift((t6 - t7) * COSPI_16_64);

    let odd_out = [t0, t1, s2, s3, t4, t5, s6, s7];

    // Combine even and odd
    for i in 0..8 {
        output[i] = even_out[i] + odd_out[7 - i];
        output[15 - i] = even_out[i] - odd_out[7 - i];
    }
}

/// 16x16 Inverse ADST
pub fn iadst16(input: &[i32], output: &mut [i32]) {
    // Simplified IADST16 - use DCT as fallback for now
    // Full ADST16 would have more stages
    let mut temp = [0i32; 16];

    // Reorder and apply DCT
    for i in 0..16 {
        temp[i] = input[i];
    }

    idct16(&temp, output);

    // Apply sign changes for ADST
    for i in 0..16 {
        if i % 2 == 1 {
            output[i] = -output[i];
        }
    }
}

/// 16x16 2D Inverse Transform
pub fn idct16x16(input: &[i32; 256], output: &mut [i32; 256], tx_type: TxType) {
    let mut temp = [[0i32; 16]; 16];
    let mut temp2 = [[0i32; 16]; 16];

    // Copy input to 2D array
    for i in 0..16 {
        for j in 0..16 {
            temp[i][j] = input[i * 16 + j];
        }
    }

    // First pass: columns
    for j in 0..16 {
        let col: Vec<i32> = (0..16).map(|i| temp[i][j]).collect();
        let mut out = [0i32; 16];

        match tx_type {
            TxType::DctDct | TxType::DctAdst => idct16(&col, &mut out),
            TxType::AdstDct | TxType::AdstAdst => iadst16(&col, &mut out),
        }

        for i in 0..16 {
            temp2[i][j] = out[i];
        }
    }

    // Second pass: rows
    for i in 0..16 {
        let row: Vec<i32> = (0..16).map(|j| temp2[i][j]).collect();
        let mut out = [0i32; 16];

        match tx_type {
            TxType::DctDct | TxType::AdstDct => idct16(&row, &mut out),
            TxType::DctAdst | TxType::AdstAdst => iadst16(&row, &mut out),
        }

        for j in 0..16 {
            output[i * 16 + j] = (out[j] + 32) >> 6;
        }
    }
}

// =============================================================================
// 32x32 Transforms
// =============================================================================

/// 32x32 Inverse DCT
pub fn idct32(input: &[i32], output: &mut [i32]) {
    // Simplified 32-point IDCT using recursive decomposition
    let mut even = [0i32; 16];
    let mut odd = [0i32; 16];

    // Extract even and odd indexed inputs
    for i in 0..16 {
        even[i] = input[i * 2];
        odd[i] = input[i * 2 + 1];
    }

    // 16-point IDCT on even
    let mut even_out = [0i32; 16];
    idct16(&even, &mut even_out);

    // Process odd samples (simplified)
    let mut odd_out = [0i32; 16];
    for i in 0..16 {
        odd_out[i] = dct_round_shift(odd[i] * COSPI_16_64);
    }

    // Combine
    for i in 0..16 {
        output[i] = even_out[i] + odd_out[15 - i];
        output[31 - i] = even_out[i] - odd_out[15 - i];
    }
}

/// 32x32 2D Inverse DCT (DCT only, no ADST for 32x32)
pub fn idct32x32(input: &[i32; 1024], output: &mut [i32; 1024]) {
    let mut temp = [[0i32; 32]; 32];
    let mut temp2 = [[0i32; 32]; 32];

    // Copy input to 2D array
    for i in 0..32 {
        for j in 0..32 {
            temp[i][j] = input[i * 32 + j];
        }
    }

    // First pass: columns
    for j in 0..32 {
        let col: Vec<i32> = (0..32).map(|i| temp[i][j]).collect();
        let mut out = [0i32; 32];
        idct32(&col, &mut out);

        for i in 0..32 {
            temp2[i][j] = out[i];
        }
    }

    // Second pass: rows
    for i in 0..32 {
        let row: Vec<i32> = (0..32).map(|j| temp2[i][j]).collect();
        let mut out = [0i32; 32];
        idct32(&row, &mut out);

        for j in 0..32 {
            output[i * 32 + j] = (out[j] + 64) >> 7;
        }
    }
}

// =============================================================================
// Walsh-Hadamard Transform (Lossless)
// =============================================================================

/// 4x4 Inverse Walsh-Hadamard Transform
pub fn iwht4x4(input: &[i32; 16], output: &mut [i32; 16]) {
    let mut temp = [0i32; 16];

    // Row transform
    for i in 0..4 {
        let row_offset = i * 4;
        let a = input[row_offset] + input[row_offset + 3];
        let b = input[row_offset + 1] + input[row_offset + 2];
        let c = input[row_offset + 1] - input[row_offset + 2];
        let d = input[row_offset] - input[row_offset + 3];

        temp[row_offset] = a + b;
        temp[row_offset + 1] = c + d;
        temp[row_offset + 2] = a - b;
        temp[row_offset + 3] = d - c;
    }

    // Column transform
    for j in 0..4 {
        let a = temp[j] + temp[12 + j];
        let b = temp[4 + j] + temp[8 + j];
        let c = temp[4 + j] - temp[8 + j];
        let d = temp[j] - temp[12 + j];

        output[j] = (a + b) >> 2;
        output[4 + j] = (c + d) >> 2;
        output[8 + j] = (a - b) >> 2;
        output[12 + j] = (d - c) >> 2;
    }
}

// =============================================================================
// Dispatch Functions
// =============================================================================

/// Perform inverse transform based on size and type
pub fn inverse_transform(
    input: &[i32],
    output: &mut [i32],
    tx_size: TxSize,
    tx_type: TxType,
    lossless: bool,
) {
    if lossless && tx_size == TxSize::Tx4x4 {
        // Use WHT for lossless mode
        let mut in_arr = [0i32; 16];
        let mut out_arr = [0i32; 16];
        in_arr[..16.min(input.len())].copy_from_slice(&input[..16.min(input.len())]);
        iwht4x4(&in_arr, &mut out_arr);
        output[..16].copy_from_slice(&out_arr);
        return;
    }

    match tx_size {
        TxSize::Tx4x4 => {
            let mut in_arr = [0i32; 16];
            let mut out_arr = [0i32; 16];
            in_arr[..16.min(input.len())].copy_from_slice(&input[..16.min(input.len())]);
            idct4x4(&in_arr, &mut out_arr, tx_type);
            output[..16].copy_from_slice(&out_arr);
        }
        TxSize::Tx8x8 => {
            let mut in_arr = [0i32; 64];
            let mut out_arr = [0i32; 64];
            in_arr[..64.min(input.len())].copy_from_slice(&input[..64.min(input.len())]);
            idct8x8(&in_arr, &mut out_arr, tx_type);
            output[..64].copy_from_slice(&out_arr);
        }
        TxSize::Tx16x16 => {
            let mut in_arr = [0i32; 256];
            let mut out_arr = [0i32; 256];
            in_arr[..256.min(input.len())].copy_from_slice(&input[..256.min(input.len())]);
            idct16x16(&in_arr, &mut out_arr, tx_type);
            output[..256].copy_from_slice(&out_arr);
        }
        TxSize::Tx32x32 => {
            let mut in_arr = [0i32; 1024];
            let mut out_arr = [0i32; 1024];
            in_arr[..1024.min(input.len())].copy_from_slice(&input[..1024.min(input.len())]);
            idct32x32(&in_arr, &mut out_arr);
            output[..1024].copy_from_slice(&out_arr);
        }
    }
}

/// Add residual to prediction and clamp to pixel range
pub fn add_residual(
    residual: &[i32],
    prediction: &[u8],
    output: &mut [u8],
    width: usize,
    height: usize,
    pred_stride: usize,
    out_stride: usize,
) {
    for y in 0..height {
        for x in 0..width {
            let pred_val = prediction[y * pred_stride + x] as i32;
            let res_val = residual[y * width + x];
            output[y * out_stride + x] = (pred_val + res_val).clamp(0, 255) as u8;
        }
    }
}

// =============================================================================
// Forward Transforms (for encoding)
// =============================================================================

/// 4x4 Forward DCT
pub fn fdct4(input: &[i32], output: &mut [i32]) {
    let x0 = input[0];
    let x1 = input[1];
    let x2 = input[2];
    let x3 = input[3];

    // Stage 1: Sum/difference
    let s0 = x0 + x3;
    let s1 = x1 + x2;
    let s2 = x1 - x2;
    let s3 = x0 - x3;

    // Output
    output[0] = dct_round_shift((s0 + s1) * COSPI_16_64);
    output[1] = dct_round_shift(s3 * COSPI_8_64 + s2 * COSPI_24_64);
    output[2] = dct_round_shift((s0 - s1) * COSPI_16_64);
    output[3] = dct_round_shift(s3 * COSPI_24_64 - s2 * COSPI_8_64);
}

/// 4x4 Forward ADST
pub fn fadst4(input: &[i32], output: &mut [i32]) {
    let x0 = input[0];
    let x1 = input[1];
    let x2 = input[2];
    let x3 = input[3];

    // ADST calculations
    let s0 = SINPI_1_9 * x0;
    let s1 = SINPI_2_9 * x0;
    let s2 = SINPI_3_9 * x1;
    let s3 = SINPI_4_9 * x2;
    let s4 = SINPI_1_9 * x2;
    let s5 = SINPI_2_9 * x3;
    let s6 = SINPI_4_9 * x3;

    let a0 = s0 + s3 + s5;
    let a1 = s1 - s4 - s6;
    let a2 = SINPI_3_9 * (x0 - x2 + x3);
    let a3 = s2;

    output[0] = dct_round_shift(a0 + a3);
    output[1] = dct_round_shift(a1 + a3);
    output[2] = dct_round_shift(a2);
    output[3] = dct_round_shift(a0 + a1 - a3);
}

/// 4x4 2D Forward Transform
pub fn fdct4x4(input: &[i32; 16], output: &mut [i32; 16], tx_type: TxType) {
    let mut temp = [[0i32; 4]; 4];
    let mut temp2 = [[0i32; 4]; 4];

    // Copy input to 2D array
    for i in 0..4 {
        for j in 0..4 {
            temp[i][j] = input[i * 4 + j];
        }
    }

    // First pass: rows
    for i in 0..4 {
        let row = [temp[i][0], temp[i][1], temp[i][2], temp[i][3]];
        let mut out = [0i32; 4];

        match tx_type {
            TxType::DctDct | TxType::DctAdst => fdct4(&row, &mut out),
            TxType::AdstDct | TxType::AdstAdst => fadst4(&row, &mut out),
        }

        for j in 0..4 {
            temp2[i][j] = out[j];
        }
    }

    // Second pass: columns
    for j in 0..4 {
        let col = [temp2[0][j], temp2[1][j], temp2[2][j], temp2[3][j]];
        let mut out = [0i32; 4];

        match tx_type {
            TxType::DctDct | TxType::AdstDct => fdct4(&col, &mut out),
            TxType::DctAdst | TxType::AdstAdst => fadst4(&col, &mut out),
        }

        for i in 0..4 {
            output[i * 4 + j] = out[i];
        }
    }
}

/// 8x8 Forward DCT
pub fn fdct8(input: &[i32], output: &mut [i32]) {
    // Even part (4-point DCT)
    let s0 = input[0] + input[7];
    let s1 = input[1] + input[6];
    let s2 = input[2] + input[5];
    let s3 = input[3] + input[4];

    let s4 = input[3] - input[4];
    let s5 = input[2] - input[5];
    let s6 = input[1] - input[6];
    let s7 = input[0] - input[7];

    // Even: 4-point DCT on s0-s3
    let x0 = s0 + s3;
    let x1 = s1 + s2;
    let x2 = s1 - s2;
    let x3 = s0 - s3;

    output[0] = dct_round_shift((x0 + x1) * COSPI_16_64);
    output[4] = dct_round_shift((x0 - x1) * COSPI_16_64);
    output[2] = dct_round_shift(x3 * COSPI_8_64 + x2 * COSPI_24_64);
    output[6] = dct_round_shift(x3 * COSPI_24_64 - x2 * COSPI_8_64);

    // Odd part
    let t0 = dct_round_shift((s4 + s5) * COSPI_16_64);
    let t1 = dct_round_shift((s4 - s5) * COSPI_16_64);
    let t2 = s6 + t0;
    let t3 = s7 + t1;
    let t4 = s6 - t0;
    let t5 = s7 - t1;

    output[1] = dct_round_shift(t3 * COSPI_4_64 + t4 * COSPI_28_64);
    output[7] = dct_round_shift(t3 * COSPI_28_64 - t4 * COSPI_4_64);
    output[5] = dct_round_shift(t2 * COSPI_20_64 + t5 * COSPI_12_64);
    output[3] = dct_round_shift(t2 * COSPI_12_64 - t5 * COSPI_20_64);
}

/// 8x8 Forward ADST
pub fn fadst8(input: &[i32], output: &mut [i32]) {
    // Forward ADST8 implementation
    let x0 = input[0];
    let x1 = input[1];
    let x2 = input[2];
    let x3 = input[3];
    let x4 = input[4];
    let x5 = input[5];
    let x6 = input[6];
    let x7 = input[7];

    // Stage 1
    let s0 = COSPI_2_64 * x0 + COSPI_30_64 * x7;
    let s1 = COSPI_30_64 * x0 - COSPI_2_64 * x7;
    let s2 = COSPI_10_64 * x1 + COSPI_22_64 * x6;
    let s3 = COSPI_22_64 * x1 - COSPI_10_64 * x6;
    let s4 = COSPI_18_64 * x2 + COSPI_14_64 * x5;
    let s5 = COSPI_14_64 * x2 - COSPI_18_64 * x5;
    let s6 = COSPI_26_64 * x3 + COSPI_6_64 * x4;
    let s7 = COSPI_6_64 * x3 - COSPI_26_64 * x4;

    let t0 = dct_round_shift(s0 + s4);
    let t1 = dct_round_shift(s1 + s5);
    let t2 = dct_round_shift(s2 + s6);
    let t3 = dct_round_shift(s3 + s7);
    let t4 = dct_round_shift(s0 - s4);
    let t5 = dct_round_shift(s1 - s5);
    let t6 = dct_round_shift(s2 - s6);
    let t7 = dct_round_shift(s3 - s7);

    // Stage 2
    let s0 = t0;
    let s1 = t1;
    let s2 = t2;
    let s3 = t3;
    let s4 = COSPI_8_64 * t4 + COSPI_24_64 * t5;
    let s5 = COSPI_24_64 * t4 - COSPI_8_64 * t5;
    let s6 = -COSPI_24_64 * t6 + COSPI_8_64 * t7;
    let s7 = COSPI_8_64 * t6 + COSPI_24_64 * t7;

    let t0 = s0 + s2;
    let t1 = s1 + s3;
    let t2 = s0 - s2;
    let t3 = s1 - s3;
    let t4 = dct_round_shift(s4 + s6);
    let t5 = dct_round_shift(s5 + s7);
    let t6 = dct_round_shift(s4 - s6);
    let t7 = dct_round_shift(s5 - s7);

    // Stage 3
    let s2 = COSPI_16_64 * (t2 + t3);
    let s3 = COSPI_16_64 * (t2 - t3);
    let s6 = COSPI_16_64 * (t6 + t7);
    let s7 = COSPI_16_64 * (t6 - t7);

    output[0] = t0;
    output[1] = -t4;
    output[2] = dct_round_shift(s6);
    output[3] = -dct_round_shift(s2);
    output[4] = dct_round_shift(s3);
    output[5] = -dct_round_shift(s7);
    output[6] = t5;
    output[7] = -t1;
}

/// 8x8 2D Forward Transform
pub fn fdct8x8(input: &[i32; 64], output: &mut [i32; 64], tx_type: TxType) {
    let mut temp = [[0i32; 8]; 8];
    let mut temp2 = [[0i32; 8]; 8];

    // Copy input to 2D array
    for i in 0..8 {
        for j in 0..8 {
            temp[i][j] = input[i * 8 + j];
        }
    }

    // First pass: rows
    for i in 0..8 {
        let row: Vec<i32> = (0..8).map(|j| temp[i][j]).collect();
        let mut out = [0i32; 8];

        match tx_type {
            TxType::DctDct | TxType::DctAdst => fdct8(&row, &mut out),
            TxType::AdstDct | TxType::AdstAdst => fadst8(&row, &mut out),
        }

        for j in 0..8 {
            temp2[i][j] = out[j];
        }
    }

    // Second pass: columns
    for j in 0..8 {
        let col: Vec<i32> = (0..8).map(|i| temp2[i][j]).collect();
        let mut out = [0i32; 8];

        match tx_type {
            TxType::DctDct | TxType::AdstDct => fdct8(&col, &mut out),
            TxType::DctAdst | TxType::AdstAdst => fadst8(&col, &mut out),
        }

        for i in 0..8 {
            output[i * 8 + j] = out[i];
        }
    }
}

/// 16x16 Forward DCT
pub fn fdct16(input: &[i32], output: &mut [i32]) {
    // Even/odd decomposition
    let mut even = [0i32; 8];
    let mut odd = [0i32; 8];

    for i in 0..8 {
        even[i] = input[i] + input[15 - i];
        odd[i] = input[i] - input[15 - i];
    }

    // 8-point DCT on even
    let mut even_out = [0i32; 8];
    fdct8(&even, &mut even_out);

    // Place even outputs
    output[0] = even_out[0];
    output[2] = even_out[1];
    output[4] = even_out[2];
    output[6] = even_out[3];
    output[8] = even_out[4];
    output[10] = even_out[5];
    output[12] = even_out[6];
    output[14] = even_out[7];

    // Process odd part
    let x0 = odd[0];
    let x1 = odd[1];
    let x2 = odd[2];
    let x3 = odd[3];
    let x4 = odd[4];
    let x5 = odd[5];
    let x6 = odd[6];
    let x7 = odd[7];

    let s0 = dct_round_shift(x0 * COSPI_2_64 + x7 * COSPI_30_64);
    let s7 = dct_round_shift(x0 * COSPI_30_64 - x7 * COSPI_2_64);
    let s1 = dct_round_shift(x1 * COSPI_10_64 + x6 * COSPI_22_64);
    let s6 = dct_round_shift(x1 * COSPI_22_64 - x6 * COSPI_10_64);
    let s2 = dct_round_shift(x2 * COSPI_18_64 + x5 * COSPI_14_64);
    let s5 = dct_round_shift(x2 * COSPI_14_64 - x5 * COSPI_18_64);
    let s3 = dct_round_shift(x3 * COSPI_26_64 + x4 * COSPI_6_64);
    let s4 = dct_round_shift(x3 * COSPI_6_64 - x4 * COSPI_26_64);

    output[1] = s0;
    output[3] = s1;
    output[5] = s2;
    output[7] = s3;
    output[9] = s4;
    output[11] = s5;
    output[13] = s6;
    output[15] = s7;
}

/// 16x16 Forward ADST
pub fn fadst16(input: &[i32], output: &mut [i32]) {
    // Simplified implementation using DCT with sign adjustments
    let mut temp = [0i32; 16];
    temp.copy_from_slice(&input[..16]);

    fdct16(&temp, output);

    // Apply sign changes for ADST approximation
    for i in 0..16 {
        if i % 2 == 1 {
            output[i] = -output[i];
        }
    }
}

/// 16x16 2D Forward Transform
pub fn fdct16x16(input: &[i32; 256], output: &mut [i32; 256], tx_type: TxType) {
    let mut temp = [[0i32; 16]; 16];
    let mut temp2 = [[0i32; 16]; 16];

    // Copy input
    for i in 0..16 {
        for j in 0..16 {
            temp[i][j] = input[i * 16 + j];
        }
    }

    // First pass: rows
    for i in 0..16 {
        let row: Vec<i32> = (0..16).map(|j| temp[i][j]).collect();
        let mut out = [0i32; 16];

        match tx_type {
            TxType::DctDct | TxType::DctAdst => fdct16(&row, &mut out),
            TxType::AdstDct | TxType::AdstAdst => fadst16(&row, &mut out),
        }

        for j in 0..16 {
            temp2[i][j] = out[j];
        }
    }

    // Second pass: columns
    for j in 0..16 {
        let col: Vec<i32> = (0..16).map(|i| temp2[i][j]).collect();
        let mut out = [0i32; 16];

        match tx_type {
            TxType::DctDct | TxType::AdstDct => fdct16(&col, &mut out),
            TxType::DctAdst | TxType::AdstAdst => fadst16(&col, &mut out),
        }

        for i in 0..16 {
            output[i * 16 + j] = out[i];
        }
    }
}

/// 32x32 Forward DCT
pub fn fdct32(input: &[i32], output: &mut [i32]) {
    // Even/odd decomposition
    let mut even = [0i32; 16];
    let mut odd = [0i32; 16];

    for i in 0..16 {
        even[i] = input[i] + input[31 - i];
        odd[i] = input[i] - input[31 - i];
    }

    // 16-point DCT on even
    let mut even_out = [0i32; 16];
    fdct16(&even, &mut even_out);

    // Place even outputs at even indices
    for i in 0..16 {
        output[i * 2] = even_out[i];
    }

    // Process odd part with simplified computation
    for i in 0..16 {
        output[i * 2 + 1] = dct_round_shift(odd[i] * COSPI_16_64);
    }
}

/// 32x32 2D Forward DCT
pub fn fdct32x32(input: &[i32; 1024], output: &mut [i32; 1024]) {
    let mut temp = [[0i32; 32]; 32];
    let mut temp2 = [[0i32; 32]; 32];

    // Copy input
    for i in 0..32 {
        for j in 0..32 {
            temp[i][j] = input[i * 32 + j];
        }
    }

    // First pass: rows
    for i in 0..32 {
        let row: Vec<i32> = (0..32).map(|j| temp[i][j]).collect();
        let mut out = [0i32; 32];
        fdct32(&row, &mut out);

        for j in 0..32 {
            temp2[i][j] = out[j];
        }
    }

    // Second pass: columns
    for j in 0..32 {
        let col: Vec<i32> = (0..32).map(|i| temp2[i][j]).collect();
        let mut out = [0i32; 32];
        fdct32(&col, &mut out);

        for i in 0..32 {
            output[i * 32 + j] = out[i];
        }
    }
}

/// 4x4 Forward Walsh-Hadamard Transform
pub fn fwht4x4(input: &[i32; 16], output: &mut [i32; 16]) {
    let mut temp = [0i32; 16];

    // Row transform
    for i in 0..4 {
        let row_offset = i * 4;
        let a = input[row_offset] + input[row_offset + 3];
        let b = input[row_offset + 1] + input[row_offset + 2];
        let c = input[row_offset + 1] - input[row_offset + 2];
        let d = input[row_offset] - input[row_offset + 3];

        temp[row_offset] = a + b;
        temp[row_offset + 1] = c + d;
        temp[row_offset + 2] = a - b;
        temp[row_offset + 3] = d - c;
    }

    // Column transform
    for j in 0..4 {
        let a = temp[j] + temp[12 + j];
        let b = temp[4 + j] + temp[8 + j];
        let c = temp[4 + j] - temp[8 + j];
        let d = temp[j] - temp[12 + j];

        output[j] = a + b;
        output[4 + j] = c + d;
        output[8 + j] = a - b;
        output[12 + j] = d - c;
    }
}

/// Compute residual block (original - prediction)
pub fn compute_residual(
    original: &[u8],
    prediction: &[u8],
    residual: &mut [i32],
    width: usize,
    height: usize,
    orig_stride: usize,
    pred_stride: usize,
) {
    for y in 0..height {
        for x in 0..width {
            let orig_val = original[y * orig_stride + x] as i32;
            let pred_val = prediction[y * pred_stride + x] as i32;
            residual[y * width + x] = orig_val - pred_val;
        }
    }
}

/// Forward transform dispatch based on size and type
pub fn forward_transform(
    input: &[i32],
    output: &mut [i32],
    tx_size: TxSize,
    tx_type: TxType,
    lossless: bool,
) {
    if lossless && tx_size == TxSize::Tx4x4 {
        // Use WHT for lossless mode
        let mut in_arr = [0i32; 16];
        let mut out_arr = [0i32; 16];
        in_arr[..16.min(input.len())].copy_from_slice(&input[..16.min(input.len())]);
        fwht4x4(&in_arr, &mut out_arr);
        output[..16].copy_from_slice(&out_arr);
        return;
    }

    match tx_size {
        TxSize::Tx4x4 => {
            let mut in_arr = [0i32; 16];
            let mut out_arr = [0i32; 16];
            in_arr[..16.min(input.len())].copy_from_slice(&input[..16.min(input.len())]);
            fdct4x4(&in_arr, &mut out_arr, tx_type);
            output[..16].copy_from_slice(&out_arr);
        }
        TxSize::Tx8x8 => {
            let mut in_arr = [0i32; 64];
            let mut out_arr = [0i32; 64];
            in_arr[..64.min(input.len())].copy_from_slice(&input[..64.min(input.len())]);
            fdct8x8(&in_arr, &mut out_arr, tx_type);
            output[..64].copy_from_slice(&out_arr);
        }
        TxSize::Tx16x16 => {
            let mut in_arr = [0i32; 256];
            let mut out_arr = [0i32; 256];
            in_arr[..256.min(input.len())].copy_from_slice(&input[..256.min(input.len())]);
            fdct16x16(&in_arr, &mut out_arr, tx_type);
            output[..256].copy_from_slice(&out_arr);
        }
        TxSize::Tx32x32 => {
            let mut in_arr = [0i32; 1024];
            let mut out_arr = [0i32; 1024];
            in_arr[..1024.min(input.len())].copy_from_slice(&input[..1024.min(input.len())]);
            fdct32x32(&in_arr, &mut out_arr);
            output[..1024].copy_from_slice(&out_arr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idct4_dc() {
        let input = [100, 0, 0, 0];
        let mut output = [0i32; 4];
        idct4(&input, &mut output);

        // DC should propagate uniformly
        let dc = output[0];
        for &v in &output {
            assert!((v - dc).abs() <= 2, "DC values should be similar");
        }
    }

    #[test]
    fn test_idct4x4_identity() {
        // Test with DC-only input
        let mut input = [0i32; 16];
        input[0] = 256;
        let mut output = [0i32; 16];

        idct4x4(&input, &mut output, TxType::DctDct);

        // All values should be approximately equal for DC-only input
        let avg: i32 = output.iter().sum::<i32>() / 16;
        for &v in &output {
            assert!((v - avg).abs() <= 2);
        }
    }

    #[test]
    fn test_iwht4x4() {
        let input = [64i32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut output = [0i32; 16];

        iwht4x4(&input, &mut output);

        // WHT of DC should give uniform output
        let expected = 64 >> 2; // WHT divides by 4
        for &v in &output {
            assert!((v - expected).abs() <= 1);
        }
    }

    #[test]
    fn test_add_residual() {
        let residual = [10i32, -10, 20, -20];
        let prediction = [128u8, 128, 128, 128];
        let mut output = [0u8; 4];

        add_residual(&residual, &prediction, &mut output, 2, 2, 2, 2);

        assert_eq!(output[0], 138);
        assert_eq!(output[1], 118);
        assert_eq!(output[2], 148);
        assert_eq!(output[3], 108);
    }

    #[test]
    fn test_add_residual_clamp() {
        let residual = [200i32, -200];
        let prediction = [128u8, 128];
        let mut output = [0u8; 2];

        add_residual(&residual, &prediction, &mut output, 2, 1, 2, 2);

        assert_eq!(output[0], 255); // Clamped to max
        assert_eq!(output[1], 0); // Clamped to min
    }
}
