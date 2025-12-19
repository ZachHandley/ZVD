//! Minimal 8x8 integer IDCT and dequant for ProRes.
//! This is a straightforward port of the AAN-style integer IDCT (10/12-bit capable)
//! sufficient for ProRes decoding. It assumes input coefficients are already de-zigzagged.

pub fn idct_8x8(block: &mut [i32; 64]) {
    // Use 64-bit intermediates to avoid overflow on pathological input.
    let mut tmp = [0i64; 64];

    // Row transform
    for i in 0..8 {
        let idx = i * 8;
        let s07 = block[idx + 0] as i64 + block[idx + 7] as i64;
        let d07 = block[idx + 0] as i64 - block[idx + 7] as i64;
        let s16 = block[idx + 1] as i64 + block[idx + 6] as i64;
        let d16 = block[idx + 1] as i64 - block[idx + 6] as i64;
        let s25 = block[idx + 2] as i64 + block[idx + 5] as i64;
        let d25 = block[idx + 2] as i64 - block[idx + 5] as i64;
        let s34 = block[idx + 3] as i64 + block[idx + 4] as i64;
        let d34 = block[idx + 3] as i64 - block[idx + 4] as i64;

        let s0734 = s07 + s34;
        let d0734 = s07 - s34;
        let s1625 = s16 + s25;
        let d1625 = s16 - s25;

        tmp[idx + 0] = s0734 + s1625;
        tmp[idx + 4] = s0734 - s1625;
        tmp[idx + 2] = d0734 + ((d1625 * 35468) >> 15);
        tmp[idx + 6] = d0734 - ((d1625 * 35468) >> 15);

        tmp[idx + 1] = d07 + ((d34 * 46341) >> 16) + ((d25 * 39200) >> 16);
        tmp[idx + 3] = d07 - ((d34 * 46341) >> 16) - ((d25 * 39200) >> 16);
        tmp[idx + 5] = d16 + ((d34 * 39200) >> 16) - ((d25 * 46341) >> 16);
        tmp[idx + 7] = d16 - ((d34 * 39200) >> 16) + ((d25 * 46341) >> 16);
    }

    // Column transform
    for i in 0..8 {
        let s07 = tmp[i] + tmp[56 + i];
        let d07 = tmp[i] - tmp[56 + i];
        let s16 = tmp[8 + i] + tmp[48 + i];
        let d16 = tmp[8 + i] - tmp[48 + i];
        let s25 = tmp[16 + i] + tmp[40 + i];
        let d25 = tmp[16 + i] - tmp[40 + i];
        let s34 = tmp[24 + i] + tmp[32 + i];
        let d34 = tmp[24 + i] - tmp[32 + i];

        let s0734 = s07 + s34;
        let d0734 = s07 - s34;
        let s1625 = s16 + s25;
        let d1625 = s16 - s25;

        block[i] = ((s0734 + s1625 + 4) >> 3).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        block[32 + i] = ((s0734 - s1625 + 4) >> 3).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        block[16 + i] = ((d0734 + ((d1625 * 35468) >> 15) + 4) >> 3)
            .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        block[48 + i] = ((d0734 - ((d1625 * 35468) >> 15) + 4) >> 3)
            .clamp(i32::MIN as i64, i32::MAX as i64) as i32;

        block[8 + i] = ((d07 + ((d34 * 46341) >> 16) + ((d25 * 39200) >> 16) + 4) >> 3)
            .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        block[24 + i] = ((d07 - ((d34 * 46341) >> 16) - ((d25 * 39200) >> 16) + 4) >> 3)
            .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        block[40 + i] = ((d16 + ((d34 * 39200) >> 16) - ((d25 * 46341) >> 16) + 4) >> 3)
            .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        block[56 + i] = ((d16 - ((d34 * 39200) >> 16) + ((d25 * 46341) >> 16) + 4) >> 3)
            .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }
}

pub fn dequant_block(coeffs: &[i16; 64], qmat: &[u8; 64], quant_mul: i32) -> [i32; 64] {
    let mut out = [0i32; 64];
    for i in 0..64 {
        out[i] = coeffs[i] as i32 * qmat[i] as i32 * quant_mul >> 2;
    }
    out
}
