//! ProRes Quantization and Dequantization
//!
//! ProRes uses profile-specific quantization matrices to control quality and bitrate.
//! Each profile (Proxy, LT, Standard, HQ, 4444, 4444 XQ) has different quantization tables.

use super::ProResProfile;
use crate::error::Result;

/// Quantization matrix for ProRes
pub struct QuantMatrix {
    /// Quantization values for each of the 64 DCT coefficients
    pub values: [u8; 64],
}

impl QuantMatrix {
    /// Get quantization matrix for a profile and QP
    pub fn for_profile(profile: ProResProfile, qp: u8) -> Self {
        let base_matrix = match profile {
            ProResProfile::Proxy => Self::PROXY_MATRIX,
            ProResProfile::Lt => Self::LT_MATRIX,
            ProResProfile::Standard => Self::STANDARD_MATRIX,
            ProResProfile::Hq => Self::HQ_MATRIX,
            ProResProfile::FourFourFourFour => Self::FOUR_FOUR_FOUR_FOUR_MATRIX,
            ProResProfile::FourFourFourFourXq => Self::FOUR_FOUR_FOUR_FOUR_XQ_MATRIX,
        };

        // Scale by QP (simplified - real ProRes uses complex scaling)
        let scale = (qp as u16 + 1) / 2;
        let mut values = [0u8; 64];
        for i in 0..64 {
            values[i] = ((base_matrix[i] as u16 * scale) / 8).clamp(1, 255) as u8;
        }

        Self { values }
    }

    // ProRes Proxy quantization matrix (lowest quality)
    const PROXY_MATRIX: [u8; 64] = [
        4,  7,  9,  11, 13, 14, 15, 63,
        7,  7,  11, 12, 14, 15, 63, 63,
        9,  11, 13, 14, 15, 63, 63, 63,
        11, 11, 13, 14, 63, 63, 63, 63,
        11, 13, 14, 63, 63, 63, 63, 63,
        13, 14, 63, 63, 63, 63, 63, 63,
        13, 63, 63, 63, 63, 63, 63, 63,
        63, 63, 63, 63, 63, 63, 63, 63,
    ];

    // ProRes LT quantization matrix
    const LT_MATRIX: [u8; 64] = [
        4,  5,  6,  7,  9,  11, 13, 15,
        5,  5,  7,  8,  11, 13, 15, 17,
        6,  7,  9,  11, 13, 15, 15, 17,
        7,  7,  9,  11, 13, 15, 17, 19,
        7,  9,  11, 13, 14, 16, 19, 23,
        9,  11, 13, 14, 16, 19, 23, 29,
        9,  11, 13, 15, 17, 21, 28, 35,
        11, 13, 16, 17, 21, 28, 35, 41,
    ];

    // ProRes Standard quantization matrix
    const STANDARD_MATRIX: [u8; 64] = [
        4,  4,  5,  5,  6,  7,  7,  9,
        4,  4,  5,  6,  7,  7,  9,  9,
        5,  5,  6,  7,  7,  9,  9,  10,
        5,  5,  6,  7,  7,  9,  9,  10,
        5,  6,  7,  7,  8,  9,  10, 12,
        6,  7,  7,  8,  9,  10, 12, 15,
        6,  7,  7,  9,  10, 11, 14, 17,
        7,  7,  9,  10, 11, 14, 17, 21,
    ];

    // ProRes HQ quantization matrix (higher quality)
    const HQ_MATRIX: [u8; 64] = [
        4,  4,  4,  4,  4,  4,  4,  4,
        4,  4,  4,  4,  4,  4,  4,  4,
        4,  4,  4,  4,  4,  4,  4,  4,
        4,  4,  4,  4,  4,  4,  4,  5,
        4,  4,  4,  4,  4,  4,  5,  5,
        4,  4,  4,  4,  4,  5,  5,  6,
        4,  4,  4,  4,  5,  5,  6,  7,
        4,  4,  4,  5,  5,  6,  7,  7,
    ];

    // ProRes 4444 quantization matrix (supports alpha)
    const FOUR_FOUR_FOUR_FOUR_MATRIX: [u8; 64] = [
        4,  4,  4,  4,  4,  4,  4,  4,
        4,  4,  4,  4,  4,  4,  4,  4,
        4,  4,  4,  4,  4,  4,  4,  4,
        4,  4,  4,  4,  4,  4,  4,  4,
        4,  4,  4,  4,  4,  4,  4,  4,
        4,  4,  4,  4,  4,  4,  4,  4,
        4,  4,  4,  4,  4,  4,  4,  4,
        4,  4,  4,  4,  4,  4,  4,  4,
    ];

    // ProRes 4444 XQ quantization matrix (highest quality)
    const FOUR_FOUR_FOUR_FOUR_XQ_MATRIX: [u8; 64] = [
        2,  2,  2,  2,  2,  2,  2,  2,
        2,  2,  2,  2,  2,  2,  2,  2,
        2,  2,  2,  2,  2,  2,  2,  2,
        2,  2,  2,  2,  2,  2,  2,  2,
        2,  2,  2,  2,  2,  2,  2,  2,
        2,  2,  2,  2,  2,  2,  2,  2,
        2,  2,  2,  2,  2,  2,  2,  2,
        2,  2,  2,  2,  2,  2,  2,  2,
    ];
}

/// Quantizer for ProRes
pub struct ProResQuantizer {
    matrix: QuantMatrix,
}

impl ProResQuantizer {
    /// Create new quantizer for profile and QP
    pub fn new(profile: ProResProfile, qp: u8) -> Self {
        Self {
            matrix: QuantMatrix::for_profile(profile, qp),
        }
    }

    /// Quantize DCT coefficients
    pub fn quantize(&self, dct_coeffs: &[i16; 64], quant_coeffs: &mut [i16; 64]) -> Result<()> {
        for i in 0..64 {
            let q = self.matrix.values[i] as i32;
            let coeff = dct_coeffs[i] as i32;

            // Quantization: coeff / q (with rounding)
            let quantized = if coeff >= 0 {
                (coeff + q / 2) / q
            } else {
                (coeff - q / 2) / q
            };

            quant_coeffs[i] = quantized.clamp(-32768, 32767) as i16;
        }
        Ok(())
    }

    /// Dequantize coefficients
    pub fn dequantize(&self, quant_coeffs: &[i16; 64], dct_coeffs: &mut [i16; 64]) -> Result<()> {
        for i in 0..64 {
            let q = self.matrix.values[i] as i32;
            let quant = quant_coeffs[i] as i32;

            // Dequantization: quant * q
            let dequant = quant * q;

            dct_coeffs[i] = dequant.clamp(-32768, 32767) as i16;
        }
        Ok(())
    }

    /// Get quantization parameter for a specific position
    pub fn get_qp(&self, pos: usize) -> u8 {
        self.matrix.values[pos]
    }
}

/// Scan order for DCT coefficients (zigzag)
pub struct ScanOrder;

impl ScanOrder {
    /// Zigzag scan order for 8×8 block
    pub const ZIGZAG: [usize; 64] = [
         0,  1,  5,  6, 14, 15, 27, 28,
         2,  4,  7, 13, 16, 26, 29, 42,
         3,  8, 12, 17, 25, 30, 41, 43,
         9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63,
    ];

    /// Convert from raster to zigzag order
    pub fn to_zigzag(raster: &[i16; 64], zigzag: &mut [i16; 64]) {
        for i in 0..64 {
            zigzag[i] = raster[Self::ZIGZAG[i]];
        }
    }

    /// Convert from zigzag to raster order
    pub fn from_zigzag(zigzag: &[i16; 64], raster: &mut [i16; 64]) {
        for i in 0..64 {
            raster[Self::ZIGZAG[i]] = zigzag[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_matrix_proxy() {
        let matrix = QuantMatrix::for_profile(ProResProfile::Proxy, 16);
        assert!(matrix.values[0] > 0);
        assert!(matrix.values[63] > 0);
    }

    #[test]
    fn test_quant_matrix_hq() {
        let matrix = QuantMatrix::for_profile(ProResProfile::Hq, 16);
        // HQ should have lower quantization values (higher quality)
        assert!(matrix.values[0] <= 10);
    }

    #[test]
    fn test_quant_dequant_roundtrip() {
        let quantizer = ProResQuantizer::new(ProResProfile::Standard, 16);

        let mut dct = [0i16; 64];
        dct[0] = 100;  // DC
        dct[1] = 50;   // Low freq
        dct[63] = 10;  // High freq

        let mut quant = [0i16; 64];
        quantizer.quantize(&dct, &mut quant).unwrap();

        let mut dequant = [0i16; 64];
        quantizer.dequantize(&quant, &mut dequant).unwrap();

        // Quantization is lossy, but should be close
        let dc_error = (dct[0] - dequant[0]).abs();
        assert!(dc_error < 20, "DC error too large: {}", dc_error);
    }

    #[test]
    fn test_quantize_all_zeros() {
        let quantizer = ProResQuantizer::new(ProResProfile::Standard, 16);

        let dct = [0i16; 64];
        let mut quant = [0i16; 64];

        quantizer.quantize(&dct, &mut quant).unwrap();

        assert_eq!(quant, [0i16; 64]);
    }

    #[test]
    fn test_zigzag_scan() {
        let mut raster = [0i16; 64];
        for i in 0..64 {
            raster[i] = i as i16;
        }

        let mut zigzag = [0i16; 64];
        ScanOrder::to_zigzag(&raster, &mut zigzag);

        let mut back = [0i16; 64];
        ScanOrder::from_zigzag(&zigzag, &mut back);

        assert_eq!(raster, back);
    }

    #[test]
    fn test_zigzag_dc_first() {
        let mut raster = [0i16; 64];
        raster[0] = 100;

        let mut zigzag = [0i16; 64];
        ScanOrder::to_zigzag(&raster, &mut zigzag);

        // DC should remain first in zigzag
        assert_eq!(zigzag[0], 100);
    }

    #[test]
    fn test_profile_quality_ordering() {
        // Higher profiles should have lower quantization (better quality)
        let proxy_q = QuantMatrix::for_profile(ProResProfile::Proxy, 16);
        let hq_q = QuantMatrix::for_profile(ProResProfile::Hq, 16);
        let xq_q = QuantMatrix::for_profile(ProResProfile::FourFourFourFourXq, 16);

        // Average quantization value
        let proxy_avg: u32 = proxy_q.values.iter().map(|&v| v as u32).sum::<u32>() / 64;
        let hq_avg: u32 = hq_q.values.iter().map(|&v| v as u32).sum::<u32>() / 64;
        let xq_avg: u32 = xq_q.values.iter().map(|&v| v as u32).sum::<u32>() / 64;

        assert!(proxy_avg > hq_avg, "Proxy should have higher quant than HQ");
        assert!(hq_avg > xq_avg, "HQ should have higher quant than XQ");
    }
}
